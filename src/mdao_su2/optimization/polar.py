"""The ``polar`` task: the lift curve through stall, URANS where steady stops converging.

A steady sweep over ``[polar] alpha`` runs every point at each angle with warm
starts. A point that misses its residual floor still leaves its last iterate as
the next angle's start, so a failed angle does not turn the rest of the sweep
into cold starts that measure nothing. Flagged angles (those that missed the floor
but left a finite solution), up to ``urans_max_angles``
per point, then run URANS from their steady restart (``solver/unsteady_primal.py``).

The derived values are pure functions of the per-angle records.
"""

import math
import os
import shutil
import time

import numpy as np
import openmdao.api as om

from ..mpi import on_root
from ..run_output.layout import final_artifact
from ..run_output.results import write_results
from ..run_output.solve_convergence import read_history_tail
from ..solver import point_cfg_writer as cfggen
from ..solver import unsteady_primal as unsteady
from ..solver.cfg_file import cfg_value, substitute
from ..solver.variable_names import SURF as _V

__all__ = ['polar', 'steady_accepted', 'missed_floor', 'urans_rejection',
                      'select_urans_angles', 'parabola_peak', 'summarise_point', 'summarise_max',
           'read_steady', 'carry_failed_restarts', 'angle_folder']

#: Wall-coordinate difference (m, chord 1 m) below which a design shape counts as the case mesh.
SHAPE_ZERO = 1.0e-10

#: Mach difference below which two Mach numbers are the same.
MACH_TOL = 1.0e-9

_MOMENT_COLUMNS = ('CMz', 'CMy', 'CMx')


def _finite(value):
    """``float(value)`` when finite, else None (the results file holds no NaN)."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


# ------------------------------------------------------------ pure rules

def steady_accepted(steady):
    """A steady angle counts when its final residual reached its floor."""
    residual, floor = steady.get('final_residual'), steady.get('floor')
    return residual is not None and (floor is None or residual <= floor)


def flag_angle(steady, accepted):
    """An angle goes to URANS when its steady solve is not accepted but left a finite solution."""
    return (not accepted and steady.get('cl') is not None
            and steady.get('final_residual') is not None)


def missed_floor(steady):
    """True when a finite final residual sits above the floor: the iterate is worth carrying."""
    residual, floor = steady.get('final_residual'), steady.get('floor')
    return residual is not None and floor is not None and residual > floor


def urans_rejection(reached_end, cl_range, cl_band, window_cycles, min_cycles):
    """Why a URANS run is not accepted, or None when it is.

    Two conditions hold together: the run reached ``TIME_ITER``; and the window is
    settled (``cl_range <= cl_band``) or spans at least ``min_cycles`` cycles of ``C_L``.
    """
    if not reached_end:
        return 'the run stopped before TIME_ITER'
    unsteady_window = cl_range is None or cl_range > cl_band
    if unsteady_window and window_cycles < min_cycles:
        return (f'unsteady window spans {window_cycles} C_L cycles, '
                f'fewer than urans_min_cycles={min_cycles}')
    return None


def select_urans_angles(flagged, max_angles):
    """Indices of the first ``max_angles`` flagged angles, in alpha order."""
    return [i for i, flag in enumerate(flagged) if flag][:max_angles]


def parabola_peak(alphas, cls):
    """Vertex ``(alpha, C_l)`` of the parabola through three angles, or None.

    None when the parabola is not concave or its vertex lies outside the
    outer two angles.
    """
    a, b, c = np.polyfit(alphas, cls, 2)
    if a >= 0.0:
        return None
    vertex = -b / (2.0 * a)
    if not alphas[0] < vertex < alphas[2]:
        return None
    return float(vertex), float(np.polyval([a, b, c], vertex))


def summarise_point(angles):
    """Stall peak of one point from its angle records (``alpha``, ``accepted``, ``cl``).

    The maximum is the stall maximum: the first accepted angle, in sweep order,
    whose next accepted angle has a lower ``C_l``; when no accepted angle is
    followed by a lower one (the curve still rises) it is the largest accepted
    ``C_l``. A later, higher ``C_l`` (a post-stall URANS mean) does not replace it.
    ``cl_max_sampled`` is that angle's ``C_l``; ``cl_max`` and ``alpha_stall``
    are the vertex of the parabola through that angle and its two neighbours in
    the sweep when both are accepted and the vertex lies between them, else the
    sampled values. ``peaked`` means an accepted angle above the maximum has a
    lower ``C_l``. All None when no angle is accepted.
    """
    accepted = [i for i, angle in enumerate(angles)
                if angle['accepted'] and angle.get('cl') is not None]
    if not accepted:
        return {'cl_max': None, 'alpha_stall': None, 'cl_max_sampled': None,
                'alpha_at_max_sampled': None, 'peaked': False}
    order = sorted(accepted, key=lambda i: angles[i]['alpha'])
    top = next((i for i, nxt in zip(order, order[1:])
                if angles[nxt]['cl'] < angles[i]['cl']), None)
    if top is None:
        top = max(accepted, key=lambda i: angles[i]['cl'])
    cl_top, alpha_top = angles[top]['cl'], angles[top]['alpha']
    peaked = any(angles[i]['alpha'] > alpha_top and angles[i]['cl'] < cl_top
                 for i in accepted)
    out = {'cl_max': cl_top, 'alpha_stall': alpha_top, 'cl_max_sampled': cl_top,
           'alpha_at_max_sampled': alpha_top, 'peaked': peaked}
    if 0 < top < len(angles) - 1 and top - 1 in accepted and top + 1 in accepted:
        trio = angles[top - 1:top + 2]
        vertex = parabola_peak([t['alpha'] for t in trio], [t['cl'] for t in trio])
        if vertex is not None:
            out['alpha_stall'], out['cl_max'] = vertex
    return out


def summarise_max(points, mach_floor=None):
    """The point with the largest ``cl_max`` and whether the Mach set fails to bracket it.

    ``points`` are records with ``id``, ``mach`` and ``cl_max``; ``mach_floor`` is
    the lowest Mach the sweep covers (the case's range bound). ``mach_edge`` is
    true when that point has the highest Mach of a set of several, or the lowest
    when that Mach is above ``mach_floor``: the maximum over Mach may then lie
    outside the set. ``range_bounded`` is true when the point sits at ``mach_floor``,
    where the range ends. With no ``mach_floor`` the lowest Mach counts as an edge.
    """
    valued = [p for p in points if p.get('cl_max') is not None]
    if not valued:
        return {'cl_max': None, 'point': None, 'mach': None, 'alpha_stall': None,
                'mach_edge': False, 'range_bounded': False}
    best = max(valued, key=lambda p: p['cl_max'])
    machs = sorted({p['mach'] for p in points})
    at_floor = mach_floor is not None and abs(best['mach'] - mach_floor) <= MACH_TOL
    at_lowest = best['mach'] == machs[0] and not at_floor
    return {'cl_max': best['cl_max'], 'point': best['id'], 'mach': best['mach'],
            'alpha_stall': best['alpha_stall'],
            'mach_edge': len(machs) > 1 and (best['mach'] == machs[-1] or at_lowest),
            'range_bounded': at_floor}


# --------------------------------------------------------- file helpers

class _Point:
    """One point's cfg, history and restart files, resolved from its generated primal cfg."""

    def __init__(self, cfg, point, workdir):
        self.id = point.id
        self.mach = float(point.mach)
        self.cfg_path = os.path.join(workdir, cfggen.cfg_name(cfg, point, False))
        self.workdir = workdir
        self.reynolds = _finite(cfg_value(self.cfg_path, 'REYNOLDS_NUMBER'))
        self.floor = _finite(cfg_value(self.cfg_path, 'CONV_RESIDUAL_MINVAL'))
        self.history = self._file('CONV_FILENAME', 'history', '.csv')
        self.restart = self._file('RESTART_FILENAME', 'restart_flow', '.dat')
        self.solution = self._file('SOLUTION_FILENAME', 'solution_flow', '.dat')
        self.surface = self._file('SURFACE_FILENAME', 'surface_flow', '.vtu')

    def _file(self, key, default, extension):
        stem = (cfg_value(self.cfg_path, key) or default).strip()
        name = stem if stem.endswith(extension) else stem + extension
        return name if os.path.isabs(name) else os.path.join(self.workdir, name)


def angle_folder(rundir, point_id, alpha):
    """``<run>/polar/<point>/a<alpha>``: the artifacts kept for one angle."""
    return os.path.join(rundir.polar, point_id, f'a{alpha:.2f}')


def read_steady(history, floor, since=None):
    """One steady solve's values from its history file, None-valued when it holds none.

    The coefficients are the last row's.
    A file last written before ``since`` (a clock time) belongs to an earlier solve.
    """
    out = {'final_residual': None, 'floor': floor, 'iterations': None,
           'cl': None, 'cd': None, 'cm': None, 'buffet': None}
    if not os.path.isfile(history) or (since is not None
                                       and os.path.getmtime(history) < since):
        return out
    tail = read_history_tail(history)
    if tail is not None:
        out['iterations'], _field, residual = tail
        out['final_residual'] = _finite(residual)
    rows = unsteady.read_history_rows(history)
    if rows:
        last = rows[-1]
        moment = next((c for c in _MOMENT_COLUMNS if c in last), None)
        out.update(cl=_finite(last.get('CL')), cd=_finite(last.get('CD')),
                   cm=_finite(last.get(moment)) if moment else None,
                   buffet=_finite(last.get('Buffet')))
    return out


def carry_failed_restarts(points, steady, comm=None):
    """Make the next angle start from the last iterate of every point that missed its floor.

    Copies the point's restart file over its solution file (rank 0, then
    every rank waits). A restart that holds no finite solve is not carried.
    ``steady`` maps point id to ``read_steady`` values. Returns the point ids carried.
    """
    def copy():
        carried = []
        for point in points:
            values = steady[point.id]
            if (missed_floor(values) and values.get('cl') is not None
                    and os.path.isfile(point.restart)):
                shutil.copy2(point.restart, point.solution)
                carried.append(point.id)
        return carried
    return copy() if comm is None else on_root(comm, copy, what='carry failed restarts')


def _keep_steady(rundir, point, alpha, cfg_path):
    folder = angle_folder(rundir, point.id, alpha)
    os.makedirs(folder, exist_ok=True)
    for source, kind in ((cfg_path, 'config'), (point.history, 'history'),
                         (point.restart, 'restart'), (point.surface, 'surface')):
        if os.path.isfile(source):
            shutil.copy2(source, os.path.join(folder, final_artifact('primal', kind)))
    return folder


# ----------------------------------------------------------------- lines

def _status(ok):
    return 'ok' if ok else 'FAILED'


def _steady_line(point, alpha, steady, accepted):
    residual, floor = steady['final_residual'], steady['floor']
    why = '' if accepted else '   steady missed its residual floor'
    goal = '-' if floor is None else f'{floor:g}'
    return (f'[polar steady] {_status(accepted):<6}  {point.id}  alpha {alpha:6.2f}  '
            f'{steady["iterations"] or "-"} it  res {_num(residual, ".2f")}/{goal}  '
            f'C_L {_num(steady["cl"])}  C_D {_num(steady["cd"])}{why}')


def _num(value, spec='.5f'):
    return '-' if value is None else format(value, spec)


def _urans_line(point, alpha, record, seconds):
    if record['error']:
        return (f'[polar urans ] FAILED  {point.id}  alpha {alpha:6.2f}  '
                f'{record["error"]}')
    ok = record['rejected'] is None
    how = '' if ok else f'not accepted: {record["rejected"]}  '
    return (f'[polar urans ] {_status(ok):<6}  {point.id}  alpha {alpha:6.2f}  '
            f'time iter {record["time_iter"]}  tavg C_L {_num(record["tavg_cl"])}  '
            f'tavg C_D {_num(record["tavg_cd"])}  '
            f'cycles {record["window_cycles"]}  '
            f'inner miss {_num(record["inner_miss"], ".3f")}  {how}'
            f'{seconds / 60.0:.1f} min')


# ------------------------------------------------------------------ task

def _record(point, alpha, steady, accepted, flagged):
    return {'alpha': alpha, 'mode': 'steady', 'accepted': accepted, 'flagged': flagged,
            'cl': steady['cl'], 'cd': steady['cd'], 'cm': steady['cm'],
            'buffet': steady['buffet'],
            'steady': {k: steady[k] for k in ('final_residual', 'floor', 'iterations')},
            'urans': None}


def _design_mesh(prob, cfg, point, rundir, comm):
    """The URANS mesh of the current design: the case mesh, or a deformed copy of it."""
    case_mesh = cfg_value(point.cfg_path, 'MESH_FILENAME')
    builder = prob.model._meta_builder
    surface = np.asarray(prob.get_val(
        f'{prob.model.geometry_name}.{_V.Geometry.COORDINATES_OUTPUT}', get_remote=True))
    x_aero = surface.reshape(builder.n_nodes, 3)[:, :builder.ndim].reshape(-1)
    if np.max(np.abs(x_aero - builder.x_ref)) <= SHAPE_ZERO:
        return case_mesh
    markers = getattr(builder, 'wall_markers', None) or [builder.wall_marker]
    return unsteady.write_deformed_mesh(
        point.cfg_path, os.path.join(rundir.workdir, 'polar', 'design_mesh'), x_aero,
        builder.x_ref, builder.gid_order, markers, comm)


def _reason(exc, limit=300):
    """``Type: cause`` of an exception: its last non-empty line (the cause, which an
    ``on_root`` error carries after a ``failed on rank 0`` header), truncated to ``limit``."""
    lines = [line.strip() for line in str(exc).splitlines() if line.strip()]
    last = lines[-1] if lines else ''
    name = type(exc).__name__
    text = last if last.startswith(name) else f'{name}: {last}'
    return text if len(text) <= limit else text[:limit - 3] + '...'


def _urans_record(table, error=None):
    """An angle's ``urans`` record with no measurement; ``error`` marks a failed angle."""
    limit = int(float(table['TIME_ITER']))
    return {'tavg_cl': None, 'tavg_cd': None, 'tavg_cm': None, 'tavg_buffet': None,
            'cl_range': None, 'time_iter': None,
            'averaging_length': limit - int(float(table['WINDOW_START_ITER'])),
            'window_steps': None, 'window_cycles': None, 'inner_miss': None,
            'inner_floor': None, 'rejected': None, 'error': error}


def _run_urans_angle(prob, cfg, point, alpha, steady_folder, mesh, rundir, comm,
                     kept_mesh=None):
    """URANS at one flagged angle; the angle's ``urans`` record.

    The kept ``urans.cfg`` names ``kept_mesh`` when given (``su2_workdir/`` is deleted at DONE).
    """
    polar = cfg.polar
    table = polar.urans
    work = os.path.join(rundir.workdir, 'polar', point.id, f'a{alpha:.2f}')
    cfg_path = os.path.join(work, 'urans.cfg')
    limit = int(float(table['TIME_ITER']))
    record = _urans_record(table)

    def prepare():
        os.makedirs(work, exist_ok=True)
        unsteady.write_urans_cfg(point.cfg_path, cfg_path, table, alpha, mesh, work)
        folder = angle_folder(rundir, point.id, alpha)
        os.makedirs(folder, exist_ok=True)
        kept_cfg = os.path.join(folder, 'urans.cfg')
        shutil.copy2(cfg_path, kept_cfg)
        if kept_mesh is not None:
            with open(cfg_path) as stream:
                text = substitute(stream.read(), {'MESH_FILENAME': kept_mesh})
            with open(kept_cfg, 'w') as stream:
                stream.write(text)
        stem = cfg_value(cfg_path, 'SOLUTION_FILENAME')
        unsteady.seed_time_levels(
            os.path.join(steady_folder, final_artifact('primal', 'restart')), work, stem,
            int(float(table['RESTART_ITER'])))
    on_root(comm, prepare, what=f'prepare URANS of {point.id} at {alpha:.2f}')
    result = None
    try:
        result = unsteady.run_urans(cfg_path, comm, work,
                                    cfg.constraints.moment.objective_key)
    except Exception as exc:        # noqa: BLE001 -- a failed angle is a record
        record['error'] = _reason(exc)

    def keep():
        source = result.history if result is not None else unsteady.urans_history_path(
            cfg_path, work)
        if os.path.isfile(source):
            shutil.copy2(source, os.path.join(angle_folder(rundir, point.id, alpha),
                                              'urans_history.csv'))
    on_root(comm, keep, what=f'keep URANS artifacts of {point.id} at {alpha:.2f}')
    if result is None:
        return record
    record.update(
        tavg_cl=_finite(result.tavg_cl), tavg_cd=_finite(result.tavg_cd),
        tavg_cm=_finite(result.tavg_cm), tavg_buffet=_finite(result.tavg_buffet),
        cl_range=_finite(result.cl_range), time_iter=result.time_iter,
        averaging_length=result.averaging_length, window_steps=result.window_steps,
        window_cycles=result.window_cycles, inner_miss=_finite(result.inner_miss),
        inner_floor=result.inner_floor)
    record['rejected'] = urans_rejection(
        result.reached_time_iter(limit), record['cl_range'], polar.cl_band,
        result.window_cycles, polar.urans_min_cycles)
    return record


def polar(prob, cfg, rank0, console=None, rundir=None, progress=None):
    """Steady sweep with warm starts, then URANS at the flagged angles; the polar record."""
    comm = prob.comm
    spec = cfg.polar
    points = [_Point(cfg, p, rundir.workdir) for p in cfg.points]
    say = progress.say if progress is not None else print

    def tell(line):
        if rank0:
            say(line)

    records = {p.id: [] for p in points}

    def save_partial():
        """Rank 0 writes the angles done so far, so a solver abort loses nothing already measured."""
        if rank0 and rundir is not None and all(records[p.id] for p in points):
            write_results(cfg, rundir, {**_payload(spec, points, records), 'complete': False})

    saved = {p.id: {} for p in points}
    design = {'mesh': None, 'kept': None, 'error': None}

    def build_design_mesh():
        """Build the URANS mesh once, right after the first steady angle; a failure is recorded."""
        try:
            design['mesh'] = _design_mesh(prob, cfg, points[0], rundir, comm)
            if design['mesh'] != cfg_value(points[0].cfg_path, 'MESH_FILENAME'):
                def keep_mesh():
                    path = os.path.join(rundir.polar, 'design_mesh.su2')
                    os.makedirs(rundir.polar, exist_ok=True)
                    shutil.copy2(design['mesh'], path)
                    return path
                design['kept'] = on_root(comm, keep_mesh, what='keep the design mesh')
        except Exception as exc:    # noqa: BLE001 -- the sweep goes on; the URANS angles carry the error
            design['error'] = _reason(exc)
            tell(f'[polar urans ] FAILED  design mesh: {design["error"]}')

    for alpha in spec.alpha:
        for i in range(cfg.n_points):
            prob.set_val(f'alpha_{i}', alpha)
        last_error = None
        since = time.time() - 2.0
        try:
            prob.run_model()
        except om.AnalysisError as exc:
            last_error = str(exc).splitlines()[0] if str(exc) else 'AnalysisError'

        def read():
            return {p.id: read_steady(p.history, p.floor, since)
                    for p in points}
        steady = on_root(comm, read, what='read the steady histories')

        def keep():
            return {p.id: _keep_steady(rundir, p, alpha, p.cfg_path) for p in points}
        folders = on_root(comm, keep, what='keep the steady artifacts')
        carry_failed_restarts(points, steady, comm)

        for p in points:
            accepted = steady_accepted(steady[p.id])
            record = _record(p, alpha, steady[p.id], accepted,
                             flag_angle(steady[p.id], accepted))
            if last_error is not None and (missed_floor(steady[p.id])
                                           or steady[p.id]['final_residual'] is None):
                record['steady']['error'] = last_error
            records[p.id].append(record)
            saved[p.id][alpha] = folders[p.id]
            tell(_steady_line(p, alpha, steady[p.id], accepted))
        if spec.run_urans and alpha == spec.alpha[0]:
            save_partial()      # a SU2 abort in the mesh run cannot be caught
            build_design_mesh()

    save_partial()
    if spec.run_urans:
        for p in points:
            flagged = [r['flagged'] for r in records[p.id]]
            chosen = select_urans_angles(flagged, spec.urans_max_angles)
            for index in chosen:
                record = records[p.id][index]
                alpha = record['alpha']
                started = time.monotonic()
                if design['error'] is not None:
                    result = _urans_record(spec.urans, design['error'])
                else:
                    tell(f'[polar urans ] start   {p.id}  alpha {alpha:6.2f}  '
                         f'{spec.urans["TIME_ITER"]} time steps')
                    result = _run_urans_angle(prob, cfg, p, alpha, saved[p.id][alpha],
                                              design['mesh'], rundir, comm,
                                              design['kept'])
                tell(_urans_line(p, alpha, result, time.monotonic() - started))
                record['mode'] = 'urans'
                record['urans'] = result
                record['accepted'] = (result['error'] is None
                                      and result['rejected'] is None)
                for key, source in (('cl', 'tavg_cl'), ('cd', 'tavg_cd'),
                                    ('cm', 'tavg_cm'), ('buffet', 'tavg_buffet')):
                    record[key] = result[source]
                save_partial()
            for index, flag in enumerate(flagged):
                if flag and index not in chosen:
                    tell(f'[polar urans ] skipped  {p.id}  alpha '
                         f'{records[p.id][index]["alpha"]:6.2f}  '
                         f'beyond urans_max_angles={spec.urans_max_angles}')

    result = _payload(spec, points, records)
    tell(_summary_line(result['polar']['points'], result['max']))
    return {**result, 'complete': True}


def _payload(spec, points, records):
    """The polar record of the angles run so far (``complete`` is added by the caller)."""
    entries = []
    for p in points:
        entries.append({'id': p.id, 'mach': p.mach, 'reynolds': p.reynolds,
                        'angles': records[p.id], **summarise_point(records[p.id])})
    final = {p.id: records[p.id][-1] for p in points}
    return {'polar': {'alpha': list(spec.alpha),
                      'cl_band': spec.cl_band, 'run_urans': spec.run_urans,
                      'urans_max_angles': spec.urans_max_angles,
                      'urans_min_cycles': spec.urans_min_cycles,
                      'mach_floor': spec.mach_floor, 'points': entries},
            'max': summarise_max(entries, spec.mach_floor),
            'points': [{'id': p.id, 'alpha': final[p.id]['alpha'],
                        'C_L': final[p.id]['cl'], 'C_D': final[p.id]['cd']}
                       for p in points]}


def _summary_line(entries, peak):
    lines = []
    for e in entries:
        ok = sum(1 for a in e['angles'] if a['accepted'])
        cl = '-' if e['cl_max'] is None else f'{e["cl_max"]:.4f}'
        stall = '-' if e['alpha_stall'] is None else f'{e["alpha_stall"]:.2f}'
        lines.append(f'[polar] {e["id"]}  {ok}/{len(e["angles"])} angles accepted  '
                     f'C_l,max {cl} at {stall} deg  peaked {e["peaked"]}')
    if peak['point'] is not None:
        edge = '  Mach set does not bracket the maximum' if peak['mach_edge'] else ''
        if peak['range_bounded']:
            edge += '  maximum at the lower bound of the Mach range'
        lines.append(f'[polar] largest C_l,max {peak["cl_max"]:.4f} at {peak["point"]}{edge}')
    return '\n'.join(lines)
