"""Finite-difference totals and moment-pivot verification tasks."""

import os
from contextlib import contextmanager, nullcontext

import numpy as np

from .adjoint_reuse import reuse_adjoints
from .progress import FD_DRIFT_WINDOW
from ..run_config.root import normal_force_slope

__all__ = ["check_totals", "moment_pivot_check", "moment_pivot_responses", "judge_rows",
           "format_check_totals", "summarize_check_totals", "GateError",
           "NO_BAND_WARNING", "FD_NOISE_VERDICT"]

#: Printed under the table of a run that states no band.
NO_BAND_WARNING = ('No acceptance band stated: differences are reported, not judged. '
                   'FD error depends on step size and cancellation (C_r); judge these '
                   'rows or state a band.')


class GateError(AssertionError):
    """A verification gate did not pass; ``result`` is the record to write first.

    The task has already printed its report, so ``reported`` tells the console
    and ``run/optimize.py`` to print no traceback. ``exit_code`` is 1 for a
    failed gate and 3 for an inconclusive one.
    """

    reported = True

    def __init__(self, message, result, exit_code=1):
        super().__init__(message)
        self.result = result
        self.exit_code = exit_code


#: SU2 history columns holding each solver output, in order of preference.
_HISTORY_COLUMNS = {'C_D': ('CD',), 'C_L': ('CL',), 'C_M': ('CMz', 'CMy', 'CMx'),
                    'C_EFF': ('CEff',), 'C_BUFFET': ('Buffet',)}


def check_totals(prob, cfg, rank0, console=None, rundir=None, progress=None):
    """Cold-start FD verification; the validator enforces the cold start.

    Every driver response is differentiated. Geometric constraints are
    reported and never judged.
    """
    v = cfg.verification
    prefix = prob.model._scenario_prefix
    prob.run_model()
    of = [path for path, _kind, _ in prob.model.driver_responses]
    geometry_paths = {path for path, kind, _ in prob.model.driver_responses
                      if kind == 'geometry'}
    for output in v.shape_outputs:
        for point in cfg.points:
            path = f'{prefix}{point.id}.{output}'
            if path not in of:
                of.append(path)
    capture = nullcontext(None)
    if v.record_shape_sensitivities and rundir is not None:
        from .shape_diagnostics import (capture_shape_sensitivities,
                                               write_shape_sensitivities)
        capture = capture_shape_sensitivities(prob, cfg, rundir)
    data = {}
    shape_report = None
    with reuse_adjoints(prob.model), _openmdao_report(rundir, rank0) as report:
        if 'shape' in cfg.geometry_dvs:
            with capture as fields:
                data = prob.check_totals(of=of, wrt=['shape'], step=v.shape_step,
                                         form=v.form, step_calc=v.step_calc,
                                         compact_print=False, out_stream=report)
            # FD restores OpenMDAO vectors, not SU2's solution file or coefficient
            # caches. Refresh the baseline primal before the next check's adjoints.
            prob.run_model()
            if fields is not None:
                solver_data = {key: d for key, d in data.items()
                               if key[0] not in geometry_paths}
                shape_report = write_shape_sensitivities(prob, cfg, rundir, fields,
                                                         solver_data)
                if not shape_report['contraction_pass']:
                    raise GateError(
                        'Captured sensitivity contraction disagrees with shape totals; '
                        'see the shape_sensitivities block of results_check_totals.json',
                        {'shape_sensitivities': shape_report})
        if 'twist' in cfg.geometry_dvs:
            data.update(prob.check_totals(of=of, wrt=['twist'], step=v.twist_step,
                                          form=v.form, step_calc=v.step_calc,
                                          compact_print=False, out_stream=report))
            prob.run_model()
        # Twist rows take the shape band; _row_tag keeps a shape C_r
        # table off them.
        shape_rows = set(data)
        alpha_of = list(of)
        for output in v.alpha_outputs:
            for point in cfg.points:
                path = f'{prefix}{point.id}.{output}'
                if path not in alpha_of:
                    alpha_of.append(path)
                value = float(prob.get_val(path, get_remote=True)[0])
                if output == 'C_M':
                    assert np.isfinite(value) and abs(value) < 0.5, (
                        f'{path}={value} violates the finite |C_M|<0.5 '
                        'quarter-chord sanity gate')
        # A pivot-gated moment row is normalized by dC_N/dalpha, which needs
        # each point's C_D alpha row and its baseline alpha, C_L and C_D.
        point_state = {}
        if v.alpha_band_moment_pivot > 0:
            for i, point in enumerate(cfg.points):
                path = f'{prefix}{point.id}'
                if f'{path}.C_D' not in alpha_of:
                    alpha_of.append(f'{path}.C_D')
                point_state[path] = tuple(
                    float(prob.get_val(name, get_remote=True)[0])
                    for name in (f'alpha_{i}', f'{path}.C_L', f'{path}.C_D'))
        data.update(prob.check_totals(
            of=alpha_of, wrt=[f'alpha_{i}' for i in range(cfg.n_points)],
            step=v.alpha_step, form=v.form, step_calc=v.step_calc,
            compact_print=False, out_stream=report))

    param_tag = _param_tag(cfg)
    noise = _fd_noise(prob, cfg, data, progress, geometry_paths) if rank0 else None
    comm = getattr(prob, 'comm', None)
    if comm is not None and getattr(comm, 'size', 1) > 1:
        noise = comm.bcast(noise, root=0)     # every rank must reach one verdict
    rows, failures = judge_rows(data, shape_rows, v, param_tag, geometry_paths,
                                fd_noise=noise, point_state=point_state)
    judged = v.judged
    inside_noise = [r for r in rows if r['verdict'] == FD_NOISE_VERDICT]
    verdict = ('not judged' if not judged else 'fail' if failures
               else 'inconclusive' if inside_noise else 'pass')
    result = {'parameterization': param_tag, 'judged': judged, 'verdict': verdict,
              'rows': rows, 'failures': failures,
              'pass': {'pass': True, 'fail': False}.get(verdict)}
    if shape_report is not None:
        result['shape_sensitivities'] = shape_report
    if rank0:
        print('\n' + '\n'.join(format_check_totals(rows, v)))
        print('\n' + '\n'.join(summarize_check_totals(rows, verdict)))
    if failures:
        # The caller writes the complete numerical referee table before the
        # error propagates, so a failed gate stays diagnosable.
        raise GateError('check_totals band violated:\n  ' + '\n  '.join(failures),
                        result)
    if verdict == 'inconclusive':
        raise GateError(f'check_totals inconclusive: {len(inside_noise)} row(s) outside '
                        f'their band by less than the FD can resolve', result,
                        exit_code=3)
    return result


#: Verdict of a row outside its band by less than the FD resolves.
FD_NOISE_VERDICT = 'within FD noise'


def _objective_weights(prob, cfg):
    """``{(scenario path, output): d obj / d input}`` at the current point, or None.

    Differentiates the objective's own ``ExecComp`` source on a one-rank copy,
    so the objective row's FD noise is the weighted sum of its inputs' noise.
    """
    compiled = getattr(cfg, 'objective_compiled', None)
    if compiled is None or not getattr(compiled, 'connections', None):
        return None
    try:
        import openmdao.api as om
        try:
            from mpi4py import MPI
            self_comm = MPI.COMM_SELF
        except ImportError:
            self_comm = None
        values, sources = {}, {}
        for source, target in compiled.connections:
            name = target.rsplit('.', 1)[-1]
            values[name] = float(np.asarray(prob.get_val(target)).ravel()[0])
            sources[name] = tuple(source.rsplit('.', 1))
        sub = om.Problem(comm=self_comm, reports=False)
        sub.model.add_subsystem(
            'objective', om.ExecComp(compiled.execcomp_src,
                                     **{n: {'val': values.get(n, 0.0)}
                                        for n in compiled.inputs}),
            promotes=['*'])
        sub.setup()
        sub.run_model()
        totals = sub.compute_totals(of=['obj'], wrt=list(values))
        return {sources[name]: float(np.asarray(totals['obj', name]).ravel()[0])
                for name in values}
    except Exception:
        return None


def _fd_noise(prob, cfg, data, progress, geometry_paths):
    """``{(of, wrt, k, i): smallest |adjoint - FD| this run's FD resolves}``.

    From ``Progress.fd_resolution``: each perturbed primal's drift over its
    last iterations, divided by the step. Rows without a recorded drift (no
    progress counter, a relative step, an output with no history column) get
    no entry and are judged against their band alone.
    """
    if progress is None:
        return {}
    v = cfg.verification
    if v.step_calc != 'abs':
        return {}
    weights = None

    def noise_or_nan(value):
        return np.nan if value is None else value

    def resolution(group, index, scenario, output, step):
        prefix = prob.model._scenario_prefix
        point = scenario[len(prefix):] if prefix and scenario.startswith(prefix) else scenario
        for column in _HISTORY_COLUMNS.get(output, (output,)):
            value = progress.fd_resolution(group, index, point, column, step, v.form)
            if value is not None:
                return value
        return None

    # A linear constraint's FD noise is the coefficient-weighted noise of its
    # terms, as the objective's is.
    linear = {c.path: compiled for c, compiled in getattr(cfg, 'linear_compiled', ())}
    noise = {}
    for (of, wrt), d in data.items():
        if of in geometry_paths:
            continue
        if wrt.startswith('alpha_'):
            group, step, fixed = 'alpha', v.alpha_step, int(wrt.split('_', 1)[1])
        else:
            group, fixed = wrt, None
            step = v.twist_step if wrt == 'twist' else v.shape_step
        scenario, _, output = of.rpartition('.')
        n_dv = _jacobian(d, 'J_fd').shape[1]
        for i in range(n_dv):
            index = fixed if fixed is not None else i
            if of in linear:
                compiled = linear[of]
                coeffs = compiled.partials({n: 0.0 for n in compiled.inputs})
                prefix = prob.model._scenario_prefix
                parts = [abs(coeffs[n]) * noise_or_nan(resolution(
                    group, index, f'{prefix}{pid}', out, step))
                         for n, (pid, out) in compiled.sources.items() if coeffs[n] != 0.0]
                value = float(sum(parts)) if parts and np.all(np.isfinite(parts)) else None
            elif output in ('obj', 'obj_out'):
                if weights is None:
                    weights = _objective_weights(prob, cfg) or {}
                parts = [abs(w) * noise_or_nan(resolution(group, index, src, out, step))
                         for (src, out), w in weights.items() if w != 0.0]
                value = float(sum(parts)) if parts and np.all(np.isfinite(parts)) else None
            else:
                value = resolution(group, index, scenario, output, step)
            if value is not None:
                noise[(of, wrt, 0, i)] = value
    return noise


#: OpenMDAO's own check_totals report, kept beside the run's other logs.
OPENMDAO_REPORT = 'check_totals_openmdao.txt'


@contextmanager
def _openmdao_report(rundir, rank0):
    """Rank 0's stream for OpenMDAO's report: ``logs/<OPENMDAO_REPORT>``, or None.

    The table this task prints carries every entry of that report (adjoint,
    FD, absolute and relative difference), so the report leaves the terminal
    and stays on disk. Without a run ``logs/`` it is dropped.
    """
    logs = getattr(rundir, 'logs', None)
    if not rank0 or logs is None:
        yield None
        return
    os.makedirs(logs, exist_ok=True)
    with open(os.path.join(logs, OPENMDAO_REPORT), 'w') as stream:
        yield stream


def moment_pivot_responses(prob, cfg):
    """The responses the moment-pivot gate differentiates: each point's ``C_M``."""
    prefix = prob.model._scenario_prefix
    return [f'{prefix}{p.id}.C_M' for p in cfg.points]


def moment_pivot_check(prob, cfg, rank0, console=None, rundir=None, progress=None):
    """FD-free moment gate: the exact two-pivot transfer identity.

    Verifies dC_M/dalpha where FD cannot.  About the aerodynamic center the
    quantity is ~0 by construction, so C_r is ~10^5 and the FD reference is
    cancellation-limited at ~6e-4 relative regardless of h or primal
    convergence.  Shifting the pivot away from the AC amplifies the same
    physics by ~74x and is not cancellation-limited, so the identity referees
    it against the FD-verified force derivatives instead of against a second
    finite difference.

    See MomentPivotConfig for why the identity is not a tautology: the
    frame-rotation term ties the angle-of-attack force projection on the force side to the
    body-axis moment's lack of one (``solver/wall_seam.py::project_alpha``),
    which is the one path -- moment AND
    alpha together -- that the shape rows and the C_D/C_L alpha rows leave
    untested.
    """
    mp = cfg.verification.moment_pivot
    # resolved_pivot, NOT case.reference.origin_moment: point_cfg_writer emits
    # REF_ORIGIN_MOMENT_* from constraints.moment.resolved_pivot(), so a
    # [constraints.moment] pivot silently overrides the case reference. Reading
    # the case field made the first run of this gate report "pivot x=0" while
    # SU2 integrated about 0.25 -- the gate caught the contradiction only
    # because the identity failed by exactly 0.25 chords.
    pivot_x = cfg.constraints.moment.resolved_pivot(
        cfg.case.reference.origin_moment)[0]
    prob.run_model()

    wrt = [f'alpha_{i}' for i in range(cfg.n_points)]
    prefix = prob.model._scenario_prefix
    of = moment_pivot_responses(prob, cfg)
    totals = prob.compute_totals(of=of, wrt=wrt)

    dcn = mp.dcn_dalpha()
    results = []
    failures = []
    for i, point in enumerate(cfg.points):
        key = (f'{prefix}{point.id}.C_M', f'alpha_{i}')
        measured = float(np.asarray(totals[key]).ravel()[0])
        predicted = mp.predict(pivot_x)
        # In chords of equivalent pivot error -- the well-posed metric, and the
        # same one alpha_band_moment_pivot uses (VerificationConfig).
        chords = abs(measured - predicted) / abs(dcn)
        ok = bool(np.isfinite(chords) and chords <= mp.band_pivot)
        results.append({
            'point': point.id, 'pivot_x': pivot_x,
            'reference_pivot_x': mp.reference_pivot_x,
            'reference_run': mp.reference_run,
            'dcm_dalpha_measured': measured,
            'dcm_dalpha_predicted': predicted,
            'dcn_dalpha': dcn,
            'absolute_error': abs(measured - predicted),
            'pivot_chord_error': chords,
            'band_pivot': mp.band_pivot,
            'amplification_vs_reference': (abs(predicted / mp.dcm_dalpha_ref)
                                           if mp.dcm_dalpha_ref else None),
            'x_ac_chords': (mp.reference_pivot_x + mp.dcm_dalpha_ref / dcn
                            if dcn else None),
            'pass': ok,
        })
        if not ok:
            failures.append(
                f'{point.id}: dC_M/dalpha at pivot x={pivot_x:g} measured '
                f'{measured: .10e} vs identity {predicted: .10e} -- '
                f'{chords:.3e} chords of pivot error > {mp.band_pivot:.1e}')

    if rank0:
        print(f'\n=== moment-transfer identity (pivot x={pivot_x:g} vs '
              f'reference x={mp.reference_pivot_x:g}) ===')
        print(f'  reference run       : {mp.reference_run or "-"}')
        print(f'  dC_N/dalpha         : {dcn: .10e} /deg  (body frame, from '
              f'the FD-verified dC_L/dalpha and dC_D/dalpha)')
        for r in results:
            print(f'  [{r["point"]}] predicted {r["dcm_dalpha_predicted"]: .10e}'
                  f'   measured {r["dcm_dalpha_measured"]: .10e}')
            print(f'        abs err {r["absolute_error"]:.4e}   '
                  f'pivot error {r["pivot_chord_error"]:.4e} chords   '
                  f'band {r["band_pivot"]:.1e}   '
                  f'{"PASS" if r["pass"] else "FAIL"}')
            if r['x_ac_chords'] is not None:
                print(f'        implied x_AC {r["x_ac_chords"]:.6f} c')
    result = {'rows': results, 'failures': failures, 'pass': not failures}
    if failures:
        raise GateError('moment-transfer identity violated:\n  '
                        + '\n  '.join(failures), result)
    return result


def _param_tag(cfg):
    """Parameterization tag for C_r table matching."""
    p = cfg.geometry.parameterization
    if p == 'ffd':
        return cfg.geometry.ffd
    if p == 'cst':
        return f'cst{cfg.geometry.n_cst}'
    return p


def _analytic(d):
    return np.atleast_1d(np.squeeze(d['J_rev' if 'J_rev' in d else 'J_fwd']))


def _row_functional(of, verification):
    """'sp0.C_M' -> 'C_M'; the objective path -> the objective's C_r key."""
    functional = of.split('.')[-1]
    return (verification.OBJECTIVE_CR_KEY if functional in ('obj', 'obj_out')
            else functional)


def _pivot_normalizer(data, of, wrt, point_state):
    """|dC_N/dalpha| for the point owning ``of``, or None when absent.

    A body-axis moment transfers with the body-normal force, so a moment-row
    error read as a pivot shift is |adjoint - FD| / |dC_N/dalpha|
    (``run_config/root.py::normal_force_slope``). It needs the point's C_L and
    C_D alpha rows and its baseline alpha, C_L and C_D; without one the
    pivot-gated row fails rather than falling back to a relative band.
    """
    point = of.rsplit('.', 1)[0]
    dcl = data.get((f'{point}.C_L', wrt))
    dcd = data.get((f'{point}.C_D', wrt))
    state = (point_state or {}).get(point)
    if dcl is None or dcd is None or state is None:
        return None
    alpha_deg, cl, cd = state
    return abs(normal_force_slope(float(_analytic(dcl)[0]), float(_analytic(dcd)[0]),
                                  cl, cd, alpha_deg))


def _shape_cr(verification, functional, i, tag):
    """Measured cancellation ratio for a shape row, or None when not measured."""
    cr = verification.cancellation_cr.get(functional)
    if cr and i < len(cr) and (tag is None
                               or verification.cancellation_cr_param == tag):
        return float(cr[i])
    return None


#: Plain-language reasons printed under the check_totals table.
CHECK_TOTALS_LEGEND = {
    'cancellation': (
        'C_r (cancellation ratio): the response is a small difference of large '
        'surface integrals, so the finite difference loses digits to round-off '
        'and solver noise. The band grows with C_r; it is shown only where C_r '
        'was measured for this parameterization.'),
    'flat': (
        'flat band: no C_r measurement for this row, so the flat relative '
        'band applies.'),
    'unverifiable': (
        'not FD-verifiable: the band would exceed the cap, so even a zeroed or '
        'sign-flipped derivative would pass. The comparison is shown but not '
        'judged. Near-tangential shape rows land here.'),
    'pivot': (
        'pivot check: a moment row compared as pivot error in chords, '
        '|adjoint - FD| / |dC_N/dalpha|.'),
    'moment': 'moment band: moment alpha rows use their own relative band.',
    'geometry': (
        'geometric constraint: reported, never judged. Its derivative comes '
        'from the geometry alone, and an exactly zero entry would fail a '
        'relative band on FD round-off.'),
}


def _row_tag(wrt, param_tag):
    """C_r tables are per parameterization; twist rows have their own tag."""
    return 'twist' if wrt == 'twist' else param_tag


def _jacobian(d, key):
    """An OpenMDAO check_totals block as (output index, DV index)."""
    return np.atleast_2d(np.asarray(d[key], dtype=float))


def judge_rows(data, shape_rows, verification, param_tag, geometry_paths=(),
               fd_noise=None, point_state=None):
    """Report every derivative entry, and judge it when the run states bands.

    Returns ``(rows, failures)``. A row is one (output index, DV index) entry.
    Every row carries the adjoint, the FD value, ``|adjoint - FD|``, the
    relative difference, and ``C_r`` where it was measured. A pivot-gated row
    also carries its pivot-chord error. Without a stated band every row reads
    ``not judged`` with ``pass: None`` and ``failures`` is empty. A row is
    named ``d <of> / d <wrt>[i]``, or ``d <of>[k] / d <wrt>[i]`` when the
    output is a vector, as geometric constraints are. With bands,
    a row is inside its band, outside it, or NOT FD-VERIFIABLE: a tiered band
    past ``shape_band_cap``, where even a sign-flipped derivative would score
    inside. The last fails unless ``shape_rows_unverifiable`` names it,
    because reporting it as a pass is the gate-that-does-not-gate failure
    forbids. Rows of ``geometry_paths`` are reported and never judged.

    ``fd_noise`` maps ``(of, wrt, k, i)`` to the smallest ``|adjoint - FD|``
    the run's finite difference resolves. A row outside its band whose
    difference is below that reads ``within FD noise`` with ``pass: None``:
    the FD cannot tell it from agreement, so it neither passes nor fails.
    Each judged row also carries ``band_ratio``, its metric over its band.

    ``point_state`` maps a point path to its baseline ``(alpha_deg, C_L, C_D)``,
    which the pivot-chord normalizer needs.
    """
    v = verification
    judged = v.judged
    rows, failures = [], []
    for key, d in data.items():
        of, wrt = key
        functional = _row_functional(of, v)
        analytic = _jacobian(d, 'J_rev' if 'J_rev' in d else 'J_fwd')
        fd = _jacobian(d, 'J_fd')
        is_shape = key in shape_rows
        tag = _row_tag(wrt, param_tag)
        kind = ('twist' if wrt == 'twist' else 'shape') if is_shape else 'alpha'
        n_out = analytic.shape[0]
        for k in range(n_out):
            for i in range(analytic.shape[1]):
                a, f = float(analytic[k, i]), float(fd[k, i])
                output = of if n_out == 1 else f'{of}[{k}]'
                row = {'derivative': f'd {output} / d {wrt}[{i}]',
                       'of': of, 'wrt': wrt, 'output_index': k, 'index': i,
                       'analytic': a, 'fd': f, 'abs_difference': abs(a - f),
                       'relative_error': abs(a - f) / max(abs(f), 1e-30),
                       'kind': kind}
                cr = _shape_cr(v, functional, i, tag) if is_shape else None
                if cr is not None:
                    row['cancellation_ratio'] = cr
                resolved = (fd_noise or {}).get((of, wrt, k, i))
                if resolved is not None and of not in geometry_paths:
                    row['fd_noise_abs'] = resolved
                    if f != 0.0:     # a relative noise needs a nonzero FD value
                        row['fd_noise'] = resolved / abs(f)
                failure = _judge_row(row, data, functional, tag, key, v, judged,
                                     of in geometry_paths, cr, output, point_state)
                rows.append(row)
                if failure:
                    row['failure'] = failure
                    failures.append(failure)
    return rows, failures


def _judge_row(row, data, functional, tag, key, v, judged, geometric, cr, output,
               point_state=None):
    """Fill ``row``'s band, verdict and note; return its failure line or None."""
    of, wrt = key
    a, f, i = row['analytic'], row['fd'], row['index']
    label = f'd({output})/d({wrt})[{i}]'
    pivot = (row['kind'] == 'alpha' and not geometric
             and v.alpha_row_is_pivot_gated(functional))
    scale = _pivot_normalizer(data, of, wrt, point_state) if pivot else None
    if pivot:
        row['normalizer_dcn_dalpha'] = scale
        row['pivot_chord_error'] = (abs(a - f) / scale if scale else float('inf'))
    if geometric or not judged:
        row.update(band=None, verdict='not judged', note_kind=None, note='')
        row['pass'] = None
        if geometric:
            row.update(note_kind='geometry', note='geometric constraint')
        elif cr is not None:
            row['note_kind'] = 'cancellation'
        elif row['kind'] != 'alpha':
            row['note'] = 'C_r not measured'
        return None
    if row['kind'] != 'alpha':
        band = v.shape_row_band(functional, i, param=tag)
        row['band'] = band
        row['metric'] = 'relative'
        row['note_kind'] = 'cancellation' if cr is not None else 'flat'
        row['note'] = 'tiered band' if cr is not None else 'flat band, C_r not measured'
        why = (f' [C_r={cr:.1f}]' if cr is not None else ' [flat: C_r unmeasured]')
        if band is None:
            return _unbanded(row, label, 'shape_band')
        verifiable = v.shape_row_is_verifiable(functional, i, param=tag)
        row['fd_verifiable'] = bool(verifiable)
        if not verifiable:
            # Never 'pass': the band has no discriminating power.
            row['acknowledged'] = bool(v.shape_row_unverifiable_ack(functional, i))
            row['pass'] = False
            row['verdict'] = ('unverifiable_acknowledged'
                              if row['acknowledged'] else 'unverifiable')
            row['note_kind'] = 'unverifiable'
            row['note'] += '; not FD-verifiable'
            if row['acknowledged']:
                return None
            return (f'{label}: NOT FD-VERIFIABLE{why} -- band {band:.2e} exceeds the '
                    f'{v.shape_band_cap:.1e} cap, so a zeroed or sign-flipped '
                    f'derivative would score inside it. Acknowledge the row in '
                    f'[verification] shape_rows_unverifiable, or reduce C_r.')
        metric, unit = row['relative_error'], 'rel'
    elif pivot:
        band = v.alpha_band_moment_pivot
        row['band'] = band
        row['metric'] = 'pivot_chords'
        row['fd_verifiable'] = True
        row['note_kind'] = 'pivot'
        row['note'] = f'pivot error {row["pivot_chord_error"]:.2e} chords'
        if scale is None or not np.isfinite(scale) or scale <= 0:
            row['pass'] = False
            row['verdict'] = 'fail'
            return (f'{label}: pivot-chord gate needs dC_N/d({wrt}) (the C_L and C_D '
                    f'rows and the point state) as its normalizer and it is '
                    f'{"missing" if scale is None else repr(scale)}.')
        if 'fd_noise_abs' in row:
            row['fd_noise'] = row['fd_noise_abs'] / scale
        metric, unit = row['pivot_chord_error'], 'chords'
        why = f' [pivot-chords, /|dC_N/dalpha|={scale:.6e}]'
    else:
        band = v.alpha_row_band(functional)
        row['band'] = band
        row['metric'] = 'relative'
        row['fd_verifiable'] = True
        moment = functional == 'C_M' and v.alpha_band_moment > 0
        row['note_kind'] = 'moment' if moment else None
        row['note'] = 'moment band' if moment else ''
        if band is None:
            return _unbanded(row, label, 'alpha_band')
        metric, unit = row['relative_error'], 'rel'
        why = ' [moment]' if moment else ''
    row['pass'] = bool(np.all(np.isfinite([a, f, metric, band])) and metric <= band)
    row['verdict'] = 'pass' if row['pass'] else 'fail'
    if np.isfinite(metric) and np.isfinite(band) and band > 0:
        row['band_ratio'] = float(metric / band)
    if row['pass']:
        return None
    noise = row.get('fd_noise')
    if noise is not None and np.isfinite(metric) and metric <= noise:
        row['pass'] = None
        row['verdict'] = FD_NOISE_VERDICT
        return None
    return (f'{label}: analytic={a: .8e}  fd={f: .8e}  '
            f'{unit}={metric:.3e} > {band:.1e}{why}')


def _unbanded(row, label, band_key):
    """A judged run reached a row no stated band covers: fail it, by name."""
    row['pass'] = False
    row['verdict'] = 'fail'
    return (f'{label}: no band stated for this row; set [verification] '
            f'{band_key}')


def _fmt_band(value, spec):
    """``value`` in ``spec``, or a ``-`` of the same width when it is unset."""
    width = spec.split('.')[0]
    return format('-', f'>{width}') if value is None else format(value, spec)


def format_check_totals(rows, verification):
    """Adjoint and FD side by side, with a legend for the notes that occur.

    Geometric-constraint rows are summarized as one line per constraint (row
    count and largest ``|adj-FD|``); the results file keeps every row.
    """
    geometric = [r for r in rows if r.get('note_kind') == 'geometry']
    rows = [r for r in rows if r.get('note_kind') != 'geometry']
    head = (f'{"derivative":42s} {"adjoint":>15s} {"FD":>15s} {"|adj-FD|":>9s} '
            f'{"rel diff":>9s} {"FD noise":>9s} {"C_r":>7s} {"band":>8s} {"x band":>7s}  '
            f'{"verdict":15s} note')
    lines = ['check_totals: adjoint vs finite difference', head, '-' * len(head)]
    for r in rows:
        cr = r.get('cancellation_ratio')
        lines.append(f'{r["derivative"]:42s} {r["analytic"]:15.8e} {r["fd"]:15.8e} '
                     f'{r["abs_difference"]:9.2e} {r["relative_error"]:9.2e} '
                     f'{_fmt_band(r.get("fd_noise"), "9.2e")} '
                     f'{_fmt_band(cr, "7.1f")} {_fmt_band(r["band"], "8.1e")} '
                     f'{_fmt_band(r.get("band_ratio"), "7.2f")}  '
                     f'{r["verdict"]:15s} {r["note"]}')
    lines += _geometry_summary(geometric)
    kinds = [k for k in CHECK_TOTALS_LEGEND
             if any(r.get('note_kind') == k for r in rows + geometric)]
    if kinds:
        lines.append('')
        lines.append('Why adjoint and FD can differ here:')
        lines += [f'  - {CHECK_TOTALS_LEGEND[k]}' for k in kinds]
    if any(r.get('fd_noise') is not None for r in rows):
        lines.append('FD noise: the smallest difference this run\'s finite difference '
                     'resolves, from each perturbed primal\'s drift over its last '
                     f'{FD_DRIFT_WINDOW} iterations divided by the step (same units as '
                     'the row\'s metric). x band: the metric divided by its band.')
    v = verification
    if v.judged:
        def stated(value):
            return 'unset' if value is None else f'{value:.0e}'
        tiered = (f'max({stated(v.shape_band_floor)}, {stated(v.shape_band_slope)} x C_r) '
                  f'where C_r is measured, else '
                  if v.shape_band_floor is not None or v.shape_band_slope is not None
                  else '')
        lines.append(f'Bands: alpha {stated(v.alpha_band)} relative; shape {tiered}'
                     f'{stated(v.shape_band)} flat; rows whose band would exceed '
                     f'{v.shape_band_cap:.1e} are not judged.')
    return lines


def _geometry_summary(rows):
    """One line per geometric constraint: row count, largest ``|adj-FD|``, not judged."""
    if not rows:
        return []
    by_output = {}
    for r in rows:
        by_output.setdefault(r['of'], []).append(r)
    width = max(len(of) for of in by_output)
    lines = ['', 'Geometric constraints (every row is in results_check_totals.json):']
    for of, group in by_output.items():
        largest = max(r['abs_difference'] for r in group)
        lines.append(f'  {of:<{width}}  {len(group):6d} rows   largest |adj-FD| '
                     f'{largest:9.2e}   not judged')
    return lines


def _row_reason(row):
    """One line: how far a row is from its band, and what the FD can resolve."""
    ratio = row.get('band_ratio')
    if ratio is None:
        return row.get('failure') or row['derivative']
    metric = row['pivot_chord_error'] if row.get('metric') == 'pivot_chords' \
        else row['relative_error']
    what = 'pivot error' if row.get('metric') == 'pivot_chords' else 'rel diff'
    text = (f'{row["derivative"]}: {what} {metric:.2e} is {ratio:.1f}x its band '
            f'{row["band"]:.1e}')
    noise = row.get('fd_noise')
    if noise is not None:
        relation = 'inside' if metric <= noise else 'above'
        text += f', {relation} the FD noise {noise:.1e}'
    return text


def summarize_check_totals(rows, verdict):
    """The verdict and, for each row that did not pass, how far off it is."""
    judged = [r for r in rows if r.get('band') is not None]
    failed = [r for r in rows if r.get('failure')]
    noisy = [r for r in rows if r['verdict'] == FD_NOISE_VERDICT]
    if verdict == 'not judged':
        return [NO_BAND_WARNING]
    if verdict == 'pass':
        return ['check_totals PASSED: every FD-verifiable row is inside its band.']
    lines = []
    if verdict == 'fail':
        lines.append(f'check_totals FAILED: {len(failed)} of {len(judged)} judged rows '
                     f'are outside their band by more than the FD resolves:')
        lines += [f'  {_row_reason(r)}' for r in failed]
        if noisy:
            lines.append(f'Also {len(noisy)} row(s) outside their band but inside the FD noise:')
    else:
        lines.append(f'check_totals INCONCLUSIVE: no row fails, but {len(noisy)} of '
                     f'{len(judged)} judged rows are outside their band by less than the '
                     f'FD resolves, so FD cannot tell them from agreement:')
    lines += [f'  {_row_reason(r)}' for r in noisy]
    if noisy:
        lines.append('To judge them, converge the primal further ([solver.primal] '
                     'conv_residual_minval) or use a larger [verification] step.')
    return lines
