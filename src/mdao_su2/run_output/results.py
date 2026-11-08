"""What a run leaves behind: results JSON and the ``run_driver`` evaluation tree.

``evaluations/`` holds ``baseline/``, ``calls/call_NNNN/`` (one per function
evaluation the optimizer requests), ``iterations/iter_NNN/`` (one per major
iteration, hard links to its call folder's files) and ``final/``. Every folder
has a ``call_summary.json``; the run root has ``optimization_summary.json``
"""

import glob
import hashlib
import json
import os
import shutil

import numpy as np
from openmdao.recorders.case_recorder import CaseRecorder

from ..mpi import on_root
from ..optimization.progress import adjoint_label
from ..vtu import extract_mesh_from_vtu, vtu_to_csv
from .geometry_artifacts import (geometry_record, write_parameterization_artifacts,
                                 write_wing_description)
from .layout import ARTIFACTS, adjoint_stage_name, keep_volume, source_artifact

__all__ = ['write_snapshot', 'call_summary', 'IterationSnapshotter',
           'IterationPrinter']

#: Optimizers whose history file marks major iterations (``isMajor``).
#: pyOptSparse sets ``isMajor`` on every record for any other optimizer.
HISTORY_MAJOR_OPTIMIZERS = ('SNOPT', 'IPOPT', 'Uno')

#: The volume field of a primal or adjoint stage.
VOLUME = ARTIFACTS['volume']

_REASON_REJECTED = 'no gradient requested at this design'
_REASON_NOT_MAJOR = 'not marked major in the optimizer history'


# ------------------------------------------------------- file helpers

def _write_json(path, data):
    """Write JSON atomically: a reader never sees a half-written file."""
    temporary = f'{path}.tmp'
    with open(temporary, 'w') as stream:
        json.dump(data, stream, indent=2)
    os.replace(temporary, path)


def stage_name(kind, point_id, cfg):
    """``primal`` or ``adjoint_<functional>``, suffixed with the point when multipoint."""
    return kind if len(cfg.points) == 1 else f'{kind}_{point_id}'


def _copy_if(source, directory, name):
    if source is None or not os.path.isfile(source):
        return False
    os.makedirs(directory, exist_ok=True)
    shutil.copy2(source, os.path.join(directory, name))
    return True


def _is_stage(name):
    return name == 'geometry' or name.startswith(('primal', 'adjoint_'))


def _manifest_from_disk(folder):
    """``{stage: [files]}`` for every stage folder with files."""
    manifest = {}
    if not os.path.isdir(folder):
        return manifest
    for name in sorted(os.listdir(folder)):
        directory = os.path.join(folder, name)
        if _is_stage(name) and os.path.isdir(directory):
            files = sorted(entry for entry in os.listdir(directory)
                           if os.path.isfile(os.path.join(directory, entry)))
            if files:
                manifest[name] = files
    return manifest


def _organize_stages(work, folder, cfg, rundir, want_volume):
    """Copy each point's primal output into ``primal[_<point>]/`` with plain names."""
    def _cfg_name_safe(point):
        try:
            from ..solver.point_cfg_writer import cfg_name
            return cfg_name(cfg, point, False)
        except (AttributeError, TypeError):
            return None

    su2 = cfg.output.su2
    for p in cfg.points:
        stage = os.path.join(folder, stage_name('primal', p.id, cfg))
        copied = []
        runtime = (
            (source_artifact(work, 'history', p, '.csv'), ARTIFACTS['history']),
            (source_artifact(work, su2.surface_basename, p, '.vtu'), ARTIFACTS['surface']),
            (source_artifact(work, su2.restart_basename, p, '.dat'), ARTIFACTS['restart']),
        )
        for source, name in runtime:
            if _copy_if(source, stage, name):
                copied.append(name)
                if name == ARTIFACTS['surface']:
                    vtu_to_csv(os.path.join(stage, name),
                               os.path.join(stage, 'surface.csv'))
        if want_volume:
            _copy_if(source_artifact(work, su2.volume_basename, p, '.vtu'), stage, VOLUME)
        # A generated config alone does not prove that a solve ran.
        if copied and rundir is not None:
            name = _cfg_name_safe(p)
            if name:
                _copy_if(os.path.join(rundir.su2_cfg, name), stage, ARTIFACTS['config'])
    return _manifest_from_disk(folder)


def _write_geometry(rundir, cfg, folder, baseline, surface, shape_dvs,
                    geometry_values, baseline_surface):
    """``geometry/``: design, surface, mesh, indicator; returns the summary block."""
    geometry_dir = os.path.join(folder, 'geometry')
    os.makedirs(geometry_dir, exist_ok=True)
    np.save(os.path.join(geometry_dir, 'surface_coordinates.npy'), surface)
    if shape_dvs is not None:
        np.save(os.path.join(geometry_dir, 'design_variables.npy'), shape_dvs)
    record = geometry_record(geometry_values, surface)
    pts = np.asarray(surface)
    primal_surfaces = sorted(glob.glob(os.path.join(folder, 'primal*', ARTIFACTS['surface'])))
    if pts.ndim == 2 and pts.shape[1] == 3 and np.ptp(pts[:, 2]) > 1e-12 \
            and primal_surfaces:
        record['wing_section_failures'] = write_wing_description(
            os.path.join(geometry_dir, 'wing_description.csv'), primal_surfaces[0])
    flow = (source_artifact(rundir.workdir, cfg.output.su2.volume_basename, cfg.points[0], '.vtu')
            if not baseline and cfg.points else None)
    mesh_source = None
    baseline_mesh = getattr(cfg, 'mesh_path', None)
    if baseline and baseline_mesh and os.path.isfile(baseline_mesh):
        mesh_source = baseline_mesh
    elif not baseline:
        generated = glob.glob(os.path.join(rundir.mesh, 'mesh_current*.su2'))
        if generated:
            mesh_source = max(generated, key=os.path.getmtime)
    if mesh_source:
        shutil.copy2(mesh_source, os.path.join(geometry_dir, 'mesh.su2'))
    elif flow and baseline_mesh and os.path.isfile(baseline_mesh):
        from ..mesh.su2_mesh_file import write_deformed_su2_from_vtu
        write_deformed_su2_from_vtu(
            baseline_mesh, flow, os.path.join(geometry_dir, 'mesh.su2'))
    if flow:
        try:
            extract_mesh_from_vtu(flow, os.path.join(geometry_dir, 'mesh.vtu'))
        except Exception:
            pass
    elif mesh_source:
        try:
            from ..mesh.su2_mesh_file import su2_mesh_to_vtu
            su2_mesh_to_vtu(mesh_source, os.path.join(geometry_dir, 'mesh.vtu'))
        except Exception:
            pass
    write_parameterization_artifacts(cfg, geometry_dir, surface, shape_dvs,
                                     baseline_surface)
    return record


# ------------------------------------------------------- call_summary.json

def call_summary(cfg, call=None, status='baseline', cds=None, cls=None,
                 alphas=None, failure=None, geometry=None, adjoints=None,
                 stages=None, major_iteration=None, has_gradient=False):
    """One shape for ``baseline``, every call, and ``final``.

    A failed call records ``null`` coefficients and its failure message,
    never the values the model still holds from an earlier call.
    """
    failed = failure is not None

    def _at(seq, i):
        return None if (seq is None or failed) else float(seq[i])

    points = [{'id': p.id, 'alpha': _at(alphas, i), 'C_L': _at(cls, i),
               'C_D': _at(cds, i)} for i, p in enumerate(cfg.points)]
    return {'call': call,
            'major_iteration': major_iteration,
            'has_gradient': bool(has_gradient),
            'status': status,
            'failure': failure,
            'points': points,
            'geometry_constraints': geometry,
            'adjoints': adjoints or {},
            'stages': stages or {}}


def write_snapshot(rundir, cfg, folder, comm=None, call=None, status='baseline',
                   surface=None, shape_dvs=None, cds=None, cls=None,
                   alphas=None, fields='none', want_volume=False,
                   geometry_values=None, baseline_surface=None, failure=None):
    """Write one evaluation folder and its ``call_summary.json``.

    ``folder`` is relative to ``evaluations/`` (``baseline``,
    ``calls/call_0003``, ``final``). Collective: call it on every rank.
    Returns the summary.
    """
    def _do_write():
        path = os.path.join(rundir.evaluations, folder)
        os.makedirs(path, exist_ok=True)
        if fields != 'none':
            _organize_stages(rundir.workdir, path, cfg, rundir,
                             want_volume or fields == 'volume')
        geometry = None
        if surface is not None and hasattr(cfg, 'geometry'):
            geometry = _write_geometry(rundir, cfg, path, folder == 'baseline', surface,
                                       shape_dvs, geometry_values,
                                       baseline_surface)
        summary = call_summary(cfg, call=call, status=status, cds=cds, cls=cls,
                               alphas=alphas, failure=failure,
                               geometry=geometry,
                               stages=_manifest_from_disk(path))
        _write_json(os.path.join(path, 'call_summary.json'), summary)
        return summary

    if comm is None:
        return _do_write()
    return on_root(comm, _do_write, what=f'write evaluation {folder!r}')


# ------------------------------------------------------- the call tracker

def _design_vector(dv):
    """A pyOptSparse design dict as one flat vector, keys in sorted order."""
    if dv is None:
        return None
    return np.concatenate([np.asarray(dv[k], dtype=float).ravel()
                           for k in sorted(dv)] or [np.zeros(0)])


def _digest(*arrays):
    sha = hashlib.sha1()
    for array in arrays:
        sha.update(np.ascontiguousarray(np.asarray(array, dtype=float)).tobytes())
    return sha.hexdigest()


def _link_tree(source, destination):
    """Hard-link every file of ``source`` under ``destination``; copy if links fail.

    ``call_summary.json`` is always copied: it is rewritten atomically, which
    would leave a hard link pointing at the old file.
    """
    for base, _dirs, names in os.walk(source):
        target = os.path.join(destination, os.path.relpath(base, source))
        os.makedirs(target, exist_ok=True)
        for name in names:
            src, dst = os.path.join(base, name), os.path.join(target, name)
            if name.endswith('.tmp'):
                continue
            if name == 'call_summary.json':
                shutil.copy2(src, dst)
                continue
            if os.path.exists(dst):
                continue
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)


class IterationSnapshotter(CaseRecorder):
    """Write every optimizer function evaluation and classify major iterations.

    ``run_driver`` wraps the driver's ``_objfunc`` and ``_gradfunc``
    (``solve_hooks.install_solve_hooks``) to call ``start_call``,
    ``end_call`` and ``gradient_requested`` on every rank; the driver
    recorder hook writes the call's folder. A call's adjoints arrive after
    its record, so a call folder is complete, and its iteration folder is
    synced, when the next call starts and at ``finish``.

    Classification: under SNOPT, IPOPT and Uno, pyOptSparse's ``isMajor`` in
    the history file. Otherwise (SLSQP) a call is major exactly when the next
    request is a gradient at the same design
    The human guide explains the call and major-iteration layout.
    """

    def __init__(self, rundir, cfg, comm, baseline_surface=None,
                 baseline_dvs=None, hist_file=None):
        super().__init__(record_viewer_data=False)
        self.rundir = rundir
        self.cfg = cfg
        self.comm = comm
        self.snapshots = cfg.output.snapshots
        self.optimizer = cfg.optimizer.name
        self.hist_file = hist_file
        self.baseline_surface = baseline_surface
        self.baseline_dvs = baseline_dvs
        self.calls = []            # one entry per optimizer function evaluation
        self.summaries = {}        # folder -> call_summary dict (rank 0)
        self.synced = {}           # major iteration -> iteration folder
        self.volumes = []          # folders still holding volume fields
        self.current = None        # folder receiving this call's stages
        self.in_call = False
        self.finished = False
        self.failure = None
        self.printer = None
        self._final = None
        # Fire on every rank; do NOT make OpenMDAO gather variables for us --
        # the snapshot reads what it needs from the problem itself.
        self._parallel = True
        self._record_on_proc = True
        self._do_gather = False

    # -- recorder protocol: everything except the driver hook is a no-op ----
    def startup(self, recording_requester, comm=None):
        pass

    def record_metadata_system(self, *a, **k):
        pass

    def record_metadata_solver(self, *a, **k):
        pass

    def record_viewer_data(self, *a, **k):
        pass

    def record_derivatives(self, *a, **k):
        pass

    def record_derivatives_driver(self, *a, **k):
        pass

    def record_iteration_system(self, *a, **k):
        """Not recorded: a System records many times per optimizer call."""

    def record_iteration_solver(self, *a, **k):
        pass

    def record_iteration_problem(self, *a, **k):
        pass

    def shutdown(self):
        pass

    def record_iteration_driver(self, driver, data, metadata):
        self.take(driver=driver)

    # -- what the console and the hooks ask ------------------------------
    @property
    def writing(self):
        return self.snapshots.mode != 'off'

    def call_label(self, kind=None):
        """The progress-line name of the solve now running: ``call <n>``.

        A primal outside any call once the optimizer has started is the
        driver's re-run at the final design. Adjoints run between calls,
        for the call before them.
        """
        if self.finished or (kind == 'primal' and self.calls and not self.in_call):
            return 'final'
        if not self.calls:
            return 'call initial'
        return f'call {self.calls[-1]["call"]}'

    def call_failed(self, message):
        """A primal of the current call raised; its coefficients are not valid."""
        if self.in_call and self.failure is None:
            self.failure = str(message)

    def adjoint_done(self, objective, point, dj_dalpha, design_hash):
        """Rank 0: one functional adjoint captured into the current folder."""
        if self.current is None or self.current not in self.summaries:
            return
        entry = self.summaries[self.current]['adjoints'].setdefault(
            adjoint_label(objective), {})
        entry[point] = {'su2_objective': str(objective),
                        'dJ_dalpha': float(dj_dalpha),
                        'design_hash': design_hash}

    # -- call boundaries (every rank) -------------------------------------
    def start_call(self, dv):
        """The optimizer requests a new function evaluation."""
        self._close_previous()
        index = len(self.calls)
        self.calls.append({'call': index, 'folder': None, 'status': 'pending',
                           'major_iteration': None, 'has_gradient': False,
                           'reason': None, 'objective': None,
                           'design': _design_vector(dv), 'digest': None})
        self.failure = None
        self.in_call = True
        every = max(int(getattr(self.snapshots, 'every_n', 1) or 1), 1)
        self.current = (f'calls/call_{index:04d}'
                        if self.writing and index % every == 0 else None)
        self._notify('call_started')

    def end_call(self):
        self.in_call = False
        self._notify('call_ended')

    def _notify(self, name):
        """Tell the console printer that the latest call started or ended."""
        hook = getattr(self.printer, name, None)
        if hook is not None and self.calls:
            hook(self.calls[-1])

    def gradient_requested(self, dv):
        """The optimizer asks for totals; under SLSQP this marks a major iteration."""
        design = _design_vector(dv)
        match = None
        for entry in reversed(self.calls):
            if entry['design'] is not None and design is not None \
                    and entry['design'].shape == design.shape \
                    and np.array_equal(entry['design'], design):
                match = entry
                break
        if match is None:
            return
        match['has_gradient'] = True
        if self.optimizer not in HISTORY_MAJOR_OPTIMIZERS \
                and match is self.calls[-1] and match['status'] != 'failed':
            self._promote(match)
        self._write_summaries([match])

    def take(self, driver=None):
        """Write the current call's folder once its primal has run."""
        if not self.in_call or not self.calls:
            return None          # pyOptSparse's initial run or the final re-run
        entry = self.calls[-1]
        failures = [self.failure]
        if self.comm is not None:
            failures = self.comm.allgather(self.failure)
        failure = next((f for f in failures if f is not None), None)
        cds = cls = alphas = None
        objective = None
        surface = shape_dvs = geometry_values = None
        if driver is not None:
            cds, cls, alphas = _driver_coefficients(driver, self.cfg)
            objective = _driver_objective(driver)
            surface, shape_dvs, geometry_values = _driver_geometry_state(driver)
        if failure is not None:
            entry['status'] = 'failed'
            entry['reason'] = failure
            objective = None
        entry['objective'] = objective
        if surface is not None:
            entry['digest'] = _digest(surface, shape_dvs if shape_dvs is not None else [])
        entry['summary_points'] = (cds, cls, alphas)
        entry['dv_norms'] = _norms(driver, self.cfg) if driver is not None else None
        if self.current is None:
            return None
        if entry['call'] == 0 and failure is None and self._is_baseline(surface, shape_dvs):
            # The first call is the baseline solve: it completes baseline/,
            # and no call folder repeats it.
            self.current = 'baseline'
        entry['folder'] = self.current
        keep = keep_volume(self.snapshots)
        summary = write_snapshot(
            self.rundir, self.cfg, self.current, comm=self.comm,
            call=entry['call'], status=entry['status'], surface=surface,
            shape_dvs=shape_dvs, cds=cds, cls=cls, alphas=alphas,
            fields='surface', want_volume=keep, geometry_values=geometry_values,
            baseline_surface=self._baseline_surface(), failure=failure)
        self.summaries[self.current] = summary
        if keep and self.current not in self.volumes:
            self.volumes.append(self.current)
        self._prune_volumes()
        self._write_overview()
        return self.current

    def finish(self, surface=None, shape_dvs=None, cds=None, cls=None,
               alphas=None, geometry_values=None):
        """Run end: classify the last call, sync, write ``final/`` and both summaries."""
        self._close_previous()
        self.in_call = False
        self.finished = True
        for entry in self.calls:
            if entry['status'] == 'pending':
                self._reject(entry)
        self._write_summaries(self.calls)
        final = None
        if surface is not None:
            final = self._write_final(surface, shape_dvs, cds, cls, alphas,
                                      geometry_values)
        self._prune_volumes()
        self._write_overview(final=final, state='finished')
        return final

    # -- internals ---------------------------------------------------------
    def _baseline_surface(self):
        if self.baseline_surface is None:
            path = os.path.join(self.rundir.evaluations, 'baseline', 'geometry',
                                'surface_coordinates.npy')
            if os.path.isfile(path):
                self.baseline_surface = np.load(path)
        return self.baseline_surface

    def _is_baseline(self, surface, shape_dvs):
        if surface is None or shape_dvs is None:
            return False
        reference = self._baseline_surface()
        design = self.baseline_dvs
        if design is None:
            path = os.path.join(self.rundir.evaluations, 'baseline', 'geometry',
                                'design_variables.npy')
            design = np.load(path) if os.path.isfile(path) else None
        return (reference is not None and design is not None
                and np.array_equal(np.asarray(surface), np.asarray(reference))
                and np.array_equal(np.asarray(shape_dvs), np.asarray(design)))

    def _promote(self, entry):
        entry['status'] = 'accepted'
        entry['reason'] = None
        entry['major_iteration'] = sum(
            1 for e in self.calls if e['major_iteration'] is not None)
        if self.printer is not None:
            self.printer.major(entry)

    def _reject(self, entry):
        entry['status'] = 'rejected'
        entry['reason'] = (_REASON_NOT_MAJOR
                           if self.optimizer in HISTORY_MAJOR_OPTIMIZERS
                           else _REASON_REJECTED)

    def _close_previous(self):
        """Classify what is known, sync iteration folders, print new majors."""
        if not self.calls:
            return
        if self.optimizer in HISTORY_MAJOR_OPTIMIZERS:
            self._classify_from_history()
        else:
            last = self.calls[-1]
            if last['status'] == 'pending':
                self._reject(last)
        self._write_summaries(self.calls)
        self._sync_iterations()
        self._prune_volumes()
        self._write_overview()

    def _classify_from_history(self):
        """Mark calls whose design the history file records as major."""
        def read():
            # The optimizer writes the file as it runs; read the records
            # directly rather than through pyOptSparse's History post-processing.
            from sqlitedict import SqliteDict
            if not self.hist_file or not os.path.isfile(self.hist_file):
                return []
            designs = []
            with SqliteDict(self.hist_file, flag='r') as db:
                for key in db.keys():
                    if not str(key).isdigit():
                        continue
                    record = db[key]
                    if record.get('isMajor') and 'xuser' in record:
                        designs.append(_design_vector(record['xuser']))
            return designs

        designs = (read() if self.comm is None
                   else on_root(self.comm, read, what='read optimizer history'))
        for entry in self.calls:
            if entry['major_iteration'] is not None or entry['status'] == 'failed':
                continue
            if any(d is not None and entry['design'] is not None
                   and d.shape == entry['design'].shape
                   and np.array_equal(d, entry['design']) for d in designs):
                self._promote(entry)
        for entry in self.calls[:-1]:
            if entry['status'] == 'pending':
                later = [e for e in self.calls if e['call'] > entry['call']
                         and e['major_iteration'] is not None]
                if later:
                    self._reject(entry)

    def _iteration_folder(self, entry):
        return f'iterations/iter_{entry["major_iteration"]:03d}'

    def _sync_iterations(self):
        """Fill ``iterations/iter_NNN`` for every classified major call folder."""
        if not self.writing:
            return
        # Called when the next call starts and at run end: every classified
        # major call folder is complete by then, adjoints included.
        todo = [e for e in self.calls if e['major_iteration'] is not None
                and e['folder'] is not None and e['major_iteration'] not in self.synced]
        if not todo:
            return

        def sync():
            for entry in todo:
                source = os.path.join(self.rundir.evaluations, entry['folder'])
                target = os.path.join(self.rundir.evaluations,
                                      self._iteration_folder(entry))
                if os.path.isdir(source):
                    _link_tree(source, target)

        self._root(sync, 'sync iteration folders')
        for entry in todo:
            folder = self._iteration_folder(entry)
            self.synced[entry['major_iteration']] = folder
            if entry['folder'] in self.volumes and folder not in self.volumes:
                self.volumes.append(folder)

    def _summary_for(self, entry):
        summary = self.summaries.get(entry['folder'])
        if summary is None:
            return None
        summary.update(call=entry['call'], status=entry['status'],
                       major_iteration=entry['major_iteration'],
                       has_gradient=entry['has_gradient'])
        if entry['status'] == 'failed':
            summary['failure'] = entry['reason']
        return summary

    def _write_summaries(self, entries):
        """Rewrite each entry's ``call_summary.json`` (and its iteration copy)."""
        if not self.writing:
            return
        entries = [e for e in entries if e['folder'] is not None]
        if not entries:
            return

        def write():
            for entry in entries:
                summary = self._summary_for(entry)
                if summary is None:
                    continue
                folder = os.path.join(self.rundir.evaluations, entry['folder'])
                summary['stages'] = _manifest_from_disk(folder)
                _write_json(os.path.join(folder, 'call_summary.json'), summary)
                synced = self.synced.get(entry['major_iteration'])
                if synced is not None:
                    _write_json(os.path.join(self.rundir.evaluations, synced,
                                             'call_summary.json'), summary)

        self._root(write, 'write call summaries')

    def _prune_volumes(self):
        """``baseline_latest``: one rule for both trees.

        Keep the volume fields on ``baseline/``, the newest call, and the
        latest major iteration (its call folder and its iteration folder).
        """
        if self.snapshots.keep_volume != 'baseline_latest' or not self.writing:
            return
        keep = {'baseline', 'final'}
        if self.calls and self.calls[-1]['folder'] is not None:
            keep.add(self.calls[-1]['folder'])
        majors = [e for e in self.calls if e['major_iteration'] is not None]
        if majors:
            latest = majors[-1]
            if latest['folder'] is not None:
                keep.add(latest['folder'])
            keep.add(self._iteration_folder(latest))
        drop = [folder for folder in self.volumes if folder not in keep]
        if not drop:
            return

        def prune():
            for folder in drop:
                path = os.path.join(self.rundir.evaluations, folder)
                for stage in glob.glob(os.path.join(path, '*', VOLUME)):
                    os.remove(stage)

        self._root(prune, 'prune superseded volume fields')
        self.volumes = [folder for folder in self.volumes if folder in keep]
        touched = [e for e in self.calls if e['folder'] in drop
                   or self.synced.get(e['major_iteration']) in drop]
        self._write_summaries(touched)

    def _write_final(self, surface, shape_dvs, cds, cls, alphas, geometry_values):
        """``final/``: links to the matching call folder, or a fresh write."""
        digest = _digest(surface, shape_dvs if shape_dvs is not None else [])
        matches = [e for e in self.calls if e['digest'] == digest
                   and e['folder'] is not None and e['status'] != 'failed']
        majors = [e for e in matches if e['major_iteration'] is not None]
        match = (majors or matches or [None])[-1]
        keep = self.snapshots.keep_volume != 'none'
        if match is None:
            summary = write_snapshot(
                self.rundir, self.cfg, 'final', comm=self.comm, call=None,
                status='final', surface=surface, shape_dvs=shape_dvs, cds=cds,
                cls=cls, alphas=alphas,
                fields='none' if not self.writing else 'surface', want_volume=keep,
                geometry_values=geometry_values,
                baseline_surface=self._baseline_surface())
            self.summaries['final'] = summary
            return {'folder': 'evaluations/final', 'call': None}

        def link():
            source = os.path.join(self.rundir.evaluations, match['folder'])
            target = os.path.join(self.rundir.evaluations, 'final')
            _link_tree(source, target)
            summary = dict(self.summaries.get(match['folder']) or {})
            summary['status'] = 'final'
            summary['stages'] = _manifest_from_disk(target)
            _write_json(os.path.join(target, 'call_summary.json'), summary)
            return summary

        self.summaries['final'] = self._root(link, 'write final evaluation')
        return {'folder': 'evaluations/final', 'call': match['call']}

    def _write_overview(self, final=None, state='running'):
        """Rewrite the run-root ``optimization_summary.json``."""
        def relative(folder):
            return None if folder is None else f'evaluations/{folder}'

        calls = [{'call': e['call'], 'folder': relative(e['folder']),
                  'status': e['status'], 'major_iteration': e['major_iteration'],
                  'has_gradient': e['has_gradient'], 'objective': e['objective'],
                  'reason': e['reason']} for e in self.calls]
        iterations = [{'iteration': e['major_iteration'], 'call': e['call'],
                       'folder': relative(self.synced.get(e['major_iteration'])),
                       'objective': e['objective']}
                      for e in self.calls if e['major_iteration'] is not None]
        rule = ('pyOptSparse isMajor' if self.optimizer in HISTORY_MAJOR_OPTIMIZERS
                else 'gradient request at the same design')
        overview = {'optimizer': self.optimizer, 'major_iteration_rule': rule,
                    'state': state, 'baseline': 'evaluations/baseline',
                    'final': final if final is not None else self._final,
                    'iterations': iterations, 'calls': calls}
        if final is not None:
            self._final = final

        def write():
            _write_json(os.path.join(self.rundir.root, 'optimization_summary.json'),
                        overview)

        self._root(write, 'write optimization summary')

    def _root(self, fn, what):
        if self.comm is None:
            return fn()
        return on_root(self.comm, fn, what=what)


def _norms(driver, cfg):
    """``(count, ||x||, max|x|)`` over every geometry DV, for the console."""
    try:
        prob = driver._problem()
        values = np.concatenate([np.asarray(prob.get_val(n)).ravel()
                                 for n in cfg.geometry_dvs] or [np.zeros(0)])
        return (len(values), float(np.linalg.norm(values)),
                float(np.max(np.abs(values))) if len(values) else 0.0)
    except Exception:
        return None


def _driver_objective(driver):
    try:
        return float(driver._problem().get_val('obj_comb.obj')[0])
    except Exception:
        return None


class IterationPrinter:
    """Console lines of ``run_driver``'s optimizer calls and major iterations.

    ``IterationSnapshotter`` calls ``call_started`` and ``call_ended`` around
    every function evaluation, which ``progress`` turns into one call line,
    and ``major`` once a call is classified as a major iteration: a
    ``major <n> (call <c>)`` line with the objective and its change from the
    baseline, then a block with each point's coefficients and the DV norms.
    The major iteration ends its segment: ``progress`` separates the next
    call from it with a rule.
    """

    def __init__(self, cfg, terminal, progress=None):
        self.cfg = cfg
        self._term = terminal
        self.progress = progress
        self._baseline_cds = None
        self._baseline_objective = None

    def set_baseline(self, cds):
        self._baseline_cds = list(cds) if cds is not None else None

    def call_started(self, entry):
        if self.progress is not None:
            self.progress.call_started(entry)

    def call_ended(self, entry):
        if self.progress is not None:
            self.progress.call_ended(entry)

    def _say(self, line=''):
        if self.progress is not None:
            self.progress.say(line)
        else:
            print(line, file=self._term, flush=True)

    def major(self, entry):
        if self._term is None and self.progress is None:
            return
        cds, cls, alphas = entry.get('summary_points') or (None, None, None)
        objective = entry.get('objective')
        if self._baseline_objective is None:
            self._baseline_objective = objective
        compare = self._baseline_cds is not None
        if self._baseline_cds is None and cds is not None:
            # Without a trimmed baseline, major 0 (the starting design) is it.
            self._baseline_cds = list(cds)
        line = f'major {entry["major_iteration"]} (call {entry["call"]})'
        if objective is not None:
            line += f'  obj {objective:.6e}'
            base = self._baseline_objective
            if entry['major_iteration'] == 0 or base is None:
                line += '  (baseline)'
            elif base:
                line += f'  ({(objective / base - 1.0) * 100:+.2f}% vs baseline)'
        sep = '-' * 72
        self._say(line)
        self._say(sep)
        if cds is not None:
            for i, p in enumerate(self.cfg.points):
                delta = ''
                if compare and self._baseline_cds[i]:
                    delta = (f'  dC_D={(cds[i] / self._baseline_cds[i] - 1.0) * 100:+.2f}%')
                self._say(f'  {p.id:>8s}  M={p.mach:.2f}  alpha={alphas[i]:8.4f}  '
                          f'C_L={cls[i]:.6f}  C_D={cds[i]:.6f}{delta}')
        norms = entry.get('dv_norms')
        if norms is not None:
            n_dv, dv_norm, dv_max = norms
            self._say(f'  Shape DVs ({n_dv}):  ||x||={dv_norm:.6e}  max|x|={dv_max:.6e}')
        self._say(sep)
        if self.progress is not None:
            self.progress.end_segment()


def _driver_coefficients(driver, cfg):
    """Best-effort C_D/C_L/alpha for this iteration, from the driver's own vectors."""
    from ..optimization.model import gather_coeffs
    try:
        prob = driver._problem()
        # gather_coeffs is itself collective and is called here on every rank,
        # which is the whole reason this recorder sets _parallel = True.
        cls, cds = gather_coeffs(prob, cfg)
        alphas = [float(prob.get_val(f'alpha_{i}')[0])
                  for i in range(len(cfg.points))]
        return cds, cls, alphas
    except Exception:
        return None, None, None


def _driver_geometry_state(driver):
    """Collect the current distributed geometry and registered constraint values."""
    try:
        prob = driver._problem()
        builder = prob.model._meta_builder
        geom = prob.model.geometry_name
        from mphys import MPhysVariables
        output = MPhysVariables.Aerodynamics.Surface.Geometry.COORDINATES_OUTPUT
        xyz = prob.get_val(f'{geom}.{output}', get_remote=True)
        surface = np.asarray(xyz).reshape(builder.n_nodes, 3)[:, :builder.ndim]
        cfg = prob.model.options['cfg']
        shape = (np.asarray(prob.get_val('shape')).copy()
                 if 'shape' in cfg.geometry_dvs else np.zeros(0))
        component = getattr(prob.model, geom, None)
        values = {n: np.asarray(prob.get_val(n)).copy()
                  for n in cfg.geometry_dvs if n != 'shape'}
        for name in getattr(component, '_constraint_names', ()):
            try:
                values[name] = np.asarray(prob.get_val(f'{geom}.{name}')).copy()
            except Exception:
                pass
        return surface, shape, values
    except Exception:
        return None, None, None


# ------------------------------------------------------- run-level results

def clear_stale_solutions(cfg, comm):
    """Drop per-point solution/restart files from an earlier run."""
    def _do_clear():
        for p in cfg.points:
            for pattern in (f'solution_flow_{p.id}.dat',
                            f'{cfg.output.su2.restart_basename}_{p.id}.dat',
                            f'solution_adj_{p.id}_*.dat',
                            f'restart_adj_{p.id}_*.dat'):
                for f in glob.glob(pattern):
                    os.remove(f)

    if comm is not None:
        on_root(comm, _do_clear, what='clear stale solutions')
    elif comm is None:
        _do_clear()


def build_result(cfg, cls, cds, alphas, baseline_cds=None):
    points = []
    for i, p in enumerate(cfg.points):
        entry = {'id': p.id, 'alpha': float(alphas[i]), 'C_L': float(cls[i]),
                 'C_D': float(cds[i])}
        if baseline_cds is not None:
            entry['baseline_C_D'] = float(baseline_cds[i])
            entry['delta_CD_pct'] = (
                1.0 - entry['C_D'] / entry['baseline_C_D']) * 100.0
        points.append(entry)
    # run.toml and resolved_config.json hold the name, task, optimizer, Mach
    # and weights; the objective's value and expression are its `responses`
    # entry.
    return {'points': points}


def write_results(cfg, rd, result):
    """The run's one results file, ``results_<task>.json``, at the run root."""
    path = os.path.join(rd.root, f'results_{cfg.execution.task}.json')
    _write_json(path, result)
