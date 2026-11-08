"""Resolve paths inside a run directory or an archived run. No default filenames."""
import glob
import json
import os

from ..optimization.progress import adjoint_label

#: Created for every task; ``run_driver`` also gets ``DRIVER_SUBDIRS``.
SUBDIRS = ('su2_workdir', 'logs')
DRIVER_SUBDIRS = ('opt_hst', 'evaluations')

__all__ = [
    'SUBDIRS', 'DRIVER_SUBDIRS', 'Run', 'RunLayoutError', 'stage_dir', 'keep_volume',
    'point_artifact', 'source_artifact', 'find_run', 'read_status',
    'adjoint_stage_name', 'ARTIFACTS',
    'final_artifact',
]

#: File of each artifact kind inside an evaluation stage folder
#: (``evaluations/<name>/<stage>[_<point>]/``).
ARTIFACTS = {'history': 'history.csv', 'surface': 'surface.vtu', 'volume': 'volume.vtu',
             'restart': 'restart.dat', 'config': 'config.cfg'}

_FINAL_PRIMAL = {'history': 'primal_history.csv', 'surface': 'surface.vtu',
                 'volume': 'volume.vtu', 'restart': 'restart.dat', 'config': 'primal.cfg'}


def final_artifact(stage, kind):
    """File of an artifact kind in the run-root ``final/<point>/`` folder.

    ``primal`` keeps the stage-folder names except ``primal_history.csv`` and
    ``primal.cfg``; an adjoint stage prefixes them (``adjoint_C_D_history.csv``,
    ``adjoint_C_D.cfg``).
    """
    if kind not in ARTIFACTS:
        raise RunLayoutError(f'unknown artifact kind {kind!r}; known: {sorted(ARTIFACTS)}')
    if stage == 'primal':
        return _FINAL_PRIMAL[kind]
    return f'{stage}.cfg' if kind == 'config' else f'{stage}_{ARTIFACTS[kind]}'


def stage_dir(folder, stage, point=None):
    """Return a stage directory (``primal``, ``adjoint_C_D``) inside an evaluation folder."""
    suffix = f'_{point}' if point is not None else ''
    return os.path.join(folder, f'{stage}{suffix}')


def keep_volume(snapshots):
    """Return whether snapshot settings retain volume fields."""
    return (snapshots.mode == 'volume'
            or snapshots.keep_volume in ('all', 'baseline_latest'))


def point_artifact(stem, point, extension):
    """Return the generated point-suffixed artifact name."""
    return f'{stem}_{point}{extension}'


def source_artifact(work, stem, point, extension):
    """Find a point-suffixed SU2 file, falling back to its bare name."""
    for name in (point_artifact(stem, point.id, extension), f'{stem}{extension}'):
        path = os.path.join(work, name)
        if os.path.isfile(path):
            return path
    return None


def find_run(root, run_id=None):
    """Resolve an explicit run directory, or the newest run below ``root``."""
    if run_id is not None:
        path = os.path.join(root, run_id)
        if os.path.isdir(path):
            return path
        raise RunLayoutError(f'Run directory not found: {path}')
    runs = [os.path.join(root, name) for name in os.listdir(root)
            if os.path.isdir(os.path.join(root, name))]
    if not runs:
        raise RunLayoutError(f'No run directories found in {root}')
    return max(runs, key=os.path.getmtime)


def read_status(path):
    """Read a run STATUS file, returning ``None`` when it is absent."""
    status = os.path.join(path, 'STATUS')
    if not os.path.isfile(status):
        return None
    with open(status) as stream:
        return stream.read().strip()


class RunLayoutError(Exception):
    """Raised when a run directory cannot be navigated."""


class Run:
    """Wraps ``cases/<case>/output/<run_id>/`` or an archived run.

    Two layouts: ``evaluations`` (``run_driver``: ``evaluations/<name>/<stage>/``)
    and ``final`` (other tasks: only ``final/<point>/``, file names from
    :func:`final_artifact`). Evaluation names: ``baseline``, ``final``,
    ``call_NNNN`` (under ``evaluations/calls/``), ``iter_NNN`` (under
    ``evaluations/iterations/``), and ``latest`` (``final``, else the last major
    iteration, else ``baseline``); a ``final`` layout knows ``final`` and
    ``latest`` only. A run with several points needs ``point``.
    """

    def __init__(self, path):
        self.path = os.path.abspath(path)
        if not os.path.isdir(self.path):
            raise RunLayoutError(f'Run directory not found: {self.path}')
        if os.path.isdir(os.path.join(self.path, 'evaluations')):
            self.layout = 'evaluations'
        elif os.path.isdir(os.path.join(self.path, 'final')):
            self.layout = 'final'
        else:
            raise RunLayoutError(f'{self.path} has neither evaluations/ nor final/')

    def _final_points(self):
        root = os.path.join(self.path, 'final')
        return sorted(n for n in os.listdir(root) if os.path.isdir(os.path.join(root, n)))

    def _final_point(self, point):
        names = self._final_points()
        if point is None:
            if len(names) != 1:
                raise RunLayoutError(f'{self.path} has points {names}: name one with --point')
            return names[0]
        if point not in names:
            raise RunLayoutError(f'point {point!r} not in {self.path}/final: {names}')
        return point

    def _listing(self, sub, prefix):
        root = os.path.join(self.path, 'evaluations', sub)
        if not os.path.isdir(root):
            return []
        return sorted(name for name in os.listdir(root)
                      if name.startswith(prefix)
                      and os.path.isdir(os.path.join(root, name)))

    def evaluations(self):
        """Every evaluation folder once: baseline, each call, final.

        Iteration folders repeat their call folders and are listed by
        ``iterations()``.
        """
        root = os.path.join(self.path, 'evaluations')
        names = [name for name in ('baseline',)
                 if os.path.isdir(os.path.join(root, name))]
        names += self._listing('calls', 'call_')
        if os.path.isdir(os.path.join(root, 'final')):
            names.append('final')
        return names

    def iterations(self):
        """The major-iteration folders, ``iter_000`` first."""
        return self._listing('iterations', 'iter_')

    def optimization_summary(self):
        """The run-root ``optimization_summary.json`` (``run_driver`` runs)."""
        path = os.path.join(self.path, 'optimization_summary.json')
        if not os.path.isfile(path):
            raise RunLayoutError(f'No optimization_summary.json in run {self.path}')
        with open(path) as stream:
            return json.load(stream)

    def summary(self, snapshot=None):
        """Load an evaluation's ``call_summary.json``."""
        path = os.path.join(self._resolve_snap_dir(snapshot or 'latest'),
                            'call_summary.json')
        if not os.path.isfile(path):
            raise RunLayoutError(
                f'No call_summary.json found in run {self.path} '
                f'(snapshot={snapshot!r})')
        with open(path) as stream:
            return json.load(stream)

    def points(self):
        """Return the list of operating-point ids."""
        if self.layout == 'final':
            return self._final_points()
        try:
            return [point['id'] for point in self.summary().get('points', [])]
        except RunLayoutError:
            return []

    def config(self):
        """Return the run's ``run.toml`` text, or None."""
        root_toml = os.path.join(self.path, 'run.toml')
        if not os.path.isfile(root_toml):
            return None
        with open(root_toml) as f:
            return f.read()

    def status(self):
        """Return the run's STATUS string, or None."""
        return read_status(self.path)

    # ---- path resolvers ----

    def _stage(self, snapshot, stage, point=None):
        """``<folder>/<stage>[_<point>]``; the point suffix only when multipoint."""
        folder = self._resolve_snap_dir(snapshot)
        if point is not None:
            pointed = stage_dir(folder, stage, point)
            if os.path.isdir(pointed):
                return pointed
        plain = stage_dir(folder, stage)
        if os.path.isdir(plain):
            return plain
        candidates = sorted(glob.glob(stage_dir(folder, stage) + '_*'))
        if len(candidates) == 1:
            return candidates[0]
        if candidates:
            names = [os.path.basename(c)[len(stage) + 1:] for c in candidates]
            raise RunLayoutError(f'{folder} has {stage} stages for points {names}: '
                                 f'name one with --point')
        raise RunLayoutError(f'No {stage} stage in {folder}')

    def _file(self, snapshot, stage, name, point=None):
        if self.layout == 'final':
            if snapshot not in ('final', 'latest'):
                raise RunLayoutError(f'{self.path} keeps only final/; snapshot '
                                     f'{snapshot!r} does not exist')
            kind = {v: k for k, v in ARTIFACTS.items()}.get(name)
            if kind is None or stage == 'geometry':
                raise RunLayoutError(f'{name} ({stage}) is not kept in final/<point>/')
            path = os.path.join(self.path, 'final', self._final_point(point),
                                final_artifact(stage, kind))
            if not os.path.isfile(path):
                raise RunLayoutError(f'No {os.path.basename(path)} in {os.path.dirname(path)}')
            return path
        path = os.path.join(self._stage(snapshot, stage, point), name)
        if not os.path.isfile(path):
            raise RunLayoutError(
                f'No {name} in {os.path.dirname(path)}. Volume fields are pruned '
                f'under keep_volume = "baseline_latest".')
        return path

    def artifact_path(self, snapshot, stage, kind, point=None):
        """Any stage's artifact (``kind`` a key of :data:`ARTIFACTS`) in either layout."""
        return self._file(snapshot, stage, ARTIFACTS[kind], point)

    def surface_path(self, snapshot, point=None):
        """The primal surface VTU. No default filename."""
        return self._file(snapshot, 'primal', ARTIFACTS['surface'], point)

    def volume_path(self, snapshot, point=None):
        """The primal volume VTU. No default filename."""
        return self._file(snapshot, 'primal', ARTIFACTS['volume'], point)

    def history_path(self, snapshot, point=None, stage='primal', objective=None):
        """A stage's SU2 history CSV; ``stage='adjoint'`` needs ``objective``."""
        if stage == 'adjoint':
            stage = adjoint_stage_name(objective)
        return self._file(snapshot, stage, ARTIFACTS['history'], point)

    def mesh_vtu_path(self, snapshot):
        """The evaluation's volume mesh VTU in ``geometry/``."""
        return self._file(snapshot, 'geometry', 'mesh.vtu')

    def gradient_path(self, snapshot, point=None, objective=None):
        """The surface sensitivity CSV of a functional's adjoint."""
        return self._file(snapshot, adjoint_stage_name(objective),
                          'sensitivity.csv', point)

    def design_variables_path(self, snapshot):
        """The evaluation's shape DVs, ``geometry/design_variables.npy``."""
        return self._file(snapshot, 'geometry', 'design_variables.npy')

    def _resolve_snap_dir(self, snapshot):
        """Resolve an evaluation name to its folder."""
        if snapshot == 'latest':
            snapshot = self._latest_name()
        root = os.path.join(self.path, 'evaluations')
        if snapshot.startswith('call_'):
            folder = os.path.join(root, 'calls', snapshot)
            if not os.path.isdir(folder):
                folder = self._mapped_call(snapshot) or folder
        elif snapshot.startswith('iter_'):
            folder = os.path.join(root, 'iterations', snapshot)
        else:
            folder = os.path.join(root, snapshot)
        if os.path.isdir(folder):
            return folder
        raise RunLayoutError(
            f'Evaluation not found: {snapshot!r} in run {self.path}')

    def _mapped_call(self, snapshot):
        """A call with no ``calls/`` folder of its own (call 0 is ``baseline/``)."""
        try:
            calls = self.optimization_summary()['calls']
        except (RunLayoutError, KeyError):
            return None
        number = int(snapshot.split('_')[1])
        for entry in calls:
            if entry['call'] == number and entry.get('folder'):
                return os.path.join(self.path, entry['folder'])
        return None

    def _latest_name(self):
        if os.path.isdir(os.path.join(self.path, 'evaluations', 'final')):
            return 'final'
        iterations = self.iterations()
        return iterations[-1] if iterations else 'baseline'


def adjoint_stage_name(objective):
    """``DRAG``/``drag``/``C_D`` -> ``adjoint_C_D``; a moment -> ``adjoint_C_M``."""
    if objective is None:
        raise RunLayoutError('An adjoint stage needs its functional')
    return f'adjoint_{adjoint_label(objective)}'
