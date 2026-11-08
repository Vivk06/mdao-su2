"""Per-run output directories. Rank 0 creates them; all ranks synchronize."""

import ast
import functools
import glob
import gzip
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
import tomllib
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone

from ..mesh.recipe import sha256
from ..mpi import on_root
from ..paths import REPO
from ..run_config.errors import ConfigError
from ..run_config.points import resolved_points_dict
from .layout import DRIVER_SUBDIRS, SUBDIRS


def _toml_val(value):
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, (list, tuple)):
        return '[' + ', '.join(_toml_val(item) for item in value) + ']'
    return json.dumps(str(value))


def _toml_key(key):
    return key if re.fullmatch(r'[A-Za-z0-9_-]+', str(key)) else json.dumps(str(key))


def to_toml(data, prefix=''):
    """TOML text for scalars, lists, tables, and arrays of tables."""
    lines, tables = [], []
    for key, value in data.items():
        if isinstance(value, dict):
            tables.append((key, value))
        elif isinstance(value, list) and value and all(isinstance(item, dict) for item in value):
            tables.append((key, value))
        else:
            lines.append(f'{_toml_key(key)} = {_toml_val(value)}')
    for key, value in tables:
        path = f'{prefix}{_toml_key(key)}'
        if isinstance(value, list):
            for item in value:
                lines += ['', f'[[{path}]]', to_toml(item, path + '.').rstrip('\n')]
        else:
            body = to_toml(value, path + '.').rstrip('\n')
            lines += ['', f'[{path}]'] + ([body] if body else [])
    return '\n'.join(line for line in lines if line is not None) + '\n'

__all__ = ['RunDir', 'make_run_dir', 'finish', 'record_provenance',
           'resolve_run_id', 'seed_solutions', 'seed_plan', 'seed_lines',
           'source_path', 'restart_points', 'resolved_config']

@dataclass(frozen=True)
class RunDir:
    """Absolute paths for one run; ``root`` owns the layout contract."""
    run_id: str
    root: str

    @property
    def workdir(self):
        return os.path.join(self.root, 'su2_workdir')

    @property
    def su2_cfg(self):
        return self.workdir

    @property
    def mesh(self):
        return os.path.join(self.workdir, 'mesh')

    @property
    def opt(self):
        return os.path.join(self.root, 'opt_hst')

    @property
    def evaluations(self):
        return os.path.join(self.root, 'evaluations')

    @property
    def final(self):
        """Each point's last solver artifacts, kept when ``su2_workdir/`` goes."""
        return os.path.join(self.root, 'final')

    @property
    def polar(self):
        """The ``polar`` task's per-point, per-angle solver artifacts."""
        return os.path.join(self.root, 'polar')

    @property
    def logs(self):
        """Solver output, progress lines and per-solve histories."""
        return os.path.join(self.root, 'logs')

    @property
    def run_toml(self):
        return os.path.join(self.root, 'run.toml')

    @property
    def zip_path(self):
        return self.root.rstrip(os.sep) + '.zip'

    def zip(self, **kwargs):
        from .archive import zip_run
        return zip_run(self.root, **kwargs)

    def enter(self):
        os.chdir(self.workdir)
        return self.workdir


def resolve_run_id(cfg, now=None):
    run_id = cfg.output.run_id
    if run_id == 'auto':
        now = now or datetime.now(timezone.utc)
        run_id = f'{now.strftime("%Y%m%dT%H%M")}_{cfg.name}'
    if (run_id in ('', '.', '..') or os.path.basename(run_id) != run_id
            or '\\' in run_id):
        raise ConfigError('[output] resolved run_id must name one directory under root')
    return run_id


def _output_root(cfg):
    return cfg.abs_path(cfg.output.root)


def resolved_config(cfg):
    """What the run was set up to solve: each resolved point once, then the formulation.

    ``resolved_config.json``. The formulation's per-point
    weight records repeat the resolved points, so only their index is kept.
    A run with ``[dv] start`` or ``[[output.seed]]`` also records the design
    source and every seed file with its SHA-256.
    """
    from .design_start import read_start, start_record
    formulation = dict(cfg.objective_formulation)
    index = {entry['id']: entry['index'] for entry in formulation.pop('points')}
    points = [{**point, 'index': index.get(point['id'])}
              for point in resolved_points_dict(cfg.resolved_points)]
    out = {'schema_version': 1, 'points': points, 'formulation': formulation}
    start = read_start(cfg)
    if start is not None:
        out['design_start'] = start_record(start)
    if cfg.output.seed:
        out['seeds'] = [{**entry, 'path': os.path.relpath(entry['path'], REPO),
                         'sha256': sha256(entry['path'])} for entry in seed_plan(cfg)]
    return out


def source_path(cfg, name, ctx):
    """``name`` resolved under the output root; anything else is a ``ConfigError``.

    Sources are run ids and paths inside them, never absolute paths or paths
    that leave the root, so every input still enters through the case.
    """
    root = os.path.realpath(_output_root(cfg))
    path = os.path.realpath(os.path.join(root, name))
    if os.path.isabs(name) or not path.startswith(root + os.sep):
        raise ConfigError(f'{ctx}={name!r} must name a path under the output root '
                          f'{root}, such as a run id')
    if not os.path.isdir(path):
        raise ConfigError(f'{ctx}={name!r}: {path} is not a directory')
    return path


#: Leading int32 words of an SU2 binary restart: magic, fields, points.
_BINARY_RESTART = 535532


def restart_points(path):
    """The number of points in an SU2 restart file, binary or ASCII."""
    import struct
    with open(path, 'rb') as stream:
        head = stream.read(12)
        if len(head) == 12:
            magic, _, n_points = struct.unpack('<3i', head)
            if magic == _BINARY_RESTART:
                return n_points
        stream.seek(0)
        # ASCII: one header line, then one row per point.
        return sum(1 for line in stream if line.strip()) - 1


def seed_plan(cfg):
    """Every seed file as ``{kind, from, point, source_point, path, name}``.

    Raises ``ConfigError`` listing every problem: a source outside the output
    root, a listed point with no file, a restart whose point count differs
    from this mesh, a flow seed with ``[solver.primal] warm_start = false``
    or an adjoint seed with ``[execution] adjoint_restart = false``.
    """
    from ..mesh.su2_mesh_file import read_mesh_header
    from .final_artifacts import seed_files
    errors, plan = [], []
    ids = set(cfg.point_ids)
    n_mesh = read_mesh_header(cfg.mesh_path)['npoin'] if os.path.isfile(cfg.mesh_path) else None
    for index, seed in enumerate(cfg.output.seed):
        ctx = f'[[output.seed]] {index} ({seed.kind})'
        if seed.kind == 'flow' and not cfg.solver.primal.warm_start:
            errors.append(f'{ctx}: [solver.primal] warm_start = false never reads a '
                          f'flow seed')
        if seed.kind == 'adjoint' and not cfg.execution.adjoint_restart:
            errors.append(f'{ctx}: [execution] adjoint_restart = false never reads '
                          f'an adjoint seed')
        try:
            directory = source_path(cfg, seed.source, f'{ctx} from')
        except ConfigError as exc:
            errors.append(str(exc))
            continue
        if os.sep not in os.path.relpath(directory, os.path.realpath(_output_root(cfg))):
            # A run id seeds from the run's final/; a failed run has none.
            if not os.path.isdir(os.path.join(directory, 'final')):
                errors.append(f'{ctx}: run {seed.source!r} has no final/; name its '
                              f'solver directory, {seed.source}/su2_workdir')
                continue
            directory = os.path.join(directory, 'final')
        pairs = seed.points or tuple((pid, pid) for pid in cfg.point_ids)
        for destination, source in pairs:
            if destination not in ids:
                errors.append(f'{ctx}: point {destination!r} is not a point of '
                              f'this configuration')
                continue
            files = seed_files(directory, cfg, seed.kind, destination, source)
            if not files:
                errors.append(f'{ctx}: no {seed.kind} restart for point {source!r} '
                              f'in {directory}')
            for path, name in files:
                n = restart_points(path)
                if n_mesh is not None and n != n_mesh:
                    errors.append(f'{ctx}: {path} has {n} points; the mesh '
                                  f'{cfg.case.mesh} has {n_mesh}')
                plan.append({'kind': seed.kind, 'from': seed.source,
                             'point': destination, 'source_point': source,
                             'path': path, 'name': name})
    if errors:
        raise ConfigError('; '.join(errors))
    return plan


def seed_lines(plan):
    """Banner lines: one per seed kind and source, with its point mapping."""
    groups = {}
    for entry in plan:
        pairs = groups.setdefault((entry['kind'], entry['from']), {})
        pairs[entry['point']] = entry['source_point']
    return [f'[seed] {kind} from {source}: ' + ', '.join(
                dst if dst == src else f'{dst} <- {src}' for dst, src in pairs.items())
            for (kind, source), pairs in groups.items()]


def make_run_dir(cfg, comm=None, argv=None, now=None):
    """Create or resume a run tree without entering its solver directory."""
    rank0 = comm is None or comm.rank == 0
    run_id = resolve_run_id(cfg, now) if rank0 else None
    if comm is not None:
        run_id = comm.bcast(run_id, root=0)
    rd = RunDir(run_id=run_id, root=os.path.join(_output_root(cfg), run_id))

    def create():
        _prepare_tree(cfg, rd)
        _write_run_toml(cfg, rd, comm, argv)
        with open(os.path.join(rd.root, 'resolved_config.json'), 'w') as stream:
            json.dump(resolved_config(cfg), stream, indent=2)
            stream.write('\n')
        _set_status(rd, 'RUNNING')

    if comm is not None:
        on_root(comm, create, what='create run directory', error=ConfigError)
    elif rank0:
        create()
    return rd


def _prepare_tree(cfg, rd):
    mode = cfg.output.on_exists
    if os.path.islink(rd.root):
        raise ConfigError(f'run directory is a symbolic link: {rd.root}')
    if os.path.lexists(rd.root) and not os.path.isdir(rd.root):
        raise ConfigError(f'run directory is not a directory: {rd.root}')
    if os.path.isdir(rd.root):
        if mode == 'fail':
            raise ConfigError(f'run directory already exists: {rd.root}')
        if mode == 'overwrite':
            shutil.rmtree(rd.root)
    driver = cfg.execution.task == 'run_driver'
    for subdir in SUBDIRS + (DRIVER_SUBDIRS if driver else ()):
        os.makedirs(os.path.join(rd.root, subdir), exist_ok=True)


def _set_status(rd, status):
    with open(os.path.join(rd.root, 'STATUS'), 'w') as stream:
        stream.write(f'{status}\n')


@functools.cache
def git_head(repo):
    """``(HEAD sha, dirty)`` of ``repo``, asked once per process: the run.toml
    provenance and the console banner report the same state."""
    def git(*args):
        return subprocess.run(('git', '-C', repo) + args, capture_output=True,
                              text=True, timeout=10)
    try:
        head = git('rev-parse', 'HEAD')
        if head.returncode:
            return 'unknown', None
        return head.stdout.strip(), bool(git('status', '--porcelain').stdout.strip())
    except (OSError, subprocess.SubprocessError):
        return 'unknown', None


def _provenance(cfg, rd, comm, argv):
    from ..paths import REPO
    from ..run_config.checks.task import resolve_derivative_mode
    sha, dirty = git_head(REPO)
    requested, resolved, reason = resolve_derivative_mode(cfg)
    return {
        'run_id': rd.run_id, 'config_source': cfg.source,
        'overrides': list(cfg.overrides), 'git_sha': sha, 'git_dirty': dirty,
        'np': 1 if comm is None else comm.size, 'host': socket.gethostname(),
        'python': sys.version.split()[0], 'su2_run': os.environ.get('SU2_RUN', ''),
        'argv': list(argv if argv is not None else sys.argv),
        'start_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'start_monotonic': time.monotonic(),
        'derivative_mode_requested': requested,
        'derivative_mode_resolved': resolved,
        'derivative_mode_reason': reason,
    }


def _header(provenance):
    return ''.join(f'# provenance: {key}={value!r}\n'
                   for key, value in provenance.items())


def _source_path(cfg):
    from ..paths import REPO
    if not cfg.source:
        return None
    return cfg.source if os.path.isabs(cfg.source) else os.path.join(REPO, cfg.source)


def _override(text):
    """``'a.b=1.5'`` (a ``cfg.overrides`` entry, ``key=repr(value)``) -> ``('a.b', 1.5)``."""
    key, _, value = text.partition('=')
    return key, ast.literal_eval(value)


def effective_toml(cfg):
    """The TOML the run solved: includes merged, ``--set`` overrides applied.

    Returns None when the text does not read back through ``tomllib`` as the
    same tables (a value TOML cannot hold, such as a mixed list).
    """
    from ..run_config import resolve_dict
    path = _source_path(cfg)
    if path is None or not os.path.isfile(path):
        return None
    try:
        data = resolve_dict(path, dict(_override(item) for item in cfg.overrides))
        text = to_toml(data)
        if tomllib.loads(text) != data:
            return None
    except (ValueError, SyntaxError, tomllib.TOMLDecodeError, ConfigError):
        return None
    return text


def _write_run_toml(cfg, rd, comm, argv):
    """The merged effective TOML under a comment-only provenance header.

    When the merged tables do not round-trip, the source file is written
    instead, with a warning.
    """
    body = effective_toml(cfg)
    note = f'# The effective TOML of {cfg.source}: includes merged, overrides applied.\n'
    if body is None:
        warnings.warn(f'run.toml: the merged config of {cfg.source} does not read back '
                      f'as the same tables; the source file is written instead',
                      stacklevel=2)
        path = _source_path(cfg)
        body = ''
        if path is not None and os.path.isfile(path):
            with open(path) as stream:
                body = stream.read()
        note = f'# The source file {cfg.source}, not merged.\n'
    with open(rd.run_toml, 'w') as stream:
        stream.write(_header(_provenance(cfg, rd, comm, argv)))
        stream.write(note)
        stream.write(body)


def _read_run_toml(rd):
    provenance, body = {}, []
    with open(rd.run_toml) as stream:
        for line in stream:
            if line.startswith('# provenance: ') and '=' in line:
                key, value = line[len('# provenance: '):].split('=', 1)
                provenance[key] = ast.literal_eval(value)
            else:
                body.append(line)
    return provenance, body


def _rewrite_run_toml(rd, provenance, body):
    with open(rd.run_toml, 'w') as stream:
        stream.write(_header(provenance))
        stream.writelines(body)


def record_provenance(rd, items):
    """Add or replace provenance keys in ``run.toml``; call on rank 0 only."""
    if not os.path.isfile(rd.run_toml):
        return
    provenance, body = _read_run_toml(rd)
    provenance.update(items)
    _rewrite_run_toml(rd, provenance, body)


def _compress_logs(rd):
    """Gzip ``logs/solver.log`` and each ``logs/solves/*.csv``, one file each.

    Appends a gzip member when the ``.gz`` exists (a resumed run), so the
    earlier attempt's output is kept; ``zcat`` reads both members in order.
    """
    paths = [os.path.join(rd.logs, 'solver.log')]
    paths += sorted(glob.glob(os.path.join(rd.logs, 'solves', '*.csv')))
    for path in paths:
        if not os.path.isfile(path):
            continue
        with open(path, 'rb') as source, gzip.open(f'{path}.gz', 'ab') as target:
            shutil.copyfileobj(source, target)
        os.remove(path)


def finish(rd, status='DONE', reason='', comm=None, remove_workdir=False):
    """Record the end state, gzip the logs; ``remove_workdir`` once ``final/`` is kept.

    Rank 0 only. ``su2_workdir/`` is removed only on ``DONE``; a failed run
    keeps it.
    """
    if comm is not None and comm.rank != 0:
        return
    _set_status(rd, f'{status}:{reason}' if reason else status)
    _compress_logs(rd)
    if remove_workdir and status == 'DONE' and os.path.isdir(rd.workdir):
        shutil.rmtree(rd.workdir)
    if not os.path.isfile(rd.run_toml):
        return
    old, body = _read_run_toml(rd)
    old['end_utc'] = datetime.now(timezone.utc).isoformat(timespec='seconds')
    old['wall_seconds'] = round(time.monotonic() - old.pop('start_monotonic'), 1)
    old['status'] = status
    _rewrite_run_toml(rd, old, body)


def seed_solutions(cfg, rd, comm=None):
    """Copy the ``[[output.seed]]`` restarts into ``su2_workdir/``; returns the plan.

    Called after ``clean_restart`` has cleared the solver directory, so a
    seeded ``run_driver`` starts warm.
    """
    plan = seed_plan(cfg)

    def copy():
        for entry in plan:
            shutil.copy2(entry['path'], os.path.join(rd.workdir, entry['name']))
    if comm is None:
        copy()
    else:
        on_root(comm, copy, what='seed solutions', error=ConfigError)
    return plan
