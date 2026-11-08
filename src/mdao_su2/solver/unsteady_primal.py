"""Time-domain (URANS) primal solves: the cfg, the restart levels, the mesh and the run.

Primal only, and no OpenMDAO: the ``polar`` task calls these functions after a
steady sweep, one angle at a time, on a point whose steady solve did not
converge. The frozen ``SU2AeroComp`` stays the steady component; it never sees
a time-domain cfg (its acceptance compares the final residual with
``CONV_RESIDUAL_MINVAL``, which does not describe a time average).

A run reads ``TAVG_LIFT``, ``TAVG_DRAG`` and the moment's time average from the
driver after ``StartSolver()``, and the ``C_L`` range over the averaging window
from the history file, with the cycles of ``C_L`` and the inner-loop misses over it.
"""

import math
import os
import re
import shutil
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

from .cfg_file import parse_cfg, parse_cfg_content, substitute, write_cfg
from .su2_options import fmt_passthrough

__all__ = ['UnsteadyResult', 'write_urans_cfg', 'seed_time_levels',
           'RESTART_SUFFIX', 'write_deformed_mesh', 'check_design_mesh',
           'write_mesh_with_coordinates', 'read_restart_coordinates', 'run_urans',
           'read_history_rows', 'window_cl_range', 'window_cycles',
           'window_inner_miss', 'HISTORY_GROUPS']

#: History groups a URANS cfg requests besides the coefficients: the iteration counters,
#: the residuals and the time averages (``TAVG_*`` fields). ``AERO_COEFF`` (the ``CL``
#: column) is added only when the steady cfg lists no ``LIFT``/``DRAG`` field.
HISTORY_GROUPS = ('ITER', 'RMS_RES', 'TAVG_AERO_COEFF')

#: Individual coefficient fields whose presence makes ``AERO_COEFF`` a duplicate.
_AERO_FIELDS = ('LIFT', 'DRAG')

#: Output stems the URANS cfg names relative to the working directory.
_STEM_KEYS = ('SOLUTION_FILENAME', 'RESTART_FILENAME', 'CONV_FILENAME',
              'SURFACE_FILENAME', 'VOLUME_FILENAME')

#: SU2 copies a restart path into ``char fname[100]`` (``CSolver::Read_SU2_Restart_Binary``,
#: ``CSolver.cpp:2973-2974``), so a name of this length or more overruns the buffer.
SU2_RESTART_NAME_LIMIT = 100

#: Extension of the restart files a URANS run reads. SU2 picks the read format from
#: ``READ_BINARY_RESTART`` (default YES; ``CTurbSolver.cpp:114-120``,
#: ``CFVMFlowSolverBase.inl:889-896``), and the seeded steady restart is binary.
RESTART_SUFFIX = '.dat'

#: The first time level SU2 opens: ``<SOLUTION_FILENAME>_00000<suffix>``.
_LEVEL_TAG = '_00000'


# ------------------------------------------------------------------- cfg

def _with_history_groups(value):
    """``HISTORY_OUTPUT`` text with every group of ``HISTORY_GROUPS`` present."""
    tokens = [t.strip() for t in (value or '').strip('() ').split(',') if t.strip()]
    groups = HISTORY_GROUPS
    if not any(field in tokens for field in _AERO_FIELDS):
        groups = ('ITER', 'RMS_RES', 'AERO_COEFF', 'TAVG_AERO_COEFF')
    tokens += [group for group in groups if group not in tokens]
    return ', '.join(tokens)


def write_urans_cfg(steady_cfg_path, out_path, keys, alpha, mesh_path, workdir):
    """Write the time-domain cfg of one angle, derived from a steady point primal cfg.

    ``ITER`` is dropped (SU2 rejects it in a time-domain run); ``TIME_DOMAIN``,
    ``RESTART_SOL``, ``DEFORM_MESH``, ``AOA``, ``MESH_FILENAME`` and
    ``WINDOW_CAUCHY_CRIT= NO`` are set (a URANS run is a fixed length that ends on ``TIME_ITER``);
    ``keys`` (the ``[polar.urans]`` table) is applied on top; ``MAX_TIME`` is set
    to twice ``TIME_ITER * TIME_STEP`` so SU2's 1 s default never ends the run;
    the solution, restart, history, surface and volume stems are bare file names
    that SU2 resolves in its working directory, ``workdir``; and ``HISTORY_OUTPUT``
    carries ``HISTORY_GROUPS``. ``MESH_FILENAME`` stays as given (an absolute
    path). Raises ``ValueError`` when a restart file name reaches SU2's 100-character
    buffer. Returns ``out_path``.
    """
    with open(steady_cfg_path) as stream:
        text = stream.read()
    text = re.sub(r'^ITER\s*=[^\n]*\n?', '', text, flags=re.MULTILINE)
    fixed = {'TIME_DOMAIN': 'YES', 'RESTART_SOL': 'YES', 'DEFORM_MESH': 'NO',
             'AOA': repr(float(alpha)), 'MESH_FILENAME': mesh_path,
             'WINDOW_CAUCHY_CRIT': 'NO'}
    stems = {key: os.path.basename(value)
             for key, value in parse_cfg_content(text).items() if key in _STEM_KEYS}
    text = substitute(text, {**fixed, **stems})
    text = substitute(text, {key: fmt_passthrough(value) for key, value in keys.items()})
    values = parse_cfg_content(text)
    span = float(values['TIME_ITER']) * float(values['TIME_STEP'])
    text = substitute(text, {'MAX_TIME': repr(2.0 * span),
                             'HISTORY_OUTPUT': _with_history_groups(values.get('HISTORY_OUTPUT'))})
    _check_restart_names(parse_cfg_content(text), workdir)
    note = (f'% Time-domain (URANS) cfg derived from {os.path.basename(steady_cfg_path)}'
            f' at AOA= {float(alpha):g}\n')
    return write_cfg(out_path, note + text)


def _check_restart_names(values, workdir):
    """Raise when a restart file name SU2 opens is too long for its ``fname`` buffer."""
    suffix = RESTART_SUFFIX
    for key in ('SOLUTION_FILENAME', 'RESTART_FILENAME'):
        stem = values.get(key)
        if stem is None:
            continue
        name = f'{stem}{_LEVEL_TAG}{suffix}'
        if len(name) >= SU2_RESTART_NAME_LIMIT:
            raise ValueError(
                f'{key} {stem!r} gives the restart file name {name!r} of {len(name)} '
                f'characters; SU2 copies it into a {SU2_RESTART_NAME_LIMIT}-character '
                f'buffer (char fname[100] in CSolver::Read_SU2_Restart_Binary) and '
                f'aborts with signal 6 at {SU2_RESTART_NAME_LIMIT} or more. Keep the '
                f'stem short and relative to the working directory {workdir!r}.')


def seed_time_levels(steady_restart, workdir, solution_stem, levels, suffix=RESTART_SUFFIX):
    """Copy a steady restart to the ``levels`` time-level files a dual-time restart reads.

    SU2 reads ``<SOLUTION_FILENAME>_00000<suffix>`` ... ``_0000<levels-1><suffix>``
    for ``RESTART_ITER = levels`` (``CDriver.cpp:1109-1132`` loads the iteration
    ``RESTART_ITER - 1 - (second order)``, and a second-order run the one after
    it, ``CFVMFlowSolverBase.inl:1147``). A relative
    ``solution_stem`` is taken inside ``workdir`` (the directory SU2 runs in). Returns the paths written.
    """
    stem = solution_stem if os.path.isabs(solution_stem) \
        else os.path.join(workdir, solution_stem)
    os.makedirs(os.path.dirname(stem), exist_ok=True)
    paths = []
    for level in range(levels):
        path = f'{stem}_{level:05d}{suffix}'
        shutil.copy2(steady_restart, path)
        paths.append(path)
    return paths


# ------------------------------------------------------------------ mesh

#: Largest miss (chord units) between a wall node of the design mesh and ``x_aero``.
WALL_NODE_TOLERANCE = 1.0e-8

#: Node count of each SU2 element type (VTK numbering) in a ``.su2`` mesh.
_NODES_OF_TYPE = {3: 2, 5: 3, 9: 4, 10: 4, 12: 8, 13: 6, 14: 5}

#: Name of the deformed mesh inside the working directory.
DESIGN_MESH_NAME = 'deformed_mesh.su2'


def _scan_mesh(path, want_elements=False):
    """Header and sections of a ``.su2`` mesh: ``(ndim, npoin, markers, elements)``.

    ``markers`` is ``{name: node ids}``. ``elements`` is ``[(type, nodes)]`` when
    ``want_elements``, else ``[]``. Raises ``ValueError`` for a multizone mesh or one
    without an ``NPOIN`` block.
    """
    ndim = npoin = None
    markers, elements = {}, []
    with open(path) as stream:
        line = stream.readline()
        while line:
            key, _, rest = line.partition('=')
            key = key.strip()
            if key in ('NZONE', 'IZONE'):
                raise ValueError(f'{path}: multizone mesh (NZONE) is not supported')
            if key == 'NDIME':
                ndim = int(rest.split()[0])
            elif key == 'NELEM':
                for _ in range(int(rest.split()[0])):
                    tokens = stream.readline().split()
                    if want_elements:
                        kind = int(tokens[0])
                        elements.append((kind, [int(t) for t in tokens[1:1 + _NODES_OF_TYPE[kind]]]))
            elif key == 'NPOIN':
                npoin = int(rest.split()[0])
                for _ in range(npoin):
                    stream.readline()
            elif key == 'MARKER_TAG':
                name = rest.strip()
                count_key, _, count = stream.readline().partition('=')
                if count_key.strip() != 'MARKER_ELEMS':
                    raise ValueError(f'{path}: MARKER_TAG {name} lacks MARKER_ELEMS')
                nodes = []
                for _ in range(int(count.split()[0])):
                    tokens = stream.readline().split()
                    nodes.extend(int(t) for t in tokens[1:1 + _NODES_OF_TYPE[int(tokens[0])]])
                markers[name] = nodes
            line = stream.readline()
    if npoin is None or ndim is None:
        raise ValueError(f'{path}: no NDIME or NPOIN block')
    return ndim, npoin, markers, elements


def _read_points(path, ndim, npoin):
    """The ``(npoin, ndim)`` coordinates of a ``.su2`` mesh's NPOIN block."""
    import numpy as np

    with open(path) as stream:
        for line in stream:
            if line.partition('=')[0].strip() == 'NPOIN':
                return np.array([[float(t) for t in stream.readline().split()[:ndim]]
                                 for _ in range(npoin)])
    raise ValueError(f'{path}: no NPOIN block')


def read_restart_coordinates(path, ndim):
    """Coordinates ``(npoin, ndim)`` of an SU2 ASCII restart, ordered by ``PointID``.

    Columns are found by header name (``"PointID"``, ``"x"``, ``"y"``, ``"z"``).
    Raises ``RuntimeError`` when a name is missing or the PointIDs are not exactly
    ``0..npoin-1``.
    """
    import numpy as np

    with open(path) as stream:
        names = [n.strip().strip('"').strip() for n in stream.readline().split(',')]
    wanted = ['PointID', 'x', 'y', 'z'][:ndim + 1]
    missing = [n for n in wanted if n not in names]
    if missing:
        raise RuntimeError(f'restart {path} lacks columns {missing}; header {names[:6]}')
    data = np.genfromtxt(path, delimiter=',', skip_header=1,
                         usecols=[names.index(n) for n in wanted], ndmin=2)
    data = data[np.argsort(data[:, 0], kind='stable')]
    if not np.array_equal(data[:, 0], np.arange(len(data))):
        raise RuntimeError(f'restart {path}: PointIDs are not exactly 0..{len(data) - 1} '
                           f'({len(data)} rows, first {data[:3, 0]}, last {data[-3:, 0]})')
    return data[:, 1:]


def write_mesh_with_coordinates(case_mesh_path, coords, out_path):
    """Write the case mesh with the coordinates of its ``NPOIN`` block replaced.

    Every other line (elements, markers, each point's trailing global index) is
    copied verbatim; text only, no SU2. ``coords`` is ``(npoin, ndim)``. Raises
    ``ValueError`` for a multizone mesh or one without ``NPOIN``, ``RuntimeError``
    when ``coords`` does not match the block.
    """
    ndim, npoin, _, _ = _scan_mesh(case_mesh_path)
    if tuple(coords.shape) != (npoin, ndim):
        raise RuntimeError(f'coordinates {tuple(coords.shape)} do not match the case mesh '
                           f'NPOIN {npoin} and NDIME {ndim}')
    with open(case_mesh_path) as source, open(out_path, 'w') as sink:
        for line in source:
            sink.write(line)
            if line.partition('=')[0].strip() == 'NPOIN':
                for i in range(npoin):
                    tail = ' '.join(source.readline().split()[ndim:])
                    sink.write(' '.join(repr(float(v)) for v in coords[i])
                               + (' ' + tail if tail else '') + '\n')


def _signed_areas(coords, elements):
    """Signed area of each triangle or quad (2D), 0 for other types."""
    import numpy as np

    areas = np.zeros(len(elements))
    for k, (kind, nodes) in enumerate(elements):
        if kind in (5, 9):
            x, y = coords[nodes, 0], coords[nodes, 1]
            areas[k] = 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
    return areas


def check_design_mesh(mesh_path, case_mesh_path, x_aero, gid_order, wall_markers):
    """Validate a design mesh on one rank; raise ``RuntimeError`` naming the failed check.

    Checks: NPOIN equals the case mesh's; all coordinates are finite; every node of the
    ``wall_markers`` (ids from the case mesh) equals ``x_aero`` mapped through
    ``gid_order`` within ``WALL_NODE_TOLERANCE``; in 2D no triangle or quad changes the
    sign of its signed area relative to the case mesh (the count of inverted cells is
    reported). The area check is skipped in 3D.
    """
    import numpy as np

    ndim, npoin, markers, _ = _scan_mesh(case_mesh_path)
    ndim_new, npoin_new, _, _ = _scan_mesh(mesh_path)
    if (ndim_new, npoin_new) != (ndim, npoin):
        raise RuntimeError(f'design mesh check "NPOIN": {npoin_new} points in {ndim_new}D, '
                           f'case mesh {npoin} in {ndim}D')
    new = _read_points(mesh_path, ndim, npoin)
    if not np.all(np.isfinite(new)):
        raise RuntimeError(f'design mesh check "finite": {int(np.sum(~np.isfinite(new)))} '
                           f'non-finite coordinates')

    x_aero = np.asarray(x_aero).reshape(-1)
    gid_pos = {gid: i for i, gid in enumerate(gid_order)}
    worst, worst_node = 0.0, None
    for name in wall_markers:
        if name not in markers:
            raise RuntimeError(f'design mesh check "wall": marker {name!r} is not in the '
                               f'case mesh (markers {sorted(markers)})')
        for node in set(markers[name]):
            if node not in gid_pos:
                raise RuntimeError(f'design mesh check "wall": node {node} of {name} is not '
                                   f'in gid_order')
            pos = ndim * gid_pos[node]
            miss = float(np.max(np.abs(new[node] - x_aero[pos:pos + ndim])))
            if miss > worst:
                worst, worst_node = miss, node
    if worst > WALL_NODE_TOLERANCE:
        raise RuntimeError(f'design mesh check "wall": worst wall-node miss {worst:.3e} '
                           f'at node {worst_node} exceeds {WALL_NODE_TOLERANCE:.0e}')

    if ndim == 2:
        elements = _scan_mesh(case_mesh_path, want_elements=True)[3]
        before = _signed_areas(_read_points(case_mesh_path, ndim, npoin), elements)
        inverted = int(np.sum(before * _signed_areas(new, elements) <= 0.0))
        if inverted:
            raise RuntimeError(f'design mesh check "cell orientation": {inverted} of '
                               f'{len(elements)} cells changed the sign of their area')


def _build_design_mesh(case_mesh, restart_path, out_path, x_aero, gid_order, wall_markers):
    """Rank 0: case mesh text with the restart coordinates, checked, at ``out_path``."""
    ndim = _scan_mesh(case_mesh)[0]
    if not os.path.isfile(restart_path):
        raise RuntimeError(f'SU2 wrote no ASCII restart at {restart_path}')
    coords = read_restart_coordinates(restart_path, ndim)
    temporary = out_path + '.tmp'
    try:
        write_mesh_with_coordinates(case_mesh, coords, temporary)
        check_design_mesh(temporary, case_mesh, x_aero, gid_order, wall_markers)
        os.replace(temporary, out_path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)
    return out_path


def write_deformed_mesh(steady_cfg_path, workdir, x_aero, x_ref, gid_order,
                        wall_markers, comm):
    """Write the mesh of a design shape: the case mesh with the wall at ``x_aero``.

    One fresh steady iteration with the displacement ``x_aero - x_ref`` (always
    from the reference surface) applied through ``wall_seam.apply_displacements``,
    writing only the ASCII restart (``OUTPUT_FILES= (RESTART_ASCII)``; SU2's native
    mesh writer needs a ``boundary.dat`` only SU2_DEF writes). Rank 0 then
    copies the case mesh text (``MESH_FILENAME`` of the steady cfg) with the NPOIN
    coordinates replaced by the restart's, checks it (``check_design_mesh``) and
    writes ``<workdir>/deformed_mesh.su2`` atomically. Collective on ``comm``: every
    rank returns the path or raises the same error. The area-sign check covers 2D
    only. ``ValueError`` for a case mesh the text path cannot handle (multizone, no
    NPOIN); ``RuntimeError`` for a failed build or check.
    """
    import pysu2
    from . import wall_seam
    from ..mpi import on_root, raise_together

    workdir = os.path.abspath(workdir)
    out_path = os.path.join(workdir, DESIGN_MESH_NAME)
    cfg_path = os.path.join(workdir, 'deformed_mesh.cfg')
    restart_stem = 'deformed_mesh_restart'

    def prepare():
        os.makedirs(workdir, exist_ok=True)
        for stale in (out_path, out_path + '.tmp', os.path.join(workdir, restart_stem + '.csv')):
            if os.path.exists(stale):
                os.remove(stale)
        with open(steady_cfg_path) as stream:
            text = stream.read()
        case_mesh = parse_cfg_content(text)['MESH_FILENAME']
        _scan_mesh(case_mesh)       # a mesh the text path cannot handle fails before the solve
        text = substitute(text, {
            'ITER': 1, 'RESTART_SOL': 'NO', 'OUTPUT_FILES': '( RESTART_ASCII )',
            'VOLUME_FILENAME': 'deformed_mesh',
            'SOLUTION_FILENAME': 'deformed_mesh_solution',
            'RESTART_FILENAME': restart_stem,
            'CONV_FILENAME': 'deformed_mesh_history',
            'SURFACE_FILENAME': 'deformed_mesh_surface'})
        write_cfg(cfg_path, text)
        return case_mesh
    case_mesh = on_root(comm, prepare, what='prepare the design mesh run',
                        error=ValueError)
    with _working_directory(workdir):
        driver = pysu2.CSinglezoneDriver(cfg_path, 1, comm)
        try:
            failure = None
            try:
                markers = wall_seam.local_markers(driver, wall_markers)
                gid_pos = {gid: i for i, gid in enumerate(gid_order)}
                wall_seam.apply_displacements(driver, markers, x_aero, x_ref, gid_pos,
                                              driver.GetNumberDimensions())
            except Exception as exc:    # noqa: BLE001 -- agreed across ranks below
                failure = f'rank {comm.rank}: {type(exc).__name__}: {exc}'
            raise_together(comm, failure)   # no rank enters StartSolver alone
            driver.StartSolver()
        finally:
            driver.Finalize()
    comm.Barrier()

    outcome = None
    if comm.rank == 0:
        try:
            outcome = (True, _build_design_mesh(
                case_mesh, os.path.join(workdir, restart_stem + '.csv'), out_path,
                x_aero, gid_order, wall_markers))
        except Exception as exc:    # noqa: BLE001 -- broadcast, then every rank raises
            outcome = (False, f'{type(exc).__name__}: {exc}')
    ok, value = comm.bcast(outcome, root=0)
    if not ok:
        raise RuntimeError(value)
    return value


# --------------------------------------------------------------- history

def read_history_rows(path):
    """``[{column: float}]`` of an SU2 history CSV (NaN where a value does not parse)."""
    with open(path) as stream:
        lines = [line for line in stream if line.strip()]
    if len(lines) < 2:
        return []
    names = [name.strip().strip('"').strip() for name in lines[0].split(',')]
    rows = []
    for line in lines[1:]:
        row = {}
        for name, value in zip(names, line.split(',')):
            try:
                row[name] = float(value)
            except ValueError:
                row[name] = math.nan
        rows.append(row)
    return rows


def window_cl_range(rows, column='CL', first=None, last=None):
    """``max - min`` of ``column`` over a row selection, or None when it is empty.

    ``first``/``last`` select by position (``rows[-last:]``) or ``Time_Iter``
    (``first`` is a minimum ``Time_Iter``); a row with a NaN value is skipped.
    """
    if first is not None:
        rows = [row for row in rows if row.get('Time_Iter', -1) >= first]
    if last is not None:
        rows = rows[-last:]
    values = [row[column] for row in rows if column in row and math.isfinite(row[column])]
    return (max(values) - min(values)) if values else None


def window_cycles(rows, column='CL', first=None):
    """Oscillation cycles of ``column`` over the rows from ``Time_Iter >= first``.

    The number of up-crossings of the window's mean, minus one (the first crossing
    opens a cycle and does not close one), floored at 0. NaN rows are skipped.
    """
    if first is not None:
        rows = [row for row in rows if row.get('Time_Iter', -1) >= first]
    values = [row[column] for row in rows if column in row and math.isfinite(row[column])]
    if len(values) < 2:
        return 0
    mean = sum(values) / len(values)
    ups = sum(1 for a, b in zip(values, values[1:]) if a < mean <= b)
    return max(ups - 1, 0)


def window_inner_miss(rows, floor, first=None):
    """Fraction of the rows from ``Time_Iter >= first`` whose residual exceeds ``floor``.

    The residual is the first ``rms[`` column, the one ``read_history_tail`` reads.
    A row with no finite residual counts as a miss. None when no row qualifies.
    """
    if first is not None:
        rows = [row for row in rows if row.get('Time_Iter', -1) >= first]
    if not rows:
        return None
    column = next((name for name in rows[0] if name.startswith('rms[')), None)
    if column is None:
        return None
    missed = sum(1 for row in rows
                 if not (math.isfinite(row.get(column, math.nan)) and row[column] <= floor))
    return missed / len(rows)


# ------------------------------------------------------------------- run

@dataclass
class UnsteadyResult:
    """What one URANS run measured."""
    tavg_cl: float
    tavg_cd: float
    tavg_cm: float
    #: Time average of the buffet sensor, None when the solver does not expose it.
    tavg_buffet: Optional[float]
    #: ``C_L`` range over the rows from ``WINDOW_START_ITER`` on.
    cl_range: Optional[float]
    #: Rows of the history from ``WINDOW_START_ITER`` on.
    window_steps: int
    #: Oscillation cycles of ``C_L`` in those rows (``window_cycles``).
    window_cycles: int
    #: Fraction of those rows whose inner loop ended above ``inner_floor``; a diagnostic.
    inner_miss: Optional[float]
    #: ``CONV_RESIDUAL_MINVAL`` of the cfg (SU2's per-step inner target), None when the cfg omits it.
    inner_floor: Optional[float]
    #: Last time iteration the run executed.
    time_iter: int
    #: ``TIME_ITER - WINDOW_START_ITER``: the time steps the average may span.
    averaging_length: int
    history: str = ''

    def reached_time_iter(self, time_iter_limit):
        return self.time_iter >= time_iter_limit - 1


@contextmanager
def _working_directory(path):
    previous = os.getcwd()
    os.makedirs(path, exist_ok=True)
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _output(driver, name):
    """``GetOutputValue(name)``, or None when the cfg does not define the field."""
    try:
        return float(driver.GetOutputValue(name))
    except Exception:
        return None


def urans_history_path(cfg_path, workdir):
    """History file a restarted time-domain cfg writes, joined with ``workdir`` when relative.

    ``CConfig::GetHistory_FileName`` (``CConfig.hpp:5806-5832``) trims ``.dat``
    and ``.csv`` from ``CONV_FILENAME``, then, for a time-domain restart, appends
    ``_%05d`` of ``RESTART_ITER`` and the extension (``.csv``, or ``.dat`` for a
    TECPLOT ``TABULAR_FORMAT``).
    """
    cfg = parse_cfg(cfg_path)
    name = cfg['CONV_FILENAME']
    for extension in ('.dat', '.csv'):
        if name.endswith(extension):
            name = name[:-len(extension)]
    tecplot = cfg.get('TABULAR_FORMAT', 'CSV').strip().upper() == 'TECPLOT'
    name += f"_{int(float(cfg['RESTART_ITER'])):05d}" + ('.dat' if tecplot else '.csv')
    return name if os.path.isabs(name) else os.path.join(workdir, name)


def run_urans(cfg_path, comm, workdir, moment_key='MOMENT_Z'):
    """Run the time-domain cfg to its end; return an ``UnsteadyResult``.

    A fresh ``pysu2.CSinglezoneDriver`` on ``comm`` (``StartSolver()`` runs the
    whole time loop). The time averages come from the driver; the ``C_L`` range and
    the inner-loop miss against the cfg's ``CONV_RESIDUAL_MINVAL`` from the
    history file the cfg names. Collective on
    ``comm``; the run executes inside ``workdir``.
    """
    import pysu2

    cfg_path = os.path.abspath(cfg_path)
    cfg = parse_cfg(cfg_path)
    limit = int(float(cfg['TIME_ITER']))
    start = int(float(cfg['WINDOW_START_ITER']))
    inner_floor = float(cfg['CONV_RESIDUAL_MINVAL']) if 'CONV_RESIDUAL_MINVAL' in cfg else None
    history = urans_history_path(cfg_path, workdir)

    with _working_directory(workdir):
        driver = pysu2.CSinglezoneDriver(cfg_path, 1, comm)
        try:
            driver.StartSolver()
            tavg = {name: _output(driver, f'TAVG_{name}')
                    for name in ('LIFT', 'DRAG', moment_key, 'BUFFET')}
            time_iter = _output(driver, 'TIME_ITER')
        finally:
            driver.Finalize()
    comm.Barrier()

    rows = read_history_rows(history) if os.path.isfile(history) else []
    if time_iter is None and rows:
        time_iter = rows[-1].get('Time_Iter')
    if time_iter is None or any(tavg[name] is None for name in ('LIFT', 'DRAG', moment_key)):
        raise RuntimeError(f'URANS run of {cfg_path} exposed no time-averaged '
                           f'coefficients (TAVG fields {tavg})')
    time_iter = int(time_iter)
    inner_miss = (window_inner_miss(rows, inner_floor, first=start)
                  if inner_floor is not None else None)
    return UnsteadyResult(
        tavg_cl=tavg['LIFT'], tavg_cd=tavg['DRAG'], tavg_cm=tavg[moment_key],
        tavg_buffet=tavg['BUFFET'], cl_range=window_cl_range(rows, first=start),
        window_steps=sum(1 for row in rows if row.get('Time_Iter', -1) >= start),
        window_cycles=window_cycles(rows, first=start),
        inner_miss=inner_miss, inner_floor=inner_floor,
        time_iter=time_iter, averaging_length=limit - start, history=history)
