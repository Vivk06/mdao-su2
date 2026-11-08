"""File readers for post-processing: one reader per format, each takes a path.

Headless (no matplotlib) and independent of any run-directory layout. Every
reader raises :class:`ReaderError` for a missing, empty or malformed file.

| Reader | Format | Returns |
| --- | --- | --- |
| :func:`read_surface` | surface ``.vtu``, SU2 surface ``.csv`` | ``{points, point_data, cells, cell_types, cell_data}`` |
| :func:`read_volume` | volume ``.vtu`` | same as the surface reader |
| :func:`read_history` | SU2 history ``.csv`` or tecplot ``.dat`` | ``{column: array}`` |
| :func:`read_hst` | pyOptSparse ``.hst`` | iterations, objectives, constraints, design variables |
| :func:`read_su2_cfg` | SU2 ``.cfg`` | ``{KEY: str}`` |
| :func:`read_su2_stdout` | SU2 screen output | ``{column: array}`` of the convergence table |
| :func:`read_table` | plain CSV, one sample per row | ``[{column: float}]``, non-numeric cells left out |
| :func:`read_reference_cp` | reference Cp CSV (``x_c``, ``Cp`` or ``cp``) | ``{x, cp, surface, point, alpha}`` |
"""
import csv
import os
import re

import numpy as np

from ..vtu import VtuFormatError, read_su2_vtu

__all__ = [
    'ReaderError', 'read_surface', 'read_volume', 'read_history', 'read_hst',
    'read_su2_cfg', 'cfg_float', 'read_su2_stdout', 'read_table', 'read_reference_cp',
    'file_kind',
]


class ReaderError(Exception):
    """A post-processing input file is missing or cannot be read."""


def _require_file(path, what):
    if not os.path.isfile(path):
        raise ReaderError(f'{what} not found: {path}')


def _read_vtu(path, what):
    _require_file(path, what)
    try:
        return read_su2_vtu(path)
    except (VtuFormatError, KeyError, ValueError) as exc:
        raise ReaderError(f'{what} {path} is not a readable VTU file: {exc}') from exc


def _strip(token):
    return token.strip().strip('"').strip()


def _to_float(token):
    try:
        return float(_strip(token))
    except ValueError:
        return float('nan')


def read_surface(path):
    """A surface file as ``{points, point_data, cells, cell_types, cell_data}``.

    ``.vtu`` is read by ``mdao_su2/vtu.py::read_su2_vtu``. A SU2 surface ``.csv``
    (header ``"PointID","x","y",...``) gives ``points`` of shape ``(n, 2)`` or
    ``(n, 3)`` from its ``x``, ``y`` and ``z`` columns, every other column as
    ``point_data`` and no cells.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == '.vtu':
        return _read_vtu(path, 'Surface file')
    if ext != '.csv':
        raise ReaderError(f'Surface file {path}: expected .vtu or .csv')
    table = read_history(path)
    names = {name.lower(): name for name in table}
    if 'x' not in names or 'y' not in names:
        raise ReaderError(f'Surface file {path} has no x and y columns')
    cols = [names[a] for a in ('x', 'y', 'z') if a in names]
    points = np.column_stack([table[c] for c in cols])
    skip = set(cols)
    point_data = {k: v for k, v in table.items() if k not in skip and k.lower() != 'pointid'}
    return {'points': points, 'point_data': point_data, 'cell_data': {}}


def read_volume(path):
    """A volume ``.vtu`` as ``{points, point_data, cells, cell_types, cell_data}``."""
    if os.path.splitext(path)[1].lower() != '.vtu':
        raise ReaderError(f'Volume file {path}: expected .vtu')
    return _read_vtu(path, 'Volume file')


def _read_tecplot(path):
    names, rows = None, []
    with open(path) as stream:
        for line in stream:
            text = line.strip()
            upper = text.upper()
            if not text or upper.startswith(('TITLE', 'ZONE', '#', '%')):
                continue
            if upper.startswith('VARIABLES'):
                names = [_strip(t) for t in text.partition('=')[2].replace(',', ' ').split('"')
                         if _strip(t)]
                if not names:
                    raise ReaderError(f'{path}: empty VARIABLES line')
                continue
            rows.append([_to_float(t) for t in text.replace(',', ' ').split()])
    if names is None:
        raise ReaderError(f'{path}: tecplot file has no VARIABLES line')
    return names, rows


def read_history(path):
    """An SU2 history (``.csv``, or tecplot ``.dat``) as ``{column: np.ndarray}``.

    Column names are the file's, quotes and padding stripped (``rms[Rho]``,
    ``CD``). A value that is not a number becomes NaN.
    """
    _require_file(path, 'History file')
    if os.path.splitext(path)[1].lower() == '.dat':
        header, rows = _read_tecplot(path)
    else:
        with open(path, newline='') as stream:
            reader = csv.reader(stream)
            try:
                header = [_strip(h) for h in next(reader)]
            except StopIteration:
                raise ReaderError(f'History file {path} is empty') from None
            rows = [[_to_float(v) for v in row] for row in reader if row]
    data = {h: [] for h in header}
    for row in rows:
        for h, v in zip(header, row):
            data[h].append(v)
    if not any(data.values()):
        raise ReaderError(f'History file {path} has no data rows')
    return {k: np.array(v) for k, v in data.items()}


def read_hst(path):
    """A pyOptSparse ``.hst`` file at its major iterations.

    Returns ``{'iterations': int array, 'objectives': {name: array},
    'constraints': {name: array}, 'dvs': {name: array (n_iter, size)}}``. A
    constraint or design variable of size 1 keeps one column.
    """
    _require_file(path, 'Optimizer history')
    try:
        from pyoptsparse import History
    except ImportError as exc:
        raise ReaderError('pyoptsparse is not available: cannot read .hst files') from exc
    hist = History(path, flag='r')
    iters = hist.getValues(names='iter', major=True)
    iterations = np.asarray(iters['iter'], dtype=int).ravel()
    n = len(iterations)

    def column(name):
        values = hist.getValues(names=name, major=True)[name]
        return np.asarray(values)

    result = {'iterations': iterations, 'objectives': {}, 'constraints': {}, 'dvs': {}}
    for name in hist.getObjNames():
        result['objectives'][name] = column(name).ravel()
    for name in hist.getConNames():
        result['constraints'][name] = column(name).reshape(n, -1) if n else column(name)
    for name in hist.getDVNames():
        result['dvs'][name] = column(name).reshape(n, -1) if n else column(name)
    result['constraints'] = {k: (v[:, 0] if v.ndim == 2 and v.shape[1] == 1 else v)
                             for k, v in result['constraints'].items()}
    result['dvs'] = {k: (v[:, 0] if v.ndim == 2 and v.shape[1] == 1 else v)
                     for k, v in result['dvs'].items()}
    return result


def read_su2_cfg(path):
    """An SU2 cfg as ``{KEY: value}``: upper-case keys, comment-stripped strings.

    ``%`` starts a comment; the first occurrence of a key wins.
    """
    _require_file(path, 'SU2 cfg')
    values = {}
    with open(path) as stream:
        for line in stream:
            head, sep, val = line.split('%')[0].strip().partition('=')
            if sep and head.strip():
                values.setdefault(head.strip().upper(), val.strip())
    return values


def cfg_float(cfg, key, default=None):
    """``cfg[key]`` (a :func:`read_su2_cfg` dict) as a float, else ``default``.

    A value that is absent or not a single number returns ``default``.
    """
    text = cfg.get(key.upper())
    if text is None:
        return default
    try:
        return float(text.split()[0].strip('()'))
    except (ValueError, IndexError):
        return default


_NUMBER = re.compile(r'^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$')


def read_su2_stdout(path, table=-1):
    """The convergence table of SU2 screen output as ``{column: np.ndarray}``.

    The table is the ``|``-delimited block whose header row names the columns
    (``Inner_Iter``, ``rms[Rho]``, ``CL``, ...); rows are the lines under it
    whose cells are all numbers. SU2 repeats the header every few dozen rows;
    a repeat with the same columns continues the table, a header with other
    columns starts a new one. ``table`` selects one of them (default the last).
    """
    _require_file(path, 'SU2 stdout')
    tables = []
    with open(path, errors='replace') as stream:
        for line in stream:
            text = line.strip()
            if not (text.startswith('|') and text.endswith('|')):
                continue
            cells = [c.strip() for c in text.strip('|').split('|')]
            if cells and all(_NUMBER.match(c) for c in cells):
                if tables and len(tables[-1][0]) == len(cells):
                    tables[-1][1].append([float(c) for c in cells])
            elif cells and all(cells) and not any(_NUMBER.match(c) for c in cells):
                if tables and tables[-1][0] == cells:
                    continue
                tables.append((cells, []))
    tables = [t for t in tables if t[1]]
    if not tables:
        raise ReaderError(f'{path} has no SU2 convergence table')
    try:
        header, rows = tables[table]
    except IndexError:
        raise ReaderError(f'{path} has {len(tables)} convergence tables, not {table}') from None
    array = np.array(rows, dtype=float)
    return {name: array[:, i] for i, name in enumerate(header)}


def _read_rows(path, what):
    """``(header, [dict of str])`` of a CSV file with a header row."""
    _require_file(path, what)
    with open(path, newline='') as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        header = reader.fieldnames
    if not header or not rows:
        raise ReaderError(f'{what} {path} has no data rows')
    return [h.strip() for h in header], [{k.strip(): v for k, v in r.items() if k} for r in rows]


def _number(text):
    """The finite float of a cell, else None (empty, text, NaN)."""
    try:
        value = float(text)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def read_table(path):
    """A plain CSV as one ``{column: float}`` per data row.

    Cells that are empty, not a number or not finite are left out of that
    row's dict, so rows may hold different columns.
    """
    _, rows = _read_rows(path, 'Table file')
    return [{k: v for k, v in ((k, _number(c)) for k, c in row.items()) if v is not None}
            for row in rows]


def read_reference_cp(path, point=None):
    """A reference Cp distribution (CSV) as ``{x, cp, surface, point, alpha}``.

    ``x`` comes from ``x_c``; ``cp`` from ``Cp`` or ``cp`` (whichever exists).
    An optional ``surface`` column gives the label of each tap (else ``surface``
    is None). Rows whose Cp or x is not finite are dropped. A ``point`` column
    with several values needs ``point`` (a number); none selected raises with
    the points and their ``ALPHA``. ``point`` and ``alpha`` are the selected
    point and its ``ALPHA`` (None when the file has no such column). ``x`` and
    ``cp`` are arrays.
    """
    header, rows = _read_rows(path, 'Reference Cp file')
    cp_col = next((c for c in ('Cp', 'cp') if c in header), None)
    if cp_col is None or 'x_c' not in header:
        raise ReaderError(f'{path}: a reference Cp file needs x_c and Cp (or cp) columns; '
                          f'it has {header}')
    selected, alpha = None, None
    if 'point' in header:
        found = {}
        for row in rows:
            found.setdefault(row['point'].strip(), _number(row.get('ALPHA')))
        if point is None:
            if len(found) > 1:
                listing = ', '.join(f'{k} (ALPHA {v})' for k, v in found.items())
                raise ReaderError(f'{path} holds several points; select one with point=N: '
                                  f'{listing}')
            point = next(iter(found))
        key = next((k for k in found if _number(k) == _number(point)), None)
        if key is None:
            raise ReaderError(f'{path} has no point {point}; it has {list(found)}')
        rows = [r for r in rows if r['point'].strip() == key]
        selected, alpha = _number(key), found[key]
    elif point is not None:
        raise ReaderError(f'{path} has no point column, point={point} cannot be selected')
    x, cp, surface = [], [], []
    for row in rows:
        xv, cv = _number(row['x_c']), _number(row[cp_col])
        if xv is None or cv is None:
            continue
        x.append(xv)
        cp.append(cv)
        surface.append(row.get('surface', '').strip() if 'surface' in header else None)
    if not x:
        raise ReaderError(f'{path}: no finite Cp rows' + (f' for point {point}' if point else ''))
    return {'x': np.array(x), 'cp': np.array(cp),
            'surface': surface if 'surface' in header else None,
            'point': selected, 'alpha': alpha}


def file_kind(path):
    """``history`` (``.csv``, ``.dat``), ``hst``, ``cfg``, ``stdout`` (``.out``, ``.log``, ``.txt``) or ``polar`` (``.json``).

    Surface and volume files are not told apart by name: the caller says
    which it passes. Any other extension raises :class:`ReaderError`.
    """
    ext = os.path.splitext(path)[1].lower()
    kinds = {'.csv': 'history', '.dat': 'history', '.hst': 'hst', '.cfg': 'cfg',
             '.out': 'stdout', '.log': 'stdout', '.txt': 'stdout', '.json': 'polar'}
    if ext not in kinds:
        raise ReaderError(f'Cannot tell the kind of {path} from its extension')
    return kinds[ext]
