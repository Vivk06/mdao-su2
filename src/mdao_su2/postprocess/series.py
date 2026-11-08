"""Samples, expressions and series: the scalar side of post-processing.

A *sample* is a dict ``{name: float}`` of the named scalars one file gives
(the last row of a history, one major iteration of an ``.hst``, a row of SU2
screen output), plus ``KEY=VAL`` overrides and values read from an SU2 cfg. A
*series* is the samples of many files, each expression of them evaluated and
sorted by the x expression. Headless: no matplotlib.

Expression rule (:func:`evaluate`)
----------------------------------
An expression is Python arithmetic over column names, evaluated by an AST
whitelist (no attribute access, subscripts, comprehensions or any call other
than the numpy functions below). Allowed: ``+ - * / // % **``, unary ``+ -``,
numbers, the constants ``pi`` and ``e``, and the functions ``sqrt log log10
exp abs sin cos tan min max`` (``min`` and ``max`` of one argument reduce an
array, of several arguments are element-wise).

A column name that is a valid identifier (``CL``, ``MACH``) is used as it is.
Any other name is addressed by its *alias*: every run of non-word characters
becomes one ``_`` and leading and trailing ``_`` are dropped, so ``rms[Rho]``
is ``rms_Rho``, ``point.C_L`` is ``point_C_L`` and ``x[3]`` is ``x_3``. Two
columns with the same alias raise. The exact name wins over an alias.

Derived names (:func:`derive_forces`)
-------------------------------------
Wherever expressions are evaluated over samples or history columns, a missing
``CN`` is ``CL cos(a) + CD sin(a)`` and a missing ``CL`` is
``(CN - CD sin(a)) / cos(a)``, with ``a`` the angle ``AoA`` (else ``AOA``) in
degrees. A name already present is never replaced.

Sample kinds
------------
``history``, ``hst`` and ``stdout`` give one sample per file; ``polar``
(a ``results_polar.json``) gives one per accepted angle of one polar point;
``table`` (a plain CSV) one per row. An override ``NAME=VALUE`` whose value is
not a number is an expression over the sample built so far.
"""
import ast
import json
import math
import operator
import re
from dataclasses import dataclass, field

import numpy as np

from .readers import (cfg_float, file_kind, read_history, read_hst,
                      read_su2_cfg, read_su2_stdout, read_table)

__all__ = [
    'SeriesError', 'SampleSpec', 'alias', 'sample_from_history', 'sample_from_hst',
    'sample_from_stdout', 'sample_from_cfg', 'parse_overrides', 'load_sample',
    'load_samples', 'sample_from_polar', 'derive_forces',
    'expression_names', 'evaluate', 'build_series',
    'fit_peak', 'evaluate_linear_row', 'linear_row_margin', 'linear_terms',
]


class SeriesError(Exception):
    """An expression or a sample set cannot be evaluated."""


def alias(name):
    """The identifier form of a column name: runs of non-word characters become ``_``."""
    return re.sub(r'\W+', '_', name).strip('_')


# ---- samples ---------------------------------------------------------------

def _row_sample(columns, row):
    """``{name: float}`` of one row of ``{column: array}``."""
    sample = {}
    for name, values in columns.items():
        try:
            sample[name] = float(values[row])
        except IndexError:
            raise SeriesError(f'row {row} is outside a column of {len(values)} values') from None
    return sample


def sample_from_history(path, row=-1):
    """The ``row``-th row (default the last) of an SU2 history file as a sample."""
    return _row_sample(read_history(path), row)


def sample_from_stdout(path, row=-1, table=-1):
    """The ``row``-th row of an SU2 screen-output convergence table as a sample."""
    return _row_sample(read_su2_stdout(path, table=table), row)


def sample_from_hst(path, iteration=-1):
    """One major iteration of a pyOptSparse ``.hst`` as a sample.

    ``iteration`` indexes the major iterations (default the last). The sample
    holds ``iteration`` (its number), each objective and constraint under its
    name, and each design variable under its name (``name[i]`` per element when
    the variable has several).
    """
    hst = read_hst(path)
    n = len(hst['iterations'])
    if n == 0:
        raise SeriesError(f'{path} has no major iterations')
    try:
        k = range(n)[iteration]
    except IndexError:
        raise SeriesError(f'{path} has {n} major iterations, not {iteration}') from None
    sample = {'iteration': float(hst['iterations'][k])}
    for group in ('objectives', 'constraints', 'dvs'):
        for name, values in hst[group].items():
            row = np.atleast_1d(values[k])
            if row.size == 1:
                sample[name] = float(row[0])
            else:
                sample.update({f'{name}[{i}]': float(v) for i, v in enumerate(row)})
    return sample


#: SU2 cfg keys exposed under a short name.
_CFG_ALIASES = {'MACH_NUMBER': 'MACH', 'GAMMA_VALUE': 'GAMMA', 'REYNOLDS_NUMBER': 'RE'}


def sample_from_cfg(path):
    """The numeric keys of an SU2 cfg as a sample, ``MACH_NUMBER`` also as ``MACH``.

    ``GAMMA_VALUE`` is also ``GAMMA`` and ``REYNOLDS_NUMBER`` also ``RE``;
    ``AOA`` keeps its name. Keys whose value is not a single number are left out.
    """
    cfg = read_su2_cfg(path)
    sample = {}
    for key in cfg:
        value = cfg_float(cfg, key)
        if value is not None and math.isfinite(value) and len(cfg[key].split()) == 1:
            sample[key] = value
            if key in _CFG_ALIASES:
                sample[_CFG_ALIASES[key]] = value
    return sample


def parse_overrides(items):
    """``['MACH=0.7', 'AOA=2']`` (or a dict) as ``{name: float}``."""
    if isinstance(items, dict):
        return {k: float(v) for k, v in items.items()}
    out = {}
    for item in items or ():
        name, _, value = item.partition('=')
        try:
            out[name.strip()] = float(value)
        except ValueError:
            raise SeriesError(f'override {item!r} is not NAME=NUMBER') from None
        if not name.strip():
            raise SeriesError(f'override {item!r} is not NAME=NUMBER')
    return out


def _split_override(item):
    name, _, value = item.partition('=')
    name = name.strip()
    if not name or not _:
        raise SeriesError(f'override {item!r} is not NAME=VALUE')
    return name, value.strip()


def sample_from_polar(path, point=None):
    """The samples of one polar point of a ``results_polar.json``: one per accepted angle.

    ``point`` is the point ``id``; a file with one point needs none, one with
    several and no ``point`` raises listing the ids. An angle whose
    ``accepted`` is false is left out: its values are not a converged result.
    A partial file (``complete`` false) gives the angles it has. Each sample
    holds ``AoA`` (degrees), ``CL``, ``CD``, ``CM``, ``BUFFET`` (when not
    null), ``MACH``, ``RE`` and ``URANS`` (1 for a URANS angle, else 0).
    """
    try:
        with open(path) as stream:
            payload = json.load(stream)
        points = payload['polar']['points']
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise SeriesError(f'{path} is not a results_polar.json: {exc}') from None
    ids = [p['id'] for p in points]
    if point is None:
        if len(points) != 1:
            raise SeriesError(f'{path} has several points; select one with point=ID: {ids}')
        chosen = points[0]
    else:
        found = [p for p in points if p['id'] == point]
        if not found:
            raise SeriesError(f'{path} has no point {point!r}; it has {ids}')
        chosen = found[0]
    samples = []
    for angle in chosen['angles']:
        if not angle.get('accepted'):
            continue
        sample = {'AoA': float(angle['alpha']), 'CL': float(angle['cl']),
                  'CD': float(angle['cd']), 'CM': float(angle['cm']),
                  'MACH': float(chosen['mach']), 'RE': float(chosen['reynolds']),
                  'URANS': 1.0 if angle.get('mode') == 'urans' else 0.0}
        if angle.get('buffet') is not None:
            sample['BUFFET'] = float(angle['buffet'])
        samples.append(sample)
    if not samples:
        raise SeriesError(f'{path}: point {chosen["id"]!r} has no accepted angle')
    return samples


@dataclass(frozen=True)
class SampleSpec:
    """Where samples come from: a file, a row or point, an optional cfg and overrides.

    ``kind`` is ``history``, ``hst``, ``stdout``, ``polar`` or ``table``; None
    takes it from the extension (``.csv`` and ``.dat`` history, ``.hst``,
    ``.out``/``.log`` stdout, ``.json`` polar; ``table`` is never inferred).
    ``row`` is the history or stdout row, the ``.hst`` major iteration, or for
    a ``table`` the 0-based data row (default -1 means all rows). ``point`` is
    the polar point id. ``cfg`` is an SU2 cfg whose numeric keys join each
    sample; ``overrides`` (``NAME=VALUE`` strings or a dict) win over
    everything: a number sets the name, anything else is an expression over
    the sample built so far (:func:`load_samples`).
    """
    path: str
    row: int = -1
    kind: str = None
    cfg: str = None
    overrides: tuple = field(default_factory=tuple)
    point: str = None

    def inputs(self):
        """``[(role, path)]`` of the files this sample reads."""
        found = [(self.kind or file_kind(self.path), self.path)]
        if self.cfg:
            found.append(('cfg', self.cfg))
        return found


def derive_forces(values):
    """``values`` (``{name: scalar or array}``) with ``CN`` or ``CL`` added when derivable.

    Without ``CN`` but with ``CL``, ``CD`` and an angle in degrees (``AoA``,
    else ``AOA``): ``CN = CL cos(a) + CD sin(a)``. Without ``CL`` but with
    ``CN``, ``CD`` and the angle: ``CL = (CN - CD sin(a)) / cos(a)``. A name
    already in ``values`` is never replaced. Returns a new dict.
    """
    out = dict(values)
    angle = next((out[n] for n in ('AoA', 'AOA') if n in out), None)
    if angle is None or 'CD' not in out:
        return out
    a = np.radians(np.asarray(angle, dtype=float))
    cd = np.asarray(out['CD'], dtype=float)
    scalar = np.ndim(angle) == 0 and np.ndim(cd) == 0

    def result(value):
        return float(value) if scalar else value

    with np.errstate(all='ignore'):
        if 'CN' not in out and 'CL' in out:
            out['CN'] = result(np.asarray(out['CL'], dtype=float) * np.cos(a) + cd * np.sin(a))
        elif 'CL' not in out and 'CN' in out:
            out['CL'] = result((np.asarray(out['CN'], dtype=float) - cd * np.sin(a)) / np.cos(a))
    return out


def _apply_overrides(sample, overrides):
    """Overrides in order: numbers set the name; other values are expressions.

    An expression naming a column the sample lacks (an empty cell of a table
    row) leaves the name unset.
    """
    items = overrides.items() if isinstance(overrides, dict) else \
        [_split_override(i) for i in overrides or ()]
    for name, value in items:
        try:
            sample[name] = float(value)
            continue
        except (TypeError, ValueError):
            pass
        view = derive_forces(sample)
        known = _lookup_table(view)
        if any(n not in known for n in expression_names(value)):
            continue
        result = evaluate(value, view)
        if result.size != 1:
            raise SeriesError(f'override {name}={value} is not a scalar')
        sample[name] = float(result)
    return sample


def load_samples(spec):
    """The samples of a :class:`SampleSpec`: file values, cfg values, overrides, derived names.

    ``history``, ``hst`` and ``stdout`` give one sample (``row``); ``polar``
    one per accepted angle of the ``point`` (:func:`sample_from_polar`);
    ``table`` one per CSV row, or only data row ``row`` when ``row`` >= 0
    (the default -1 is all rows). A file value wins over the cfg's of the same
    name; overrides apply in order; then :func:`derive_forces`.
    """
    kind = spec.kind or file_kind(spec.path)
    if kind == 'history':
        samples = [sample_from_history(spec.path, spec.row)]
    elif kind == 'hst':
        samples = [sample_from_hst(spec.path, spec.row)]
    elif kind == 'stdout':
        samples = [sample_from_stdout(spec.path, spec.row)]
    elif kind == 'polar':
        samples = sample_from_polar(spec.path, spec.point)
    elif kind == 'table':
        samples = read_table(spec.path)
        if spec.row >= 0:
            if spec.row >= len(samples):
                raise SeriesError(f'{spec.path} has {len(samples)} rows, not row {spec.row}')
            samples = [samples[spec.row]]
    else:
        raise SeriesError(f'{spec.path}: no sample from a {kind!r} file')
    cfg = sample_from_cfg(spec.cfg) if spec.cfg else {}
    return [derive_forces(_apply_overrides({**cfg, **s}, spec.overrides)) for s in samples]


def load_sample(spec):
    """The one sample of a single-sample :class:`SampleSpec` (``history``, ``hst``, ``stdout``)."""
    samples = load_samples(spec)
    if len(samples) != 1:
        raise SeriesError(f'{spec.path}: {len(samples)} samples, use load_samples')
    return samples[0]


# ---- expressions -------------------------------------------------------------

def _reduce(many, one):
    def apply(*args):
        if len(args) == 1:
            return one(args[0])
        out = args[0]
        for arg in args[1:]:
            out = many(out, arg)
        return out
    return apply


_FUNCTIONS = {
    'sqrt': np.sqrt, 'log': np.log, 'log10': np.log10, 'exp': np.exp, 'abs': np.abs,
    'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
    'min': _reduce(np.minimum, np.min), 'max': _reduce(np.maximum, np.max),
}
_CONSTANTS = {'pi': math.pi, 'e': math.e}
_BINARY = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
           ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
           ast.Mod: operator.mod, ast.Pow: operator.pow}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _parse(expr):
    try:
        return ast.parse(expr.strip(), mode='eval').body
    except SyntaxError as exc:
        raise SeriesError(f'cannot parse expression {expr!r}: {exc.msg}') from None


def expression_names(expr):
    """The column names (or aliases) an expression uses, in first-seen order."""
    names = []
    for node in ast.walk(_parse(expr)):
        if isinstance(node, ast.Name) and node.id not in _FUNCTIONS and node.id not in _CONSTANTS:
            if node.id not in names:
                names.append(node.id)
    return names


def _lookup_table(columns):
    """Names an expression may use: identifier columns as they are, others by alias."""
    owners = {}
    for name in columns:
        key = alias(name)
        if name.isidentifier() or not key:
            continue
        if owners.setdefault(key, name) != name:
            raise SeriesError(f'columns {owners[key]!r} and {name!r} share the alias {key!r}')
    table = {key: columns[name] for key, name in owners.items()}
    table.update({name: value for name, value in columns.items() if name.isidentifier()})
    return table


def evaluate(expr, columns):
    """The value of ``expr`` over ``columns`` (``{name: array or float}``) as a float array.

    See the module docstring for the allowed syntax and the alias rule for
    names such as ``rms[Rho]``. Anything outside the whitelist raises
    :class:`SeriesError`.
    """
    table = _lookup_table(columns)

    def walk(node):
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise SeriesError(f'{expr!r}: only numbers are allowed as constants')
            return float(node.value)
        if isinstance(node, ast.Name):
            if node.id in table:
                return np.asarray(table[node.id], dtype=float)
            if node.id in _CONSTANTS:
                return _CONSTANTS[node.id]
            raise SeriesError(f'{expr!r}: no column {node.id!r}; available '
                              f'{sorted(set(table))}')
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
            return _BINARY[type(node.op)](walk(node.left), walk(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
            return _UNARY[type(node.op)](walk(node.operand))
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id in _FUNCTIONS and not node.keywords and node.args):
            return _FUNCTIONS[node.func.id](*[walk(a) for a in node.args])
        raise SeriesError(f'{expr!r}: {type(node).__name__} is not allowed in an expression')

    with np.errstate(all='ignore'):
        return np.asarray(walk(_parse(expr)), dtype=float)


def build_series(samples, x_expr, y_exprs, skip_incomplete=False):
    """``{'x', 'y': {expr: array}, 'samples'}`` of ``samples``, sorted by ``x_expr``.

    ``samples`` is a list of sample dicts. Every column an expression names
    must be in every sample, unless ``skip_incomplete`` drops the samples
    that lack one (for reference tables with empty cells).
    """
    if not samples:
        raise SeriesError('no samples')
    needed = []
    for expr in (x_expr, *y_exprs):
        needed += [n for n in expression_names(expr) if n not in needed]
    if skip_incomplete:
        samples = [s for s in samples if all(n in _lookup_table(s) for n in needed)]
        if not samples:
            raise SeriesError(f'no sample has all of {needed}')
    arrays = {}
    for name in {n for sample in samples for n in sample}:
        if all(name in sample for sample in samples):
            arrays[name] = np.array([sample[name] for sample in samples], dtype=float)
    missing = [n for n in needed if n not in _lookup_table(arrays)]
    if missing:
        raise SeriesError(f'columns {missing} are missing from at least one sample; '
                          f'the first sample has {sorted(samples[0])}')
    x = evaluate(x_expr, arrays) * np.ones(len(samples))
    order = np.argsort(x, kind='stable')
    return {'x': x[order],
            'y': {e: (evaluate(e, arrays) * np.ones(len(samples)))[order] for e in y_exprs},
            'samples': [samples[i] for i in order]}


# ---- pure array helpers --------------------------------------------------------

def fit_peak(x, y):
    """Maximum of ``y`` and the vertex of the parabola through it and its neighbours.

    Returns ``(x_max, y_max, x_vertex, y_vertex)``; the vertex is ``(None,
    None)`` when the maximum is an end point or the three points do not form a
    maximum.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    k = int(np.argmax(y))
    if k == 0 or k == len(x) - 1:
        return float(x[k]), float(y[k]), None, None
    a, b, c = np.polyfit(x[k - 1:k + 2], y[k - 1:k + 2], 2)
    if a >= 0:
        return float(x[k]), float(y[k]), None, None
    xv = -b / (2.0 * a)
    return float(x[k]), float(y[k]), float(xv), float(a * xv ** 2 + b * xv + c)


def evaluate_linear_row(columns, terms, offset=0.0):
    """``sum(coeff * columns[name]) + offset`` for ``terms`` of ``(column, coeff)``.

    ``columns`` maps a column name to an array (one value per iteration); every
    column has the same length.
    """
    value = 0.0
    for name, coeff in terms:
        if name not in columns:
            raise SeriesError(f'no column {name!r}; available {sorted(columns)}')
        value = value + coeff * np.asarray(columns[name], dtype=float)
    return value + float(offset)


def linear_row_margin(columns, terms, offset=0.0, lower=None, upper=None, equals=None):
    """Margin of a linear constraint row per element of the columns, in percent.

    ``terms`` are ``(column, coeff)``. The margin is the signed distance to the
    bound (positive: met), divided by the sum of the ``|coeff * value|`` of the
    terms that work against the bound (the negative terms for a lower bound,
    the positive terms for an upper bound, all terms for ``equals``). With both
    bounds the smaller margin is taken.
    """
    contributions = np.array([coeff * evaluate_linear_row(columns, [(name, 1.0)])
                              for name, coeff in terms])
    value = contributions.sum(axis=0) + float(offset)
    margins = []

    def percent(distance, against):
        scale = np.abs(contributions * against).sum(axis=0)
        total = np.abs(contributions).sum(axis=0)
        return 100.0 * distance / np.where(scale > 0.0, scale, total)

    if lower is not None:
        margins.append(percent(value - lower, (contributions < 0.0).astype(float)))
    if upper is not None:
        margins.append(percent(upper - value, (contributions > 0.0).astype(float)))
    if equals is not None:
        margins.append(percent(-np.abs(value - equals), np.ones_like(contributions)))
    if not margins:
        raise SeriesError('A linear row needs a bound: lower, upper or equals')
    return np.min(margins, axis=0)


def linear_terms(expr):
    """``(terms, offset)`` of a linear expression: ``'2*x_1 - x_2 + 3'`` is
    ``([('x_1', 2.0), ('x_2', -1.0)], 3.0)``.

    Allowed: names, numbers, ``+ -``, a product or quotient with a number, and
    parentheses. Anything non-linear raises :class:`SeriesError`. The terms are
    the ``terms`` and ``offset`` of :func:`linear_row_margin`.
    """
    def walk(node):
        """``({name: coeff}, constant)`` of a node."""
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return {}, float(node.value)
        if isinstance(node, ast.Name):
            return {node.id: 1.0}, 0.0
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            coeffs, const = walk(node.operand)
            sign = -1.0 if isinstance(node.op, ast.USub) else 1.0
            return {k: sign * v for k, v in coeffs.items()}, sign * const
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
            (ca, ka), (cb, kb) = walk(node.left), walk(node.right)
            sign = 1.0 if isinstance(node.op, ast.Add) else -1.0
            out = dict(ca)
            for k, v in cb.items():
                out[k] = out.get(k, 0.0) + sign * v
            return out, ka + sign * kb
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Mult, ast.Div)):
            (ca, ka), (cb, kb) = walk(node.left), walk(node.right)
            if isinstance(node.op, ast.Div):
                if cb or kb == 0.0:
                    raise SeriesError(f'{expr!r}: divide only by a non-zero number')
                return {k: v / kb for k, v in ca.items()}, ka / kb
            if ca and cb:
                raise SeriesError(f'{expr!r} is not linear: it multiplies two names')
            coeffs, const, scale = (cb, kb, ka) if cb else (ca, ka, kb)
            return {k: v * scale for k, v in coeffs.items()}, const * scale
        raise SeriesError(f'{expr!r} is not a linear expression of names and numbers')

    coeffs, const = walk(_parse(expr))
    return [(name, c) for name, c in coeffs.items()], const
