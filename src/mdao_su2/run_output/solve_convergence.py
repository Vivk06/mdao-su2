"""How each SU2 solve of a task converged, read from its history file.

Instance overrides of ``compute``, ``_run_adjoint`` and ``_run_response_adjoint``
leave the solver components unchanged. The same overrides
report each solve to an optional ``optimization.progress.Progress``.
An adjoint that ``adjoint_reuse.py`` answers from its cache never reaches
them, so it is neither counted nor reported.
"""

import math
import os
from contextlib import contextmanager
from types import MethodType

from ..solver.cfg_file import cfg_value
from .solve_hooks import component_point_id, solver_components


_HISTORY_CACHE = {}


def read_history_lines(path):
    """Non-blank lines of an SU2 history CSV, or None when the file is absent.

    The convergence tracker and the progress console both read each finished
    solve's history; a one-entry cache keyed by path, size and modification
    time serves every read after the first.
    """
    if not path or not os.path.isfile(path):
        return None
    stat = os.stat(path)
    key = (os.path.abspath(path), stat.st_size, stat.st_mtime_ns)
    if key not in _HISTORY_CACHE:
        with open(path) as stream:
            lines = [line for line in stream if line.strip()]
        _HISTORY_CACHE.clear()
        _HISTORY_CACHE[key] = lines
    return _HISTORY_CACHE[key]


def read_history_tail(path):
    """``(iterations, residual_field, final_residual)`` from an SU2 history CSV."""
    lines = read_history_lines(path)
    if lines is None or len(lines) < 2:
        return None
    names = [name.strip().strip('"').strip() for name in lines[0].split(',')]
    values = [value.strip() for value in lines[-1].split(',')]
    row = dict(zip(names, values))
    field = next((name for name in names if name.startswith('rms[')), None)
    try:
        iterations = int(float(row['Inner_Iter'])) + 1
    except (KeyError, ValueError):
        iterations = len(lines) - 1
    try:
        residual = float(row[field]) if field is not None else None
    except ValueError:
        residual = None
    return iterations, field, residual


def _history_path(cfg_path, default):
    stem = (cfg_value(cfg_path, 'CONV_FILENAME') or default).strip()
    return stem if stem.endswith('.csv') else f'{stem}.csv'


def _limits(cfg_path):
    """``(CONV_RESIDUAL_MINVAL, ITER)`` of a cfg, each None when absent."""
    def number(key, kind):
        try:
            return kind(float(cfg_value(cfg_path, key)))
        except (TypeError, ValueError):
            return None
    return number('CONV_RESIDUAL_MINVAL', float), number('ITER', int)


class SolveConvergence:
    """Per (kind, point, functional): solve count, last iterations and residual."""

    def __init__(self):
        self.entries = {}

    def note(self, kind, point, functional, history):
        key = (kind, point, functional)
        entry = self.entries.setdefault(key, {'solves': 0})
        entry['solves'] += 1
        tail = read_history_tail(history)
        if tail is None:
            return
        iterations, field, residual = tail
        entry.update(iterations=iterations, residual_field=field,
                     final_residual=residual)
        if residual is not None and math.isfinite(residual):
            worst = entry.get('worst_final_residual')
            entry['worst_final_residual'] = (residual if worst is None
                                             else max(worst, residual))

    def record(self, comm=None):
        """The JSON block; collective when ``comm`` spans several ranks."""
        entries = dict(self.entries)
        if comm is not None and getattr(comm, 'size', 1) > 1:
            for part in comm.allgather(self.entries):
                entries.update(part)
        out = {}
        for (kind, point, functional), entry in sorted(
                entries.items(), key=lambda item: tuple(str(k) for k in item[0])):
            target = out.setdefault(kind, {})
            if functional is None:
                target[point] = entry
            else:
                target.setdefault(point, {})[functional] = entry
        return out


class _Solve:
    """Report one solve to ``progress`` around the original call."""

    def __init__(self, progress, kind, point, functional=None, fd=False, limits=(None, None)):
        self.progress, self.args = progress, (kind, point, functional, fd)
        self.limits = limits
        self.token = None
        self.error = None

    def __enter__(self):
        if self.progress is not None:
            self.token = self.progress.started(*self.args)
        return self

    def __exit__(self, *exc):
        return False

    def done(self, history):
        if self.token is not None:
            target, limit = self.limits
            self.progress.finished(self.token, history, error=self.error,
                                   target=target, limit=limit)


@contextmanager
def track_solve_convergence(model, cfg, progress=None):
    """Yield a ``SolveConvergence`` filled by every solve run inside the block.

    ``progress`` (rank 0 only) is told when each solve starts and finishes.
    """
    tracker = SolveConvergence()
    installed = []
    systems = solver_components(model) if hasattr(model, 'system_iter') else ()
    for component in systems:
        point = component_point_id(component, cfg)
        primal_history = _history_path(component.options['primal_cfg'], 'history')
        adjoint_history = _history_path(component.options['adjoint_cfg'], 'history_adj')
        primal_limits = _limits(component.options['primal_cfg'])
        adjoint_limits = _limits(component.options['adjoint_cfg'])
        root = component.comm.rank == 0
        report = progress if root else None

        def compute(self, inputs, outputs, _original=component.compute,
                    _point=point, _history=primal_history, _root=root,
                    _report=report, _limits=primal_limits):
            solve = _Solve(_report, 'primal', _point,
                           fd=bool(getattr(self, 'under_approx', False)), limits=_limits)
            with solve:
                try:
                    return _original(inputs, outputs)
                except BaseException as exc:
                    solve.error = exc
                    raise
                finally:
                    if _root:
                        tracker.note('primal', _point, None, _history)
                        solve.done(_history)

        def run_adjoint(self, objective, inputs, _original=component._run_adjoint,
                        _point=point, _history=adjoint_history, _root=root,
                        _report=report, _limits=adjoint_limits):
            solve = _Solve(_report, 'adjoint', _point, str(objective), limits=_limits)
            with solve:
                try:
                    return _original(objective, inputs)
                except BaseException as exc:
                    solve.error = exc
                    raise
                finally:
                    if _root:
                        tracker.note('adjoint', _point, str(objective), _history)
                        solve.done(_history)

        names = ['compute', '_run_adjoint']
        wrappers = [compute, run_adjoint]
        if hasattr(component, '_run_response_adjoint'):
            def run_response(self, record, inputs,
                             _original=component._run_response_adjoint,
                             _point=point, _history=adjoint_history, _root=root,
                             _report=report, _limits=adjoint_limits):
                solve = _Solve(_report, 'adjoint', _point, str(record.su2_objective),
                               limits=_limits)
                with solve:
                    try:
                        return _original(record, inputs)
                    except BaseException as exc:
                        solve.error = exc
                        raise
                    finally:
                        if _root:
                            tracker.note('adjoint', _point,
                                         str(record.su2_objective), _history)
                            solve.done(_history)
            names.append('_run_response_adjoint')
            wrappers.append(run_response)
        for name, wrapper in zip(names, wrappers):
            had = name in component.__dict__
            installed.append((component, name, had, component.__dict__.get(name)))
            setattr(component, name, MethodType(wrapper, component))
    try:
        yield tracker
    finally:
        for component, name, had, previous in reversed(installed):
            if had:
                setattr(component, name, previous)
            else:
                component.__dict__.pop(name, None)
