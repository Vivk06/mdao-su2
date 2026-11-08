"""Planned and finished SU2 solves of a task, one terminal line per solve.

``plan_task`` counts the solves a task will run from the config,
the resolved derivative mode and the built model's DV sizes. ``Progress``
receives each solve from the instance overrides of
``run_output/solve_convergence.py`` and prints, in aligned columns,

    [primal   7/41] ok      sp0  FD shape[5] +h     1807 it  -10.02/-10     12.4 s   left ~8 min

(counter, status, point, label, iterations, final residual over its target or
``CAP`` when the solve stopped at its iteration limit, time, time left) to the
console (terminal and ``logs/progress.log``), writes a header line into
``logs/solver.log`` before the solve (the console trims a successful solve's
output after it), and copies the solve's history to
``logs/solves/<seq>_<kind>_<point>_<tag>.csv``. A blank line separates
consecutive groups: a change of solve kind, or under ``run_driver`` each
optimizer call.

Under ``run_driver`` the primals of one optimizer call print as one call line
(``call_started`` / ``call_ended``, from ``IterationSnapshotter``), with the
call's objective, or ``FAILED`` and the reason. A major iteration closes a
segment (``end_segment``): the next call line, or the ``final`` primal,
follows a ``SEGMENT_RULE`` between blank lines.

An FD line names the DV entry whose value differs from the last plain
primal's, and the sign of that difference; an alpha step reads
``FD alpha[<point>] +h``. The plan counts FD solves the way OpenMDAO runs
them: each index of a ``wrt`` group in turn, +h before -h under the central
form, re-running only the systems the DV reaches (every point for shape and
twist, the owning point for ``alpha_i``). Without DV values the k-th FD solve
takes the k-th planned step, and ``FD ?`` when its point does not match.
"""

import os
import shutil
import time

import numpy as np
from dataclasses import dataclass, field

from ..run_config.root import DERIVATIVE_TASKS

__all__ = ['FDStep', 'Plan', 'plan_task', 'Progress', 'adjoint_label',
           'read_history_drift', 'FD_DRIFT_WINDOW']

#: Iterations at the end of a primal's history over which a functional's
#: remaining drift is measured; the drift bounds how well that solve resolves
#: the functional, and so how small a finite difference it can resolve.
FD_DRIFT_WINDOW = 50

#: Sign order of OpenMDAO's finite-difference forms.
FD_SIGNS = {'forward': (1,), 'backward': (-1,), 'central': (1, -1)}

_ADJOINT_LABEL = {'DRAG': 'C_D', 'LIFT': 'C_L'}

#: Counter digits under ``run_driver``, where no total is known.
OPEN_COUNT_WIDTH = 4
#: The point column of a ``run_driver`` call line that solved several points.
ALL_POINTS = 'all'
#: The widest ``run_driver`` label the label column makes room for.
DRIVER_LABEL_SPAN = 'call 999  C_D'
#: Status column: a solve or call that raised reads ``FAILED``.
STATUS_WIDTH = 6
#: Separates ``run_driver`` segments: the calls leading to one major
#: iteration, its block and its adjoints.
SEGMENT_RULE = '=' * 72
#: ``logs/solves`` tag of a primal the plan did not foresee.
UNPLANNED_TAG = 'unplanned'


def read_history_drift(path, window=FD_DRIFT_WINDOW):
    """``{column: max - min over the last window rows}`` of an SU2 history CSV."""
    from ..run_output.solve_convergence import read_history_lines
    lines = read_history_lines(path)
    if lines is None or len(lines) < 2:
        return None
    names = [name.strip().strip('"').strip() for name in lines[0].split(',')]
    rows = []
    for line in lines[1:][-window:]:
        try:
            rows.append([float(value) for value in line.split(',')])
        except ValueError:
            continue
    if not rows:
        return None
    columns = np.array([row for row in rows if len(row) == len(names)])
    if columns.size == 0:
        return None
    return {name: float(np.ptp(columns[:, k])) for k, name in enumerate(names)}


def adjoint_label(objective):
    """``DRAG`` -> ``C_D``; a moment objective -> ``C_M``; others unchanged."""
    name = str(objective)
    if name.upper().startswith('MOMENT_'):
        return 'C_M'
    return _ADJOINT_LABEL.get(name.upper(), name)


@dataclass(frozen=True)
class FDStep:
    """One FD primal: DV group, DV index, sign of the step, solving point.

    An alpha step also names the point that owns the alpha DV (``owner``)
    and reads ``FD alpha[<owner>] +h``.
    """
    group: str
    index: int
    sign: int
    point: str
    owner: str = None

    @property
    def _step(self):
        return '+h' if self.sign > 0 else '-h'

    @property
    def label(self):
        if self.group == 'alpha':
            return f'FD alpha[{self.owner}] {self._step}'
        return f'FD {self.group}[{self.index}] {self._step}'

    @property
    def tag(self):
        side = 'p' if self.sign > 0 else 'm'
        if self.group == 'alpha':
            return f'alpha_{self.owner}{side}'
        return f'{self.group}{self.index}{side}'


@dataclass
class Plan:
    """What a task will solve. ``primals`` labels the plain primals in order."""
    task: str
    points: tuple
    dv_counts: dict
    mode: str
    fd_form: str = None
    #: Every point ID in order; ``alpha_i`` belongs to ``all_points[i]``.
    all_points: tuple = ()
    primals: list = field(default_factory=list)
    fd: list = field(default_factory=list)
    adjoints: dict = field(default_factory=dict)
    bounded: bool = False
    open_ended: bool = False
    note: str = ''

    @property
    def totals(self):
        """``{'primal': n, 'adjoint': m}``; ``None`` where the optimizer decides."""
        if self.open_ended:
            return {'primal': None, 'adjoint': None}
        return {'primal': len(self.primals) + len(self.fd),
                'adjoint': sum(len(v) for v in self.adjoints.values())}

    @property
    def count_width(self):
        """Digits of the widest total of any kind; the optimizer's count gets 4."""
        if self.open_ended:
            return OPEN_COUNT_WIDTH
        return max([len(str(n)) for n in self.totals.values() if n] or [1])

    @property
    def point_width(self):
        """Width of the longest point ID (``all`` names every point of a call)."""
        return max([len(p) for p in self.all_points or self.points]
                   + ([len(ALL_POINTS)] if self.open_ended else []))

    @property
    def label_width(self):
        """Width of the longest label the plan can print."""
        labels = list(self.primals) + [step.label for step in self.fd]
        labels += [adjoint_label(f) for names in self.adjoints.values() for f in names]
        if self.open_ended:
            labels.append(DRIVER_LABEL_SPAN)
        return max([len(label) for label in labels] or [1])

    def header(self):
        """The lines printed before the first solve."""
        dvs = ' '.join(f'{name} {count}' for name, count in self.dv_counts.items())
        fd = f'  FD {self.fd_form}' if self.fd_form else ''
        mode = f'  mode {self.mode}' if self.task in DERIVATIVE_TASKS else ''
        lines = [f'[plan] {self.task}  points {" ".join(self.points)}  '
                 f'DVs {dvs or "none"}{mode}{fd}']
        if self.open_ended:
            text = 'solves: as the optimizer requests'
            if self.primals:
                text += f'; trim first, up to {len(self.primals)} primal'
            lines.append(f'[plan] {text}')
        else:
            counts = {}
            for label in self.primals:
                key = label.split()[0]
                counts[key] = counts.get(key, 0) + 1
            parts = [f'{n} {k}' for k, n in counts.items()]
            if self.fd:
                parts.append(f'{len(self.fd)} FD')
            bound = 'up to ' if self.bounded else ''
            total = self.totals
            text = (f'[plan] solves: {bound}{total["primal"]} primal'
                    f' ({", ".join(parts)})' if parts else
                    f'[plan] solves: {bound}{total["primal"]} primal')
            if total['adjoint']:
                per = '; '.join(f'{p}: {" ".join(f)}' for p, f in self.adjoints.items() if f)
                text += f', {total["adjoint"]} adjoint ({per})'
            lines.append(text)
        if self.note:
            lines.append(f'[plan] {self.note}')
        return lines


def _derivative_functionals(cfg, task, mode, points):
    """Per point, the SU2AeroComp outputs whose adjoint the task runs once.

    Reverse mode runs one adjoint per seeded output; ``check_totals`` reuses
    them across groups (``adjoint_reuse.py``). Forward mode computes every
    output the builder names (``cfg.aero_functionals``) once per point.
    """
    if task == 'moment_pivot_check':
        return {p: ('C_M',) for p in points}
    if mode == 'fwd':
        return {p: tuple(cfg.aero_functionals) for p in points}
    base = cfg.objective_compiled.base_functionals
    out = {}
    for point in points:
        names = list(base.get(point, ()))
        if cfg.constraints.lift.applies(point):
            names.append('C_L')
        if cfg.constraints.moment.applies(point):
            names.append('C_M')
        for _constraint, compiled in cfg.linear_compiled:
            names += list(compiled.base_functionals.get(point, ()))
        if task == 'check_totals':
            names += list(cfg.verification.shape_outputs)
            names += list(cfg.verification.alpha_outputs)
        out[point] = tuple(dict.fromkeys(names))
    return out


def plan_task(cfg, mode, dv_counts, points=None):
    """The solves ``cfg.execution.task`` will run.

    ``mode`` is the resolved derivative mode (``rev`` or ``fwd``),
    ``dv_counts`` the built model's sizes ``{'shape': n, 'twist': m}``, and
    ``points`` the point IDs this rank reports (default: every point).
    """
    task = cfg.execution.task
    all_points = tuple(p.id for p in cfg.points)
    points = tuple(points) if points is not None else all_points
    counts = {name: int(dv_counts.get(name, 0)) for name in cfg.geometry_dvs}
    counts['alpha'] = len(all_points)
    plan = Plan(task=task, points=points, dv_counts=counts, mode=mode,
                all_points=all_points)
    if len(points) != len(all_points):
        plan.note = (f'this rank reports {len(points)} of {len(all_points)} points '
                     f'(parallel_points)')
    per_call = list(points)

    def run_model(label):
        plan.primals.extend([label] * len(per_call))

    def trim():
        for k in range(cfg.trim.max_iter):
            run_model(f'trim {k + 1}')

    if task == 'run_model':
        run_model('baseline')
    elif task == 'trim':
        trim()
        plan.bounded = True
        plan.note = (plan.note + '; ' if plan.note else '') + (
            f'upper bound: secant limit {cfg.trim.max_iter} per point')
    elif task in ('compute_totals', 'moment_pivot_check'):
        run_model('baseline')
        plan.adjoints = _derivative_functionals(cfg, task, mode, points)
    elif task == 'check_totals':
        form = cfg.verification.form
        plan.fd_form = form
        signs = FD_SIGNS[form]
        run_model('baseline')
        for group in ('shape', 'twist'):
            if group not in cfg.geometry_dvs:
                continue
            plan.fd += [FDStep(group, i, s, p) for i in range(counts[group])
                        for s in signs for p in points]
            run_model('refresh')
        plan.fd += [FDStep('alpha', i, s, p, owner=p)
                    for i, p in enumerate(all_points) for s in signs if p in points]
        plan.adjoints = _derivative_functionals(cfg, task, mode, points)
    elif task == 'polar':
        for angle in cfg.polar.alpha:
            run_model(f'alpha {angle:.2f}')
    elif task == 'run_driver':
        plan.open_ended = True
        if cfg.trim.enabled:
            trim()
    return plan


def _duration(seconds):
    if seconds < 90:
        return f'~{seconds:.0f} s'
    if seconds < 7200:
        return f'~{seconds / 60:.0f} min'
    return f'~{seconds / 3600:.1f} h'


def _reason(error):
    """The first line of an exception's message, or its type."""
    return str(error).splitlines()[0] if str(error) else type(error).__name__


def _convergence(history, target, limit):
    """``(iterations, final residual, target, capped)`` of one solve, or None."""
    if not history:
        return None
    from ..run_output.solve_convergence import read_history_tail
    tail = read_history_tail(history)
    if tail is None:
        return None
    iterations, _field, residual = tail
    capped = (limit is not None and iterations >= limit
              and (residual is None or target is None or residual > target))
    return iterations, residual, target, capped


def _worst(records):
    """One call's convergence over its points: most iterations, worst residual."""
    records = [r for r in records if r is not None]
    if not records:
        return None
    residuals = [r[1] for r in records if r[1] is not None]
    return (max(r[0] for r in records), max(residuals) if residuals else None,
            records[0][2], any(r[3] for r in records))


def _format_convergence(conv):
    """``('  1807 it', ' -10.02/-10')``: iterations and residual over target or CAP."""
    if conv is None:
        return f'{"-":>6} it', f'{"-":>11}'
    iterations, residual, target, capped = conv
    shown = '-' if residual is None else f'{residual:.2f}'
    goal = 'CAP' if capped else ('-' if target is None else f'{target:g}')
    return f'{iterations:>6} it', f'{shown + "/" + goal:>11}'


class Progress:
    """Count, label and time each solve against a ``Plan``.

    ``console`` provides ``terminal`` (a stream, or ``None``), ``solver_note``
    and ``truncate_last_output``; ``solves_dir`` receives history copies
    (``None`` under ``run_driver``); ``call(kind)`` returns the label of the
    optimizer call the solve of that kind belongs to (``call 3`` names
    ``evaluations/calls/call_0003``), or the driver's counter;
    ``dv_values`` returns ``{name: array}`` for every DV (``shape``,
    ``twist``, ``alpha_<i>``) as the model holds them now.

    An FD solve is labelled from the DV values themselves: the one entry that
    differs from the last plain primal's values names the DV, its index and
    the sign of the step. The planned FD sequence is the fallback when the
    values are not available or differ in more than one entry.
    """

    def __init__(self, plan, console=None, solves_dir=None, call=None,
                 dv_values=None, clock=time.monotonic):
        self.plan = plan
        self.dv_values = dv_values
        self._baseline = None
        self.console = console
        self.solves_dir = solves_dir
        self.call = call
        self.clock = clock
        self.seq = 0
        self.done = {'primal': 0, 'adjoint': 0}
        self.durations = {'primal': [], 'adjoint': []}
        self._plain = 0
        self._fd = 0
        self.lines = []
        #: Per FD step ``(group, index, sign, point)``: the perturbed primal's
        #: history drift and the drift of the plain primal before it.
        self.fd_drift = {}
        self._base_drift = {}
        self._group = None
        #: The primals of the optimizer call in progress, printed as one line.
        self._call_solves = None
        #: A major iteration ended its segment; the next call prints the rule.
        self._segment_ended = False

    # -- console --------------------------------------------------------
    def _say(self, line):
        self.lines.append(line)
        stream = getattr(self.console, 'terminal', None)
        if stream is not None:
            print(line, file=stream, flush=True)

    def say(self, line):
        """Print a line that is not a solve (the ``major`` line and its block)."""
        self._say(line)

    def end_segment(self):
        """A major iteration closes the segment; the rule waits for the next call."""
        self._segment_ended = True

    def _segment_rule(self):
        """Blank line, ``SEGMENT_RULE``, blank line, once after a major iteration."""
        if not self._segment_ended:
            return
        self._segment_ended = False
        self._say('')
        self._say(SEGMENT_RULE)
        self._say('')
        self._group = None

    def _say_solve(self, line, group):
        """Print a solve or call line, after a blank line when the group changes."""
        if self._group is not None and group != self._group:
            self._say('')
        self._group = group
        self._say(line)

    def announce(self):
        for line in self.plan.header():
            self._say(line)
        self._say('')

    def _call(self, kind=None):
        """The call the solve belongs to: a label, or the driver's counter."""
        number = self.call(kind)
        if isinstance(number, str):
            return number
        return f'call {number}' if number else 'call initial'

    def _snapshot(self):
        return {name: np.array(value, dtype=float).ravel()
                for name, value in self.dv_values().items()}

    def _perturbation(self, point):
        """The ``FDStep`` the current DV values show, or None."""
        if self.dv_values is None or self._baseline is None:
            return None
        moved = []
        for name, value in self._snapshot().items():
            base = self._baseline.get(name)
            if base is None or base.shape != value.shape:
                return None
            for index in np.flatnonzero(value != base):
                moved.append((name, int(index), float(value[index] - base[index])))
        if len(moved) != 1:
            return None
        name, index, delta = moved[0]
        sign = 1 if delta > 0 else -1
        if name.startswith('alpha_') and name[6:].isdigit():
            number = int(name[6:])
            owners = self.plan.all_points
            owner = owners[number] if number < len(owners) else name[6:]
            return FDStep('alpha', number, sign, point, owner=owner)
        return FDStep(name, index, sign, point)

    # -- one solve ------------------------------------------------------
    def started(self, kind, point, functional=None, fd=False):
        """Label the solve about to run; return the token ``finished`` takes."""
        self.seq += 1
        seen = None
        if kind == 'adjoint':
            label, tag = adjoint_label(functional), adjoint_label(functional)
        elif fd:
            step = self.plan.fd[self._fd] if self._fd < len(self.plan.fd) else None
            self._fd += 1
            seen = self._perturbation(point)
            if seen is None and step is not None and step.point == point:
                seen = step
            if seen is not None:
                label, tag = seen.label, seen.tag
            else:
                label, tag = 'FD ?', 'fd'
        else:
            if self.dv_values is not None:
                self._baseline = self._snapshot()
            planned = self._plain < len(self.plan.primals)
            label = self.plan.primals[self._plain] if planned else ''
            self._plain += 1
            tag = label.replace(' ', '') if planned else UNPLANNED_TAG
            if self.call is not None and not planned:
                label = self._call(kind)
                if label == 'final':
                    self._segment_rule()
        group = kind
        if self.call is not None:
            # Under run_driver a group is one optimizer call and its adjoints.
            group = self._call(kind)
            if kind == 'adjoint':
                label = f'{group}  {label}'
        token = {'seq': self.seq, 'kind': kind, 'point': point, 'label': label,
                 'tag': tag, 'start': self.clock(), 'group': group,
                 'fd': bool(fd) and kind == 'primal',
                 'step': seen if (fd and kind == 'primal') else None}
        console = self.console
        if console is not None and getattr(console, 'solver_stdout', 'log') == 'discard':
            console.truncate_last_output()
        header = f'==== solve {self.seq}: {kind} {point} {label} ===='
        note = (getattr(console, 'solve_started', None)
                or getattr(console, 'solver_note', None))
        if note is not None:
            note(header)
        return token

    def finished(self, token, history=None, error=None, target=None, limit=None):
        """Print the solve's line, marked FAILED with its reason, and keep its history.

        ``target`` is the cfg's ``CONV_RESIDUAL_MINVAL`` and ``limit`` its
        ``ITER``; the line reads the solve's iterations and final residual
        from ``history``. Inside an optimizer call a primal waits for the
        call's line (``call_ended``).
        """
        kind = token['kind']
        elapsed = self.clock() - token['start']
        trim = getattr(self.console, 'solve_finished', None)
        if trim is not None:
            trim(error is None)
        self.done[kind] += 1
        self.durations[kind].append(elapsed)
        token.update(elapsed=elapsed, error=error,
                     convergence=_convergence(history, target, limit))
        if kind == 'primal' and self._call_solves is not None:
            self._call_solves.append(token)
        else:
            self._say_solve(self.line(token, elapsed), token['group'])
        if kind == 'primal' and error is None:
            drift = read_history_drift(history)
            if token['fd']:
                step = token['step']
                if step is not None:
                    key = (step.group, step.index, step.sign, token['point'])
                    self.fd_drift[key] = (drift, self._base_drift.get(token['point']))
            else:
                self._base_drift[token['point']] = drift
        if self.solves_dir is not None and history and os.path.isfile(history):
            os.makedirs(self.solves_dir, exist_ok=True)
            name = f'{token["seq"]:04d}_{kind}_{token["point"]}_{token["tag"]}.csv'
            shutil.copyfile(history, os.path.join(self.solves_dir, name))

    # -- optimizer calls (run_driver) -------------------------------------
    def call_started(self, entry=None):
        """An optimizer call begins: hold its primals for one call line."""
        self._call_solves = []
        self._segment_rule()

    def call_ended(self, entry):
        """Print the call's line: its primals, objective, or FAILED and the reason.

        ``entry`` is the ``IterationSnapshotter`` call record (``call``,
        ``status``, ``reason``, ``objective``).
        """
        solves = self._call_solves or []
        self._call_solves = None
        number = entry['call']
        points = {t['point'] for t in solves}
        point = solves[0]['point'] if len(points) == 1 else ALL_POINTS
        error = next((t['error'] for t in solves if t['error'] is not None), None)
        failed = entry.get('status') == 'failed' or error is not None
        reason = entry.get('reason') or (_reason(error) if error is not None else '')
        reason = str(reason).splitlines()[0] if reason else ''
        conv = _worst([t['convergence'] for t in solves])
        elapsed = sum(t['elapsed'] for t in solves)
        objective = entry.get('objective')
        tail = (f'   {reason}' if failed else
                f'   obj {objective:.6e}' if objective is not None else '')
        count = f'{number:>{self.plan.count_width}}'
        self._say_solve(self._columns('call', count, 'FAILED' if failed else 'ok',
                                      point, 'primal', conv, elapsed) + tail,
                        f'call {number}')

    # -- line format ------------------------------------------------------
    def _columns(self, kind, count, status, point, label, conv, elapsed):
        plan = self.plan
        iterations, residual = _format_convergence(conv)
        return (f'[{kind:<7} {count}] {status:<{STATUS_WIDTH}}  '
                f'{point:<{plan.point_width}}  {label:<{plan.label_width}}  '
                f'{iterations}  {residual}  {elapsed:8.1f} s')

    def fd_resolution(self, group, index, point, column, step, form):
        """Smallest derivative difference this run's FD can resolve, or None.

        From the drift of ``column`` over the last ``FD_DRIFT_WINDOW``
        iterations of the solves the difference used: central
        ``(d+ + d-) / 2h``, forward ``(d+ + d0) / h``, backward
        ``(d- + d0) / h``, d0 being the plain primal before the step. Returns
        0.0 when the step ran at other points but not at ``point`` (OpenMDAO
        did not re-solve it, so its value is exact), and None when the step
        or a drift is unknown.
        """
        signs = FD_SIGNS.get(form)
        if signs is None or not step:
            return None
        ran = {key for key in self.fd_drift if key[:2] == (group, index)}
        if not ran:
            return None
        if not any(key[3] == point for key in ran):
            return 0.0

        def drift(sign, base=False):
            entry = self.fd_drift.get((group, index, sign, point))
            if entry is None:
                return None
            record = entry[1] if base else entry[0]
            return None if record is None else record.get(column)

        if form == 'central':
            parts, divisor = [drift(1), drift(-1)], 2.0 * step
        else:
            sign = signs[0]
            parts, divisor = [drift(sign), drift(sign, base=True)], step
        if any(part is None for part in parts):
            return None
        return float(sum(parts) / divisor)

    def line(self, token, elapsed):
        kind = token['kind']
        n = self.done[kind]
        total = self.plan.totals[kind]
        width = self.plan.count_width
        if total is None:
            count = f'{n:>{width}}'
        elif n > total:
            count = f'{total}/{total} +{n - total} unplanned'
        else:
            count = f'{n:>{width}}/{total:<{width}}'
        error = token.get('error')
        text = self._columns(kind, count, 'ok' if error is None else 'FAILED',
                             token['point'], token['label'],
                             token.get('convergence'), elapsed)
        if error is not None:
            text += f'   {_reason(error)}'
            return text
        left = self.remaining()
        if left is not None:
            text += f'   left {_duration(left)}'
        return text

    def remaining(self):
        """Seconds left from each kind's mean solve time; ``None`` if open-ended."""
        totals = self.plan.totals
        if totals['primal'] is None:
            return None
        every = self.durations['primal'] + self.durations['adjoint']
        if not every:
            return None
        overall = sum(every) / len(every)
        left = 0.0
        for kind in ('primal', 'adjoint'):
            todo = max((totals[kind] or 0) - self.done[kind], 0)
            seen = self.durations[kind]
            left += todo * (sum(seen) / len(seen) if seen else overall)
        return left
