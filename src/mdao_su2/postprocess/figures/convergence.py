"""Convergence and history figures over N labelled datasets.

``entries`` are ``[(label, data)]``. ``data`` is a ``{column: array}`` dict
for the solver figures (``readers.read_history`` or ``readers.read_su2_stdout``)
and a ``readers.read_hst`` result for the optimizer figures. Colours and line
styles are per entry (the default palette when None).
"""
import numpy as np

from ..extract import ExtractError
from ..series import derive_forces, evaluate
from ..style import TEXT, GRID
from .panels import resolve_styles

__all__ = [
    'residual_history', 'coefficient_history', 'history_columns',
    'adjoint_residuals', 'optimizer_history', 'constraint_history',
]

#: Line styles that tell several columns of one entry apart.
_DASHES = ('-', '--', ':', '-.')


def _finish(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel, fontsize=9, color=TEXT)
    ax.set_ylabel(ylabel, fontsize=9, color=TEXT)
    ax.grid(True, alpha=0.25, color=GRID)
    ax.tick_params(labelsize=8, colors=TEXT)


def _x_of(data, x):
    """The x values of a history: the named column, else the row index."""
    if x is None:
        n = len(next(iter(data.values())))
        return np.arange(n)
    return evaluate(x, derive_forces(data))


def residual_history(ax, entries, colors=None, styles=None, x=None):
    """Solver RMS residuals (SU2 writes log10 values) against iteration, every ``rms[...]`` column.

    One colour per entry and one dash pattern per column; with one entry, one
    colour per column.
    """
    colors, styles = resolve_styles(len(entries), colors, styles)
    names = list(dict.fromkeys(k for _, d in entries for k in d if k.startswith('rms[')))
    if not names:
        raise ExtractError('No rms[...] columns in the history')
    single = len(entries) == 1
    for i, (label, data) in enumerate(entries):
        xs = _x_of(data, x)
        for j, name in enumerate(names):
            if name in data:
                ax.plot(xs, data[name], lw=1.0,
                        color=f'C{j}' if single else colors[i],
                        ls='-' if single else _DASHES[j % len(_DASHES)],
                        label=name if single else f'{label} {name}')
    _finish(ax, 'Iteration' if x is None else x, 'log10 RMS residual')
    ax.legend(fontsize=7, framealpha=0.9)


def _coefficient_key(data, letter):
    names = (f'C{letter}', f'C_{letter}', f'C{letter.lower()}')
    return next((k for k in data if k in names), None) or next(
        (k for k in data if k.startswith(names[0])), None)


def coefficient_history(ax, entries, colors=None, styles=None, x=None):
    """``C_D`` (counts, solid) and ``C_L`` (right axis, dashed) against iteration, N entries."""
    colors, styles = resolve_styles(len(entries), colors, styles)
    ax2 = None
    cl_name = 'C_L'
    for i, (label, data) in enumerate(entries):
        xs = _x_of(data, x)
        cd_key, cl_key = _coefficient_key(data, 'D'), _coefficient_key(data, 'L')
        if cd_key:
            ax.plot(xs, data[cd_key] * 1e4, color=colors[i], ls=styles[i], lw=1.2,
                    label=f'{label} {cd_key} (counts)')
        if cl_key:
            ax2 = ax2 or ax.twinx()
            cl_name = cl_key
            ax2.plot(xs, data[cl_key], color=colors[i], ls='--', lw=1.0,
                     label=f'{label} {cl_key}')
    if ax2 is not None:
        ax2.set_ylabel(cl_name, fontsize=9, color=TEXT)
        ax2.tick_params(labelsize=8, colors=TEXT)
    _finish(ax, 'Iteration' if x is None else x, '$C_D$ (counts)')
    handles, labels = ax.get_legend_handles_labels()
    if ax2 is not None:
        h2, l2 = ax2.get_legend_handles_labels()
        handles, labels = handles + h2, labels + l2
    if len(labels) > 1:
        ax.legend(handles, labels, fontsize=6, framealpha=0.9)


def history_columns(ax, entries, exprs, colors=None, styles=None, x=None, ylabel=None,
                    log_y=False):
    """Expressions of the history columns against ``x`` (default the row index), N entries.

    ``exprs`` use the ``series.evaluate`` syntax (``rms[Rho]`` is ``rms_Rho``)
    and may name the derived ``CN`` and ``CL`` (``series.derive_forces``).
    One colour per entry and one dash pattern per expression.
    """
    colors, styles = resolve_styles(len(entries), colors, styles)
    for i, (label, data) in enumerate(entries):
        xs = _x_of(data, x)
        data = derive_forces(data)
        for j, expr in enumerate(exprs):
            ys = evaluate(expr, data) * np.ones(len(xs))
            ax.plot(xs, ys, color=colors[i], ls=_DASHES[j % len(_DASHES)] if len(exprs) > 1
                    else styles[i], lw=1.1,
                    label=f'{label} {expr}' if len(exprs) > 1 else label)
    if log_y:
        ax.set_yscale('log')
    _finish(ax, 'Iteration' if x is None else x, ylabel or ', '.join(exprs))
    if sum(len(l) > 0 for l, _ in entries) or len(exprs) > 1:
        ax.legend(fontsize=7, framealpha=0.9)


def adjoint_residuals(ax, entries, column='rms[A_Rho]', floor=None, colors=None):
    """The final value of an adjoint residual column of each adjoint history.

    ``entries`` are ``[(label, history)]`` of adjoint solves, drawn as one
    marker each against the entry number; ``floor`` (the adjoint
    ``CONV_RESIDUAL_MINVAL``) is a dashed line.
    """
    colors, _ = resolve_styles(len(entries), colors, None)
    for i, (label, data) in enumerate(entries):
        if column not in data:
            raise ExtractError(f'{label}: no {column} column; available {list(data)}')
        ax.plot([i + 1], [data[column][-1]], ls='none', marker='o', ms=5, color=colors[i],
                label=label)
    if floor is not None:
        ax.axhline(floor, color='black', ls='--', lw=0.9, label=f'residual floor {floor:g}')
    ax.set_xticks(range(1, len(entries) + 1))
    _finish(ax, 'adjoint solve', f'final {column}')
    ax.legend(fontsize=6, frameon=False, loc='center left', bbox_to_anchor=(1.01, 0.5))


def optimizer_history(ax, entries, colors=None, styles=None):
    """Every objective of each ``.hst`` against major iteration, N entries."""
    colors, styles = resolve_styles(len(entries), colors, styles)
    for i, (label, hst) in enumerate(entries):
        names = list(hst['objectives'])
        for j, name in enumerate(names):
            ax.plot(hst['iterations'], hst['objectives'][name], color=colors[i],
                    ls=styles[i], lw=1.2, marker='o', markersize=2.5, markeredgewidth=0,
                    label=label if len(names) == 1 else f'{label} {name}')
    _finish(ax, 'Major iteration', 'Objective')
    ax.legend(fontsize=7, framealpha=0.9)


def constraint_history(axes, entries, constraints, colors=None, styles=None):
    """Constraint values against major iteration, one axes per constraint, N entries.

    ``constraints`` is ``{'C_L': {'target': 0.824, 'kind': 'eq'}, ...}``; a
    key matches the first ``.hst`` constraint whose name contains it, and
    ``target`` (when not None) is a dashed (``eq``) or dotted line.
    """
    colors, styles = resolve_styles(len(entries), colors, styles)
    for ax, (name, spec) in zip(axes, constraints.items()):
        for i, (label, hst) in enumerate(entries):
            hit = next(((k, v) for k, v in hst['constraints'].items() if name in k), None)
            if hit is None:
                ax.text(0.5, 0.5 - 0.1 * i, f'{label}: {name} not in history',
                        transform=ax.transAxes, ha='center', fontsize=8, color=TEXT)
                continue
            key, values = hit
            values = np.asarray(values)
            if values.ndim > 1:
                values = values[:, 0]
            ax.plot(hst['iterations'][:len(values)], values, color=colors[i], ls=styles[i],
                    lw=1.2, marker='o', markersize=2.5, markeredgewidth=0,
                    label=f'{label} {key}')
        target = spec.get('target')
        if target is not None:
            ax.axhline(target, color='black', ls='--' if spec.get('kind', 'eq') == 'eq' else ':',
                       lw=0.9, label=f'target = {target}')
        _finish(ax, 'Major iteration', name)
        ax.ticklabel_format(useOffset=False)
        ax.legend(fontsize=7, framealpha=0.9)
