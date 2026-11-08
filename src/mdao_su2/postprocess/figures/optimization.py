"""Optimization comparison figures: change by iteration, constraint margins, DV profiles, wall displacement.

``entries`` are ``[(label, data)]``; colours and line styles are per entry.
"""
import numpy as np

from ..extract import ExtractError, delta_y
from ..series import evaluate, linear_row_margin
from ..style import series_styles
from .panels import legend_below, resolve_styles, series_kwargs, style_axes

__all__ = ['iteration_change', 'linear_margins', 'dv_profile', 'dv_history',
           'delta_y_figure']


def _grid(fig, count, width=4.2, height=3.0, max_cols=3):
    """``count`` panels in rows of at most ``max_cols``; returns the used axes."""
    cols = min(count, 2 if count == 4 else max_cols)
    rows = -(-count // cols)
    fig.set_size_inches(width * cols, height * rows)
    axes = fig.subplots(rows, cols, squeeze=False).ravel()
    for ax in axes[count:]:
        ax.axis('off')
    return axes[:count]


def _finish(fig, axes):
    handles, labels = axes[0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    if len(labels) > 1:
        legend_below(fig, handles, labels)


def _iterations(data):
    """The x values of an iteration table: its ``iteration`` column, else the row index."""
    if 'iteration' in data:
        return np.asarray(data['iteration'], dtype=float)
    return np.arange(len(next(iter(data.values()))), dtype=float)


def iteration_change(fig, entries, exprs, colors=None, styles=None, percent=True,
                     y_labels=None):
    """Change of each expression from its first row, one panel per expression, N entries.

    ``entries`` are ``[(label, columns)]`` with ``columns`` a ``{name: array}``
    iteration table (one value per iteration; an ``iteration`` column is the
    x axis, else the row index). The change is a percentage of the first value
    when ``percent`` else the difference. ``y_labels`` maps an expression to
    its y axis label (default ``change from iteration 0``, with ``(%)`` for
    a percentage).
    """
    colors, styles = resolve_styles(len(entries), colors, styles)
    axes = _grid(fig, len(exprs))
    for index, (label, columns) in enumerate(entries):
        kw = series_kwargs(colors, styles, index)
        x = _iterations(columns)
        for ax, expr in zip(axes, exprs):
            y = evaluate(expr, columns) * np.ones(len(x))
            ax.plot(x, 100.0 * (y / y[0] - 1.0) if percent else y - y[0], label=label, **kw)
    for ax, expr in zip(axes, exprs):
        ax.axhline(0.0, color='black', lw=0.8)
        style_axes(ax, xlabel='iteration', title=expr,
                   ylabel=(y_labels or {}).get(expr) or (
                       'change from iteration 0 (%)' if percent
                       else 'change from iteration 0'))
    _finish(fig, axes)


def linear_margins(fig, entries, rows, colors=None, styles=None):
    """Margin of each linear constraint row by iteration (%), one panel per row, N entries.

    ``entries`` are ``[(label, columns)]`` as for :func:`iteration_change`.
    ``rows`` are dicts with ``name``, ``terms`` (``(column, coeff)``),
    ``offset`` and the bounds ``lower``, ``upper``, ``equals``. The margin is
    the distance to the bound over the magnitude of the terms working against
    it (``series.linear_row_margin``); zero is the bound and positive is met.
    """
    colors, styles = resolve_styles(len(entries), colors, styles)
    axes = _grid(fig, len(rows), width=4.6)
    for ax, row in zip(axes, rows):
        for index, (label, columns) in enumerate(entries):
            margin = linear_row_margin(columns, row['terms'], row.get('offset', 0.0),
                                       row.get('lower'), row.get('upper'), row.get('equals'))
            ax.plot(_iterations(columns), margin, label=label,
                    **series_kwargs(colors, styles, index))
        ax.axhline(0.0, color='black', lw=1.0)
        style_axes(ax, xlabel='iteration', title=row['name'],
                   ylabel='margin to the bound (%)')
    _finish(fig, axes)


def dv_profile(fig, entries, crest, colors=None, styles=None, lower=None, upper=None,
               unit=1e-3, groups=('upper surface', 'lower surface'), unit_label=r'$10^{-3}\,c$'):
    """Design-variable values against bump crest, one panel per group, N entries.

    ``entries`` are ``[(label, vector)]``; each vector holds ``len(groups)``
    consecutive blocks of ``len(crest)`` values (upper surface then lower).
    Values and the dotted ``lower``/``upper`` bounds are divided by ``unit``.
    """
    colors, styles = resolve_styles(len(entries), colors, styles)
    n = len(crest)
    axes = fig.subplots(1, len(groups), sharey=True, squeeze=False)[0]
    fig.set_size_inches(4.5 * len(groups), 3.4)
    for index, (label, vector) in enumerate(entries):
        vector = np.asarray(vector, dtype=float).ravel()
        if vector.size != n * len(groups):
            raise ExtractError(f'{label}: {vector.size} design variables, expected '
                               f'{len(groups)} x {n}')
        kw = series_kwargs(colors, styles, index)
        for g, ax in enumerate(axes):
            ax.plot(crest, vector[g * n:(g + 1) * n] / unit, label=label, **kw)
    for ax, name in zip(axes, groups):
        for bound in (lower, upper):
            if bound is not None:
                ax.axhline(bound / unit, color='black', ls=':', lw=1.0)
        ax.axhline(0.0, color='0.5', lw=0.6)
        style_axes(ax, xlabel='bump crest x/c', title=name)
    axes[0].set_ylabel(f'amplitude ({unit_label}); dotted: bounds', fontsize=8)
    _finish(fig, axes)


def delta_y_figure(fig, entries, colors=None, styles=None):
    """Wall displacement of every entry against the first (the base), by x/c.

    ``entries`` are ``[(label, surface)]`` with 2D walls; draws the upper and
    lower surface ``dy``, the thickness change and the camber-line change of
    entries 1 onward (``extract.delta_y``).
    """
    if len(entries) < 2:
        raise ExtractError('delta_y needs a base surface and at least one more')
    colors, styles = resolve_styles(len(entries), colors, styles)
    axes = fig.subplots(2, 2, sharex=True).ravel()
    fig.set_size_inches(9.0, 5.8)
    titles = (r'upper surface $\Delta y$', r'lower surface $\Delta y$',
              r'thickness change $\Delta y_u-\Delta y_l$',
              r'camber-line change $(\Delta y_u+\Delta y_l)/2$')
    base = entries[0][1]
    for index in range(1, len(entries)):
        label, surface = entries[index]
        profile = delta_y(base, surface)
        for ax, key in zip(axes, ('upper', 'lower', 'thickness', 'camber')):
            ax.plot(profile['x'], profile[key] * 1e3, color=colors[index],
                    ls=styles[index], lw=1.4, label=f'{label} - {entries[0][0]}')
    for ax, title in zip(axes, titles):
        ax.axhline(0.0, color='black', lw=0.8)
        style_axes(ax, title=title)
    for ax in axes[2:]:
        ax.set_xlabel('x/c', fontsize=8)
    for ax in axes[::2]:
        ax.set_ylabel(r'$10^{-3}\,c$', fontsize=8)
    _finish(fig, axes)


def _dv_columns(hst):
    """``[(name, values per major iteration)]``: one per design-variable element, ``name[i]``."""
    columns = []
    for name, table in hst['dvs'].items():
        table = np.asarray(table, dtype=float)
        if table.ndim == 1:
            columns.append((name, table))
        else:
            columns += [(f'{name}[{j}]', table[:, j]) for j in range(table.shape[1])]
    return columns


def dv_history(ax, entries, lower=None, upper=None):
    """Every design-variable element against major iteration, one line each, N entries.

    ``entries`` are ``[(label, hst)]`` (``readers.read_hst``); a vector design
    variable draws one line per element, named ``name[i]``. Colour is per
    element; the entries are solid, dashed, dash-dotted and dotted in turn
    (more than four raise). ``lower`` and ``upper`` (when not None) are dotted
    lines.
    """
    dashes = ['-', '--', '-.', ':']
    if len(entries) > len(dashes):
        raise ValueError(f'dv_history draws at most {len(dashes)} entries')
    colors, _ = series_styles(len(_dv_columns(entries[0][1])))
    for i, (label, hst) in enumerate(entries):
        for j, (name, values) in enumerate(_dv_columns(hst)):
            ax.plot(hst['iterations'], values, color=colors[j], ls=dashes[i], lw=1.2,
                    marker='o', ms=2.5, mew=0,
                    label=name if len(entries) == 1 else f'{label} {name}')
    for bound in (lower, upper):
        if bound is not None:
            ax.axhline(bound, color='black', ls=':', lw=0.9)
    style_axes(ax, xlabel='Major iteration', ylabel='Design variable value')
    ax.legend(fontsize=7, frameon=False, loc='center left', bbox_to_anchor=(1.01, 0.5))
