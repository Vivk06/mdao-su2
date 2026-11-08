"""Sweep figure: any y expressions against an x expression, one panel per expression."""
from matplotlib.lines import Line2D

from ..series import fit_peak
from .panels import legend_below, resolve_styles, series_kwargs, style_axes

__all__ = ['sweep']


def sweep(fig, entries, colors=None, styles=None, x_label='x', y_labels=None, peak=(),
          log_y=(), ncols=None, reference=None):
    """One panel per y expression, N series overlaid.

    ``entries`` are ``[(label, series)]`` with ``series`` the
    ``series.build_series`` result (``x`` and ``y`` = ``{expression: array}``);
    every series holds the same y expressions, in the first one's order.
    ``y_labels`` maps an expression to its axis label (default the expression).
    A star marks the maximum of each expression in ``peak`` and an open
    triangle the vertex of the parabola through the maximum and its two
    neighbours; expressions in ``log_y`` get a log axis. ``reference`` is one
    bool per entry: a True entry is drawn as open circle markers only, in its
    entry colour, and gets no peak markers.
    """
    colors, styles = resolve_styles(len(entries), colors, styles)
    reference = reference or [False] * len(entries)
    exprs = list(entries[0][1]['y'])
    ncols = ncols or len(exprs)
    nrows = -(-len(exprs) // ncols)
    axes = fig.subplots(nrows, ncols, squeeze=False).ravel()
    for ax in axes[len(exprs):]:
        ax.axis('off')
    fig.set_size_inches(3.7 * ncols, 3.4 * nrows)
    for index, (label, data) in enumerate(entries):
        kw = series_kwargs(colors, styles, index)
        if reference[index]:
            kw = {'color': colors[index], 'ls': 'none', 'marker': 'o', 'ms': 5.5, 'mfc': 'none',
                  'mew': 1.2}
        for ax, expr in zip(axes, exprs):
            ax.plot(data['x'], data['y'][expr], label=label, **kw)
        for ax, expr in zip(axes, exprs):
            if expr not in peak or reference[index]:
                continue
            x_max, y_max, x_vertex, y_vertex = fit_peak(data['x'], data['y'][expr])
            ax.plot([x_max], [y_max], ls='none', marker='*', ms=11, color=colors[index],
                    mec='black', mew=0.5)
            if x_vertex is not None:
                ax.plot([x_vertex], [y_vertex], ls='none', marker='v', ms=8, mfc='white',
                        mec=colors[index], mew=1.4)
    labels_of = y_labels or {}
    for ax, expr in zip(axes, exprs):
        if expr in log_y:
            ax.set_yscale('log')
        style_axes(ax, xlabel=x_label, ylabel=labels_of.get(expr, expr))
    handles, labels = axes[0].get_legend_handles_labels()
    if peak:
        handles += [Line2D([], [], ls='none', marker='*', ms=10, color='0.4', mec='black'),
                    Line2D([], [], ls='none', marker='v', ms=7, mfc='white', mec='0.4',
                           mew=1.4)]
        labels += ['maximum', 'parabola vertex']
    fig.tight_layout()
    legend_below(fig, handles, labels, ncol=4)
