"""Shared drawing helpers of the multi-run comparison figures."""
from ..style import GRID, TEXT, series_styles

__all__ = ['style_axes', 'legend_below', 'series_kwargs', 'resolve_styles']


def style_axes(ax, xlabel=None, ylabel=None, title=None):
    """Grid, tick size and optional labels in the comparison style."""
    ax.grid(True, alpha=0.25, color=GRID)
    ax.tick_params(labelsize=7, colors=TEXT)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=8, color=TEXT)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8, color=TEXT)
    if title:
        ax.set_title(title, fontsize=8, color=TEXT)


def series_kwargs(colors, styles, index, marker='o', lw=1.3):
    """Line keywords of the ``index``-th entry."""
    return {'color': colors[index], 'ls': styles[index], 'lw': lw,
            'marker': marker, 'ms': 3.5, 'mec': 'white', 'mew': 0.4}


def legend_below(fig, handles, labels, ncol=None):
    """One legend under the whole figure (saved with ``bbox_inches='tight'``)."""
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.0),
               ncol=ncol or min(len(labels), 4), fontsize=7, frameon=False)


def resolve_styles(count, colors=None, styles=None):
    """``(colors, styles)`` for ``count`` entries: the given ones, else the default palette."""
    default_colors, default_styles = series_styles(count)
    return colors or default_colors, styles or default_styles
