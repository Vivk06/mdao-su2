"""Standalone legend and colour-bar images shared by the comparison figures."""
from ..style import TEXT, series_styles

__all__ = ['save_legend', 'save_colorbar']


def save_legend(labels, path, dpi=100, figsize=(4.0, 0.4), colors=None):
    """Save a legend image of ``labels`` in entry order."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_axis_off()
    colors = colors or series_styles(len(labels))[0]
    handles = [mpatches.Patch(color=colors[i], label=label) for i, label in enumerate(labels)]
    ax.legend(handles=handles, loc='center', ncol=len(labels), fontsize=7, frameon=False)
    fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white', pad_inches=0.02)
    plt.close(fig)


def save_colorbar(path, vmin, vmax, label, cmap='coolwarm', dpi=100, figsize=(4.0, 0.35)):
    """Save a horizontal colour bar image for a ``vmin`` to ``vmax`` scale named ``label``."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize

    fig, ax = plt.subplots(figsize=figsize)
    sm = cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cb = fig.colorbar(sm, cax=ax, orientation='horizontal')
    cb.set_label(label, fontsize=7, color=TEXT)
    cb.ax.tick_params(labelsize=6)
    fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white', pad_inches=0.02)
    plt.close(fig)
