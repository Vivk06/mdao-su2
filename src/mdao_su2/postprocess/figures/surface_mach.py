"""Isentropic surface Mach number on the upper surface of a 2D wall."""

from ..extract import ExtractError, isentropic_mach, split_wall_loop, surface_field
from .panels import resolve_styles, style_axes

__all__ = ['upper_surface_mach', 'surface_mach']


def upper_surface_mach(surface, mach, gamma, field='Pressure_Coefficient'):
    """``(x/c, M_is)`` along the upper surface, from the wall's pressure coefficient."""
    pts, cp = surface_field(surface, field)
    cells = [c for c in surface.get('cells', ()) if len(c) == 2]
    if not cells:
        raise ExtractError('The surface has no line cells: cannot find the upper surface')
    upper, _ = split_wall_loop(pts[:, :2], cells)
    x = pts[:, 0]
    return (x[upper] - x.min()) / (x.max() - x.min()), isentropic_mach(cp, mach, gamma)[upper]


def surface_mach(ax, entries, machs, gamma, colors=None, styles=None):
    """Upper-surface isentropic Mach number of N entries; the line at 1 is sonic.

    ``entries`` are ``[(label, surface)]`` and ``machs`` the freestream Mach
    number of each.
    """
    colors, styles = resolve_styles(len(entries), colors, styles)
    for i, ((label, surface), mach) in enumerate(zip(entries, machs)):
        x, m = upper_surface_mach(surface, mach, gamma)
        ax.plot(x, m, color=colors[i], ls=styles[i], lw=1.4, label=label)
    ax.axhline(1.0, color='black', lw=0.8)
    ax.set_xlim(0.0, 1.0)
    style_axes(ax, xlabel='x/c', ylabel='isentropic surface Mach number',
               title='upper surface')
    ax.legend(fontsize=7, frameon=False, loc='best')
