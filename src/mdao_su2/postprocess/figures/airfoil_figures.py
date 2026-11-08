"""2D airfoil figures. Each draws N labelled surfaces or one volume onto a caller-supplied Axes.

``entries`` are ``[(label, surface)]`` with ``surface`` a
``readers.read_surface`` result; colours and line styles are per entry (the
default palette when None).
"""
import numpy as np

from ..extract import ExtractError, surface_field, surface_ndim, wall_series
from ...vtu import triangles
from ..style import GRID, TEXT
from .panels import resolve_styles

__all__ = ['cp_chord', 'shape_overlay', 'field_contour', 'field_surface']


def _check_2d(surface):
    if surface_ndim(surface) != 2:
        raise ExtractError('This figure needs a 2D wall (line cells); use the wing figures '
                           'for a 3D surface')


def _decorate(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel, fontsize=9, color=TEXT)
    ax.set_ylabel(ylabel, fontsize=9, color=TEXT)
    ax.grid(True, alpha=0.25, color=GRID)
    ax.tick_params(labelsize=8, colors=TEXT)


def field_surface(ax, entries, colors=None, styles=None, field='Pressure_Coefficient',
                  ylabel=None):
    """A wall field against x/c, N entries; upper solid, lower dashed and lighter."""
    colors, styles = resolve_styles(len(entries), colors, styles)
    for i, (label, surface) in enumerate(entries):
        _check_2d(surface)
        _, values = surface_field(surface, field)
        xu, vu, xl, vl = wall_series(surface, values)
        lw = 1.6 if i == 0 else 1.2
        ax.plot(xu, vu, color=colors[i], ls=styles[i], lw=lw, label=label)
        ax.plot(xl, vl, color=colors[i], ls='--', lw=0.7 * lw, alpha=0.7)
    _decorate(ax, 'x/c', ylabel or field)
    if len(entries) > 1 or entries[0][0]:
        ax.legend(fontsize=7, frameon=False, loc='best')


def cp_chord(ax, entries, colors=None, styles=None, field='Pressure_Coefficient',
             references=(), reference_colors=None):
    """Pressure coefficient against x/c, y inverted, N entries.

    ``references`` are ``[(label, reference)]`` with ``reference`` a
    ``readers.read_reference_cp`` result, drawn as markers after the lines in
    ``reference_colors`` (default the palette after the entries): filled for
    ``upper`` and ``leading_edge`` taps, open for ``lower`` and
    ``trailing_edge`` taps, one style when the file has no surface column.
    """
    field_surface(ax, entries, colors, styles, field=field, ylabel='$C_p$')
    ax.invert_yaxis()
    if not references:
        return
    if reference_colors is None:
        reference_colors = resolve_styles(len(entries) + len(references))[0][len(entries):]
    for (label, ref), color in zip(references, reference_colors):
        x, cp = np.asarray(ref['x']), np.asarray(ref['cp'])
        if ref['surface'] is None:
            groups = [(np.ones(len(x), dtype=bool), True, label)]
        else:
            surface = np.asarray(ref['surface'])
            filled = np.isin(surface, ('upper', 'leading_edge'))
            groups = [(filled, True, label), (~filled, False, None)]
        for mask, solid, name in groups:
            if mask.any():
                ax.plot(x[mask], cp[mask], ls='none', marker='o', ms=4.5, mew=1.0,
                        mec=color, mfc=color if solid else 'none', label=name)
    ax.legend(fontsize=7, frameon=False, loc='best')


def shape_overlay(ax, entries, colors=None, styles=None):
    """The wall outline of every entry, equal aspect, in the wall's length unit."""
    colors, styles = resolve_styles(len(entries), colors, styles)
    for i, (label, surface) in enumerate(entries):
        _check_2d(surface)
        pts = np.asarray(surface['points'], dtype=float)
        x, y = pts[:, 0], pts[:, 1]
        order = np.argsort(np.arctan2(y - y.mean(), x - x.mean()))
        ax.plot(x[order], y[order], color=colors[i], ls=styles[i],
                lw=1.6 if i == 0 else 1.2, label=label)
    ax.set_aspect('equal')
    _decorate(ax, 'x/c', 'y/c')
    ax.legend(fontsize=7, frameon=False, loc='best')


def field_range(volumes, field):
    """``(vmin, vmax)`` of a volume point-data field over ``volumes``, to share one scale."""
    lo, hi = np.inf, -np.inf
    for volume in volumes:
        values = _scalar(volume, field)
        lo, hi = min(lo, float(values.min())), max(hi, float(values.max()))
    return lo, hi


def _scalar(volume, field):
    if field not in volume['point_data']:
        raise ExtractError(f'Field {field!r} not in the volume file. '
                           f'Available: {list(volume["point_data"])}')
    values = volume['point_data'][field].astype(float)
    return np.linalg.norm(values, axis=1) if values.ndim > 1 else values


def field_contour(ax, volume, wall=None, field='Mach', margin=0.3, levels=28, length=1.0,
                  vmin=None, vmax=None, cmap='coolwarm', title=None):
    """A 2D volume point-data field as a filled contour; returns the contour set.

    ``volume`` is a ``readers.read_volume`` result and ``wall`` an optional
    surface of the same case that sets the window to its bounding box plus
    ``margin`` chords (without it the window is the whole domain). Coordinates
    are divided by ``length`` (``[case.reference] length``). The line of unit
    Mach is drawn for ``field='Mach'``; ``vmin``/``vmax`` share one scale
    across volumes.
    """
    import matplotlib.tri as mtri
    from matplotlib.colors import Normalize

    pts = volume['points'].astype(float) / length
    values = _scalar(volume, field)
    tri = mtri.Triangulation(pts[:, 0], pts[:, 1], triangles(volume))
    vmin = float(values.min()) if vmin is None else vmin
    vmax = float(values.max()) if vmax is None else vmax
    cf = ax.tricontourf(tri, values, levels=levels, norm=Normalize(vmin=vmin, vmax=vmax),
                        cmap=cmap)
    if field == 'Mach' and vmin < 1.0 < vmax:
        ax.tricontour(tri, values, levels=[1.0], colors='k', linewidths=0.8,
                      linestyles='--')
    if wall is not None:
        sx = wall['points'][:, 0].astype(float) / length
        sy = wall['points'][:, 1].astype(float) / length
        chord = sx.max() - sx.min()
        xpad = margin * chord
        ypad = margin * max(chord, sy.max() - sy.min())
        ax.set_xlim(sx.min() - xpad, sx.max() + xpad)
        ax.set_ylim(sy.min() - ypad, sy.max() + ypad)
    ax.set_aspect('equal')
    if title:
        ax.set_title(title, fontsize=8, color=TEXT)
    _decorate(ax, 'x/c', 'y/c')
    return cf
