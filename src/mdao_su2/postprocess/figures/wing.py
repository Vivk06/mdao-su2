"""3D wing panel figures on the case's span and lift axes (``[case.reference]``).

Sections are cut through the surface triangles and chained by connectivity,
so a cambered or thin section draws as one ordered loop.
"""
import numpy as np

from ..style import LABEL, TEXT, diverging
from ...vtu import triangles
from .panels import legend_below, resolve_styles, style_axes

__all__ = [
    'surface_mesh', 'split_surfaces', 'section',
    'panel_field_planform', 'panel_delta_planform', 'panel_sections', 'field_sections',
    'panel_root_section', 'surface_3d', 'surface_3d_overlay', 'interactive_surface',
    'VIEW_ELEV', 'VIEW_AZIM', 'AXIS',
]

AXIS = {'x': 0, 'y': 1, 'z': 2}

#: Oblique view of every 3D wing figure (matplotlib degrees).
VIEW_ELEV = 30.0
VIEW_AZIM = -60.0

#: Section colour of the first surface of :func:`surface_3d_overlay`.
OVERLAY_BASE_LINE = '#8a8a8a'


def surface_mesh(surface):
    """``(points (n,3), tris (m,3), point_data)`` of a ``readers.read_surface`` 3D surface."""
    if not len(surface.get('cells', ())):
        raise ValueError('The surface file has no cells: a 3D figure needs a VTU surface')
    return surface['points'].astype(float), triangles(surface), surface['point_data']


def split_surfaces(pts, tris, lift_axis='z'):
    """``(upper_tris, lower_tris)`` by the sign of the facet normal along ``lift_axis``."""
    p0, p1, p2 = pts[tris[:, 0]], pts[tris[:, 1]], pts[tris[:, 2]]
    n = np.cross(p1 - p0, p2 - p0)[:, AXIS[lift_axis]]
    return tris[n > 0], tris[n < 0]


def _chain(segments, tol):
    """Join 2-point segments into the longest connected polyline."""
    keys = lambda p: tuple(np.round(np.asarray(p) / tol).astype(np.int64))
    ends = {}
    for k, (a, b) in enumerate(segments):
        ends.setdefault(keys(a), []).append((k, 0))
        ends.setdefault(keys(b), []).append((k, 1))
    used, best = set(), []
    for start in range(len(segments)):
        if start in used:
            continue
        used.add(start)
        line = [segments[start][0], segments[start][1]]
        for grow_front in (False, True):
            while True:
                tip = line[0] if grow_front else line[-1]
                nxt = next(((k, e) for k, e in ends.get(keys(tip), ()) if k not in used), None)
                if nxt is None:
                    break
                k, e = nxt
                used.add(k)
                other = segments[k][1 - e]
                line.insert(0, other) if grow_front else line.append(other)
        if len(line) > len(best):
            best = line
    return np.array(best)


def section(pts, tris, station, span_axis='y', lift_axis='z', values=None):
    """``(n, 2)`` ordered (x, lift) loop of the wall cut at ``span = station``, or None.

    With nodal ``values`` the loop is ``(n, 3)``: the third column is the
    field interpolated linearly along each cut edge.
    """
    s, h = AXIS[span_axis], AXIS[lift_axis]
    cols = [0, h]
    if values is not None:
        pts = np.column_stack([pts, np.asarray(values, float)])
        cols = [0, h, 3]
    segments = []
    for f in tris:
        hits = []
        for a in range(3):
            i, j = f[a], f[(a + 1) % 3]
            da, db = pts[i, s] - station, pts[j, s] - station
            if da == 0.0:
                hits.append(pts[i])
            if da * db < 0.0:
                hits.append(pts[i] + da / (da - db) * (pts[j] - pts[i]))
        if len(hits) >= 2:
            segments.append((hits[0][cols], hits[1][cols]))
    if not segments:
        return None
    tol = 1e-9 * max(float(np.ptp(pts[:, 0])), 1.0)
    return _chain(segments, tol)


def panel_field_planform(ax, pts, tris, values, vmin, vmax, cmap=None,
                         span_axis='y', **kwargs):
    """Top-down (x, span) map of a nodal field over the given facets."""
    import matplotlib.tri as mtri
    t = mtri.Triangulation(pts[:, 0], pts[:, AXIS[span_axis]], tris)
    col = ax.tripcolor(t, np.asarray(values, float), shading='gouraud',
                       cmap=cmap or diverging(), vmin=vmin, vmax=vmax,
                       rasterized=True, **kwargs)
    ax.set_aspect('equal')
    ax.set_xlabel('x'); ax.set_ylabel(f'{span_axis} (span)')
    return col


def panel_delta_planform(ax, pts, tris, delta, span_axis='y', **kwargs):
    """Top-down (x, span) map of a nodal difference on a colour scale symmetric about 0.

    ``delta`` is the field of a later entry minus that of the first, on the
    first entry's facets; the scale is ``+-max|delta|``.
    """
    delta = np.asarray(delta, float)
    m = float(np.nanmax(np.abs(delta))) or 1.0
    return panel_field_planform(ax, pts, tris, delta, -m, m, span_axis=span_axis, **kwargs)


def panel_sections(ax, entries, stations, colors=None, styles=None, step=0.16,
                   span_axis='y', lift_axis='z'):
    """Sections of every entry at each station, stacked with an offset.

    ``entries`` are ``[(label, surface)]``; each is cut through its own facets.
    """
    colors, styles = resolve_styles(len(entries), colors, styles)
    meshes = [surface_mesh(surface)[:2] for _, surface in entries]
    for n, y in enumerate(stations):
        drawn = []
        for i, (pts, tris) in enumerate(meshes):
            line = section(pts, tris, y, span_axis, lift_axis)
            if line is None:
                continue
            ax.plot(line[:, 0], line[:, 1] - n * step, color=colors[i], ls=styles[i],
                    lw=1.0 if i == 0 else 1.4,
                    label=entries[i][0] if n == 0 else None)
            drawn.append(line)
        if drawn:
            ax.text(drawn[0][:, 0].min() - 0.04, -n * step, f'{span_axis} = {y:g}',
                    ha='right', va='center', fontsize=8, color=LABEL)
    ax.set_aspect('equal')
    ax.axis('off')


def field_sections(fig, entries, stations, values, label, colors=None, styles=None,
                   span_axis='y', lift_axis='z', max_cols=3):
    """A nodal field along the wall section at each station, one panel per station.

    ``entries`` are ``[(label, surface)]`` and ``values`` the matching nodal
    arrays; the y axis is inverted (pressure convention) and shared by the
    panels. The first entry is drawn thinner.
    """
    colors, styles = resolve_styles(len(entries), colors, styles)
    meshes = [surface_mesh(surface)[:2] for _, surface in entries]
    cols = min(len(stations), max_cols)
    rows = -(-len(stations) // cols)
    fig.set_size_inches(3.0 * cols + 0.6, 2.4 * rows + 0.8)
    axes = fig.subplots(rows, cols, squeeze=False, sharey=True).ravel()
    for ax in axes[len(stations):]:
        ax.axis('off')
    for ax, y in zip(axes, stations):
        for i, ((pts, tris), v) in enumerate(zip(meshes, values)):
            loop = section(pts, tris, y, span_axis, lift_axis, values=v)
            if loop is not None:
                ax.plot(loop[:, 0], loop[:, 2], color=colors[i], ls=styles[i],
                        lw=1.0 if i == 0 else 1.4, label=entries[i][0])
        ax.set_title(f'{span_axis} = {y:g}', fontsize=8, color=LABEL)
        style_axes(ax, xlabel='x' if ax in axes[-cols:] else None)
    axes[0].invert_yaxis()
    for ax in axes[::cols]:
        ax.set_ylabel(label, fontsize=8, color=TEXT)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    legend_below(fig, handles, labels)


def panel_root_section(ax, entries, y=0.0, colors=None, styles=None, span_axis='y',
                       lift_axis='z'):
    """Section of every entry at ``span = y`` at true scale.

    Returns the trailing-edge lift shift of the last entry against the first,
    or None with a single entry or a missing section.
    """
    colors, styles = resolve_styles(len(entries), colors, styles)
    lines = []
    for i, (label, surface) in enumerate(entries):
        pts, tris, _ = surface_mesh(surface)
        line = section(pts, tris, y, span_axis, lift_axis)
        lines.append(line)
        if line is not None:
            ax.plot(line[:, 0], line[:, 1], color=colors[i], ls=styles[i],
                    lw=1.2 if i == 0 else 1.6, label=label)
    ax.set_aspect('equal')
    ax.set_xlabel('x'); ax.set_ylabel(lift_axis)
    s0, s1 = lines[0], lines[-1]
    if len(lines) < 2 or s0 is None or s1 is None:
        return None
    ite = int(np.argmax(s0[:, 0]))
    near = np.argsort(np.abs(s1[:, 0] - s0[ite, 0]))[:4]
    return float(s1[near, 1].mean() - s0[ite, 1])


def surface_3d(ax, pts, tris, values, label, vmin=None, vmax=None, lift_axis='z',
               elev=VIEW_ELEV, azim=VIEW_AZIM):
    """Shaded 3D surface on a ``projection='3d'`` axes, lift axis up (isometric view)."""
    v = np.asarray(values, float)
    vmin = float(np.nanmin(v)) if vmin is None else vmin
    vmax = float(np.nanmax(v)) if vmax is None else vmax
    h = AXIS[lift_axis]
    order = [a for a in range(3) if a != h] + [h]
    P = pts[:, order]
    col = ax.plot_trisurf(P[:, 0], P[:, 1], P[:, 2], triangles=tris, cmap=diverging(),
                          linewidth=0, antialiased=False, shade=False)
    col.set_array(v[tris].mean(1))
    col.set_clim(vmin, vmax)
    ax.figure.colorbar(col, ax=ax, shrink=0.6, label=label)
    span = P.max(0) - P.min(0)
    ax.set_box_aspect(tuple(span / span.max()))
    names = 'xyz'
    ax.set_xlabel(names[order[0]]); ax.set_ylabel(names[order[1]])
    ax.set_zlabel(names[order[2]])
    ax.set_zticks([])  # the wing is thin: lift-axis tick labels would overprint
    ax.view_init(elev=elev, azim=azim)
    return col


def surface_3d_overlay(ax, surfaces, colors, span_axis='y', lift_axis='z', n_sections=9,
                       elev=VIEW_ELEV, azim=VIEW_AZIM):
    """Several 3D wings on one ``projection='3d'`` axes, with a legend and no field colouring.

    ``surfaces`` are ``[(label, pts, tris)]``. Each is drawn as ``n_sections``
    wall sections at evenly spaced span stations (2 to 98 % of the first
    surface's span), so that the wings stay visible through each other; the
    first surface's sections are grey and the others take ``colors[i]``. The view is that of
    :func:`surface_3d` and the box follows the joint extent, so the wing fills
    the frame.
    """
    from matplotlib.lines import Line2D
    h, s = AXIS[lift_axis], AXIS[span_axis]
    order = [a for a in range(3) if a != s and a != h] + [s, h]
    allp = np.vstack([pts[:, order] for _, pts, _ in surfaces])
    lo, hi = float(surfaces[0][1][:, s].min()), float(surfaces[0][1][:, s].max())
    stations = lo + np.linspace(0.02, 0.98, n_sections) * (hi - lo)
    handles = []
    for i, (label, pts, tris) in enumerate(surfaces):
        color = OVERLAY_BASE_LINE if i == 0 else colors[i]
        for y in stations:
            loop = section(pts, tris, y, span_axis, lift_axis)
            if loop is not None:
                ax.plot(loop[:, 0], np.full(len(loop), y), loop[:, 1], color=color, lw=1.4)
        handles.append(Line2D([], [], color=color, lw=1.6, label=label))
    span = allp.max(0) - allp.min(0)
    ax.set_box_aspect(tuple(span / span.max()), zoom=1.45)
    names = 'xyz'
    ax.set_xlabel(names[order[0]]); ax.set_ylabel(names[order[1]], labelpad=10)
    ax.set_zlabel(names[order[2]])
    ax.set_zticks([])
    ax.view_init(elev=elev, azim=azim)
    ax.legend(handles=handles, fontsize=8, loc='upper left', framealpha=0.9)


def interactive_surface(pts, tris, values, label, vmin=None, vmax=None, lift_axis='z'):
    """Open a rotatable 3D window (needs an interactive matplotlib backend)."""
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection='3d')
    surface_3d(ax, pts, tris, values, label, vmin, vmax, lift_axis)
    ax.set_title(f'{label} -- drag to rotate, scroll to zoom')
    plt.show()
    return fig
