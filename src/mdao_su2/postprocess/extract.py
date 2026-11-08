"""Derived wall data from reader outputs: arrays in, arrays out. No plotting imports, headless."""
import csv
import os

import numpy as np

__all__ = [
    'surface_field', 'surface_ndim', 'wall_split', 'chord_series', 'split_wall_loop',
    'wall_series', 'isentropic_mach', 'delta_y', 'write_table', 'ExtractError',
]


class ExtractError(Exception):
    """Raised when extraction fails."""


def surface_field(surface, name):
    """A named point-data array of a :func:`readers.read_surface` result: ``(pts, values)``."""
    available = list(surface['point_data'])
    if name not in surface['point_data']:
        raise ExtractError(f'Field {name!r} not in the surface file. '
                           f'Available: {", ".join(available)}')
    return surface['points'].astype(float), surface['point_data'][name].astype(float)


def surface_ndim(surface):
    """2 for a surface of line cells (an airfoil wall), else 3.

    A surface without cells (a SU2 ``.csv``) is 2D when its points have two
    columns or no extent along z.
    """
    types = surface.get('cell_types')
    if types is not None and len(types):
        return 2 if bool(np.all(np.asarray(types) == 3)) else 3
    pts = surface['points']
    return 2 if pts.shape[1] < 3 or float(np.ptp(pts[:, 2])) == 0.0 else 3


def wall_split(pts, ndim, cells=None):
    """Split wall nodes into (upper_idx, lower_idx). 2D: chord cross-product; 3D: facet-normal z-sign."""
    if ndim == 2:
        return _wall_split_2d(pts)
    elif ndim == 3:
        if cells is None:
            raise ExtractError('wall_split(ndim=3) requires cells')
        return _wall_split_3d(pts, cells)
    else:
        raise ExtractError(f'ndim must be 2 or 3, got {ndim}')


def _wall_split_2d(pts):
    """2D split: chord-line cross product, matching the legacy implementation."""
    x, y = pts[:, 0], pts[:, 1]
    ile = int(np.argmin(x))
    ite = int(np.argmax(x))
    dx = x[ite] - x[ile]
    dy = y[ite] - y[ile]
    cross = dx * (y - y[ile]) - dy * (x - x[ile])
    ui = np.where(cross > 0)[0]
    li = np.where(cross < 0)[0]
    ui = ui[np.argsort(x[ui])]
    li = li[np.argsort(x[li])]
    ui = np.concatenate([[ile], ui, [ite]])
    li = np.concatenate([[ile], li, [ite]])
    return ui, li


def _wall_split_3d(pts, cells):
    """3D split: sign of facet-normal z-component, matching surface_projection.split_surfaces."""
    tris = []
    for c in cells:
        c = np.asarray(c)
        for t in range(1, len(c) - 1):
            tris.append((c[0], c[t], c[t + 1]))
    tris = np.asarray(tris, dtype=int)
    if tris.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    p0, p1, p2 = pts[tris[:, 0]], pts[tris[:, 1]], pts[tris[:, 2]]
    nz = np.cross(p1 - p0, p2 - p0)[:, 2]
    upper_nodes = set(tris[nz > 0].ravel())
    lower_nodes = set(tris[nz < 0].ravel())
    return np.sort(list(upper_nodes)), np.sort(list(lower_nodes))


def chord_series(pts, values, ndim):
    """Normalize x by measured chord and split upper/lower. Returns (x_c_u, v_u, x_c_l, v_l)."""
    x = pts[:, 0].astype(float)
    x_min, x_max = x.min(), x.max()
    chord = x_max - x_min
    if chord < 1e-15:
        raise ExtractError('Degenerate chord (x_max ≈ x_min)')
    x_c = (x - x_min) / chord

    ui, li = _wall_split_2d(pts)
    vals = np.asarray(values, dtype=float)
    return x_c[ui], vals[ui], x_c[li], vals[li]


def split_wall_loop(pts, cells):
    """Split a closed 2D wall loop into ``(upper_idx, lower_idx)``.

    Walks the line cells from the leading edge (minimum x) round to the
    leading edge again, cuts the loop at the trailing edge (maximum x) and
    takes the arc with the larger mean y as the upper surface. Each index
    array runs from the leading to the trailing edge in ascending x. A
    chord-line split is wrong where a surface crosses the chord line (the
    RAE 2822 aft lower surface rises above it); the walk does not depend on
    the chord line.
    """
    pts = np.asarray(pts, dtype=float)
    neighbours = {}
    for cell in cells:
        if len(cell) != 2:
            continue
        a, b = int(cell[0]), int(cell[1])
        neighbours.setdefault(a, []).append(b)
        neighbours.setdefault(b, []).append(a)
    if not neighbours:
        raise ExtractError('No line cells in the surface file: cannot walk the wall loop')
    le = int(np.argmin(pts[:, 0]))
    if le not in neighbours:
        raise ExtractError('The leading-edge node is in no line cell')
    loop, previous, current = [le], None, le
    while True:
        onward = [k for k in neighbours[current] if k != previous]
        if not onward:
            raise ExtractError('The wall line cells do not form a closed loop')
        nxt = onward[0]
        if nxt == le:
            break
        loop.append(nxt)
        previous, current = current, nxt
        if len(loop) > len(neighbours):
            raise ExtractError('The wall line cells do not form a simple loop')
    loop = np.array(loop)
    te = int(np.argmax(pts[loop, 0]))
    arcs = (loop[:te + 1], np.concatenate([loop[te:], [le]]))
    upper = max(arcs, key=lambda arc: pts[arc, 1].mean())
    lower = arcs[1] if upper is arcs[0] else arcs[0]
    return (upper[np.argsort(pts[upper, 0], kind='stable')],
            lower[np.argsort(pts[lower, 0], kind='stable')])


def isentropic_mach(cp, mach, gamma):
    """Isentropic Mach number from a pressure coefficient and the freestream Mach.

    ``p/p_inf = 1 + gamma/2 M_inf^2 Cp`` and ``p0/p_inf = (1 + (gamma-1)/2
    M_inf^2)^(gamma/(gamma-1))``; the result is clipped at zero where the
    pressure exceeds the total pressure.
    """
    cp = np.asarray(cp, dtype=float)
    p_ratio = 1.0 + 0.5 * gamma * mach ** 2 * cp
    p0_ratio = (1.0 + 0.5 * (gamma - 1.0) * mach ** 2) ** (gamma / (gamma - 1.0))
    arg = (p0_ratio / p_ratio) ** ((gamma - 1.0) / gamma) - 1.0
    return np.sqrt(np.maximum(2.0 / (gamma - 1.0) * arg, 0.0))


def wall_series(surface, values):
    """``(x_c_upper, v_upper, x_c_lower, v_lower)`` of a 2D wall field, by x/c.

    Walks the line cells when the surface has them (:func:`split_wall_loop`),
    else cuts at the chord line (:func:`chord_series`). ``values`` is a nodal
    array of the surface.
    """
    pts = np.asarray(surface['points'], dtype=float)
    values = np.asarray(values, dtype=float)
    cells = [c for c in surface.get('cells', ()) if len(c) == 2]
    if not cells:
        return chord_series(pts, values, 2)
    upper, lower = split_wall_loop(pts, cells)
    x = pts[:, 0]
    xc = (x - x.min()) / (x.max() - x.min())
    return xc[upper], values[upper], xc[lower], values[lower]


def delta_y(base, other, n=401):
    """Wall displacement ``other - base`` of a 2D wall on a common x/c grid.

    ``base`` and ``other`` are :func:`readers.read_surface` results with line
    cells; the node order and count may differ. Each wall is cut at the
    leading edge and trailing edge (:func:`split_wall_loop`) and its
    ``y(x/c)`` interpolated on ``n`` points, with x/c measured on ``base``.
    Returns ``{'x', 'upper', 'lower', 'thickness', 'camber'}``: ``upper`` and
    ``lower`` are ``dy`` of each surface, ``thickness = dy_u - dy_l`` and
    ``camber = (dy_u + dy_l) / 2``, in the wall's length unit.
    """
    x0, chord = float(base['points'][:, 0].min()), float(np.ptp(base['points'][:, 0]))
    if chord < 1e-15:
        raise ExtractError('Degenerate chord (x_max = x_min)')
    grid = np.linspace(0.0, 1.0, n)
    profiles = []
    for wall in (base, other):
        pts = np.asarray(wall['points'], dtype=float)
        upper, lower = split_wall_loop(pts, wall.get('cells', ()))
        xc = (pts[:, 0] - x0) / chord
        profiles.append([np.interp(grid, xc[arc], pts[arc, 1]) for arc in (upper, lower)])
    du = profiles[1][0] - profiles[0][0]
    dl = profiles[1][1] - profiles[0][1]
    return {'x': grid, 'upper': du, 'lower': dl, 'thickness': du - dl, 'camber': 0.5 * (du + dl)}


def write_table(path, header, rows):
    """Write a CSV file from header + rows (list-of-lists or list-of-dicts)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        if rows and isinstance(rows[0], dict):
            for row in rows:
                writer.writerow([row.get(h) for h in header])
        else:
            writer.writerows(rows)
