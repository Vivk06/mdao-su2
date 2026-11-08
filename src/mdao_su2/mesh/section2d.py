"""All-triangle 2D Euler mesh around a sharp-TE section, sized as a wing3d span station.

The mesh is the 2D analogue of ``wing3d.generate`` at one span
station away from the root and tip: the same OCC upper and lower splines
through the section points, the same D-shaped outer boundary (upstream arc,
straight upper and lower reaches, outlet at ``x = cx + r``), and the ``y =
const`` slice of its size field. Section coordinates ``(x, z)`` of wing3d are
``(x, y)`` here. The fields are

- wall: ``min(far, wall + growth * d_wall)``;
- LE and TE: ``min(far, size + (wall - size) / edge_band * d)``, with ``d`` the
  distance to the LE or TE point (wing3d measures it to the LE or TE line);
- the TE diamond ("wake"): ``min(far, te + growth * (|x - x_te| + |y|))``;

each multiplied by ``level_scale[level]``. With the wing3d recipe's sizes and
level factor this is the section resolution that the 3D recipe *requests*.
A smaller factor refines every size uniformly, which gives a grid family in
the sense of Vassberg & Jameson (2010).

wing3d samples its LE line with gmsh's default ``Distance`` sampling, so the
built 3D LE is coarser than requested between samples. Here every distance is
exact (a point, or a curve sampled at a quarter of the smallest size), so an
as-built 3D station is reproduced by overriding ``le_size`` rather than by
copying the sampling gap. Every value is an argument.
"""

import math

import numpy as np

from .wing3d import format_difference, split_section


def _spline(occ, points):
    tags = [occ.addPoint(float(x), float(y), 0.0) for x, y in points]
    return tags, occ.addSpline(tags)


def build_section(gmsh, section):
    """Add the wall loop (upper TE->LE, lower LE->TE splines); returns ``(loop, curves)``."""
    occ = gmsh.model.occ
    upper, lower = split_section(section)
    up_pts, up = _spline(occ, upper)
    lo_tags = ([up_pts[-1]] + [occ.addPoint(float(x), float(y), 0.0)
                               for x, y in lower[1:-1]] + [up_pts[0]])
    lo = occ.addSpline(lo_tags)
    return occ.addCurveLoop([up, lo]), [up, lo]


def build_domain(gmsh, farfield_radius, farfield_center):
    """Add the D-shaped outer loop; returns ``(loop, curves)``."""
    occ = gmsh.model.occ
    cx, cy = (float(v) for v in farfield_center)
    r = float(farfield_radius)
    center = occ.addPoint(cx, cy, 0.0)
    top = occ.addPoint(cx, cy + r, 0.0)
    left = occ.addPoint(cx - r, cy, 0.0)
    bottom = occ.addPoint(cx, cy - r, 0.0)
    outlet_bottom = occ.addPoint(cx + r, cy - r, 0.0)
    outlet_top = occ.addPoint(cx + r, cy + r, 0.0)
    curves = [occ.addCircleArc(top, center, left),
              occ.addCircleArc(left, center, bottom),
              occ.addLine(bottom, outlet_bottom),
              occ.addLine(outlet_bottom, outlet_top),
              occ.addLine(outlet_top, top)]
    return occ.addCurveLoop(curves), curves


def generate(section, output_path, *, farfield_radius, farfield_center, wall_size,
             le_size, te_size, far_size, size_growth, edge_band, level_scale,
             wall_marker, farfield_marker, verbose=False):
    """Write the 2D mesh; returns a dict of counts and the resolved scale."""
    import gmsh
    section = np.asarray(section, dtype=float)
    upper, _ = split_section(section)
    le = section[int(np.argmin(section[:, 0]))]
    te = upper[0]
    k = float(level_scale)
    gmsh.initialize()
    try:
        gmsh.option.setNumber('General.Terminal', 1 if verbose else 0)
        gmsh.model.add('section2d')
        wall_loop, wall = build_section(gmsh, section)
        far_loop, far = build_domain(gmsh, farfield_radius, farfield_center)
        fluid = gmsh.model.occ.addPlaneSurface([far_loop, wall_loop])
        gmsh.model.occ.synchronize()
        f = gmsh.model.mesh.field
        # Sample each wall curve at a quarter of the smallest requested size, so
        # the wall distance is exact to that tolerance (gmsh's default is 20).
        h_min = k * min(wall_size, le_size, te_size)
        arc = float(np.linalg.norm(np.diff(upper, axis=0), axis=1).sum())
        dist_wall = f.add('Distance')
        f.setNumbers(dist_wall, 'CurvesList', wall)
        f.setNumber(dist_wall, 'Sampling', int(math.ceil(arc / (0.25 * h_min))))
        grow = f.add('MathEval')
        f.setString(grow, 'F', f'{k!r}*min({far_size!r}, {wall_size!r} + '
                               f'{size_growth!r}*F{dist_wall})')
        fields = [grow]
        for (px, py), size in ((le, le_size), (te, te_size)):
            d = f'sqrt({format_difference("x", px)}^2+{format_difference("y", py)}^2)'
            edge = f.add('MathEval')
            slope = (wall_size - size) / edge_band
            f.setString(edge, 'F', f'{k!r}*min({far_size!r}, {size!r}+{slope!r}*{d})')
            fields.append(edge)
        wake = f.add('MathEval')
        f.setString(wake, 'F', f'{k!r}*min({far_size!r}, {te_size!r}+{size_growth!r}*'
                               f'(abs{format_difference("x", te[0])}+'
                               f'abs{format_difference("y", te[1])}))')
        fields.append(wake)
        mn = f.add('Min')
        f.setNumbers(mn, 'FieldsList', fields)
        f.setAsBackgroundMesh(mn)
        for opt in ('MeshSizeExtendFromBoundary', 'MeshSizeFromPoints',
                    'MeshSizeFromCurvature'):
            gmsh.option.setNumber(f'Mesh.{opt}', 0)
        gmsh.option.setNumber('Mesh.Algorithm', 6)       # Frontal-Delaunay, as wing3d
        gmsh.option.setNumber('Mesh.RecombineAll', 0)     # triangles only
        gmsh.model.addPhysicalGroup(1, wall, name=wall_marker)
        gmsh.model.addPhysicalGroup(1, far, name=farfield_marker)
        gmsh.model.addPhysicalGroup(2, [fluid], name='FLUID')
        gmsh.model.mesh.generate(2)
        gmsh.option.setNumber('Mesh.SaveAll', 0)
        gmsh.write(output_path)
        _, tris, _ = gmsh.model.mesh.getElements(2)
        info = {'n_nodes': int(len(gmsh.model.mesh.getNodes()[0])),
                'n_tri': int(sum(len(t) for t in tris)),
                'level_scale': k, 'gmsh_version': gmsh.__version__}
    finally:
        gmsh.finalize()
    return info


def wall_spacing(mesh_path, wall_marker):
    """Wall segment lengths by chord band and upper/lower node counts (reported, not gated)."""
    from .su2_mesh_file import parse_su2_mesh
    points, _, markers = parse_su2_mesh(mesh_path)
    edges = np.array([(e[1], e[2]) for e in markers.get(wall_marker, ())])
    if not len(edges):
        return {}
    xy = np.array([points[i] for i in range(len(points))])
    a, b = xy[edges[:, 0]], xy[edges[:, 1]]
    length = np.linalg.norm(b - a, axis=1)
    mid = 0.5 * (a + b)
    x0, x1 = float(mid[:, 0].min()), float(mid[:, 0].max())
    xc = (mid[:, 0] - x0) / (x1 - x0)
    out = {}
    for name, lo, hi in (('le_0_0.01', 0.0, 0.01), ('fore_0.01_0.2', 0.01, 0.2),
                         ('mid_0.2_0.8', 0.2, 0.8), ('aft_0.8_0.99', 0.8, 0.99),
                         ('te_0.99_1', 0.99, 1.0 + 1e-12)):
        sel = (xc >= lo) & (xc < hi)
        if sel.any():
            out[name] = {'n': int(sel.sum()), 'min': float(length[sel].min()),
                         'max': float(length[sel].max())}
    nodes = np.unique(edges)
    out['n_upper_nodes'] = int((xy[nodes, 1] > 1e-12).sum())
    out['n_lower_nodes'] = int((xy[nodes, 1] < -1e-12).sum())
    return out


def check(measured, spacing, quality, wall_marker, farfield_marker):
    """``[(name, ok, detail), ...]`` for an all-triangle section mesh."""
    results = []

    def add(name, ok, detail):
        results.append((name, bool(ok), detail))

    if 'error' in measured:
        add('mesh_structure', False, measured['error'])
        return results
    need = {wall_marker, farfield_marker}
    add('markers', need <= set(measured['markers']),
        f"{measured['markers']} (need {sorted(need)})")
    add('triangles_only', measured['n_quad'] == 0 and measured['n_tri'] > 0,
        f"{measured['n_tri']} tri + {measured['n_quad']} quad")
    add('wall_closed', measured['n_wall_nodes'] == measured['n_wall_elements'],
        f"{measured['n_wall_nodes']} wall nodes, {measured['n_wall_elements']} segments")
    add('no_degenerate_cells', measured['degenerate_cells'] == 0,
        f"{measured['degenerate_cells']} cells with area <= 1e-20")
    add('skewness', measured['max_skewness'] <= quality['max_skewness'],
        f"max {measured['max_skewness']:.3f}, allowed {quality['max_skewness']:g}")
    add('aspect_ratio', measured['aspect_ratio_p99'] <= quality['max_aspect_ratio_p99'],
        f"p99 {measured['aspect_ratio_p99']:.2f}, allowed "
        f"{quality['max_aspect_ratio_p99']:g}")
    bands = ', '.join(f"{k} {v['n']}x[{v['min']:.5f}, {v['max']:.5f}]"
                      for k, v in spacing.items() if isinstance(v, dict))
    add('wall_spacing (reported)', True,
        f"{bands}; upper/lower nodes {spacing.get('n_upper_nodes')}/"
        f"{spacing.get('n_lower_nodes')}")
    return results
