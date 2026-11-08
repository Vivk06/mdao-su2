"""Unstructured tetrahedral half-wing mesh with a revolved tip.

Geometry (gmsh OpenCASCADE kernel), span along +y, lift along +z:

- the section (a Selig loop in the x-z plane) at y = 0, extruded to y = ``span``;
- a tip cap: the upper half of the section at y = ``span``, revolved by 180
  degrees about its chord line, so it runs from z > 0 through y > span to
  z < 0 (ADODG paper p. 23: "a surface of revolution based on the NACA 0012
  aerofoil"). The cap needs a symmetric section and reaches
  y = span + max half-thickness;
- a D-shaped outer cross-section at y = 0: circular upstream arc, straight
  upper/lower reaches, and a downstream outlet. The section is extruded to a
  spanwise farfield plane ``span + farfield_radius`` and the wing is removed.

Sizes come from distance fields: a wall size that grows toward the farfield,
the LE/TE fields measured to the LE/TE lines (``edge_distance``: ``"sampled"``
is gmsh's ``Distance`` field with its default 20 samples per curve, which leaves
the LE up to 5x coarser than requested between samples on a 3 c span;
``"exact"`` is the closed-form distance to the straight edge line),
gradual LE/TE/tip refinement, and a downstream wake corridor. This gives the
unstructured symmetry-plane mesh the C-shaped clustering shown in Fig. 29 of
Yang et al. (2018), while retaining this project's tetrahedral volume format.
``level_scale`` multiplies every size. Every value is an argument.
"""

import numpy as np


def split_section(section):
    """``(upper TE->LE, lower LE->TE)`` of a closed Selig loop (x, z)."""
    s = np.asarray(section, dtype=float)
    if not np.array_equal(s[0], s[-1]):
        raise ValueError('wing3d needs a sharp-TE section (first row == last row)')
    le = int(np.argmin(s[:, 0]))
    return s[:le + 1], s[le:]


def format_difference(var, value):
    """``var - value`` for gmsh's MathEval, which rejects ``x--0.5`` and ``(-0.0)``."""
    value = float(value) + 0.0                     # no -0.0
    return f'({var}+{-value!r})' if value < 0.0 else f'({var}-{value!r})'


def check_symmetric(section, tol=1e-9):
    upper, lower = split_section(section)
    if len(upper) != len(lower) or np.abs(upper[::-1] * [1, -1] - lower).max() > tol:
        raise ValueError('tip = "revolved" needs a symmetric section: the lower '
                         'surface must mirror the upper one')


def build_wing(gmsh, section, span):
    """Add the wing solid (extruded section + revolved cap); returns its volume tag."""
    occ = gmsh.model.occ
    upper, lower = split_section(section)

    def curve(points, y):
        tags = [occ.addPoint(x, y, z) for x, z in points]
        return tags, occ.addSpline(tags)

    # root section face at y = 0
    up_pts, up = curve(upper, 0.0)                     # TE -> LE
    lo_tags = [up_pts[-1]] + [occ.addPoint(x, 0.0, z) for x, z in lower[1:-1]] + [up_pts[0]]
    lo = occ.addSpline(lo_tags)                        # LE -> TE
    root = occ.addPlaneSurface([occ.addCurveLoop([up, lo])])
    prism = [e for e in occ.extrude([(2, root)], 0.0, span, 0.0) if e[0] == 3]

    # cap: upper half face at y = span, revolved about the chord line
    tip_pts, tip_up = curve(upper, span)
    chord = occ.addLine(tip_pts[-1], tip_pts[0])        # LE -> TE on z = 0
    half = occ.addPlaneSurface([occ.addCurveLoop([tip_up, chord])])
    cap = [e for e in occ.revolve([(2, half)], 0.0, span, 0.0, 1.0, 0.0, 0.0, -np.pi)
           if e[0] == 3]
    wing, _ = occ.fuse(prism, cap)
    occ.synchronize()
    return [t for d, t in wing if d == 3]


def build_domain(gmsh, wing_vols, span, farfield_radius, farfield_center):
    occ = gmsh.model.occ
    cx, cy, cz = (float(v) for v in farfield_center)
    r = float(farfield_radius)
    if abs(cy) > 1e-12:
        raise ValueError('wing3d farfield_center must lie on the symmetry plane y=0')
    center = occ.addPoint(cx, 0.0, cz)
    top = occ.addPoint(cx, 0.0, cz + r)
    left = occ.addPoint(cx - r, 0.0, cz)
    bottom = occ.addPoint(cx, 0.0, cz - r)
    outlet_bottom = occ.addPoint(cx + r, 0.0, cz - r)
    outlet_top = occ.addPoint(cx + r, 0.0, cz + r)
    boundary = [occ.addCircleArc(top, center, left),
                occ.addCircleArc(left, center, bottom),
                occ.addLine(bottom, outlet_bottom),
                occ.addLine(outlet_bottom, outlet_top),
                occ.addLine(outlet_top, top)]
    root_face = occ.addPlaneSurface([occ.addCurveLoop(boundary)])
    outer = [e for e in occ.extrude([(2, root_face)], 0.0, span + r, 0.0)
             if e[0] == 3]
    fluid, _ = occ.cut(outer, [(3, t) for t in wing_vols])
    occ.synchronize()
    return [t for d, t in fluid if d == 3]


def classify_surfaces(gmsh, section, span, tol=1e-6):
    """``{'wing': [...], 'tip': [...], 'symmetry': [...], 'farfield': [...]}`` by bounding box."""
    out = {'wing': [], 'tip': [], 'symmetry': [], 'farfield': []}
    section = np.asarray(section)
    x_le, x_te = section[:, 0].min(), section[:, 0].max()
    half_thickness = np.abs(section[:, 1]).max()
    for _, tag in gmsh.model.getEntities(2):
        x0, y0, z0, x1, y1, z1 = gmsh.model.getBoundingBox(2, tag)
        if y1 - y0 < tol and abs(y0) < tol:
            out['symmetry'].append(tag)
        elif (y0 > span - tol and y1 < span + half_thickness + tol and
              x0 >= x_le - tol and x1 <= x_te + tol and
              z0 >= -half_thickness - tol and z1 <= half_thickness + tol):
            out['tip'].append(tag)
        elif (y1 <= span + tol and x0 >= x_le - tol and x1 <= x_te + tol and
              z0 >= -half_thickness - tol and z1 <= half_thickness + tol):
            out['wing'].append(tag)
        else:
            out['farfield'].append(tag)
    return out


def orient_boundaries(gmsh, surfaces):
    """Reverse every surface whose triangles face away from the fluid (SU2's rule).

    SU2 wants each boundary triangle's normal to point into the domain
    (``CPhysicalGeometry.cpp:4219``) and flips any other at start-up. gmsh
    orients each surface by its parametrization, so some face the wrong way.
    A surface is reversed as a whole; a surface with triangles both ways is an
    error. Returns the reversed surface tags.
    """
    from .admission import boundary_orientation
    tags, xyz, _ = gmsh.model.mesh.getNodes()
    index = np.full(int(tags.max()) + 1, -1, dtype=np.int64)
    index[tags.astype(np.int64)] = np.arange(len(tags))
    coords = xyz.reshape(-1, 3)
    types, _, nodes = gmsh.model.mesh.getElements(3)
    tets = np.concatenate([index[n.astype(np.int64)].reshape(-1, 4)
                           for t, n in zip(types, nodes) if t == 4])
    reversed_ = []
    for tag in surfaces:
        types, _, nodes = gmsh.model.mesh.getElements(2, tag)
        tri = np.concatenate([index[n.astype(np.int64)].reshape(-1, 3)
                              for t, n in zip(types, nodes) if t == 2])
        into, away, orphan = boundary_orientation(coords, tets, tri)
        if orphan or (into and away):
            raise RuntimeError(f'surface {tag}: {into} triangles into the fluid, '
                               f'{away} away, {orphan} with no tetrahedron')
        if away:
            gmsh.model.mesh.reverse([(2, tag)])
            reversed_.append(tag)
    return reversed_


def cad_semispan(gmsh, tip_surfaces, n=81):
    """Largest y on the tip cap surfaces (grid search, then a local refinement)."""
    from scipy.optimize import minimize
    best = -np.inf
    for tag in tip_surfaces:
        lo, hi = gmsh.model.getParametrizationBounds(2, tag)
        uu, vv = np.meshgrid(np.linspace(lo[0], hi[0], n), np.linspace(lo[1], hi[1], n))
        uv = np.c_[uu.ravel(), vv.ravel()].ravel()
        y = gmsh.model.getValue(2, tag, uv)[1::3]
        k = int(np.argmax(y))
        start = uv[2 * k:2 * k + 2]

        def neg_y(p):
            p = np.clip(p, lo, hi)
            return -gmsh.model.getValue(2, tag, list(p))[1]
        res = minimize(neg_y, start, method='Nelder-Mead',
                       options={'xatol': 1e-12, 'fatol': 1e-14, 'maxiter': 4000})
        best = max(best, float(y[k]), -float(res.fun))
    return best


def _edges_at(gmsh, x, span, tol=1e-6):
    """Spanwise edge(s) of the wing at chord station ``x`` (the LE or TE line)."""
    tags = []
    for _, tag in gmsh.model.getEntities(1):
        x0, y0, z0, x1, y1, z1 = gmsh.model.getBoundingBox(1, tag)
        if abs(x0 - x) < tol and abs(x1 - x) < tol and y1 - y0 > 0.5 * span:
            tags.append(tag)
    return tags


EDGE_DISTANCES = ('sampled', 'exact')
def generate(section, output_path, *, span, farfield_radius, farfield_center,
             wall_size, le_size, te_size, tip_size, far_size, size_growth,
             edge_band, te_band, te_normal_scale, level_scale, wing_marker, tip_marker, symmetry_marker,
             farfield_marker, edge_distance, verbose=False, threads=1):
    """Write the tetrahedral half-wing mesh; returns a dict of counts and surfaces."""
    import gmsh
    check_symmetric(section)
    upper, _ = split_section(section)
    x_le, x_te = float(np.min(section[:, 0])), float(upper[0, 0])
    k = float(level_scale)
    if edge_distance not in EDGE_DISTANCES:
        raise ValueError(f'wing3d edge_distance={edge_distance!r} must be one of '
                         f'{list(EDGE_DISTANCES)}')
    gmsh.initialize()
    try:
        gmsh.option.setNumber('General.Terminal', 1 if verbose else 0)
        gmsh.option.setNumber('General.NumThreads', int(threads))
        gmsh.model.add('wing3d')
        wing = build_wing(gmsh, section, span)
        fluid = build_domain(gmsh, wing, span, farfield_radius, farfield_center)
        groups = classify_surfaces(gmsh, section, span)
        empty = [g for g, v in groups.items() if not v]
        if empty or len(fluid) != 1:
            raise RuntimeError(f'wing3d geometry: surface groups {groups}, fluid '
                               f'volumes {fluid}')
        walls = groups['wing'] + groups['tip']
        f = gmsh.model.mesh.field
        dist_wall = f.add('Distance')
        f.setNumbers(dist_wall, 'SurfacesList', walls)
        grow = f.add('MathEval')
        # h(d) = k * min(far, wall + growth * d): every size scales with the level.
        f.setString(grow, 'F', f'{k!r}*min({far_size!r}, {wall_size!r} + '
                               f'{size_growth!r}*F{dist_wall})')
        fields = [grow]
        te_distance = None
        for name, size, target in (('le', le_size, ('curves', _edges_at(gmsh, x_le, span))),
                                   ('te', te_size, ('curves', _edges_at(gmsh, x_te, span))),
                                   ('tip', tip_size, ('surfaces', groups['tip']))):
            if not target[1]:
                raise RuntimeError(f'wing3d missing {target[0]} refinement entity')
            if target[0] == 'curves' and edge_distance == 'exact':
                # The LE and TE are straight spanwise lines at z = 0 from the
                # root (y = 0) to the cap (y = span). Inlined: gmsh 4.15 stalls
                # in 1D meshing when a MathEval field reads another MathEval.
                x0 = x_le if name == 'le' else x_te
                d = (f'sqrt({format_difference("x", x0)}^2+z^2+'
                     f'max(0,{format_difference("y", span)})^2)')
            else:
                tag = f.add('Distance')
                f.setNumbers(tag, 'CurvesList' if target[0] == 'curves' else 'SurfacesList',
                             target[1])
                d = f'F{tag}'
            if name == 'te':
                te_distance = d
            # A Threshold to the farfield size over 0.1c jumps to nearly the
            # coarse size immediately: the previous mesh left a 0.012c last
            # panel at the TE despite a requested 0.0023c edge size. Let the
            # field grow only to wall_size over edge_band and continue with the
            # same slope; the wall field takes over farther away.
            edge = f.add('MathEval')
            slope = (wall_size - size) / edge_band
            f.setString(edge, 'F', f'{k!r}*min({far_size!r}, '
                                   f'{size!r}+{slope!r}*{d})')
            fields.append(edge)
        if te_band > edge_band:
            # Widen the TE cluster along the aft wall, not throughout a radial
            # volume. A separate thin downstream corridor resolves the wake.
            slope = (wall_size - te_size) / te_band
            fade_slope = size_growth + (wall_size - te_size) / edge_band
            def aft_size(distance):
                return (f'{k!r}*min({far_size!r}, {te_size!r}+'
                        f'{slope!r}*min({distance},{te_band!r})+'
                        f'{fade_slope!r}*max({distance}-{te_band!r},0))')
            wall_distance = f'max({te_distance},{te_normal_scale!r}*F{dist_wall})'
            wake_distance = (f'{te_normal_scale!r}*max(0,{x_te!r}-x)'
                             f'+max(0,x-{x_te!r})+{te_normal_scale!r}*abs(z)'
                             f'+{te_normal_scale!r}*max(0,y-{float(span)!r})')
            for distance in (wall_distance, wake_distance):
                aft = f.add('MathEval')
                f.setString(aft, 'F', aft_size(distance))
                fields.append(aft)
        # An unstructured analogue of a C-grid wake: resolve the TE and its
        # downstream trace on every span station, then relax streamwise and
        # cross-stream. The max terms keep the corridor downstream of the TE
        # and inside the wing's span (apart from smooth growth at its edges).
        wake = f.add('MathEval')
        f.setString(wake, 'F',
                    f'{k!r}*min({far_size!r}, {te_size!r}+'
                    f'{size_growth!r}*(max(0,{x_te!r}-x)'
                    f'+max(0,x-{x_te!r})+abs(z)'
                    f'+max(0,y-{float(span)!r})))')
        fields.append(wake)
        mn = f.add('Min')
        f.setNumbers(mn, 'FieldsList', fields)
        f.setAsBackgroundMesh(mn)
        for opt in ('MeshSizeExtendFromBoundary', 'MeshSizeFromPoints',
                    'MeshSizeFromCurvature'):
            gmsh.option.setNumber(f'Mesh.{opt}', 0)
        gmsh.option.setNumber('Mesh.Algorithm', 6)       # Frontal-Delaunay surfaces
        gmsh.option.setNumber('Mesh.Algorithm3D', 1)     # Delaunay tetrahedra
        gmsh.option.setNumber('Mesh.Optimize', 1)
        gmsh.option.setNumber('Mesh.OptimizeNetgen', 1)
        for name, tags in ((wing_marker, groups['wing']), (tip_marker, groups['tip']),
                           (symmetry_marker, groups['symmetry']),
                           (farfield_marker, groups['farfield'])):
            gmsh.model.addPhysicalGroup(2, tags, name=name)
        gmsh.model.addPhysicalGroup(3, fluid, name='FLUID')
        gmsh.model.mesh.generate(3)
        flipped = orient_boundaries(gmsh, [t for v in groups.values() for t in v])
        gmsh.option.setNumber('Mesh.SaveAll', 0)
        gmsh.write(output_path)
        _, tets, _ = gmsh.model.mesh.getElements(3)
        info = {'n_nodes': int(len(gmsh.model.mesh.getNodes()[0])),
                'n_tets': int(sum(len(t) for t in tets)),
                'surfaces': {k2: list(v) for k2, v in groups.items()},
                'reversed_surfaces': flipped,
                'cad_semispan': cad_semispan(gmsh, groups['tip']),
                'level_scale': k, 'gmsh_version': gmsh.__version__}
    finally:
        gmsh.finalize()
    return info
