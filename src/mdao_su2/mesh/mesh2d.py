#!/usr/bin/env python
"""Hybrid boundary-layer mesh around a closed 2D section via gmsh.

Every size, marker name and the far-field centre is an argument; nothing names
a case with explicit inputs. ``run/mesh.py mesh2d`` is the command line; its
recipe keys are the argument names below (``RemeshConfig.gmsh_kwargs()`` plus
``wall_marker``, ``farfield_marker`` and ``farfield_center``).
"""

import math
import os

import numpy as np

from .su2_mesh_file import parse_su2_mesh


# ── gmsh mesh generation ───────────────────────────────────────────────

def remesh_airfoil(surface_coords, output_path, loop_order, *,
                   first_cell_height, growth_ratio, n_bl_layers,
                   fan_angle_deg, fan_elements,
                   farfield_radius, farfield_meshsize,
                   near_meshsize, far_growth_slope,
                   wall_marker, farfield_marker, farfield_center,
                   bl_thickness=None, wall_meshsize=None,
                   wake_length=0.0, wake_angles_deg=None, wake_meshsize=None,
                   wake_stream_slope=None,
                   viewable_formats=(), verbose=False):
    """Generate a hybrid mesh around a closed 2D wall loop.

    ``bl_thickness=None`` derives the depth from the layer count and
    ``wall_meshsize=None`` sizes the wall from the input spacing; both are
    stated by ``RemeshConfig.gmsh_kwargs()`` and by the mesh2d recipe.
    ``wake_length > 0`` adds the wake wedge to the size field
    (:func:`size_field_expression`); 0 leaves the field as the wall-distance law.
    """
    import gmsh

    gmsh.initialize()
    gmsh.option.setNumber('General.Terminal', 1 if verbose else 0)
    gmsh.model.add('mesh2d')

    N = len(surface_coords)
    loop = np.asarray(loop_order)
    loop_coords = surface_coords[loop]

    # Sharp-vertex detection on the SURFACE, before any meshing: the included
    # angle is a property of the geometry, so this is the one place that knows
    fan_idx = []
    if fan_angle_deg > 0.0:
        for k in range(N):
            a, b, c = loop_coords[k - 1], loop_coords[k], loop_coords[(k + 1) % N]
            v1, v2 = a - b, c - b
            d1, d2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if d1 == 0.0 or d2 == 0.0:
                continue
            angle = np.degrees(np.arccos(
                np.clip(np.dot(v1, v2) / (d1 * d2), -1.0, 1.0)))
            if angle < fan_angle_deg:
                fan_idx.append(k)

    if bl_thickness is None:
        bl_thickness = (first_cell_height * n_bl_layers if growth_ratio == 1.0
                        else first_cell_height
                        * (growth_ratio ** n_bl_layers - 1) / (growth_ratio - 1))
    thickness = float(bl_thickness)

    # ── airfoil geometry: line segments through input points ──
    #

    pt_tags = []
    for k in range(N):
        x, y = float(loop_coords[k, 0]), float(loop_coords[k, 1])
        ms = wall_meshsize
        if ms is None:
            # local spacing: distance to the next point in the loop
            k_next = (k + 1) % N
            ms = float(np.linalg.norm(loop_coords[k] - loop_coords[k_next]))
        tag = gmsh.model.geo.addPoint(x, y, 0.0, ms)
        pt_tags.append(tag)

    line_tags = []
    for k in range(N):
        lt = gmsh.model.geo.addLine(pt_tags[k], pt_tags[(k + 1) % N])
        line_tags.append(lt)

    airfoil_loop = gmsh.model.geo.addCurveLoop(line_tags)

    # ── far-field boundary: circle ──
    cx, cy = (float(v) for v in farfield_center)
    r = farfield_radius
    p_center = gmsh.model.geo.addPoint(cx, cy, 0.0, farfield_meshsize)
    p_right = gmsh.model.geo.addPoint(cx + r, cy, 0.0, farfield_meshsize)
    p_top = gmsh.model.geo.addPoint(cx, cy + r, 0.0, farfield_meshsize)
    p_left = gmsh.model.geo.addPoint(cx - r, cy, 0.0, farfield_meshsize)
    p_bottom = gmsh.model.geo.addPoint(cx, cy - r, 0.0, farfield_meshsize)

    arc1 = gmsh.model.geo.addCircleArc(p_right, p_center, p_top)
    arc2 = gmsh.model.geo.addCircleArc(p_top, p_center, p_left)
    arc3 = gmsh.model.geo.addCircleArc(p_left, p_center, p_bottom)
    arc4 = gmsh.model.geo.addCircleArc(p_bottom, p_center, p_right)
    ff_arcs = [arc1, arc2, arc3, arc4]
    ff_loop = gmsh.model.geo.addCurveLoop(ff_arcs)

    # ── fluid domain surface (hole = airfoil) ──
    fluid_surf = gmsh.model.geo.addPlaneSurface([ff_loop, airfoil_loop])

    gmsh.model.geo.synchronize()

    # ── retain every input vertex, optionally subdividing long segments ──
    # One element per input segment preserves the source discretisation.  An
    # explicit wall_meshsize may refine those straight segments without moving
    # their endpoints, which keeps the surface-node map valid.
    for k, lt in enumerate(line_tags):
        n_segments = 1
        if wall_meshsize is not None:
            k_next = (k + 1) % N
            length = float(np.linalg.norm(loop_coords[k] - loop_coords[k_next]))
            n_segments = max(1, int(np.ceil(length / wall_meshsize)))
        gmsh.model.mesh.setTransfiniteCurve(lt, n_segments + 1)

    # ── physical groups → SU2 markers ──
    gmsh.model.addPhysicalGroup(1, line_tags, name=wall_marker)
    gmsh.model.addPhysicalGroup(1, ff_arcs, name=farfield_marker)
    gmsh.model.addPhysicalGroup(2, [fluid_surf], name='FLUID')

    # ── boundary-layer field (structured quad BL) ──
    bl_tag = gmsh.model.mesh.field.add('BoundaryLayer')
    gmsh.model.mesh.field.setNumbers(bl_tag, 'CurvesList', line_tags)
    gmsh.model.mesh.field.setNumbers(bl_tag, 'PointsList', pt_tags)
    gmsh.model.mesh.field.setNumber(bl_tag, 'Size', first_cell_height)
    gmsh.model.mesh.field.setNumber(bl_tag, 'Ratio', growth_ratio)
    gmsh.model.mesh.field.setNumber(bl_tag, 'NbLayers', n_bl_layers)
    # Thickness is what actually terminates the extrusion; NbLayers alone is
    # advisory.  See the parameter docs above.
    gmsh.model.mesh.field.setNumber(bl_tag, 'Thickness', thickness)
    gmsh.model.mesh.field.setNumber(bl_tag, 'Quads', 1)
    if fan_idx:
        # A fan opens the cusp out into `fan_elements` wedges instead of forcing
        # the quads from both surfaces to meet at the vertex angle.
        gmsh.model.mesh.field.setNumbers(
            bl_tag, 'FanPointsList', [float(pt_tags[k]) for k in fan_idx])
        gmsh.model.mesh.field.setNumbers(
            bl_tag, 'FanPointsSizesList',
            [float(fan_elements) for _ in fan_idx])
        if verbose:
            print(f'[remesh] fanning {len(fan_idx)} sharp vertex/vertices '
                  f'at {[tuple(np.round(loop_coords[k], 6)) for k in fan_idx]}')
    gmsh.model.mesh.field.setAsBoundaryLayer(bl_tag)

    # ── mesh size field: proportional to wall distance, plus the wake wedge ──
    dist_tag = gmsh.model.mesh.field.add('Distance')
    gmsh.model.mesh.field.setNumbers(dist_tag, 'CurvesList', line_tags)
    size_tag = gmsh.model.mesh.field.add('MathEval')
    wake = None
    if wake_length > 0.0:
        wake = dict(apex=tuple(0.5 * (loop_coords[0] + loop_coords[-1])),
                    length=wake_length, angles_deg=tuple(wake_angles_deg),
                    meshsize=wake_meshsize, stream_slope=wake_stream_slope)
    gmsh.model.mesh.field.setString(
        size_tag, 'F',
        size_field_expression(farfield_meshsize, near_meshsize, far_growth_slope,
                              f'F{dist_tag}', wake))
    gmsh.model.mesh.field.setAsBackgroundMesh(size_tag)

    # ── mesh generation ──
    gmsh.option.setNumber('Mesh.Algorithm', 6)  # Frontal-Delaunay
    gmsh.option.setNumber('Mesh.RecombineAll', 0)  # tri in far-field
    gmsh.option.setNumber('Mesh.MeshSizeExtendFromBoundary', 0)
    gmsh.option.setNumber('Mesh.MeshSizeFromPoints', 0)
    gmsh.option.setNumber('Mesh.MeshSizeFromCurvature', 0)
    gmsh.model.mesh.generate(2)

    # ── export ──
    gmsh.option.setNumber('Mesh.SaveAll', 0)  # only physical groups
    gmsh.write(output_path)
    write_viewable(gmsh, output_path, viewable_formats, verbose=verbose)

    # ── mesh statistics ──
    mesh_info = _mesh_stats(gmsh)

    gmsh.finalize()
    return mesh_info


def size_field_expression(farfield_meshsize, near_meshsize, far_growth_slope,
                          distance_field, wake=None):
    """The gmsh ``MathEval`` background-size expression.

    Wall law: ``min(farfield_meshsize, max(near_meshsize, far_growth_slope * d))``
    with ``d`` the wall distance (``distance_field`` names the Distance field,
    e.g. ``"F1"``). A ``wake`` dict (``apex``, ``length``, ``angles_deg`` = the
    wedge's two ray angles from +x, ``meshsize``, ``stream_slope``) takes the
    minimum with ``h_wake = meshsize + stream_slope * t + far_growth_slope * d_w``,
    where ``t = max(0, e.q)`` runs along the wedge bisector ``e`` and
    ``d_w = max(0, n_lo.q, n_hi.q, e.q - length * cos(half_angle), -e.q)``,
    ``q = p - apex``, is the largest signed distance to the wedge's four bounding
    lines (outward ray normals ``n_lo``, ``n_hi``, the far edge, and the line
    through the apex normal to ``e``). ``d_w`` is a lower bound of the distance to
    the wedge, so the size is conservative (finer) off the wedge. Without a
    ``wake`` the string is the wall law alone.
    """
    wall = (f'min({farfield_meshsize!r}, '
            f'max({near_meshsize!r}, {far_growth_slope!r}*{distance_field}))')
    if wake is None:
        return wall
    lo, hi = (math.radians(a) for a in wake['angles_deg'])
    mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo)
    x0, y0 = (float(v) for v in wake['apex'])
    dx, dy = f'(x-({x0!r}))', f'(y-({y0!r}))'

    def along(ux, uy):
        return f'(({ux!r})*{dx}+({uy!r})*{dy})'

    e = along(math.cos(mid), math.sin(mid))
    d_w = (f'max(max(max(0, {along(math.sin(lo), -math.cos(lo))}), '
           f'max({along(-math.sin(hi), math.cos(hi))}, '
           f'{e}-{wake["length"] * math.cos(half)!r})), 0-{e})')
    h_wake = (f'({wake["meshsize"]!r}+{wake["stream_slope"]!r}*max(0, {e})'
              f'+{far_growth_slope!r}*{d_w})')
    return f'min({wall}, {h_wake})'


#: Formats a mesh can be written in for INSPECTION.  ``.su2`` is what the solver
#: reads and is not viewable: gmsh cannot open it back, and ParaView has no
VIEWABLE_FORMATS = ('msh', 'vtk')


def write_viewable(gmsh, output_path, formats=(), verbose=False):
    """Write viewable twins of ``output_path`` beside it, one per format."""
    written = []
    for fmt in formats:
        fmt = fmt.lower().lstrip('.')
        if fmt not in VIEWABLE_FORMATS:
            raise ValueError(
                f'unsupported viewable mesh format {fmt!r}; '
                f'expected one of {VIEWABLE_FORMATS}')
        path = os.path.splitext(output_path)[0] + '.' + fmt
        try:
            gmsh.write(path)
            written.append(path)
        except Exception as exc:                       # noqa: BLE001
            print(f'[remesh] WARNING: could not write {path}: {exc}')
    if verbose and written:
        print(f'[remesh] viewable mesh: {", ".join(written)}')
    return written


def _mesh_stats(gmsh):
    """Collect mesh statistics."""
    node_tags, _, _ = gmsh.model.mesh.getNodes()
    elem_types, _, _ = gmsh.model.mesh.getElements(dim=2)
    n_tri = 0
    n_quad = 0
    for et in elem_types:
        name, _, _, _, _, _ = gmsh.model.mesh.getElementProperties(et)
        if 'Triangle' in name:
            tags, _ = gmsh.model.mesh.getElementsByType(et)
            n_tri += len(tags)
        elif 'Quadrilateral' in name or 'Quad' in name:
            tags, _ = gmsh.model.mesh.getElementsByType(et)
            n_quad += len(tags)

    return {
        'n_nodes': len(node_tags),
        'n_tri': n_tri,
        'n_quad': n_quad,
        'n_elements': n_tri + n_quad,
    }


# ── mesh quality verification ──────────────────────────────────────────

def verify_mesh(mesh_path, wall_marker, farfield_marker, verbose=True):
    """Verify mesh quality and marker structure for SU2 compatibility.

    The report keeps its historical key names (``n_airfoil_nodes`` ...) for the
    wall marker, whatever it is called.
    """
    points, elements, markers = parse_su2_mesh(mesh_path)
    report = {}
    passed = True

    # 1. marker check
    report['markers'] = list(markers.keys())
    if wall_marker not in markers:
        report['error_airfoil'] = f'{wall_marker} marker missing'
        passed = False
    if farfield_marker not in markers:
        report['error_farfield'] = f'{farfield_marker} marker missing'
        passed = False

    # 2. node and element counts
    report['n_nodes'] = len(points)
    n_tri = sum(1 for e in elements if e[0] == 5)
    n_quad = sum(1 for e in elements if e[0] == 9)
    report['n_tri'] = n_tri
    report['n_quad'] = n_quad
    report['n_elements'] = n_tri + n_quad

    # 3. wall boundary stats
    if wall_marker in markers:
        af_nodes = set()
        for e in markers[wall_marker]:
            af_nodes.add(e[1])
            af_nodes.add(e[2])
        report['n_airfoil_nodes'] = len(af_nodes)
        report['n_airfoil_elems'] = len(markers[wall_marker])

    # 4. far-field boundary stats
    if farfield_marker in markers:
        ff_nodes = set()
        for e in markers[farfield_marker]:
            ff_nodes.add(e[1])
            ff_nodes.add(e[2])
        report['n_farfield_nodes'] = len(ff_nodes)

    # 5. element quality (negative area check)
    coords = np.array([points[i] for i in range(len(points))])
    neg_area = 0
    min_area = float('inf')
    max_aspect = 0.0
    for e in elements:
        if e[0] == 5:  # triangle
            p0, p1, p2 = coords[e[1]], coords[e[2]], coords[e[3]]
            area = 0.5 * abs((p1[0] - p0[0]) * (p2[1] - p0[1]) -
                             (p2[0] - p0[0]) * (p1[1] - p0[1]))
            if area < 1e-20:
                neg_area += 1
            min_area = min(min_area, area)
            edges = [np.linalg.norm(p1 - p0), np.linalg.norm(p2 - p1),
                     np.linalg.norm(p0 - p2)]
            if min(edges) > 0:
                max_aspect = max(max_aspect, max(edges) / min(edges))
        elif e[0] == 9:  # quad
            p0, p1 = coords[e[1]], coords[e[2]]
            p2, p3 = coords[e[3]], coords[e[4]]
            area1 = 0.5 * abs((p1[0] - p0[0]) * (p2[1] - p0[1]) -
                              (p2[0] - p0[0]) * (p1[1] - p0[1]))
            area2 = 0.5 * abs((p2[0] - p0[0]) * (p3[1] - p0[1]) -
                              (p3[0] - p0[0]) * (p2[1] - p0[1]))
            area = area1 + area2
            if area < 1e-20:
                neg_area += 1
            min_area = min(min_area, area)
            edges = [np.linalg.norm(p1 - p0), np.linalg.norm(p2 - p1),
                     np.linalg.norm(p3 - p2), np.linalg.norm(p0 - p3)]
            if min(edges) > 0:
                max_aspect = max(max_aspect, max(edges) / min(edges))

    report['negative_area_cells'] = neg_area
    report['min_area'] = min_area
    report['max_aspect_ratio'] = max_aspect

    if neg_area > 0:
        passed = False
    if max_aspect > 1000:
        report['warning_aspect'] = f'max aspect ratio {max_aspect:.1f} > 1000'

    # 6. first cell height estimate (distance from wall node to nearest interior)
    if wall_marker in markers:
        af_nodes_set = set()
        for e in markers[wall_marker]:
            af_nodes_set.add(e[1])
            af_nodes_set.add(e[2])
        interior = set(range(len(points))) - af_nodes_set
        if interior:
            from scipy.spatial import KDTree
            af_coords = coords[sorted(af_nodes_set)]
            int_coords = coords[sorted(interior)]
            tree = KDTree(int_coords)
            dists, _ = tree.query(af_coords)
            report['min_first_cell_height'] = float(dists.min())
            report['max_first_cell_height'] = float(dists.max())
            report['mean_first_cell_height'] = float(dists.mean())

    if verbose:
        print('=== Mesh Verification Report ===')
        for k, v in report.items():
            print(f'  {k}: {v}')
        print(f'  VERDICT: {"PASS" if passed else "FAIL"}')

    return passed, report
