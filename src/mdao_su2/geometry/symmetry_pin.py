"""Hold wall nodes that lie on a symmetry plane on that plane (``[geometry] pin_symmetry``).

On a half model the wall meets the symmetry plane at corner nodes (ADODG Case 1: the
LE and TE on y = 0). A shape basis that does not vanish exactly there -- the paper's
control-point FFD box ends at x = 1.001 (Yang & Da Ronch 2018, Table 3), so its last
free column keeps a Bernstein weight of about 0.0099 at the TE -- lifts the corner
off the plane and kinks it. The MG3 primal then stalls on a residual floor that
scales with the lift. The human guide summarizes this limit.

The pin zeroes each such node's displacement normal to the plane, and the matching
derivative rows, inside the geometry component. Every consumer of the geometry
output -- SU2, the snapshots, verification -- therefore sees the same surface.
"""

import numpy as np

__all__ = ['pinned_wall_nodes', 'apply_symmetry_pin']

#: SU2 boundary element VTK type -> node count (line, triangle, quadrilateral).
_BOUNDARY_NODES = {3: 2, 5: 3, 9: 4}
#: A symmetry plane is axis-aligned when its smallest coordinate extent is below
#: this fraction of its largest.
_PLANE_TOL = 1e-9


def _marker_nodes(markers, tags):
    nodes = set()
    for tag in tags:
        for element in markers[tag]:
            n = _BOUNDARY_NODES.get(element[0])
            if n is None:
                raise ValueError(
                    f'{tag}: unsupported boundary element type {element[0]}')
            nodes.update(element[1:1 + n])
    return nodes


def pinned_wall_nodes(mesh_path, wall_markers, symmetry_markers):
    """``(sorted wall node ids on the plane, normal axis)``. The plane must be axis-aligned."""
    from ..mesh.su2_mesh_file import parse_su2_mesh

    points, _, markers = parse_su2_mesh(mesh_path)
    plane = _marker_nodes(markers, symmetry_markers)
    coords = np.array([points[g] for g in sorted(plane)], dtype=float)
    extent = np.ptp(coords, axis=0)  # coordinate extent
    axis = int(np.argmin(extent))
    if extent[axis] > _PLANE_TOL * max(1.0, float(np.max(extent))):
        raise ValueError(
            f'symmetry markers {list(symmetry_markers)} do not lie on one '
            f'axis-aligned plane (smallest coordinate extent {extent[axis]:.3e}); '
            f'[geometry] pin_symmetry supports x, y or z = const planes only')
    wall = _marker_nodes(markers, wall_markers)
    return sorted(wall & plane), axis


def symmetry_pinned_rows(mesh_path, wall_markers, symmetry_markers, gid_order):
    """Rows of the gid-ordered wall surface that lie on the plane, and its normal axis."""
    nodes, axis = pinned_wall_nodes(mesh_path, wall_markers, symmetry_markers)
    nodes = set(nodes)
    rows = np.array([i for i, gid in enumerate(gid_order) if int(gid) in nodes],
                    dtype=int)
    return rows, axis


def pin_dvgeo_point_set(dvgeo, pt_name, reference, rows, axis):
    """Make one DVGeometry keep ``rows`` of ``pt_name`` on their reference ``axis`` value.

    Wraps the three calls ``OM_DVGEOCOMP`` makes for a point set: ``update``
    (output), ``totalSensitivity`` (reverse) and ``totalSensitivityProd``
    (forward). Other point sets, such as DVConstraints', pass through unchanged.
    The forward product is also flattened. That covers the empty-rank case
    ``FFDGeoBidirComp`` patches, which skips a DVGeometry whose
    ``totalSensitivityProd`` is already an instance attribute.
    """
    reference = np.asarray(reference, dtype=float).reshape(-1, 3)
    rows = np.asarray(rows, dtype=int)
    flat = rows * 3 + axis
    update = dvgeo.update
    sensitivity = dvgeo.totalSensitivity
    product = dvgeo.totalSensitivityProd

    def pinned_update(ptSetName, *args, **kwargs):
        out = update(ptSetName, *args, **kwargs)
        if ptSetName != pt_name or rows.size == 0:
            return out
        out = np.array(out, dtype=float)  # a copy: never write into pyGeo's storage
        out.reshape(-1, 3)[rows, axis] = reference[rows, axis]
        return out

    def pinned_sensitivity(dIdpt, ptSetName, *args, **kwargs):
        if ptSetName == pt_name and rows.size:
            dIdpt = np.array(dIdpt, dtype=float)
            dIdpt[..., rows, axis] = 0.0
        return sensitivity(dIdpt, ptSetName, *args, **kwargs)

    def pinned_product(vec, ptSetName, *args, **kwargs):
        out = np.asarray(product(vec, ptSetName, *args, **kwargs),
                         dtype=float).reshape(-1)
        if ptSetName == pt_name and rows.size:
            out = out.copy()
            out[flat] = 0.0
        return out

    dvgeo.update = pinned_update
    dvgeo.totalSensitivity = pinned_sensitivity
    dvgeo.totalSensitivityProd = pinned_product


def apply_symmetry_pin(geometry, builder, cfg):
    """Apply ``[geometry] pin_symmetry`` to the embedded wall point set.

    Call after the wall points are embedded. ``builder`` supplies the rank-local
    surface (``get_surface_points``) and the gid order of the whole wall
    (``gid_order``), as ``SU2Builder`` does. Returns the rank-local pinned row count.
    """
    if not cfg.geometry.pin_symmetry:
        return 0
    from mphys import MPhysVariables
    pt_name = MPhysVariables.Aerodynamics.Surface.Geometry.COORDINATES_OUTPUT

    wall, sym = cfg.case.design_markers, cfg.case.markers_of('sym')
    gid_order = np.asarray(builder.gid_order)
    rows, axis = symmetry_pinned_rows(cfg.mesh_path, wall, sym, gid_order)
    if rows.size == 0:
        raise ValueError(
            f'[geometry] pin_symmetry is on, but no {list(wall)} node lies '
            f'on {list(sym)}; a pin that holds nothing is a mistake')
    reference = np.asarray(builder.get_surface_points(), dtype=float).reshape(-1, 3)
    if len(reference) == 0:
        rows = rows[:0]  # rank-0 surface partition: this rank embeds no points
    elif len(reference) != len(gid_order):
        raise ValueError(
            f'pin_symmetry needs the whole wall on one rank (SU2Builder partition); '
            f'this rank holds {len(reference)} of {len(gid_order)} points')

    parameterization = cfg.geometry.parameterization
    if parameterization == 'ffd':
        pin_dvgeo_point_set(geometry.nom_getDVGeo(), pt_name, reference, rows, axis)
    elif parameterization == 'hicks_henne':
        geometry.nom_pin_points(pt_name, rows, axis)
    else:
        raise ValueError(f'[geometry] pin_symmetry does not support '
                         f'parameterization={parameterization!r}')
    return int(rows.size)
