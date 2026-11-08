"""3D mesh admission for SU2 ASCII meshes.

Gates (each reported, passing ones included):

- **markers**: the mesh has exactly the expected markers.
- **positive_volumes**: every tetrahedron and prism has positive volume in the
  VTK/SU2 node order (``CPhysicalGeometry::Check_IntElem_Orientation``).
- **boundary_orientation**: every boundary triangle's normal points into the
  fluid, so the triangle and the domain node opposite it make a positive
  tetrahedron. SU2 flips any other triangle
  (``CPhysicalGeometry.cpp:4219``); a mesh SU2 must repair is not admitted.
  On the wing this is the normal pointing out of the body.
- **closed_wall**: every edge of the wall markers is shared by two wall
  triangles, except edges on the symmetry plane, which have one.
- **symmetry_plane**: every node of the symmetry marker has ``|y| <= tol``.
- **dihedral** and **aspect_ratio**: tetrahedron shape bounds from the recipe.

Nothing here names a case or a marker by default.
"""

import numpy as np

TET, PRISM, PYRAMID, HEXA, TRI, QUAD = 10, 13, 14, 12, 5, 9
_NODES = {3: 2, TRI: 3, QUAD: 4, TET: 4, HEXA: 8, PRISM: 6, PYRAMID: 5}


# -------------------------------------------------------------------- read

def read_su2(path):
    """``(coords (N,3), volume {vtk: (M,k) int}, markers {name: {vtk: (M,k) int}})``.

    Point indices are the trailing IDs of the NPOIN block.
    """
    with open(path) as fh:
        lines = fh.read().splitlines()
    i, ndim = 0, None
    coords, volume, markers = None, {}, {}
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('NDIME='):
            ndim = int(line.split('=')[1])
            if ndim != 3:
                raise ValueError(f'{path}: NDIME={ndim}; quality3d admits 3D meshes')
            i += 1
        elif line.startswith('NPOIN='):
            n = int(line.split('=')[1].split()[0])
            block = np.array([ln.split() for ln in lines[i + 1:i + 1 + n]], dtype=float)
            ids = block[:, -1].astype(np.int64)
            coords = np.zeros((n, 3))
            coords[ids, :ndim] = block[:, :ndim]
            i += 1 + n
        elif line.startswith('NELEM='):
            n = int(line.split('=')[1].split()[0])
            volume = _group(lines[i + 1:i + 1 + n], with_id=True)
            i += 1 + n
        elif line.startswith('MARKER_TAG='):
            tag = line.split('=', 1)[1].strip()
            n = int(lines[i + 1].split('=')[1])
            markers[tag] = _group(lines[i + 2:i + 2 + n], with_id=False)
            i += 2 + n
        else:
            i += 1
    if ndim != 3:
        raise ValueError(f'{path}: NDIME={ndim}; quality3d admits 3D meshes')
    return coords, volume, markers


def _group(rows, with_id):
    out = {}
    for ln in rows:
        parts = ln.split()
        vtk = int(parts[0])
        k = _NODES[vtk]
        out.setdefault(vtk, []).append([int(p) for p in parts[1:1 + k]])
    return {vtk: np.array(v, dtype=np.int64) for vtk, v in out.items()}


# --------------------------------------------------------------- geometry

def tet_volumes(coords, tets):
    p0, p1, p2, p3 = (coords[tets[:, j]] for j in range(4))
    return np.einsum('ij,ij->i', np.cross(p1 - p0, p2 - p0), p3 - p0) / 6.0


def prism_volumes(coords, prisms):
    """Wedge (0,1,2 | 3,4,5) as three tetrahedra in a consistent orientation."""
    split = ([0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5])
    return sum(tet_volumes(coords, prisms[:, s]) for s in split)


def dihedral_angles(coords, tets):
    """``(M, 6)`` interior dihedral angles in degrees (orientation-free).

    At edge (a, b) the angle is between the other two vertices projected onto
    the plane normal to the edge.
    """
    p = [coords[tets[:, j]] for j in range(4)]
    out = []
    for a, b, c, d in ((0, 1, 2, 3), (0, 2, 1, 3), (0, 3, 1, 2),
                       (1, 2, 0, 3), (1, 3, 0, 2), (2, 3, 0, 1)):
        e = p[b] - p[a]
        e = e / np.linalg.norm(e, axis=1, keepdims=True)
        u, v = p[c] - p[a], p[d] - p[a]
        u = u - np.einsum('ij,ij->i', u, e)[:, None] * e
        v = v - np.einsum('ij,ij->i', v, e)[:, None] * e
        cos = np.einsum('ij,ij->i', u, v) / (np.linalg.norm(u, axis=1)
                                             * np.linalg.norm(v, axis=1))
        out.append(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))
    return np.stack(out, axis=1)


def aspect_ratios(coords, tets):
    """Circumradius / (3 x inradius); 1 for a regular tetrahedron."""
    p0, p1, p2, p3 = (coords[tets[:, j]] for j in range(4))
    a, b, c = p1 - p0, p2 - p0, p3 - p0
    vol6 = np.abs(np.einsum('ij,ij->i', np.cross(a, b), c))
    area = sum(0.5 * np.linalg.norm(np.cross(q - p, r - p), axis=1)
               for p, q, r in ((p1, p2, p3), (p0, p2, p3), (p0, p1, p3), (p0, p1, p2)))
    inradius = vol6 / 2.0 / area
    num = (np.einsum('ij,ij->i', a, a)[:, None] * np.cross(b, c)
           + np.einsum('ij,ij->i', b, b)[:, None] * np.cross(c, a)
           + np.einsum('ij,ij->i', c, c)[:, None] * np.cross(a, b))
    circum = np.linalg.norm(num, axis=1) / np.maximum(2.0 * vol6, 1e-300)
    return circum / (3.0 * np.maximum(inradius, 1e-300))


def _face_keys(tri, n_nodes):
    s = np.sort(tri, axis=1)
    return (s[:, 0] * n_nodes + s[:, 1]) * n_nodes + s[:, 2]


def boundary_orientation(coords, tets, faces):
    """``(n_into_fluid, n_away, n_orphan)`` for boundary triangles under SU2's rule."""
    n = len(coords)
    opposite = ((1, 2, 3, 0), (0, 2, 3, 1), (0, 1, 3, 2), (0, 1, 2, 3))
    keys, other = [], []
    for a, b, c, d in opposite:
        keys.append(_face_keys(tets[:, [a, b, c]], n))
        other.append(tets[:, d])
    keys, other = np.concatenate(keys), np.concatenate(other)
    order = np.argsort(keys)
    keys, other = keys[order], other[order]
    fk = _face_keys(faces, n)
    pos = np.searchsorted(keys, fk)
    pos = np.minimum(pos, len(keys) - 1)
    found = keys[pos] == fk
    domain = other[pos]
    p1, p2, p3, pd = coords[faces[:, 0]], coords[faces[:, 1]], coords[faces[:, 2]], coords[domain]
    vol = np.einsum('ij,ij->i', np.cross(p2 - p1, p3 - p1), pd - p1)
    good = found & (vol > 0.0)
    return int(good.sum()), int((found & ~(vol > 0.0)).sum()), int((~found).sum())


def edge_counts(tri):
    """``{(a, b): count}`` of undirected triangle edges, as arrays (edges, counts)."""
    e = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    e = np.sort(e, axis=1)
    edges, counts = np.unique(e, axis=0, return_counts=True)
    return edges, counts


# ------------------------------------------------------------------ gates

def measure(path, wall_markers, symmetry_marker, farfield_marker,
            farfield_center=None, symmetry_tol=1e-12):
    """Everything :func:`check` gates on, plus sizes for the report."""
    coords, volume, markers = read_su2(path)
    tets = volume.get(TET, np.zeros((0, 4), dtype=np.int64))
    prisms = volume.get(PRISM, np.zeros((0, 6), dtype=np.int64))
    out = {'mesh_path': path, 'n_nodes': int(len(coords)),
           'n_tets': int(len(tets)), 'n_prisms': int(len(prisms)),
           'other_volume_types': sorted(int(k) for k in volume if k not in (TET, PRISM)),
           'n_cells': int(sum(len(v) for v in volume.values())),
           'markers': {k: int(sum(len(v) for v in m.values())) for k, m in markers.items()}}
    vt = tet_volumes(coords, tets) if len(tets) else np.zeros(0)
    vp = prism_volumes(coords, prisms) if len(prisms) else np.zeros(0)
    vols = np.concatenate([vt, vp])
    out.update(min_volume=float(vols.min()), n_nonpositive=int((vols <= 0).sum()),
               total_volume=float(vols.sum()))
    if len(tets):
        dih = dihedral_angles(coords, tets)
        ar = aspect_ratios(coords, tets)
        out.update(min_dihedral_deg=float(dih.min()), max_dihedral_deg=float(dih.max()),
                   aspect_ratio_median=float(np.median(ar)),
                   aspect_ratio_p99=float(np.percentile(ar, 99)),
                   aspect_ratio_max=float(ar.max()))
    # boundary orientation: every triangle of every marker
    tris = [m[TRI] for m in markers.values() if TRI in m]
    if tris and len(tets):
        into, away, orphan = boundary_orientation(coords, tets, np.concatenate(tris))
        out.update(boundary_into_fluid=into, boundary_away=away, boundary_orphan=orphan)
    # closed wall
    wall = [markers[w][TRI] for w in wall_markers if w in markers and TRI in markers[w]]
    if wall:
        wall = np.concatenate(wall)
        edges, counts = edge_counts(wall)
        on_sym = np.all(np.abs(coords[edges][:, :, 1]) <= symmetry_tol, axis=1)
        out.update(wall_edges=int(len(edges)),
                   wall_edges_open=int(((counts != 2) & ~on_sym).sum()),
                   wall_edges_on_symmetry=int(on_sym.sum()),
                   wall_symmetry_edges_bad=int(((counts != 1) & on_sym).sum()),
                   wall_nodes=int(len(np.unique(wall))))
        wall_nodes = np.unique(wall)
        out.update(wall_y_max=float(coords[wall_nodes, 1].max()),
                   wall_x_range=[float(coords[wall_nodes, 0].min()),
                                 float(coords[wall_nodes, 0].max())])
    if symmetry_marker in markers:
        s = np.unique(np.concatenate([v.ravel() for v in markers[symmetry_marker].values()]))
        out['symmetry_max_abs_y'] = float(np.abs(coords[s, 1]).max())
    if farfield_marker in markers and farfield_center is not None:
        f = np.unique(np.concatenate([v.ravel() for v in markers[farfield_marker].values()]))
        out['farfield_bounds'] = [[float(coords[f, j].min()),
                                   float(coords[f, j].max())] for j in range(3)]
    return out


def check(measured, expected_markers, quality, symmetry_tol=1e-12):
    """``[(name, ok, detail)]`` for every gate. ``quality``: min/max dihedral, max aspect p99."""
    res = []

    def add(name, ok, detail):
        res.append((name, bool(ok), detail))

    got = set(measured['markers'])
    add('markers', got == set(expected_markers),
        f'{sorted(got)} (expected {sorted(expected_markers)})')
    add('tetrahedral', measured['other_volume_types'] == [] and measured['n_tets'] > 0,
        f"{measured['n_tets']} tets, {measured['n_prisms']} prisms, other VTK types "
        f"{measured['other_volume_types']}")
    add('positive_volumes', measured['n_nonpositive'] == 0,
        f"{measured['n_nonpositive']} non-positive of {measured['n_cells']}; min "
        f"{measured['min_volume']:.3e}")
    if 'boundary_into_fluid' in measured:
        add('boundary_orientation',
            measured['boundary_away'] == 0 and measured['boundary_orphan'] == 0,
            f"{measured['boundary_into_fluid']} into the fluid, {measured['boundary_away']} "
            f"away (SU2 would flip them), {measured['boundary_orphan']} with no volume cell")
    else:
        add('boundary_orientation', False, 'no boundary triangles or no tetrahedra')
    if 'wall_edges' in measured:
        add('closed_wall',
            measured['wall_edges_open'] == 0 and measured['wall_symmetry_edges_bad'] == 0,
            f"{measured['wall_edges']} edges: {measured['wall_edges_open']} not shared by "
            f"two faces, {measured['wall_edges_on_symmetry']} on the symmetry trace "
            f"({measured['wall_symmetry_edges_bad']} not shared by exactly one)")
    else:
        add('closed_wall', False, 'no wall triangles')
    if 'symmetry_max_abs_y' in measured:
        add('symmetry_plane', measured['symmetry_max_abs_y'] <= symmetry_tol,
            f"max |y| {measured['symmetry_max_abs_y']:.3e} (allowed {symmetry_tol:g})")
    else:
        add('symmetry_plane', False, 'no symmetry marker')
    if 'min_dihedral_deg' in measured:
        lo, hi = quality['min_dihedral_deg'], quality['max_dihedral_deg']
        add('dihedral', lo <= measured['min_dihedral_deg'] and measured['max_dihedral_deg'] <= hi,
            f"{measured['min_dihedral_deg']:.2f}..{measured['max_dihedral_deg']:.2f} deg "
            f"(allowed {lo:g}..{hi:g})")
        add('aspect_ratio', measured['aspect_ratio_p99'] <= quality['max_aspect_ratio_p99'],
            f"p99 {measured['aspect_ratio_p99']:.2f}, max {measured['aspect_ratio_max']:.2f}, "
            f"median {measured['aspect_ratio_median']:.2f} (p99 allowed "
            f"{quality['max_aspect_ratio_p99']:g})")
    return res


def report_lines(measured, results):
    lines = [f"=== 3D mesh admission: {measured['mesh_path']} ===",
             f"  {measured['n_nodes']} nodes, {measured['n_cells']} cells, "
             f"markers {measured['markers']}"]
    for name, ok, detail in results:
        lines.append(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    bad = sum(not ok for _, ok, _ in results)
    lines.append(f"  VERDICT: {'PASS' if not bad else f'FAIL ({bad})'}")
    return lines
