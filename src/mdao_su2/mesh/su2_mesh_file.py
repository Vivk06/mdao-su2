#!/usr/bin/env python
"""su2_mesh_file.py — SU2 mesh file (.su2) parsing."""

import os
import tempfile
from collections import Counter

import numpy as np


def read_mesh_header(mesh_path):
    """Dimensionality, sizes and marker inventory via header-only streaming (no full parse)."""
    header = {'ndim': None, 'npoin': None, 'nelem': None, 'markers': {}}
    with open(mesh_path) as fh:
        def skip(n):
            for _ in range(n):
                next(fh, None)

        for line in fh:
            line = line.strip()
            if line.startswith('NDIME='):
                header['ndim'] = int(line.split('=')[1])
            elif line.startswith('NPOIN='):
                # NPOIN can carry trailing tokens in some writers; take the first.
                header['npoin'] = int(line.split('=')[1].split()[0])
                skip(header['npoin'])
            elif line.startswith('NELEM='):
                header['nelem'] = int(line.split('=')[1].split()[0])
                skip(header['nelem'])
            elif line.startswith('MARKER_TAG='):
                # Names are transcribed EXACTLY -- ONERA M6 ships a marker
                # spelled YNORMAL_FACE while its siblings are plural, and the
                tag = line.split('=', 1)[1].strip()
                nxt = next(fh, '')
                n = int(nxt.split('=')[1]) if 'MARKER_ELEMS' in nxt else 0
                header['markers'][tag] = n
                skip(n)
            elif line.startswith('FFD_NBOX='):
                break        # the FFD tail: nothing past here is mesh
    return header


def read_points(fh, npoin, ndim):
    """``(n, 3)`` coordinates of the next ``npoin`` rows of ``fh``, keyed by point ID.

    Each row's last token, the point ID, is its index; a 2D row may
    carry a zero z column before it, and a row without an ID takes its row
    index. IDs a non-contiguous mesh skips stay NaN, and 2D rows get
    z = 0.
    """
    ids = np.empty(npoin, dtype=int)
    xyz = np.zeros((npoin, 3))
    for row in range(npoin):
        parts = next(fh).split()
        xyz[row, :ndim] = [float(v) for v in parts[:ndim]]
        ids[row] = int(parts[-1]) if len(parts) > ndim else row
    points = np.full((int(ids.max()) + 1 if npoin else 0, 3), np.nan)
    points[ids] = xyz
    return points


def read_wall_surface(mesh_path, wall_markers):
    """``(points (n,3), faces list[tuple[int]])`` for the named markers. Streamed, skips volume elements.

    Faces are 2D lines (VTK 3), triangles (5) or quadrilaterals (9).
    """
    wanted = set(wall_markers)
    ndim = 3
    points = None
    faces = []
    found = set()
    with open(mesh_path) as fh:
        def skip(n):
            for _ in range(n):
                next(fh, None)

        for line in fh:
            line = line.strip()
            if line.startswith('NDIME='):
                ndim = int(line.split('=')[1])
            elif line.startswith('NPOIN='):
                points = read_points(fh, int(line.split('=')[1].split()[0]), ndim)
            elif line.startswith('NELEM='):
                skip(int(line.split('=')[1].split()[0]))
            elif line.startswith('MARKER_TAG='):
                tag = line.split('=', 1)[1].strip()
                nxt = next(fh, '')
                n = int(nxt.split('=')[1]) if 'MARKER_ELEMS' in nxt else 0
                if tag in wanted:
                    found.add(tag)
                    for _ in range(n):
                        parts = next(fh).split()
                        # col 0 is the VTK element type (5 tri, 9 quad); the
                        # node ids follow.  Trailing element-id columns exist in
                        etype = int(parts[0])
                        nn = {3: 2, 5: 3, 9: 4}.get(etype)
                        if nn is None:
                            raise ValueError(
                                f'{tag}: unsupported surface element type {etype}')
                        faces.append(tuple(int(p) for p in parts[1:1 + nn]))
                else:
                    skip(n)
            elif line.startswith('FFD_NBOX='):
                break
    missing = wanted - found
    if missing:
        raise KeyError(f'markers not in {mesh_path}: {sorted(missing)}')
    return points, faces


def triangulate_faces(points, faces):
    """``[p0, p1, p2]`` corner arrays for pyGeo ``surfFormat='point-point'``. Must be list, not tuple (DVCon constraint)."""
    p0, p1, p2 = [], [], []
    for face in faces:
        for t in range(1, len(face) - 1):
            p0.append(points[face[0]])
            p1.append(points[face[t]])
            p2.append(points[face[t + 1]])
    return [np.array(p0), np.array(p1), np.array(p2)]


def parse_su2_mesh(mesh_path):
    """Parse an SU2 mesh file. Returns (points, elements, markers).

    ``points`` is ``{gid: (x, y)}`` for 2D or ``{gid: (x, y, z)}`` for 3D.
    ``elements`` is ``[(vtk_type, node0, node1, ..., eid), ...]``.
    ``markers``  is ``{tag: [(vtk_type, node0, ..., eid), ...]}``.
    """
    with open(mesh_path) as f:
        lines = f.readlines()

    ndim = 2
    points = {}
    elements = []
    markers = {}
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('NDIME='):
            ndim = int(line.split('=')[1])
            i += 1
        elif line.startswith('NPOIN='):
            npoin = int(line.split('=')[1].split()[0])
            for j in range(npoin):
                parts = lines[i + 1 + j].split()
                gid = int(parts[-1])
                if ndim == 3:
                    points[gid] = (float(parts[0]), float(parts[1]),
                                   float(parts[2]))
                else:
                    points[gid] = (float(parts[0]), float(parts[1]))
            i += 1 + npoin
        elif line.startswith('NELEM='):
            nelem = int(line.split('=')[1].split()[0])
            for j in range(nelem):
                parts = lines[i + 1 + j].split()
                elements.append(tuple(int(p) for p in parts))
            i += 1 + nelem
        elif line.startswith('MARKER_TAG='):
            tag = line.split('=')[1].strip()
            nme = int(lines[i + 1].split('=')[1])
            elems = []
            for j in range(nme):
                parts = lines[i + 2 + j].split()
                elems.append(tuple(int(p) for p in parts))
            markers[tag] = elems
            i += 2 + nme
        else:
            i += 1
    return points, elements, markers


def collect_line_elements(markers, marker_name):
    """Line elements of one marker or of several joined (``str`` or sequence)."""
    names = [marker_name] if isinstance(marker_name, str) else list(marker_name)
    missing = [n for n in names if n not in markers]
    if missing:
        raise KeyError(f'marker(s) {missing} not in mesh; it has {sorted(markers)}')
    return [e for n in names for e in markers[n]]


def get_loop_order(mesh_path, marker_name):
    """Derive the airfoil loop traversal order. Returns (gid_sorted, loop_perm).

    ``marker_name`` is one marker or a sequence of markers that together form
    one closed loop (for example an upper and a lower surface); no marker is
    assumed.
    """
    points, _, markers = parse_su2_mesh(mesh_path)
    elems = collect_line_elements(markers, marker_name)

    # build adjacency (line elements: type=3, node_a, node_b)
    nodes = set()
    adj = {}
    for e in elems:
        a, b = e[1], e[2]
        nodes.add(a)
        nodes.add(b)
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)

    gid_sorted = sorted(nodes)
    gid_to_pos = {g: i for i, g in enumerate(gid_sorted)}

    # walk the loop starting from the TE node (highest x)
    coords = {g: points[g] for g in nodes}
    start = max(nodes, key=lambda g: coords[g][0])
    visited = [start]
    prev = None
    current = start
    for _ in range(len(nodes)):
        nbrs = [n for n in adj[current] if n != prev]
        if not nbrs or nbrs[0] == start:
            break
        nxt = nbrs[0]
        visited.append(nxt)
        prev = current
        current = nxt

    # choose the direction that goes to upper surface first (positive y)
    if len(visited) > 1 and coords[visited[1]][1] < 0:
        nbrs0 = [n for n in adj[start] if n != visited[1]]
        if nbrs0:
            visited = [start]
            prev = None
            current = start
            nxt = nbrs0[0]
            visited.append(nxt)
            prev = current
            current = nxt
            for _ in range(len(nodes) - 2):
                nbrs = [n for n in adj[current] if n != prev]
                if not nbrs or nbrs[0] == start:
                    break
                nxt = nbrs[0]
                visited.append(nxt)
                prev = current
                current = nxt

    loop_perm = [gid_to_pos[g] for g in visited]
    return gid_sorted, loop_perm


def extract_baseline_surface(mesh_path, marker_name):
    """Extract baseline surface coordinates in gid-sorted order (x_aero layout).

    ``marker_name`` is one marker or a sequence of markers; none is assumed.
    """
    points, _, markers = parse_su2_mesh(mesh_path)
    nodes = set()
    for e in collect_line_elements(markers, marker_name):
        nodes.add(e[1])
        nodes.add(e[2])
    gid_sorted = sorted(nodes)
    return np.array([points[g] for g in gid_sorted])


# VTK element type → number of nodes (volume + surface types used by SU2)
_VTK_NODES = {
    3: 2, 5: 3, 9: 4,      # line, triangle, quad
    10: 4, 12: 8, 13: 6, 14: 5,  # tet, hex, wedge, pyramid
}


def write_deformed_su2_from_vtu(reference_path, vtu_path, output_path):
    """Write a deformed SU2 mesh using a flow VTU's point coordinates.

    The reference supplies topology, marker metadata, and explicit point IDs.
    SU2's VTU writer keeps the mesh-row ordering but does not export those IDs,
    so this function replaces coordinates row by row and preserves every token
    after the coordinate columns.
    """
    from ..vtu import read_su2_vtu

    with open(reference_path) as stream:
        lines = stream.readlines()
    data = read_su2_vtu(vtu_path)
    coordinates = np.asarray(data['points'])

    ndim = None
    point_start = None
    point_count = None
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('NDIME='):
            ndim = int(stripped.split('=', 1)[1])
        elif stripped.startswith('NPOIN='):
            if point_start is not None:
                raise ValueError(f'{reference_path}: multiple NPOIN sections are unsupported')
            point_count = int(stripped.split('=', 1)[1].split()[0])
            point_start = index + 1

    if ndim not in (2, 3) or point_start is None:
        raise ValueError(f'{reference_path}: missing valid NDIME/NPOIN header')
    if coordinates.ndim != 2 or coordinates.shape[0] != point_count \
            or coordinates.shape[1] < ndim:
        raise ValueError(
            f'mesh point mismatch: {reference_path} has {point_count} {ndim}D points; '
            f'{vtu_path} has shape {coordinates.shape}')

    rewritten = list(lines)
    point_ids = set()
    point_rows = {}
    for row in range(point_count):
        index = point_start + row
        tokens = lines[index].split()
        if len(tokens) < ndim + 1:
            raise ValueError(
                f'{reference_path}: point row {row} lacks an explicit point ID')
        # The ID is the last token: some 2D meshes (RAE 2822) carry a zero
        # third coordinate column before it, as parse_su2_mesh also assumes.
        try:
            point_id = int(tokens[-1])
        except ValueError as exc:
            raise ValueError(
                f'{reference_path}: invalid point ID on row {row}') from exc
        if point_id in point_ids:
            raise ValueError(f'{reference_path}: duplicate point ID {point_id}')
        point_ids.add(point_id)
        point_rows[point_id] = row
        coords = ' '.join(f'{value:.17g}' for value in coordinates[row, :ndim])
        rewritten[index] = f'{coords} {" ".join(tokens[ndim:])}\n'

    _points, elements, _markers = parse_su2_mesh(reference_path)
    reference_cells = Counter()
    for element in elements:
        vtk_type = element[0]
        node_count = _VTK_NODES.get(vtk_type)
        if node_count is None:
            raise ValueError(
                f'{reference_path}: unsupported element type {vtk_type}')
        node_rows = tuple(sorted(point_rows[node]
                                 for node in element[1:1 + node_count]))
        reference_cells[(vtk_type, node_rows)] += 1
    if 'cells' not in data or 'cell_types' not in data:
        raise ValueError(f'{vtu_path}: volume topology is unavailable')
    vtu_cells = Counter(
        (int(vtk_type), tuple(sorted(int(node) for node in cell)))
        for vtk_type, cell in zip(data['cell_types'], data['cells']))
    if vtu_cells != reference_cells:
        raise ValueError(
            f'mesh topology mismatch between {reference_path} and {vtu_path}')

    output_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_dir, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile('w', dir=output_dir, delete=False) as stream:
            temporary = stream.name
            stream.writelines(rewritten)
        os.replace(temporary, output_path)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.remove(temporary)


def _vtk_cells(elements, index, what):
    """``(cells, cell_types)`` with node ids mapped through ``index``; an element type
    outside ``_VTK_NODES`` raises."""
    cells, cell_types = [], []
    for e in elements:
        nn = _VTK_NODES.get(e[0])
        if nn is None:
            raise ValueError(f'{what}: unsupported element type {e[0]}')
        cells.append(np.array([index[g] for g in e[1:1 + nn]], dtype=int))
        cell_types.append(e[0])
    return cells, np.array(cell_types, dtype=int)


def su2_mesh_to_vtu(su2_path, vtu_path, boundary_path=None, markers=None):
    """Convert an SU2 mesh to a mesh-only VTU (no flow field) for ParaView.

    Point ids are mapped to contiguous VTU indices in id order. With
    ``boundary_path`` a second VTU holds the marker elements of ``markers`` (every
    marker when ``None``) on the same point array. Each boundary cell carries an
    integer ``marker`` cell array, the marker's position in the header, and the
    legend is written as FieldData: one Int32 array per marker tag, named by the
    tag and holding that position.

    Returns ``{'ndim', 'n_points', 'cells': {vtk_type: n}, 'markers': [{'tag', 'index',
    'n_elements', 'written'}]}``.
    """
    from ..vtu import write_unstructured

    points, elements, mesh_markers = parse_su2_mesh(su2_path)
    ids = sorted(points)
    index = {g: i for i, g in enumerate(ids)}
    pts = np.array([points[g] for g in ids])

    cells, cell_types = _vtk_cells(elements, index, su2_path)
    write_unstructured(vtu_path, pts, cells, cell_types)

    tags = list(mesh_markers)                    # header order
    wanted = tags if markers is None else list(markers)
    unknown = [t for t in wanted if t not in mesh_markers]
    if unknown:
        raise KeyError(f'markers {unknown} not in {su2_path}; it has {tags}')
    if boundary_path:
        b_cells, b_types, b_marker = [], [], []
        for tag in wanted:
            c, t = _vtk_cells(mesh_markers[tag], index, f'{su2_path} marker {tag}')
            b_cells += c
            b_types.append(t)
            b_marker += [tags.index(tag)] * len(c)
        write_unstructured(
            boundary_path, pts, b_cells,
            np.concatenate(b_types) if b_types else np.zeros(0, dtype=int),
            cell_data={'marker': np.array(b_marker, dtype=np.int32)},
            field_data={tag: np.array([i], dtype=np.int32) for i, tag in enumerate(tags)})
    return {'ndim': len(next(iter(points.values()))), 'n_points': len(ids),
            'cells': dict(sorted(Counter(cell_types.tolist()).items())),
            'markers': [{'tag': t, 'index': i, 'n_elements': len(mesh_markers[t]),
                         'written': bool(boundary_path) and t in wanted}
                        for i, t in enumerate(tags)]}
