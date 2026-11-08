#!/usr/bin/env python
"""Reproduce SU2's FFD_CONTINUITY / FFD_SYMMETRY_PLANE plane freeze (G11/G12)."""
import argparse
import sys

import numpy as np

from .ffd_lattice import read_ffd_boxes

#: Moller-Trumbore parallel-ray cutoff, ``CGeometry::RayIntersectsTriangle``.
RAY_EPSILON = 1e-6

#: Planes frozen inward from an intersecting face, per FFD_CONTINUITY level.
#: ``2ND_DERIVATIVE`` is SU2's default (CConfig.cpp:2865).
CONTINUITY_DEPTH = {
    'USER_INPUT': 0,
    'NO_DERIVATIVE': 1,
    '1ST_DERIVATIVE': 2,
    '2ND_DERIVATIVE': 3,
}

#: SU2 element type -> node count, for the surface elements a 3D wall carries.
SURFACE_ELEMENT_NODES = {5: 3, 9: 4}

#: Face triangles from CSurfaceMovement.cpp:1087-1146 — copied, not derived
#: (second triangle of each face is not the obvious complement).
FACE_TRIANGLES = {
    'i0': ([(0, 0, 0), (0, 0, 'N'), (0, 'M', 0)],
           [(0, 'M', 'N'), (0, 'M', 0), (0, 0, 'N')]),
    'i1': ([('L', 0, 0), ('L', 0, 'N'), ('L', 'M', 0)],
           [('L', 'M', 'N'), ('L', 'M', 0), ('L', 0, 'N')]),
    'j0': ([(0, 0, 0), (0, 0, 'N'), ('L', 0, 0)],
           [('L', 0, 'N'), ('L', 0, 0), (0, 0, 'N')]),
    'j1': ([(0, 'M', 0), (0, 'M', 'N'), ('L', 'M', 0)],
           [('L', 'M', 'N'), ('L', 'M', 0), (0, 'M', 'N')]),
    'k0': ([(0, 0, 0), (0, 'M', 0), ('L', 0, 0)],
           [('L', 'M', 0), ('L', 0, 0), (0, 'M', 0)]),
    'k1': ([(0, 0, 'N'), (0, 'M', 'N'), ('L', 0, 'N')],
           [('L', 'M', 'N'), ('L', 0, 'N'), (0, 'M', 'N')]),
}


# ── mesh reading — wall element edges ──────────────────────

def read_wall_edges(mesh_path, wall_markers):
    """Node coords keyed by point ID and unique wall element edges. Includes quad diagonals (SU2 tests all node pairs)."""
    from ..mesh.su2_mesh_file import read_points

    wanted = set(wall_markers)
    found = set()
    ndim = 3
    coords = None
    segments = set()

    with open(mesh_path) as fh:
        def skip(n):
            for _ in range(n):
                next(fh, None)

        for line in fh:
            line = line.strip()
            if line.startswith('NDIME='):
                ndim = int(line.split('=')[1])
            elif line.startswith('NPOIN='):
                coords = read_points(fh, int(line.split('=')[1].split()[0]), ndim)
            elif line.startswith('NELEM='):
                skip(int(line.split('=')[1].split()[0]))
            elif line.startswith('MARKER_TAG='):
                tag = line.split('=', 1)[1].strip()
                n = int(next(fh).split('=')[1])
                if tag not in wanted:
                    skip(n)
                    continue
                found.add(tag)
                for _ in range(n):
                    parts = [int(v) for v in next(fh).split()]
                    n_nodes = SURFACE_ELEMENT_NODES.get(parts[0])
                    if n_nodes is None:
                        raise ValueError(
                            f'{mesh_path}: marker {tag!r} carries element type '
                            f'{parts[0]}, which is not a 3D surface element')
                    nodes = parts[1:1 + n_nodes]
                    for a_i, a in enumerate(nodes):
                        for b in nodes[a_i + 1:]:
                            segments.add((min(a, b), max(a, b)))
            elif line.startswith('FFD_NBOX='):
                break

    if coords is None:
        raise ValueError(f'{mesh_path}: no NPOIN block')
    missing = wanted - found
    if missing:
        raise ValueError(f'{mesh_path}: markers {sorted(missing)} not found')
    return coords, np.array(sorted(segments))


# ── geometry — SU2's own intersection test, vectorized ─────

def segments_intersect_triangle(p0, p1, v0, v1, v2):
    """Vectorized Möller-Trumbore + SU2 segment containment check."""
    p0 = np.asarray(p0, float)
    p1 = np.asarray(p1, float)
    d = p1 - p0

    edge1 = v1 - v0
    edge2 = v2 - v0
    pvec = np.cross(d, edge2)
    det = pvec @ edge1

    ok = np.abs(det) >= RAY_EPSILON
    if not ok.any():
        return ok
    inv_det = np.zeros_like(det)
    inv_det[ok] = 1.0 / det[ok]

    tvec = p0 - v0
    u = inv_det * np.einsum('ij,ij->i', tvec, pvec)
    ok &= (u >= 0.0) & (u <= 1.0)
    if not ok.any():
        return ok

    qvec = np.cross(tvec, edge1)
    v = inv_det * np.einsum('ij,ij->i', d, qvec)
    ok &= (v >= 0.0) & (u + v <= 1.0)
    if not ok.any():
        return ok

    t = inv_det * (qvec @ edge2)
    hit = p0 + t[:, None] * d

    normal = np.cross(edge1, edge2)
    uu = hit - p0
    vv = hit - p1
    # SU2 rejects when the product is strictly positive (both on one side).
    ok &= ((uu @ normal) * (vv @ normal)) <= 0.0
    return ok


def _face_vertices(control_points, selector):
    n_i, n_j, n_k = control_points.shape[:3]
    last = {'L': n_i - 1, 'M': n_j - 1, 'N': n_k - 1}
    out = []
    for i, j, k in selector:
        out.append(control_points[last.get(i, i) if isinstance(i, str) else i,
                                  last.get(j, j) if isinstance(j, str) else j,
                                  last.get(k, k) if isinstance(k, str) else k])
    return out


def face_intersections(box, coords, segments, symmetry_plane=False):
    """Which box faces the design surface cuts. symmetry_plane skips J faces (CSurfaceMovement.cpp:1249/1261)."""
    p0 = coords[segments[:, 0]]
    p1 = coords[segments[:, 1]]

    out = {}
    for face, triangles in FACE_TRIANGLES.items():
        if symmetry_plane and face.startswith('j'):
            out[face] = False
            continue
        hit = False
        for selector in triangles:
            v0, v1, v2 = _face_vertices(box.control_points, selector)
            if segments_intersect_triangle(p0, p1, v0, v1, v2).any():
                hit = True
                break
        out[face] = hit
    return out


def fixed_planes(box, intersections, continuity='2ND_DERIVATIVE'):
    """Frozen lattice indices per axis. Only freezes where degree > 1 (CheckFFDIntersections L1428-1510)."""
    try:
        depth = CONTINUITY_DEPTH[continuity]
    except KeyError:
        raise ValueError(f'unknown FFD_CONTINUITY {continuity!r}; '
                         f'choose from {sorted(CONTINUITY_DEPTH)}')

    out = {}
    for axis, (degree, lo_face, hi_face) in enumerate(
            zip(box.degree, ('i0', 'j0', 'k0'), ('i1', 'j1', 'k1'))):
        frozen = set()
        if degree > 1:
            if intersections[lo_face]:
                frozen |= set(range(depth))
            if intersections[hi_face]:
                frozen |= set(range(degree, degree - depth, -1))
        out['ijk'[axis]] = sorted(frozen)
    return out


def compute(mesh_path, wall_markers, continuity='2ND_DERIVATIVE',
            symmetry_plane=False, box_tag=None):
    """``(box, intersections, frozen)`` for one mesh."""
    boxes = read_ffd_boxes(mesh_path, with_surface_points=False)
    if box_tag is not None:
        boxes = [b for b in boxes if b.tag == box_tag]
    if len(boxes) != 1:
        raise ValueError(f'{mesh_path}: expected exactly 1 FFD box, '
                         f'found {len(boxes)}')
    box, = boxes
    if box.ndim != 3:
        # Only the 3D intersection branch is reproduced; 2D boxes take a separate code path.
        # Under-reporting intersections silently frees CPs SU2 would hold.
        raise NotImplementedError(
            f'{mesh_path}: FFD box {box.tag!r} is 2D. This tool reproduces only '
            f"SU2's 3D intersection branch; the 2D one is a separate code path.")
    coords, segments = read_wall_edges(mesh_path, wall_markers)
    inter = face_intersections(box, coords, segments, symmetry_plane)
    return box, inter, fixed_planes(box, inter, continuity), len(segments)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('mesh')
    ap.add_argument('--wall', action='append', required=True,
                    help='design-surface marker (repeat for several)')
    ap.add_argument('--continuity', default='2ND_DERIVATIVE',
                    choices=sorted(CONTINUITY_DEPTH))
    ap.add_argument('--symmetry-plane', action='store_true',
                    help='FFD_SYMMETRY_PLANE= YES (skips the J-plane tests)')
    ap.add_argument('--box', default=None)
    args = ap.parse_args(argv)

    box, inter, frozen, n_seg = compute(
        args.mesh, args.wall, args.continuity, args.symmetry_plane, args.box)

    print(f'FFD_TAG            : {box.tag}   lattice {box.shape}')
    print(f'design markers     : {", ".join(args.wall)}  '
          f'({n_seg} unique element edges)')
    print(f'FFD_CONTINUITY     : {args.continuity} '
          f'(freeze depth {CONTINUITY_DEPTH[args.continuity]})')
    print(f'FFD_SYMMETRY_PLANE : {"YES -- J-plane tests skipped" if args.symmetry_plane else "NO"}')
    print('face intersections :')
    for face in ('i0', 'i1', 'j0', 'j1', 'k0', 'k1'):
        note = ''
        if args.symmetry_plane and face.startswith('j'):
            note = '   (not tested: FFD_SYMMETRY_PLANE)'
        print(f'    {face} : {"CUT" if inter[face] else "clear"}{note}')
    print('\nSU2 freezes these lattice planes -- put them in [dv.shape]:')
    for axis in 'ijk':
        print(f'    freeze_{axis} = {frozen[axis]}')
    n_free = 1
    for axis, n in zip('ijk', box.shape):
        n_free *= n - len(frozen[axis])
    print(f'\nfree control points: {n_free} of {box.n_control_points}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
