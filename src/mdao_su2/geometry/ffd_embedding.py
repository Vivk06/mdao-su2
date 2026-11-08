#!/usr/bin/env python
"""FFD box checks against the mesh.

The embedding check compares pyGeo's parametric coordinates of the wall nodes
with SU2's FFD_SURFACE_POINTS, which holds only when pyGeo's basis order equals
SU2's Bezier order (kmax = nCtl). The containment check proves only that every
wall node lies inside the box; it is the fallback for a mesh with no
FFD_SURFACE_POINTS block.
"""
import argparse
import os
import tempfile

import numpy as np

from .ffd_lattice import read_ffd_boxes, recommended_kmax, write_plot3d

# Embedding-check tolerance: compares two independent Newton inversions, not stored digits.
PARAM_TOL = 1e-6

#: Containment-check tolerance: pyGeo clamps inversion to the box, so exterior nodes show up
#: as residual (not u>1). 1e-9 is below geometric features, above convergence.
ENCLOSURE_TOL = 1e-9


def read_point_coordinates(path):
    """NPOIN coordinate array as (npoin, 3). Standalone — su2_mesh_file.py is 2D-only (G4)."""
    coords = None
    with open(path) as fh:
        for line in fh:
            if not line.startswith('NPOIN='):
                continue
            npoin = int(line.split('=')[1].split()[0])
            coords = np.zeros((npoin, 3))
            for i in range(npoin):
                parts = next(fh).split()
                # 2D: "x y [idx]" -> 2 or 3 tokens; 3D: "x y z [idx]" -> 3 or 4.
                ndim = 2 if len(parts) <= 3 else 3
                coords[i, :ndim] = [float(v) for v in parts[:ndim]]
                if len(parts) == ndim + 1 and int(parts[ndim]) != i:
                    raise ValueError(
                        f'{path}: NPOIN node {i} carries index {parts[ndim]}; '
                        'this reader assumes file order is index order')
            break
    if coords is None:
        raise ValueError(f'{path}: no NPOIN block')
    return coords


def check_enclosure(mesh_path, wall_markers, ffd_path=None, box_tag=None, kmax=None):
    """Containment check: every wall node embeds inside the box. Weaker than the
    embedding check; the fallback when the mesh has no FFD_SURFACE_POINTS."""
    from pygeo import DVGeometry

    boxes = read_ffd_boxes(mesh_path, with_surface_points=False)
    if box_tag is not None:
        boxes = [b for b in boxes if b.tag == box_tag]
    if len(boxes) != 1:
        raise ValueError(f'{mesh_path}: expected exactly 1 FFD box, found {len(boxes)}')
    box, = boxes

    tmpdir = None
    if ffd_path is None:
        tmpdir = tempfile.TemporaryDirectory()
        ffd_path = os.path.join(tmpdir.name, f'{box.tag.lower()}.xyz')
        write_plot3d(box.control_points, ffd_path)
    if kmax is None:
        kmax = recommended_kmax(box)

    coords = read_point_coordinates(mesh_path)
    idx = read_marker_nodes(mesh_path, wall_markers)
    pts = coords[idx]

    dvgeo = DVGeometry(ffd_path, kmax=kmax)
    dvgeo.addPointSet(pts, 'wall')
    emb = dvgeo.FFD.embeddedVolumes['wall']
    uvw = np.column_stack([emb.u.ravel(), emb.v.ravel(), emb.w.ravel()])

    if tmpdir is not None:
        tmpdir.cleanup()

    # Exterior nodes show as projection residuals (pyGeo clamps to box boundary).
    back = dvgeo.update('wall')
    resid = np.linalg.norm(back - pts, axis=1)
    return {
        'tag': box.tag,
        'kmax': kmax,
        'n_points': len(idx),
        'uvw_min': tuple(float(v) for v in uvw.min(axis=0)),
        'uvw_max': tuple(float(v) for v in uvw.max(axis=0)),
        'max_residual': float(resid.max()),
        'lattice_stored': box.control_points_stored,
        'ok': bool(resid.max() <= ENCLOSURE_TOL),
        'tol': ENCLOSURE_TOL,
    }


#: VTK type -> node count. Must use TYPE, not line length — some writers append
#: a trailing element index that would be misread as an extra node.
_MARKER_ELEM_NODES = {3: 2, 5: 3, 9: 4}


def read_marker_nodes(path, markers):
    """Unique node indices carried by the named markers of a ``.su2`` mesh."""
    want = set(markers)
    out = set()
    seen = set()
    with open(path) as fh:
        for line in fh:
            if not line.startswith('MARKER_TAG='):
                continue
            tag = line.split('=')[1].strip()
            seen.add(tag)
            n = int(next(fh).split('=')[1])
            for _ in range(n):
                parts = next(fh).split()
                if tag not in want:
                    continue
                etype = int(parts[0])
                try:
                    nn = _MARKER_ELEM_NODES[etype]
                except KeyError:
                    raise ValueError(
                        f'{path}: marker {tag!r} holds element type {etype}, '
                        f'which is not a surface element '
                        f'({sorted(_MARKER_ELEM_NODES)})') from None
                out.update(int(v) for v in parts[1:1 + nn])
    missing = sorted(want - seen)
    if missing:
        raise ValueError(f'{path}: no MARKER_TAG matched {missing}; '
                         f'the mesh holds {sorted(seen)}')
    return np.array(sorted(out))


def check_box(mesh_path, ffd_path=None, tol=PARAM_TOL, box_tag=None, kmax=None):
    """Embedding check: compare pyGeo parametric coords against FFD_SURFACE_POINTS."""
    from pygeo import DVGeometry

    boxes = read_ffd_boxes(mesh_path, with_surface_points=True)
    if box_tag is not None:
        boxes = [b for b in boxes if b.tag == box_tag]
    if len(boxes) != 1:
        raise ValueError(f'{mesh_path}: expected exactly 1 FFD box, found {len(boxes)}')
    box, = boxes

    if not box.surface_points:
        raise ValueError(f'{mesh_path}: FFD box {box.tag!r} has no FFD_SURFACE_POINTS block')

    tmpdir = None
    if ffd_path is None:
        tmpdir = tempfile.TemporaryDirectory()
        ffd_path = os.path.join(tmpdir.name, f'{box.tag.lower()}.xyz')
        write_plot3d(box.control_points, ffd_path)

    if kmax is None:
        kmax = recommended_kmax(box)

    coords = read_point_coordinates(mesh_path)
    idx = np.array([sp[1] for sp in box.surface_points])
    uvw_su2 = np.array([sp[2:] for sp in box.surface_points])
    pts = coords[idx]

    dvgeo = DVGeometry(ffd_path, kmax=kmax)
    dvgeo.addPointSet(pts, 'surface')
    emb = dvgeo.FFD.embeddedVolumes['surface']
    uvw_pygeo = np.column_stack([emb.u.ravel(), emb.v.ravel(), emb.w.ravel()])

    if tmpdir is not None:
        tmpdir.cleanup()

    err = np.abs(uvw_pygeo - uvw_su2)
    return {
        'tag': box.tag,
        'kmax': kmax,
        'order': tuple(min(kmax, n) for n in box.shape),
        'n_points': len(idx),
        'max_err': float(err.max()),
        'axis_max_err': tuple(float(e) for e in err.max(axis=0)),
        'ok': bool(err.max() <= tol),
        'tol': tol,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('mesh', help='.su2 mesh carrying an embedded FFD box')
    ap.add_argument('--ffd', default=None,
                    help='Plot3D lattice to check (default: convert from the mesh)')
    ap.add_argument('--box', default=None, help='FFD_TAG when the mesh holds several')
    ap.add_argument('--tol', type=float, default=PARAM_TOL)
    ap.add_argument('--kmax', type=int, default=None,
                    help='override pyBlock kmax (default: order-matched to nCtl)')
    ap.add_argument('--enclosure', metavar='MARKER', nargs='+', default=None,
                    help='run the containment check instead: every node of these wall '
                         'markers must embed inside the box. For a mesh with no '
                         'FFD_SURFACE_POINTS block, where the embedding check cannot run.')
    args = ap.parse_args(argv)

    if args.enclosure:
        r = check_enclosure(args.mesh, args.enclosure, ffd_path=args.ffd,
                            box_tag=args.box, kmax=args.kmax)
        print(f'FFD_TAG      : {r["tag"]}')
        print(f'lattice      : {"STORED in the mesh" if r["lattice_stored"] else "DERIVED from the 8 corners"}')
        print(f'basis        : kmax={r["kmax"]}')
        print(f'wall nodes   : {r["n_points"]}')
        print('u,v,w range  : ' + '  '.join(
            f'{a:.4f}..{b:.4f}' for a, b in zip(r['uvw_min'], r['uvw_max'])))
        print(f'containment  : max embed residual = {r["max_residual"]:.3e}'
              f'  (tol {r["tol"]:.0e})  {"PASS" if r["ok"] else "FAIL"}')
        print('               NOTE: the containment check is WEAKER than the '
              'embedding check -- it proves the '
              'surface is inside the box, not that the basis matches SU2.')
        return 0 if r['ok'] else 1

    r = check_box(args.mesh, ffd_path=args.ffd, tol=args.tol,
                  box_tag=args.box, kmax=args.kmax)
    print(f'FFD_TAG      : {r["tag"]}')
    print(f'basis        : kmax={r["kmax"]}  ->  spline order {r["order"]}')
    print(f'points       : {r["n_points"]}')
    print(f'max |du,dv,dw|: {r["axis_max_err"][0]:.3e}  {r["axis_max_err"][1]:.3e}'
          f'  {r["axis_max_err"][2]:.3e}')
    print(f'embedding    : max |err| = {r["max_err"]:.3e}  (tol {r["tol"]:.0e})'
          f'  {"PASS" if r["ok"] else "FAIL"}')
    return 0 if r['ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
