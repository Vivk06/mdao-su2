#!/usr/bin/env python
"""Convert .su2 embedded FFD box to Plot3D .xyz for pyGeo. See CSurfaceMovement.cpp ReadFFDInfo / CFreeFormDefBox SetControlPoints_Parallelepiped for format."""
import argparse
import os
import sys
from dataclasses import dataclass, field

import numpy as np

SPACINGS = ('uniform', 'cosine')


def _stations(n, lo, hi, spacing, le_te=None):
    if spacing == 'uniform':
        return np.linspace(lo, hi, n)
    if spacing == 'cosine':
        if le_te is None:
            raise ValueError('cosine spacing needs le_te = [x_le, x_te]; none is assumed')
        theta = np.linspace(0, np.pi, n - 2)
        interior = (1.0 - np.cos(theta)) / 2.0
        interior = le_te[0] + (le_te[1] - le_te[0]) * interior
        return np.concatenate([[lo], interior, [hi]])
    raise ValueError(f'spacing={spacing!r} must be one of {list(SPACINGS)}')


def airfoil_box(nx, spacing, zspan, x_bounds=None, y_bounds=None,
                x_margin=None, y_margin=None, coords=None, le_te=None):
    if x_margin is not None or y_margin is not None:
        if coords is None or x_margin is None or y_margin is None:
            raise ValueError('margins need x_margin, y_margin and the section coords')
        c = np.asarray(coords, dtype=float)
        x_bounds = (c[:, 0].min() - x_margin, c[:, 0].max() + x_margin)
        y_bounds = (c[:, 1].min() - y_margin, c[:, 1].max() + y_margin)
    if x_bounds is None or y_bounds is None:
        raise ValueError('airfoil_box needs x_bounds and y_bounds, or margins')
    if nx < 2:
        raise ValueError('nx must be >= 2')
    xline = _stations(nx, x_bounds[0], x_bounds[1], spacing, le_te)
    X = np.zeros((nx, 2, 2)); Y = np.zeros((nx, 2, 2)); Z = np.zeros((nx, 2, 2))
    for i in range(nx):
        for j in range(2):
            for k in range(2):
                X[i, j, k] = xline[i]
                Y[i, j, k] = y_bounds[0] if j == 0 else y_bounds[1]
                Z[i, j, k] = -zspan if k == 0 else zspan
    return X, Y, Z


def wing_box(nx, n_span, nz, x_bounds, y_bounds, z_bounds, spacing='uniform'):
    if min(nx, n_span, nz) < 2:
        raise ValueError('nx, n_span and nz must each be >= 2')
    if spacing != 'uniform':
        raise ValueError('wing_box supports spacing="uniform" only')
    xs = np.linspace(x_bounds[0], x_bounds[1], nx)
    ys = np.linspace(y_bounds[0], y_bounds[1], n_span)
    zs = np.linspace(z_bounds[0], z_bounds[1], nz)
    return np.meshgrid(xs, ys, zs, indexing='ij')


def write_lattice_plot3d(path, X, Y, Z):
    ni, nj, nk = X.shape
    with open(path, 'w') as f:
        f.write('1\n'); f.write(f'{ni} {nj} {nk}\n')
        for arr in (X, Y, Z):
            vals = [f'{arr[i, j, k]:.10f}'
                    for k in range(nk) for j in range(nj) for i in range(ni)]
            f.write(' '.join(vals) + '\n')
    return path


def read_lattice_plot3d(path):
    with open(path) as f:
        tokens = f.read().split()
    if int(tokens[0]) != 1:
        raise ValueError(f'{path}: {tokens[0]} blocks; one expected')
    ni, nj, nk = (int(t) for t in tokens[1:4])
    vals = np.array(tokens[4:], dtype=float); n = ni * nj * nk
    if len(vals) != 3 * n:
        raise ValueError(f'{path}: {len(vals)} values for a {ni}x{nj}x{nk} block')
    return tuple(vals[a*n:(a+1)*n].reshape((nk, nj, ni)).transpose(2, 1, 0)
                 for a in range(3))


def axis_x_fraction(x_bounds, x_axis):
    return (x_axis - x_bounds[0]) / (x_bounds[1] - x_bounds[0])

#: 2D boxes put their k planes at z = -/+ ZSPAN (SU2's 2D FFD convention,
#: CSurfaceMovement.cpp:4504/4519); ``[geometry] zspan`` must agree with it.
ZSPAN = 0.5

# Corner-check tolerance: same arithmetic in two places, so above round-off = real disagreement.
CORNER_RECON_TOL = 1e-12

# Sentinel for FFD_CONTROL_POINTS=0 (derive from corners). Distinct from None (missing key = truncated file).
_DERIVE = object()


@dataclass
class FFDBox:
    """One ``FFD_NBOX`` group read out of a ``.su2`` tail."""

    tag: str
    level: int
    ndim: int
    degree: tuple            # (dI, dJ, dK) -- dK is 1 for a 2D box
    blending: str
    corner_points: np.ndarray        # (8, 3), VTK hexahedron order
    control_points: np.ndarray       # (nI, nJ, nK, 3), SU2 index order
    surface_points: list = field(default_factory=list)  # (marker, index, u, v, w)
    # False when lattice was DERIVED from corners (FFD_CONTROL_POINTS=0); the
    # corner check then reports INAPPLICABLE.
    control_points_stored: bool = True

    @property
    def shape(self):
        """Control-point lattice size ``(nI, nJ, nK)``."""
        return tuple(d + 1 for d in self.degree)

    @property
    def n_control_points(self):
        n_i, n_j, n_k = self.shape
        return n_i * n_j * n_k


def recommended_kmax(box):
    """kmax for DVGeometry: order = nCtl, so pyGeo's B-spline basis is SU2's Bezier basis."""
    return int(max(box.shape))


# ── reading ────────────────────────────────────────────────

def _tail_lines(path):
    """Yield FFD tail lines starting at FFD_NBOX=. Streamed to avoid slurping large meshes."""
    with open(path) as fh:
        for line in fh:
            if line.startswith('FFD_NBOX='):
                yield line
                break
        else:
            raise ValueError(f'{path}: no FFD_NBOX block (mesh carries no embedded FFD)')
        yield from fh


def _kv(line):
    """Split ``KEY= value`` on the first ``=``. Returns (key, value-string)."""
    key, _, val = line.partition('=')
    return key.strip(), val.strip()


def read_ffd_boxes(path, with_surface_points=True):
    """Parse every FFD box embedded in the .su2 mesh."""
    lines = _tail_lines(path)

    key, val = _kv(next(lines))
    assert key == 'FFD_NBOX', key
    n_box = int(val)

    key, val = _kv(next(lines))
    if key != 'FFD_NLEVEL':
        raise ValueError(f'{path}: expected FFD_NLEVEL after FFD_NBOX, got {key!r}')

    boxes = []
    for _ in range(n_box):
        boxes.append(_read_one_box(lines, path, with_surface_points))
    return boxes


def _read_one_box(lines, path, with_surface_points):
    tag = level = blending = None
    degrees = {}
    corner_points = control_points = None
    surface_points = []
    ndim = 3

    for line in lines:
        line = line.strip()
        if not line:
            continue
        key, val = _kv(line)

        if key == 'FFD_TAG':
            tag = val
        elif key == 'FFD_LEVEL':
            level = int(val)
        elif key in ('FFD_DEGREE_I', 'FFD_DEGREE_J', 'FFD_DEGREE_K'):
            degrees[key[-1]] = int(val)
        elif key == 'FFD_BLENDING':
            blending = val
        elif key in ('FFD_PARENTS', 'FFD_CHILDREN'):
            for _ in range(int(val)):
                next(lines)
        elif key == 'FFD_CORNER_POINTS':
            n_corner = int(val)
            # 4 corners => 2D box; SU2 duplicates them to 8 at z = -/+0.5.
            ndim = 2 if n_corner == 4 else 3
            corner_points = _read_corner_points(lines, n_corner, ndim)
        elif key == 'FFD_CONTROL_POINTS':
            degrees.setdefault('K', 1)
            degree = (degrees['I'], degrees['J'], degrees['K'])
            if int(val) == 0:
                control_points = _DERIVE
            else:
                control_points = _read_control_points(lines, int(val), degree,
                                                      path, tag)
        elif key == 'FFD_SURFACE_POINTS':
            n_surf = int(val)
            if with_surface_points:
                surface_points = _read_surface_points(lines, n_surf)
            else:
                for _ in range(n_surf):
                    next(lines)
            break
        else:
            raise ValueError(f'{path}: unexpected key {key!r} in the FFD tail block')

    if corner_points is None:
        raise ValueError(f'{path}: FFD box {tag!r} is missing its corner points')
    if control_points is None:
        raise ValueError(f'{path}: FFD box {tag!r} declares no FFD_CONTROL_POINTS '
                         f'key at all (0 is legal and means "derive from the '
                         f'corners"; a missing key is a truncated tail block)')

    degrees.setdefault('K', 1)
    stored = control_points is not _DERIVE
    if not stored:
        shape = tuple(degrees[a] + 1 for a in 'IJK')
        control_points = corner_derived_control_points(corner_points, shape)

    return FFDBox(
        tag=tag,
        level=level,
        ndim=ndim,
        degree=(degrees['I'], degrees['J'], degrees['K']),
        blending=blending or 'BEZIER',   # SU2's default when the key is absent
        corner_points=corner_points,
        control_points=control_points,
        surface_points=surface_points,
        control_points_stored=stored,
    )


def _read_corner_points(lines, n_corner, ndim):
    """Return the eight corners in VTK hexahedron order, always as (8, 3)."""
    raw = []
    for _ in range(n_corner):
        raw.append([float(v) for v in next(lines).split()])
    raw = np.asarray(raw, dtype=float)

    if ndim == 3:
        return raw

    # 2D: 4 stored corners duplicated to 8 at z = -/+0.5 (CSurfaceMovement.cpp:4498-4513).
    lower = np.column_stack([raw, np.full(4, -ZSPAN)])
    upper = np.column_stack([raw, np.full(4, +ZSPAN)])
    return np.vstack([lower, upper])


def _read_control_points(lines, n_points, degree, path, tag):
    n_i, n_j, n_k = (d + 1 for d in degree)
    expected = n_i * n_j * n_k
    if n_points != expected:
        raise ValueError(
            f'{path}: FFD box {tag!r} declares {n_points} control points but '
            f'degree {degree} implies {n_i}x{n_j}x{n_k} = {expected}')

    cps = np.full((n_i, n_j, n_k, 3), np.nan)
    for _ in range(n_points):
        parts = next(lines).split()
        i, j, k = int(parts[0]), int(parts[1]), int(parts[2])
        cps[i, j, k] = [float(parts[3]), float(parts[4]), float(parts[5])]

    if np.isnan(cps).any():
        missing = int(np.isnan(cps[..., 0]).sum())
        raise ValueError(f'{path}: FFD box {tag!r} left {missing} lattice slots unset')
    return cps


def _read_surface_points(lines, n_points):
    out = []
    for _ in range(n_points):
        marker, idx, u, v, w = next(lines).split()
        out.append((marker, int(idx), float(u), float(v), float(w)))
    return out


# ── corner check: corner-derived vs stored control points ───

def corner_derived_control_points(corner_points, shape):
    """Trilinear interpolation of VTK hex corners (CFreeFormDefBox.cpp:200-208)
    for the corner check."""
    n_i, n_j, n_k = shape
    p = np.asarray(corner_points, dtype=float)

    # Single-plane axis: clamp parametric coord to 0 to avoid div-by-zero.
    u = np.linspace(0.0, 1.0, n_i) if n_i > 1 else np.zeros(1)
    v = np.linspace(0.0, 1.0, n_j) if n_j > 1 else np.zeros(1)
    w = np.linspace(0.0, 1.0, n_k) if n_k > 1 else np.zeros(1)
    U, V, W = np.meshgrid(u, v, w, indexing='ij')

    weights = [
        ((1 - U) * (1 - V) * (1 - W), p[0]),
        (U * (1 - V) * (1 - W), p[1]),
        (U * V * (1 - W), p[2]),
        ((1 - U) * V * (1 - W), p[3]),
        ((1 - U) * (1 - V) * W, p[4]),
        (U * (1 - V) * W, p[5]),
        (U * V * W, p[6]),
        ((1 - U) * V * W, p[7]),
    ]
    out = np.zeros((n_i, n_j, n_k, 3))
    for coeff, corner in weights:
        out += coeff[..., None] * corner
    return out


def check_corner_reconstruction(box, tol=CORNER_RECON_TOL):
    """Corner check: (ok, max_abs_error). Returns None error on derived box."""
    if not box.control_points_stored:
        return True, None
    derived = corner_derived_control_points(box.corner_points, box.shape)
    err = float(np.abs(derived - box.control_points).max())
    return err <= tol, err


# ── writing ────────────────────────────────────────────────

def write_plot3d(control_points, path):
    """Write lattice as ASCII Plot3D (single block, Fortran order) for pyBlock._readPlot3D."""
    cps = np.asarray(control_points, dtype=float)
    n_i, n_j, n_k = cps.shape[:3]
    with open(path, 'w') as fh:
        fh.write('1\n')
        fh.write(f'{n_i} {n_j} {n_k}\n')
        for idim in range(3):
            arr = cps[..., idim]
            vals = [f'{arr[i, j, k]:.16g}'
                    for k in range(n_k) for j in range(n_j) for i in range(n_i)]
            fh.write(' '.join(vals) + '\n')
    return path


def read_plot3d(path):
    """Read back a single-block Plot3D lattice as ``(ni, nj, nk, 3)``."""
    with open(path) as fh:
        tokens = fh.read().split()
    n_vol = int(tokens[0])
    if n_vol != 1:
        raise ValueError(f'{path}: expected 1 block, found {n_vol}')
    n_i, n_j, n_k = (int(t) for t in tokens[1:4])
    vals = np.asarray(tokens[4:4 + 3 * n_i * n_j * n_k], dtype=float)
    out = np.zeros((n_i, n_j, n_k, 3))
    per = n_i * n_j * n_k
    for idim in range(3):
        out[..., idim] = vals[idim * per:(idim + 1) * per].reshape(
            (n_i, n_j, n_k), order='F')
    return out


# ── CLI ────────────────────────────────────────────────────

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('mesh', help='.su2 mesh carrying an embedded FFD box')
    ap.add_argument('--out', default=None,
                    help='output Plot3D .xyz (default: <ffd_tag>.xyz beside the mesh)')
    ap.add_argument('--box', default=None,
                    help='FFD_TAG to convert when the mesh holds several')
    ap.add_argument('--verify', action='store_true',
                    help='run the corner check (corner-derived vs stored control points)')
    args = ap.parse_args(argv)

    boxes = read_ffd_boxes(args.mesh, with_surface_points=False)
    if args.box is not None:
        boxes = [b for b in boxes if b.tag == args.box]
        if not boxes:
            ap.error(f'no FFD box tagged {args.box!r} in {args.mesh}')
    if len(boxes) > 1:
        ap.error(f'{args.mesh} holds {len(boxes)} FFD boxes; pass --box <FFD_TAG>')
    box, = boxes

    n_i, n_j, n_k = box.shape
    print(f'FFD_TAG      : {box.tag}  (level {box.level}, {box.ndim}D)')
    print(f'blending     : {box.blending}')
    print(f'lattice      : {"STORED in the mesh" if box.control_points_stored else "DERIVED from the 8 corners (FFD_CONTROL_POINTS= 0)"}')
    print(f'degree       : {box.degree}  ->  lattice {n_i} x {n_j} x {n_k}'
          f' = {box.n_control_points} control points')
    print(f'pyGeo kmax   : {recommended_kmax(box)}   (order = nCtl, no interior knots)')
    lo = box.control_points.reshape(-1, 3).min(axis=0)
    hi = box.control_points.reshape(-1, 3).max(axis=0)
    print(f'bounds       : x [{lo[0]:.6g}, {hi[0]:.6g}]'
          f'  y [{lo[1]:.6g}, {hi[1]:.6g}]  z [{lo[2]:.6g}, {hi[2]:.6g}]')

    ok = True
    if args.verify:
        passed, err = check_corner_reconstruction(box)
        if err is None:
            print('corner check : INAPPLICABLE -- FFD_CONTROL_POINTS= 0, so the '
                  'lattice below IS the corner-derived one and there is nothing '
                  'independent to compare it against.')
            print('               Verify the box with the containment check instead '
                  '(run/mesh.py check-ffd --enclosure): every wall node must fall inside it.')
        else:
            print(f'corner check : corner-derived vs stored, max |err| = {err:.3e}'
                  f'  (tol {CORNER_RECON_TOL:.0e})  {"PASS" if passed else "FAIL"}')
        ok = ok and passed

    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.mesh)),
                                   f'{box.tag.lower()}.xyz')
    write_plot3d(box.control_points, out)
    print(f'wrote        : {out}')

    # Round-trip verification.
    err = float(np.abs(read_plot3d(out) - box.control_points).max())
    print(f'round-trip   : max |err| = {err:.3e}')
    ok = ok and err <= CORNER_RECON_TOL

    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
