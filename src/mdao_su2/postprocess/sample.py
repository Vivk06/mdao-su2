"""Cell-based point locator and shape-function interpolation. Points in no cell return NaN."""
import numpy as np
from scipy.spatial import KDTree

__all__ = ['CellLocator', 'sample_at']

# VTK cell type codes this locator handles.
VTK_TRIANGLE = 5
VTK_QUAD = 9
VTK_TETRA = 10
VTK_HEXAHEDRON = 12
VTK_WEDGE = 13
VTK_PYRAMID = 14

_SUPPORTED = {VTK_TRIANGLE, VTK_QUAD, VTK_TETRA,
              VTK_HEXAHEDRON, VTK_WEDGE, VTK_PYRAMID}


def _tri_bary(p, v0, v1, v2):
    """Barycentric coordinates for point p in triangle v0-v1-v2 (2D or 3D)."""
    e0 = v1 - v0
    e1 = v2 - v0
    d = p - v0
    d00 = e0 @ e0
    d01 = e0 @ e1
    d11 = e1 @ e1
    d20 = d @ e0
    d21 = d @ e1
    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-30:
        return None
    u = (d11 * d20 - d01 * d21) / denom
    v = (d00 * d21 - d01 * d20) / denom
    w = 1.0 - u - v
    return np.array([w, u, v])


def _quad_bary(p, v0, v1, v2, v3):
    """Bilinear weights for point p in quad v0-v1-v2-v3, split into two triangles."""
    # Triangle 0-1-2
    b = _tri_bary(p, v0, v1, v2)
    if b is not None and b.min() >= -1e-8:
        return np.array([b[0], b[1], b[2], 0.0])
    # Triangle 0-2-3
    b = _tri_bary(p, v0, v2, v3)
    if b is not None and b.min() >= -1e-8:
        return np.array([b[0], 0.0, b[1], b[2]])
    return None


def _tetra_bary(p, v0, v1, v2, v3):
    """Barycentric coordinates in a tetrahedron."""
    mat = np.column_stack([v1 - v0, v2 - v0, v3 - v0])
    det = np.linalg.det(mat)
    if abs(det) < 1e-30:
        return None
    uvw = np.linalg.solve(mat, p - v0)
    w0 = 1.0 - uvw.sum()
    return np.array([w0, uvw[0], uvw[1], uvw[2]])


def _hex_bary(p, verts):
    """Weights for a hexahedron (8 vertices), split into 5 tets."""
    v = verts
    tets = [
        (v[0], v[1], v[3], v[4]),
        (v[1], v[2], v[3], v[6]),
        (v[1], v[4], v[5], v[6]),
        (v[3], v[4], v[6], v[7]),
        (v[1], v[3], v[4], v[6]),
    ]
    tet_idx_map = [
        [0, 1, 3, 4],
        [1, 2, 3, 6],
        [1, 4, 5, 6],
        [3, 4, 6, 7],
        [1, 3, 4, 6],
    ]
    for tet_verts, idx_map in zip(tets, tet_idx_map):
        b = _tetra_bary(p, *tet_verts)
        if b is not None and b.min() >= -1e-8:
            weights = np.zeros(8)
            for i, gi in enumerate(idx_map):
                weights[gi] += b[i]
            return weights
    return None


def _wedge_bary(p, verts):
    """Weights for a wedge/prism (6 vertices), split into 3 tets."""
    v = verts
    tets = [
        (v[0], v[1], v[2], v[3]),
        (v[1], v[2], v[3], v[4]),
        (v[2], v[3], v[4], v[5]),
    ]
    tet_idx_map = [
        [0, 1, 2, 3],
        [1, 2, 3, 4],
        [2, 3, 4, 5],
    ]
    for tet_verts, idx_map in zip(tets, tet_idx_map):
        b = _tetra_bary(p, *tet_verts)
        if b is not None and b.min() >= -1e-8:
            weights = np.zeros(6)
            for i, gi in enumerate(idx_map):
                weights[gi] += b[i]
            return weights
    return None


def _pyramid_bary(p, verts):
    """Weights for a pyramid (5 vertices), split into 2 tets."""
    v = verts
    tets = [
        (v[0], v[1], v[3], v[4]),
        (v[1], v[2], v[3], v[4]),
    ]
    tet_idx_map = [
        [0, 1, 3, 4],
        [1, 2, 3, 4],
    ]
    for tet_verts, idx_map in zip(tets, tet_idx_map):
        b = _tetra_bary(p, *tet_verts)
        if b is not None and b.min() >= -1e-8:
            weights = np.zeros(5)
            for i, gi in enumerate(idx_map):
                weights[gi] += b[i]
            return weights
    return None


class CellLocator:
    """KD-tree over cell centroids with point-in-cell shape functions."""

    def __init__(self, points, cells, cell_types):
        self.points = np.asarray(points, dtype=float)
        self.cells = cells
        self.cell_types = np.asarray(cell_types, dtype=int)
        self.ndim = self.points.shape[1]

        for ct in np.unique(self.cell_types):
            if int(ct) not in _SUPPORTED:
                raise ValueError(
                    f'Unsupported VTK cell type {ct}. '
                    f'Supported: {sorted(_SUPPORTED)}')

        centroids = np.array([
            self.points[np.asarray(c)].mean(axis=0) for c in self.cells
        ])
        self._tree = KDTree(centroids)
        self._cell_sizes = np.array([
            np.max(np.ptp(self.points[np.asarray(c)], axis=0))
            for c in self.cells
        ])

    def find(self, p):
        """Find the cell containing point p. Returns (cell_index, weights) or None."""
        p = np.asarray(p, dtype=float)
        # Search nearest centroids, expanding until we find a containing cell
        max_k = min(32, len(self.cells))
        dists, idxs = self._tree.query(p, k=max_k)
        if max_k == 1:
            dists = [dists]
            idxs = [idxs]

        for dist, ci in zip(dists, idxs):
            ct = int(self.cell_types[ci])
            cell = np.asarray(self.cells[ci])
            verts = self.points[cell]
            tol = self._cell_sizes[ci] * 1e-6

            w = self._weights(p, verts, ct)
            if w is not None and w.min() >= -tol:
                return ci, np.maximum(w, 0.0)
        return None

    def _weights(self, p, verts, ct):
        if ct == VTK_TRIANGLE:
            return _tri_bary(p, verts[0], verts[1], verts[2])
        elif ct == VTK_QUAD:
            return _quad_bary(p, verts[0], verts[1], verts[2], verts[3])
        elif ct == VTK_TETRA:
            return _tetra_bary(p, verts[0], verts[1], verts[2], verts[3])
        elif ct == VTK_HEXAHEDRON:
            return _hex_bary(p, verts)
        elif ct == VTK_WEDGE:
            return _wedge_bary(p, verts)
        elif ct == VTK_PYRAMID:
            return _pyramid_bary(p, verts)
        return None


def sample_at(source, locator, pts, field_names=None):
    """Sample field(s) at arbitrary coordinates. Returns ({name: array}, n_missed)."""
    pts = np.asarray(pts, dtype=float)
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)

    pd = source['point_data']
    if field_names is None:
        field_names = list(pd.keys())
    else:
        for name in field_names:
            if name not in pd:
                raise ValueError(
                    f'Field {name!r} not in source. '
                    f'Available: {", ".join(pd.keys())}')

    results = {}
    for name in field_names:
        arr = np.asarray(pd[name], dtype=float)
        if arr.ndim == 2:
            results[name] = np.full((len(pts), arr.shape[1]), np.nan)
        else:
            results[name] = np.full(len(pts), np.nan)

    n_missed = 0
    for i, p in enumerate(pts):
        hit = locator.find(p)
        if hit is None:
            n_missed += 1
            continue
        ci, weights = hit
        cell = np.asarray(locator.cells[ci])
        for name in field_names:
            arr = np.asarray(pd[name], dtype=float)
            vals = arr[cell]
            if vals.ndim == 2:
                results[name][i] = weights @ vals
            else:
                results[name][i] = weights @ vals

    return results, n_missed
