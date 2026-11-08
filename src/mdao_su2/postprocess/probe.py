"""Geometric probes: Plot Over Line (1D) and Slice/region (2D). No plotting imports."""
import csv
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .sample import CellLocator, sample_at
from ..vtu import read_su2_vtu, write_polydata, write_unstructured

__all__ = [
    'LineProbe', 'RegionProbe', 'ProbeError',
    'probe_line', 'probe_plane', 'probe_box',
    'write_vtp', 'write_vtu', 'write_csv',
]


class ProbeError(Exception):
    """Raised when a probe operation fails."""


@dataclass
class LineProbe:
    """Result of a line probe."""
    points: np.ndarray       # (n, 3)
    arc_length: np.ndarray   # (n,)
    fields: Dict[str, np.ndarray]
    n_missed: int
    n_total: int
    p0: np.ndarray
    p1: np.ndarray


@dataclass
class RegionProbe:
    """Result of a region probe (box or plane)."""
    points: np.ndarray       # (n, 3) or (ny, nx, 3) for structured
    fields: Dict[str, np.ndarray]
    n_missed: int
    n_total: int
    shape: Tuple[int, ...]   # grid shape for structured probes


def _ensure_3d(p):
    """Complete a 2-tuple to 3D with z=0."""
    p = np.asarray(p, dtype=float)
    if p.shape == (2,):
        p = np.array([p[0], p[1], 0.0])
    return p


def _validate_fields(source, field_names):
    """Validate field names against the source, return list."""
    available = list(source['point_data'].keys())
    if field_names is None:
        return available
    for name in field_names:
        if name not in source['point_data']:
            raise ProbeError(
                f'Field {name!r} not in source. '
                f'Available: {", ".join(available)}')
    return list(field_names)


def _detect_ndim(source):
    """Detect mesh dimensionality from the point coordinates."""
    pts = source['points']
    z_range = pts[:, 2].max() - pts[:, 2].min()
    if z_range < 1e-10:
        return 2
    return 3


def probe_line(source, p0, p1, n=200, fields=None):
    """Sample fields along a line from p0 to p1. Returns LineProbe."""
    if 'cells' not in source or not source.get('cells'):
        raise ProbeError(
            'probe_line requires a volume source with cell connectivity. '
            'For a surface mesh, use extract.section instead.')

    p0 = _ensure_3d(p0)
    p1 = _ensure_3d(p1)
    field_names = _validate_fields(source, fields)

    locator = CellLocator(source['points'], source['cells'],
                          source['cell_types'])

    t = np.linspace(0, 1, n)
    sample_pts = p0[None, :] + t[:, None] * (p1 - p0)[None, :]
    arc = np.linalg.norm(p1 - p0) * t

    results, n_missed = sample_at(source, locator, sample_pts, field_names)

    return LineProbe(
        points=sample_pts,
        arc_length=arc,
        fields=results,
        n_missed=n_missed,
        n_total=n,
        p0=p0,
        p1=p1,
    )


def probe_plane(source, origin, normal, bounds=None, n=(200, 200),
                fields=None):
    """Sample fields on a plane (3D only). Returns RegionProbe."""
    ndim = _detect_ndim(source)
    if ndim == 2:
        raise ProbeError(
            f'probe_plane is meaningless on a 2D mesh (ndim={ndim}). '
            f'Use probe_box(source, lo, hi) for a rectangular region '
            f'in the mesh plane.')

    if 'cells' not in source or not source.get('cells'):
        raise ProbeError('probe_plane requires a volume source.')

    origin = np.asarray(origin, dtype=float)
    normal = np.asarray(normal, dtype=float)
    normal = normal / np.linalg.norm(normal)

    # Build local coordinate frame on the plane
    if abs(normal[2]) < 0.9:
        u = np.cross(normal, np.array([0, 0, 1.0]))
    else:
        u = np.cross(normal, np.array([1, 0, 0.0]))
    u = u / np.linalg.norm(u)
    v = np.cross(normal, u)

    field_names = _validate_fields(source, fields)

    if bounds is None:
        # Project all mesh points onto the plane
        pts = source['points'].astype(float)
        proj_u = (pts - origin) @ u
        proj_v = (pts - origin) @ v
        bounds = ((proj_u.min(), proj_u.max()),
                  (proj_v.min(), proj_v.max()))

    n0, n1 = n
    s0 = np.linspace(bounds[0][0], bounds[0][1], n0)
    s1 = np.linspace(bounds[1][0], bounds[1][1], n1)
    ss0, ss1 = np.meshgrid(s0, s1, indexing='ij')
    sample_pts = (origin[None, None, :] +
                  ss0[:, :, None] * u[None, None, :] +
                  ss1[:, :, None] * v[None, None, :])
    flat_pts = sample_pts.reshape(-1, 3)

    locator = CellLocator(source['points'], source['cells'],
                          source['cell_types'])
    results, n_missed = sample_at(source, locator, flat_pts, field_names)

    # Reshape results to grid
    for name in results:
        arr = results[name]
        if arr.ndim == 2:
            results[name] = arr.reshape(n0, n1, arr.shape[1])
        else:
            results[name] = arr.reshape(n0, n1)

    return RegionProbe(
        points=sample_pts,
        fields=results,
        n_missed=n_missed,
        n_total=n0 * n1,
        shape=(n0, n1),
    )


def probe_box(source, lo=None, hi=None, n=(200, 200), fields=None):
    """Sample fields on a rectangular region in the mesh plane. Returns RegionProbe."""
    if 'cells' not in source or not source.get('cells'):
        raise ProbeError('probe_box requires a volume source.')

    pts = source['points'].astype(float)
    if lo is None:
        lo = (pts[:, 0].min(), pts[:, 1].min())
    if hi is None:
        hi = (pts[:, 0].max(), pts[:, 1].max())

    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)

    field_names = _validate_fields(source, fields)

    nx, ny = n
    x = np.linspace(lo[0], hi[0], nx)
    y = np.linspace(lo[1], hi[1], ny)
    xx, yy = np.meshgrid(x, y, indexing='ij')
    sample_pts = np.column_stack([xx.ravel(), yy.ravel(),
                                  np.zeros(nx * ny)])

    locator = CellLocator(source['points'], source['cells'],
                          source['cell_types'])
    results, n_missed = sample_at(source, locator, sample_pts, field_names)

    # Reshape
    grid_pts = np.column_stack([xx.ravel(), yy.ravel(),
                                np.zeros(nx * ny)]).reshape(nx, ny, 3)
    for name in results:
        arr = results[name]
        if arr.ndim == 2:
            results[name] = arr.reshape(nx, ny, arr.shape[1])
        else:
            results[name] = arr.reshape(nx, ny)

    return RegionProbe(
        points=grid_pts,
        fields=results,
        n_missed=n_missed,
        n_total=nx * ny,
        shape=(nx, ny),
    )


def write_vtp(probe, path):
    """Write a LineProbe as ASCII VTP (PolyData) for ParaView."""
    if not isinstance(probe, LineProbe):
        raise ProbeError('write_vtp requires a LineProbe')

    point_data = {}
    for name, arr in probe.fields.items():
        point_data[name] = arr

    conn = [np.arange(len(probe.points))]
    write_polydata(path, probe.points, conn, point_data=point_data)


def write_vtu(probe, path):
    """Write a RegionProbe as ASCII VTU (UnstructuredGrid) for ParaView."""
    if not isinstance(probe, RegionProbe):
        raise ProbeError('write_vtu requires a RegionProbe')

    flat_pts = probe.points.reshape(-1, 3)
    point_data = {}
    for name, arr in probe.fields.items():
        if arr.ndim == 3:
            point_data[name] = arr.reshape(-1, arr.shape[2])
        else:
            point_data[name] = arr.ravel()

    # Build quad cells from the structured grid
    nx, ny = probe.shape
    cells = []
    cell_types = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            c0 = i * ny + j
            c1 = (i + 1) * ny + j
            c2 = (i + 1) * ny + (j + 1)
            c3 = i * ny + (j + 1)
            cells.append(np.array([c0, c1, c2, c3]))
            cell_types.append(9)  # VTK_QUAD

    write_unstructured(path, flat_pts, cells, np.array(cell_types),
                       point_data=point_data)


def write_csv(probe, path):
    """Write probe data as CSV."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    if isinstance(probe, LineProbe):
        pts = probe.points
        header = ['x', 'y', 'z', 'arc_length']
        field_names = list(probe.fields.keys())
        # Expand vector field headers
        expanded_header = list(header)
        for name in field_names:
            arr = probe.fields[name]
            if arr.ndim == 2:
                for k in range(arr.shape[1]):
                    expanded_header.append(f'{name}_{k}')
            else:
                expanded_header.append(name)

        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(expanded_header)
            for i in range(len(pts)):
                row = [pts[i, 0], pts[i, 1], pts[i, 2], probe.arc_length[i]]
                for name in field_names:
                    arr = probe.fields[name]
                    if arr.ndim == 2:
                        for k in range(arr.shape[1]):
                            row.append(arr[i, k])
                    else:
                        row.append(arr[i])
                writer.writerow(row)

    elif isinstance(probe, RegionProbe):
        flat_pts = probe.points.reshape(-1, 3)
        header = ['x', 'y', 'z']
        field_names = list(probe.fields.keys())
        expanded_header = list(header)
        for name in field_names:
            arr = probe.fields[name]
            flat = arr.reshape(-1) if arr.ndim <= 2 else arr.reshape(-1, arr.shape[-1])
            if flat.ndim == 2:
                for k in range(flat.shape[1]):
                    expanded_header.append(f'{name}_{k}')
            else:
                expanded_header.append(name)

        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(expanded_header)
            for i in range(len(flat_pts)):
                row = [flat_pts[i, 0], flat_pts[i, 1], flat_pts[i, 2]]
                for name in field_names:
                    arr = probe.fields[name]
                    if arr.ndim == 3:
                        flat_arr = arr.reshape(-1, arr.shape[2])
                        for k in range(flat_arr.shape[1]):
                            row.append(flat_arr[i, k])
                    else:
                        row.append(arr.ravel()[i])
                writer.writerow(row)
