"""Small, dependency-free geometry exports used by run snapshots."""

import csv
import os

import numpy as np

from ..vtu import write_unstructured
from ..vtu import read_su2_vtu, triangles
from ..geometry.ffd import shape_functions
from ..postprocess.span_load import section_cut


def _load_lattice(path):
    """Plot3D reader matching postprocess.dvmap.load_lattice's i-fastest convention."""
    with open(path) as fh:
        tok = fh.read().split()
    ni, nj, nk = map(int, tok[1:4])
    vals, n = np.asarray(tok[4:], dtype=float), ni * nj * nk
    out = np.empty((ni, nj, nk, 3))
    for axis in range(3):
        out[..., axis] = vals[axis * n:(axis + 1) * n].reshape(
            (ni, nj, nk), order='F')
    return out


def compute_bump_locations(n_dv, distribution):
    i = np.arange(1, n_dv + 1, dtype=float)
    if distribution == 'even':
        return i / (n_dv + 1.0)
    if distribution == '1-cos':
        return 0.5 * (1.0 - np.cos(np.pi * i / (n_dv + 1.0)))
    raise ValueError(f'Unknown Hicks-Henne distribution {distribution!r}')


def _lines_for_lattice(shape):
    lines = []
    for axis in range(3):
        for index in np.ndindex(shape):
            if index[axis] + 1 < shape[axis]:
                other = list(index)
                other[axis] += 1
                a = np.ravel_multi_index(index, shape)
                b = np.ravel_multi_index(tuple(other), shape)
                lines.append([a, b])
    return lines


def ffd_lattice_pair(cfg, shape_dvs):
    """Return baseline and shape-function-deformed FFD control lattices."""
    base = _load_lattice(cfg.ffd_path)
    deformed = base.copy()
    dvs = np.asarray(shape_dvs if shape_dvs is not None else [], dtype=float)
    if not cfg.dv.shape.enabled or dvs.size == 0:
        # Twist moves the lattice through pyGeo's reference axis, not through
        # shape functions; the deformed lattice here shows shape DVs only.
        return base, deformed
    for value, function in zip(dvs, shape_functions(cfg.dv.shape, base.shape)):
        for flat_index, direction in function.items():
            ijk = np.unravel_index(flat_index, base.shape[:3])
            deformed[ijk] += value * np.asarray(direction)
    return base, deformed


def write_lattice_vtu(path, lattice):
    """Write an FFD lattice ``(ni, nj, nk, 3)`` as VTK line cells along i, j and k."""
    _write_lines(path, lattice, _lines_for_lattice(lattice.shape[:3]))
    return path


def _write_lines(path, points, lines, point_data=None):
    write_unstructured(path, np.asarray(points).reshape(-1, 3), lines,
                       np.full(len(lines), 3, dtype=int), point_data=point_data)


def _surface_branches(surface):
    points = np.asarray(surface, dtype=float).reshape(-1, surface.shape[-1])
    if points.shape[1] == 2:
        points = np.column_stack([points, np.zeros(len(points))])
    x = points[:, 0]
    lo, hi = float(np.min(x)), float(np.max(x))
    mid = np.interp(x, [lo, hi], [np.mean(points[np.isclose(x, lo), 1]),
                                  np.mean(points[np.isclose(x, hi), 1])])
    return points, points[:, 1] >= mid


def write_parameterization_artifacts(cfg, geometry_dir, surface, shape_dvs,
                                     baseline_surface=None):
    """Write this design's control geometry, or a labelled visualization guide.

    One file per evaluation folder, drawn at that folder's design: the
    baseline folder holds the baseline indicator, every other folder its own.
    Returns the file name, or None.
    """
    param = cfg.geometry.parameterization
    if param == 'ffd':
        _base, deformed = ffd_lattice_pair(cfg, shape_dvs)
        write_lattice_vtu(os.path.join(geometry_dir, 'ffd_box.vtu'), deformed)
        return 'ffd_box.vtu'

    reference = baseline_surface if baseline_surface is not None else surface
    ref, upper = _surface_branches(reference)
    cur, _ = _surface_branches(surface)
    if param == 'cst':
        # Stations identify coefficient positions on this design's surface;
        # they are a guide, not a control mesh.
        order = np.argsort(ref[:, 0])
        lines = [[int(order[i]), int(order[i + 1])] for i in range(len(order)-1)]
        station = (ref[:, 0] - ref[:, 0].min()) / max(np.ptp(ref[:, 0]), 1e-30)
        data = {'coefficient_station': station, 'surface_side': upper.astype(int)}
        _write_lines(os.path.join(geometry_dir, 'cst_stations.vtu'), cur, lines, data)
        return 'cst_stations.vtu'

    # Each bump is a segment at its station from the baseline wall to this
    # design's wall, so a deformed design's file differs from the baseline's.
    # Both ends were once taken from the reference surface, which made every
    # deformed indicator a copy of the baseline one.
    loc = compute_bump_locations(cfg.dv.shape.n_dv, cfg.dv.shape.distribution)
    x0, chord = float(ref[:, 0].min()), max(float(np.ptp(ref[:, 0])), 1e-30)
    guide_points, lines, dv_index, side, width = [], [], [], [], []
    for is_upper, label in ((True, 1), (False, 0)):
        if not is_upper and len(shape_dvs) == cfg.dv.shape.n_dv:
            continue
        branch = upper if is_upper else ~upper
        for i, loc_i in enumerate(loc):
            x = x0 + loc_i * chord
            y_ref = np.interp(x, ref[branch, 0], ref[branch, 1])
            y_cur = np.interp(x, cur[branch, 0], cur[branch, 1])
            a = len(guide_points)
            guide_points.extend([[x, y_ref, 0.0], [x, y_cur, 0.0]])
            lines.append([a, a + 1])
            dv_index.extend([i, i])
            side.extend([label, label])
            width.extend([cfg.dv.shape.bump_width] * 2)
    pdata = {'dv_index': np.asarray(dv_index), 'surface_side': np.asarray(side),
             'bump_width': np.asarray(width)}
    _write_lines(os.path.join(geometry_dir, 'hicks_henne_bumps.vtu'),
                 guide_points, lines, pdata)
    return 'hicks_henne_bumps.vtu'


def surface_metrics(surface):
    """Deterministic 2-D surface metrics; 3-D-only quantities remain null."""
    if surface is None:
        return {'chord': None, 'area': None, 'maximum_thickness': None,
                'thickness_to_chord': None}
    p, upper = _surface_branches(surface)
    if np.ptp(p[:, 2]) > 1e-12:
        return {'chord': float(np.ptp(p[:, 0])), 'area': None,
                'maximum_thickness': None, 'thickness_to_chord': None}
    if upper.all() or not upper.any():
        # A half-model wall has one branch; the symmetry plane is not on it.
        return {'chord': float(np.ptp(p[:, 0])), 'area': None,
                'maximum_thickness': None, 'thickness_to_chord': None}
    x = np.unique(p[:, 0]); up = np.interp(x, p[upper, 0], p[upper, 1]); lo = np.interp(x, p[~upper, 0], p[~upper, 1])
    chord = float(np.ptp(x)); thickness = up - lo
    return {'chord': chord, 'area': float(np.trapezoid(thickness, x)),
            'maximum_thickness': float(np.max(thickness)),
            'thickness_to_chord': float(np.max(thickness) / chord) if chord else None}


#: Geometry DVs recorded beside the constraints, with their units.
DESIGN_UNITS = {'twist': 'deg'}


def geometry_record(constraints, surface):
    """The ``geometry_constraints`` block of ``call_summary.json``.

    Surface metrics, every registered geometric constraint value, and the
    non-shape geometry DVs with their units.
    """
    values, design = {}, {}
    for name, value in sorted((constraints or {}).items()):
        values_list = [float(v) for v in np.asarray(value).reshape(-1)]
        if name in DESIGN_UNITS:
            design[name] = {'value': values_list, 'units': DESIGN_UNITS[name]}
        else:
            values[name] = values_list
    return {'surface': surface_metrics(surface), 'constraints': values,
            'design_variables': design}


def write_wing_description(path, surface_vtu, n_sections=11):
    """Cut a triangulated 3-D wall at span (y) stations for a compact report.

    The wings span y with lift along z. Twist is positive nose-up (LE toward
    +z), the twist DV's sign; thickness is measured normal to the chord line.
    """
    data = read_su2_vtu(surface_vtu)
    p = np.asarray(data['points'], dtype=float)
    faces = triangles(data)
    y0, y1 = float(np.min(p[:, 1])), float(np.max(p[:, 1]))
    # Inset the root plane: a root node a round-off above y0 (3e-42 on the
    # ADODG3 LE) is otherwise missed. A rounded tip stays a counted failure.
    ys = np.linspace(y0 + 1e-6 * (y1 - y0), y1, n_sections)
    rows, failures = [], 0
    for y in ys:
        try:
            cut = section_cut(p, faces, y)
        except ValueError:
            cut = np.empty((0, 2))
        if len(cut) < 3:
            rows.append([y] + [np.nan] * 8); failures += 1; continue
        x, z = cut[:, 0], cut[:, 1]
        le, te = float(x.min()), float(x.max())
        le_z = float(np.mean(z[np.isclose(x, le)])); te_z = float(np.mean(z[np.isclose(x, te)]))
        chord = float(np.hypot(te - le, te_z - le_z))
        if not chord:
            rows.append([y] + [np.nan] * 8); failures += 1; continue
        twist = float(np.degrees(np.arctan2(le_z - te_z, te - le)))
        # Chord-frame coordinates: s along the chord, h normal to it (+z side).
        cx, cz = (te - le) / chord, (te_z - le_z) / chord
        s = (x - le) * cx + (z - le_z) * cz
        h = (z - le_z) * cx - (x - le) * cz
        upper = h >= 0.0
        thickness = np.nan
        if upper.any() and (~upper).any():
            grid = np.linspace(0.0, float(s.max()), 201)
            iu, il = np.argsort(s[upper]), np.argsort(s[~upper])
            thickness = float(np.max(np.interp(grid, s[upper][iu], h[upper][iu])
                                     - np.interp(grid, s[~upper][il], h[~upper][il])))
        rows.append([y, np.nan, chord, thickness, thickness / chord,
                     le, te, twist, np.nan])
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['span_location', 'planform_area_contribution', 'chord',
                         'maximum_thickness', 'thickness_to_chord', 'le_x',
                         'te_x', 'twist_deg', 'dihedral_deg'])
        writer.writerows(rows)
    return failures
