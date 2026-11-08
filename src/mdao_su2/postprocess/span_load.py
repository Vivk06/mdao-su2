"""Span load and span efficiency of a wing from its surface pressure.

- Sectional lift: each wall triangle carries the force ``-Cp * A`` (per unit
  dynamic pressure; ``A`` is its area vector pointing out of the body), which is
  split into lift and drag with the angle of attack. Triangles are binned by
  their centroid's spanwise coordinate, so the strips add up to the total
  exactly: ``sum(l' dy) / S_ref = C_L``.
- ``e = C_L^2 / (pi AR C_D)`` and ``e0 = C_L^2 / (pi AR (C_D - C_D0))`` with
  ``C_D0`` from the alpha = 0 run (ADODG paper p. 23, Eq. 9).
- ``e_trefftz``: a sine-series fit of the span load, ``l'(theta) = sum A_n
  sin(n theta)`` with ``y = s cos(theta)``, gives ``A_1^2 / sum n A_n^2``: the
  efficiency the load shape alone implies (1 for an elliptic load).

``section_cut`` (a wall slice at one spanwise station) lives here;
``postprocess/thickness.py`` imports it.
"""

import math
import os

import numpy as np

AXES = {'x': 0, 'y': 1, 'z': 2}


def section_cut(points, faces, y_station):
    """Points (m,2) as (x, z) where the wall crosses y=const. Nodes on the plane are included."""
    hits = []
    for face in faces:
        n = len(face)
        for a in range(n):
            i, j = face[a], face[(a + 1) % n]
            ya, yb = points[i, 1], points[j, 1]
            if ya == y_station:
                hits.append((points[i, 0], points[i, 2]))
            if (ya - y_station) * (yb - y_station) < 0.0:
                t = (y_station - ya) / (yb - ya)
                p = points[i] + t * (points[j] - points[i])
                hits.append((p[0], p[2]))
    if not hits:
        raise ValueError(f'no wall intersection at y={y_station}')
    return np.unique(np.round(np.array(hits), 12), axis=0)


def area_vectors(points, tris, lift_axis='z'):
    """Triangle area vectors, oriented out of the body.

    A wing's upper surface faces +lift and its lower surface -lift, so the
    outward set has ``sum(A_lift * (h - h_mid)) > 0``; the whole set is flipped
    otherwise (a mesh's own orientation convention does not matter here).
    """
    p = np.asarray(points, dtype=float)
    t = np.asarray(tris, dtype=int)
    a = 0.5 * np.cross(p[t[:, 1]] - p[t[:, 0]], p[t[:, 2]] - p[t[:, 0]])
    k = AXES[lift_axis]
    h = p[t].mean(axis=1)[:, k]
    if np.sum(a[:, k] * (h - 0.5 * (h.min() + h.max()))) < 0.0:
        a = -a
    return a


def span_load(points, tris, cp, alpha_deg, s_ref, n_stations, span_axis='y',
              lift_axis='z', flow_axis='x'):
    """Sectional lift ``l' = c_l c`` (per unit span, per unit dynamic pressure) in strips."""
    p = np.asarray(points, dtype=float)
    t = np.asarray(tris, dtype=int)
    a = area_vectors(p, t, lift_axis)
    force = -np.asarray(cp, dtype=float)[t].mean(axis=1)[:, None] * a
    al = math.radians(alpha_deg)
    e_flow, e_lift = np.zeros(3), np.zeros(3)
    e_flow[AXES[flow_axis]] = 1.0
    e_lift[AXES[lift_axis]] = 1.0
    lift = force @ (-math.sin(al) * e_flow + math.cos(al) * e_lift)
    drag = force @ (math.cos(al) * e_flow + math.sin(al) * e_lift)
    y = p[t].mean(axis=1)[:, AXES[span_axis]]
    y0, y1 = float(p[:, AXES[span_axis]].min()), float(p[:, AXES[span_axis]].max())
    edges = np.linspace(y0, y1, int(n_stations) + 1)
    k = np.clip(np.searchsorted(edges, y, side='right') - 1, 0, n_stations - 1)
    dy = np.diff(edges)
    strip_lift = np.bincount(k, weights=lift, minlength=n_stations)
    return {'y': 0.5 * (edges[:-1] + edges[1:]), 'dy': dy, 'edges': edges,
            'lprime': strip_lift / dy, 'CL': float(lift.sum() / s_ref),
            'CD_pressure': float(drag.sum() / s_ref), 'n_triangles': int(len(t))}


def elliptic_load(y, cl, s_ref, semispan):
    """``l'(y)`` of an elliptic load with the same C_L on a half wing of area ``s_ref``."""
    l0 = 4.0 * cl * s_ref / (math.pi * semispan)
    return l0 * np.sqrt(np.clip(1.0 - (np.asarray(y) / semispan) ** 2, 0.0, None))


def trefftz_efficiency(y, lprime, semispan, n_modes=9):
    """``A_1^2 / sum n A_n^2`` from a least-squares odd sine series of the load."""
    theta = np.arccos(np.clip(np.asarray(y, dtype=float) / semispan, -1.0, 1.0))
    modes = np.arange(1, 2 * n_modes, 2)
    basis = np.sin(np.outer(theta, modes))
    coef, *_ = np.linalg.lstsq(basis, np.asarray(lprime, dtype=float), rcond=None)
    return float(coef[0] ** 2 / np.sum(modes * coef ** 2)), coef


def span_efficiency(cl, cd, aspect_ratio, cd0=None):
    """``(e, e0)``; ``e0`` is ``None`` without ``cd0`` (ADODG Eq. 9)."""
    e = cl ** 2 / (math.pi * aspect_ratio * cd)
    e0 = None if cd0 is None else cl ** 2 / (math.pi * aspect_ratio * (cd - cd0))
    return e, e0


def from_surface_vtu(path, alpha_deg, cd, s_ref, sl):
    """The full record for one SU2 ``surface_flow.vtu`` and a ``[postprocess.span_load]`` block."""
    from ..vtu import read_su2_vtu, triangles
    data = read_su2_vtu(path)
    pts = np.asarray(data['points'], dtype=float)
    load = span_load(pts, triangles(data), data['point_data']['Pressure_Coefficient'],
                     alpha_deg, s_ref, sl.n_stations, sl.span_axis, sl.lift_axis)
    e, e0 = span_efficiency(load['CL'], cd, sl.aspect_ratio, sl.cd0)
    e_tr, coef = trefftz_efficiency(load['y'], load['lprime'], sl.semispan)
    return {
        'surface': path, 'alpha_deg': alpha_deg, 'CL_integrated': load['CL'],
        'CD': cd, 'CD0': sl.cd0, 'aspect_ratio': sl.aspect_ratio, 'e': e, 'e0': e0,
        'e_trefftz': e_tr, 'sine_coefficients': coef.tolist(),
        'y': load['y'].tolist(), 'lprime': load['lprime'].tolist(),
        'lprime_elliptic': elliptic_load(load['y'], load['CL'], s_ref,
                                         sl.semispan).tolist(),
        'note': ('e and e0 use the solver C_D; e0 omitted without cd0' if sl.cd0 is None
                 else 'e0 = C_L^2 / (pi AR (C_D - C_D0)), ADODG Eq. 9'),
    }


def compute_run_span_load(cfg, rundir, result):
    """The ``span_load`` block of a finished run's results file, one record per point.

    Reads each point's ``final/<point>/surface.vtu`` (``run_output/final_artifacts.py``).
    """
    sl = cfg.postprocess.span_load
    records = []
    for point, res in zip(cfg.points, result['points']):
        vtu = os.path.join(rundir.final, point.id, 'surface.vtu')
        if not os.path.isfile(vtu):
            records.append({'point': point.id, 'error': f'no final/{point.id}/surface.vtu'})
            continue
        rec = from_surface_vtu(vtu, float(res['alpha']), float(res['C_D']),
                               cfg.case.reference.area, sl)
        rec['point'] = point.id
        records.append(rec)
    return {'points': records}
