"""FFD shape DVs and geometric constraints. All numbers from RunConfig."""

import warnings

import numpy as np
from pygeo.mphys import OM_DVGEOCOMP


__all__ = ['add_ffd_shape_dvs', 'add_geometry_constraints_from_config',
           'frozen_lattice_points', 'shape_functions', 'thickness_probe_lines',
           'FFDGeoBidirComp']


class FFDGeoBidirComp(OM_DVGEOCOMP):
    """Flatten forward products on ranks that have no surface points."""

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        if mode == 'fwd':
            for dvgeo in self.DVGeos.values():
                if 'totalSensitivityProd' not in vars(dvgeo):
                    product = dvgeo.totalSensitivityProd
                    dvgeo.totalSensitivityProd = (
                        lambda vec, name, *args, _product=product, **kwargs:
                        np.asarray(_product(vec, name, *args, **kwargs)).reshape(-1))
        super().compute_jacvec_product(inputs, d_inputs, d_outputs, mode)

# Case values live in the TOML, not here.


def _control_point_2d_shapes(li):
    """Identity shape functions: one DV per interior FFD point per j-row."""
    diry = np.array([0.0, 1.0, 0.0])
    shapes = []
    for i in range(1, li.shape[0] - 1):
        for j in range(li.shape[1]):
            shapes.append({li[i, j, 0]: diry, li[i, j, 1]: diry})
    return shapes


def _bspline_shapes(li, n_basis, degree):
    """B-spline basis shape functions: n_basis smooth DOFs per j-row."""
    from scipy.interpolate import BSpline

    ni = li.shape[0]
    n_int = ni - 2                       # interior points (skip LE/TE)
    if n_basis <= 0:
        n_basis = n_int                  # auto: same count as control_point_2d
    if n_basis > n_int:
        # Silently clamping here used to make a config asking for 40 basis
        # functions run with 28 and report neither -- warn, do not surprise.
        warnings.warn(
            f'n_basis={n_basis} exceeds the {n_int} interior FFD stations; '
            f'clamped to {n_int}. Use a larger FFD lattice for more DOFs.',
            stacklevel=2)
        n_basis = n_int
    if n_basis < degree + 1:
        raise ValueError(
            f'n_basis={n_basis} must be >= degree+1={degree + 1}')

    n_knots = n_basis + degree + 1
    knots = np.concatenate([
        np.zeros(degree + 1),
        np.linspace(0, 1, n_knots - 2 * (degree + 1) + 2)[1:-1],
        np.ones(degree + 1),
    ])

    t = np.linspace(0, 1, n_int)
    diry = np.array([0.0, 1.0, 0.0])

    shapes = []
    for b in range(n_basis):
        coeffs = np.zeros(n_basis)
        coeffs[b] = 1.0
        weights = BSpline(knots, coeffs, degree)(t)

        for j in range(li.shape[1]):
            shape_dict = {}
            for k, i_ffd in enumerate(range(1, ni - 1)):
                if abs(weights[k]) > 1e-14:
                    shape_dict[li[i_ffd, j, 0]] = weights[k] * diry
                    shape_dict[li[i_ffd, j, 1]] = weights[k] * diry
            if shape_dict:
                shapes.append(shape_dict)
    return shapes


# Modified Gram-Schmidt tolerance; 0.1 fixes the n_le=4 conditioning issue.
_INDEPENDENCE_TOL = 0.1


def _independent_shapes(shapes, npts, tol=_INDEPENDENCE_TOL):
    """Drop near-redundant shapes by modified Gram-Schmidt. Priority order: first in wins."""
    kept, basis = [], []
    for shape in shapes:
        vec = np.zeros(npts)
        for idx, direction in shape.items():
            vec[idx] += direction[1]
        norm = np.linalg.norm(vec)
        if norm == 0.0:
            continue
        residual = vec.copy()
        for q in basis:
            residual -= (q @ residual) * q
        r_norm = np.linalg.norm(residual)
        if r_norm > tol * norm:
            basis.append(residual / r_norm)
            kept.append(shape)
    return kept


def _le_te_shapes(li, free_te=False):
    """Anti-symmetric LE/TE shapes. free_te adds symmetric TE translation (rear-loading)."""
    diry = np.array([0.0, 1.0, 0.0])
    shapes = []
    for i in [0, li.shape[0] - 1]:
        shapes.append({li[i, 0, 0]: diry, li[i, 0, 1]: diry,
                       li[i, 1, 0]: -diry, li[i, 1, 1]: -diry})
    if free_te:
        i = li.shape[0] - 1
        shapes.append({li[i, 0, 0]: diry, li[i, 0, 1]: diry,
                       li[i, 1, 0]: diry, li[i, 1, 1]: diry})
    return shapes


def _le_refine_shapes(li, n_le):
    """Local single-station DVs on n_le stations nearest the LE, additive with the global pattern."""
    ni = li.shape[0]
    n_int = ni - 2
    if n_le > n_int:
        warnings.warn(
            f'n_le={n_le} exceeds the {n_int} interior FFD stations; '
            f'clamped to {n_int}.', stacklevel=2)
        n_le = n_int

    diry = np.array([0.0, 1.0, 0.0])
    shapes = []
    for i in range(1, 1 + n_le):
        for j in range(li.shape[1]):
            shapes.append({li[i, j, 0]: diry, li[i, j, 1]: diry})
    return shapes


def _ffd3d_shapes(li, directions, link_k, freeze, freeze_points=()):
    """One DV per free control point per direction. Direction-major order (contract: D5)."""
    ni, nj, nk = li.shape
    fi, fj, fk = ((set(x % n for x in idx)
                   for idx, n in zip(freeze, (ni, nj, nk))))
    fp = {(i % ni, j % nj, k % nk) for i, j, k in freeze_points}

    shapes = []
    for diry in (np.asarray(d, dtype=float) for d in directions):
        for i in range(ni):
            if i in fi:
                continue
            for j in range(nj):
                if j in fj:
                    continue
                if link_k:
                    free_k = [k for k in range(nk)
                              if k not in fk and (i, j, k) not in fp]
                    if len(free_k) == nk - len(fk) and free_k:
                        shapes.append({li[i, j, k]: diry for k in free_k})
                else:
                    for k in range(nk):
                        if k not in fk and (i, j, k) not in fp:
                            shapes.append({li[i, j, k]: diry})
    return shapes


def _thickness2d_shapes(li, freeze_i=()):
    """Paired thickness DVs: per free column, upper row +y and lower row -y.

    The FFD thickness method of Yang & Da Ronch (2018, Fig 8b): a positive DV moves
    both surfaces outward by the same amount, so the section stays symmetric.
    """
    ni, nj, nk = li.shape
    if nj != 2:
        raise ValueError(f'thickness2d needs a 2-row FFD lattice, got {nj} rows')
    frozen = {i % ni for i in freeze_i}
    diry = np.array([0.0, 1.0, 0.0])
    shapes = []
    for i in range(ni):
        if i in frozen:
            continue
        shape = {li[i, 1, k]: diry for k in range(nk)}
        shape.update({li[i, 0, k]: -diry for k in range(nk)})
        shapes.append(shape)
    return shapes


def shape_functions(shape_cfg, lattice_shape, control_point_indices=None):
    """Return shape-function dicts for the configured pattern, without pyGeo/MPI."""
    if control_point_indices is None:
        ni, nj, nk = lattice_shape[:3]
        li = np.arange(ni * nj * nk).reshape(ni, nj, nk)
    else:
        li = np.asarray(control_point_indices)

    pattern = shape_cfg.pattern
    if pattern == 'thickness2d':
        return _thickness2d_shapes(li, shape_cfg.freeze_i)
    if pattern == 'ffd3d':
        return _ffd3d_shapes(
            li, shape_cfg.resolved_directions, shape_cfg.link_k,
            (shape_cfg.freeze_i, shape_cfg.freeze_j, shape_cfg.freeze_k),
            shape_cfg.freeze_points)

    if pattern == 'bspline':
        shapes = _bspline_shapes(li, shape_cfg.n_basis, shape_cfg.degree)
    else:
        shapes = _control_point_2d_shapes(li)

    free_te = shape_cfg.free_te
    if shape_cfg.n_le > 0:
        shapes = _independent_shapes(
            _le_refine_shapes(li, shape_cfg.n_le)
            + _le_te_shapes(li, free_te=free_te) + shapes, li.size)
    else:
        shapes.extend(_le_te_shapes(li, free_te=free_te))
    return shapes


def embed_wall(geometry, builder):
    """Embed the wall points in the FFD (``nom_add_discipline_coords``)."""
    from mphys import MPhysVariables
    _V = MPhysVariables.Aerodynamics.Surface
    geometry.nom_add_discipline_coords(_V.Geometry, points=builder.get_surface_points())


def add_ffd_shape_dvs(geometry, builder, shape_cfg=None):
    """Embed wall points and add shape DVs. Returns (n_total, per_direction_counts)."""
    embed_wall(geometry, builder)
    return add_ffd_shapes(geometry, shape_cfg)


def add_ffd_shapes(geometry, shape_cfg=None):
    """Shape DVs on an FFD whose wall points are already embedded."""
    li = geometry.nom_getDVGeo().getLocalIndex(0)

    pattern = 'control_point_2d' if shape_cfg is None else shape_cfg.pattern
    if pattern == 'thickness2d':
        n = _add_shapes(geometry, _thickness2d_shapes(li, shape_cfg.freeze_i))
        return n, (n,)
    if pattern == 'ffd3d':
        # ffd3d owns its own freeze; 2D LE/TE shapes not appended.
        dirs = shape_cfg.resolved_directions
        freeze = (shape_cfg.freeze_i, shape_cfg.freeze_j, shape_cfg.freeze_k)
        if len(dirs) == 1:
            n = _add_shapes(geometry, _ffd3d_shapes(
                li, dirs, shape_cfg.link_k, freeze, shape_cfg.freeze_points))
            return n, (n,)
        counts = []
        for d in dirs:
            counts.append(len(_ffd3d_shapes(
                li, [d], shape_cfg.link_k, freeze, shape_cfg.freeze_points)))
        all_shapes = _ffd3d_shapes(
            li, dirs, shape_cfg.link_k, freeze, shape_cfg.freeze_points)
        n = _add_shapes(geometry, all_shapes)
        return n, tuple(counts)
    if pattern == 'bspline':
        shapes = _bspline_shapes(li, shape_cfg.n_basis, shape_cfg.degree)
    else:
        shapes = _control_point_2d_shapes(li)
    free_te = shape_cfg is not None and shape_cfg.free_te
    # n_le > 0: local LE DVs go first so they survive independence filter.
    if shape_cfg is not None and shape_cfg.n_le > 0:
        shapes = _independent_shapes(
            _le_refine_shapes(li, shape_cfg.n_le)
            + _le_te_shapes(li, free_te=free_te) + shapes, li.size)
    else:
        shapes.extend(_le_te_shapes(li, free_te=free_te))

    n = _add_shapes(geometry, shapes)
    return n, (n,)


def frozen_lattice_points(shape_cfg, lattice_shape):
    """Set of (i,j,k) control points no DV moves, derived from _ffd3d_shapes by difference."""
    ni, nj, nk = lattice_shape[:3]
    li = np.arange(ni * nj * nk).reshape(ni, nj, nk)
    shapes = _ffd3d_shapes(li, shape_cfg.resolved_directions, shape_cfg.link_k,
                           (shape_cfg.freeze_i, shape_cfg.freeze_j,
                            shape_cfg.freeze_k), shape_cfg.freeze_points)
    touched = {idx for shape in shapes for idx in shape}
    return {(i, j, k)
            for i in range(ni) for j in range(nj) for k in range(nk)
            if li[i, j, k] not in touched}


def _add_shapes(geometry, shapes):
    if not shapes:
        raise ValueError(
            'the shape-DV pattern produced no design variables -- every control '
            'point is frozen. Check [dv.shape] freeze_i/freeze_j/freeze_k '
            'against the FFD lattice size.')
    geometry.nom_addShapeFunctionDV(dvName='shape', shapes=shapes)
    return len(shapes)


def thickness_probe_lines(t, zm):
    """Return (name, leList, teList, lower) per thickness station, axes-aware."""
    from ..run_config.root import AXES_SPAN_Y

    rows = []
    if t.axes == AXES_SPAN_Y:
        lo, hi = t.span_offsets
        for (y, tmin), px in zip(t.stations, t.probe_x):
            # Absolute y plane + absolute thickness bound; pyGeo projects +-z.
            name = f'thick_y{int(round(y * 100)):03d}'
            leList = [[px, y + lo, 0.0], [px, y + hi, 0.0]]
            teList = [[px + t.te_offset, y + lo, 0.0],
                      [px + t.te_offset, y + hi, 0.0]]
            rows.append((name, leList, teList, tmin))
        return rows

    for x, tmin in t.stations:
        name = f'thick_{int(round(x * 100)):02d}c'
        # Flip TE-side near trailing edge to stay on airfoil.
        xte = x + t.te_offset if x + t.te_offset <= 0.999 else x - t.te_offset
        leList = [[x, 0.0, -zm], [x, 0.0, zm]]
        teList = [[xte, 0.0, -zm], [xte, 0.0, zm]]
        rows.append((name, leList, teList, tmin))
    return rows


def add_geometry_constraints_from_config(geometry, builder, cfg):
    """Add thickness/volume/LE-radius/curvature constraints from RunConfig. Returns (name, kwargs) list."""
    c = cfg.constraints
    zm = cfg.geometry.zspan / 2.0

    geometry.nom_setConstraintSurface(
        builder.get_triangulated_surface(zspan=cfg.geometry.zspan),
        surfFormat='point-point')

    cons = []
    if c.thickness.enabled:
        for name, leList, teList, tmin in thickness_probe_lines(c.thickness, zm):
            geometry.nom_addThicknessConstraints2D(
                name, leList, teList, nSpan=c.thickness.nSpan,
                nChord=c.thickness.nChord, scaled=c.thickness.scaled)
            bound = 'upper' if c.thickness.kind == 'max' else 'lower'
            cons.append((name, {bound: tmin, 'scaler': 1.0}))

    if c.volume.enabled:
        v = c.volume
        leList = [[v.le_x, 0.0, -zm], [v.le_x, 0.0, zm]]
        teList = [[v.te_x, 0.0, -zm], [v.te_x, 0.0, zm]]
        geometry.nom_addVolumeConstraint('volcon', leList, teList,
                                         nSpan=v.nSpan, nChord=v.nChord)
        cons.append(('volcon', dict(lower=v.lower, scaler=1.0)))

    if c.curvature.enabled:
        k = c.curvature
        # Two constraints (upper/lower), probing +-y from the chord line.
        for name, axis in (('curv_upper', [0.0, 1.0, 0.0]),
                           ('curv_lower', [0.0, -1.0, 0.0])):
            geometry.nom_addCurvatureConstraint1D(
                name, [k.start_x, 0.0, 0.0], [k.end_x, 0.0, 0.0],
                nPts=k.n_pts, axis=axis, curvatureType=k.curvature_type,
                scaled=True, KSCoeff=k.ks_coeff)
            cons.append((name, dict(upper=k.upper, scaler=k.scaler)))

    if c.le_radius.enabled:
        r = c.le_radius
        leList = [[r.le_x, 0.0, -zm], [r.le_x, 0.0, zm]]
        geometry.nom_addLERadiusConstraints('rcon', leList, r.nSpan,
                                            list(r.axis), list(r.chord_dir))
        cons.append(('rcon', dict(lower=r.lower, scaler=1.0)))

    return cons

