"""Design-variable and geometry visualisation — from config alone, no solve."""
import numpy as np

from ..geometry.ffd import shape_functions, frozen_lattice_points
from .style import (
    LATTICE, EDGE, FROZEN, FROZEN_DARK, INACTIVE, INACTIVE_EDGE,
    LABEL, BASELINE, OPTIMIZED, diverging,
)

__all__ = [
    'load_lattice', 'deformed_lattice',
    'ffd_box', 'dv_map', 'dv_influence', 'basis_functions',
    'cst_classification', 'moment_reference',
    'panel_ffd_planform', 'panel_freeze_map',
    'TUTORIAL_CP_RANGE',
]

TUTORIAL_CP_RANGE = (-1.0, 0.4)


def _cst_problem(cfg):
    """Build the same CST component and wall point set used by the model."""
    from mphys import MPhysVariables
    import openmdao.api as om

    from ..geometry.cst import CSTGeoComp
    from ..mesh.su2_mesh_file import extract_baseline_surface

    surface = extract_baseline_surface(cfg.mesh_path, cfg.case.wall_single)
    points = np.zeros((len(surface), 3))
    points[:, :cfg.case.ndim] = surface
    variables = MPhysVariables.Aerodynamics.Surface

    class _CSTMapGroup(om.Group):
        def setup(self):
            self.add_subsystem('dvs', om.IndepVarComp(), promotes=['*'])
            self.add_subsystem(
                'geometry',
                CSTGeoComp(dat_file=cfg.dat_path,
                            n_cst=cfg.geometry.n_cst),
            )

        def configure(self):
            self.geometry.nom_add_discipline_coords(variables.Geometry, points)
            n_shape, _ = self.geometry.add_shape_dv()
            self.dvs.add_output('shape', val=np.zeros(n_shape))
            self.connect('shape', 'geometry.shape')

    problem = om.Problem(model=_CSTMapGroup())
    problem.setup(mode='rev')
    problem.run_model()
    return problem, problem.model.geometry, points


def _hh_problem(cfg):
    """Build the same Hicks--Henne component and wall point set used by the model."""
    from mphys import MPhysVariables
    import openmdao.api as om

    from ..geometry.factory import build_geometry_component
    from ..mesh.su2_mesh_file import extract_baseline_surface, parse_su2_mesh

    surface = extract_baseline_surface(cfg.mesh_path, cfg.case.wall_single)
    points = np.zeros((len(surface), 3))
    points[:, :cfg.case.ndim] = surface
    _mesh_points, _elements, markers = parse_su2_mesh(cfg.mesh_path)
    node_ids = sorted({node for element in markers[cfg.case.wall_single]
                       for node in element[1:]})
    node_positions = {node: index for index, node in enumerate(node_ids)}
    segments = [
        (node_positions[element[1]], node_positions[element[2]])
        for element in markers[cfg.case.wall_single]
    ]
    variables = MPhysVariables.Aerodynamics.Surface

    class _HHMapGroup(om.Group):
        def setup(self):
            self.add_subsystem('dvs', om.IndepVarComp(), promotes=['*'])
            self.add_subsystem('geometry', build_geometry_component(cfg))

        def configure(self):
            self.geometry.nom_set_surface_segments(segments)
            self.geometry.nom_add_discipline_coords(variables.Geometry, points)
            n_shape, _ = self.geometry.add_shape_dv()
            self.dvs.add_output('shape', val=np.zeros(n_shape))
            self.connect('shape', 'geometry.shape')

    problem = om.Problem(model=_HHMapGroup())
    problem.setup(mode='rev')
    problem.run_model()
    return (problem, problem.model.geometry, points,
            variables.Geometry.COORDINATES_OUTPUT)


def load_lattice(path):
    """Plot3D ``.xyz`` FFD lattice -> ``(ni, nj, nk, 3)``. Fortran order, i-fastest."""
    with open(path) as fh:
        tok = fh.read().split()
    n_blocks = int(tok[0])
    if n_blocks < 1:
        raise ValueError(f'Expected >= 1 block, got {n_blocks}')
    ni, nj, nk = int(tok[1]), int(tok[2]), int(tok[3])
    vals = np.array(tok[4:], dtype=float)
    per = ni * nj * nk
    out = np.empty((ni, nj, nk, 3))
    for d in range(3):
        out[..., d] = vals[d * per:(d + 1) * per].reshape(
            (ni, nj, nk), order='F')
    return out


def _proj_axes(projection):
    """Return (ax0, ax1, label0, label1) for the projection."""
    if projection in ('xy', 'planform'):
        return 0, 1, 'x (chord)', 'y (span)'
    elif projection == 'xz':
        return 0, 2, 'x (chord)', 'z'
    elif projection == 'yz':
        return 1, 2, 'y (span)', 'z'
    else:
        raise ValueError(f'projection must be xy/xz/yz/planform, got {projection!r}')


def moment_reference(ax, cfg, projection='xy'):
    """Draw the configured moment pivot and return its plotted coordinates."""
    a0, a1, _, _ = _proj_axes(projection)
    moment = cfg.constraints.moment
    pivot = np.asarray(moment.resolved_pivot(
        cfg.case.reference.origin_moment), dtype=float)
    xy = pivot[[a0, a1]]
    label = f'{moment.objective_key} origin'
    ax.scatter([xy[0]], [xy[1]], marker='X', s=70, color=FROZEN_DARK,
               label=label, zorder=7)
    ax.annotate(f'{label}\n({xy[0]:g}, {xy[1]:g})', xy=xy,
                xytext=(6, 7), textcoords='offset points', color=LABEL,
                fontsize=7, zorder=8)
    return tuple(float(value) for value in xy)


def ffd_box(ax, lattice, wall=None, projection='xy', **kwargs):
    """Draw FFD lattice + bounding box, optionally with wall. Returns n_outside."""
    a0, a1, lab0, lab1 = _proj_axes(projection)
    ni, nj, nk = lattice.shape[:3]

    for k in range(nk):
        for i in range(ni):
            ax.plot(lattice[i, :, k, a0], lattice[i, :, k, a1],
                    color=LATTICE, lw=0.6, zorder=1)
        for j in range(nj):
            ax.plot(lattice[:, j, k, a0], lattice[:, j, k, a1],
                    color=LATTICE, lw=0.6, zorder=1)

    flat = lattice.reshape(-1, 3)
    lo = flat[:, [a0, a1]].min(axis=0)
    hi = flat[:, [a0, a1]].max(axis=0)
    from matplotlib.patches import Rectangle
    rect = Rectangle(lo, hi[0] - lo[0], hi[1] - lo[1],
                     fill=False, edgecolor=EDGE, lw=1.2, ls='--', zorder=2)
    ax.add_patch(rect)

    n_outside = 0
    if wall is not None:
        w = np.asarray(wall, dtype=float)
        ax.plot(w[:, a0], w[:, a1], color=BASELINE, lw=0.8, zorder=3,
                label='wall', **kwargs)
        tol = 1e-6
        outside = ((w[:, a0] < lo[0] - tol) | (w[:, a0] > hi[0] + tol) |
                   (w[:, a1] < lo[1] - tol) | (w[:, a1] > hi[1] + tol))
        n_outside = int(outside.sum())

    ax.set_aspect('equal')
    ax.set_xlabel(lab0)
    ax.set_ylabel(lab1)
    return n_outside


def dv_entries(cfg, lattice):
    """Every DV in DV-vector order: shape, then twist root to tip, then alpha.

    Twist entries name their FFD section ``j``, its station ``y`` and the axis
    ``x`` (``[dv.twist] axis_x_fraction`` of the section's chord extent).
    """
    out = []
    if cfg.dv.shape.enabled:
        for k, _ in enumerate(shape_functions(cfg.dv.shape, lattice.shape)):
            out.append({'name': 'shape', 'index': k})
    t = cfg.dv.twist
    if t.enabled:
        first = 0 if t.include_root else 1
        for k, j in enumerate(range(first, lattice.shape[1])):
            section = lattice[:, j, :, :]
            x0, x1 = section[..., 0].min(), section[..., 0].max()
            out.append({'name': 'twist', 'index': k, 'section_j': j,
                        'y': float(section[..., 1].mean()),
                        'axis_x': float(x0 + t.axis_x_fraction * (x1 - x0)),
                        'units': 'deg'})
    for i, point in enumerate(cfg.points):
        if point.alpha == 'design':
            out.append({'name': f'alpha_{i}', 'index': 0, 'point': point.id,
                        'units': 'deg'})
    return out


def dv_map(ax, cfg, lattice, projection='xy', **kwargs):
    """Per control point: has-DV / frozen / linked, with direction arrows. Returns (n_free, n_frozen, n_total).

    Twist DVs (a rotation of a whole section about its axis point) are marked at
    their axis points and counted in ``n_free``.
    """
    a0, a1, lab0, lab1 = _proj_axes(projection)
    ni, nj, nk = lattice.shape[:3]
    shape_cfg = cfg.dv.shape

    shapes = shape_functions(shape_cfg, lattice.shape) if shape_cfg.enabled else []
    n_dvs = len(shapes)

    touched = set()
    directions = {}
    for shape in shapes:
        for idx, direction in shape.items():
            touched.add(idx)
            directions[idx] = np.asarray(direction, dtype=float)

    if shape_cfg.pattern == 'ffd3d':
        frozen = frozen_lattice_points(shape_cfg, lattice.shape)
    else:
        frozen = set()

    marks = {0: 'o', 1: 's'}
    off = 1  # 1-based labels

    for k in range(nk):
        for i in range(ni):
            ax.plot(lattice[i, :, k, a0], lattice[i, :, k, a1],
                    color=LATTICE, lw=0.4, zorder=1)
        for j in range(nj):
            ax.plot(lattice[:, j, k, a0], lattice[:, j, k, a1],
                    color=LATTICE, lw=0.4, zorder=1)

    flat_idx = np.arange(ni * nj * nk).reshape(ni, nj, nk)
    n_frozen_count = 0

    for i in range(ni):
        for j in range(nj):
            for k in range(nk):
                idx = flat_idx[i, j, k]
                x = lattice[i, j, k, a0]
                y = lattice[i, j, k, a1]
                marker = marks.get(k, 'o')

                if (i, j, k) in frozen:
                    ax.plot(x, y, marker, ms=7, mfc=FROZEN, mec=FROZEN_DARK,
                            mew=0.6, zorder=4)
                    n_frozen_count += 1
                elif idx in touched:
                    ax.plot(x, y, marker, ms=5, mfc=BASELINE, mec=EDGE,
                            mew=0.4, zorder=3)
                    d = directions.get(idx)
                    if d is not None and np.linalg.norm(d) > 0:
                        scale = 0.02 * max(
                            np.ptp(lattice[..., a0]), np.ptp(lattice[..., a1]))
                        ax.annotate('', xy=(x + d[a0] * scale,
                                            y + d[a1] * scale),
                                    xytext=(x, y),
                                    arrowprops=dict(arrowstyle='->', color=EDGE,
                                                    lw=0.8))
                else:
                    ax.plot(x, y, marker, ms=3.5, mfc=INACTIVE,
                            mec=INACTIVE_EDGE, mew=0.4, zorder=2)

    twists = [e for e in dv_entries(cfg, lattice) if e['name'] == 'twist']
    for e in twists:
        p = {0: e['axis_x'], 1: e['y'], 2: 0.0}
        ax.plot(p[a0], p[a1], 'D', ms=5, mfc=OPTIMIZED, mec=EDGE, mew=0.4, zorder=5)
        ax.annotate(f"twist[{e['index']}]", (p[a0], p[a1]), fontsize=6, color=LABEL,
                    xytext=(3, 3), textcoords='offset points')
    n_dvs += len(twists)

    ax.set_aspect('equal')
    ax.set_xlabel(lab0)
    ax.set_ylabel(lab1)
    moment_reference(ax, cfg, projection=projection)
    return n_dvs, n_frozen_count, ni * nj * nk


def dv_influence(ax, cfg, lattice, dvs, projection='xy', **kwargs):
    """Surface displacement magnitude per DV (an evaluation's geometry/design_variables.npy)."""
    if cfg.geometry.parameterization == 'cst':
        problem, geometry, points = _cst_problem(cfg)
        output_name = geometry._point_names[0]
        baseline = problem.get_val(
            f'geometry.{output_name}', get_remote=True).reshape(-1, 3).copy()
        n_dvs = geometry._n_upper + geometry._n_lower
        influences = np.zeros((n_dvs,) + baseline.shape)
        for index in range(n_dvs):
            shape = np.zeros(n_dvs)
            shape[index] = 1.0
            problem.set_val('shape', shape)
            problem.run_model()
            updated = problem.get_val(
                f'geometry.{output_name}', get_remote=True).reshape(-1, 3)
            influences[index] = updated - baseline

        point_set = geometry._dvgeo.points[output_name]
        upper = np.asarray(point_set['upper'], dtype=int)
        lower = np.asarray(point_set['lower'], dtype=int)
        for index, displacement in enumerate(influences):
            rows = upper if index < geometry._n_upper else lower
            order = rows[np.argsort(points[rows, 0])]
            surface = 'upper' if index < geometry._n_upper else 'lower'
            weight = index if surface == 'upper' else index - geometry._n_upper
            ax.plot(points[order, 0], displacement[order, 1],
                    color=BASELINE if surface == 'upper' else OPTIMIZED,
                    ls='-' if surface == 'upper' else '--', lw=1.0,
                    label=f'{surface} w{weight}', **kwargs)
        ax.axhline(0.0, color=EDGE, lw=0.5)
        ax.set_xlabel('x (chord)')
        ax.set_ylabel('Vertical displacement at unit perturbation')
        ax.set_xlim(float(points[:, 0].min()), float(points[:, 0].max()))
        return influences

    if cfg.geometry.parameterization == 'hicks_henne':
        problem, geometry, points, output_name = _hh_problem(cfg)
        baseline = problem.get_val(
            f'geometry.{output_name}', get_remote=True).reshape(-1, 3).copy()
        n_dvs = problem.get_val('shape', get_remote=True).size
        influences = np.zeros((n_dvs,) + baseline.shape)
        for index in range(n_dvs):
            shape = np.zeros(n_dvs)
            shape[index] = 1.0
            problem.set_val('shape', shape)
            problem.run_model()
            updated = problem.get_val(
                f'geometry.{output_name}', get_remote=True).reshape(-1, 3)
            influences[index] = updated - baseline

        upper_mask = geometry._point_sets[output_name]['upper']
        n_per_surface = cfg.dv.shape.n_dv
        for index, displacement in enumerate(influences):
            upper = index < n_per_surface
            rows = np.flatnonzero(upper_mask if upper else ~upper_mask)
            order = rows[np.argsort(points[rows, 0])]
            surface = 'upper' if upper else 'lower'
            weight = index if upper else index - n_per_surface
            ax.plot(points[order, 0], displacement[order, 1],
                    color=BASELINE if upper else OPTIMIZED,
                    ls='-' if upper else '--', lw=1.0,
                    label=f'{surface} h{weight}', **kwargs)
        ax.axhline(0.0, color=EDGE, lw=0.5)
        ax.set_xlabel('x (chord)')
        ax.set_ylabel('Vertical displacement at unit perturbation')
        ax.set_xlim(float(points[:, 0].min()), float(points[:, 0].max()))
        return influences

    from .style import diverging as _div
    a0, a1, lab0, lab1 = _proj_axes(projection)
    shape_cfg = cfg.dv.shape
    shapes = shape_functions(shape_cfg, lattice.shape)

    if len(shapes) != len(dvs):
        raise ValueError(
            f'{len(shapes)} shape functions but {len(dvs)} DVs')

    flat = lattice.reshape(-1, 3).copy()
    disp = np.zeros(len(flat))
    for dv_val, shape in zip(dvs, shapes):
        for idx, direction in shape.items():
            d = np.asarray(direction, dtype=float)
            disp[idx] += abs(dv_val) * np.linalg.norm(d)

    ni, nj, nk = lattice.shape[:3]
    lim = max(float(disp.max()), 1e-12)
    cmap = _div()

    for k in range(nk):
        xy = lattice[:, :, k, [a0, a1]].reshape(-1, 2)
        dd = disp[np.arange(ni * nj * nk).reshape(ni, nj, nk)[:, :, k].ravel()]
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=dd, cmap=cmap, vmin=0, vmax=lim,
                        s=30, edgecolors=EDGE, linewidths=0.3, zorder=3,
                        **kwargs)

    ax.set_aspect('equal')
    ax.set_xlabel(lab0)
    ax.set_ylabel(lab1)
    return sc


def basis_functions(ax, cfg, **kwargs):
    """Plot B-spline basis curves, or identity shapes for other patterns."""
    if cfg.geometry.parameterization == 'cst':
        _problem, geometry, _points = _cst_problem(cfg)
        dvgeo = geometry._dvgeo
        x = np.linspace(0.0, 1.0, 200)
        basis = {}
        for surface, count, colour, linestyle in (
                ('upper', geometry._n_upper, BASELINE, '-'),
                ('lower', geometry._n_lower, OPTIMIZED, '--')):
            curves = dvgeo.computeShapeFunctions(x, np.ones(count))
            basis[surface] = curves
            for index, curve in enumerate(curves):
                ax.plot(x, curve, color=colour, ls=linestyle, lw=1.0,
                        alpha=0.35 + 0.65 * (index + 1) / count,
                        label=f'{surface} w{index}', **kwargs)
        ax.set_xlabel('Normalised chord station', fontsize=9, color=LABEL)
        ax.set_ylabel('Bernstein basis weight', fontsize=9, color=LABEL)
        ax.set_xlim(0, 1)
        return basis

    if cfg.geometry.parameterization == 'hicks_henne':
        from ..geometry.hicks_henne import build_basis_matrix, compute_bump_locations

        shape = cfg.dv.shape
        x = np.linspace(0.0, 1.0, 200)
        curves = build_basis_matrix(
            x, compute_bump_locations(shape.n_dv, shape.distribution),
            shape.bump_width).T
        basis = {'upper': curves}
        for index, curve in enumerate(curves):
            ax.plot(x, curve, color=BASELINE, lw=1.0,
                    alpha=0.35 + 0.65 * (index + 1) / shape.n_dv,
                    label=f'upper h{index}', **kwargs)
        upper_only = all(
            point.aoa_lower == point.aoa_upper == 0.0
            for point in cfg.points)
        if not upper_only:
            basis['lower'] = -curves
            for index, curve in enumerate(curves):
                ax.plot(x, -curve, color=OPTIMIZED, ls='--', lw=1.0,
                        alpha=0.35 + 0.65 * (index + 1) / shape.n_dv,
                        label=f'lower h{index}', **kwargs)
        ax.axhline(0.0, color=EDGE, lw=0.5)
        ax.set_xlabel('Normalised chord station', fontsize=9, color=LABEL)
        ax.set_ylabel('Vertical displacement at unit perturbation',
                      fontsize=9, color=LABEL)
        ax.set_xlim(0, 1)
        return basis

    from scipy.interpolate import BSpline

    shape_cfg = cfg.dv.shape
    pattern = shape_cfg.pattern

    if pattern == 'bspline':
        n_basis = shape_cfg.n_basis
        degree = shape_cfg.degree
        if n_basis <= 0:
            n_basis = 8  # fallback

        n_knots = n_basis + degree + 1
        knots = np.concatenate([
            np.zeros(degree + 1),
            np.linspace(0, 1, n_knots - 2 * (degree + 1) + 2)[1:-1],
            np.ones(degree + 1),
        ])

        t = np.linspace(0, 1, 200)
        for b in range(n_basis):
            coeffs = np.zeros(n_basis)
            coeffs[b] = 1.0
            w = BSpline(knots, coeffs, degree)(t)
            ax.plot(t, w, lw=1.2, label=f'B{b}', color=f'C{b % 10}',
                    **kwargs)
    else:
        ax.text(0.5, 0.5, f'Pattern {pattern!r} uses identity shapes',
                ha='center', va='center', transform=ax.transAxes,
                fontsize=10, color=LABEL)

    ax.set_xlabel('Normalised chord station', fontsize=9, color=LABEL)
    ax.set_ylabel('Weight', fontsize=9, color=LABEL)
    ax.set_xlim(0, 1)


def cst_classification(ax, cfg, *, dvgeo=None, points=None,
                       projected=None, split=None, **kwargs):
    """Draw the upper/lower split stored by the model's ``DVGeometryCST``."""
    if dvgeo is None:
        _problem, geometry, points = _cst_problem(cfg)
        dvgeo = geometry._dvgeo
        point_name = geometry._point_names[0]
    else:
        point_name = 'wall'
        points = np.asarray(points, dtype=float)

    point_set = dvgeo.points[point_name]
    upper = np.asarray(point_set['upper'], dtype=int)
    lower = np.asarray(point_set['lower'], dtype=int)
    endpoints = np.unique((np.argmin(points[:, 0]), np.argmax(points[:, 0])))
    endpoint_set = set(endpoints.tolist())
    upper_plot = np.array([i for i in upper if i not in endpoint_set], dtype=int)
    lower_plot = np.array([i for i in lower if i not in endpoint_set], dtype=int)
    classified = set(upper.tolist()) | set(lower.tolist())
    unclassified = sorted(set(range(len(points))) - classified)

    ax.scatter(points[upper_plot, 0], points[upper_plot, 1], s=14,
               color=BASELINE, label=f'upper ({len(upper)})', **kwargs)
    ax.scatter(points[lower_plot, 0], points[lower_plot, 1], s=14,
               color=OPTIMIZED, label=f'lower ({len(lower)})', **kwargs)
    ax.scatter(points[endpoints, 0], points[endpoints, 1], s=75, marker='*',
               color=EDGE, label='LE / TE', zorder=5)
    if unclassified:
        ax.scatter(points[unclassified, 0], points[unclassified, 1], s=60,
                   facecolors='none', edgecolors=FROZEN_DARK,
                   label=f'unclassified ({len(unclassified)})')

    moment_reference(ax, cfg, projection='xy')

    mixed = []
    if projected is not None and split is not None:
        for name, (probe_upper, probe_lower) in split.items():
            probe_points = projected[name]
            ax.scatter(probe_points[probe_upper, 0], probe_points[probe_upper, 1],
                       s=70, marker='^', facecolors='none',
                       edgecolors=BASELINE, linewidths=1.4)
            ax.scatter(probe_points[probe_lower, 0], probe_points[probe_lower, 1],
                       s=70, marker='v', facecolors='none',
                       edgecolors=OPTIMIZED, linewidths=1.4)
            upper_set = set(probe_upper)
            lower_set = set(probe_lower)
            for index in range(0, len(probe_points), 2):
                pair = {index, index + 1}
                bad = pair <= upper_set or pair <= lower_set
                if bad:
                    mixed.append(f'{name}@x={probe_points[index, 0]:.3f}')
                ax.plot(probe_points[[index, index + 1], 0],
                        probe_points[[index, index + 1], 1],
                        color=FROZEN_DARK if bad else EDGE,
                        lw=2.2 if bad else 0.9, zorder=4)

    ax.set_aspect('equal')
    ax.set_xlabel('x (chord)')
    ax.set_ylabel('y')
    ax.legend(loc='upper right', fontsize=8, frameon=False)
    return {
        'upper': upper,
        'lower': lower,
        'endpoints': endpoints,
        'n_points': len(points),
        'unclassified': unclassified,
        'mixed': mixed,
    }


def deformed_lattice(lattice, dvs, shape_cfg):
    """Apply a shape-DV vector to an FFD lattice via geometry.ffd mappings."""
    shapes = shape_functions(shape_cfg, lattice.shape)
    if len(shapes) != len(dvs):
        raise ValueError(f'{len(shapes)} shape functions but {len(dvs)} DVs -- '
                         f'the config does not match the saved DV vector')
    flat = lattice.reshape(-1, 3).copy()
    for dv, shape in zip(dvs, shapes):
        for idx, direction in shape.items():
            flat[idx] += dv * np.asarray(direction, dtype=float)
    return flat.reshape(lattice.shape)


def panel_ffd_planform(ax, L0, L1, frozen=(), moved_tol=1e-12):
    """Planform view of which control points moved. Returns (scatter, lim, n_moved, n_total)."""
    ni, nj, nk = L0.shape[:3]
    d = L1[..., 2] - L0[..., 2]
    lim = max(float(np.abs(d).max()), 1e-12)

    for i in range(ni):
        ax.plot(L0[i, :, 0, 0], L0[i, :, 0, 1], color=LATTICE, lw=0.6, zorder=1)
    for j in range(nj):
        ax.plot(L0[:, j, 0, 0], L0[:, j, 0, 1], color=LATTICE, lw=0.6, zorder=1)

    fz = {tuple(int(v) % n for v, n in zip(p, (ni, nj, nk))) for p in frozen}
    cmap = diverging()
    marks = {0: 'o', 1: 's'}
    sc = None
    for k in range(nk):
        xy = L0[:, :, k, :2].reshape(-1, 2)
        dz = d[:, :, k].reshape(-1)
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=dz, cmap=cmap, vmin=-lim, vmax=lim,
                        s=46 if k == 0 else 20, marker=marks[k],
                        edgecolors=EDGE, linewidths=0.4, zorder=3)
    if fz:
        pf = np.array([L0[i, j, k, :2] for (i, j, k) in sorted(fz)])
        ax.scatter(pf[:, 0], pf[:, 1], s=150, facecolors='none',
                   edgecolors=FROZEN, linewidths=1.6, zorder=4,
                   label=f'frozen ({len(fz)})')
        ax.legend(loc='upper left', fontsize=8, frameon=False)
    ax.set_aspect('equal')
    ax.set_xlabel('x (chord)')
    ax.set_ylabel('y (span)')
    return sc, lim, int((np.abs(d) > moved_tol).sum()), int(d.size)


def panel_freeze_map(ax, L0, frozen, one_based_labels=True):
    """Mark all lattice points; ring and label frozen ones. Returns n_frozen."""
    ni, nj, nk = L0.shape[:3]
    fz = {tuple(int(v) % n for v, n in zip(p, (ni, nj, nk))) for p in frozen}
    off = 1 if one_based_labels else 0
    for i in range(ni):
        ax.plot(L0[i, :, 0, 0], L0[i, :, 0, 1], color=LATTICE, lw=0.6)
    for j in range(nj):
        ax.plot(L0[:, j, 0, 0], L0[:, j, 0, 1], color=LATTICE, lw=0.6)
    for i in range(ni):
        for j in range(nj):
            frozen_here = [k for k in range(nk) if (i, j, k) in fz]
            x, y = L0[i, j, 0, :2]
            if frozen_here:
                ax.plot(x, y, 'o', ms=9, mfc=FROZEN, mec=FROZEN_DARK, mew=0.8)
                ax.annotate(','.join(str(k + off) for k in frozen_here),
                            (x, y), textcoords='offset points', xytext=(0, 9),
                            ha='center', fontsize=6.5, color=FROZEN)
            else:
                ax.plot(x, y, 'o', ms=4.5, mfc=INACTIVE, mec=INACTIVE_EDGE, mew=0.5)
    ax.set_aspect('equal')
    return len(fz)


def _hh_classification(ax, cfg):
    """Wall nodes the Hicks--Henne bumps move as the upper or the lower surface."""
    _problem, geometry, points, output_name = _hh_problem(cfg)
    upper = np.asarray(geometry._point_sets[output_name]['upper'], dtype=bool)
    ax.scatter(points[upper, 0], points[upper, 1], s=10, color=BASELINE,
               label=f'upper ({int(upper.sum())})')
    ax.scatter(points[~upper, 0], points[~upper, 1], s=10, color=OPTIMIZED,
               label=f'lower ({int((~upper).sum())})')
    moment_reference(ax, cfg, projection='xy')
    ax.set_aspect('equal')
    ax.set_xlabel('x (chord)')
    ax.set_ylabel('y')
    ax.legend(loc='upper right', fontsize=8, frameon=False)


def dv_map_panels(fig, cfg, lattice=None, dvs=None, wall=None):
    """One DV map for any parameterization: returns the panel titles drawn.

    2D: (a) the basis over the wall (CST Bernstein weights, Hicks--Henne bumps,
    or the FFD box around the wall), (b) each DV's wall or control-point
    influence (unit perturbation for CST and Hicks--Henne, the saved DVs or a
    unit vector for FFD), (c) the classification (CST and Hicks--Henne
    upper/lower split, FFD free/frozen control points). 3D FFD: (a) the
    control points the DVs move (planform), (b) the freeze map, (c) the
    control-point map.
    """
    param = cfg.geometry.parameterization
    axes = fig.subplots(1, 3)
    fig.set_size_inches(18, 5.5)
    if cfg.case.ndim == 3:
        if lattice is None:
            raise ValueError('a 3D DV map needs the FFD lattice')
        shapes = shape_functions(cfg.dv.shape, lattice.shape) if cfg.dv.shape.enabled else []
        moved = deformed_lattice(lattice, np.ones(len(shapes)) if dvs is None else dvs,
                                 cfg.dv.shape)
        frozen = (frozen_lattice_points(cfg.dv.shape, lattice.shape)
                  if cfg.dv.shape.pattern == 'ffd3d' else set())
        panel_ffd_planform(axes[0], lattice, moved, frozen)
        panel_freeze_map(axes[1], lattice, frozen)
        dv_map(axes[2], cfg, lattice)
        titles = ('control points moved' + (' (unit DVs)' if dvs is None else ''),
                  'frozen control points', 'DV per control point')
    elif param == 'cst':
        basis_functions(axes[0], cfg)
        dv_influence(axes[1], cfg, lattice, None)
        cst_classification(axes[2], cfg)
        titles = ('CST basis', 'influence per DV (unit)', 'upper / lower split')
    elif param == 'hicks_henne':
        basis_functions(axes[0], cfg)
        dv_influence(axes[1], cfg, lattice, None)
        _hh_classification(axes[2], cfg)
        titles = ('Hicks-Henne bumps', 'influence per DV (unit)', 'upper / lower split')
    else:
        shapes = shape_functions(cfg.dv.shape, lattice.shape)
        ffd_box(axes[0], lattice, wall=wall)
        dv_influence(axes[1], cfg, lattice, np.ones(len(shapes)) if dvs is None else dvs)
        dv_map(axes[2], cfg, lattice)
        titles = ('FFD box and wall', 'influence per DV' + (' (unit)' if dvs is None else ''),
                  'free / frozen control points')
    for ax, title in zip(axes, titles):
        ax.set_title(title, fontsize=9)
    return list(titles)
