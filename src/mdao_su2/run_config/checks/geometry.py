"""Checks of the parameterization, ``[dv.shape]``, ``[dv.twist]`` and the
curvature constraint."""

from ...geometry.factory import PARAMETERIZATIONS
from ..errors import ConfigError
from ..root import AXES_CHORD_ZSPAN


_FFD_ONLY_GEOMETRY_KEYS = frozenset(('ffd', 'ffd_files', 'ffd_order'))


def check_shape(cfg):
    """The parameterization and [dv.shape] rules; run only when shape DVs are on."""
    errors = []
    parameterization = cfg.geometry.parameterization
    shape = cfg.dv.shape
    if parameterization not in PARAMETERIZATIONS:
        errors.append(
            f'[geometry] parameterization={parameterization!r} is not '
            f'implemented; supported: {list(PARAMETERIZATIONS)}')
    if parameterization == 'cst':
        if cfg.case.ndim != 2:
            errors.append(
                f'[geometry] parameterization="cst" requires ndim=2, got '
                f'ndim={cfg.case.ndim} (DVGeometryCST fits a single '
                f'airfoil section; use "ffd" for 3D cases)')
        if not cfg.geometry.dat_file:
            errors.append(
                '[geometry] parameterization="cst" requires dat_file '
                '(Selig-format airfoil .dat, case-root-relative)')
        thick = cfg.constraints.thickness
        if thick.enabled and thick.axes != AXES_CHORD_ZSPAN:
            errors.append(
                f'[constraints.thickness] CST requires axes="chord_zspan"; '
                f'got axes={thick.axes!r}')
        own_keys = getattr(cfg, '_own_geometry_keys', frozenset())
        own_ffd = sorted(own_keys & _FFD_ONLY_GEOMETRY_KEYS)
        if own_ffd:
            errors.append(
                f'[geometry] parameterization="cst" but the run config sets '
                f'FFD-only key(s) {own_ffd}; remove them (inherited FFD keys '
                f'from the case.toml include are ignored)')
    if parameterization == 'hicks_henne':
        if cfg.case.ndim != 2:
            errors.append(
                f'[geometry] parameterization="hicks_henne" requires ndim=2, '
                f'got ndim={cfg.case.ndim}')
        thick = cfg.constraints.thickness
        if thick.enabled and thick.axes != AXES_CHORD_ZSPAN:
            errors.append(
                '[constraints.thickness] Hicks-Henne requires '
                f'axes="chord_zspan"; got axes={thick.axes!r}')
        if shape.n_dv < 1:
            errors.append(f'[dv.shape] n_dv must be >= 1, got {shape.n_dv}')
        if shape.distribution not in shape.DISTRIBUTIONS:
            errors.append(
                f'[dv.shape] distribution={shape.distribution!r} must be one '
                f'of {list(shape.DISTRIBUTIONS)}')
        if not 1 <= shape.bump_width <= 15:
            errors.append(
                f'[dv.shape] bump_width must be in [1, 15], got '
                f'{shape.bump_width}')
    if shape.bound <= 0.0:
        errors.append(f'[dv.shape] bound must be > 0, got {shape.bound}')
    else:
        try:
            shape.resolved_scaler
        except ConfigError as exc:
            errors.append(str(exc))
    from ..root import SHAPE_PATTERNS
    if parameterization == 'ffd' and shape.pattern not in SHAPE_PATTERNS:
        errors.append(f'[dv.shape] pattern={shape.pattern!r} is not '
                      f'implemented; allowed: {SHAPE_PATTERNS} '
                      f'(geometry/ffd.add_ffd_shape_dvs)')
    if parameterization == 'ffd' and shape.pattern == 'bspline':
        if shape.degree < 1:
            errors.append(f'[dv.shape] degree must be >= 1, got {shape.degree}')
        if shape.n_basis < 0:
            errors.append(f'[dv.shape] n_basis must be >= 0 '
                          f'(0 = auto), got {shape.n_basis}')
    curv = cfg.constraints.curvature
    if curv.enabled:
        if curv.end_x <= curv.start_x:
            errors.append(f'[constraints.curvature] end_x must exceed start_x, '
                          f'got {curv.start_x} .. {curv.end_x}')
        if curv.n_pts < 5:
            errors.append(f'[constraints.curvature] pyGeo requires n_pts >= 5, '
                          f'got {curv.n_pts}')
        if curv.upper <= 1.0:
            errors.append(
                f'[constraints.curvature] upper={curv.upper} is a bound on '
                f'mean(kappa^2) scaled so the BASELINE is 1.0; <= 1.0 forbids '
                f'the starting shape')
        if curv.scaler <= 0.0:
            errors.append(
                f'[constraints.curvature] scaler={curv.scaler} must be > 0: '
                f'pyOptSparse multiplies both the value and the bound by it, '
                f'so 0 deletes the constraint and a negative flips its sense')
        if curv.curvature_type not in ('mean', 'aggregated'):
            errors.append(
                f'[constraints.curvature] curvature_type must be "mean" or '
                f'"aggregated", got {curv.curvature_type!r}')
    dirs = (() if parameterization == 'hicks_henne'
            else shape.resolved_directions)
    if (parameterization != 'hicks_henne'
            and len(shape.bounds) not in (0, len(dirs))):
        errors.append(f'[dv.shape] bounds has {len(shape.bounds)} entries but '
                      f'there are {len(dirs)} directions; must be 0 (broadcast '
                      f'from bound) or {len(dirs)}')
    for i, b in enumerate(
            () if parameterization == 'hicks_henne'
            else shape.resolved_bounds):
        if b <= 0.0:
            errors.append(f'[dv.shape] bounds[{i}] must be > 0, got {b}')
    if len(dirs) > 1:
        if shape.pattern != 'ffd3d':
            errors.append(f'[dv.shape] multiple directions require '
                          f'pattern="ffd3d", got {shape.pattern!r}')
        for i in range(len(dirs)):
            for j in range(i + 1, len(dirs)):
                dot = sum(a * b for a, b in zip(dirs[i], dirs[j]))
                if abs(dot) > 1 - 1e-12:
                    errors.append(
                        f'[dv.shape] directions[{i}] and [{j}] are parallel '
                        f'or duplicate — the Jacobian would be rank-deficient')
    if parameterization != 'hicks_henne' and shape.n_le < 0:
        errors.append(f'[dv.shape] n_le must be >= 0 (0 = off), '
                      f'got {shape.n_le}')
    only_y = (len(dirs) == 1
              and all(abs(a - b) < 1e-12
                      for a, b in zip(dirs[0], (0.0, 1.0, 0.0))))
    if parameterization == 'ffd' and shape.pattern == 'thickness2d' and not only_y:
        errors.append('[dv.shape] pattern="thickness2d" moves control points in '
                      'y only; set the single direction "y"')
    if getattr(shape, 'outward_only', False):
        # A zero lower bound is only "outward" when a positive DV moves every
        # control point it touches away from the section.
        if parameterization != 'ffd' or shape.pattern not in ('ffd3d',
                                                              'thickness2d'):
            errors.append(
                '[dv.shape] outward_only needs parameterization="ffd" with '
                'pattern "thickness2d" or "ffd3d"; Hicks-Henne on a fixed-alpha '
                'case is already outward-only')
        elif not only_y:
            errors.append('[dv.shape] outward_only needs the single direction "y"')
        elif shape.pattern == 'ffd3d' and tuple(shape.freeze_j) != (0,):
            errors.append(
                '[dv.shape] outward_only with pattern="ffd3d" needs freeze_j = '
                '[0]: only the upper row may move, so that +y is outward')
    return errors


def check_twist(cfg):
    """[dv.twist]: an FFD wing, sane bounds, a TE axis inside the box."""
    t = cfg.dv.twist
    if not t.enabled:
        return []
    errors = []
    if cfg.geometry.parameterization != 'ffd':
        errors.append('[dv.twist] needs [geometry] parameterization = "ffd" (pyGeo '
                      f'reference axis); got {cfg.geometry.parameterization!r}')
    if cfg.case.ndim != 3:
        errors.append(f'[dv.twist] needs a 3D case; [case] ndim = {cfg.case.ndim}')
    if t.n < 2:
        errors.append(f'[dv.twist] n = {t.n}: a twist axis needs >= 2 sections')
    if not t.lower_deg < t.upper_deg:
        errors.append(f'[dv.twist] lower_deg = {t.lower_deg} must be < upper_deg = '
                      f'{t.upper_deg}')
    elif not t.lower_deg <= t.init_deg <= t.upper_deg:
        errors.append(f'[dv.twist] init_deg = {t.init_deg} is outside '
                      f'[{t.lower_deg}, {t.upper_deg}]')
    if not t.scaler > 0.0:
        errors.append(f'[dv.twist] scaler must be > 0, got {t.scaler}')
    if not 0.0 < t.axis_x_fraction <= 1.0:
        errors.append(f'[dv.twist] axis_x_fraction = {t.axis_x_fraction} must be in '
                      f'(0, 1] (a fraction of the FFD box chord)')
    if cfg.execution.task == 'check_totals':
        step = cfg.verification.twist_step
        if step is None or not step > 0.0:
            errors.append('[verification] twist_step (degrees) must be set > 0 for a '
                          'check_totals run with [dv.twist] on; none is assumed')
    return errors


def ffd_lattice_shape(path):
    """``(ni, nj, nk)`` from the header of a single-block Plot3D file."""
    with open(path) as fh:
        tokens = fh.read(256).split()
    return tuple(int(t) for t in tokens[1:4])
