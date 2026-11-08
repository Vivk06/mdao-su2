"""Checks of ``[execution] task``, ``[optimizer]`` and ``[trim]``, and the single
derivative-routing decision point."""

import warnings

from ..root import OPTIMIZERS, TASKS
from .verification import (check_band_discriminating_power, check_cancellation_cr,
                           check_moment_pivot, check_verification_bands)

FD_FORMS = ('central', 'forward', 'backward')
_STANDARD_OUTPUTS = ('C_L', 'C_D', 'C_M')


def check_optimizer_and_task(cfg):
    errors = []
    task = cfg.execution.task
    record = cfg.verification.record_shape_sensitivities
    if not isinstance(record, bool):
        errors.append('[verification] record_shape_sensitivities must be a bool')
    elif record and (task != 'check_totals' or cfg.execution.parallel_points
                     or cfg.geometry.parameterization != 'ffd' or cfg.remesh.enabled):
        errors.append('[verification] record_shape_sensitivities requires '
                      'fixed-mesh, nonparallel FFD check_totals')
    if task not in TASKS:
        errors.append(f'[execution] task={task!r} unknown; '
                      f'choose from {list(TASKS)}')
    if cfg.optimizer.name not in OPTIMIZERS:
        errors.append(f'[optimizer] name={cfg.optimizer.name!r} unknown; '
                      f'choose from {list(OPTIMIZERS)}')
    if cfg.verification.form not in FD_FORMS:
        errors.append(f'[verification] form={cfg.verification.form!r} unknown; '
                      f'choose from {list(FD_FORMS)}')
    for name in ('shape_step', 'alpha_step'):
        if getattr(cfg.verification, name) <= 0.0:
            errors.append(f'[verification] {name} must be > 0')
    from ...solver.response_catalog import CATALOG
    allowed_alpha_outputs = {'C_L', 'C_D', 'C_M'} | {r.output for r in CATALOG}
    unknown_alpha_outputs = sorted(
        set(cfg.verification.alpha_outputs) - allowed_alpha_outputs)
    if unknown_alpha_outputs:
        errors.append('[verification] alpha_outputs contains unknown scenario '
                      f'output(s) {unknown_alpha_outputs}; allowed: '
                      f'{sorted(allowed_alpha_outputs)}')
    if len(set(cfg.verification.alpha_outputs)) != \
            len(cfg.verification.alpha_outputs):
        errors.append('[verification] alpha_outputs must not contain duplicates')
    allowed_shape_outputs = allowed_alpha_outputs
    unknown_shape_outputs = sorted(
        set(cfg.verification.shape_outputs) - allowed_shape_outputs)
    if unknown_shape_outputs:
        errors.append('[verification] shape_outputs contains unknown scenario '
                      f'output(s) {unknown_shape_outputs}; allowed: '
                      f'{sorted(allowed_shape_outputs)}')
    if len(set(cfg.verification.shape_outputs)) != \
            len(cfg.verification.shape_outputs):
        errors.append('[verification] shape_outputs must not contain duplicates')
    errors += check_cancellation_cr(cfg)
    errors += check_band_discriminating_power(cfg)
    errors += check_verification_bands(cfg)
    errors += check_moment_pivot(cfg)
    trim = cfg.trim
    if task == 'trim':
        # Both are the task's whole content; without either it would run one
        # analysis and call it a trim.
        if not trim.enabled:
            errors.append('[execution] task="trim" needs [trim] enabled=true')
        if not _trim_has_targets(cfg):
            errors.append('[execution] task="trim" needs [constraints.lift] '
                          'enabled=true, or a start_cl on every alpha="design" '
                          'point -- there is no target to trim to')
    if trim.enabled:
        lo, hi = trim.alpha_clip
        if lo >= hi:
            errors.append(f'[trim] alpha_clip must be increasing, got {lo} >= {hi}')
        if trim.max_iter < 1:
            errors.append(f'[trim] max_iter must be >= 1, got {trim.max_iter}')
        if trim.tol <= 0.0:
            errors.append(f'[trim] tol must be > 0, got {trim.tol}')
        if trim.dcl_da_init == 0.0:
            errors.append('[trim] dcl_da_init must be non-zero (it is the '
                          'secant seed and appears in a denominator)')
        if task == 'run_driver' and not _trim_has_targets(cfg):
            # alpha_trim reads lift.trim_target unless every trimmed point has
            # its own start_cl; with neither there is no target to trim to.
            errors.append('[trim] enabled=true with [constraints.lift] '
                          'enabled=false: run_driver would trim alpha to a C_L '
                          'target nothing asked for. Set [trim] enabled=false, '
                          'or give every alpha="design" point a start_cl.')
        fixed = [p.id for p in cfg.points if p.aoa_lower == p.aoa_upper]
        if fixed:
            errors.append(f'[trim] enabled=true but alpha is fixed '
                          f'(aoa_lower == aoa_upper) at point(s) {fixed}: trim '
                          f'clips to alpha_clip, not the point bounds. '
                          f'Set [trim] enabled=false.')
    lift = cfg.constraints.lift
    if lift.enabled and lift.kind != 'equals':
        for i, pt in enumerate(cfg.points):
            if lift.applies(pt.id) and pt.alpha == 'design' and pt.aoa_upper <= pt.aoa_init:
                errors.append(
                    f'[constraints.lift] kind={lift.kind!r} with point '
                    f'{pt.id!r}: aoa_upper ({pt.aoa_upper}) must be strictly '
                    f'above aoa_init ({pt.aoa_init}) -- under C_L >= x '
                    f'nothing bounds alpha from above except aoa_upper')
    return errors


def _trim_has_targets(cfg):
    """True when the trim has a lift target: the lift constraint is enabled, or
    every alpha="design" point carries its own ``start_cl``."""
    if cfg.constraints.lift.enabled:
        return True
    design = [p for p in cfg.points if p.alpha == 'design']
    return bool(design) and all(p.start_cl is not None for p in design)


def _response_records(point):
    from ...solver.response_catalog import lookup
    try:
        return tuple(lookup(name) for name in getattr(point, 'responses', ()))
    except ValueError:
        # The points check owns the user-facing unknown-response error;
        # routing must remain best-effort so validation reports all errors.
        return ()


def _uses_response_path(cfg, point):
    """True when ``point`` needs the reverse-only seam: a multi-marker or 3D wall, or catalog responses.

    ``SU2Builder`` reads its wall as 2D line elements, so a 3D wall
    with one design marker also takes the seam.
    """
    case = getattr(cfg, 'case', None)
    return (getattr(case, 'is_multi_marker_wall', False)
            or getattr(case, 'ndim', 2) != 2
            or any(record.output not in _STANDARD_OUTPUTS
                   for record in _response_records(point)))


def resolve_derivative_mode(cfg):
    """Return ``(requested, resolved, reason)`` for ``[execution] derivative_mode``.

    Forward products exist only on the 2D fixed-mesh path. A
    ``fwd`` or ``auto`` request on any other configuration falls back to
    ``rev``; the totals are the same, only the cost differs. ``auto``
    on the forward-capable path stays ``auto`` for OpenMDAO to resolve.
    """
    requested = getattr(getattr(cfg, 'execution', None), 'derivative_mode', 'rev')
    if requested == 'rev':
        return requested, 'rev', 'requested'
    rev_only = _reverse_only_reasons(cfg)
    if rev_only:
        return requested, 'rev', f'{"; ".join(rev_only)} is reverse-mode only'
    return requested, requested, 'fixed-mesh 2D path provides forward products'


def _reverse_only_reasons(cfg):
    """Every condition of ``cfg`` whose components assert reverse mode."""
    return [why for flag, why in (
        (any(getattr(point, 'responses', ()) for point in cfg.points),
         'catalog responses (ResponseSU2AeroComp)'),
        (cfg.remesh.enabled, 'remesh (no forward product through a regenerated mesh)'),
        (cfg.execution.parallel_points,
         'parallel_points (failure handling supports reverse mode only)'),
        (cfg.case.is_multi_marker_wall,
         'a multi-marker wall (ResponseSU2AeroComp)'),
        (not cfg.case.is_multi_marker_wall and cfg.case.ndim != 2,
         'a 3D wall (ResponseSU2AeroComp)'),
    ) if flag]


def warn_derivative_fallback(cfg):
    """Warn when an unsupported ``fwd``/``auto`` request falls back to ``rev``."""
    requested, resolved, reason = resolve_derivative_mode(cfg)
    if resolved != requested:
        warnings.warn(f'[execution] derivative_mode={requested!r} falls back to '
                      f'{resolved!r}: {reason}', stacklevel=3)


def derivative_route(cfg, point=None):
    """Return ``(builder, geometry, allowed_modes, reason)`` for ``cfg``.

    ``builder`` is the solver builder for ``point`` (the first point when
    omitted). This centralizes the dispatch predicates without instantiating
    either component, preserving lazy imports and the component assertions.
    """
    from ...geometry.factory import PARAMETERIZATIONS
    from ...solver.builder import SU2Builder

    _requested, mode, reason = resolve_derivative_mode(cfg)
    if point is None and cfg.points:
        point = cfg.points[0]
    if _uses_response_path(cfg, point):
        from ...solver.seam_comp import ResponseSU2Builder
        builder = ResponseSU2Builder
    elif mode == 'rev':
        builder = SU2Builder
    else:
        from ...solver.bidir_comp import BidirSU2Builder
        builder = BidirSU2Builder
    allowed = ('rev',) if mode == 'rev' else ('fwd', 'auto', 'rev')

    if cfg.geometry.parameterization == 'ffd':
        from pygeo.mphys import OM_DVGEOCOMP
        from ...geometry.ffd import FFDGeoBidirComp
        geometry = OM_DVGEOCOMP if mode == 'rev' else FFDGeoBidirComp
    elif cfg.geometry.parameterization == 'cst':
        from ...geometry.cst import CSTGeoComp
        geometry = CSTGeoComp
    elif cfg.geometry.parameterization == 'hicks_henne':
        from ...geometry.hicks_henne import HicksHenneGeoComp
        geometry = HicksHenneGeoComp
    else:
        raise ValueError(f'geometry parameterization {cfg.geometry.parameterization!r} '
                         f'is not supported; choose from {list(PARAMETERIZATIONS)}')
    return builder, geometry, allowed, reason

