"""Checks of ``[objective]``, ``[dv.alpha]`` and the lift, moment and thickness
constraints."""

import math

from ..errors import ConfigError
from ..root import AXES_SPAN_Y, MOMENT_AXES, MOMENT_KINDS
from .geometry import check_shape


#: Which keys each lift mode reads.  A key outside its mode's set, written to
#: something other than its default, is an ERROR: "I set `lower` and it silently
_LIFT_MODE_KEYS = {'equals': ('equals',),
                   'min': ('lower',),
                   'range': ('lower', 'upper')}


def _lift_errors(lift):
    """Check the kind and the bound(s) it uses; reject the bounds it ignores.

    This is the one load-time check of the lift bounds.  The accessor it calls
    raises the same text when a caller builds the constraint around the loader.
    """
    if not lift.enabled:
        return []
    try:
        lift.constraint_kwargs
    except ConfigError as exc:
        return [str(exc)]

    errors = []
    used = _LIFT_MODE_KEYS[lift.kind]
    for key in used:
        val = getattr(lift, key)
        if not 0.0 < val < 2.0:
            errors.append(f'[constraints.lift] {key}={val} is not a plausible '
                          f'2D airfoil C_L target')
    for key in ('equals', 'lower', 'upper'):
        if key in used:
            continue
        val = getattr(lift, key)
        if val is not None:
            errors.append(f'[constraints.lift] {key}={val} is set but '
                          f'kind={lift.kind!r} ignores it')
    if lift.kind == 'range' and not lift.lower < lift.upper:
        errors.append(f'[constraints.lift] kind="range" needs lower < upper, '
                      f'got lower={lift.lower}, upper={lift.upper}')
    return errors


def _moment_errors(moment):
    """Check an enabled moment constraint's kind, bound presence, and range.

    This is the one load-time check of the moment bounds; the bound-presence
    text comes from the accessor the model reads.
    """
    if moment.kind not in MOMENT_KINDS:
        return [f'[constraints.moment] kind={moment.kind!r} must be one of '
                f'{list(MOMENT_KINDS)}']
    try:
        moment.constraint_kwargs
    except ConfigError as exc:
        return [str(exc)]
    if moment.kind == 'range' and moment.lower >= moment.upper:
        return ['[constraints.moment] kind="range" needs lower < '
                f'upper, got lower={moment.lower}, upper={moment.upper}']
    return []


def check_dv_and_objective(cfg):
    # An off [dv.shape] block is not range-checked; model.py
    # refuses to build without shape DVs when the selected path requires them.
    errors = check_shape(cfg) if cfg.dv.shape.enabled else []
    if not cfg.dv.alpha.per_point:
        errors.append('[dv.alpha] per_point=false is not implemented: each '
                      'point trims its own alpha to the lift constraint, so a '
                      'shared alpha would over-constrain the problem')
    obj = cfg.objective
    weights = [p.weight for p in cfg.points]
    weights_ok = (all(math.isfinite(w) and w >= 0.0 for w in weights)
                  and any(w > 0.0 for w in weights))
    try:
        # Bad weights have their own error; compiling would repeat it.
        compiled = cfg.objective_compiled if weights_ok else None
    except ConfigError as exc:
        errors.append(str(exc))
        compiled = None
    if compiled is not None:
        from ...solver.response_catalog import status
        from ..physics import profile_id
        profile = profile_id(cfg.case.physics)
        for name in compiled.responses:
            state, evidence = status(name, profile)
            if cfg.execution.task != 'run_driver':
                continue
            if state in ('unsupported', 'fd_failed'):
                errors.append(f'[objective] response {name!r} has {state} '
                              f'derivative status for {profile}; '
                              f'evidence={evidence}')
    if obj.scaler is None:
        errors.append('[objective] scaler is required next to expression')
    if obj.scaler is not None and float(obj.scaler) == 0.0:
        errors.append('[objective] scaler=0 zeroes the objective gradient')

    errors.extend(_lift_errors(cfg.constraints.lift))
    moment = cfg.constraints.moment
    if moment.enabled:
        errors.extend(_moment_errors(moment))
    if moment.axis not in MOMENT_AXES:
        errors.append(f'[constraints.moment] axis={moment.axis!r} must be one '
                      f'of {list(MOMENT_AXES)}')
    elif cfg.case.ndim == 2 and moment.axis != 'z':
        errors.append('[constraints.moment] a 2D case requires axis="z"; '
                      f'got axis={moment.axis!r}')
    pivot = moment.resolved_pivot(cfg.case.reference.origin_moment)
    if not all(math.isfinite(value) for value in pivot):
        errors.append(f'[constraints.moment] pivot must contain finite '
                      f'coordinates, got {pivot}')
    thick = cfg.constraints.thickness
    if thick.enabled:
        # The station RANGE check depends on what a station is.  Under
        # `chord_zspan` it is a chord fraction and (0, 1) is the whole domain;
        span_y = thick.axes == AXES_SPAN_Y
        for x, tmin in thick.stations:
            if span_y:
                if x < 0.0:
                    errors.append(f'[constraints.thickness] axes="span_y" '
                                  f'station y={x} must be >= 0')
            elif not 0.0 < x < 1.0:
                errors.append(f'[constraints.thickness] station x/c={x} must '
                              f'be strictly inside (0, 1)')
            if tmin <= 0.0:
                errors.append(f'[constraints.thickness] station {x}: '
                              f'min thickness must be > 0, got {tmin}')
        if span_y and thick.scaled:
            # `scaled=True` normalises each constraint by its own baseline
            # value, which turns an absolute bound of 0.077 m into "77x the
            errors.append('[constraints.thickness] axes="span_y" bounds are '
                          'absolute thicknesses; set scaled = false')
    return errors
