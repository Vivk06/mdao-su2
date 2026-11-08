"""Resolved point-local inputs and the sole operating-point weight sum."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ResolvedPoint:
    id: str
    mach: float
    weight: float
    weight_normalized: float
    alpha_mode: str
    aoa_init: float
    aoa_lower: float
    aoa_upper: float
    aoa_scaler: float
    freestream: object
    target_cl: object
    start_cl: object
    responses: tuple
    source: dict


def weight_total(points):
    """Return the total relative weight for configured or resolved points."""
    return float(sum(point.weight for point in points))


def resolve_points(cfg):
    """Resolve point overrides against case defaults and normalize positive weights."""
    normalizer = weight_total(point for point in cfg.points if point.weight > 0.0)
    if not normalizer:
        return tuple()
    result = []
    for point in cfg.points:
        base = cfg.case.freestream
        freestream_values = {key: getattr(base, key) for key in
                             ('temperature', 'reynolds_number', 'reynolds_length',
                              'pressure', 'dimensionalization')}
        freestream_values.update(point.freestream or {})
        freestream = type(base)(**freestream_values)
        source = {
            'mach': 'points', 'weight': 'points', 'alpha': 'points',
            'aoa_init': 'points', 'aoa_lower': 'points', 'aoa_upper': 'points',
            'aoa_scaler': 'points', 'target_cl': 'points' if point.target_cl is not None else 'constraints.lift',
            'start_cl': 'points' if point.start_cl is not None else 'none',
            'responses': 'points',
            'freestream': {key: ('points.freestream' if key in (point.freestream or {}) else 'case.freestream')
                           for key in ('temperature', 'reynolds_number', 'reynolds_length', 'pressure', 'dimensionalization')},
        }
        result.append(ResolvedPoint(
            id=point.id, mach=point.mach, weight=point.weight,
            weight_normalized=point.weight / normalizer if point.weight > 0 else 0.0,
            alpha_mode=point.alpha, aoa_init=point.aoa_init,
            aoa_lower=point.aoa_lower, aoa_upper=point.aoa_upper,
            aoa_scaler=point.aoa_scaler, freestream=freestream,
            target_cl=(point.target_cl if point.target_cl is not None else
                       (cfg.constraints.lift.trim_target if cfg.constraints.lift.applies(point.id) else None)),
            start_cl=point.start_cl,
            responses=point.responses,
            source=source))
    return tuple(result)


def resolved_points_dict(points):
    """JSON-ready resolved point records."""
    return [asdict(point) for point in points]
