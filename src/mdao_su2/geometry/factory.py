"""Geometry-parameterization dispatch. Physics values stay in RunConfig."""

PARAMETERIZATIONS = ('ffd', 'cst', 'hicks_henne')

__all__ = ['PARAMETERIZATIONS', 'build_geometry_component', 'add_geometry_dvs',
           'add_shape_dvs', 'embed_and_add_shape_dvs', 'add_geometry_constraints']


def _ffd_path(cfg):
    """Resolve the selected entry from ``geometry.ffd_files``."""
    return cfg.abs_path(cfg.geometry.ffd_files[cfg.geometry.ffd])


def _unsupported(cfg):
    raise ValueError(
        f'geometry parameterization {cfg.geometry.parameterization!r} is not '
        f'supported; choose from {list(PARAMETERIZATIONS)}')


def build_geometry_component(cfg):
    """Build the OpenMDAO geometry component selected by ``cfg``."""
    parameterization = cfg.geometry.parameterization
    from ..run_config.checks.task import derivative_route
    _builder, route_geometry, _allowed, _reason = derivative_route(cfg)

    if parameterization == 'ffd':
        ffd_file = _ffd_path(cfg)
        options = cfg.geometry.dvgeo_options(ffd_file) or None
        return route_geometry(file=ffd_file, type='ffd', options=options)

    if parameterization == 'cst':
        return route_geometry(
            dat_file=cfg.dat_path,
            n_cst=cfg.geometry.n_cst,
        )

    if parameterization == 'hicks_henne':
        shape = cfg.dv.shape
        upper_only = all(
            point.aoa_lower == point.aoa_upper == 0.0
            for point in cfg.points)
        return route_geometry(
            n_dv=shape.n_dv,
            distribution=shape.distribution,
            bump_width=shape.bump_width,
            upper_only=upper_only,
        )

    _unsupported(cfg)


def add_geometry_dvs(geometry, builder, cfg):
    """Embed the wall and register the enabled geometry DVs, in ``cfg.geometry_dvs`` order.

    Returns ``{name: (n, per_direction_counts)}``. ``[geometry] pin_symmetry``
    is applied once the wall points are embedded, so the pinned surface is the
    geometry component's own output.
    """
    dvs = {}
    if cfg.dv.shape.enabled:
        dvs['shape'] = embed_and_add_shape_dvs(geometry, builder, cfg)
    elif cfg.geometry.parameterization == 'ffd':
        from .ffd import embed_wall
        embed_wall(geometry, builder)
    else:
        from ..run_config import ConfigError
        raise ConfigError(f'[dv.shape] off is supported for parameterization "ffd" '
                          f'only; got {cfg.geometry.parameterization!r}')
    if cfg.dv.twist.enabled:
        from .twist import add_twist_dvs
        n = add_twist_dvs(geometry, cfg.dv.twist)
        dvs['twist'] = (n, (n,))
    from .symmetry_pin import apply_symmetry_pin
    apply_symmetry_pin(geometry, builder, cfg)
    return dvs


def add_shape_dvs(geometry, builder, cfg):
    """``(n_shape, counts)`` of :func:`add_geometry_dvs` (callers that need shape only)."""
    return add_geometry_dvs(geometry, builder, cfg).get('shape', (0, (0,)))


def embed_and_add_shape_dvs(geometry, builder, cfg):
    parameterization = cfg.geometry.parameterization

    if parameterization == 'ffd':
        from .ffd import add_ffd_shape_dvs
        return add_ffd_shape_dvs(geometry, builder, shape_cfg=cfg.dv.shape)

    if parameterization == 'cst':
        from mphys import MPhysVariables
        _V = MPhysVariables.Aerodynamics.Surface
        geometry.nom_add_discipline_coords(_V.Geometry, points=builder.get_surface_points())
        return geometry.add_shape_dv()

    if parameterization == 'hicks_henne':
        from mphys import MPhysVariables
        _V = MPhysVariables.Aerodynamics.Surface
        geometry.nom_set_surface_segments(
            getattr(builder, 'wall_segments', None))
        geometry.nom_add_discipline_coords(
            _V.Geometry, points=builder.get_surface_points())
        return geometry.add_shape_dv()

    _unsupported(cfg)


def add_geometry_constraints(geometry, builder, cfg):
    """Register geometric constraints for the selected parameterization."""
    parameterization = cfg.geometry.parameterization
    if parameterization not in PARAMETERIZATIONS:
        _unsupported(cfg)
    if parameterization in ('cst', 'hicks_henne'):
        from ..run_config import ConfigError
        from ..run_config.root import AXES_CHORD_ZSPAN

        axes = cfg.constraints.thickness.axes
        if axes != AXES_CHORD_ZSPAN:
            raise ConfigError(
                f'[constraints.thickness] {parameterization} requires '
                'axes="chord_zspan"; '
                f'got {axes!r}')

    from .ffd import add_geometry_constraints_from_config

    return add_geometry_constraints_from_config(geometry, builder, cfg)
