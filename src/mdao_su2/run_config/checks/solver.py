"""Checks of the ``[solver]`` blocks, ``[solver.su2_extra]`` passthrough keys
and the enum values the cfg writer emits."""



def check_solver(cfg):
    errors = []
    primal, adjoint = cfg.solver_blocks

    # zero-level adjoint multigrid, deliberately unoverridable: a non-zero adjoint MGLEVEL does not
    # fail, it silently corrupts the mesh sensitivities.
    if adjoint.mglevel != 0:
        errors.append(
            f'[solver.adjoint] mglevel must be 0, got {adjoint.mglevel} '
            f'(multigrid corrupts discrete-adjoint mesh sensitivities; '
            f'this rule has no override)')
    if primal.mglevel < 0:
        errors.append(f'[solver.primal] mglevel must be >= 0, '
                      f'got {primal.mglevel}')
    for name, block in (('primal', primal), ('adjoint', adjoint)):
        if block.iter < 1:
            errors.append(f'[solver.{name}] iter must be >= 1, got {block.iter}')
        if block.conv_residual_minval > 0:
            errors.append(
                f'[solver.{name}] conv_residual_minval={block.conv_residual_minval} '
                f'must be negative (it is a log10 residual floor)')

    # CFL_ADAPT and CFL_ADAPT_PARAM are no-ops on the discrete adjoint --
    # CFluidIteration.cpp:135 guards the adaptation with !disc_adj.  Not an
    if 'cfl_adapt' in cfg.solver.adjoint._explicit_keys \
            or 'cfl_adapt_param' in cfg.solver.adjoint._explicit_keys:
        import warnings
        warnings.warn(
            '[solver.adjoint] cfl_adapt/cfl_adapt_param have NO effect on the '
            'discrete adjoint (CFluidIteration.cpp:135 guards CFL '
            'adaptation with !disc_adj). Use '
            '[solver.adjoint_settings] relaxation_factor for adjoint stability.',
            stacklevel=2)

    # a warm-started primal makes the FD reference depend on the previous
    # solve's state, and that noise is larger than the shape derivative signal.
    if cfg.execution.task == 'check_totals' and primal.warm_start:
        errors.append(
            '[solver.primal] warm_start must be false when '
            'task="check_totals" (warm-start noise swamps the shape '
            'signal in the FD reference)')
    return errors


def check_su2_extra(cfg):
    """passthrough keys exist in SU2 v8.5, shadow nothing typed, and
    use scanned enum values."""
    from ...solver.su2_options import (enum_violations, fmt_passthrough,
                                      passthrough_blocked, side_passthrough_blocked)
    errors = []
    for side, table in cfg.solver.su2_extra.items():
        where = '[solver.su2_extra]' + ('' if side == 'both' else f'.{side}')
        for key, value in table.items():
            if key == 'TIME_DOMAIN' and fmt_passthrough(value).upper() == 'YES':
                errors.append(
                    f'{where} {key}: a time-domain run is not judged by the steady '
                    f'components (final residual against CONV_RESIDUAL_MINVAL) and '
                    f'adjoint tasks never run one; time-domain keys go in '
                    f'[polar.urans]')
                continue
            why = side_passthrough_blocked(key) if side != 'both' else ''
            if why:
                errors.append(f'{where} {key}: {why}')
                continue
            bad = enum_violations(key, fmt_passthrough(value))
            if bad:
                errors.append(f'{where} {key}: {bad} not accepted by SU2 v8.5')
                continue
            why = passthrough_blocked(key)
            if why:
                errors.append(f'{where} {key}: {why}')
    return errors


def check_emitted_enums(cfg):
    """Every enum-valued key the writer emits uses a value SU2 v8.5 accepts."""
    from ...solver.cfg_file import parse_cfg_content
    from ...solver.point_cfg_writer import generate_cfg_content
    from ...solver.su2_options import enum_violations
    errors = []
    point = cfg.points[0]
    for is_adjoint in (False, True):
        text = generate_cfg_content(cfg, point, is_adjoint)
        for key, value in parse_cfg_content(text).items():
            bad = enum_violations(key, value)
            if bad:
                errors.append(f'{key}= {value}: {bad} not accepted by SU2 v8.5')
    return sorted(set(errors))
