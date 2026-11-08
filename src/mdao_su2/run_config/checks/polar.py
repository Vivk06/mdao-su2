"""Checks of ``[polar]`` and ``[polar.urans]``, and what the ``polar`` task needs."""

#: Keys an enabled ``[polar.urans]`` states: the dual-time marching, its window and its output.
URANS_REQUIRED = ('TIME_MARCHING', 'TIME_STEP', 'TIME_ITER', 'INNER_ITER', 'RESTART_ITER',
                  'WINDOW_START_ITER', 'OUTPUT_WRT_FREQ', 'CFL_NUMBER')

#: SU2's windowed convergence test; the run is a fixed length, so the task turns it off.
URANS_WINDOW_TEST = ('CONV_WINDOW_FIELD', 'CONV_WINDOW_CAUCHY_EPS', 'CONV_WINDOW_CAUCHY_ELEMS',
                     'CONV_WINDOW_STARTITER')

#: Keys the task sets for each angle; a table that names one would be overridden silently.
URANS_TASK_OWNED = ('RESTART_SOL', 'TIME_DOMAIN', 'MESH_FILENAME', 'SOLUTION_FILENAME',
                    'RESTART_FILENAME', 'CONV_FILENAME', 'AOA', 'DEFORM_MESH',
                    'WINDOW_CAUCHY_CRIT')

#: Restart time levels each marching order reads (``RESTART_ITER``).
URANS_RESTART_LEVELS = {'DUAL_TIME_STEPPING-1ST_ORDER': 1,
                        'DUAL_TIME_STEPPING-2ND_ORDER': 2}


def _number(table, key, kind, errors):
    """``table[key]`` as ``kind``, or None after an error when it is not one."""
    if key not in table:
        return None
    try:
        return kind(float(table[key])) if kind is int else kind(table[key])
    except (TypeError, ValueError):
        errors.append(f'[polar.urans] {key}={table[key]!r} must be a number')
        return None


def check_urans_table(polar):
    """Problems of the ``[polar.urans]`` table of an enabled block."""
    from ...solver import su2_options_v85 as v85
    from ...solver.su2_options import enum_violations, fmt_passthrough
    table = polar.urans
    errors = []
    for key, value in table.items():
        if key not in v85.OPTION_KEYS:
            errors.append(f'[polar.urans] {key} is not an SU2 v8.5 option '
                          f'(commit {v85.SU2_COMMIT[:10]})')
            continue
        bad = enum_violations(key, fmt_passthrough(value))
        if bad:
            errors.append(f'[polar.urans] {key}: {bad} not accepted by SU2 v8.5')
    missing = [key for key in URANS_REQUIRED if key not in table]
    if missing:
        errors.append(f'[polar.urans] must state {missing}: the URANS run takes no '
                      f'value from the steady cfg for them (CFL_NUMBER: the steady CFL does not carry '
                      f'over to a dual-time inner loop)')
    if 'ITER' in table:
        errors.append('[polar.urans] ITER is rejected by SU2 in a time-domain run '
                      '(CConfig.cpp:4349); the pseudo-time limit is INNER_ITER and '
                      'the physical-time limit TIME_ITER')
    if 'MAX_TIME' in table:
        errors.append('[polar.urans] MAX_TIME is set by the task to twice TIME_ITER * '
                      'TIME_STEP, so the run ends on TIME_ITER')
    for key in URANS_TASK_OWNED:
        if key in table:
            errors.append(f'[polar.urans] {key} is set by the task for each angle')
    for key in URANS_WINDOW_TEST:
        if key in table:
            errors.append(f'[polar.urans] {key} is not used: the run is a fixed length '
                          f'that ends on TIME_ITER, and SU2\'s window test is off')
    for key in ('WRT_RESTART_OVERWRITE', 'WRT_SURFACE_OVERWRITE', 'WRT_VOLUME_OVERWRITE'):
        if key in table and fmt_passthrough(table[key]).upper() == 'NO':
            errors.append(f'[polar.urans] {key}= NO fails in a time-domain run '
                          f'(CConfig.cpp:4002)')

    files = table.get('OUTPUT_FILES')
    if files is not None:
        tokens = {t.strip() for t in fmt_passthrough(files).strip('() ').split(',') if t.strip()}
        if 'MESH' in tokens:
            errors.append('[polar.urans] OUTPUT_FILES may not hold MESH: SU2 v8.5\'s MESH '
                          'writer needs an undeclared boundary.dat')
        if not tokens & {'RESTART', 'RESTART_BINARY'}:
            errors.append('[polar.urans] OUTPUT_FILES must hold RESTART (or RESTART_BINARY): '
                          'the run reads the binary steady restart as its time levels')
    if 'READ_BINARY_RESTART' in table and fmt_passthrough(
            table['READ_BINARY_RESTART']).upper() == 'NO':
        errors.append('[polar.urans] READ_BINARY_RESTART= NO: the seeded time levels are '
                      'the binary steady restart (CTurbSolver.cpp:114-120)')

    marching = str(table.get('TIME_MARCHING', ''))
    if 'TIME_MARCHING' in table and marching not in URANS_RESTART_LEVELS:
        errors.append(f'[polar.urans] TIME_MARCHING={marching!r} must be one of '
                      f'{list(URANS_RESTART_LEVELS)}')
    restart = _number(table, 'RESTART_ITER', int, errors)
    start = _number(table, 'WINDOW_START_ITER', int, errors)
    total = _number(table, 'TIME_ITER', int, errors)
    step = _number(table, 'TIME_STEP', float, errors)
    levels = URANS_RESTART_LEVELS.get(marching)
    if restart is not None and levels is not None and restart != levels:
        errors.append(f'[polar.urans] RESTART_ITER={restart} must be {levels} for '
                      f'{marching}: the restart seeds that many time levels')
    if restart is not None and start is not None and start < restart:
        errors.append(f'[polar.urans] WINDOW_START_ITER={start} must be at least '
                      f'RESTART_ITER={restart} (CConfig.cpp:3979)')
    if start is not None and total is not None and total <= start:
        errors.append(f'[polar.urans] TIME_ITER={total} must exceed '
                      f'WINDOW_START_ITER={start}')
    if step is not None and step <= 0.0:
        errors.append(f'[polar.urans] TIME_STEP={step:g} must be > 0')
    return errors


def check_polar(cfg):
    errors = []
    polar = cfg.polar
    task = cfg.execution.task
    if task == 'polar' and not polar.enabled:
        errors.append('[execution] task="polar" needs [polar] enabled=true')
    if polar.enabled:
        alpha = polar.alpha
        if len(alpha) < 2:
            errors.append(f'[polar] alpha needs at least 2 angles, got {len(alpha)}')
        if any(b <= a for a, b in zip(alpha, alpha[1:])):
            errors.append(f'[polar] alpha must be strictly increasing, got {list(alpha)}')
        if len(cfg.points) > 1 and polar.mach_floor <= 0.0:
            errors.append(f'[polar] mach_floor must be > 0, got {polar.mach_floor:g}')
        if polar.run_urans:
            if polar.cl_band <= 0.0:
                errors.append(f'[polar] cl_band must be > 0, got {polar.cl_band:g}')
            if polar.urans_max_angles < 1:
                errors.append(f'[polar] urans_max_angles must be >= 1, '
                              f'got {polar.urans_max_angles}')
            if polar.urans_min_cycles < 1:
                errors.append(f'[polar] urans_min_cycles must be >= 1, '
                              f'got {polar.urans_min_cycles}')
            errors += check_urans_table(polar)
    if task == 'polar':
        free = [p.id for p in cfg.points if p.alpha != 'fixed']
        if free:
            errors.append(f'[execution] task="polar" needs alpha="fixed" at every '
                          f'point; the task sets the angle. Design alpha at {free}')
        if not cfg.solver.primal.warm_start:
            errors.append('[execution] task="polar" needs [solver.primal] '
                          'warm_start=true: each angle starts from the one before')
        if cfg.n_points > 1 and not cfg.execution.parallel_points:
            errors.append('[execution] task="polar" with several points needs '
                          '[execution] parallel_points=true: the serial multipoint '
                          'group stops at the first failed point, and the points after '
                          'it would not run that angle')
    return errors
