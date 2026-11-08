"""Checks of ``[execution]`` parallel points and the ``[output]`` table."""

import os


def check_execution(cfg, comm):
    errors = []
    if cfg.execution.parallel_points:
        if cfg.n_points < 2:
            errors.append('[execution] parallel_points needs at least 2 points')
        if comm is not None and comm.size < cfg.n_points:
            errors.append(
                f'[execution] parallel_points requires np >= n_points '
                f'({cfg.n_points}), got np={comm.size}')
        if comm is not None and comm.size % cfg.n_points != 0:
            errors.append(
                f'[execution] parallel_points requires np divisible by '
                f'n_points ({cfg.n_points}); np={comm.size} gives '
                f'{comm.size % cfg.n_points} leftover rank(s) that would '
                f'IndexError in SolverConsole._log_path')
    if cfg.output.on_exists not in ('fail', 'resume', 'overwrite'):
        errors.append(f'[output] on_exists={cfg.output.on_exists!r} unknown; '
                      f'choose from ["fail", "resume", "overwrite"]')
    if cfg.output.run_id == 'auto' and (
            os.path.basename(cfg.name) != cfg.name or '\\' in cfg.name):
        errors.append('[output] auto run_id needs a name without path separators')
    # [output.snapshots]: which surface and volume files each evaluation keeps.
    snaps = cfg.output.snapshots
    if snaps.mode not in ('off', 'surface', 'volume'):
        errors.append(f'[output.snapshots] mode={snaps.mode!r} unknown; '
                      f'choose from ["off", "surface", "volume"]')
    if snaps.keep_volume not in ('none', 'baseline_latest', 'all'):
        errors.append(
            f'[output.snapshots] keep_volume={snaps.keep_volume!r} unknown; '
            f'choose from ["none", "baseline_latest", "all"]')
    if snaps.every_n < 1:
        errors.append(f'[output.snapshots] every_n={snaps.every_n} must be >= 1')
    return errors
