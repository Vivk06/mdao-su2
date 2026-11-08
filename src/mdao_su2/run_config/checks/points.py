"""Checks of the ``[[points]]`` table: ids, weights, Mach, angle of attack,
free-stream overrides and responses; and of the constraints that name points
(``[constraints.lift|moment] points``, ``[[constraints.linear]]``)."""

import math



def _constraint_point_errors(cfg, ids):
    """``[constraints.lift] points`` and ``[constraints.moment] points``: known,
    unique, non-empty ids; every point with its own ``target_cl`` listed."""
    errors = []
    for name in ('lift', 'moment'):
        constraint = getattr(cfg.constraints, name)
        listed = constraint.points
        if not constraint.enabled or listed is None:
            continue
        ctx = f'[constraints.{name}] points'
        if not listed:
            errors.append(f'{ctx} is empty; disable the constraint instead')
        unknown = [pid for pid in listed if pid not in ids]
        if unknown:
            errors.append(f'{ctx}: {unknown} are not [[points]] ids')
        dupes = sorted({pid for pid in listed if listed.count(pid) > 1})
        if dupes:
            errors.append(f'{ctx}: {dupes} listed more than once')
        if name == 'lift':
            dropped = [p.id for p in cfg.points
                       if p.target_cl is not None and p.id not in listed]
            if dropped:
                errors.append(f'{ctx}: point(s) {dropped} set target_cl but are not '
                              f'listed, so their lift target would be ignored')
    return errors


def _linear_constraint_errors(cfg, physics):
    """``[[constraints.linear]]``: unique names that make model
    paths, known points, responses the point provides, finite numbers, and
    either ``equals`` or ``lower``/``upper``. Under ``run_driver`` a response
    whose derivative status is ``unsupported`` or ``fd_failed`` is rejected."""
    from ...optimization.expression import LEGACY_OUTPUTS
    from ...solver.response_catalog import lookup, status
    errors = []
    names = [c.name for c in cfg.constraints.linear]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        errors.append(f'[[constraints.linear]] duplicate name(s) {dupes}')
    by_id = {p.id: p for p in cfg.points}
    for c in cfg.constraints.linear:
        ctx = f'[[constraints.linear]] {c.name!r}'
        if not c.name.isidentifier():
            errors.append(f'{ctx}: name must be letters, digits and underscores, '
                          f'not starting with a digit (it names the model '
                          f'subsystem {c.component!r})')
        if not c.terms:
            errors.append(f'{ctx}: terms is empty')
        numbers = {'offset': c.offset, 'scaler': c.scaler,
                   **{k: getattr(c, k) for k in ('equals', 'lower', 'upper')
                      if getattr(c, k) is not None}}
        numbers.update({f'terms[{i}] coeff': t.coeff for i, t in enumerate(c.terms)})
        bad = [k for k, v in numbers.items() if not math.isfinite(v)]
        if bad:
            errors.append(f'{ctx}: {bad} must be finite')
        if math.isfinite(c.scaler) and c.scaler <= 0.0:
            errors.append(f'{ctx}: scaler must be > 0, got {c.scaler}')
        if c.equals is None and c.lower is None and c.upper is None:
            errors.append(f'{ctx}: states no bound; give equals, or lower, '
                          f'upper or both')
        elif c.equals is not None and (c.lower is not None or c.upper is not None):
            errors.append(f'{ctx}: equals excludes lower and upper; give one form')
        elif c.lower is not None and c.upper is not None and not c.lower < c.upper:
            errors.append(f'{ctx}: needs lower < upper, got lower={c.lower}, '
                          f'upper={c.upper}; write equals for one value')
        pairs = [(t.point, t.response) for t in c.terms]
        repeated = sorted({p for p in pairs if pairs.count(p) > 1})
        if repeated:
            errors.append(f'{ctx}: (point, response) {repeated} listed more than '
                          f'once; give each pair one coefficient')
        for i, term in enumerate(c.terms):
            where = f'{ctx} terms[{i}]'
            point = by_id.get(term.point)
            if point is None:
                errors.append(f'{where}: point {term.point!r} is not a [[points]] id')
                continue
            try:
                record = lookup(term.response)
            except ValueError as exc:
                errors.append(f'{where}: {exc}')
                continue
            if record.output not in LEGACY_OUTPUTS:
                requested = set()
                for name in point.responses:
                    try:
                        requested.add(lookup(name).output)
                    except ValueError:
                        pass        # the [[points]] check names it
                if record.output not in requested:
                    errors.append(f'{where}: response {term.response!r} '
                                  f'({record.output}) is not in point '
                                  f'{term.point!r} responses; add it there so the '
                                  f'solver component provides it')
            state, evidence = status(term.response, physics)
            if cfg.execution.task == 'run_driver' and state in ('unsupported', 'fd_failed'):
                errors.append(f'{where}: response {term.response!r} has {state} '
                              f'derivative status for {physics}; evidence={evidence}')
    return errors


def check_points(cfg):
    errors = []
    if cfg.n_points < 1:
        errors.append('at least one [[points]] entry is required')
        return errors

    ids = cfg.point_ids
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        errors.append(f'[[points]] duplicate id(s) {dupes}; ids name output '
                      f'files, so they must be unique')
    if all(p.weight == 0.0 for p in cfg.points):
        errors.append('[[points]] at least one weight must be > 0')
    errors += _constraint_point_errors(cfg, ids)
    from ...solver.response_catalog import for_profile, lookup, status, supports
    from ..physics import profile_id
    from ...solver.su2_options import FIXED_CL_KEYS
    fixed_cl = any(set(table) & FIXED_CL_KEYS
                   for table in cfg.solver.su2_extra.values())
    physics = profile_id(cfg.case.physics)
    errors += _linear_constraint_errors(cfg, physics)
    for p in cfg.points:
        if not p.id or any(c in p.id for c in ' /\\'):
            errors.append(f'[[points]] id={p.id!r} must be non-empty and free '
                          f'of spaces and path separators (it becomes a '
                          f'filename suffix)')
        if not math.isfinite(p.weight) or p.weight < 0.0:
            errors.append(f'[[points]] {p.id}: weight must be finite and >= 0, '
                          f'got {p.weight}')
        if p.alpha not in ('design', 'fixed'):
            errors.append(f'[[points]] {p.id}: alpha must be "design" or "fixed", got {p.alpha!r}')
        if p.target_cl is not None:
            if p.alpha == 'fixed':
                errors.append(f'[[points]] {p.id}.target_cl cannot be used with alpha="fixed"')
            if not cfg.constraints.lift.enabled:
                errors.append(f'[[points]] {p.id}.target_cl requires [constraints.lift].enabled = true')
            if fixed_cl:
                errors.append(f'[[points]] {p.id}.target_cl cannot be combined with SU2 fixed-CL keys')
            if not math.isfinite(p.target_cl) or not 0.0 < p.target_cl < 2.0:
                errors.append(f'[[points]] {p.id}.target_cl={p.target_cl!r} must be finite and in (0, 2)')
        if p.start_cl is not None:
            if p.alpha == 'fixed':
                errors.append(f'[[points]] {p.id}.start_cl cannot be used with alpha="fixed"')
            if p.target_cl is not None:
                errors.append(f'[[points]] {p.id}: start_cl and target_cl are mutually '
                              f'exclusive (target_cl is a constraint, start_cl only the trim aim)')
            if not math.isfinite(p.start_cl):
                errors.append(f'[[points]] {p.id}.start_cl={p.start_cl!r} must be finite')
        for name in p.responses:
            try:
                item = lookup(name)
            except ValueError as exc:
                errors.append(f'[[points]] {p.id}.responses: {exc}')
                continue
            if not supports(item, cfg):
                errors.append(f'[[points]] {p.id}.responses={name!r} requires {item.requires}; '
                              f'case is ndim={cfg.case.ndim}, physics={cfg.case.physics.kind!r}')
            response_status = for_profile(name, physics)
            evidence = status(name, physics)[1]
            if cfg.execution.task == 'run_driver' and response_status in ('unsupported', 'fd_failed'):
                errors.append(f'[[points]] {p.id}.responses={name!r} has {response_status} derivative status for {physics}; evidence={evidence}')
        if len(set(p.responses)) != len(p.responses):
            errors.append(f'[[points]] {p.id}.responses contains duplicate catalog names')
        if not 0.0 < p.mach < 1.5:
            errors.append(f'[[points]] {p.id}: mach={p.mach} is outside the '
                          f'transonic range this pipeline is validated for')
        if not (p.aoa_lower <= p.aoa_init <= p.aoa_upper):
            errors.append(
                f'[[points]] {p.id}: aoa_init={p.aoa_init} is outside '
                f'[{p.aoa_lower}, {p.aoa_upper}] -- the optimizer would clip '
                f'the starting point and the trim would fight the bound')
        # A value the point inherits is checked once, by check_case; only
        # the point's own [[points]].freestream values are checked here.
        fs_values = p.freestream or {}
        if 'temperature' in fs_values:
            temp = float(fs_values['temperature'])
            if not math.isfinite(temp) or temp <= 0.0:
                errors.append(f'[[points]] {p.id}.freestream.temperature must be finite and > 0')
        reynolds = fs_values.get('reynolds_number', cfg.case.freestream.reynolds_number)
        if cfg.case.physics.turbulent and reynolds is None:
            errors.append(f'[[points]] {p.id}: turbulent physics needs '
                          f'[case.freestream] reynolds_number (no default)')
        elif cfg.case.physics.turbulent:
            if 'reynolds_number' in fs_values:
                try:
                    re = float(reynolds)
                except (TypeError, ValueError):
                    errors.append(f'[[points]] {p.id}.freestream.reynolds_number={reynolds!r} is not numeric')
                else:
                    if not math.isfinite(re) or re <= 0.0:
                        errors.append(f'[[points]] {p.id}.freestream.reynolds_number must be finite and > 0')
            if 'reynolds_length' in fs_values:
                re_length = float(fs_values['reynolds_length'])
                if not math.isfinite(re_length) or re_length <= 0.0:
                    errors.append(f'[[points]] {p.id}.freestream.reynolds_length must be finite and > 0')
        # alpha is DEGREES everywhere.  A value that looks like radians is
        # the single most expensive unit slip in this project's history.
        if abs(p.aoa_init) < 0.09 and p.aoa_init != 0.0:
            errors.append(
                f'[[points]] {p.id}: aoa_init={p.aoa_init} looks like radians; '
                f'alpha is in DEGREES everywhere')
    return errors
