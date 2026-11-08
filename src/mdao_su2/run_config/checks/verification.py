"""Checks of ``[verification]``: cancellation-ratio tables, band discriminating
power, stated bands and the moment-pivot identity."""



def check_cancellation_cr(cfg):
    """A measured C_r table must declare the parameterization it was measured on."""
    v = cfg.verification
    if not v.cancellation_cr or not v.cancellation_cr_param:
        return []
    param = cfg.geometry.parameterization
    if param == 'ffd':
        expected_tag = cfg.geometry.ffd
    elif param == 'cst':
        expected_tag = f'cst{cfg.geometry.n_cst}'
    else:
        expected_tag = param
    if v.cancellation_cr_param != expected_tag:
        return [f'[verification] cancellation_cr_param={v.cancellation_cr_param!r} '
                f'but the running parameterization resolves to {expected_tag!r}. '
                f'C_r is measured per parameterization -- one row is one design '
                f'variable. Measure it for {expected_tag!r}, '
                f'or set cancellation_cr_param = "" to fall back to the flat '
                f'shape_band (assigning an empty cancellation_cr does NOT '
                f'clear an inherited table: tables merge, scalars replace)']
    return []


def check_band_discriminating_power(cfg):
    """Bands must be able to fail; an always-passing band cannot verify a result."""
    v = cfg.verification
    errors = []
    if v.alpha_band_moment > 0 and v.alpha_band_moment_pivot > 0:
        errors.append(
            '[verification] set alpha_band_moment OR alpha_band_moment_pivot, '
            'not both: they judge the same C_M alpha row by different metrics '
            '(relative error vs chords of equivalent pivot error) and which '
            'one applied would not be readable from the config')
    if v.shape_band_cap < 0:
        errors.append('[verification] shape_band_cap must be >= 0 '
                      '(0 disables the cap)')
    for func, rows in v.shape_rows_unverifiable.items():
        for i in rows:
            if i < 0:
                errors.append(f'[verification] shape_rows_unverifiable[{func!r}] '
                              f'has a negative row index {i}')
                continue
            crs = v.cancellation_cr.get(func)
            if not crs or i >= len(crs):
                errors.append(
                    f'[verification] shape_rows_unverifiable[{func!r}] names '
                    f'row {i}, but cancellation_cr[{func!r}] has '
                    f'{len(crs) if crs else 0} entries. Only a TIERED band can '
                    f'inflate past the cap, so only a row with a measured C_r '
                    f'can be acknowledged')
                continue
            band = v.shape_row_band(func, i)
            if band is not None and v.shape_band_cap > 0 and band <= v.shape_band_cap:
                errors.append(
                    f'[verification] shape_rows_unverifiable[{func!r}] names '
                    f'row {i}, whose band is {band:.2e} -- inside the '
                    f'{v.shape_band_cap:.1e} cap, so the row IS verifiable. '
                    f'Remove it rather than record a real gate as unverified')
    return errors


def check_verification_bands(cfg):
    """A check_totals run judges every row or none.

    With no band stated the run reports the differences unjudged. A run that states some
    bands must state every band its rows use, or a row nothing tested would
    read as passed. Rows past the end of a C_r table take the flat band at
    run time; the verdict fails such a row when ``shape_band`` is unset.
    """
    if cfg.execution.task != 'check_totals':
        return []
    v = cfg.verification
    stated = v.stated_bands
    if not stated:
        return []
    needed = {'alpha_band'}
    param = cfg.geometry.parameterization
    tag = {'ffd': cfg.geometry.ffd,
           'cst': f'cst{cfg.geometry.n_cst}'}.get(param, param)
    functionals = ['C_D', *(['C_L'] if cfg.constraints.lift.enabled else []),
                   *(['C_M'] if cfg.constraints.moment.enabled else []),
                   *[c.path.rpartition('.')[2] for c in cfg.constraints.linear],
                   *v.shape_outputs]
    groups = [(name, tag if name == 'shape' else 'twist')
              for name in cfg.geometry_dvs]
    for _group, group_tag in groups:
        for functional in functionals:
            if v.shape_row_is_tiered(functional, param=group_tag):
                needed |= {'shape_band_floor', 'shape_band_slope'}
            else:
                needed.add('shape_band')
    missing = sorted(needed - set(stated))
    if missing:
        return [f'[verification] states {", ".join(stated)} but not '
                f'{", ".join(missing)}: a check_totals run that judges must '
                f'state a band for every row it produces, or none']
    return []


def check_moment_pivot(cfg):
    """``task="moment_pivot_check"`` needs a complete, usable reference state."""
    v = cfg.verification
    mp = v.moment_pivot
    task = cfg.execution.task
    errors = []
    if task == 'moment_pivot_check' and not mp.enabled:
        errors.append('[verification.moment_pivot] task="moment_pivot_check" '
                      'needs enabled = true')
    if not mp.enabled:
        return errors
    if task != 'moment_pivot_check':
        errors.append('[verification.moment_pivot] enabled = true but '
                      f'[execution] task={task!r}; the identity is only '
                      'evaluated by task="moment_pivot_check"')
    if mp.dcl_dalpha == 0.0:
        errors.append('[verification.moment_pivot] dcl_dalpha = 0 leaves '
                      'dC_N/dalpha with no signal; supply the reference run\'s '
                      'measured value')
    if mp.band_pivot <= 0:
        errors.append('[verification.moment_pivot] band_pivot must be > 0')
    # The pivot SU2 actually integrates about is resolved_pivot(): a
    # [constraints.moment] pivot overrides [case.reference] origin_moment in
    # point_cfg_writer. Checking the case field alone let the first run of this
    # gate proceed with an unmoved pivot and burn a full solve.
    pivot_x = cfg.constraints.moment.resolved_pivot(
        cfg.case.reference.origin_moment)[0]
    if pivot_x == mp.reference_pivot_x:
        source = ('[constraints.moment] pivot'
                  if cfg.constraints.moment.pivot is not None
                  else '[case.reference] origin_moment')
        errors.append(
            f'[verification.moment_pivot] this run integrates the moment about '
            f'x = {pivot_x:g} (from {source}, which is what '
            f'REF_ORIGIN_MOMENT_X becomes), the same pivot the reference '
            f'derivatives were measured at. The identity reduces to '
            f'0 == 0 and tests nothing -- move the pivot off the aerodynamic '
            f'center (x=0 amplifies the signal ~74x on RAE 2822). Setting only '
            f'origin_moment is NOT enough when constraints.moment.pivot is set')
    if not cfg.verification.alpha_outputs:
        errors.append('[verification.moment_pivot] needs C_M in '
                      '[verification] alpha_outputs')
    elif 'C_M' not in cfg.verification.alpha_outputs:
        errors.append('[verification.moment_pivot] [verification] '
                      f'alpha_outputs={list(cfg.verification.alpha_outputs)} '
                      'does not contain "C_M"')
    return errors
