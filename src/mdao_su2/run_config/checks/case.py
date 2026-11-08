"""Checks of ``[case]``: boundary roles, reference values, free stream,
numerics and SU2's boundary-physics guards."""

import math



def check_case(cfg):
    """Markers, reference values and the free-stream: the block that decides
    what the reported coefficients MEAN."""
    errors = []
    case = cfg.case
    wall = set(case.design_markers)
    sym = set(case.markers_of('sym'))
    if not wall:
        errors.append('[[case.boundary]] no marker has role "design"; the wall '
                      'seam needs the surface the shape DVs move')
    # coefficients integrated over one marker while pyGeo deforms another
    # is a silent wrong answer -- the run converges and reports drag for a
    # surface nobody moved.
    for role in ('monitor', 'plot'):
        named = case.markers_with(role)
        if set(named) != wall:
            errors.append(
                f'[[case.boundary]] role "{role}" is on {list(named)}, not the '
                f'design wall ({sorted(wall)}). C_L/C_D are integrated over '
                f'MARKER_MONITORING and pyGeo pushes displacements onto '
                f'MARKER_DEFORM_MESH; a disagreement reports coefficients for a '
                f'surface the shape DVs do not move.')
    # ``deform`` may exceed the wall only by symmetry planes: naming a plane is
    # how a wall node lying on it keeps its displacement.
    deform = set(case.markers_with('deform'))
    extra = sorted(deform - wall - sym)
    if not deform >= wall or extra:
        errors.append(
            f'[[case.boundary]] role "deform" is on {sorted(deform)}; it must '
            f'cover the design wall ({sorted(wall)}) and may add only symmetry '
            f'planes; {extra or "the wall is incomplete"}. pyGeo pushes '
            f'displacements onto MARKER_DEFORM_MESH, so a surface missing from it '
            f'does not move.')
    # a symmetry plane that is neither deform nor deform_sym is CLAMPED
    # by SU2's mesh solver, and the clamp zeroes wall nodes shared with it.
    deform_sym = set(case.markers_with('deform_sym'))
    missing = sorted(sym - deform - deform_sym)
    if missing:
        errors.append(
            f'[[case.boundary]] symmetry plane(s) {missing} have neither role '
            f'"deform" nor "deform_sym". SU2 clamps any marker that is neither '
            f'DEFORM_MESH nor DEFORM_MESH_SYM_PLANE, and the clamp zeroes the '
            f'boundary displacement of every node it touches -- including wall '
            f'nodes shared with the plane. Give it "deform" (plane PINNED, reverse '
            f'pass intact) or "deform_sym" (plane SLIDES, reverse pass reads 0 at '
            f'every wall node on it).')
    both = sorted(deform & deform_sym)
    if both:
        errors.append(
            f'[[case.boundary]] {both} has both "deform" and "deform_sym". SU2 '
            f'tests DEFORM_MESH first and never reaches the sym-plane branch -- '
            f'name exactly one.')
    errors += check_boundary_physics(cfg)
    # the pin holds wall nodes on the symmetry plane inside the geometry
    # component; it needs a plane, and it is built for one geometry path.
    if cfg.geometry.pin_symmetry:
        if not sym:
            errors.append('[geometry] pin_symmetry = true needs a bc = "sym" '
                          'plane to hold the wall on; this case has none')
        if cfg.geometry.parameterization not in ('ffd', 'hicks_henne'):
            errors.append(f'[geometry] pin_symmetry supports ffd and hicks_henne, '
                          f'not {cfg.geometry.parameterization!r}')
        if (cfg.case.is_multi_marker_wall or cfg.case.ndim != 2
                or cfg.execution.parallel_points or cfg.remesh.mode != 'none'):
            errors.append('[geometry] pin_symmetry is built for the 2D single-marker, '
                          'sequential, fixed-mesh path; it refuses a multi-marker '
                          'or 3D wall, parallel_points and remesh.mode="gmsh"')

    ref = cfg.case.reference
    if ref.length <= 0.0:
        errors.append('[case.reference] length must be > 0 '
                      '(it divides every reported moment coefficient)')
    # REF_AREA= 0 is SU2's "compute the planform area yourself" sentinel,
    # so 0 is legal and NEGATIVE is not.
    if ref.area < 0.0:
        errors.append(f'[case.reference] area={ref.area} must be > 0, or '
                      f'exactly 0 to mean "let SU2 compute the planform area '
                      f'from MARKER_MONITORING" (REF_AREA= 0)')

    phys = cfg.case.physics
    fs = cfg.case.freestream
    if not math.isfinite(fs.temperature) or fs.temperature <= 0.0:
        errors.append(f'[case.freestream] temperature={fs.temperature} must be '
                      f'> 0 (it is absolute, in Kelvin)')
    if phys.turbulent:
        if not math.isfinite(fs.reynolds_length) or fs.reynolds_length <= 0.0:
            errors.append('[case.freestream] reynolds_length must be > 0')
        try:
            re = float(fs.reynolds_number)
        except TypeError:
            pass            # unset here: every point must supply it (checked per point)
        except ValueError:
            errors.append(f'[case.freestream] reynolds_number='
                          f'{fs.reynolds_number!r} is not a number')
        else:
            if not math.isfinite(re) or re <= 0.0:
                errors.append('[case.freestream] reynolds_number must be > 0')
    else:
        # An inviscid case emits no REYNOLDS_NUMBER, so the free stream is
        # fixed by Mach + temperature + pressure instead.  Leaving the pressure
        if fs.pressure < 0.0:
            errors.append(f'[case.freestream] pressure={fs.pressure} must be '
                          f'>= 0 (0 = omit the key)')

    mg = cfg.solver.multigrid
    for name in ('pre_smooth', 'post_smooth', 'correction_smooth'):
        if getattr(mg, name) < 0:
            errors.append(f'[solver.multigrid] {name} must be >= 0')
    if cfg.solver.linear.iter < 1:
        errors.append('[solver.linear] iter must be >= 1')
    if cfg.solver.numerics.cfl_number <= 0.0:
        errors.append(f'[solver.numerics] cfl_number='
                      f'{cfg.solver.numerics.cfl_number} must be > 0')
    if cfg.deform.emit:
        d = cfg.deform
        if d.linear_solver_iter < 1:
            errors.append('[deform] linear_solver_iter must be >= 1')
        if d.nonlinear_iter < 1:
            errors.append('[deform] nonlinear_iter must be >= 1')
        if d.linear_solver_error <= 0.0:
            errors.append('[deform] linear_solver_error must be > 0')
    return errors


def check_boundary_physics(cfg):
    """SU2's own load-time guards (CConfig.cpp:3642-3663), mirrored so a bad TOML
    fails here instead of inside an MPI job; plus the SA-only RANS contract."""
    from ...solver.su2_options import BC_FAMILIES
    errors = []
    phys = cfg.case.physics
    extra = cfg.solver.su2_extra
    bounds = cfg.case.boundaries
    if not any(BC_FAMILIES[b.bc].outer for b in bounds):
        outer = sorted(n for n, f in BC_FAMILIES.items() if f.outer)
        errors.append(f'[[case.boundary]] has no outer boundary; one entry needs bc '
                      f'in {outer}, or SU2 stops at startup on an unassigned marker')
    if phys.inviscid:
        viscous = [b.markers for b in bounds if BC_FAMILIES[b.bc].viscous]
        if viscous:
            errors.append(f'[[case.boundary]] {viscous} are viscous walls; SU2: '
                          f'"Euler solvers are only compatible with slip walls '
                          f'(MARKER_EULER)"')
    else:
        slip = [b.markers for b in bounds if 'design' in b.roles
                and not BC_FAMILIES[b.bc].viscous]
        if slip:
            errors.append(f'[[case.boundary]] design wall(s) {slip} are slip walls '
                          f'under RANS; a RANS wall needs a viscous bc (heatflux, '
                          f'isothermal, heattransfer)')
    trans = phys.transition.model
    if trans != 'NONE' and phys.inviscid:
        errors.append(f'KIND_TRANS_MODEL={trans} under EULER; SU2: "KIND_TURB_MODEL '
                      f'cannot be NONE to use a transition model"')
    if trans == 'LM2015' and cfg.case.ndim != 3:
        errors.append('KIND_TRANS_MODEL=LM2015 requires ndim=3 (SU2 CConfig.cpp:3642-3690)')
    if phys.turbulence.model == 'SA' and any(
            item not in ('NONE', 'NEGATIVE', 'EDWARDS', 'WITHFT2', 'QCR2000',
                         'COMPRESSIBILITY', 'ROTATION', 'BCM', 'EXPERIMENTAL')
            for item in phys.turbulence.options):
        errors.append('SA_OPTIONS contains a value not accepted by SU2 v8.5')
    for name, value in (('FREESTREAM_TURBULENCEINTENSITY', phys.turbulence_intensity),
                        ('FREESTREAM_TURB2LAMVISCRATIO', phys.turb2lam_viscosity_ratio)):
        if value is not None and value <= 0.0:
            errors.append(f'{name} must be positive, got {value}')
    return errors
