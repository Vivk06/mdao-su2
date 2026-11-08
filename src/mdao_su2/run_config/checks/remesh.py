"""Checks of ``[remesh]`` and of the mesh2d recipe values it shares."""



REMESH_MODES = ('none', 'gmsh')


def check_remesh(cfg):
    errors = []
    r = cfg.remesh
    if r.mode not in REMESH_MODES:
        errors.append(f'[remesh] mode={r.mode!r} unknown; '
                      f'choose from {list(REMESH_MODES)}')
        return errors
    if not r.enabled:
        return errors
    # Gmsh meshing runs before the optimization.
    errors.append(
        '[remesh] mode="gmsh" is retired from the run path: '
        'generate the mesh once before the run with `run/mesh.py mesh2d` (which '
        'gates its own output), point [case] mesh at it, and run with '
        'su2_native deformation (remesh.mode = "none")')
    return errors


def wake_value_errors(r):
    """Checks of the optional wake wedge: off at ``wake_length = 0``, else all four keys."""
    if r.wake_length < 0.0:
        return [f'[remesh] wake_length={r.wake_length} must be >= 0 (0 turns the wake off)']
    if r.wake_length == 0.0:
        return []
    errors = []
    if len(r.wake_angles_deg) != 2:
        errors.append('[remesh] wake_length > 0 needs wake_angles_deg = [lo, hi], the '
                      'angles of the wedge rays from +x in degrees')
    else:
        lo, hi = r.wake_angles_deg
        if not (lo < hi and hi - lo < 180.0):
            errors.append(f'[remesh] wake_angles_deg={[lo, hi]} must satisfy lo < hi and '
                          f'hi - lo < 180 (the wedge must be convex)')
    if r.wake_meshsize <= 0.0:
        errors.append('[remesh] wake_length > 0 needs wake_meshsize > 0 (size at the TE apex)')
    if r.wake_stream_slope < 0.0:
        errors.append('[remesh] wake_length > 0 needs wake_stream_slope >= 0 '
                      '(size increase per unit length downstream)')
    return errors


def remesh_value_errors(r, table='[remesh]'):
    """Size, BL and quality-band checks for a RemeshConfig (also the mesh2d recipe)."""
    errors = []
    if r.first_cell_height <= 0.0:
        errors.append(f'[remesh] first_cell_height must be > 0, '
                      f'got {r.first_cell_height}')
    if r.growth_ratio <= 1.0:
        errors.append(f'[remesh] growth_ratio={r.growth_ratio} must be > 1 '
                      f'(a ratio <= 1 never grows out of the boundary layer)')
    if r.n_bl_layers < 1:
        errors.append(f'[remesh] n_bl_layers must be >= 1, got {r.n_bl_layers}')
    if r.farfield_radius <= 1.0:
        errors.append(f'[remesh] farfield_radius={r.farfield_radius} chords is '
                      f'too close to the airfoil for a transonic solution')
    if r.near_meshsize <= 0.0:
        errors.append(f'[remesh] near_meshsize must be > 0, '
                      f'got {r.near_meshsize}')
    if not 0.0 < r.far_growth_slope < 1.0:
        errors.append(
            f'[remesh] far_growth_slope={r.far_growth_slope} must be in (0, 1). '
            f'It is an element size divided by its distance from the wall; the '
            f'validated baseline mesh runs at 0.06-0.17, and >= 1 means elements '
            f'larger than their own distance from the airfoil')
    if r.farfield_meshsize <= r.near_meshsize:
        errors.append(
            f'[remesh] farfield_meshsize={r.farfield_meshsize} must exceed '
            f'near_meshsize={r.near_meshsize}; it is the ceiling on the size '
            f'field and the other is its floor')
    if r.wall_meshsize < 0.0:
        errors.append('[remesh] wall_meshsize must be >= 0 (0 means auto)')
    errors += wake_value_errors(r)
    # 'vtu' is the natural typo -- it is what SU2 writes for FLOW fields, while
    # the mesh writer is gmsh's and takes 'vtk'.  Catching it here costs a
    from ...mesh.mesh2d import VIEWABLE_FORMATS
    for fmt in r.viewable_mesh:
        if not isinstance(fmt, str) or fmt.lower().lstrip('.') not in VIEWABLE_FORMATS:
            errors.append(
                f'[remesh] viewable_mesh entry {fmt!r} is not a supported '
                f'format; expected any of {list(VIEWABLE_FORMATS)} '
                f"(e.g. viewable_mesh = [\"msh\"])")
    rs = r.solver
    for side in ('primal', 'adjoint'):
        val = getattr(rs, f'{side}_conv_residual_minval')
        if val > 0:
            errors.append(
                f'[remesh.solver] {side}_conv_residual_minval={val} must be '
                f'negative (it is a log10 residual)')
    # A primal floor looser than -7 was how an unconverged field reached the
    # objective: rms crosses the floor on a limit-cycle downswing and the
    if rs.primal_conv_residual_minval > -7.0:
        errors.append(
            f'[remesh.solver] primal_conv_residual_minval='
            f'{rs.primal_conv_residual_minval} is too loose to trust the '
            f'coefficients; use -8.5 or tighter, or state why in the config')
    q = r.quality
    if not 0.0 < q.min_first_cell_height_ratio <= 1.0 <= q.max_first_cell_height_ratio:
        errors.append(
            '[remesh.quality] first-cell-height ratio band must bracket 1.0 '
            f'(got [{q.min_first_cell_height_ratio}, '
            f'{q.max_first_cell_height_ratio}])')
    if not 0.0 < q.min_bl_thickness_ratio <= 1.0 <= q.max_bl_thickness_ratio:
        errors.append(
            '[remesh.quality] bl-thickness ratio band must bracket 1.0 '
            f'(got [{q.min_bl_thickness_ratio}, {q.max_bl_thickness_ratio}])')
    # 1.0 is ALLOWED and means "do not gate on skewness": equiangle skew is
    # bounded above by 1.0, so the check can never fire.  Deliberately not
    if (r.first_cell_height > 0.0 and r.growth_ratio > 1.0
            and r.n_bl_layers >= 1 and r.farfield_radius > 1.0):
        _, _, n_layers, thickness = r.resolved_bl()
    else:
        n_layers, thickness = 0, 0.0
    if thickness > 0.1 * r.farfield_radius:
        errors.append(
            f'[remesh] the boundary layer works out to {thickness:.4g} chords '
            f'({n_layers} layers), which is more than 10% of '
            f'farfield_radius={r.farfield_radius:g}. Either bl_thickness is too '
            f'large, or first_cell_height x growth_ratio^n_bl_layers is: a '
            f'geometric BL grows fast, and gmsh fails with "Edge not recovered" '
            f'rather than anything readable')
    if q.max_aspect_ratio_p99 <= 1.0:
        errors.append(f'[remesh.quality] max_aspect_ratio_p99='
                      f'{q.max_aspect_ratio_p99} must be > 1 (1 is a perfectly '
                      f'isotropic element; a boundary layer is never that)')
    if r.fan_angle_deg < 0.0 or r.fan_angle_deg >= 180.0:
        errors.append(f'[remesh] fan_angle_deg={r.fan_angle_deg} must be in '
                      f'[0, 180); 0 disables fanning')
    if r.fan_elements < 1:
        errors.append(f'[remesh] fan_elements={r.fan_elements} must be >= 1')
    if not 0.0 < q.max_skewness <= 1.0:
        errors.append(f'[remesh.quality] max_skewness={q.max_skewness} must be '
                      f'in (0, 1]; 1.0 disables the gate')
    if '{id}' not in r.mesh_output:
        errors.append('[remesh] mesh_output must contain "{id}" so the points '
                      'do not overwrite each other\'s meshes')
    if table != '[remesh]':
        errors = [e.replace('[remesh.quality]', f'{table[:-1]}.quality]')
                  .replace('[remesh.solver]', f'{table[:-1]}.solver]')
                  .replace('[remesh]', table) for e in errors]
    return errors
