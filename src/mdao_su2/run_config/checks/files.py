"""Checks that read input files: the mesh and its markers, the CST surface, the
symmetry pin, FFD boxes, design start and seeds, and primal/adjoint cfg parity."""

import os

from ...solver.cfg_file import CfgParityError, assert_cfg_parity
from ..errors import ConfigError
from .geometry import ffd_lattice_shape


def check_markers_against_mesh(cfg):
    """Every configured marker exists in the mesh, and ``ndim`` agrees."""
    from ...mesh.su2_mesh_file import read_mesh_header
    errors = []
    try:
        header = read_mesh_header(cfg.mesh_path)
    except (OSError, ValueError) as exc:
        return [f'[case] mesh could not be read: {cfg.mesh_path}: {exc}']

    present = header['markers']
    if present:
        for name in cfg.case.all_markers():
            if name not in present:
                near = [p for p in present
                        if p.lower().replace('_', '') == name.lower().replace('_', '')]
                hint = f' Did you mean {near[0]!r}?' if near else ''
                errors.append(
                    f'[[case.boundary]] {name!r} is not a marker in '
                    f'{os.path.basename(cfg.mesh_path)}. The mesh has '
                    f'{sorted(present)}.{hint}')
        for name in cfg.case.design_markers:
            if present.get(name) == 0:
                errors.append(f'[[case.boundary]] design marker {name!r} exists in '
                              f'the mesh but has no elements')

    ndim = header['ndim']
    if ndim is not None and ndim != cfg.case.ndim:
        errors.append(
            f'[case] ndim={cfg.case.ndim} but {os.path.basename(cfg.mesh_path)} '
            f'declares NDIME= {ndim}. Dimensionality decides the x_aero stride '
            f'and every geometry path; a disagreement is not recoverable later.')
    return errors


#: CST surface-to-mesh wall tolerance, in reference lengths.
CST_SURFACE_TOL = 1e-6


def check_cst_surface(cfg):
    """The CST ``dat_file`` is the mesh wall: same node count, every node on it.

    Catches a surface from one mesh level paired with a mesh from another.
    """
    import numpy as np
    from ...geometry.airfoil import read_dat
    from ...mesh.su2_mesh_file import read_wall_surface
    try:
        rows = read_dat(cfg.dat_path)
    except ValueError as exc:
        return [f'[geometry] dat_file could not be read: {exc}']
    if np.array_equal(rows[0], rows[-1]):
        rows = rows[:-1]                    # closed Selig loop repeats its TE row
    try:
        points, faces = read_wall_surface(cfg.mesh_path, cfg.case.design_markers)
    except (KeyError, ValueError) as exc:
        return [f'[geometry] dat_file could not be compared with the mesh wall: {exc}']
    wall = points[sorted({n for f in faces for n in f}), :2]
    name = (os.path.basename(cfg.dat_path), os.path.basename(cfg.mesh_path))
    if len(wall) != len(rows):
        return [f'[geometry] dat_file {name[0]} has {len(rows)} surface points but '
                f'{name[1]} has {len(wall)} wall nodes: pair the surface with the mesh '
                f'level it was meshed from']
    tol = CST_SURFACE_TOL * cfg.case.reference.length
    gap = max(float(np.min(np.hypot(*(rows - p).T))) for p in wall)
    if gap > tol:
        return [f'[geometry] dat_file {name[0]} is up to {gap:.3g} from the {name[1]} '
                f'wall (tolerance {tol:.3g}): pair the surface with the mesh it '
                f'was meshed from']
    return []


def check_symmetry_pin_mesh(cfg):
    """the pinned plane is axis-aligned and holds at least one wall node."""
    from ...geometry.symmetry_pin import pinned_wall_nodes
    wall, sym = cfg.case.design_markers, cfg.case.markers_of('sym')
    try:
        nodes, _axis = pinned_wall_nodes(cfg.mesh_path, wall, sym)
    except (KeyError, ValueError) as exc:
        return [f'[geometry] pin_symmetry: {exc}']
    if not nodes:
        return [f'[geometry] pin_symmetry = true, but no {list(wall)} node lies '
                f'on {list(sym)} in {os.path.basename(cfg.mesh_path)}; a '
                f'pin that holds nothing is a mistake']
    return []


def check_input_files(cfg):
    """Every input file, checked before anything expensive starts."""
    errors = []
    paths = [('case.mesh', cfg.mesh_path)]
    for label, path in paths:
        if not os.path.isfile(path):
            errors.append(f'[case] {label} not found: {path}')
    if os.path.isfile(cfg.mesh_path):
        errors += check_markers_against_mesh(cfg)
        if cfg.geometry.pin_symmetry and cfg.case.markers_of('sym'):
            errors += check_symmetry_pin_mesh(cfg)
    if cfg.geometry.parameterization == 'cst':
        if cfg.geometry.dat_file:
            dat = cfg.dat_path
            if not os.path.isfile(dat):
                errors.append(f'[geometry] dat_file not found: {dat}')
            elif os.path.isfile(cfg.mesh_path):
                errors += check_cst_surface(cfg)
    elif cfg.geometry.parameterization == 'ffd':
        try:
            ffd = cfg.ffd_path
        except ConfigError as exc:
            errors.append(str(exc))
        else:
            if not os.path.isfile(ffd):
                errors.append(f'[geometry] ffd file not found: {ffd}')
            elif cfg.dv.twist.enabled:
                nj = ffd_lattice_shape(ffd)[1]
                if nj != cfg.dv.twist.n:
                    errors.append(f'[dv.twist] n = {cfg.dv.twist.n} but {ffd} has {nj} '
                                  f'spanwise (j) sections; twist sets one per section')

    # A design start or seed that does not resolve would start the run from
    # the wrong state without a message, so every problem fails here.
    from ...run_output.design_start import read_start
    from ...run_output.rundir import seed_plan
    for check in (read_start, seed_plan):
        try:
            check(cfg)
        except ConfigError as exc:
            errors.append(str(exc))
    return errors


def check_cfg_parity(cfg):
    """Primal/adjoint parity: the two cfgs must describe the same problem:
    the physics keys agree, and every key that differs is a declared difference
    (registry ``parity``/``side`` or a side-specific passthrough)."""
    from ...solver.cfg_file import parse_cfg_content
    from ...solver.point_cfg_writer import generate_cfg_content, intentional_differences
    errors = []
    allowed = intentional_differences(cfg)
    for point in cfg.points:
        primal = generate_cfg_content(cfg, point, is_adjoint=False)
        adjoint = generate_cfg_content(cfg, point, is_adjoint=True)
        try:
            assert_cfg_parity(primal, adjoint, as_text=True)
        except CfgParityError as exc:
            errors.append(f'point {point.id}: {exc}')
        p, a = parse_cfg_content(primal), parse_cfg_content(adjoint)
        undeclared = sorted(k for k in set(p) | set(a)
                            if p.get(k) != a.get(k) and k not in allowed)
        if undeclared:
            errors.append(f'point {point.id}: primal/adjoint differ on undeclared '
                          f'key(s) {undeclared}; the adjoint would linearize a '
                          f'flow the primal did not solve')
    return errors
