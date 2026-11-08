#!/usr/bin/env python
"""Prepare geometry and mesh files before a configured run.

Outputs go to the explicit ``-o`` path or the recipe's ``[mesh.outputs]``
paths, separately from a run's output directory.

    python run/mesh.py extract --mesh M.su2 --markers UPPER LOWER -o s.dat
    python run/mesh.py airfoil --config recipe.toml [--level L] -o s.dat
    python run/mesh.py ffd     --config recipe.toml [--level L] -o box.xyz
    python run/mesh.py mesh2d  --config recipe.toml [--level L] --surface s.dat -o m.su2 [--json]
    python run/mesh.py wing3d  --config recipe.toml --surface s.dat --level medium -o w.su2 [--json]
    python run/mesh.py section2d --config recipe.toml --surface s.dat --level x2 -o m.su2 [--json]
    python run/mesh.py view    M.su2 -o m.vtu [--boundary b.vtu [--markers TAG ...]] [--json]
    python run/mesh.py ffd-from-su2 M.su2 -o box.xyz [--verify --tol T]
    python run/mesh.py check-ffd M.su2 --ffd box.xyz --tol T [--json]
    python run/mesh.py ffd-freeze M.su2 --wall WING --continuity C [--json]
    python run/mesh.py all     --config recipe.toml --level L|all [--output-dir D]

Values come from CLI flags over ``[mesh.<step>.levels.<L>]`` over the
``[mesh.<step>]`` table of ``--config``; there are no defaults. A step runs
only when its table says ``enabled = true`` or ``--enable`` is given, and
``--config`` may be left out when the flags state every value (``--quality
KEY=VALUE`` and ``--level-scale LEVEL=VALUE`` give the two table-valued
keys). ``-o/--output`` is required on every step that writes a file; ``all``
takes its file names from ``[mesh.outputs]``. ``mesh2d``, ``wing3d`` and ``section2d`` gate their own output. With ``--json`` stdout carries
one JSON object and progress text goes to stderr.

Exit status: 0 done, 1 a gate failed or a value or file could not be used,
2 a usage error.
"""

import argparse
import contextlib
import json
import os
import sys
from types import SimpleNamespace

os.environ.setdefault('OPENMDAO_REPORTS', '0')      # importing mdao_su2 imports OpenMDAO


class UsageError(Exception):
    """A command-line combination that cannot be used (exit 2)."""


class _KeyValue(argparse.Action):
    """``KEY=VALUE`` (repeatable) collected into a dict of floats."""

    def __call__(self, parser, namespace, values, option_string=None):
        key, sep, number = values.partition('=')
        try:
            if not sep or not key:
                raise ValueError
            value = float(number)
        except ValueError:
            parser.error(f'{option_string} takes KEY=NUMBER, got {values!r}')
        merged = dict(getattr(namespace, self.dest) or {})
        merged[key] = value
        setattr(namespace, self.dest, merged)


def _common(p, recipe=True, level=True):
    if recipe:
        p.add_argument('--config', metavar='RECIPE.toml',
                       help='recipe TOML with a [mesh.<step>] table')
        p.add_argument('--enable', action='store_true',
                       help='run the step even if the recipe does not enable it')
    if recipe and level:
        p.add_argument('--level', help='apply [mesh.<step>.levels.<LEVEL>] over the step')
    p.add_argument('-o', '--output', required=True, help='file to write (no default)')


def _quality(p, what):
    p.add_argument('--quality', action=_KeyValue, metavar='KEY=VALUE',
                   help=f'{what} gate limit; repeatable, per key over the recipe')


def _level_scale(p, what):
    p.add_argument('--level-scale', dest='level_scale', action=_KeyValue,
                   metavar='LEVEL=VALUE', help=f'{what} level scales; repeatable, per key '
                                               f'over the recipe')


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='step', required=True)

    p = sub.add_parser('extract', help='wall markers of a 2D SU2 mesh -> Selig .dat')
    p.add_argument('--mesh', help='SU2 mesh (a recipe path is relative to the recipe, '
                                  'a flag path to the working directory)')
    p.add_argument('--markers', nargs='+',
                   help='one or more wall markers; several are joined into one loop')
    _common(p)

    p = sub.add_parser('airfoil', help='import, repanel, TE edit -> Selig .dat + report')
    p.add_argument('--source', help='naca4:<code>:<open|closed>:<n> | mesh:<path>:<m>[,<m>] | <file.dat>')
    p.add_argument('--n-points', type=int)
    p.add_argument('--spacing', help='cosine | conical | polynomial | bigeometric')
    p.add_argument('--te', help='keep | sharp:<x_cut> | blunt:<x_cut> | round:<x_cut>')
    p.add_argument('--le-spacing', type=float, help='bigeometric LE spacing (chord units)')
    p.add_argument('--te-spacing', type=float, help='bigeometric TE spacing (chord units)')
    p.add_argument('--spacing-ratio', type=float, help='bigeometric growth ratio, > 1')
    _common(p)

    p = sub.add_parser('ffd-from-su2', help='embedded SU2 FFD box -> Plot3D .xyz')
    p.add_argument('mesh')
    p.add_argument('-o', '--output', required=True, help='file to write (no default)')
    p.add_argument('--box')
    p.add_argument('--verify', action='store_true',
                   help='run the check-ffd embedding check on the written box (needs --tol)')
    p.add_argument('--tol', type=float, help='parameter tolerance of --verify')

    p = sub.add_parser('ffd-freeze', help='report FFD planes SU2 freezes at wall intersections')
    p.add_argument('mesh')
    p.add_argument('--wall', action='append', required=True)
    p.add_argument('--continuity', required=True,
                   help='FFD_CONTINUITY of the run, e.g. 2ND_DERIVATIVE (no default)')
    p.add_argument('--symmetry-plane', action='store_true')
    p.add_argument('--box')
    p.add_argument('--json', action='store_true', help='print the result as JSON')

    p = sub.add_parser('check-ffd', help='verify an embedded FFD box and optional wall enclosure')
    p.add_argument('mesh')
    p.add_argument('--ffd')
    p.add_argument('--box')
    p.add_argument('--tol', type=float, required=True,
                   help='parameter tolerance of the embedding check (no default)')
    p.add_argument('--kmax', type=int)
    p.add_argument('--enclosure', nargs='+')
    p.add_argument('--json', action='store_true', help='print the result as JSON')

    p = sub.add_parser('ffd', help='airfoil or wing FFD box -> Plot3D .xyz')
    p.add_argument('--kind', choices=('airfoil', 'wing'))
    p.add_argument('--nx', type=int)
    p.add_argument('--spacing', help='uniform | cosine (airfoil only)')
    p.add_argument('--zspan', type=float)
    for axis in 'xyz':
        p.add_argument(f'--{axis}-bounds', nargs=2, type=float, metavar=('LO', 'HI'))
    p.add_argument('--x-margin', type=float)
    p.add_argument('--y-margin', type=float)
    p.add_argument('--le-te', nargs=2, type=float, metavar=('X_LE', 'X_TE'))
    p.add_argument('--n-span', type=int)
    p.add_argument('--nz', type=int)
    p.add_argument('--surface', help='Selig .dat the margins are measured from')
    _common(p)

    p = sub.add_parser('mesh2d', help='hybrid BL mesh around a Selig section -> .su2, gated')
    p.add_argument('--surface', required=True, help='Selig .dat of the wall (run/mesh.py airfoil)')
    for key, typ in (('first_cell_height', float), ('growth_ratio', float),
                     ('n_bl_layers', int), ('bl_thickness', float),
                     ('fan_angle_deg', float), ('fan_elements', int),
                     ('farfield_radius', float), ('farfield_meshsize', float),
                     ('near_meshsize', float), ('far_growth_slope', float),
                     ('wall_meshsize', float), ('wall_marker', str),
                     ('farfield_marker', str), ('wake_length', float),
                     ('wake_meshsize', float), ('wake_stream_slope', float),
                     ('attempts', int)):
        p.add_argument('--' + key.replace('_', '-'), type=typ)
    p.add_argument('--wake-angles-deg', nargs=2, type=float, metavar=('LO', 'HI'))
    p.add_argument('--farfield-center', nargs=2, type=float, metavar=('X', 'Y'))
    _quality(p, 'mesh2d')
    from mdao_su2.mesh.mesh2d import VIEWABLE_FORMATS
    p.add_argument('--view', action='append', default=[], choices=list(VIEWABLE_FORMATS),
                   help='also write a viewable copy beside the .su2; repeatable')
    p.add_argument('--verbose', action='store_true')
    p.add_argument('--json', action='store_true', help='print the gate report as JSON')
    _common(p)

    p = sub.add_parser('wing3d', help='tetrahedral half-wing with revolved tip -> .su2, gated')
    p.add_argument('--surface', required=True, help='sharp-TE Selig .dat (run/mesh.py airfoil)')
    p.add_argument('--level', help='a key of [mesh.wing3d] level_scale (or of --level-scale)')
    for key in ('span', 'farfield_radius', 'wall_size', 'le_size', 'te_size', 'tip_size',
                'far_size', 'size_growth', 'edge_band', 'te_band', 'te_normal_scale'):
        p.add_argument('--' + key.replace('_', '-'), type=float)
    p.add_argument('--edge-distance', choices=('sampled', 'exact'),
                   help='LE/TE distance: gmsh-sampled curve or exact line distance')
    p.add_argument('--tip', help='tip kind, revolved')
    p.add_argument('--farfield-center', nargs=3, type=float, metavar=('X', 'Y', 'Z'))
    for key in MARKERS_3D:
        p.add_argument('--' + key.replace('_', '-'))
    _quality(p, 'wing3d')
    _level_scale(p, 'wing3d')
    p.add_argument('--threads', type=int, default=0,
                   help='gmsh threads (execution only; 0 = gmsh decides)')
    p.add_argument('--verbose', action='store_true')
    p.add_argument('--json', action='store_true', help='print the gate report as JSON')
    _common(p, level=False)

    p = sub.add_parser('section2d', help='all-triangle 2D section mesh sized as a '
                                         'wing3d span station -> .su2, gated')
    p.add_argument('--surface', required=True, help='sharp-TE Selig .dat (run/mesh.py airfoil)')
    p.add_argument('--level', help='a key of [mesh.section2d] level_scale (or of --level-scale)')
    for key in SECTION2D_SIZES:
        p.add_argument('--' + key.replace('_', '-'), type=float)
    p.add_argument('--farfield-center', nargs=2, type=float, metavar=('X', 'Y'))
    p.add_argument('--wall-marker')
    p.add_argument('--farfield-marker')
    _quality(p, 'section2d')
    _level_scale(p, 'section2d')
    p.add_argument('--verbose', action='store_true')
    p.add_argument('--json', action='store_true', help='print the gate report as JSON')
    _common(p, level=False)

    p = sub.add_parser('view', help='SU2 mesh -> .vtu volume, optionally a marker .vtu')
    p.add_argument('mesh')
    p.add_argument('-o', '--output', required=True, help='volume .vtu to write (no default)')
    p.add_argument('--boundary', metavar='BOUNDARY.vtu',
                   help='also write the marker elements, with a `marker` cell array and '
                        'the legend as FieldData')
    p.add_argument('--markers', nargs='+', metavar='TAG',
                   help='only these markers in --boundary (default: all)')
    p.add_argument('--json', action='store_true', help='print the summary as JSON')

    p = sub.add_parser('all', help='run enabled recipe steps in dependency order')
    p.add_argument('--config', required=True, metavar='RECIPE.toml')
    p.add_argument('--output-dir', help='resolve [mesh.outputs] against this directory '
                                        'instead of the recipe\'s')
    p.add_argument('--surface', help='existing Selig .dat if airfoil is disabled')
    p.add_argument('--level', help='a recipe level, "all" for every level, or a wing3d/'
                                   'section2d level_scale key')
    p.add_argument('--view', nargs='*', default=[], help='mesh2d viewable copies (msh, vtk)')
    return ap


SECTION2D_SIZES = ('farfield_radius', 'wall_size', 'le_size', 'te_size', 'far_size',
                   'size_growth', 'edge_band')
#: wing3d marker names, in the order of ``Wing3DStep.markers`` without the far field.
MARKERS_3D = ('wing_marker', 'tip_marker', 'symmetry_marker', 'farfield_marker')

#: Real stdout while ``--json`` sends progress text to stderr.
_json_out = None


def _cli_values(args, keys):
    return {k: getattr(args, k, None) for k in keys}


def _emit(args, payload):
    """The ``--json`` object on stdout."""
    if getattr(args, 'json', False):
        print(json.dumps(payload, indent=2, default=str), file=_json_out or sys.stdout)


def _prepare_output(*paths):
    for path in paths:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)


def do_extract(args):
    from mdao_su2.geometry.airfoil import selig_from_mesh, write_selig
    from mdao_su2.mesh.recipe import resolve
    if args.config:
        r = resolve('extract', args.config, _cli_values(args, ('mesh', 'markers')),
                    args.enable, args.level)
        mesh = r.values.mesh
        if r.sources['mesh'] != 'cli':             # a recipe path is relative to the recipe
            mesh = os.path.join(os.path.dirname(os.path.abspath(args.config)), mesh)
        markers = list(r.values.markers)
    elif args.mesh and args.markers:
        r, mesh, markers = None, args.mesh, args.markers
    else:
        raise UsageError('extract needs --mesh and --markers, or --config with [mesh.extract]')
    _prepare_output(args.output)
    coords = selig_from_mesh(mesh, markers)
    n = write_selig(args.output, coords)
    print(f'extract: {mesh} {markers} -> {args.output} ({n} rows)')
    return r


def do_airfoil(args):
    import numpy as np
    from mdao_su2.geometry.airfoil import (geometry_report, read_surface,
                                                repanel, write_selig)
    from mdao_su2.mesh.recipe import resolve
    r = resolve('airfoil', args.config, _cli_values(
        args, ('source', 'n_points', 'spacing', 'te', 'le_spacing', 'te_spacing',
               'spacing_ratio')), args.enable, args.level)
    v = r.values
    _prepare_output(args.output)
    # a recipe's source is relative to the recipe, a flag's to the working directory
    base = (os.path.dirname(os.path.abspath(args.config))
            if args.config and r.sources['source'] != 'cli' else os.getcwd())
    coords = repanel(read_surface(v.source, base_dir=base), v.n_points, v.spacing, v.te,
                     le_spacing=v.le_spacing, te_spacing=v.te_spacing,
                     spacing_ratio=v.spacing_ratio)
    rows = write_selig(args.output, coords)
    report = geometry_report(coords)
    if v.le_spacing is not None:
        panels = np.linalg.norm(np.diff(coords, axis=0), axis=1)
        le = int(np.argmin(coords[:, 0]))
        report['panel_spacing'] = {
            'le': float(min(panels[le - 1], panels[le])),
            'te': float(max(panels[0], panels[-1] if panels[-1] > 0 else panels[-2])),
            'max': float(panels.max())}
    print(f'airfoil: {v.source} -> {args.output} ({rows} rows)')
    print(json.dumps(report, indent=2))
    return r


def do_ffd(args):
    from mdao_su2.geometry import ffd_lattice
    from mdao_su2.geometry.airfoil import read_surface
    from mdao_su2.mesh.recipe import resolve
    keys = ('kind', 'nx', 'spacing', 'zspan', 'x_bounds', 'y_bounds', 'z_bounds',
            'x_margin', 'y_margin', 'le_te', 'n_span', 'nz')
    r = resolve('ffd', args.config, _cli_values(args, keys), args.enable, args.level)
    v = r.values
    _prepare_output(args.output)
    if v.kind == 'airfoil':
        coords = None
        if v.x_margin is not None:
            if not args.surface:
                raise UsageError('margins need --surface <selig.dat>')
            coords = read_surface(args.surface)
        X, Y, Z = ffd_lattice.airfoil_box(v.nx, v.spacing, v.zspan, x_bounds=v.x_bounds,
                                      y_bounds=v.y_bounds, x_margin=v.x_margin,
                                      y_margin=v.y_margin, coords=coords, le_te=v.le_te)
    else:
        X, Y, Z = ffd_lattice.wing_box(v.nx, v.n_span, v.nz, v.x_bounds, v.y_bounds,
                                   v.z_bounds, spacing=v.spacing)
    ffd_lattice.write_lattice_plot3d(args.output, X, Y, Z)
    print(f'ffd: {v.kind} {X.shape} -> {args.output}')
    return r


def do_ffd_from_su2(args):
    from mdao_su2.geometry.ffd_lattice import read_ffd_boxes, write_plot3d
    if args.verify and args.tol is None:
        raise UsageError('--verify needs --tol')
    _prepare_output(args.output)
    boxes = read_ffd_boxes(args.mesh, with_surface_points=False)
    if args.box:
        boxes = [box for box in boxes if box.tag == args.box]
    if len(boxes) != 1:
        raise UsageError('select exactly one embedded FFD box with --box')
    box, = boxes
    write_plot3d(box.control_points, args.output)
    print(f'ffd-from-su2: {args.mesh} [{box.tag}] {box.shape} -> {args.output}')
    if args.verify:
        from mdao_su2.geometry.ffd_embedding import check_box
        r = check_box(args.mesh, args.output, args.tol, box.tag)
        print(_embedding_line(r))
        if not r['ok']:
            raise SystemExit(1)


def do_ffd_freeze(args):
    from mdao_su2.geometry.ffd_freeze import compute
    _, _, frozen, _ = compute(args.mesh, args.wall, args.continuity, args.symmetry_plane, args.box)
    for axis in 'ijk':
        print(f'freeze_{axis} = {frozen[axis]}')
    _emit(args, {'mesh': args.mesh, 'wall': args.wall, 'continuity': args.continuity,
                 'box': args.box, 'symmetry_plane': args.symmetry_plane,
                 **{f'freeze_{a}': [int(v) for v in frozen[a]] for a in 'ijk'}})


def _embedding_line(r):
    return (f"FFD_TAG {r['tag']}: {r['n_points']} points, kmax {r['kmax']}, "
            f"max |err| {r['max_err']:.3e} (tol {r['tol']:.0e}) "
            f"{'PASS' if r['ok'] else 'FAIL'}")


def do_check_ffd(args):
    from mdao_su2.geometry import ffd_embedding
    if args.enclosure:
        r = ffd_embedding.check_enclosure(args.mesh, args.enclosure, ffd_path=args.ffd,
                                          box_tag=args.box, kmax=args.kmax)
        print(f"FFD_TAG {r['tag']}: {r['n_points']} wall nodes, max embed residual "
              f"{r['max_residual']:.3e} (tol {r['tol']:.0e}) {'PASS' if r['ok'] else 'FAIL'}")
    else:
        r = ffd_embedding.check_box(args.mesh, args.ffd, args.tol, args.box, args.kmax)
        print(_embedding_line(r))
    _emit(args, {'mesh': args.mesh, 'ffd': args.ffd, 'enclosure': args.enclosure,
                 'passed': r['ok'], **r})
    if not r['ok']:
        raise SystemExit(1)


MESH2D_KEYS = ('first_cell_height', 'growth_ratio', 'n_bl_layers', 'bl_thickness',
               'fan_angle_deg', 'fan_elements', 'farfield_radius', 'farfield_meshsize',
               'near_meshsize', 'far_growth_slope', 'wall_meshsize', 'wall_marker',
               'farfield_marker', 'farfield_center', 'attempts', 'wake_length',
               'wake_angles_deg', 'wake_meshsize', 'wake_stream_slope', 'quality')


def _check_2d(mesh_path, step):
    """``(passed, measured, results)`` of ``mesh/quality2d.py`` and the marker check
    against a mesh2d step."""
    from mdao_su2.mesh.mesh2d import verify_mesh
    from mdao_su2.mesh.quality2d import check, measure
    measured = measure(mesh_path, step.wall_marker)
    results = check(measured, step.remesh_config(), step.wall_marker)
    markers_ok, report = verify_mesh(mesh_path, step.wall_marker, step.farfield_marker,
                                     verbose=False)
    results.append(('markers', markers_ok,
                    f"{report['markers']} (need {step.wall_marker}, {step.farfield_marker})"))
    return all(ok for _, ok, _ in results), measured, results


def _print_check(measured, results):
    from mdao_su2.mesh.quality2d import report_lines
    print('\n'.join(report_lines(measured, results)))


def _check_payload(mesh_path, passed, measured, results, **extra):
    """The JSON object of a check: the mesh, the verdict, the measurement, each gate."""
    from mdao_su2.mesh.quality2d import json_safe
    from mdao_su2.mesh.su2_mesh_file import read_mesh_header
    return {'mesh': os.path.abspath(mesh_path), 'header': read_mesh_header(mesh_path),
            'passed': bool(passed), 'measured': json_safe(measured),
            'gates': [{'name': n, 'ok': ok, 'detail': d} for n, ok, d in results], **extra}


def do_mesh2d(args):
    import numpy as np
    from mdao_su2.geometry.airfoil import read_dat
    from mdao_su2.mesh.mesh2d import remesh_airfoil
    from mdao_su2.mesh.recipe import resolve
    r = resolve('mesh2d', args.config, _cli_values(args, MESH2D_KEYS), args.enable,
                args.level)
    step = r.values
    _prepare_output(args.output)
    wall = read_dat(args.surface)
    if np.array_equal(wall[0], wall[-1]):
        wall = wall[:-1]                   # sharp TE: one wall node at the TE
    kwargs = step.gmsh_kwargs()
    kwargs['viewable_formats'] = tuple(args.view)
    kwargs['verbose'] = args.verbose
    attempts = step.attempts or 1
    for attempt in range(1, attempts + 1):
        if attempt == 1:
            remesh_airfoil(wall, args.output, np.arange(len(wall)), **kwargs)
        else:
            # gmsh can return another mesh in a new process.
            import multiprocessing
            with multiprocessing.get_context('spawn').Pool(1) as pool:
                pool.apply(remesh_airfoil, (wall, args.output, np.arange(len(wall))),
                           kwargs)
        passed, measured, results = _check_2d(args.output, step)
        _print_check(measured, results)
        if passed:
            break
        if attempt < attempts:
            print(f'mesh2d: attempt {attempt}/{attempts} failed its gates; regenerating')
    _emit(args, _check_payload(args.output, passed, measured, results,
                               attempts={'used': attempt, 'allowed': attempts}))
    if not passed:
        raise SystemExit(f'{args.output}: the mesh check failed; the mesh is kept as '
                         f'evidence and is not admitted')
    return r


def _check_3d(mesh_path, step):
    from mdao_su2.mesh import admission
    measured = admission.measure(mesh_path, [step.wing_marker, step.tip_marker],
                                 step.symmetry_marker, step.farfield_marker,
                                 step.farfield_center)
    results = admission.check(measured, step.markers, step.quality)
    return all(ok for _, ok, _ in results), measured, results


def _check_section2d(mesh_path, step):
    from mdao_su2.mesh import section2d as S
    from mdao_su2.mesh.quality2d import measure
    measured = measure(mesh_path, step.wall_marker)
    measured['wall_spacing'] = S.wall_spacing(mesh_path, step.wall_marker)
    results = S.check(measured, measured['wall_spacing'], step.quality,
                      step.wall_marker, step.farfield_marker)
    return all(ok for _, ok, _ in results), measured, results


def _level_of(args):
    if not args.level:
        raise UsageError('--level <name> picks the level_scale entry')
    return args.level


def do_wing3d(args):
    from mdao_su2.geometry.airfoil import read_dat
    from mdao_su2.mesh.admission import report_lines
    from mdao_su2.mesh.recipe import resolve
    from mdao_su2.mesh.wing3d import generate
    keys = ('span', 'farfield_radius', 'wall_size', 'le_size', 'te_size', 'tip_size',
            'far_size', 'size_growth', 'edge_band', 'te_band', 'te_normal_scale',
            'edge_distance', 'tip', 'farfield_center', 'quality', 'level_scale') + MARKERS_3D
    r = resolve('wing3d', args.config, _cli_values(args, keys), args.enable)
    step = r.values
    kwargs = step.generate_kwargs(_level_of(args))
    _prepare_output(args.output)
    section = read_dat(args.surface)
    generate(section, args.output, verbose=args.verbose, threads=args.threads, **kwargs)
    passed, measured, results = _check_3d(args.output, step)
    print('\n'.join(report_lines(measured, results)))
    _emit(args, _check_payload(args.output, passed, measured, results, level=args.level,
                               level_scale=kwargs['level_scale']))
    if not passed:
        raise SystemExit(f'{args.output}: the 3D check failed; the mesh is kept as '
                         f'evidence and is not admitted')
    return r


def do_section2d(args):
    from mdao_su2.geometry.airfoil import read_dat
    from mdao_su2.mesh.recipe import resolve
    from mdao_su2.mesh.section2d import generate
    keys = SECTION2D_SIZES + ('farfield_center', 'wall_marker', 'farfield_marker',
                              'quality', 'level_scale')
    r = resolve('section2d', args.config, _cli_values(args, keys), args.enable)
    step = r.values
    kwargs = step.generate_kwargs(_level_of(args))
    _prepare_output(args.output)
    generate(read_dat(args.surface), args.output, verbose=args.verbose, **kwargs)
    passed, measured, results = _check_section2d(args.output, step)
    _print_check(measured, results)
    _emit(args, _check_payload(args.output, passed, measured, results, level=args.level,
                               level_scale=kwargs['level_scale']))
    if not passed:
        raise SystemExit(f'{args.output}: the section2d check failed; the mesh is kept '
                         f'as evidence and is not admitted')
    return r


def do_view(args):
    from mdao_su2.mesh.su2_mesh_file import su2_mesh_to_vtu
    if args.markers and not args.boundary:
        raise UsageError('--markers selects what --boundary writes; give --boundary FILE')
    _prepare_output(*[p for p in (args.output, args.boundary) if p])
    summary = su2_mesh_to_vtu(args.mesh, args.output, args.boundary, args.markers)
    cells = ', '.join(f'VTK {t}: {n}' for t, n in summary['cells'].items())
    print(f"view: {args.mesh} -> {args.output} ({summary['ndim']}D, "
          f"{summary['n_points']} points, {sum(summary['cells'].values())} cells: {cells})")
    for m in summary['markers']:
        print(f"  marker {m['index']}: {m['tag']} ({m['n_elements']} elements)"
              + (' -> ' + args.boundary if m['written'] else ''))
    _emit(args, {'mesh': os.path.abspath(args.mesh), 'volume': os.path.abspath(args.output),
                 'boundary': os.path.abspath(args.boundary) if args.boundary else None,
                 **summary})


def do_all(args):
    """extract -> airfoil -> ffd -> mesh2d | wing3d | section2d, per level.

    File names come from ``[mesh.outputs]``: ``extract`` (default: the
    airfoil step's ``source``), ``surface``, ``ffd`` and ``mesh``. Each mesh
    step runs its own check and stops on a failed gate.
    """
    from mdao_su2.mesh.recipe import RecipeError, levels, load_recipe, output_path
    recipe = load_recipe(args.config)
    on = [s for s in ('extract', 'airfoil', 'ffd', 'mesh2d', 'wing3d', 'section2d')
          if recipe.get(s, {}).get('enabled') is True]
    for step in ('extract', 'airfoil', 'ffd', 'mesh2d', 'wing3d', 'section2d'):
        if step not in on:
            print(f'all: skip {step} (disabled)')
    base = os.path.abspath(args.output_dir) if args.output_dir else None
    declared = levels(recipe, on)
    if args.level == 'all':
        run_levels = declared or sorted({k for s in ('wing3d', 'section2d') if s in on
                                         for k in recipe[s].get('level_scale', {})})
        if not run_levels:
            raise RecipeError(f'{args.config}: --level all, but no step declares levels')
    else:
        run_levels = [args.level]
    if declared and args.level is None:
        raise RecipeError(f'all: the recipe declares levels {declared}; pass --level '
                          f'<name> or --level all')

    def ns(output, level, **kw):
        return SimpleNamespace(config=args.config, enable=False, output=output,
                               level=level, view=list(args.view),
                               verbose=False, threads=0, **kw)

    source = None
    if 'extract' in on:
        out = (output_path(recipe, args.config, 'extract', base_dir=base)
               if 'extract' in recipe.get('outputs', {}) else
               os.path.normpath(os.path.join(base or os.path.dirname(
                   os.path.abspath(args.config)), recipe.get('airfoil', {})['source'])))
        do_extract(ns(out, None, mesh=None, markers=None))
        source = out
    for level in run_levels:
        surface = args.surface
        if 'airfoil' in on:
            surface = output_path(recipe, args.config, 'surface', level, base)
            do_airfoil(ns(surface, level, source=source, n_points=None, spacing=None,
                          te=None))
        if 'ffd' in on:
            do_ffd(ns(output_path(recipe, args.config, 'ffd', level, base), level,
                      surface=surface))
        for step in ('mesh2d', 'wing3d', 'section2d'):
            if step not in on:
                continue
            if not surface:
                raise RecipeError(f'all: {step} needs an enabled airfoil step or --surface')
            if step != 'mesh2d' and not level:
                raise RecipeError(f'all: --level is required for enabled [mesh.{step}]')
            STEPS[step](ns(output_path(recipe, args.config, 'mesh', level, base), level,
                           surface=surface))
    return 0


STEPS = {'extract': do_extract, 'airfoil': do_airfoil, 'ffd': do_ffd,
         'ffd-from-su2': do_ffd_from_su2, 'ffd-freeze': do_ffd_freeze,
         'check-ffd': do_check_ffd,
         'mesh2d': do_mesh2d, 'wing3d': do_wing3d, 'section2d': do_section2d,
         'view': do_view,
         'all': do_all}


def main(argv=None):
    global _json_out
    args = build_parser().parse_args(argv)
    from mdao_su2.mesh.recipe import RecipeError
    json_mode = getattr(args, 'json', False)
    _json_out = sys.stdout if json_mode else None
    try:
        with contextlib.redirect_stdout(sys.stderr) if json_mode else contextlib.nullcontext():
            STEPS[args.step](args)
    except UsageError as exc:
        print(f'run/mesh.py {args.step}: {exc}', file=sys.stderr)
        return 2
    except (RecipeError, ValueError, KeyError, OSError) as exc:     # OSError: FileNotFoundError
        message = exc.args[0] if isinstance(exc, KeyError) and exc.args else exc
        print(f'run/mesh.py {args.step}: {message}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
