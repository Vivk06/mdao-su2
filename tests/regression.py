#!/usr/bin/env python3
"""Small, explicit regression suite for shipped configurations and runs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib
import json
import math
import os
from pathlib import Path
import pkgutil
import shlex
import subprocess
import sys
import tempfile
import time
import traceback
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'src'))
os.environ.setdefault('OPENMDAO_REPORTS', '0')


@dataclass(frozen=True)
class TestCase:
    name: str
    toml: str | None
    task: str
    iteration_cap: int | None
    reference_values: dict[str, float] | None
    tolerance: float
    lane: str


# Solver references stay unset until a user run supplies reviewed values.
CASES = (
    TestCase('module-imports', None, 'imports', None, None, 0.0, 'fast'),
    TestCase('runnable-configs', None, 'configs', None, None, 0.0, 'fast'),
    TestCase('geometry-signs', None, 'geometry', None, None, 1e-10, 'fast'),
    TestCase('mesh2d-recipe', None, 'mesh', None, None, 0.0, 'fast'),
    TestCase('run-layout', 'cases/rae2822_rans_sa_hh/input/configs/run_sp.toml',
             'layout', None, None, 0.0, 'fast'),
    TestCase('euler-2d-primal', 'cases/adodg1_naca0012_euler_ffd/input/configs/run_sp.toml',
             'run_model', None, None, 0.01, 'solver'),
    TestCase('rans-sa-2d-primal', 'cases/rae2822_rans_sa_hh/input/configs/run_sp.toml',
             'run_model', None, None, 0.01, 'solver'),
    TestCase('euler-3d-ffd-primal', 'cases/oneram6_euler_ffd/input/configs/run_sp.toml',
             'run_model', None, None, 0.01, 'solver'),
    TestCase('euler-hh-adjoint-row',
             'cases/adodg1_naca0012_euler_ffd/input/configs/opt_hh_even_n20.toml',
             'compute_totals', None, None, 0.05, 'solver'),
    TestCase('two-iteration-optimization',
             'cases/adodg1_naca0012_euler_ffd/input/configs/opt_ffd_cp.toml',
             'run_driver', 2, None, 0.01, 'solver'),
)


def _require(test, message):
    if not test:
        raise AssertionError(message)


def _imports(_scratch: Path):
    import mdao_su2

    names = [entry.name for entry in pkgutil.walk_packages(
        mdao_su2.__path__, mdao_su2.__name__ + '.')]
    _require(names, 'no package modules discovered')
    for name in names:
        importlib.import_module(name)


def _configs(scratch: Path):
    from mdao_su2.run_config import load_config
    from mdao_su2.solver.cfg_file import parse_cfg
    from mdao_su2.solver.point_cfg_writer import generate_point_cfgs

    paths = sorted(REPO.glob('cases/*/input/configs/*.toml'))
    _require(paths, 'no runnable TOMLs found')
    for path in paths:
        cfg = load_config(path)
        folder = scratch / path.parent.parent.parent.name / path.stem
        pairs = generate_point_cfgs(cfg, str(folder))
        _require(len(pairs) == cfg.n_points, f'{path}: point count changed')
        for primal, adjoint in pairs:
            p, a = parse_cfg(folder / primal), parse_cfg(folder / adjoint)
            _require(p.get('MATH_PROBLEM') == 'DIRECT', f'{path}: primal side')
            _require(a.get('MATH_PROBLEM') == 'DISCRETE_ADJOINT',
                     f'{path}: adjoint side')
            _require(bool(p.get('MESH_FILENAME')) and bool(a.get('MESH_FILENAME')),
                     f'{path}: missing mesh input')


def _geometry(_scratch: Path):
    import numpy as np
    from pygeo import DVGeometry, DVGeometryCST
    from mdao_su2.geometry.hicks_henne import _HicksHenneDVGeo
    from mdao_su2.geometry.ffd import shape_functions
    from mdao_su2.geometry.twist import add_twist_dvs_dvgeo, station_angles
    from mdao_su2.mesh.su2_mesh_file import extract_baseline_surface
    from mdao_su2.postprocess.dvmap import load_lattice
    from mdao_su2.run_config import load_config

    surface = np.array([[0., 0., 0.], [.5, .1, 0.], [1., 0., 0.],
                        [.5, -.1, 0.]])
    hh = _HicksHenneDVGeo(surface, None, np.array([.5]), 1, False)
    hh.addPointSet(surface, 'surface')
    hh.setDesignVars({'shape': np.zeros(2)})
    _require(np.allclose(hh.update('surface'), surface), 'HH zero DV')
    hh.setDesignVars({'shape': np.ones(2)})
    moved = hh.update('surface')
    _require(moved[1, 1] > surface[1, 1] and moved[3, 1] < surface[3, 1],
             'HH outward sign')

    dat = REPO / 'cases/rae2822_rans_sa_hh/input/geom/rae2822_surface.dat'
    cst_cfg = load_config(REPO /
        'cases/rae2822_rans_sa_hh/input/configs/run_sp.toml')
    xy = extract_baseline_surface(cst_cfg.mesh_path, cst_cfg.case.wall_single)
    points = np.column_stack((xy, np.zeros(len(xy))))
    cst = DVGeometryCST(str(dat), numCST=8, idxChord=0, idxVertical=1)
    cst.addPointSet(points, 'surface')
    cst.addDV(dvName='upper', dvType='upper')
    cst.addDV(dvName='lower', dvType='lower')
    upper = cst.DVs['upper'].value.copy()
    lower = cst.DVs['lower'].value.copy()
    cst.setDesignVars({'upper': upper, 'lower': lower})
    zero = cst.update('surface')
    fit_error = np.sqrt(np.mean(np.sum((zero - points) ** 2, axis=1)))
    _require(fit_error < 0.00011, f'CST zero DV fit error {fit_error}')
    changed = upper.copy()
    changed[2] += 1.0
    cst.setDesignVars({'upper': changed, 'lower': lower})
    shifted = cst.update('surface')
    _require(np.max(shifted[:, 1] - zero[:, 1]) > 0.0, 'CST upper sign')

    cfg = load_config(REPO / CASES[9].toml)
    lattice = load_lattice(cfg.ffd_path)
    ffd = DVGeometry(cfg.ffd_path)
    shapes = shape_functions(cfg.dv.shape, lattice.shape,
                             ffd.getLocalIndex(0))
    _require(shapes, 'no FFD shape directions')
    ffd.addShapeFunctionDV('shape', shapes)
    wall_xy = extract_baseline_surface(cfg.mesh_path, cfg.case.wall_single)
    wall = np.column_stack((wall_xy, np.zeros(len(wall_xy))))
    ffd.addPointSet(wall, 'wall')
    ffd.setDesignVars({'shape': np.zeros(len(shapes))})
    _require(np.max(np.abs(ffd.update('wall') - wall)) < 1e-10,
             'FFD zero DV')
    ffd.setDesignVars({'shape': np.ones(len(shapes))})
    shift = ffd.update('wall')[:, 1] - wall[:, 1]
    _require(np.max(shift) > 0 and np.min(shift) >= -1e-10,
             'FFD positive displacement sign')

    wing = REPO / 'cases/adodg3_naca0012_euler_ffd/input/ffd/ffd_adodg3_wing_n5.xyz'
    twist_cfg = load_config(REPO /
        'cases/adodg3_naca0012_euler_ffd/input/configs/opt_ffd_twist_n5.toml')
    geo = DVGeometry(str(wing))
    n = add_twist_dvs_dvgeo(geo, twist_cfg.dv.twist)
    ys = np.linspace(0.0, 3.1, 5)
    ys[-1] -= 1e-9
    chords = np.array([[x, y, 0.0] for y in ys for x in (0.0, 1.0)])
    geo.addPointSet(chords, 'chords')
    geo.setDesignVars({'twist': np.zeros(n)})
    _require(np.max(np.abs(geo.update('chords') - chords)) < 1e-10,
             'twist zero DV')
    geo.setDesignVars({'twist': np.ones(n)})
    angles = station_angles(geo.update('chords'), 5)
    _require(np.max(np.abs(angles - 1.0)) < 1e-5, 'twist nose-up sign')


_RECIPE = '''[mesh.airfoil]
enabled = true
source = "naca4:0012:closed:129"
n_points = 97
spacing = "cosine"
te = "keep"
[mesh.mesh2d]
enabled = true
first_cell_height = 1e-3
growth_ratio = 1.3
n_bl_layers = 6
fan_angle_deg = 60.0
fan_elements = 5
farfield_radius = 10.0
farfield_meshsize = 2.0
near_meshsize = 0.01
far_growth_slope = 0.3
wall_meshsize = 0.0
wall_marker = "WALL_T"
farfield_marker = "FAR_T"
farfield_center = [0.5, 0.0]
[mesh.mesh2d.quality]
min_first_cell_height_ratio = 0.5
max_first_cell_height_ratio = 2.0
min_bl_thickness_ratio = 0.5
max_bl_thickness_ratio = 2.0
max_skewness = 0.85
max_aspect_ratio_p99 = 2000.0
'''


def _mesh(scratch: Path):
    from run import mesh as mesh_cli
    from mdao_su2.mesh.mesh2d import verify_mesh
    from mdao_su2.mesh.quality2d import check, measure
    from mdao_su2.mesh.recipe import resolve

    recipe = scratch / 'recipe.toml'
    recipe.write_text(_RECIPE)
    surface, mesh = scratch / 'naca.dat', scratch / 'naca.su2'
    _require(mesh_cli.main(['airfoil', '--config', str(recipe), '--output',
                            str(surface)]) == 0, 'airfoil recipe failed')
    _require(mesh_cli.main(['mesh2d', '--config', str(recipe), '--surface',
                            str(surface), '--output', str(mesh)]) == 0,
             'mesh recipe failed')
    passed, report = verify_mesh(str(mesh), 'WALL_T', 'FAR_T', verbose=False)
    _require(passed, f'mesh admission: {report}')
    step = resolve('mesh2d', str(recipe), {}).values
    failures = [row for row in check(measure(str(mesh), 'WALL_T'),
                                     step.remesh_config()) if not row[1]]
    _require(not failures, f'mesh quality: {failures}')


def _layout(scratch: Path):
    from run.optimize import build_parser, collect_overrides
    from mdao_su2.run_config import load_config
    from mdao_su2.run_output.rundir import finish, make_run_dir

    external = scratch / 'external'
    args = build_parser().parse_args([
        '--config', str(REPO / CASES[4].toml), '--output-root',
        str(external), '--run-id', 'regression'])
    cfg = load_config(REPO / CASES[4].toml, collect_overrides(args))
    rd = make_run_dir(cfg)
    _require(Path(rd.root) == external / 'regression', 'output root ignored')
    _require(Path(rd.run_toml).is_file(), 'run TOML missing')
    _require((Path(rd.root) / 'resolved_config.json').is_file(),
             'resolved configuration missing')
    finish(rd)
    _require((Path(rd.root) / 'STATUS').read_text().strip() == 'DONE',
             'run status not finalized')


_FAST = {'imports': _imports, 'configs': _configs, 'geometry': _geometry,
         'mesh': _mesh, 'layout': _layout}


def _observed(case: TestCase, result: dict) -> dict[str, float]:
    if case.task == 'compute_totals':
        totals = result.get('totals', {})
        for by_dv in totals.values():
            if 'shape' in by_dv:
                values = by_dv['shape']
                while isinstance(values, list):
                    values = values[0]
                return {'gradient_row_0': float(values)}
        raise ValueError('no shape derivative row in results')
    if case.task == 'run_driver':
        for entry in result.get('responses', []):
            if entry.get('kind') == 'objective':
                return {'objective': float(entry['value'])}
        raise ValueError('no objective response in results')
    point = result['points'][0]
    return {'C_L': float(point['C_L']), 'C_D': float(point['C_D'])}


def _run_solver(case: TestCase, out: Path, comm, print_references: bool):
    from run.optimize import main as optimize

    args = ['--config', str(REPO / case.toml), '--task', case.task,
            '--output-root', str(out), '--run-id', case.name]
    if case.task == 'run_driver':
        args += ['--set', f'optimizer.SLSQP.MAXIT={case.iteration_cap}']
    rc = optimize(args)
    codes = comm.allgather(rc)
    if any(code != 0 for code in codes):
        return False, f'run exit codes {codes}'
    if comm.rank == 0:
        try:
            run_root = out / case.name
            path = run_root / f'results_{case.task}.json'
            result = json.loads(path.read_text())
            _require((run_root / 'STATUS').read_text().strip() == 'DONE',
                     'run did not finish cleanly')
            if case.task == 'compute_totals':
                mode = result.get('derivative_mode', {}).get('resolved')
                _require(mode == 'rev', f'expected reverse adjoint, got {mode!r}')
            if case.task == 'run_driver':
                required = ('run.toml', 'resolved_config.json',
                            'optimization_summary.json', 'evaluations', 'final')
                missing = [name for name in required
                           if not (run_root / name).exists()]
                _require(not missing, f'optimization layout missing {missing}')
                majors = result.get('optimizer', {}).get('major_iterations')
                _require(isinstance(majors, int) and 1 <= majors <= 2,
                         f'expected one or two major iterations, got {majors!r}')
            observed = _observed(case, result)
            _require(all(math.isfinite(v) for v in observed.values()),
                     'non-finite result')
            if print_references:
                print(f'REFERENCE {case.name}: {json.dumps(observed, sort_keys=True)}')
            if case.reference_values is None:
                verdict = (False, 'reference pending; result recorded for review')
            else:
                errors = {key: abs(observed[key] - expected)
                          for key, expected in case.reference_values.items()}
                verdict = (all(error <= case.tolerance for error in errors.values()),
                           f'observed={observed}; absolute errors={errors}')
        except Exception as exc:
            verdict = (False, f'result inspection: {exc}')
    else:
        verdict = None
    return comm.bcast(verdict, root=0)


def _check_wiring(case: TestCase, scratch: Path, comm):
    """Exercise CLI, cfg routing and solver entry points with inert drivers."""
    import types
    import numpy as np
    from mdao_su2.optimization import tasks
    from mdao_su2.optimization.model import TopOpt
    from mdao_su2.run_config import load_config
    from mdao_su2.solver.cfg_file import parse_cfg
    from mdao_su2.solver.point_cfg_writer import generate_point_cfgs
    from run.optimize import main as optimize

    calls = []
    def intercepted_execute(cfg, comm=None, argv=None):
        if case.task == 'run_driver':
            _require(cfg.optimizer.settings['SLSQP']['MAXIT'] == case.iteration_cap,
                     'optimization iteration cap did not reach task dispatch')
        calls.append(('dispatch', cfg.execution.task, Path(cfg.source).resolve()))
        return None, None

    argv = ['--config', str(REPO / case.toml), '--task', case.task,
            '--output-root', str(scratch), '--run-id', case.name]
    if case.task == 'run_driver':
        argv += ['--set', f'optimizer.SLSQP.MAXIT={case.iteration_cap}']
    with patch.object(tasks, 'execute', intercepted_execute), \
         patch('mdao_su2.optimization.solver_console.show_warnings_once'):
        _require(optimize(argv) == 0, 'CLI did not dispatch')
    _require(calls == [('dispatch', case.task, (REPO / case.toml).resolve())],
             f'wrong CLI task or TOML: {calls}')

    cfg = load_config(REPO / case.toml, {'execution.task': case.task}, comm=comm)
    pairs = generate_point_cfgs(cfg, str(scratch), comm=comm)
    _require(len(pairs) == cfg.n_points, 'point cfg count')
    expected_solver = 'RANS' if case.name == 'rans-sa-2d-primal' else 'EULER'
    expected_builder = ('SU2Builder' if expected_solver == 'RANS' else
                        'ResponseSU2Builder' if case.name == 'euler-3d-ffd-primal'
                        else 'BidirSU2Builder')
    for point, (primal, adjoint) in zip(cfg.points, pairs):
        primal, adjoint = scratch / primal, scratch / adjoint
        p, a = parse_cfg(primal), parse_cfg(adjoint)
        _require(p.get('MATH_PROBLEM') == 'DIRECT' and
                 a.get('MATH_PROBLEM') == 'DISCRETE_ADJOINT', 'cfg side routing')
        _require(p.get('SOLVER') == a.get('SOLVER') == expected_solver,
                 'wrong primal/adjoint physics')
        _require(Path(p['MESH_FILENAME']).resolve() == Path(cfg.mesh_path).resolve(),
                 'wrong primal mesh')
        _require(p.get('KIND_TURB_MODEL') == a.get('KIND_TURB_MODEL'),
                 'turbulence differs between primal and adjoint')
        if expected_solver == 'RANS':
            _require(p.get('KIND_TURB_MODEL') == 'SA', 'RANS-SA route lost')
        builder = TopOpt._solver_builder(cfg, str(primal), str(adjoint), point)
        _require(type(builder).__name__ == expected_builder, 'wrong builder')
        _require(builder.primal_cfg == str(primal) and
                 builder.adjoint_cfg == str(adjoint), 'builder cfg connections')

        builder.n_nodes = 1
        builder.ndim = cfg.case.ndim
        coupling = builder.get_coupling_group_subsystem()
        coupling.setup()
        adapter_output = ('x_aero_3d' if cfg.case.ndim == 3 else 'x_aero_2d')
        _require(coupling._static_manual_connections.get('aero.x_aero',
                 (None,))[0] == f'adapter.{adapter_output}',
                 'geometry adapter is not connected to solver x_aero')
        aero = coupling._static_subsystems_allprocs['aero'].system
        _require(aero.options['primal_cfg'] == str(primal) and
                 aero.options['adjoint_cfg'] == str(adjoint),
                 'aero component received wrong cfg pair')
        expected_component = ('SU2AeroComp' if expected_builder == 'SU2Builder'
                              else 'ResponseSU2AeroComp' if
                              expected_builder == 'ResponseSU2Builder'
                              else 'SU2AeroBidirComp')
        _require(type(aero).__name__ == expected_component,
                 'wrong aero component in coupling group')
        aero.comm = comm
        aero._cl = aero._cd = None
        aero.x_aero0 = np.zeros(cfg.case.ndim)
        aero._ndim = cfg.case.ndim
        aero._gid_pos = {}
        aero._solution_file = str(scratch / f'fake_solution_{comm.rank}.dat')
        aero._restart_file = str(scratch / f'fake_restart_{comm.rank}.dat')
        aero._conv_minval = None
        aero._fixed_cl_mode = False
        aero._response_records = tuple(getattr(builder, 'responses', ()))
        aero._response_values = {}
        aero._apply_surface = lambda driver, x: (0, 0)
        aero._sync_solution = lambda: Path(aero._solution_file).touch()
        events = []

        class DummyDriver:
            def __init__(self, path, _zones, _comm):
                self.config = parse_cfg(path)
                events.append(('construct', self.config['MATH_PROBLEM'],
                               self.config['SOLVER'],
                               self.config.get('OBJECTIVE_FUNCTION')))
            def SetFarFieldAoA(self, value):
                events.append(('alpha', value))
            def StartSolver(self):
                events.append(('start', self.config['MATH_PROBLEM']))
            def GetOutputValue(self, key):
                return {'LIFT': .2, 'DRAG': .01, 'SENS_AOA': .1}.get(key, 0.0)
            def Finalize(self):
                pass

        fake_primal = types.SimpleNamespace(CSinglezoneDriver=DummyDriver)
        fake_adjoint = types.SimpleNamespace(CDiscAdjSinglezoneDriver=DummyDriver)
        inputs = {'x_aero': np.zeros(cfg.case.ndim),
                  'alpha': np.array([point.aoa_init])}
        outputs = {name: np.zeros(1) for name in ('C_L', 'C_D', 'C_M')}
        outputs.update({record.output: np.zeros(1)
                        for record in aero._response_records})
        with patch.dict(sys.modules, {'pysu2': fake_primal, 'pysu2ad': fake_adjoint}):
            aero.compute(inputs, outputs)
            if case.task in ('compute_totals', 'run_driver'):
                d_inputs = {'x_aero': np.zeros(cfg.case.ndim),
                            'alpha': np.zeros(1)}
                aero.compute_jacvec_product(
                    inputs, d_inputs, {'C_D': np.ones(1)}, 'rev')
        _require(('start', 'DIRECT') in events, 'primal solver was not called')
        _require(any(e[:3] == ('construct', 'DIRECT', expected_solver)
                     for e in events), f'wrong primal driver: {events}')
        if case.task in ('compute_totals', 'run_driver'):
            _require(('start', 'DISCRETE_ADJOINT') in events,
                     'adjoint solver was not called')
            _require(any(e[:2] == ('construct', 'DISCRETE_ADJOINT') and
                         e[3] == 'DRAG' for e in events),
                     f'wrong adjoint objective: {events}')
        calls.extend(events)
    return f'{cfg.case.physics.kind}, {expected_builder}, {len(pairs)} point(s), dummy drivers'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lane', choices=('fast', 'wiring', 'solver'), default='fast')
    parser.add_argument('--np', type=int)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--print-references', action='store_true')
    args = parser.parse_args(argv)
    if args.np is None:
        args.np = 4 if args.lane == 'solver' else 1
    if args.np < 1:
        parser.error('--np must be positive')
    if args.lane == 'fast':
        selected = [case for case in CASES if case.lane == 'fast']
        start = time.monotonic()
        with tempfile.TemporaryDirectory(prefix='mdao-su2-regression-',
                                         dir=os.environ.get('TMPDIR')) as name:
            root = Path(name)
            failed = 0
            for case in selected:
                try:
                    folder = root / case.name
                    folder.mkdir()
                    _FAST[case.task](folder)
                    print(f'PASS {case.name}')
                except Exception as exc:
                    failed += 1
                    print(f'FAIL {case.name}: {exc}', file=sys.stderr)
                    traceback.print_exc()
        print(f'Fast lane: {len(selected) - failed}/{len(selected)} passed '
              f'in {time.monotonic() - start:.1f}s')
        return int(bool(failed))

    if args.lane == 'solver' and args.out is None:
        parser.error('--out is required for the solver lane')
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    if comm.size == 1 and args.np > 1:
        prefix = (['mpirun', '-np', str(args.np)] if args.lane == 'wiring'
                  else shlex.split(subprocess.check_output(
                      ['bash', str(REPO / 'tools/bind_cores.sh'), str(args.np)],
                      text=True).splitlines()[-1]))
        command = prefix + [sys.executable, str(Path(__file__).resolve()),
                            '--lane', args.lane, '--np', str(args.np)]
        if args.out is not None:
            command += ['--out', str(args.out)]
        if args.print_references and args.lane == 'solver':
            command.append('--print-references')
        return subprocess.call(command, cwd=REPO)
    if comm.size != args.np:
        parser.error(f'--np {args.np} does not match MPI size {comm.size}')
    if args.lane == 'wiring':
        root = (tempfile.mkdtemp(prefix='mdao-su2-wiring-',
                dir=os.environ.get('TMPDIR')) if comm.rank == 0 else None)
        root = Path(comm.bcast(root, root=0))
        failed = 0
        try:
            for case in (item for item in CASES if item.lane == 'solver'):
                try:
                    detail = _check_wiring(case, root / case.name, comm)
                    verdict = (True, detail)
                except Exception as exc:
                    verdict = (False, f'{type(exc).__name__}: {exc}')
                    traceback.print_exc()
                verdicts = comm.allgather(verdict)
                if comm.rank == 0:
                    passed = all(v[0] for v in verdicts)
                    print(f'{"PASS" if passed else "FAIL"} {case.name}: '
                          f'{verdicts[0][1]}', flush=True)
                    failed += not passed
            return int(bool(comm.bcast(failed, root=0)))
        finally:
            comm.Barrier()
            if comm.rank == 0:
                import shutil
                shutil.rmtree(root)

    out = args.out.resolve()
    if comm.rank == 0:
        out.mkdir(parents=True, exist_ok=True)
    comm.Barrier()
    failed = 0
    for case in (item for item in CASES if item.lane == 'solver'):
        try:
            passed, detail = _run_solver(case, out, comm,
                                         args.print_references)
        except Exception as exc:
            passed, detail = False, str(exc)
        if comm.rank == 0:
            print(f'{"PASS" if passed else "FAIL"} {case.name}: {detail}',
                  flush=True)
        failed += not passed
    return int(bool(failed))


if __name__ == '__main__':
    sys.exit(main())
