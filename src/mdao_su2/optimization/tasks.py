"""Task dispatch: turn a validated RunConfig into a completed run."""

import os
import time
import numpy as np

from ..run_output.results import (IterationPrinter, IterationSnapshotter,
                                  clear_stale_solutions, build_result,
                                  write_results, write_snapshot)
from ..run_config.checks.task import resolve_derivative_mode
from ..run_config.errors import ConfigError
from ..run_config.root import DERIVATIVE_TASKS
from ..run_output.final_artifacts import keep_final
from ..run_output.design_start import read_start, start_lines
from ..run_output.rundir import (finish, make_run_dir, record_provenance, seed_lines,
                                 seed_plan, seed_solutions)
from ..run_output.solve_convergence import track_solve_convergence
from ..solver import point_cfg_writer as cfggen
from ..solver.variable_names import SURF as _V
from .model import build_problem, gather_coeffs
from .solver_console import SolverConsole, show_warnings_once
from .optimizer import describe_outcome, optimizer_output, outcome
from .progress import Progress, plan_task
from .adjoint_reuse import reuse_adjoints
from .polar import polar
from .banner import run_banner
from .check_totals import (GateError, check_totals, moment_pivot_check,
                           moment_pivot_responses)

__all__ = ['run', 'execute', 'prepare', 'run_model', 'trim', 'compute_totals',
           'check_totals', 'moment_pivot_check', 'run_driver', 'polar', 'failure_line']


def prepare(cfg, comm=None, argv=None):
    """Create the run directory, generate the per-point cfgs, enter ``work/``."""
    rank0 = comm is None or comm.rank == 0
    rd = make_run_dir(cfg, comm=comm, argv=argv)
    if rank0:
        # First on the terminal and at the top of logs/progress.log.
        banner = run_banner(cfg, rd, 1 if comm is None else comm.size)
        print('\n'.join(banner) + '\n', flush=True)
        _log_progress(rd, '\n'.join(banner) + '\n')
    pairs = cfggen.generate_point_cfgs(cfg, rd.su2_cfg, comm=comm,
                                       relative_to=rd.workdir)
    if rank0:
        lines = list(cfggen.describe(cfg, pairs))
        lines.append(f'[cfg] config={cfg.source} name={cfg.name} '
                     f'points={cfg.n_points} task={cfg.execution.task} '
                     f'remesh={cfg.remesh.mode} '
                     f'parallel_points={cfg.execution.parallel_points}')
        lines += [f'[cfg] override {ov}' for ov in cfg.overrides]
        start = read_start(cfg)
        if start is not None:
            lines += start_lines(start)
        if cfg.output.seed:
            lines += seed_lines(seed_plan(cfg))
        requested, routed, reason = resolve_derivative_mode(cfg)
        if routed != requested and cfg.execution.task in DERIVATIVE_TASKS:
            lines.append(f'[cfg] WARNING: derivative_mode={requested!r} falls back to '
                         f'{routed!r}: {reason}')
        lines.append(f'[run] {rd.root}')
        # The console's progress.log tee opens later; the log keeps these too.
        print('\n'.join(lines), flush=True)
        _log_progress(rd, '\n'.join(lines))
    rd.enter()
    return pairs, rd


def run(cfg, comm=None, argv=None):
    """Dispatch on ``cfg.execution.task``.  Returns a result dict."""
    return execute(cfg, comm, argv)[0]


def failure_line(task, exc):
    """The one-line failure message: printed by the CLI, kept in ``progress.log``."""
    first = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
    if getattr(exc, 'reported', False):
        where = (f' The rows are in results_{task}.json of the run directory.'
                 if hasattr(exc, 'result') else '')
        return f'{task} did not pass ({first}); exit code {exc.exit_code}.{where}'
    return f'{task} FAILED: {type(exc).__name__}: {first}'


def _elapsed(seconds):
    if seconds < 90:
        return f'{seconds:.1f} s'
    if seconds < 7200:
        return f'{seconds / 60:.1f} min'
    return f'{seconds / 3600:.2f} h'


def _log_progress(rd, line):
    """Append ``line`` to ``logs/progress.log`` once the console has closed."""
    os.makedirs(rd.logs, exist_ok=True)
    with open(os.path.join(rd.logs, 'progress.log'), 'a') as stream:
        stream.write(line + '\n')


def execute(cfg, comm=None, argv=None):
    """Run the task; return ``(result, rd)`` with the ``RunDir`` it used."""
    physics = getattr(cfg.case, 'physics', None)
    if physics is not None and cfg.execution.task in DERIVATIVE_TASKS:
        from ..run_config.physics import profile_id
        from ..solver.response_catalog import status
        profile = profile_id(physics)
        state, evidence = status('drag', profile)
        if state in ('unsupported', 'fd_failed'):
            detail = ('CSolverFactory.cpp::CreateSolverContainer omits TRANS_SOL '
                      'for DISC_ADJ_RANS, but CDriver.cpp::Numerics_Preprocessing '
                      'dereferences it for LM. '
                      if '+lm[' in profile else '')
            raise ConfigError(
                f'{profile} derivative status is {state}: {detail}'
                f'evidence={evidence}. Use task="run_model" or "trim" '
                'for primal work.')
    show_warnings_once(comm)
    start = time.monotonic()
    pairs, rd = prepare(cfg, comm, argv)
    task = cfg.execution.task
    rank0 = comm is None or comm.rank == 0

    if task == 'run_driver' and cfg.execution.clean_restart:
        clear_stale_solutions(cfg, comm)
    # After the clean, so a seeded run_driver starts warm.
    if cfg.output.seed:
        seed_solutions(cfg, rd, comm)

    handler = {'run_model': run_model, 'trim': trim,
               'compute_totals': compute_totals,
               'check_totals': check_totals,
               'moment_pivot_check': moment_pivot_check,
               'run_driver': run_driver, 'polar': polar}[task]
    try:
        # Every task is quiet: SU2 output goes to logs/, never to the terminal
        # The console opens before the SU2 drivers are built.
        with SolverConsole(comm, logdir=rd.logs, quiet=True,
                           solver_stdout=getattr(cfg.output, 'solver_stdout', 'log')
                           ) as console:
            # Library text printed while the model is built (prefoil's CST
            # fit) belongs with the solver output, not on the terminal.
            with console.solver_output():
                prob = build_problem(cfg, pairs, comm=comm, rundir=rd,
                                     with_driver=(task == 'run_driver'))
            modes = _resolved_derivative_mode(prob, cfg, rank0)
            if rank0:
                record_provenance(rd, modes)
            progress = (_progress(prob, cfg, modes, console, rd)
                        if rank0 else None)
            gate = None
            with track_solve_convergence(prob.model, cfg, progress) as convergence:
                try:
                    result = handler(prob, cfg, rank0, console=console, rundir=rd,
                                     progress=progress)
                except GateError as exc:
                    # Write the gate's full record before the failure propagates.
                    gate, result = exc, exc.result
            if result is not None:
                result = _complete_result(result, prob, cfg, modes,
                                          convergence.record(prob.comm))
        if rank0 and result is not None:
            write_results(cfg, rd, result)
        if gate is not None:
            raise gate
    except BaseException as exc:
        # A run left saying RUNNING forever is indistinguishable from one still
        # going.  Label it, then re-raise unchanged.
        finish(rd, 'FAILED', type(exc).__name__, comm=comm)
        if rank0:
            # The CLI prints the same line; the log keeps it without a tee.
            _log_progress(rd, failure_line(task, exc))
        raise

    # final/ keeps each point's last solver artifacts; su2_workdir/ goes at
    # finish only when final/ was written.
    try:
        kept = keep_final(cfg, rd, comm)
    except Exception as exc:
        kept = False
        if rank0:
            print(f'\n[final] FAILED, su2_workdir/ is kept: {exc}', flush=True)

    span_load = getattr(getattr(getattr(cfg, 'postprocess', None),
                                'span_load', None), 'enabled', False)
    if (rank0 and result is not None and span_load
            and task in ('run_model', 'trim', 'run_driver')):
        try:
            from ..postprocess.span_load import compute_run_span_load
            result['span_load'] = compute_run_span_load(cfg, rd, result)
            print('[span_load] recorded in the results file', flush=True)
        except Exception as exc:
            # A post-processing failure must never destroy a finished run.
            print(f'\n[span_load] FAILED: {exc}', flush=True)
            result['span_load'] = {'error': str(exc)}
        write_results(cfg, rd, result)

    os.chdir(rd.root)               # su2_workdir/ may be removed below
    finish(rd, 'DONE', comm=comm, remove_workdir=kept)
    if rank0:
        line = (f'{task} DONE in {_elapsed(time.monotonic() - start)} '
                f'-> results_{task}.json')
        print(f'\n{line}', flush=True)
        _log_progress(rd, f'\n{line}')
    return result, rd


def _progress(prob, cfg, modes, console, rd):
    """Plan the task's solves, print the plan header, return the counter."""
    from ..run_output.solve_hooks import component_point_id, solver_components
    task = cfg.execution.task
    sizes = {name: int(np.size(prob.get_val(name))) for name in cfg.geometry_dvs}
    owned = []
    if hasattr(prob.model, 'system_iter'):
        owned = [component_point_id(c, cfg) for c in solver_components(prob.model)
                 if c.comm.rank == 0]
    plan = plan_task(cfg, modes['derivative_mode_resolved'], sizes,
                     points=[p.id for p in cfg.points if p.id in owned] or None)
    call = None
    if task == 'run_driver':
        call = lambda kind=None: getattr(prob.driver, 'iter_count', 0)   # noqa: E731
    solves = os.path.join(rd.logs, 'solves') if task != 'run_driver' else None
    names = list(cfg.geometry_dvs) + [f'alpha_{i}' for i in range(cfg.n_points)]

    def dv_values():
        return {name: prob.get_val(name) for name in names}

    progress = Progress(plan, console, solves_dir=solves, call=call,
                        dv_values=dv_values)
    progress.announce()
    return progress


def _complete_result(result, prob, cfg, modes, convergence):
    """Add the blocks every results file carries.

    Collective: response values and coefficients are gathered on every rank.
    """
    driver = list(getattr(prob.model, 'driver_responses', ()))
    if cfg.execution.task == 'moment_pivot_check':
        # The gate's file lists only the responses it differentiates.
        known = {path: (kind, kwargs) for path, kind, kwargs in driver}
        driver = [(path, *known.get(path, ('response', {})))
                  for path in moment_pivot_responses(prob, cfg)]
    listed = []
    for path, kind, kwargs in driver:
        value = np.asarray(prob.get_val(path, get_remote=True), dtype=float)
        entry = {'path': path, 'kind': kind,
                 'value': float(value[0]) if value.size == 1 else value.tolist()}
        if kind == 'objective':
            entry['expression'] = cfg.objective.expression
        entry.update({key: kwargs[key] for key in ('lower', 'upper', 'equals')
                      if kwargs.get(key) is not None})
        listed.append(entry)
    if 'points' not in result:
        cls, cds = gather_coeffs(prob, cfg)
        result['points'] = [
            {'id': p.id, 'alpha': float(prob.get_val(f'alpha_{i}')[0]),
             'C_L': cls[i], 'C_D': cds[i]} for i, p in enumerate(cfg.points)]
    head = {'derivative_mode': {'requested': modes['derivative_mode_requested'],
                                'resolved': modes['derivative_mode_resolved'],
                                'reason': modes['derivative_mode_reason']},
            'responses': listed}
    return {**head, **result, 'convergence': convergence}


def _resolved_derivative_mode(prob, cfg, rank0):
    """The requested direction, the one OpenMDAO uses, and why they differ.

    A ``fwd``/``auto`` request on a reverse-only configuration was set up as
    ``rev`` (``resolve_derivative_mode``); ``auto`` resolves in final_setup.
    """
    requested, routed, reason = resolve_derivative_mode(cfg)
    if routed == 'auto':
        prob.final_setup()          # collective; run_model would call it anyway
    resolved = prob._metadata['mode']
    if rank0 and cfg.execution.task in DERIVATIVE_TASKS:
        # The blank line closes the [cfg] block before the [plan] header.
        print(f'[cfg] derivative_mode={requested} -> {resolved}\n', flush=True)
    return {'derivative_mode_requested': requested,
            'derivative_mode_resolved': resolved,
            'derivative_mode_reason': reason}


# ------------------------------------------------------------------ tasks

def run_model(prob, cfg, rank0, console=None, rundir=None, progress=None):
    """One primal solve per point at the configured alpha; no derivatives."""
    prob.run_model()
    cls, cds = gather_coeffs(prob, cfg)          # collective: ALL ranks
    obj = float(prob.get_val('obj_comb.obj')[0])  # serial comp: local
    if rank0:
        for p, cl, cd in zip(cfg.points, cls, cds):
            print(f'{p.id} M={p.mach:.2f}:  C_L={cl:.6f}  C_D={cd:.6f}')
        print(f'Weighted drag (obj): {obj:.6f}')
    return build_result(cfg, cls, cds,
                   [float(prob.get_val(f'alpha_{i}')[0])
                    for i in range(cfg.n_points)])


def trim(prob, cfg, rank0, console=None, rundir=None, progress=None):
    """Secant-trim alpha to the lift target and report the coefficients there."""
    alphas, cls, cds = alpha_trim(prob, cfg, rank0)
    obj = float(prob.get_val('obj_comb.obj')[0])   # serial comp: local
    if rank0:
        targets = trim_targets(cfg)
        trimmed = [float(t) for t in targets if np.isfinite(t)]
        shown = (trimmed[0] if len(set(trimmed)) == 1
                 else ', '.join(f'{t:g}' for t in trimmed))
        print(f'\n=== Trimmed to C_L = {shown} ===')
        for i, p in enumerate(cfg.points):
            print(f'  {p.id} M={p.mach:.2f}:  alpha={alphas[i]:.6f}  '
                  f'C_L={cls[i]:.6f}  C_D={cds[i]:.6f}  '
                  f'(dC_L {cls[i] - targets[i]:+.2e})')
        print(f'  Weighted drag: {float(np.dot(cfg.weights, cds)):.6f}')
        print(f'  Objective ({cfg.objective.expression}): {obj:.6f}')
    return build_result(cfg, cls, cds, alphas)


def compute_totals(prob, cfg, rank0, console=None, rundir=None, progress=None):
    """The gradient the optimizer receives, and its cost -- no FD, no verdict."""
    from ..solver.standalone.memory_profile import get_peak_rss_mb

    comm = prob.comm
    t0 = time.time()
    prob.run_model()
    t_primal = time.time() - t0
    rss_primal = get_peak_rss_mb(comm)

    wrt = list(cfg.geometry_dvs) + [f'alpha_{i}' for i in range(cfg.n_points)]
    of = [path for path, _, _ in prob.model.driver_responses]
    t1 = time.time()
    with reuse_adjoints(prob.model) as reuse_counts:
        totals = prob.compute_totals(of=of, wrt=wrt)
    t_totals = time.time() - t1
    rss_totals = get_peak_rss_mb(comm)

    # Adjoint solves under rev, forward solves under fwd.
    kind = 'adjoint' if prob._metadata['mode'] == 'rev' else 'forward'
    cost = {'n_points': cfg.n_points, 'n_ranks': comm.size,
            'primal': {'wall_s': t_primal, 'peak_rss_mb': rss_primal},
            kind: {'wall_s': t_totals, 'peak_rss_mb': rss_totals}}
    by_name = {}
    for (output, dv), value in totals.items():
        by_name.setdefault(output, {})[dv] = np.asarray(value).tolist()
    if rank0:
        for key, val in totals.items():
            arr = np.asarray(val).ravel()
            print(f'{key}: {np.array2string(arr, precision=6)}')
        print(f'\n=== cost (np={comm.size}, {cfg.n_points} point(s)) ===')
        print(f'  primal            : {t_primal:8.1f} s   peak RSS '
              f'{rss_primal:9.1f} MB')
        print(f'  + {kind:<16s}: {t_totals:8.1f} s   peak RSS '
              f'{rss_totals:9.1f} MB')
    return {'totals': by_name, 'cost': cost, 'adjoint_reuse': dict(reuse_counts)}


def _shape_values(prob, cfg):
    """The shape DV vector, empty when [dv.shape] is off."""
    if 'shape' in cfg.geometry_dvs:
        return np.asarray(prob.get_val('shape'))
    return np.zeros(0)


def _snapshot_fields(cfg):
    """``[output.snapshots] mode`` as the ``write_snapshot`` field policy."""
    return 'none' if cfg.output.snapshots.mode == 'off' else 'surface'


def run_driver(prob, cfg, rank0, console=None, rundir=None, progress=None):
    """The full optimization: optional alpha trim, then pyOptSparse to convergence."""
    ndim = prob.model._meta_builder.ndim
    n = prob.model._meta_builder.n_nodes
    x_ref = prob.model._meta_builder.x_ref.copy()
    if rank0:
        np.save('baseline_surface.npy', x_ref.reshape(-1, ndim))

    trim_cds = trim_alphas = None
    snapshotter = None
    terminal = getattr(console, 'terminal', None)
    if cfg.trim.enabled:
        trim_alphas, trim_cls, trim_cds = alpha_trim(prob, cfg, rank0)
        if rank0 and terminal is not None:
            t = terminal
            print('\n=== Trimmed baseline ===', file=t)
            for i, p in enumerate(cfg.points):
                print(f'  {p.id} M={p.mach:.2f}:  alpha={trim_alphas[i]:.4f}'
                      f'  C_L={trim_cls[i]:.6f}  C_D={trim_cds[i]:.6f}',
                      file=t)
            print(f'  Weighted drag: '
                  f'{float(np.dot(cfg.weights, trim_cds)):.6f}', file=t)
            t.flush()

    # Collective, on EVERY rank: the write is rank-0-only inside on_root, so
    # a rank-0 failure raises everywhere instead of stranding the others at
    baseline_dvs = np.zeros_like(_shape_values(prob, cfg))
    baseline_surface = x_ref.reshape(-1, ndim)
    if cfg.dv.start and trim_cds is not None:
        # The trim solved the [dv] start design, so the baseline records it.
        baseline_dvs = _shape_values(prob, cfg).copy()
        geom = prob.model.geometry_name
        baseline_surface = np.asarray(prob.get_val(
            f'{geom}.{_V.Geometry.COORDINATES_OUTPUT}', get_remote=True)
        ).reshape(n, 3)[:, :ndim]
    if rundir is not None:
        write_snapshot(rundir, cfg, 'baseline', comm=prob.comm,
                       surface=baseline_surface, shape_dvs=baseline_dvs,
                       cds=trim_cds, alphas=trim_alphas,
                       # an untrimmed baseline has not solved yet.
                       fields=(_snapshot_fields(cfg)
                               if cfg.trim.enabled else 'none'),
                       want_volume=(cfg.trim.enabled
                                    and cfg.output.snapshots.keep_volume
                                    != 'none'))
        # The call tracker lives here, outside aero_comp: OpenMDAO
        # calls the recorder once per function evaluation on all ranks.
        snapshotter = IterationSnapshotter(
            rundir, cfg, prob.comm, baseline_surface=x_ref.reshape(-1, ndim),
            baseline_dvs=baseline_dvs, hist_file=prob.driver.options['hist_file'])
        prob.driver.add_recorder(snapshotter)
        call_label = getattr(snapshotter, 'call_label', None)
        if progress is not None and call_label is not None:
            # Progress lines name the call folder the solve belongs to.
            progress.call = call_label

    restore_hooks = (lambda: None)
    if rundir is not None and snapshotter is not None:
        from ..run_output.solve_hooks import install_solve_hooks
        restore_hooks = install_solve_hooks(prob, cfg, rundir, snapshotter)

    printer = IterationPrinter(cfg, terminal=terminal if rank0 else None,
                               progress=progress)
    if trim_cds is not None:
        printer.set_baseline(trim_cds)
    if snapshotter is not None:
        # One line per call, and a major line with its block once the call is
        # classified as a major iteration.
        snapshotter.printer = printer

    raised = None
    try:
        # pyOptSparse's tables and debug_print go to opt_hst/optimizer.log.
        with optimizer_output(rundir.opt if rundir is not None else None, rank0):
            # Outermost wrapper: a reused adjoint is not captured, printed or
            # counted as a solve.
            with reuse_adjoints(prob.model) as reuse_counts:
                prob.run_driver()
    except Exception as exc:       # noqa: BLE001 -- re-raised below
        raised = exc
    finally:
        restore_hooks()
    # A recorded stop is the cause: the hooks raise it from the optimizer's
    # callback, and OpenMDAO may re-raise an exception it stored on the way
    # (for example a gradient requested at the failed first design).
    stop = getattr(restore_hooks, 'state', {}).get('stop')
    if stop is not None:
        if snapshotter is not None:
            snapshotter.finish()
        if rank0:
            print(f'\nrun_driver stopped: {stop}', flush=True)
        if raised is not None and raised is not stop:
            raise stop from raised
        raise stop
    if raised is not None:
        raise raised

    # Collective gathers below run on ALL ranks, never inside a rank-0 branch.
    geom = prob.model.geometry_name
    x3d = prob.get_val(f'{geom}.{_V.Geometry.COORDINATES_OUTPUT}',
                       get_remote=True)
    cls, cds = gather_coeffs(prob, cfg)
    alphas = [float(prob.get_val(f'alpha_{i}')[0]) for i in range(cfg.n_points)]
    final_design = {name: np.asarray(value, dtype=float).ravel().tolist()
                    for name, value in prob.driver.get_design_var_values(
                        driver_scaling=False).items()}

    calls = getattr(snapshotter, 'calls', None) or []
    result = build_result(cfg, cls, cds, alphas, trim_cds)
    result['optimizer'] = outcome(prob.driver, calls, final_design)
    result['adjoint_reuse'] = dict(reuse_counts)
    if rank0:
        print(f'\n{describe_outcome(prob.driver, calls)}')
        for i, p in enumerate(cfg.points):
            delta = ''
            if trim_cds is not None:
                delta = (f'  dC_D={(cds[i] / trim_cds[i] - 1.0) * 100.0:+.2f}% '
                         f'vs the trimmed baseline')
            print(f'  {p.id} M={p.mach:.2f}:  alpha={alphas[i]:.4f}  '
                  f'C_L={cls[i]:.6f}  C_D={cds[i]:.6f}{delta}')

        surface = np.asarray(x3d).reshape(n, 3)[:, :ndim]
        shape_dvs = _shape_values(prob, cfg)
        np.save('optimized_surface.npy', surface)
        np.save('optimized_shape_dvs.npy', shape_dvs)
        for name in cfg.geometry_dvs:
            if name != 'shape':
                np.save(f'optimized_{name}.npy', np.asarray(prob.get_val(name)))
    else:
        surface = shape_dvs = None

    # Collective, and outside the rank-0 block for that reason.  Only
    # rank 0 has the gathered surface, so it is broadcast into the writer rather
    if rundir is not None and snapshotter is not None:
        surface = prob.comm.bcast(surface, root=0) if prob.comm else surface
        shape_dvs = prob.comm.bcast(shape_dvs, root=0) if prob.comm else shape_dvs
        geometry_values = {name: np.asarray(prob.get_val(name)).copy()
                           for name in cfg.geometry_dvs if name != 'shape'}
        component = getattr(prob.model, geom, None)
        for constraint_name in getattr(component, '_constraint_names', ()):
            try:
                geometry_values[constraint_name] = np.asarray(
                    prob.get_val(f'{geom}.{constraint_name}')).copy()
            except Exception:
                pass
        # final/ links the call folder at the optimizer's final design, or is
        # written from the final re-run when no call matches it.
        snapshotter.finish(surface=surface, shape_dvs=shape_dvs, cds=cds,
                           cls=cls, alphas=alphas,
                           geometry_values=geometry_values)
    return result


# Per-point lift trim used by task dispatch.
def trim_targets(cfg):
    """Per-point trim target: ``target_cl``, else ``start_cl``, else the run-level
    ``[constraints.lift]`` trim target (read only for a point that needs it).
    A fixed-alpha point is not trimmed and gets NaN."""
    out = []
    for p in cfg.points:
        if getattr(p, 'alpha', 'design') != 'design':
            out.append(np.nan)
            continue
        target = getattr(p, 'target_cl', None)
        if target is None:
            target = getattr(p, 'start_cl', None)
        if target is None:
            target = cfg.constraints.lift.trim_target
        out.append(float(target))
    return np.array(out)


def alpha_trim(prob, cfg, rank0=False):
    """Secant-trim all per-point alphas to lift target. Returns (alphas, cls, cds) in degrees."""
    t = cfg.trim
    targets = trim_targets(cfg)
    active = np.array([getattr(p, 'alpha', 'design') == 'design'
                       for p in cfg.points], dtype=bool)
    lo, hi = t.alpha_clip
    n = cfg.n_points

    alphas = np.array([float(prob.get_val(f'alpha_{i}')[0]) for i in range(n)])

    prob.run_model()
    cls, cds = (np.array(x) for x in gather_coeffs(prob, cfg))
    _report(cfg, alphas, cls, cds, rank0)

    if not np.any(active) or np.all(np.abs(cls[active] - targets[active]) <= t.tol):
        return alphas, cls, cds

    dcl_da = np.full(n, t.dcl_da_init)

    for _ in range(t.max_iter - 1):
        old_alphas, old_cls = alphas.copy(), cls.copy()
        alphas = old_alphas.copy()
        alphas[active] = np.clip(old_alphas[active] -
                                 (old_cls[active] - targets[active]) /
                                 dcl_da[active], lo, hi)

        for i in range(n):
            prob.set_val(f'alpha_{i}', alphas[i])

        prob.run_model()
        cls, cds = (np.array(x) for x in gather_coeffs(prob, cfg))

        # Update each point's local lift slope from its own last two iterates.
        # Guarded on both differences: a clipped alpha gives da == 0, and a
        da, dcl = alphas - old_alphas, cls - old_cls
        for i in range(n):
            if abs(da[i]) > 1e-10 and abs(dcl[i]) > 1e-10:
                dcl_da[i] = dcl[i] / da[i]

        _report(cfg, alphas, cls, cds, rank0)
        if np.all(np.abs(cls[active] - targets[active]) <= t.tol):
            break

    if rank0:
        print(f'[trim] DONE  max|CL-target| = '
              f'{np.max(np.abs(cls[active] - targets[active])):.2e}')
    return alphas, cls, cds


def _report(cfg, alphas, cls, cds, rank0):
    if not rank0:
        return
    for i, p in enumerate(cfg.points):
        print(f'[trim] {p.id} M={p.mach:.2f}: alpha={alphas[i]:.4f}  '
              f'C_L={cls[i]:.6f}  C_D={cds[i]:.6f}')
