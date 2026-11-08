#!/usr/bin/env python
"""Run a configured SU2 task or print its resolved configuration.

Tasks include a model evaluation, derivative checks, a polar sweep, and
optimization. ``--print-config`` inspects the TOML without MPI or a solver.
Run output uses the configured root (or ``--output-root``); ``--archive``
zips the run this command completed.
"""

import argparse
import json
import os
import sys

# Suppress OpenMDAO's auto-generated {problem}_out/reports/ trees. Must precede
# the openmdao import, which happens inside mdao_su2.optimization.
os.environ.setdefault('OPENMDAO_REPORTS', '0')


def build_parser():
    # Choices from the schema, never hardcoded.
    from mdao_su2.run_config.root import OPTIMIZERS, TASKS
    from mdao_su2.run_config.checks.remesh import REMESH_MODES

    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', required=True, help='path to a run TOML')
    p.add_argument('--task', choices=list(TASKS))
    p.add_argument('--optimizer', choices=list(OPTIMIZERS))
    p.add_argument('--cl-target', type=float)
    p.add_argument('--remesh', choices=list(REMESH_MODES))
    p.add_argument('--ffd', help='geometry.ffd key, e.g. ffd10 / ffd20 / ffd30')
    p.add_argument('--run-id')
    p.add_argument('--output-root', help='directory for runs (absolute or relative to the case root)')
    p.add_argument('--on-exists', choices=['fail', 'resume', 'overwrite'],
                   help='what to do if the run directory already exists')
    p.add_argument('--seed-from', metavar='RUN_ID',
                   help='seed the flow and adjoint restarts of every point from '
                        'that run\'s final/: one flow and one adjoint '
                        '[[output.seed]] entry')
    # Concurrent points: requires np % n_points == 0, changes model topology.
    p.add_argument('--parallel-points', action='store_true', default=None)
    p.add_argument('--no-trim', action='store_true')
    p.add_argument('--no-constraints', action='store_true')
    p.add_argument('--set', dest='overrides', action='append', default=[],
                   metavar='KEY=VALUE',
                   help='override any config key, repeatable '
                        '(e.g. --set remesh.n_bl_layers=35)')
    p.add_argument('--band', dest='bands', action='append', default=[],
                   metavar='NAME=VALUE',
                   help='state an acceptance band for check_totals, repeatable: '
                        'alpha, shape (the flat band) or moment_pivot; applied '
                        'as a --set of its [verification] key')
    p.add_argument('--print-config', action='store_true',
                   help='dump the resolved config as JSON and exit '
                        '(no solver, no MPI)')
    p.add_argument('--archive', action='store_true',
                   help='zip the run directory after completion; logs/progress.log '
                        'holds what the terminal showed')
    return p


# Named flag -> the dotted config key it sets.  A flag is only sugar for a
# --set, so there is exactly one override mechanism to reason about.
_FLAG_KEYS = {
    'task': 'execution.task',
    'optimizer': 'optimizer.name',
    'cl_target': 'constraints.lift.equals',   # sets the equality; no-op under kind='min'/'range'
    'remesh': 'remesh.mode',
    'ffd': 'geometry.ffd',
    'run_id': 'output.run_id',
    'output_root': 'output.root',
    'on_exists': 'output.on_exists',
    'parallel_points': 'execution.parallel_points',
}


def collect_overrides(args):
    """Named flags first, then --band, then --set, so --set always wins."""
    from mdao_su2.run_config import parse_band_override, parse_set_override

    overrides = {}
    for flag, key in _FLAG_KEYS.items():
        value = getattr(args, flag)
        if value is not None:
            overrides[key] = value
    if args.seed_from is not None:
        overrides['output.seed'] = [{'kind': kind, 'from': args.seed_from}
                                    for kind in ('flow', 'adjoint')]
    if args.no_trim:
        overrides['trim.enabled'] = False
    if args.no_constraints:
        overrides['constraints.geometry'] = False
    for item in args.bands:
        key, value = parse_band_override(item)
        overrides[key] = value
    for item in args.overrides:
        key, value = parse_set_override(item)
        overrides[key] = value
    return overrides


def main(argv=None):
    args = build_parser().parse_args(argv)

    from mdao_su2.run_config import ConfigError, load_config
    overrides = collect_overrides(args)

    if args.print_config:
        # No MPI, no solver: the fast path for checking what a sweep will run.
        from mdao_su2.run_config import resolve_dict
        from mdao_su2.run_config.toggles import report
        cfg = load_config(args.config, overrides, check_files=False)
        resolved = cfg.to_dict()
        resolved['toggles'] = report(resolve_dict(args.config, overrides),
                                     cfg.overrides)
        resolved['formulation'] = cfg.objective_formulation
        json.dump(resolved, sys.stdout, indent=2, default=str)
        sys.stdout.write('\n')
        return 0

    from mpi4py import MPI
    from mdao_su2.optimization.solver_console import show_warnings_once
    from mdao_su2.optimization.tasks import execute, failure_line

    comm = MPI.COMM_WORLD
    show_warnings_once(comm)
    try:
        cfg = load_config(args.config, overrides, comm=comm)
    except ConfigError as exc:
        # Every rank raises the same error; only one should print it.
        if comm.rank == 0:
            print(f'CONFIG ERROR: {exc}', file=sys.stderr)
        return 2

    # argv goes into provenance.json for reproducibility.
    try:
        _, rd = execute(cfg, comm=comm,
                        argv=sys.argv if argv is None else ['run/optimize.py', *argv])
    except ConfigError as exc:
        if comm.rank == 0:
            print(f'CONFIG ERROR: {exc}', file=sys.stderr)
        return 2
    except Exception as exc:
        # A gate or a stopped driver printed its own report; a traceback would
        # add nothing. Exit 1 = failed, 3 = inconclusive. execute() has put the
        # same line into logs/progress.log.
        if not getattr(exc, 'reported', False):
            raise
        if comm.rank == 0:
            print(failure_line(cfg.execution.task, exc), file=sys.stderr)
        return exc.exit_code

    if args.archive and comm.rank == 0:
        _archive_run(rd.root)

    return 0


def _archive_run(run_path):
    """Zip the run directory this process used."""
    from mdao_su2.run_output.archive import zip_run
    result = zip_run(run_path)
    compressed_mb = result['compressed_bytes'] / (1024 * 1024)
    print(f"  Archive: {result['zip_path']}")
    print(f"  Files: {result['n_files']}  Size: {compressed_mb:.1f} MB")


if __name__ == '__main__':
    sys.exit(main())
