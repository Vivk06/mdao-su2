#!/usr/bin/env python3
"""Peak RSS memory profile: primal vs adjoint (DRAG, LIFT).

Measures peak resident set size via resource.getrusage (RUSAGE_SELF) around
each solve.  Writes results to stdout and summarizes the adjoint/primal ratio
with a 10× mesh extrapolation.

Usage (serial):
    python memory_profile.py --primal-cfg /tmp/su2cfg/rae2822_primal.cfg \
                             --adjoint-cfg /tmp/su2cfg/rae2822_adjoint.cfg \
                             --case-dir cases/rae2822_rans_sa_hh

The cfg is GENERATED from the TOML; no cfg template is tracked.  Either reuse
the cfg a finished run kept::

    cases/<case>/output/<run_id>/final/<point>/primal.cfg

or emit a standalone pair (per-point suffixes stripped, so the solver writes
`restart_flow.dat`)::

    python -c "from mdao_su2.run_config import load_config; \
from mdao_su2.solver.point_cfg_writer import write_unsuffixed_pair as w; \
w(load_config('cases/rae2822_rans_sa_hh/input/configs/run_sp.toml'), '/tmp/su2cfg')"

This is a PRIMARY EXECUTION script — the user should run it tee'd to a log.
"""

import argparse
import os
import resource
import shutil

from .run_primal import run_primal
from .run_adjoint import run_adjoint_one, _ensure_primal_solution


def get_peak_rss_mb(comm=None):
    """Return peak RSS in MB (Linux: ru_maxrss is in KB).

    Under MPI, returns the SUM across all ranks (total machine footprint);
    per-rank peak is the local value before reduction.
    """
    usage = resource.getrusage(resource.RUSAGE_SELF)
    local = usage.ru_maxrss / 1024.0
    if comm is None:
        return local
    return comm.allreduce(local)


def run_memory_profile(primal_cfg, adjoint_cfg, case_dir, comm=None):
    """Profile memory for primal and adjoint solves."""
    os.chdir(case_dir)
    rank = 0 if comm is None else comm.Get_rank()

    if rank == 0:
        print("=" * 60)
        print("  Memory Profile: Primal vs Adjoint")
        print("=" * 60)

    # --- Primal ---
    rss_before = get_peak_rss_mb(comm)
    cl, cd, drv = run_primal(primal_cfg, comm=comm)

    # Get mesh size
    n_nodes = drv.GetNumberNodes()
    n_dims = drv.GetNumberDimensions()
    drv.Finalize()
    rss_primal = get_peak_rss_mb(comm)

    if rank == 0:
        print(f"\n  Primal peak RSS:     {rss_primal:.1f} MB")
        print(f"  Mesh nodes (rank):   {n_nodes}")

    # Prepare for adjoint
    _ensure_primal_solution()

    # --- Adjoint (DRAG) ---
    drag_sens, drag_aoa, _, drv = run_adjoint_one(adjoint_cfg, 'DRAG', comm=comm)
    drv.Finalize()
    rss_adj_drag = get_peak_rss_mb(comm)

    if rank == 0:
        print(f"  Adjoint DRAG RSS:    {rss_adj_drag:.1f} MB")

    # --- Adjoint (LIFT) ---
    lift_sens, lift_aoa, _, drv = run_adjoint_one(adjoint_cfg, 'LIFT', comm=comm)
    drv.Finalize()
    rss_adj_lift = get_peak_rss_mb(comm)

    if rank == 0:
        print(f"  Adjoint LIFT RSS:    {rss_adj_lift:.1f} MB")

    # --- Summary ---
    rss_adj_max = max(rss_adj_drag, rss_adj_lift)
    ratio = rss_adj_max / rss_primal if rss_primal > 0 else float('inf')

    # 10× mesh extrapolation (linear in node count)
    extrap_primal = rss_primal * 10
    extrap_adj = rss_adj_max * 10

    if rank == 0:
        print(f"\n{'='*60}")
        print(f"  SUMMARY")
        print(f"{'='*60}")
        print(f"  Mesh:              {n_nodes} nodes, {n_dims}D")
        print(f"  Primal peak RSS:   {rss_primal:.1f} MB")
        print(f"  Adjoint peak RSS:  {rss_adj_max:.1f} MB  (max of DRAG/LIFT)")
        print(f"  Adjoint/Primal:    {ratio:.2f}×")
        print(f"  10× mesh extrap:   primal ~{extrap_primal:.0f} MB, adjoint ~{extrap_adj:.0f} MB")
        print(f"{'='*60}")

    return {
        'n_nodes': n_nodes,
        'n_dims': n_dims,
        'rss_primal_mb': rss_primal,
        'rss_adj_drag_mb': rss_adj_drag,
        'rss_adj_lift_mb': rss_adj_lift,
        'rss_adj_max_mb': rss_adj_max,
        'ratio': ratio,
        'extrap_primal_10x_mb': extrap_primal,
        'extrap_adj_10x_mb': extrap_adj,
    }


def main():
    parser = argparse.ArgumentParser(description="Memory profile: primal vs adjoint")
    parser.add_argument("--primal-cfg", required=True, help="Primal config file")
    parser.add_argument("--adjoint-cfg", required=True, help="Adjoint config file template")
    parser.add_argument("--case-dir", required=True, help="Case root directory")
    args = parser.parse_args()

    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
    except ImportError:
        comm = None

    run_memory_profile(args.primal_cfg, args.adjoint_cfg, args.case_dir, comm=comm)


if __name__ == "__main__":
    main()
