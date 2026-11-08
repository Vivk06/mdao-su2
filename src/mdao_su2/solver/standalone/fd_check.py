#!/usr/bin/env python3
"""Finite-difference vs adjoint verification — Nemec Table 1 style.

Central FD on ≥5 surface nodes (y-displacement) + AoA, step sweep over
{1e-5, 1e-6, 1e-7}·chord.  Prints a comparison table and reports the plateau
FD value against the adjoint sensitivity.

Gate: |FD − adjoint| / |adjoint| ≤ 1% on ≥5 nodes (both CD and CL).

MPI-aware: works correctly with mpirun -np N.  Each SU2 solve uses all ranks;
node selection is gathered globally and perturbations are routed to the
owning rank.

Prerequisites:
    1. Primal must have converged (restart_flow.dat exists in --case-dir).
    2. Adjoint sensitivities computed (run_adjoint.py) — or this script
       computes them in-line.

Usage (MPI, recommended):
    mpirun -np 4 python fd_check.py --primal-cfg /tmp/su2cfg/rae2822_primal.cfg \
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

import numpy as np

from ..wall_seam import DEG
from .run_primal import run_primal
from .run_adjoint import run_adjoint_one, _ensure_primal_solution

CHORD = 1.0

# Node selection: avoid LE (tiny CD sensitivity → huge relative error); keep
# shock-adjacent nodes to small FD steps.
NODE_DESCRIPTIONS = [
    ("10%c upper",  lambda x, y: abs(x - 0.10*CHORD) + (0 if y > 0 else 100), 'min'),
    ("25%c upper",  lambda x, y: abs(x - 0.25*CHORD) + (0 if y > 0 else 100), 'min'),
    ("25%c lower",  lambda x, y: abs(x - 0.25*CHORD) + (0 if y < 0 else 100), 'min'),
    ("45%c upper",  lambda x, y: abs(x - 0.45*CHORD) + (0 if y > 0 else 100), 'min'),
    ("near-TE",     lambda x, y: abs(x - 0.95*CHORD), 'min'),
]


def _select_plateau(fd_vals, steps):
    """Select FD value at the convergence plateau (adjacent-step agreement).

    Returns (best_fd, best_h).  With ≤1 value, returns the only one.
    With 2+, finds the pair of adjacent steps with smallest relative change
    and returns their average (and the larger step of the pair as best_h).
    """
    if len(fd_vals) <= 1:
        return fd_vals[0], steps[0]

    best_rel = float('inf')
    best_i = 0
    for i in range(len(fd_vals) - 1):
        denom = max(abs(fd_vals[i]), abs(fd_vals[i + 1]), 1e-30)
        rel = abs(fd_vals[i] - fd_vals[i + 1]) / denom
        if rel < best_rel:
            best_rel = rel
            best_i = i

    avg = 0.5 * (fd_vals[best_i] + fd_vals[best_i + 1])
    return avg, steps[best_i]


def _gather_marker_coords(driver, marker_id, comm):
    """Gather all AIRFOIL coordinates across MPI ranks.

    Returns (identical on all ranks):
        list of (owner_rank, local_iVertex, x, y)
    """
    if marker_id >= 0:
        nVertex = driver.GetNumberMarkerNodes(marker_id)
        ini = driver.MarkerInitialCoordinates(marker_id)
        local = [(iv, float(ini.Get(iv)[0]), float(ini.Get(iv)[1]))
                 for iv in range(nVertex)]
    else:
        local = []

    if comm is None:
        return [(0, iv, x, y) for iv, x, y in local]

    all_local = comm.allgather(local)
    global_nodes = []
    for r, rank_data in enumerate(all_local):
        for iv, x, y in rank_data:
            global_nodes.append((r, iv, x, y))
    return global_nodes


def select_nodes(driver, marker_id, comm):
    """Pick ≥5 representative surface nodes by chordwise location (MPI-safe).

    Returns (identical on all ranks):
        list of (owner_rank, local_iVertex, desc, x, y)
    """
    global_nodes = _gather_marker_coords(driver, marker_id, comm)

    selected = []
    for desc, score_fn, mode in NODE_DESCRIPTIONS:
        scores = [score_fn(x, y) for _, _, x, y in global_nodes]
        if mode == 'min':
            idx = int(np.argmin(scores))
        else:
            idx = int(np.argmax(scores))
        owner_rank, local_iv, x, y = global_nodes[idx]
        selected.append((owner_rank, local_iv, desc, x, y))

    return selected


def fd_primal_perturbed(cfg_path, owner_rank, local_iv_target, dy, comm):
    """Run primal with one global node displaced by dy in the y-direction.

    Only the owning rank applies the displacement; all other ranks set zero.
    """
    import pysu2

    if comm is None:
        comm_arg = 0
        rank = 0
    else:
        comm_arg = comm
        rank = comm.Get_rank()

    driver = pysu2.CSinglezoneDriver(cfg_path, 1, comm_arg)

    marker_indices = driver.GetMarkerIndices()
    mid = marker_indices.get('AIRFOIL', -1)
    nVertex = driver.GetNumberMarkerNodes(mid) if mid >= 0 else 0

    for iv in range(nVertex):
        if rank == owner_rank and iv == local_iv_target:
            driver.SetMarkerCustomDisplacement(mid, iv, (0.0, dy))
        else:
            driver.SetMarkerCustomDisplacement(mid, iv, (0.0, 0.0))

    driver.StartSolver()
    cl = driver.GetOutputValue('LIFT')
    cd = driver.GetOutputValue('DRAG')
    driver.Finalize()
    return cl, cd


def fd_primal_aoa_perturbed(cfg_path, base_alpha_deg, dalpha_deg, comm):
    """Run primal with AoA perturbed by dalpha_deg (degrees)."""
    cl, cd, driver = run_primal(cfg_path, comm=comm, alpha_deg=base_alpha_deg + dalpha_deg)
    driver.Finalize()
    return cl, cd


def _get_adj_sensitivity(sens_disp, marker_id, owner_rank, local_iv, comp, comm):
    """Extract adjoint sensitivity at a globally-selected node.

    The owning rank reads its local sens_disp and broadcasts to all ranks.
    comp: 0 for x-component, 1 for y-component.
    """
    rank = 0 if comm is None else comm.Get_rank()

    if comm is None:
        return float(sens_disp[local_iv, comp])

    if rank == owner_rank and marker_id >= 0:
        val = float(sens_disp[local_iv, comp])
    else:
        val = 0.0

    val = comm.bcast(val, root=owner_rank)
    return val


def run_fd_check(primal_cfg, adjoint_cfg, case_dir, steps=None, comm=None):
    """Run the full FD-vs-adjoint check.

    Returns
    -------
    results : list of dict, one per node + AoA
    """
    os.chdir(case_dir)

    if steps is None:
        steps = [1e-5 * CHORD, 1e-6 * CHORD, 1e-7 * CHORD]

    rank = 0 if comm is None else comm.Get_rank()

    # --- Step 1: run baseline primal ---
    if rank == 0:
        print("=" * 70)
        print("  FD-vs-Adjoint Check (Nemec Table 1 style)")
        print("=" * 70)
        print("\n--- Baseline primal ---")

    cl0, cd0, primal_drv = run_primal(primal_cfg, comm=comm)

    marker_indices = primal_drv.GetMarkerIndices()
    marker_id = marker_indices.get('AIRFOIL', -1)
    nodes = select_nodes(primal_drv, marker_id, comm)
    primal_drv.Finalize()

    base_alpha_deg = None
    with open(primal_cfg, 'r') as f:
        for line in f:
            line = line.split('%')[0].strip()
            if line.startswith('AOA=') or line.startswith('AOA ='):
                base_alpha_deg = float(line.split('=')[1].strip())
                break
    if base_alpha_deg is None:
        base_alpha_deg = 2.70

    # --- Step 2: run adjoint for both objectives ---
    if rank == 0:
        print("\n--- Adjoint solves ---")

    _ensure_primal_solution()

    drag_sens, drag_aoa, drag_mid, drv = run_adjoint_one(adjoint_cfg, 'DRAG', comm=comm)
    drv.Finalize()
    lift_sens, lift_aoa, lift_mid, drv = run_adjoint_one(adjoint_cfg, 'LIFT', comm=comm)
    drv.Finalize()

    # --- Step 3: FD on selected nodes (y-displacement, plateau selection) ---
    results = []
    if rank == 0:
        print("\n--- FD perturbations (surface nodes, y-direction) ---")
        print(f"Steps: {steps}")

    for owner_rank, local_iv, desc, x0, y0 in nodes:
        fd_cd_vals = []
        fd_cl_vals = []

        if rank == 0:
            print(f"\n  Node (rank {owner_rank}, local_iv {local_iv}) "
                  f"({desc}) at ({x0:.4f}, {y0:.6f})")

        for h in steps:
            cl_p, cd_p = fd_primal_perturbed(primal_cfg, owner_rank, local_iv, +h, comm)
            cl_m, cd_m = fd_primal_perturbed(primal_cfg, owner_rank, local_iv, -h, comm)
            fd_cd = (cd_p - cd_m) / (2 * h)
            fd_cl = (cl_p - cl_m) / (2 * h)

            fd_cd_vals.append(fd_cd)
            fd_cl_vals.append(fd_cl)

            if rank == 0:
                print(f"    h={h:.1e}: FD_dCD/dy={fd_cd:.6e}  FD_dCL/dy={fd_cl:.6e}")

        best_fd_cd, best_h_cd = _select_plateau(fd_cd_vals, steps)
        best_fd_cl, best_h_cl = _select_plateau(fd_cl_vals, steps)

        adj_cd = _get_adj_sensitivity(drag_sens, drag_mid, owner_rank, local_iv, 1, comm)
        adj_cl = _get_adj_sensitivity(lift_sens, lift_mid, owner_rank, local_iv, 1, comm)

        err_cd = abs(best_fd_cd - adj_cd) / abs(adj_cd) * 100 if abs(adj_cd) > 1e-30 else 0.0
        err_cl = abs(best_fd_cl - adj_cl) / abs(adj_cl) * 100 if abs(adj_cl) > 1e-30 else 0.0

        results.append({
            'name': desc, 'owner_rank': owner_rank, 'local_iv': local_iv,
            'x': x0, 'y': y0,
            'adj_cd': adj_cd, 'adj_cl': adj_cl,
            'fd_cd': best_fd_cd, 'fd_cl': best_fd_cl,
            'err_cd_pct': err_cd, 'err_cl_pct': err_cl,
            'best_h_cd': best_h_cd, 'best_h_cl': best_h_cl,
        })

    # --- Step 4: FD on AoA ---
    if rank == 0:
        print("\n--- FD perturbation (AoA, degrees) ---")

    aoa_steps = [1e-3, 1e-4, 1e-5]  # degrees
    fd_cd_aoa_vals = []
    fd_cl_aoa_vals = []

    for da in aoa_steps:
        cl_p, cd_p = fd_primal_aoa_perturbed(primal_cfg, base_alpha_deg, +da, comm)
        cl_m, cd_m = fd_primal_aoa_perturbed(primal_cfg, base_alpha_deg, -da, comm)
        fd_cd_a = (cd_p - cd_m) / (2 * da)
        fd_cl_a = (cl_p - cl_m) / (2 * da)

        fd_cd_aoa_vals.append(fd_cd_a)
        fd_cl_aoa_vals.append(fd_cl_a)

        if rank == 0:
            print(f"  da={da:.1e} deg: FD_dCD/da={fd_cd_a:.6e}  FD_dCL/da={fd_cl_a:.6e}")

    best_fd_cd_aoa, _ = _select_plateau(fd_cd_aoa_vals, aoa_steps)
    best_fd_cl_aoa, _ = _select_plateau(fd_cl_aoa_vals, aoa_steps)

    # SENS_AOA is a PARTIAL (flow-path only) derivative — add the explicit
    # wind-axis force-projection term:
    #   dCD/dα_total = SENS_AOA_drag + C_L·π/180   [per degree]
    #   dCL/dα_total = SENS_AOA_lift − C_D·π/180
    k = DEG
    adj_cd_aoa = None if drag_aoa is None else drag_aoa + cl0 * k
    adj_cl_aoa = None if lift_aoa is None else lift_aoa - cd0 * k

    if rank == 0 and drag_aoa is not None:
        print(f"  SENS_AOA raw:       dCD/da={drag_aoa:.6e}  dCL/da={lift_aoa:.6e}")
        print(f"  + projection term:  dCD/da={adj_cd_aoa:.6e}  dCL/da={adj_cl_aoa:.6e}")

    aoa_err_cd = None
    aoa_err_cl = None
    if adj_cd_aoa is not None and abs(adj_cd_aoa) > 1e-30:
        aoa_err_cd = abs(best_fd_cd_aoa - adj_cd_aoa) / abs(adj_cd_aoa) * 100
    if adj_cl_aoa is not None and abs(adj_cl_aoa) > 1e-30:
        aoa_err_cl = abs(best_fd_cl_aoa - adj_cl_aoa) / abs(adj_cl_aoa) * 100

    results.append({
        'name': 'AoA', 'owner_rank': -1, 'local_iv': -1,
        'x': None, 'y': None,
        'adj_cd': adj_cd_aoa, 'adj_cl': adj_cl_aoa,
        'fd_cd': best_fd_cd_aoa, 'fd_cl': best_fd_cl_aoa,
        'err_cd_pct': aoa_err_cd, 'err_cl_pct': aoa_err_cl,
        'best_h_cd': None, 'best_h_cl': None,
    })

    # --- Print summary table ---
    if rank == 0:
        print("\n" + "=" * 100)
        print("  SUMMARY: FD vs Adjoint (plateau-selected step per node)")
        print("=" * 100)
        header = f"{'Node':<16} {'x':>7}  {'y':>9}  " \
                 f"{'FD dCD/dy':>12}  {'Adj dCD/dy':>12}  {'%err':>6}  " \
                 f"{'FD dCL/dy':>12}  {'Adj dCL/dy':>12}  {'%err':>6}"
        print(header)
        print("-" * 100)
        for r in results:
            x_s = f"{r['x']:.4f}" if r['x'] is not None else "   N/A"
            y_s = f"{r['y']:.6f}" if r['y'] is not None else "     N/A"
            fd_cd_s = f"{r['fd_cd']:.4e}" if r['fd_cd'] is not None else "      N/A"
            adj_cd_s = f"{r['adj_cd']:.4e}" if r['adj_cd'] is not None else "      N/A"
            err_cd_s = f"{r['err_cd_pct']:.2f}" if r['err_cd_pct'] is not None else "  N/A"
            fd_cl_s = f"{r['fd_cl']:.4e}" if r['fd_cl'] is not None else "      N/A"
            adj_cl_s = f"{r['adj_cl']:.4e}" if r['adj_cl'] is not None else "      N/A"
            err_cl_s = f"{r['err_cl_pct']:.2f}" if r['err_cl_pct'] is not None else "  N/A"
            print(f"{r['name']:<16} {x_s:>7}  {y_s:>9}  "
                  f"{fd_cd_s:>12}  {adj_cd_s:>12}  {err_cd_s:>6}  "
                  f"{fd_cl_s:>12}  {adj_cl_s:>12}  {err_cl_s:>6}")
        print("=" * 100)

        gate_pass = True
        for r in results:
            for key in ('err_cd_pct', 'err_cl_pct'):
                if r[key] is not None and r[key] > 1.0:
                    gate_pass = False
                    print(f"  FAIL: {r['name']} {key} = {r[key]:.2f}% > 1%")
        if gate_pass:
            print("  GATE: PASS — all FD-vs-adjoint errors ≤ 1% (surface nodes + AoA)")
        else:
            print("  GATE: FAIL — some errors > 1%")

    return results


def main():
    parser = argparse.ArgumentParser(description="FD-vs-adjoint check (Nemec Table 1 style)")
    parser.add_argument("--primal-cfg", required=True, help="Primal config file")
    parser.add_argument("--adjoint-cfg", required=True, help="Adjoint config file template")
    parser.add_argument("--case-dir", required=True, help="Case root directory")
    parser.add_argument("--steps", default=None,
                        help="Comma-separated FD steps in chord units, e.g. 1e-6,1e-7 "
                             "(default: 1e-5,1e-6,1e-7)")
    args = parser.parse_args()

    steps = None
    if args.steps is not None:
        steps = [float(s) * CHORD for s in args.steps.split(',')]

    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
    except ImportError:
        comm = None

    run_fd_check(args.primal_cfg, args.adjoint_cfg, args.case_dir, steps=steps, comm=comm)


if __name__ == "__main__":
    main()
