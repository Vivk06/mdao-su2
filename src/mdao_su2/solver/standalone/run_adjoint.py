#!/usr/bin/env python3
"""Standalone pysu2ad discrete-adjoint solve — one adjoint per functional.

Runs adjoint for DRAG and LIFT on the RAE 2822 case, extracts per-node
surface displacement sensitivities and AoA sensitivity.

Prerequisites:
    1. Primal must have converged (run_primal.py).
    2. restart_flow.dat must be copied/renamed to solution_flow.dat in --case-dir.
       (This script does the copy automatically if restart_flow.dat exists.)

Usage (serial):
    python run_adjoint.py --cfg /tmp/su2cfg/rae2822_adjoint.cfg --case-dir cases/rae2822_rans_sa_hh

Usage (MPI):
    mpirun -n 2 python run_adjoint.py --cfg /tmp/su2cfg/rae2822_adjoint.cfg --case-dir cases/rae2822_rans_sa_hh

The cfg is GENERATED from the TOML; no cfg template is tracked.  Either reuse
the cfg a finished run kept::

    cases/<case>/output/<run_id>/final/<point>/adjoint_<functional>.cfg

or emit a standalone pair (per-point suffixes stripped, so the solver writes
`restart_flow.dat`)::

    python -c "from mdao_su2.run_config import load_config; \
from mdao_su2.solver.point_cfg_writer import write_unsuffixed_pair as w; \
w(load_config('cases/rae2822_rans_sa_hh/input/configs/run_sp.toml'), '/tmp/su2cfg')"
"""

import argparse
import math
import os
import re
import shutil
import sys
import tempfile

import numpy as np


def _prepare_adjoint_cfg(base_cfg_path, objective):
    """Return path to a temporary cfg with OBJECTIVE_FUNCTION set to `objective`."""
    with open(base_cfg_path, 'r') as f:
        content = f.read()
    content = re.sub(
        r'^OBJECTIVE_FUNCTION\s*=\s*\S+',
        f'OBJECTIVE_FUNCTION= {objective}',
        content,
        flags=re.MULTILINE,
    )
    tmp = tempfile.NamedTemporaryFile(
        mode='w', suffix='.cfg', dir=os.path.dirname(base_cfg_path) or '.',
        delete=False, prefix=f'adj_{objective.lower()}_',
    )
    tmp.write(content)
    tmp.close()
    return tmp.name


def _ensure_primal_solution():
    """Copy restart_flow.dat → solution_flow.dat if needed."""
    sol = 'solution_flow.dat'
    rst = 'restart_flow.dat'
    if not os.path.exists(sol) and os.path.exists(rst):
        shutil.copy2(rst, sol)
        print(f"Copied {rst} → {sol}")
    if not os.path.exists(sol):
        raise FileNotFoundError(
            f"No {sol} found.  Run the primal first and ensure "
            f"{rst} is copied to {sol}."
        )


def run_adjoint_one(cfg_path, objective, comm=None, alpha_deg=None):
    """Run one discrete-adjoint solve and return sensitivities.

    Parameters
    ----------
    cfg_path : str
        Path to the adjoint .cfg template.
    objective : str
        'DRAG' or 'LIFT'.
    comm : mpi4py.MPI.Comm or None
    alpha_deg : float or None
        If given, override AoA (degrees) via SetFarFieldAoA.

    Returns
    -------
    sens_disp : np.ndarray, shape (nVertex, 2)
        dJ/d(displacement_x), dJ/d(displacement_y) per marker vertex.
    sens_aoa : float or None
        dJ/d(AoA) from GetOutputValue('SENS_AOA'), or None if unavailable.
    marker_id : int
    driver : pysu2ad.CDiscAdjSinglezoneDriver (caller must Finalize)
    """
    import pysu2ad

    if comm is None:
        comm_arg = 0
        rank = 0
    else:
        comm_arg = comm
        rank = comm.Get_rank()

    tmp_cfg = _prepare_adjoint_cfg(cfg_path, objective)
    try:
        driver = pysu2ad.CDiscAdjSinglezoneDriver(tmp_cfg, 1, comm_arg)
    finally:
        os.unlink(tmp_cfg)

    if alpha_deg is not None:
        driver.SetFarFieldAoA(alpha_deg)

    marker_indices = driver.GetMarkerIndices()
    wall_marker = 'AIRFOIL'
    marker_id = marker_indices.get(wall_marker, -1)
    nVertex = driver.GetNumberMarkerNodes(marker_id) if marker_id >= 0 else 0

    driver.StartSolver()

    sens_disp = np.zeros((nVertex, 2))
    for iv in range(nVertex):
        sx, sy = driver.GetMarkerDisplacementSensitivity(marker_id, iv)
        sens_disp[iv, 0] = sx
        sens_disp[iv, 1] = sy

    sens_aoa = None
    try:
        output_names = driver.GetOutputNames()
        if 'SENS_AOA' in output_names:
            sens_aoa = driver.GetOutputValue('SENS_AOA')
    except Exception:
        pass

    if rank == 0:
        print(f"\nAdjoint [{objective}] complete:")
        print(f"  Surface sens shape: {sens_disp.shape}")
        print(f"  |sens_disp| max:    {np.abs(sens_disp).max():.6e}")
        if sens_aoa is not None:
            print(f"  SENS_AOA:           {sens_aoa:.6e}")
        else:
            print(f"  SENS_AOA:           not available")

    return sens_disp, sens_aoa, marker_id, driver


def run_adjoint(cfg_path, comm=None, alpha_deg=None):
    """Run adjoint for both DRAG and LIFT, return all sensitivities.

    Returns
    -------
    result : dict with keys:
        'drag_sens_disp', 'drag_sens_aoa',
        'lift_sens_disp', 'lift_sens_aoa',
        'marker_id'
    """
    _ensure_primal_solution()

    drag_sens, drag_aoa, mid, drv = run_adjoint_one(
        cfg_path, 'DRAG', comm=comm, alpha_deg=alpha_deg)
    drv.Finalize()

    lift_sens, lift_aoa, _, drv = run_adjoint_one(
        cfg_path, 'LIFT', comm=comm, alpha_deg=alpha_deg)
    drv.Finalize()

    if comm is None or comm.Get_rank() == 0:
        diff = np.abs(drag_sens - lift_sens).max()
        print(f"\nObjective-switch check: max |DRAG_sens - LIFT_sens| = {diff:.6e}")
        assert diff > 1e-12, (
            "DRAG and LIFT adjoint sensitivities are identical — "
            "objective selection may not be working"
        )

    return {
        'drag_sens_disp': drag_sens,
        'drag_sens_aoa': drag_aoa,
        'lift_sens_disp': lift_sens,
        'lift_sens_aoa': lift_aoa,
        'marker_id': mid,
    }


def main():
    parser = argparse.ArgumentParser(description="Standalone SU2 discrete-adjoint solve")
    parser.add_argument("--cfg", required=True, help="Adjoint config file template")
    parser.add_argument("--case-dir", required=True, help="Case root directory")
    parser.add_argument("--alpha", type=float, default=None, help="Override AoA (degrees)")
    args = parser.parse_args()

    os.chdir(args.case_dir)

    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
    except ImportError:
        comm = None

    result = run_adjoint(args.cfg, comm=comm, alpha_deg=args.alpha)

    if comm is None or comm.Get_rank() == 0:
        print("\n=== Adjoint Summary ===")
        for obj in ('drag', 'lift'):
            sens = result[f'{obj}_sens_disp']
            aoa = result[f'{obj}_sens_aoa']
            print(f"  {obj.upper()}: max|sens_x|={np.abs(sens[:,0]).max():.4e}  "
                  f"max|sens_y|={np.abs(sens[:,1]).max():.4e}  "
                  f"SENS_AOA={'N/A' if aoa is None else f'{aoa:.4e}'}")


if __name__ == "__main__":
    main()
