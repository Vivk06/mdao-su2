#!/usr/bin/env python3
"""Standalone pysu2 primal solve on RAE 2822 — returns C_L, C_D.

Usage (serial):
    python run_primal.py --cfg /tmp/su2cfg/rae2822_primal.cfg --case-dir cases/rae2822_rans_sa_hh

Usage (MPI):
    mpirun -n 2 python run_primal.py --cfg /tmp/su2cfg/rae2822_primal.cfg --case-dir cases/rae2822_rans_sa_hh

The cfg is GENERATED from the TOML; no cfg template is tracked.  Either reuse
the cfg a finished run kept::

    cases/<case>/output/<run_id>/final/<point>/primal.cfg

or emit a standalone pair (per-point suffixes stripped, so the solver writes
`restart_flow.dat`)::

    python -c "from mdao_su2.run_config import load_config; \
from mdao_su2.solver.point_cfg_writer import write_unsuffixed_pair as w; \
w(load_config('cases/rae2822_rans_sa_hh/input/configs/run_sp.toml'), '/tmp/su2cfg')"

The script chdir's to --case-dir; a generated cfg carries an ABSOLUTE
MESH_FILENAME, so the mesh resolves regardless.  Restart files land in
--case-dir.
"""

import argparse
import math
import os
import sys

import numpy as np


def run_primal(cfg_path, comm=None, alpha_deg=None):
    """Run SU2 primal and return (C_L, C_D, driver).

    Parameters
    ----------
    cfg_path : str
        Path to the primal .cfg file (relative to CWD or absolute).
    comm : mpi4py.MPI.Comm or None
        MPI communicator.  None → serial (comm=0).
    alpha_deg : float or None
        If given, override the cfg AoA (degrees) via SetFarFieldAoA.

    Returns
    -------
    cl : float
    cd : float
    driver : pysu2.CSinglezoneDriver (caller must Finalize)
    """
    import pysu2

    if comm is None:
        comm_arg = 0
        rank = 0
    else:
        comm_arg = comm
        rank = comm.Get_rank()

    driver = pysu2.CSinglezoneDriver(cfg_path, 1, comm_arg)

    if alpha_deg is not None:
        driver.SetFarFieldAoA(alpha_deg)

    driver.StartSolver()

    cl = driver.GetOutputValue('LIFT')
    cd = driver.GetOutputValue('DRAG')

    if rank == 0:
        print(f"Primal converged:  C_L = {cl:.6f}   C_D = {cd:.6f}")

    return cl, cd, driver


def main():
    parser = argparse.ArgumentParser(description="Standalone SU2 primal solve")
    parser.add_argument("--cfg", required=True, help="Primal config file (relative to --case-dir)")
    parser.add_argument("--case-dir", required=True, help="Case root directory")
    parser.add_argument("--alpha", type=float, default=None, help="Override AoA (degrees)")
    args = parser.parse_args()

    os.chdir(args.case_dir)

    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
    except ImportError:
        comm = None

    cl, cd, driver = run_primal(args.cfg, comm=comm, alpha_deg=args.alpha)
    driver.Finalize()

    return cl, cd


if __name__ == "__main__":
    main()
