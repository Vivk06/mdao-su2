"""SU2AeroComp — matrix-free OpenMDAO ExplicitComponent wrapping pysu2/pysu2ad.

Surface and solver contract:
    inputs  : x_aero (flattened surface coordinates), alpha (DEGREES)
    outputs : C_L, C_D
    compute()                      = in-memory deform -> warm-started primal
    compute_jacvec_product('rev')  = one discrete-adjoint solve per seeded
                                     functional (dJ/dx_aero, dJ/dalpha)

Design decisions:
  * Single component: solver and functions are collapsed into one component,
    not the four-component table; the MPhys builder (builder.py) wraps it
    without a split.
  * Fresh SU2 driver per solve. Warm start is file-based (RESTART_SOL= YES +
    SOLUTION_FILENAME, SU2 binding behavior) — re-running StartSolver on a live
    driver is not a sanctioned operation of the pinned pysu2 interface. The
    per-call driver construction cost is accepted.
  * x_aero layout: wall-marker nodes sorted by mesh-global node index
    (GetNodeGlobalIndex), coordinates interleaved [x0, y0, x1, y1, ...].
    This ordering is identical on every rank and independent of the MPI
    partitioning, so serial and mpirun runs share one x_aero vector.
    x_aero, C_L, C_D are declared NON-distributed: every rank holds the full duplicated vector and the
    component allreduces so all ranks return identical values/derivatives.
  * Physics-agnostic: turbulence/transition model, schemes, and flow regime
    are owned entirely by the cfg pair; this class never inspects them.

Working directory: SU2 resolves MESH_FILENAME etc. relative to the CWD and
writes restart/solution files there — run with the case directory as the CWD
(same convention as standalone/*).

Solver invariants: degree-based alpha (SetFarFieldAoA takes degrees), configuration-owned objectives (no
runtime objective setter — cfg rewrite), rank-local ownership (rank-local marker/vertex ids,
halo filtering), angle-of-attack force (SENS_AOA projection correction), zero-level adjoint multigrid (adjoint cfg is
the MGLEVEL=0 working copy — never re-copy the pipeline one).
"""

import math
import os
import shutil

import numpy as np
import openmdao.api as om

# The one cfg module owns these helpers. The private aliases preserve the
# component call sites.
from .cfg_file import cfg_value as _cfg_value, rewrite_cfg as _rewrite_cfg
from .wall_seam import DEG

try:
    from mpi4py import MPI
except ImportError:
    MPI = None


def get_reference_surface(primal_cfg, comm, wall_marker='AIRFOIL'):
    """Construct a temporary primal driver (no solve) and return the global
    reference surface layout.

    MUST run before any pysu2ad driver has run in this process: afterwards
    MarkerInitialCoordinates returns corrupted memory (the surface must be read before a driver runs).
    OpenMDAO guarantees this for SU2AeroComp — all setup() calls precede any
    solve. These pristine coordinates are the displacement reference for the
    whole component lifetime.

    Returns
    -------
    x_ref : np.ndarray, shape (ndim * n_global,)
        Undeformed wall-marker coordinates, nodes sorted by mesh-global node
        index, interleaved [x0, y0, x1, y1, ...] — the x_aero layout.
        Identical on all ranks and independent of the partitioning.
    gid_order : list[int]
        Mesh-global node index of each x_aero node (same order as x_ref).
    ndim : int
    """
    import pysu2

    driver = pysu2.CSinglezoneDriver(primal_cfg, 1, comm)
    ndim = driver.GetNumberDimensions()
    marker_id = driver.GetMarkerIndices().get(wall_marker, -1)
    nVertex = driver.GetNumberMarkerNodes(marker_id) if marker_id >= 0 else 0

    local = []
    if nVertex > 0:
        ini = driver.MarkerInitialCoordinates(marker_id)
        for iv in range(nVertex):
            iPoint = driver.GetMarkerNode(marker_id, iv)
            if not driver.GetNodeDomain(iPoint):
                continue  # halo: reported by its owning rank
            local.append((driver.GetNodeGlobalIndex(iPoint), tuple(ini.Get(iv))))
    driver.Finalize()

    all_nodes = sorted(n for rank_nodes in comm.allgather(local) for n in rank_nodes)
    gid_order = [gid for gid, _ in all_nodes]
    x_ref = np.array([c for _, coords in all_nodes for c in coords])
    return x_ref, gid_order, ndim


class SU2AeroComp(om.ExplicitComponent):
    """x_aero + alpha (deg) -> C_L, C_D with reverse-mode adjoint derivatives."""

    def initialize(self):
        self.options.declare('primal_cfg', types=str,
                             desc='primal cfg (MATH_PROBLEM= DIRECT)')
        self.options.declare('adjoint_cfg', types=str,
                             desc='adjoint cfg template (MATH_PROBLEM= DISCRETE_ADJOINT, MGLEVEL= 0)')
        self.options.declare('wall_marker', types=str, default='AIRFOIL')
        self.options.declare('moment_key', types=str, default='MOMENT_Z')
        self.options.declare('restart', types=bool, default=True,
                             desc='warm-start the primal from the previous converged solution')

    def setup(self):
        if MPI is None or not isinstance(self.comm, MPI.Comm):
            raise RuntimeError(
                'SU2AeroComp requires mpi4py: pysu2 drivers need a real MPI '
                'communicator (this build rejects the serial comm=0 fallback).')

        primal_cfg = self.options['primal_cfg']

        # cfg-derived bookkeeping — physics and convergence stay cfg-owned
        aoa = _cfg_value(primal_cfg, 'AOA')
        self._alpha0 = float(aoa) if aoa is not None else 0.0
        minval = _cfg_value(primal_cfg, 'CONV_RESIDUAL_MINVAL')
        self._conv_minval = float(minval) if minval is not None else None
        field = _cfg_value(primal_cfg, 'CONV_FIELD')
        self._conv_field = (field.strip('() ').split(',')[0].strip()
                            if field else 'RMS_DENSITY')
        fcl = _cfg_value(primal_cfg, 'FIXED_CL_MODE')
        self._fixed_cl_mode = (fcl is not None and fcl.strip().upper() == 'YES')
        if self._fixed_cl_mode:
            tcl = _cfg_value(primal_cfg, 'TARGET_CL')
            self._target_cl = float(tcl) if tcl is not None else None
        else:
            self._target_cl = None
        sol = _cfg_value(primal_cfg, 'SOLUTION_FILENAME') or 'solution_flow'
        rst = _cfg_value(primal_cfg, 'RESTART_FILENAME') or 'restart_flow'
        self._solution_file = sol if sol.endswith('.dat') else sol + '.dat'
        self._restart_file = rst if rst.endswith('.dat') else rst + '.dat'

        self.x_aero0, self._gid_order, self._ndim = get_reference_surface(
            primal_cfg, self.comm, self.options['wall_marker'])
        self._gid_pos = {gid: i for i, gid in enumerate(self._gid_order)}

        # x_aero is fed by the geometry chain's 2D adapter (builder.py) — NOT an
        # mphys scenario-boundary variable, so it carries no mphys tag.
        self.add_input('x_aero', val=self.x_aero0.copy())
        # alpha stays DEGREES; NO units='rad' — the mphys aoa convention is
        # radians, ours is not, so the DV routes through unitless. mphys_input lets
        # ScenarioAerodynamic promote it to the scenario boundary.
        self.add_input('alpha', val=self._alpha0, tags=['mphys_input'])
        self.add_output('C_L', val=0.0, tags=['mphys_result'])
        self.add_output('C_D', val=0.0, tags=['mphys_result'])
        self.add_output('C_M', val=0.0, tags=['mphys_result'])

        # converged primal coefficients — required by the SENS_AOA projection
        # correction; populated by compute()
        self._cl = None
        self._cd = None

    # ------------------------------------------------------------------ SU2

    def _apply_surface(self, driver, x_aero):
        """Push displacements (x_aero − pristine reference coords) onto the wall.

        Sets every rank-local vertex (halos included) so the volume deformation
        sees a consistent surface; the gid->position map routes each local
        vertex to its slot in the globally-ordered x_aero vector.

        The reference is the setup-captured x_aero0, NEVER this driver's
        MarkerInitialCoordinates: that accessor (and MarkerCoordinates) returns
        corrupted stale memory on any driver constructed after a pysu2ad driver
        has run in the process (the surface is read before a driver runs).
        SU2's internal initial coordinates are unaffected (bit-exact post-AD
        zero-displacement solve), so displacement relative to x_aero0 lands the
        surface exactly at x_aero.

        Returns (marker_id, nVertex), rank-local.
        """
        marker_id = driver.GetMarkerIndices().get(self.options['wall_marker'], -1)
        nVertex = driver.GetNumberMarkerNodes(marker_id) if marker_id >= 0 else 0
        ndim = self._ndim
        for iv in range(nVertex):
            pos = self._gid_pos[driver.GetNodeGlobalIndex(driver.GetMarkerNode(marker_id, iv))]
            disp = tuple(float(x_aero[ndim * pos + d] - self.x_aero0[ndim * pos + d])
                         for d in range(ndim))
            driver.SetMarkerCustomDisplacement(marker_id, iv, disp)
        return marker_id, nVertex

    def _sync_solution(self):
        """restart_flow.dat -> solution_flow.dat so warm starts and
        adjoint solves read the latest converged primal."""
        if self.comm.rank == 0 and os.path.exists(self._restart_file):
            shutil.copy2(self._restart_file, self._solution_file)
        self.comm.Barrier()

    def _primal_failure(self, driver, cl, cd):
        """Return a failure description, or None if the solve is trustworthy.

        NaN coefficients or a residual short of CONV_RESIDUAL_MINVAL mean the
        line search must back off — never hand the optimizer garbage. The
        residual check uses the cfg's CONV_FIELD (default RMS_DENSITY) and is
        skipped when the cfg declares no residual criterion.

        FIXED_CL_MODE: the post-solve residual is meaningless because the
        FD step (ITER_DCL_DALPHA) perturbs AoA and resets the residual.
        Gate on CL convergence to TARGET_CL instead.
        """
        if math.isnan(cl) or math.isnan(cd):
            return f'NaN coefficients (C_L={cl}, C_D={cd})'
        if self._fixed_cl_mode:
            if self._target_cl is not None:
                delta = abs(cl - self._target_cl)
                if delta > 1e-3:
                    return (f'FIXED_CL_MODE: CL={cl:.6f} did not converge to '
                            f'TARGET_CL={self._target_cl:.6f} (delta={delta:.6f})')
            return None
        if self._conv_minval is not None:
            try:
                res = float(driver.GetOutputValue(self._conv_field))
            except Exception:
                return None  # cfg converges on a field the wrapper does not expose
            if math.isnan(res) or res > self._conv_minval:
                return (f'{self._conv_field}= {res:.3f} did not reach '
                        f'CONV_RESIDUAL_MINVAL= {self._conv_minval}')
        return None

    def _run_adjoint(self, objective, inputs):
        """One discrete-adjoint solve; returns TOTAL (dJ/dx_aero, dJ/dalpha).

        The objective is set by cfg rewrite — there is no runtime setter
        (the objective comes from the cfg). Results are allreduced: identical on every rank.
        """
        import pysu2ad

        tmp_cfg = _rewrite_cfg(self.options['adjoint_cfg'],
                               {'OBJECTIVE_FUNCTION': objective})
        try:
            driver = pysu2ad.CDiscAdjSinglezoneDriver(tmp_cfg, 1, self.comm)
        finally:
            os.unlink(tmp_cfg)

        # linearize at the same deformed shape and AoA as the primal
        marker_id, nVertex = self._apply_surface(driver, inputs['x_aero'])
        driver.SetFarFieldAoA(float(inputs['alpha'][0]))  # DEGREES
        driver.StartSolver()

        djdx = np.zeros_like(self.x_aero0)
        ndim = self._ndim
        for iv in range(nVertex):
            iPoint = driver.GetMarkerNode(marker_id, iv)
            if not driver.GetNodeDomain(iPoint):
                continue  # halo — its owner accumulates it
            pos = self._gid_pos[driver.GetNodeGlobalIndex(iPoint)]
            sens = driver.GetMarkerDisplacementSensitivity(marker_id, iv)
            for d in range(ndim):
                djdx[ndim * pos + d] = sens[d]
        sens_aoa = driver.GetOutputValue('SENS_AOA')
        driver.Finalize()

        self.comm.Allreduce(MPI.IN_PLACE, djdx, op=MPI.SUM)

        # SENS_AOA is the flow-path PARTIAL only — add the wind-axis
        # force-projection term for the total (angle-of-attack correction). Raw SENS_AOA
        # for DRAG carries the wrong sign.
        k = DEG
        # Moments are integrated in body axes, so no wind-axis projection term
        # is applied. Central alpha FD on C_M checks this branch.
        if objective == self.options['moment_key']:
            djdalpha = sens_aoa
        elif objective == 'DRAG':
            djdalpha = sens_aoa + self._cl * k
        else:  # LIFT
            djdalpha = sens_aoa - self._cd * k

        if math.isnan(djdalpha) or np.isnan(djdx).any():
            raise om.AnalysisError(f'SU2 adjoint [{objective}] produced NaN sensitivities')
        return djdx, djdalpha

    # -------------------------------------------------------------- OpenMDAO

    def compute(self, inputs, outputs):
        import pysu2

        warm = self.options['restart'] and os.path.exists(self._solution_file)
        tmp_cfg = _rewrite_cfg(self.options['primal_cfg'],
                               {'RESTART_SOL': 'YES' if warm else 'NO'})
        try:
            driver = pysu2.CSinglezoneDriver(tmp_cfg, 1, self.comm)
        finally:
            os.unlink(tmp_cfg)

        self._apply_surface(driver, inputs['x_aero'])
        driver.SetFarFieldAoA(float(inputs['alpha'][0]))  # DEGREES
        driver.StartSolver()

        cl = driver.GetOutputValue('LIFT')
        cd = driver.GetOutputValue('DRAG')
        cm = driver.GetOutputValue(self.options['moment_key'])
        failure = self._primal_failure(driver, cl, cd)
        # The moment failure path retains _primal_failure's established
        # C_L/C_D call untouched, so C_M has an adjacent additive guard.
        if math.isnan(cm):
            failure = f'NaN coefficient (C_M={cm})'
        driver.Finalize()
        if failure:
            raise om.AnalysisError(f'SU2 primal failed: {failure}')

        self._sync_solution()
        self._cl, self._cd = cl, cd
        outputs['C_L'] = cl
        outputs['C_D'] = cd
        outputs['C_M'] = cm

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        assert mode == 'rev', \
            'SU2AeroComp is reverse-mode only (one adjoint solve per functional)'
        if self._cl is None:
            raise RuntimeError('compute() must run before compute_jacvec_product '
                               '(the angle-of-attack force projection needs converged C_L/C_D)')
        if not os.path.exists(self._solution_file):
            raise FileNotFoundError(
                f'No {self._solution_file} — the adjoint reads the converged '
                f'primal solution')

        for objective, name in (('DRAG', 'C_D'), ('LIFT', 'C_L')):
            if name not in d_outputs:
                continue
            seed = float(d_outputs[name][0])
            if seed == 0.0:
                continue
            djdx, djdalpha = self._run_adjoint(objective, inputs)
            if 'x_aero' in d_inputs:
                d_inputs['x_aero'] += seed * djdx
            if 'alpha' in d_inputs:
                d_inputs['alpha'] += seed * djdalpha

        # Kept as an additive loop so the C_L/C_D objective tuple and
        # its established reverse path remain unchanged.
        for objective, name in ((self.options['moment_key'], 'C_M'),):
            if name not in d_outputs:
                continue
            seed = float(d_outputs[name][0])
            if seed == 0.0:
                continue
            djdx, djdalpha = self._run_adjoint(objective, inputs)
            if 'x_aero' in d_inputs:
                d_inputs['x_aero'] += seed * djdx
            if 'alpha' in d_inputs:
                d_inputs['alpha'] += seed * djdalpha
