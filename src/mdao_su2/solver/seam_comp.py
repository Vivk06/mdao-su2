"""A design surface spanning several wall markers, in 3D (ONERA M6)."""

import csv
import math
import os

import numpy as np
import openmdao.api as om


from ..mpi import on_root
from .aero_comp import SU2AeroComp
from .builder import SU2Builder, _SU2MeshComp
from . import wall_seam as seam
from .cfg_file import (cfg_value as _cfg_value, rewrite_cfg,
                        rewrite_cfg as _rewrite_cfg)
from .variable_names import SURF as _V

try:
    from mpi4py import MPI
except ImportError:
    MPI = None


def get_reference_surface_multi(primal_cfg, comm, wall_markers):
    """``get_reference_surface`` over several markers, deduplicated by node."""
    import pysu2

    driver = pysu2.CSinglezoneDriver(primal_cfg, 1, comm)
    return seam.reference_surface(driver, comm, list(wall_markers), primal_cfg)


class MultiMarkerSU2AeroComp(SU2AeroComp):
    """``SU2AeroComp`` whose design surface spans a list of wall markers."""

    def initialize(self):
        super().initialize()
        self.options.declare(
            'wall_markers', types=(list, tuple), default=(),
            desc='wall marker names forming the design surface, in cfg order')
        self.options.declare('adjoint_stall_window', types=int, default=2000)
        self.options.declare('adjoint_stall_delta', types=float, default=0.1)

    def setup(self):
        """Shared SU2AeroComp setup with a multi-marker design surface."""
        if MPI is None or not isinstance(self.comm, MPI.Comm):
            raise RuntimeError(
                'MultiMarkerSU2AeroComp requires mpi4py: pysu2 drivers need a '
                'real MPI communicator.')
        markers = tuple(self.options['wall_markers'])
        if not markers:
            raise ValueError(
                'MultiMarkerSU2AeroComp needs wall_markers. A single-marker '
                'case must use SU2AeroComp -- that is the component '
                'every recorded RAE 2822 number was produced with.')

        primal_cfg = self.options['primal_cfg']

        aoa = _cfg_value(primal_cfg, 'AOA')
        self._alpha0 = float(aoa) if aoa is not None else 0.0
        minval = _cfg_value(primal_cfg, 'CONV_RESIDUAL_MINVAL')
        self._conv_minval = float(minval) if minval is not None else None
        field = _cfg_value(primal_cfg, 'CONV_FIELD')
        self._conv_field = (field.strip('() ').split(',')[0].strip()
                            if field else 'RMS_DENSITY')
        # The inherited _primal_failure reads these fields. The
        # framework never writes FIXED_CL_MODE, so they stay off.
        self._fixed_cl_mode = False
        self._target_cl = None
        sol = _cfg_value(primal_cfg, 'SOLUTION_FILENAME') or 'solution_flow'
        rst = _cfg_value(primal_cfg, 'RESTART_FILENAME') or 'restart_flow'
        self._solution_file = sol if sol.endswith('.dat') else sol + '.dat'
        self._restart_file = rst if rst.endswith('.dat') else rst + '.dat'

        self.x_aero0, self._gid_order, self._ndim = \
            get_reference_surface_multi(primal_cfg, self.comm, markers)
        self._gid_pos = {gid: i for i, gid in enumerate(self._gid_order)}

        self.add_input('x_aero', val=self.x_aero0.copy())
        self.add_input('alpha', val=self._alpha0, tags=['mphys_input'])
        self.add_output('C_L', val=0.0, tags=['mphys_result'])
        self.add_output('C_D', val=0.0, tags=['mphys_result'])
        self.add_output('C_M', val=0.0, tags=['mphys_result'])

        self._cl = None
        self._cd = None

    # ------------------------------------------------------------------ SU2

    def _local_markers(self, driver):
        """``[(marker_id, n_vertex), ...]`` for the wall markers this rank owns."""
        return seam.local_markers(driver, self.options['wall_markers'])

    def _apply_surface(self, driver, x_aero):
        """Push displacements onto every wall marker; returns the marker list."""
        markers = self._local_markers(driver)
        seam.apply_displacements(driver, markers, x_aero, self.x_aero0,
                                 self._gid_pos, self._ndim)
        return markers

    def _read_sensitivities(self, driver, markers):
        """``dJ/dx_aero`` gathered across markers -- ``=``, never ``+=``."""
        return seam.read_sensitivities(driver, markers, self._gid_pos, self._ndim,
                                       self.x_aero0.size)

    def _run_adjoint(self, objective, inputs):
        """The inherited ``_run_adjoint`` with the wall made multi-marker, via the seam."""
        import pysu2ad

        tmp_cfg = _rewrite_cfg(self.options['adjoint_cfg'],
                               {'OBJECTIVE_FUNCTION': objective})
        try:
            driver = pysu2ad.CDiscAdjSinglezoneDriver(tmp_cfg, 1, self.comm)
        finally:
            os.unlink(tmp_cfg)

        markers = self._apply_surface(driver, inputs['x_aero'])
        driver.SetFarFieldAoA(float(inputs['alpha'][0]))   # DEGREES
        driver.StartSolver()
        self._check_adjoint(driver, objective)

        djdx = self._read_sensitivities(driver, markers)
        sens_aoa = driver.GetOutputValue('SENS_AOA')
        driver.Finalize()

        seam.reduce(self.comm, djdx)
        djdalpha = seam.project_alpha(objective, sens_aoa, self._cl, self._cd,
                                      self.options['moment_key'])

        if math.isnan(djdalpha) or np.isnan(djdx).any():
            raise om.AnalysisError(
                f'SU2 adjoint [{objective}] produced NaN sensitivities')
        return djdx, djdalpha

    def _check_adjoint(self, driver, objective):
        """Reject an adjoint above ``CONV_RESIDUAL_MINVAL``; finalizes the driver first."""
        try:
            residual = float(driver.GetOutputValue('RMS_ADJ_DENSITY'))
        except Exception:
            residual = float('nan')
        floor = _cfg_value(self.options['adjoint_cfg'], 'CONV_RESIDUAL_MINVAL')
        floor = float(floor) if floor is not None else -6.0
        if math.isfinite(residual) and residual <= floor:
            return
        history = self._adjoint_residual_history()
        window = self.options['adjoint_stall_window']
        improvement, span = (final_window_improvement(history, window)
                             if history else (None, 0))
        driver.Finalize()
        # Report the final window: the first flat window is usually the
        # start-up transient of an adjoint that was still converging.
        detail = ''
        if improvement is not None:
            verdict = ('stalled' if improvement < self.options['adjoint_stall_delta']
                       else 'still converging')
            detail = (f'; the last {span} iterations improved '
                      f'{improvement:.2f} decades ({verdict}; stall threshold '
                      f'{self.options["adjoint_stall_delta"]:g} decades per '
                      f'{window} iterations)')
        raise om.AnalysisError(
            f'SU2 adjoint [{objective}] did not reach '
            f'CONV_RESIDUAL_MINVAL={floor:g} (RMS_ADJ_DENSITY={residual:g}){detail}')

    def _adjoint_residual_history(self):
        def read():
            base = _cfg_value(self.options['adjoint_cfg'], 'CONV_FILENAME')
            path = f"{(base or 'history_adj').strip()}.csv"
            if not os.path.exists(path):
                return None
            with open(path, newline='') as stream:
                reader = csv.reader(stream)
                header = next(reader)
                col = next((i for i, name in enumerate(header) if 'A_Rho' in name), None)
                if col is None:
                    return None
                rows = []
                for row in reader:
                    try:
                        rows.append(float(row[col].strip()))
                    except (IndexError, ValueError):
                        continue
                return rows or None
        return on_root(self.comm, read, what='read adjoint history')


#: Adapter output feeding the 3D component's non-distributed ``x_aero``.
#: Internal connection name, mirroring ``builder._X_AERO_2D``.
_X_AERO_3D = 'x_aero_3d'


class _SurfaceBroadcastComp(om.ExplicitComponent):
    """pyGeo 3D deformed surface (distributed, rank-0) -> 3D x_aero (non-dist)."""

    def initialize(self):
        self.options.declare('n_nodes', types=int)

    def setup(self):
        n = self.options['n_nodes']
        self.add_input(_V.COORDINATES, shape_by_conn=True, distributed=True,
                       tags=['mphys_coupling'])
        self.add_output(_X_AERO_3D, shape=n * 3)

    def compute(self, inputs, outputs):
        n = self.options['n_nodes']
        if self.comm.rank == 0:
            full = inputs[_V.COORDINATES].reshape(n, 3).copy()
        else:
            full = np.empty((n, 3))
        self.comm.Bcast(full, root=0)
        outputs[_X_AERO_3D] = full.reshape(-1)

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        assert mode == 'rev', '_SurfaceBroadcastComp is reverse-mode only'
        if _X_AERO_3D in d_outputs and _V.COORDINATES in d_inputs:
            # The non-distributed seed is replicated on every rank; rank 0 alone
            # routes it back to its (rank-0-only) distributed slot, or the sum
            if self.comm.rank == 0:
                d_inputs[_V.COORDINATES] += d_outputs[_X_AERO_3D]


class _MultiMarkerCouplingGroup(om.Group):
    """3D coupling group: broadcast adapter + the multi-marker component."""

    def initialize(self):
        self.options.declare('builder', recordable=False)

    def setup(self):
        b = self.options['builder']
        self.add_subsystem('adapter', _SurfaceBroadcastComp(n_nodes=b.n_nodes),
                           promotes_inputs=[_V.COORDINATES])
        self.add_subsystem(
            'aero',
            MultiMarkerSU2AeroComp(
                primal_cfg=b.primal_cfg, adjoint_cfg=b.adjoint_cfg,
                wall_markers=b.wall_markers, moment_key=b.moment_key,
                restart=b.restart),
            promotes_inputs=['alpha'],
            promotes_outputs=['C_L', 'C_D', 'C_M'])
        self.connect(f'adapter.{_X_AERO_3D}', 'aero.x_aero')


class MultiMarkerSU2Builder(SU2Builder):
    """``SU2Builder`` for a 3D, multi-marker design surface."""

    def __init__(self, primal_cfg, adjoint_cfg, wall_markers, restart=True,
                 moment_key='MOMENT_Z'):
        markers = tuple(wall_markers)
        if not markers:
            raise ValueError('MultiMarkerSU2Builder needs at least one wall marker')
        # The inherited scalar is kept coherent (some diagnostics read it) but
        # nothing in this class resolves a marker through it.
        super().__init__(primal_cfg, adjoint_cfg,
                         wall_marker=markers[0], restart=restart,
                         moment_key=moment_key)
        self.wall_markers = markers

    def initialize(self, comm):
        self.comm = comm
        self.x_ref, self.gid_order, self.ndim = \
            get_reference_surface_multi(self.primal_cfg, comm, self.wall_markers)
        self.n_nodes = len(self.gid_order)
        self._gid_pos = {gid: i for i, gid in enumerate(self.gid_order)}
        # The builder reads wall SEGMENTS here for the 2D constraint
        # surface. A 3D wall is faces, not a closed line loop, so that call is
        self.wall_faces = self._read_wall_faces(comm)
        # A 2D wall's faces are its line elements: the segments Hicks-Henne
        # and the extruded constraint surface need.
        self.wall_segments = list(self.wall_faces) if self.ndim == 2 else None

    def get_coupling_group_subsystem(self, scenario_name=None):
        return _MultiMarkerCouplingGroup(builder=self)

    def get_mesh_coordinate_subsystem(self, scenario_name=None):
        return _SU2MeshComp(builder=self)

    def get_surface_points(self):
        """Rank-local undeformed surface points (rank 0: full ``(n, 3)``)."""
        if self.comm.rank == 0:
            return self.x_ref.reshape(self.n_nodes, self.ndim).reshape(-1)
        return np.zeros(0)

    def _read_wall_faces(self, comm):
        """Wall surface elements as tuples of x_ref positions, deduplicated."""
        import pysu2

        driver = pysu2.CSinglezoneDriver(self.primal_cfg, 1, comm)
        indices = driver.GetMarkerIndices()
        local = []
        for name in self.wall_markers:
            mid = indices.get(name, -1)
            if mid < 0:
                continue
            for e in range(driver.GetNumberMarkerElements(mid)):
                nodes = driver.GetMarkerElementNodes(mid, e)
                local.append(tuple(driver.GetNodeGlobalIndex(n) for n in nodes))
        driver.Finalize()

        seen, faces = set(), []
        for gids in (f for rank_f in comm.allgather(local) for f in rank_f):
            key = frozenset(gids)
            if key in seen:
                continue
            seen.add(key)
            faces.append(tuple(self._gid_pos[g] for g in gids))
        return faces

    def get_triangulated_surface(self, zspan=None):
        """``(p0, p1, p2)`` triangle corner arrays for ``nom_setConstraintSurface``.

        A 2D wall is extruded to ``z = -/+zspan`` as ``SU2Builder`` does.
        """
        if self.ndim == 2:
            return SU2Builder.get_triangulated_surface(self, zspan=zspan)
        pts = self.x_ref.reshape(self.n_nodes, self.ndim)
        p0, p1, p2 = [], [], []
        for face in self.wall_faces:
            # Fan-triangulate: a quad becomes (0,1,2) and (0,2,3).
            for t in range(1, len(face) - 1):
                p0.append(pts[face[0]])
                p1.append(pts[face[t]])
                p2.append(pts[face[t + 1]])
        return [np.array(p0), np.array(p1), np.array(p2)]




class ResponseSU2AeroComp(MultiMarkerSU2AeroComp):
    """Multi-response component; inherited setup establishes the solver state."""

    def initialize(self):
        super().initialize()
        self.options.declare('responses', types=(list, tuple), default=())

    def setup(self):
        super().setup()
        records = tuple(self.options['responses'])
        self._response_records = records
        self._response_values = {}
        for record in records:
            if record.output not in ('C_L', 'C_D', 'C_M'):
                self.add_output(record.output, val=0.0, tags=['mphys_result'])

    def compute(self, inputs, outputs):
        import pysu2

        warm = self.options['restart'] and os.path.exists(self._solution_file)
        tmp_cfg = rewrite_cfg(self.options['primal_cfg'],
                              {'RESTART_SOL': 'YES' if warm else 'NO'})
        try:
            driver = pysu2.CSinglezoneDriver(tmp_cfg, 1, self.comm)
        finally:
            os.unlink(tmp_cfg)
        self._apply_surface(driver, inputs['x_aero'])
        driver.SetFarFieldAoA(float(inputs['alpha'][0]))
        driver.StartSolver()
        cl = driver.GetOutputValue('LIFT')
        cd = driver.GetOutputValue('DRAG')
        cm = driver.GetOutputValue(self.options['moment_key'])
        values = {record.output: driver.GetOutputValue(record.history_field)
                  for record in self._response_records
                  if record.output not in ('C_L', 'C_D', 'C_M')}
        failure = self._primal_failure(driver, cl, cd)
        if not math.isfinite(float(cm)):
            failure = f'non-finite coefficient (C_M={cm})'
        if any(not math.isfinite(float(value)) for value in values.values()):
            failure = 'non-finite requested response value'
        driver.Finalize()
        if failure:
            raise om.AnalysisError(f'SU2 primal failed: {failure}')
        self._sync_solution()
        self._cl, self._cd = cl, cd
        self._response_values = values
        outputs['C_L'], outputs['C_D'], outputs['C_M'] = cl, cd, cm
        for name, value in values.items():
            outputs[name] = value

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        super().compute_jacvec_product(inputs, d_inputs, d_outputs, mode)
        for record in self._response_records:
            if record.output in ('C_L', 'C_D', 'C_M') or record.output not in d_outputs:
                continue
            seed = float(d_outputs[record.output][0])
            if seed == 0.0:
                continue
            djdx, djdalpha = self._run_response_adjoint(record, inputs)
            if 'x_aero' in d_inputs:
                d_inputs['x_aero'] += seed * djdx
            if 'alpha' in d_inputs:
                d_inputs['alpha'] += seed * djdalpha

    def _run_response_adjoint(self, record, inputs):
        import pysu2ad
        tmp_cfg = rewrite_cfg(self.options['adjoint_cfg'],
                              {'OBJECTIVE_FUNCTION': record.su2_objective})
        try:
            driver = pysu2ad.CDiscAdjSinglezoneDriver(tmp_cfg, 1, self.comm)
        finally:
            os.unlink(tmp_cfg)
        markers = self._apply_surface(driver, inputs['x_aero'])
        driver.SetFarFieldAoA(float(inputs['alpha'][0]))
        driver.StartSolver()
        self._check_adjoint(driver, record.su2_objective)
        djdx = self._read_sensitivities(driver, markers)
        sens_aoa = driver.GetOutputValue('SENS_AOA')
        driver.Finalize()
        seam.reduce(self.comm, djdx)
        projection = {'wind_drag': 'DRAG', 'wind_lift': 'LIFT',
                      'wind_efficiency': 'EFFICIENCY'}.get(record.alpha_rule)
        if record.alpha_rule == 'moment':
            djdalpha = sens_aoa
        elif projection:
            djdalpha = seam.project_alpha(projection, sens_aoa, self._cl, self._cd,
                                          self.options['moment_key'])
        else:
            djdalpha = sens_aoa
        if not math.isfinite(float(djdalpha)) or not np.isfinite(djdx).all():
            raise om.AnalysisError(f'SU2 adjoint [{record.su2_objective}] produced non-finite sensitivities')
        return djdx, djdalpha


class _ResponseCouplingGroup(om.Group):
    def initialize(self):
        self.options.declare('builder', recordable=False)

    def setup(self):
        from .builder import _Surface3Dto2D, _X_AERO_2D
        builder = self.options['builder']
        if builder.ndim == 2:
            adapter, output = _Surface3Dto2D(n_nodes=builder.n_nodes), _X_AERO_2D
        else:
            adapter, output = _SurfaceBroadcastComp(n_nodes=builder.n_nodes), _X_AERO_3D
        self.add_subsystem('adapter', adapter)
        # Geometry output naming follows the adapter's MPhys surface input.
        from .variable_names import SURF
        self.promotes('adapter', inputs=[SURF.COORDINATES])
        self.add_subsystem('aero', ResponseSU2AeroComp(
            primal_cfg=builder.primal_cfg, adjoint_cfg=builder.adjoint_cfg,
            wall_markers=builder.wall_markers, moment_key=builder.moment_key,
            restart=builder.restart, responses=builder.responses,
            adjoint_stall_window=builder.adjoint_stall_window,
            adjoint_stall_delta=builder.adjoint_stall_delta),
            promotes_inputs=['alpha'], promotes_outputs=['C_L', 'C_D', 'C_M'] +
            [r.output for r in builder.responses
             if r.output not in ('C_L', 'C_D', 'C_M')])
        self.connect(f'adapter.{output}', f'aero.x_aero')


class ResponseSU2Builder(MultiMarkerSU2Builder):
    """Builder for requested catalog responses on fixed meshes with native warp."""

    def __init__(self, primal_cfg, adjoint_cfg, wall_markers, responses,
                 restart=True, moment_key='MOMENT_Z',
                 adjoint_stall_window=2000, adjoint_stall_delta=0.1):
        super().__init__(primal_cfg, adjoint_cfg, wall_markers, restart, moment_key)
        self.responses = tuple(responses)
        self.adjoint_stall_window = adjoint_stall_window
        self.adjoint_stall_delta = adjoint_stall_delta

    def get_surface_points(self):
        """Return rank-zero pyGeo points in its required three-coordinate layout."""
        if self.comm.rank != 0:
            return np.zeros(0)
        points = np.zeros((self.n_nodes, 3))
        points[:, :self.ndim] = self.x_ref.reshape(self.n_nodes, self.ndim)
        return points.reshape(-1)

    def get_coupling_group_subsystem(self, scenario_name=None):
        return _ResponseCouplingGroup(builder=self)


__all__ = ['ResponseSU2AeroComp', 'ResponseSU2Builder']


def final_window_improvement(residuals, window=2000):
    """Return the best-residual improvement and span of the final window."""
    n = len(residuals)
    if window <= 0 or n < 2:
        return None, 0
    span = min(window, n - 1)
    return min(residuals[:n - span]) - min(residuals), span
