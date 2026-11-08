"""SU2Builder — MPhys 2.0.0 Builder wrapping SU2AeroComp.

Provides the mesh-coordinate
subsystem and the coupling subsystem MPhys needs to assemble a ScenarioAerodynamic,
wired to OM_DVGEOCOMP (pyGeo FFD) in MPhys's mesh -> geometry -> scenario pattern,
in mphys 2.0.0 MPhysVariables names.

Two impedance mismatches are bridged here (SU2AeroComp retains its interface):

  1. 2D SU2 <-> 3D pyGeo. SU2's RAE 2822 case is 2D (surface in the z=0 plane,
     x_aero = [x0,y0,x1,y1,...]); pyGeo is 3D. The mesh subsystem lifts the
     reference wall coords to 3D (z=0); the FFD deforms in 3D with the k=0/k=1
     span planes moving together (2D shape pattern); the coupling group's adapter
     drops z and re-interleaves back to SU2's 2D x_aero.

  2. distributed pyGeo <-> non-distributed SU2AeroComp. OM_DVGEOCOMP declares its
     surface I/O distributed=True; SU2AeroComp declares x_aero NON-distributed
     (full duplicated vector on every rank, sorted by mesh-global node index).
     We put the ENTIRE surface pointset on rank 0 (empty
     on other ranks), so pyGeo's global surface vector lives on rank 0 and the
     shape-DV Jacobian is not double-counted across ranks; the adapter then
     broadcasts rank 0's deformed surface to all ranks as the non-distributed
     x_aero (and sums reverse seeds back to rank 0). Validated serial + np2:
     check_totals of the geometry chain matches FD bit-identically on 1 and 2
     ranks.

Ordering parity: the mesh subsystem lifts get_reference_surface's x_ref (the SAME
gid-sorted layout SU2AeroComp uses), so pyGeo point i <-> x_aero node i. The
zero-DV geometry round trip is the deterministic guard.

Alpha stays DEGREES end-to-end (NO units='rad' — the
mphys aoa convention is radians; ours is not, so alpha carries no units).
Rank-local handling lives inside SU2AeroComp / the rank-0 partition.
The reference coords are read once before any pysu2ad run (get_reference_surface in
initialize()); the builder never touches MarkerInitialCoordinates itself.
"""

import numpy as np
import openmdao.api as om
from mphys import Builder, MPhysVariables

from .aero_comp import SU2AeroComp, get_reference_surface

_V = MPhysVariables.Aerodynamics.Surface
# adapter output feeding SU2AeroComp's (2D, non-distributed) x_aero — internal name
_X_AERO_2D = 'x_aero_2d'


class _SU2MeshComp(om.ExplicitComponent):
    """Emits the undeformed 3D surface coordinates (z=0) for the FFD to embed.

    x_aero0_mesh layout: get_reference_surface's gid-sorted nodes lifted to 3D,
    [x0,y0,0, x1,y1,0, ...]. Distributed with the whole vector on rank 0 (empty
    elsewhere) — the rank-0 partition the adapter and pyGeo rely on. Constant
    source (no inputs, no derivatives)."""

    def initialize(self):
        self.options.declare('builder', recordable=False)

    def setup(self):
        b = self.options['builder']
        n, ndim = b.n_nodes, b.ndim
        if self.comm.rank == 0:
            pts = np.zeros((n, 3))
            pts[:, :ndim] = b.x_ref.reshape(n, ndim)
            val = pts.reshape(-1)
        else:
            val = np.zeros(0)
        self.add_output(_V.Mesh.COORDINATES, val=val, distributed=True,
                        tags=['mphys_coordinates'])


class _Surface3Dto2D(om.ExplicitComponent):
    """pyGeo 3D deformed surface (distributed, rank-0) -> SU2 2D x_aero (non-dist).

    compute: broadcast rank 0's full 3D surface to every rank, drop z, interleave
    -> x_aero_2d (non-distributed, identical on all ranks — SU2AeroComp's contract).
    rev: the non-distributed seed is replicated on every rank; rank 0 alone routes
    it back to its (rank-0-only) distributed 3D slot (x,y components; z seed = 0).
    """

    def initialize(self):
        self.options.declare('n_nodes', types=int)

    def setup(self):
        n = self.options['n_nodes']
        # 3D surface from geometry: distributed, sized by the (rank-0) connection
        self.add_input(_V.COORDINATES, shape_by_conn=True, distributed=True,
                       tags=['mphys_coupling'])
        # 2D x_aero for SU2AeroComp: non-distributed, full on every rank
        self.add_output(_X_AERO_2D, shape=n * 2)

    def compute(self, inputs, outputs):
        n = self.options['n_nodes']
        if self.comm.rank == 0:
            full3 = inputs[_V.COORDINATES].reshape(n, 3).copy()
        else:
            full3 = np.empty((n, 3))
        self.comm.Bcast(full3, root=0)
        outputs[_X_AERO_2D] = full3[:, :2].reshape(-1)

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        assert mode == 'rev', '_Surface3Dto2D is reverse-mode only'
        if _X_AERO_2D in d_outputs and _V.COORDINATES in d_inputs:
            n = self.options['n_nodes']
            if self.comm.rank == 0:
                seed3 = np.zeros((n, 3))
                seed3[:, :2] = d_outputs[_X_AERO_2D].reshape(n, 2)
                d_inputs[_V.COORDINATES] += seed3.reshape(-1)


class _SU2CouplingGroup(om.Group):
    """Adapter (3D->2D) + SU2AeroComp. Scenario boundary (via mphys tags):
    x_aero (3D, mphys_coupling), alpha (mphys_input), C_L/C_D (mphys_result)."""

    def initialize(self):
        self.options.declare('builder', recordable=False)

    def setup(self):
        b = self.options['builder']
        # promote the 3D x_aero input up (mphys_coupling) so the scenario exposes it
        self.add_subsystem('adapter', _Surface3Dto2D(n_nodes=b.n_nodes),
                           promotes_inputs=[_V.COORDINATES])
        self.add_subsystem(
            'aero',
            SU2AeroComp(primal_cfg=b.primal_cfg, adjoint_cfg=b.adjoint_cfg,
                        wall_marker=b.wall_marker, moment_key=b.moment_key,
                        restart=b.restart),
            promotes_inputs=['alpha'],          # DEGREES; tag on SU2AeroComp
            promotes_outputs=['C_L', 'C_D', 'C_M'])  # mphys_result tags on SU2AeroComp
        self.connect(f'adapter.{_X_AERO_2D}', 'aero.x_aero')


class SU2Builder(Builder):
    """MPhys builder for the SU2 discrete-adjoint aerodynamic solver (2D RAE 2822).

    Options mirror SU2AeroComp's contract: primal_cfg, adjoint_cfg, wall_marker,
    restart. Set restart=False for cold-start FD verification."""

    def __init__(self, primal_cfg, adjoint_cfg, wall_marker='AIRFOIL', restart=True,
                 moment_key='MOMENT_Z'):
        self.primal_cfg = primal_cfg
        self.adjoint_cfg = adjoint_cfg
        self.wall_marker = wall_marker
        self.restart = restart
        self.moment_key = moment_key

    def initialize(self, comm):
        self.comm = comm
        # reference surface read ONCE, before any pysu2ad driver runs.
        # Same gid-sorted layout SU2AeroComp uses -> pyGeo/x_aero ordering parity.
        self.x_ref, self.gid_order, self.ndim = get_reference_surface(
            self.primal_cfg, comm, self.wall_marker)
        self.n_nodes = len(self.gid_order)
        self._gid_pos = {gid: i for i, gid in enumerate(self.gid_order)}
        # wall-segment connectivity (as x_ref position pairs) for the constraint
        # surface — read from the SAME pristine mesh, before any pysu2ad.
        self.wall_segments = self._read_wall_segments(comm)

    def get_mesh_coordinate_subsystem(self, scenario_name=None):
        return _SU2MeshComp(builder=self)

    def get_coupling_group_subsystem(self, scenario_name=None):
        return _SU2CouplingGroup(builder=self)

    def get_number_of_nodes(self):
        return self.n_nodes

    def get_ndof(self):
        return self.ndim

    # ---- geometry-chain helpers used by the Top's configure() (2D RAE 2822) ----

    def get_surface_points(self):
        """Rank-local undeformed 3D surface points (rank 0: full (n,3); else empty)
        — the pointset OM_DVGEOCOMP embeds (flattened), rank-0 partition."""
        if self.comm.rank == 0:
            pts = np.zeros((self.n_nodes, 3))
            pts[:, :self.ndim] = self.x_ref.reshape(self.n_nodes, self.ndim)
            return pts.reshape(-1)
        return np.zeros(0)

    def _read_wall_segments(self, comm):
        """Wall-marker line-element connectivity as x_ref position pairs.

        Reads GetMarkerElementNodes on a pristine primal driver (rank-local under
        MPI), maps element node -> mesh-global index -> x_ref position, then
        allgathers and dedups by gid pair so rank 0 (which builds the constraint
        surface) has the whole closed airfoil loop. The RAE 2822 wall is a single
        closed loop (192 segments, every node degree 2 — verified)."""
        import pysu2

        driver = pysu2.CSinglezoneDriver(self.primal_cfg, 1, comm)
        mid = driver.GetMarkerIndices().get(self.wall_marker, -1)
        nElem = driver.GetNumberMarkerElements(mid) if mid >= 0 else 0
        local = []
        for e in range(nElem):
            a, b = driver.GetMarkerElementNodes(mid, e)
            local.append((driver.GetNodeGlobalIndex(a), driver.GetNodeGlobalIndex(b)))
        driver.Finalize()

        seen, segs = set(), []
        for ga, gb in (s for rank_s in comm.allgather(local) for s in rank_s):
            key = (min(ga, gb), max(ga, gb))
            if key not in seen:
                seen.add(key)
                segs.append((self._gid_pos[ga], self._gid_pos[gb]))
        return segs

    def get_triangulated_surface(self, zspan=0.5):
        """Triangulated constraint surface for nom_setConstraintSurface.

        Each 2D wall segment (true mesh connectivity) is extruded to the two span
        planes z=-/+zspan and split into two triangles — watertight by
        construction. Returns [p0, p1, p2] (each (ntri,3)), pyGeo's DVConstraints
        format.

        REPLICATED on every rank (unlike the discipline coords, which are rank-0
        only): DVConstraints projects each thickness ray onto the *local* surface,
        so an empty surface on any rank makes that rank's projection fail and the
        run hangs under MPI. The full loop is available on all ranks (x_ref and
        wall_segments are allgathered in initialize), and the constraint outputs are
        non-distributed, so replicating gives identical values/derivatives with no
        double-count."""
        c = self.x_ref.reshape(self.n_nodes, self.ndim)
        zlo, zhi = -zspan, zspan
        p0, p1, p2 = [], [], []
        for pa, pb in self.wall_segments:
            A, B = c[pa], c[pb]
            Alo, Blo = [A[0], A[1], zlo], [B[0], B[1], zlo]
            Ahi, Bhi = [A[0], A[1], zhi], [B[0], B[1], zhi]
            # quad (Alo, Blo, Bhi, Ahi) -> two triangles
            p0.append(Alo); p1.append(Blo); p2.append(Bhi)
            p0.append(Alo); p1.append(Bhi); p2.append(Ahi)
        return [np.array(p0), np.array(p1), np.array(p2)]
