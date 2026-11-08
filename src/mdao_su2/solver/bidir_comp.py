"""Forward- and reverse-mode SU2 aero path over the existing solver components.

SU2AeroComp has three scalar outputs, and one discrete-adjoint solve returns a
whole Jacobian row (dJ/dx_aero, dJ/dalpha, angle-of-attack force projection included). Caching
those rows at the linearization point answers both OpenMDAO directions through
the frozen ``_run_adjoint``:

    rev: d_inputs  += seed * row        bit-identical to SU2AeroComp
    fwd: d_outputs += row . d_inputs    at most one adjoint per functional,
                                        however many design variables seed it

``aero_comp.py`` and ``builder.py`` stay untouched. ``derivative_mode = "rev"``
never builds these classes, so the validated reverse path is unchanged.
"""

import hashlib
import os

import numpy as np
import openmdao.api as om

from .aero_comp import SU2AeroComp
from .builder import _V, _X_AERO_2D, SU2Builder, _Surface3Dto2D

__all__ = ['FUNCTIONALS', 'SU2AeroBidirComp', 'BidirSU2Builder']

#: Every functional SU2AeroComp can differentiate, in its frozen seed order.
FUNCTIONALS = ('C_D', 'C_L', 'C_M')


class BidirectionalAdjointMixin:
    """Adjoint-row cache giving an SU2AeroComp subclass both jacvec directions."""

    def initialize(self):
        super().initialize()
        self.options.declare(
            'functionals', types=tuple, default=FUNCTIONALS,
            desc='outputs whose rows forward mode computes; forward mode cannot '
                 'tell which outputs a response needs, so the builder names them')
        self._rows = {}
        self._lin_key = None

    @staticmethod
    def _linearization_key(inputs):
        digest = hashlib.sha1(np.ascontiguousarray(inputs['x_aero']).tobytes())
        digest.update(np.ascontiguousarray(inputs['alpha']).tobytes())
        return digest.hexdigest()

    def compute(self, inputs, outputs):
        # Cleared first: a failed primal must not leave the last point's rows.
        self._rows, self._lin_key = {}, None
        super().compute(inputs, outputs)
        self._lin_key = self._linearization_key(inputs)

    def _row(self, objective, inputs):
        if objective not in self._rows:
            self._rows[objective] = self._run_adjoint(objective, inputs)
        return self._rows[objective]

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        # The frozen preconditions, verbatim in meaning.
        if self._cl is None:
            raise RuntimeError('compute() must run before compute_jacvec_product '
                               '(the angle-of-attack force projection needs converged C_L/C_D)')
        if not os.path.exists(self._solution_file):
            raise FileNotFoundError(
                f'No {self._solution_file} — the adjoint reads the converged '
                f'primal solution')
        if self._linearization_key(inputs) != self._lin_key:
            raise RuntimeError(
                'compute_jacvec_product inputs differ from the last successful '
                'compute(); the cached adjoint rows and C_L/C_D would be stale')

        objectives = (('DRAG', 'C_D'), ('LIFT', 'C_L'),
                      (self.options['moment_key'], 'C_M'))
        for objective, name in objectives:
            if name not in d_outputs:
                continue
            if mode == 'rev':
                seed = float(d_outputs[name][0])
                if seed == 0.0:
                    continue
                djdx, djdalpha = self._row(objective, inputs)
                if 'x_aero' in d_inputs:
                    d_inputs['x_aero'] += seed * djdx
                if 'alpha' in d_inputs:
                    d_inputs['alpha'] += seed * djdalpha
                continue

            if name not in self.options['functionals']:
                continue
            dx = d_inputs['x_aero'] if 'x_aero' in d_inputs else None
            dalpha = float(d_inputs['alpha'][0]) if 'alpha' in d_inputs else 0.0
            # Seeds are non-distributed and replicated, so every rank takes the
            # same branch and the collective adjoint solve cannot hang.
            if dalpha == 0.0 and (dx is None or not np.any(dx)):
                continue
            djdx, djdalpha = self._row(objective, inputs)
            product = djdalpha * dalpha
            if dx is not None:
                product += float(np.dot(djdx, dx))
            d_outputs[name] += product


class SU2AeroBidirComp(BidirectionalAdjointMixin, SU2AeroComp):
    """SU2AeroComp with forward and reverse matrix-free products."""


class _Surface3Dto2DBidir(_Surface3Dto2D):
    """The frozen 3D->2D adapter plus its forward product."""

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        if mode == 'rev':
            return super().compute_jacvec_product(inputs, d_inputs, d_outputs, mode)
        if _X_AERO_2D in d_outputs and _V.COORDINATES in d_inputs:
            # Collective, like compute(): rank 0 owns the distributed surface.
            n = self.options['n_nodes']
            if self.comm.rank == 0:
                full3 = np.asarray(d_inputs[_V.COORDINATES]).reshape(n, 3).copy()
            else:
                full3 = np.empty((n, 3))
            self.comm.Bcast(full3, root=0)
            d_outputs[_X_AERO_2D] += full3[:, :2].reshape(-1)


class _BidirCouplingGroup(om.Group):
    """``_SU2CouplingGroup`` built from the bidirectional components."""

    def initialize(self):
        self.options.declare('builder', recordable=False)

    def setup(self):
        b = self.options['builder']
        self.add_subsystem('adapter', _Surface3Dto2DBidir(n_nodes=b.n_nodes),
                           promotes_inputs=[_V.COORDINATES])
        self.add_subsystem(
            'aero',
            SU2AeroBidirComp(primal_cfg=b.primal_cfg, adjoint_cfg=b.adjoint_cfg,
                             wall_marker=b.wall_marker, moment_key=b.moment_key,
                             restart=b.restart, functionals=b.functionals),
            promotes_inputs=['alpha'],
            promotes_outputs=['C_L', 'C_D', 'C_M'])
        self.connect(f'adapter.{_X_AERO_2D}', 'aero.x_aero')


class BidirSU2Builder(SU2Builder):
    """SU2Builder whose coupling group supports ``fwd`` and ``auto`` modes."""

    def __init__(self, primal_cfg, adjoint_cfg, functionals=FUNCTIONALS, **kwargs):
        super().__init__(primal_cfg, adjoint_cfg, **kwargs)
        unknown = set(functionals) - set(FUNCTIONALS)
        if unknown:
            raise ValueError(f'unknown functionals {sorted(unknown)}; '
                             f'allowed: {list(FUNCTIONALS)}')
        self.functionals = tuple(functionals)

    def get_coupling_group_subsystem(self, scenario_name=None):
        return _BidirCouplingGroup(builder=self)
