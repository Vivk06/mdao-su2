"""OpenMDAO/MPhys pieces shared by the config-driven model."""

import glob
import os
import shutil

import openmdao.api as om
from mphys import Builder
from mphys.scenarios.aerodynamic import ScenarioAerodynamic

from ..geometry.factory import build_geometry_component
from ..mpi import on_root
from ..solver.cfg_file import cfg_value, edit_cfg
from ..solver.variable_names import SURF as _V

__all__ = ['sync_adjoint_restart', 'AdjointWarmStartComp', 'ObjectiveComp',
           'GeoBuilder', 'GeoScenarioAero']


# ------------------------------------------------- adjoint warm start (5c)

def sync_adjoint_restart(adj_cfg):
    """Copy adjoint restart files to solution files and enable RESTART_SOL."""
    rst_base = cfg_value(adj_cfg, 'RESTART_ADJ_FILENAME') or 'restart_adj'
    sol_base = cfg_value(adj_cfg, 'SOLUTION_ADJ_FILENAME') or 'solution_adj'

    any_copied = False
    for src in glob.glob(f'{rst_base}_*.dat'):
        suffix = src[len(rst_base):]
        dst = f'{sol_base}{suffix}'
        shutil.copy2(src, dst)
        any_copied = True

    if any_copied:
        # Legitimate in-place edit: adj_cfg is a GENERATED per-point cfg, not a
        # tracked template (see cfg.edit_cfg).
        edit_cfg(adj_cfg, {'RESTART_SOL': 'YES'})


class ObjectiveComp(om.ExecComp):
    """``obj_comb``: the compiled objective, with its ``log`` arguments checked.

    A non-positive ``log`` argument fails the evaluation with an
    ``AnalysisError`` that names the response and point, instead of passing
    a NaN when the objective is evaluated.
    """

    def __init__(self, compiled):
        super().__init__(compiled.execcomp_src,
                         **{n: {'val': 0.0} for n in compiled.inputs})
        self._compiled = compiled

    def compute(self, inputs, outputs):
        if self._compiled.domain:
            errors = self._compiled.domain_errors(
                {name: inputs[name][0] for name in self._compiled.inputs})
            if errors:
                raise om.AnalysisError(
                    f'objective {self._compiled.text!r}: ' + '; '.join(errors))
        super().compute(inputs, outputs)


class AdjointWarmStartComp(om.ExplicitComponent):
    """Identity pass-through managing adjoint restart files between design iterations."""

    def initialize(self):
        self.options.declare('point_cfgs', types=list,
                             desc='list of (primal_cfg, adjoint_cfg) per point')
        self.options.declare('enabled', types=bool, default=True)

    def setup(self):
        self.add_input('obj_in', val=0.0)
        self.add_output('obj_out', val=0.0)
        self.declare_partials('obj_out', 'obj_in', val=1.0)

    def compute(self, inputs, outputs):
        outputs['obj_out'] = inputs['obj_in']
        if not self.options['enabled']:
            return
        on_root(self.comm,
                lambda: [sync_adjoint_restart(a) for _, a in self.options['point_cfgs']],
                what='sync adjoint restart files')


# ---------------------------------------------- parallel geometry (partition)

class GeoBuilder(Builder):
    """Geometry Builder for the parallel-points path."""

    def __init__(self, cfg):
        self._cfg = cfg

    def initialize(self, comm):
        self.comm = comm

    def get_mesh_coordinate_subsystem(self, scenario_name=None):
        return build_geometry_component(self._cfg)


class GeoScenarioAero(ScenarioAerodynamic):
    """ScenarioAerodynamic with per-scenario geometry for the SU2 coupling (no x_aero0 sink)."""

    def initialize(self):
        super().initialize()
        self.options.declare('shape_dv_hook', default=None, recordable=False,
                             desc='callable(geometry, aero_builder) -> n_shape')

    def _mphys_scenario_setup(self):
        aero_builder = self.options['aero_builder']
        geometry_builder = self.options['geometry_builder']
        aero_builder.initialize(self.comm)
        geometry_builder.initialize(self.comm)
        self.add_subsystem(
            'mesh', aero_builder.get_mesh_coordinate_subsystem(self.name))
        self.mphys_add_subsystem(
            'geometry', geometry_builder.get_mesh_coordinate_subsystem(self.name))
        self.connect(f'mesh.{_V.Mesh.COORDINATES}', _V.Geometry.COORDINATES_INPUT)
        self.connect(_V.Geometry.COORDINATES_OUTPUT, _V.COORDINATES)
        # deliberately NO COORDINATES_OUTPUT -> COORDINATES_INITIAL (x_aero0):
        # the SU2 coupling has no x_aero0 input.
        self._mphys_add_pre_coupling_subsystem_from_builder(
            'aero', aero_builder, self.name)
        self.mphys_add_subsystem(
            'coupling', aero_builder.get_coupling_group_subsystem(self.name))
        self._mphys_add_post_coupling_subsystem_from_builder(
            'aero', aero_builder, self.name)

    def configure(self):
        hook = self.options['shape_dv_hook']
        if hook is None:
            from ..geometry.ffd import add_ffd_shape_dvs as hook
        hook(self.geometry, self.options['aero_builder'])
        self.promotes('geometry',
                      inputs=[_V.Geometry.COORDINATES_INPUT],
                      outputs=[_V.Geometry.COORDINATES_OUTPUT])
        super().configure()
