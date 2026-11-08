"""Typed SU2 option registry: what the cfg writer emits and where each key comes from.

``su2_options_v85.py`` (generated) lists every key SU2 v8.5 parses. This module
types the subset the framework owns: ``REGISTRY`` (scalar keys, in emission
order), ``BC_FAMILIES`` and ``ROLE_KEYS`` (the ``[[case.boundary]]`` keys),
``RUNTIME_KEYS`` (set by the pipeline, never by TOML), and ``STATUS`` (the
derivative evidence the compatibility matrix reports). Every other scanned key
reaches a cfg only through ``[solver.su2_extra]``.
"""

import re
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

from . import su2_options_v85 as v85

# ------------------------------------------------------------- formatting


def yesno(flag):
    return 'YES' if flag else 'NO'


def fmt_ref_origin(value):
    """REF_ORIGIN_MOMENT_*: two decimals when lossless, full precision otherwise."""
    return f'{value:.2f}' if float(f'{value:.2f}') == float(value) else repr(float(value))


def fmt_minval(value):
    """Integer form when lossless."""
    return str(int(value)) if float(value).is_integer() else repr(float(value))


def fmt_option_list(value):
    """Format an SU2 enum list without changing single-value legacy output."""
    if isinstance(value, (list, tuple)):
        return ', '.join(str(item) for item in value)
    return str(value)


def marker_tuple(items):
    """SU2's ``( A, B, C )`` list syntax, items transcribed exactly."""
    return '( ' + ', '.join(str(i) for i in items) + ' )'


def with_output_field(fields, name):
    """Append one SU2 output name without duplicating a configured field."""
    names = [item.strip() for item in fields.split(',')]
    return fields if name in names else f'{fields}, {name}'


# ---------------------------------------------------------------- records

#: Emission sections, in cfg order. ``PASSTHROUGH`` holds ``[solver.su2_extra]``.
SECTIONS = ('PROBLEM DEFINITION', 'FREE-STREAM', 'REFERENCE VALUES',
            'BOUNDARY CONDITIONS', 'MESH DEFORMATION', 'NUMERICAL METHOD',
            'CONVERGENCE', 'INPUT / OUTPUT', 'PASSTHROUGH')
SIDES = ('both', 'primal', 'adjoint')
PARITIES = ('match', 'differ')


@dataclass(frozen=True)
class Ctx:
    """What a record's value and predicate may read."""
    cfg: object
    point: object
    block: object          # the solver block configuration for this side
    is_adjoint: bool
    n_levels: int          # MG array length - 1 (primal depth, zero-level adjoint multigrid)

    @property
    def suffix(self):
        return f'_{self.point.id}'

    @property
    def turbulent(self):
        return self.cfg.case.physics.turbulent


@dataclass(frozen=True)
class Option:
    """One typed SU2 key. ``toml_path`` is None for a runtime key."""
    key: str
    section: str
    value: Callable[[Ctx], object]
    toml_path: Optional[str] = None
    side: str = 'both'
    parity: str = 'match'
    emit_when: Optional[Callable[[Ctx], bool]] = None
    fmt: Callable[[object], str] = str
    group: int = 0         # blank line between groups inside a section
    #: Enum values the TOML can produce, when the schema narrows SU2's set.
    allowed: Optional[Tuple[str, ...]] = None

    @property
    def enum(self):
        """Scanned MapType name for an enum-valued key, else None."""
        return v85.ENUM_OF.get(self.key)

    @property
    def values(self):
        return v85.ENUMS[self.enum] if self.enum else ()

    def applies(self, ctx):
        if self.side != 'both' and (self.side == 'adjoint') != ctx.is_adjoint:
            return False
        return self.emit_when is None or bool(self.emit_when(ctx))


def _o(key, section, value, toml_path=None, **kw):
    return Option(key, section, value, toml_path, **kw)


_P, _F, _R, _D, _N, _C, _IO = (SECTIONS[0], SECTIONS[1], SECTIONS[2],
                               SECTIONS[4], SECTIONS[5], SECTIONS[6], SECTIONS[7])


def _sol(c):
    return c.cfg.solver


def _point_fs(c, name):
    fs = c.point.freestream
    if fs is None:
        return getattr(c.cfg.case.freestream, name)
    if hasattr(fs, '__dataclass_fields__'):
        return getattr(fs, name)
    return fs.get(name, getattr(c.cfg.case.freestream, name))


def _mg_array(attr):
    return lambda c: marker_tuple([getattr(_sol(c).multigrid, attr)] * (c.n_levels + 1))


def _side_name(primal, adjoint):
    return lambda c: (adjoint if c.is_adjoint else primal) + c.suffix


def _history(c):
    o = c.cfg.output.su2
    if c.is_adjoint:
        return o.history_output_adjoint
    fields = with_output_field(o.history_output_primal, c.cfg.constraints.moment.objective_key)
    from .response_catalog import lookup
    for name in c.point.responses:
        fields = with_output_field(fields, lookup(name).history_field)
    return fields


def _screen(c):
    o = c.cfg.output.su2
    if c.is_adjoint:
        return o.screen_output_adjoint
    return with_output_field(o.screen_output_primal, c.cfg.constraints.moment.objective_key)


def _pivot(axis):
    def get(c):
        ref = c.cfg.case.reference
        return c.cfg.constraints.moment.resolved_pivot(ref.origin_moment)[axis]
    return get


REGISTRY = (
    # ---- problem
    _o('SOLVER', _P, lambda c: c.cfg.case.physics.su2_solver, 'case.physics.kind',
       allowed=('EULER', 'RANS')),
    # SU2 ignores KIND_TURB_MODEL under EULER; it is omitted there.
    _o('KIND_TURB_MODEL', _P, lambda c: c.cfg.case.physics.turbulence.model,
       'case.physics.turbulence.model', emit_when=lambda c: c.turbulent,
       allowed=tuple(v for v in v85.ENUMS['Turb_Model_Map'] if v != 'NONE')),
    _o('SA_OPTIONS', _P, lambda c: c.cfg.case.physics.turbulence.options,
       'case.physics.turbulence.options', emit_when=lambda c: c.turbulent and c.cfg.case.physics.turbulence.model == 'SA' and bool(c.cfg.case.physics.turbulence.options),
       allowed=tuple(v85.ENUMS['SA_Options_Map']), fmt=fmt_option_list),
    _o('SST_OPTIONS', _P, lambda c: c.cfg.case.physics.turbulence.options,
       'case.physics.turbulence.options', emit_when=lambda c: c.turbulent and c.cfg.case.physics.turbulence.model == 'SST' and bool(c.cfg.case.physics.turbulence.options),
       allowed=tuple(v85.ENUMS['SST_Options_Map']), fmt=fmt_option_list),
    _o('KIND_TRANS_MODEL', _P, lambda c: c.cfg.case.physics.transition.model,
       'case.physics.transition.model', emit_when=lambda c: c.turbulent and c.cfg.case.physics.transition.model != 'NONE'),
    _o('LM_OPTIONS', _P, lambda c: c.cfg.case.physics.transition.options,
       'case.physics.transition.options', emit_when=lambda c: c.turbulent and c.cfg.case.physics.transition.model == 'LM' and bool(c.cfg.case.physics.transition.options),
       allowed=tuple(v85.ENUMS['LM_Options_Map']), fmt=fmt_option_list),
    _o('MATH_PROBLEM', _P, lambda c: 'DISCRETE_ADJOINT' if c.is_adjoint else 'DIRECT',
       parity='differ'),
    _o('OBJECTIVE_FUNCTION', _P, lambda c: _sol(c).adjoint_settings.objective_function,
       'solver.adjoint_settings.objective_function', side='adjoint'),
    # only under RANS -- Euler has no viscous term to freeze.
    _o('FROZEN_VISC_DISC', _P, lambda c: _sol(c).adjoint_settings.frozen_visc_disc,
       'solver.adjoint_settings.frozen_visc_disc', side='adjoint',
       emit_when=lambda c: c.turbulent, fmt=yesno),
    _o('FROZEN_LIMITER_DISC', _P, lambda c: True,
       'solver.adjoint_settings.frozen_limiter_disc', side='adjoint',
       emit_when=lambda c: _sol(c).adjoint_settings.frozen_limiter_disc, fmt=yesno),
    _o('RESTART_SOL', _P, lambda c: 'NO'),       # the solver toggles it per solve

    # ---- free-stream
    _o('MACH_NUMBER', _F, lambda c: c.point.mach, 'points[].mach'),
    _o('AOA', _F, lambda c: c.point.aoa_init, 'points[].aoa_init'),      # DEGREES
    _o('FREESTREAM_TEMPERATURE', _F, lambda c: _point_fs(c, 'temperature'),
       'points[].freestream.temperature'),
    # An inviscid case has no Reynolds number; it states the pressure instead.
    _o('REYNOLDS_NUMBER', _F, lambda c: _point_fs(c, 'reynolds_number'),
       'points[].freestream.reynolds_number', emit_when=lambda c: c.turbulent),
    _o('REYNOLDS_LENGTH', _F, lambda c: _point_fs(c, 'reynolds_length'),
       'points[].freestream.reynolds_length', emit_when=lambda c: c.turbulent),
    _o('FREESTREAM_PRESSURE', _F, lambda c: _point_fs(c, 'pressure'),
       'points[].freestream.pressure', emit_when=lambda c: _point_fs(c, 'pressure')),
    _o('FREESTREAM_TURBULENCEINTENSITY', _F, lambda c: c.cfg.case.physics.turbulence_intensity,
       'case.physics.turbulence_intensity', emit_when=lambda c: c.turbulent and c.cfg.case.physics.turbulence_intensity is not None),
    _o('FREESTREAM_TURB2LAMVISCRATIO', _F, lambda c: c.cfg.case.physics.turb2lam_viscosity_ratio,
       'case.physics.turb2lam_viscosity_ratio', emit_when=lambda c: c.turbulent and c.cfg.case.physics.turb2lam_viscosity_ratio is not None),
    _o('FLUID_MODEL', _F, lambda c: c.cfg.case.physics.gas.fluid_model,
       'case.physics.gas.fluid_model', emit_when=lambda c: c.cfg.case.physics.gas.fluid_model is not None),
    _o('GAMMA_VALUE', _F, lambda c: c.cfg.case.physics.gas.gamma,
       'case.physics.gas.gamma', emit_when=lambda c: c.cfg.case.physics.gas.gamma is not None),
    _o('GAS_CONSTANT', _F, lambda c: c.cfg.case.physics.gas.gas_constant,
       'case.physics.gas.gas_constant', emit_when=lambda c: c.cfg.case.physics.gas.gas_constant is not None),
    _o('ACENTRIC_FACTOR', _F, lambda c: c.cfg.case.physics.gas.acentric_factor,
       'case.physics.gas.acentric_factor', emit_when=lambda c: c.cfg.case.physics.gas.acentric_factor is not None),
    _o('SPECIFIC_HEAT_CP', _F, lambda c: c.cfg.case.physics.gas.specific_heat_cp,
       'case.physics.gas.specific_heat_cp', emit_when=lambda c: c.cfg.case.physics.gas.specific_heat_cp is not None),
    _o('MOLECULAR_WEIGHT', _F, lambda c: c.cfg.case.physics.gas.molecular_weight,
       'case.physics.gas.molecular_weight', emit_when=lambda c: c.cfg.case.physics.gas.molecular_weight is not None),
    _o('VISCOSITY_MODEL', _F, lambda c: c.cfg.case.physics.gas.viscosity_model,
       'case.physics.gas.viscosity_model', emit_when=lambda c: c.cfg.case.physics.gas.viscosity_model is not None),
    _o('MU_REF', _F, lambda c: c.cfg.case.physics.gas.mu_ref,
       'case.physics.gas.mu_ref', emit_when=lambda c: c.cfg.case.physics.gas.mu_ref is not None),
    _o('MU_T_REF', _F, lambda c: c.cfg.case.physics.gas.mu_t_ref,
       'case.physics.gas.mu_t_ref', emit_when=lambda c: c.cfg.case.physics.gas.mu_t_ref is not None),
    _o('SUTHERLAND_CONSTANT', _F, lambda c: c.cfg.case.physics.gas.sutherland_constant,
       'case.physics.gas.sutherland_constant', emit_when=lambda c: c.cfg.case.physics.gas.sutherland_constant is not None),
    _o('PRANDTL_LAM', _F, lambda c: c.cfg.case.physics.gas.prandtl_lam,
       'case.physics.gas.prandtl_lam', emit_when=lambda c: c.cfg.case.physics.gas.prandtl_lam is not None),
    _o('PRANDTL_TURB', _F, lambda c: c.cfg.case.physics.gas.prandtl_turb,
       'case.physics.gas.prandtl_turb', emit_when=lambda c: c.cfg.case.physics.gas.prandtl_turb is not None),
    _o('CONDUCTIVITY_MODEL', _F, lambda c: c.cfg.case.physics.gas.conductivity_model,
       'case.physics.gas.conductivity_model', emit_when=lambda c: c.cfg.case.physics.gas.conductivity_model is not None),
    _o('TURBULENT_CONDUCTIVITY_MODEL', _F, lambda c: c.cfg.case.physics.gas.turbulent_conductivity_model,
       'case.physics.gas.turbulent_conductivity_model', emit_when=lambda c: c.cfg.case.physics.gas.turbulent_conductivity_model is not None),
    _o('INIT_OPTION', _F, lambda c: c.cfg.case.physics.gas.init_option,
       'case.physics.gas.init_option', emit_when=lambda c: c.cfg.case.physics.gas.init_option is not None),
    _o('FREESTREAM_OPTION', _F, lambda c: c.cfg.case.physics.gas.freestream_option,
       'case.physics.gas.freestream_option', emit_when=lambda c: c.cfg.case.physics.gas.freestream_option is not None),

    # ---- reference values
    *(_o(f'REF_ORIGIN_MOMENT_{a}', _R, _pivot(i), f'case.reference.origin_moment[{i}]',
         fmt=fmt_ref_origin) for i, a in enumerate('XYZ')),
    _o('REF_LENGTH', _R, lambda c: c.cfg.case.reference.length, 'case.reference.length'),
    # 0 = SU2 auto-computes planform area from monitored markers.
    _o('REF_AREA', _R, lambda c: (0 if c.cfg.case.reference.area_is_auto
                                  else c.cfg.case.reference.area), 'case.reference.area'),
    _o('REF_DIMENSIONALIZATION', _R, lambda c: _point_fs(c, 'dimensionalization'),
       'points[].freestream.dimensionalization',
       emit_when=lambda c: _point_fs(c, 'dimensionalization')),

    # ---- mesh deformation (the marker roles are in the boundary section)
    _o('DEFORM_MESH', _D, lambda c: 'YES'),       # remesh path rewrites it at runtime
    *(_o(f'DEFORM_{k.upper()}', _D, (lambda a: lambda c: getattr(c.cfg.deform, a))(k),
         f'deform.{k}', emit_when=lambda c: c.cfg.deform.emit,
         fmt=yesno if k == 'console_output' else str)
      for k in ('linear_solver', 'linear_solver_prec', 'linear_solver_error',
                'linear_solver_iter', 'stiffness_type', 'nonlinear_iter',
                'console_output')),

    # ---- numerics
    _o('NUM_METHOD_GRAD', _N, lambda c: _sol(c).numerics.num_method_grad,
       'solver.numerics.num_method_grad'),
    _o('CFL_NUMBER', _N, lambda c: _sol(c).numerics.cfl_number, 'solver.numerics.cfl_number'),
    # a no-op on the adjoint, written for pair-diff readability.
    _o('CFL_ADAPT', _N, lambda c: c.block.cfl_adapt, 'solver.{side}.cfl_adapt',
       parity='differ', fmt=yesno),
    _o('CFL_ADAPT_PARAM', _N, lambda c: c.block.cfl_adapt_param,
       'solver.{side}.cfl_adapt_param', parity='differ',
       emit_when=lambda c: c.block.cfl_adapt),
    _o('ITER', _N, lambda c: c.block.iter, 'solver.{side}.iter', parity='differ'),
    _o('LINEAR_SOLVER', _N, lambda c: _sol(c).linear.solver, 'solver.linear.solver'),
    _o('LINEAR_SOLVER_ERROR', _N, lambda c: _sol(c).linear.error, 'solver.linear.error'),
    _o('LINEAR_SOLVER_ITER', _N, lambda c: _sol(c).linear.iter, 'solver.linear.iter'),
    _o('LINEAR_SOLVER_PREC', _N, lambda c: _sol(c).linear.prec, 'solver.linear.prec',
       emit_when=lambda c: _sol(c).linear.prec != 'ILU'),     # ILU is SU2's default
    _o('RELAXATION_FACTOR_ADJOINT', _N, lambda c: _sol(c).adjoint_settings.relaxation_factor,
       'solver.adjoint_settings.relaxation_factor', side='adjoint',
       emit_when=lambda c: _sol(c).adjoint_settings.relaxation_factor != 1.0),
    _o('MGLEVEL', _N, lambda c: c.block.mglevel, 'solver.{side}.mglevel',
       parity='differ', group=1),
    _o('MGCYCLE', _N, lambda c: _sol(c).multigrid.cycle, 'solver.multigrid.cycle', group=1),
    _o('MG_PRE_SMOOTH', _N, _mg_array('pre_smooth'), 'solver.multigrid.pre_smooth', group=1),
    _o('MG_POST_SMOOTH', _N, _mg_array('post_smooth'), 'solver.multigrid.post_smooth',
       group=1),
    _o('MG_CORRECTION_SMOOTH', _N, _mg_array('correction_smooth'),
       'solver.multigrid.correction_smooth', group=1),
    _o('MG_DAMP_RESTRICTION', _N, lambda c: _sol(c).multigrid.damp_restriction,
       'solver.multigrid.damp_restriction', group=1),
    _o('MG_DAMP_PROLONGATION', _N, lambda c: _sol(c).multigrid.damp_prolongation,
       'solver.multigrid.damp_prolongation', group=1),
    _o('CONV_NUM_METHOD_FLOW', _N, lambda c: _sol(c).numerics.conv_method_flow,
       'solver.numerics.conv_method_flow', group=2),
    # SU2 defaults MUSCL_FLOW to YES and turns it off only for a centered scheme
    # (CConfig.cpp:2023,3795-3800): omit it only there, or an upwind case asking
    # for NO would run with MUSCL on.
    _o('MUSCL_FLOW', _N, lambda c: _sol(c).numerics.muscl_flow, 'solver.numerics.muscl_flow',
       emit_when=lambda c: (_sol(c).numerics.muscl_flow != 'NO' or
                            _sol(c).numerics.conv_method_flow not in v85.ENUMS['Centered_Map']),
       group=2),
    _o('SLOPE_LIMITER_FLOW', _N, lambda c: _sol(c).numerics.slope_limiter_flow,
       'solver.numerics.slope_limiter_flow', group=2),
    _o('VENKAT_LIMITER_COEFF', _N, lambda c: _sol(c).numerics.venkat_limiter_coeff,
       'solver.numerics.venkat_limiter_coeff', group=2),
    _o('JST_SENSOR_COEFF', _N, lambda c: _sol(c).numerics.jst_sensor_coeff,
       'solver.numerics.jst_sensor_coeff', group=2),
    _o('TIME_DISCRE_FLOW', _N, lambda c: _sol(c).numerics.time_discre_flow,
       'solver.numerics.time_discre_flow', group=2),
    _o('CONV_NUM_METHOD_TURB', _N, lambda c: _sol(c).numerics.conv_method_turb,
       'solver.numerics.conv_method_turb', emit_when=lambda c: c.turbulent, group=3),
    _o('MUSCL_TURB', _N, lambda c: _sol(c).numerics.muscl_turb, 'solver.numerics.muscl_turb',
       emit_when=lambda c: c.turbulent and _sol(c).numerics.muscl_turb != 'NO', group=3),
    _o('SLOPE_LIMITER_TURB', _N, lambda c: _sol(c).numerics.slope_limiter_turb,
       'solver.numerics.slope_limiter_turb', emit_when=lambda c: c.turbulent, group=3),
    _o('TIME_DISCRE_TURB', _N, lambda c: _sol(c).numerics.time_discre_turb,
       'solver.numerics.time_discre_turb', emit_when=lambda c: c.turbulent, group=3),

    # ---- convergence
    _o('CONV_RESIDUAL_MINVAL', _C, lambda c: c.block.conv_residual_minval,
       'solver.{side}.conv_residual_minval', parity='differ', fmt=fmt_minval),
    _o('CONV_STARTITER', _C, lambda c: _sol(c).convergence.startiter,
       'solver.convergence.startiter'),
    _o('CONV_CAUCHY_ELEMS', _C, lambda c: _sol(c).convergence.cauchy_elems,
       'solver.convergence.cauchy_elems'),
    _o('CONV_CAUCHY_EPS', _C, lambda c: _sol(c).convergence.cauchy_eps,
       'solver.convergence.cauchy_eps'),

    # ---- input / output
    _o('MESH_FILENAME', _IO, lambda c: c.cfg.mesh_path),     # absolute (R4)
    _o('MESH_FORMAT', _IO, lambda c: c.cfg.output.su2.mesh_format, 'output.su2.mesh_format'),
    _o('TABULAR_FORMAT', _IO, lambda c: c.cfg.output.su2.tabular_format,
       'output.su2.tabular_format'),
    _o('HISTORY_OUTPUT', _IO, _history, 'output.su2.history_output_{side}',
       parity='differ'),
    _o('SCREEN_OUTPUT', _IO, _screen, 'output.su2.screen_output_{side}',
       parity='differ'),
    _o('CONV_FILENAME', _IO, _side_name('history', 'history_adj'), parity='differ'),
    # Same on both sides: the adjoint linearizes about this point's flow.
    # Stems from [output.su2] *_basename; runtime keys, so passthrough cannot set them.
    _o('RESTART_FILENAME', _IO, lambda c: c.cfg.output.su2.restart_basename + c.suffix),
    _o('SOLUTION_FILENAME', _IO, lambda c: 'solution_flow' + c.suffix),
    *(_o(key, _IO, (lambda s: lambda c: s + c.suffix)(stem), side='adjoint')
      for key, stem in (('SOLUTION_ADJ_FILENAME', 'solution_adj'),
                        ('RESTART_ADJ_FILENAME', 'restart_adj'),
                        ('VOLUME_ADJ_FILENAME', 'adjoint'),
                        ('GRAD_OBJFUNC_FILENAME', 'of_grad'),
                        ('SURFACE_ADJ_FILENAME', 'surface_adjoint'))),
    _o('VOLUME_FILENAME', _IO, lambda c: c.cfg.output.su2.volume_basename + c.suffix,
       side='primal'),
    _o('SURFACE_FILENAME', _IO, lambda c: c.cfg.output.su2.surface_basename + c.suffix,
       side='primal'),
    # Same on both sides; empty lists leave SU2's defaults (not emitted).
    _o('VOLUME_OUTPUT', _IO, lambda c: marker_tuple(c.cfg.output.su2.volume_output),
       'output.su2.volume_output', emit_when=lambda c: bool(c.cfg.output.su2.volume_output)),
    _o('OUTPUT_FILES', _IO, lambda c: marker_tuple(c.cfg.output.su2.output_files),
       'output.su2.output_files', emit_when=lambda c: bool(c.cfg.output.su2.output_files)),
    _o('OUTPUT_WRT_FREQ', _IO, lambda c: c.cfg.output.su2.wrt_freq, 'output.su2.wrt_freq'),
)

#: Keys the pipeline owns. No TOML path sets them and passthrough may not name them.
RUNTIME_KEYS = frozenset(o.key for o in REGISTRY if o.toml_path is None) | {
    'OBJECTIVE_FUNCTION'}


# -------------------------------------------------------------- boundaries

@dataclass(frozen=True)
class Param:
    """One bc parameter: ``n`` numbers, or one value of a scanned enum."""
    name: str
    n: int = 1
    enum: Optional[str] = None


@dataclass(frozen=True)
class BCFamily:
    key: str
    params: Tuple[Param, ...] = ()
    #: Walls may carry the ``design`` role; SU2 rejects viscous walls under Euler.
    wall: bool = False
    viscous: bool = False
    #: Entries of two markers (inlet, outlet) rather than one tuple per marker.
    paired: bool = False
    #: Closes the domain; a case needs at least one outer boundary.
    outer: bool = False


BC_FAMILIES = {
    'euler': BCFamily('MARKER_EULER', wall=True),
    'heatflux': BCFamily('MARKER_HEATFLUX', (Param('value'),), wall=True, viscous=True),
    'isothermal': BCFamily('MARKER_ISOTHERMAL', (Param('value'),), wall=True, viscous=True),
    'heattransfer': BCFamily('MARKER_HEATTRANSFER',
                             (Param('coefficient'), Param('temperature')),
                             wall=True, viscous=True),
    'far': BCFamily('MARKER_FAR', outer=True),
    'sym': BCFamily('MARKER_SYM'),
    'internal': BCFamily('MARKER_INTERNAL'),
    'nearfield': BCFamily('MARKER_NEARFIELD'),
    'inlet': BCFamily('MARKER_INLET', (Param('total_temperature'),
                                       Param('total_pressure'), Param('direction', 3)),
                      outer=True),
    'supersonic_inlet': BCFamily('MARKER_SUPERSONIC_INLET', (
        Param('temperature'), Param('pressure'), Param('velocity', 3)), outer=True),
    'outlet': BCFamily('MARKER_OUTLET', (Param('pressure'),), outer=True),
    'supersonic_outlet': BCFamily('MARKER_SUPERSONIC_OUTLET', outer=True),
    'riemann': BCFamily('MARKER_RIEMANN', (Param('kind', enum='Riemann_Map'),
                                           Param('values', 2), Param('direction', 3)),
                           outer=True),
    'engine_inflow': BCFamily('MARKER_ENGINE_INFLOW', (Param('value'),)),
    'engine_exhaust': BCFamily('MARKER_ENGINE_EXHAUST', (Param('total_temperature'),
                                                         Param('total_pressure'))),
    'actdisk': BCFamily('MARKER_ACTDISK', (Param('values', 6),), paired=True),
}

#: Role -> cfg key. ``design`` is the wall seam surface and has no key;
#: ``MARKER_DESIGNING`` is read only by SU2_DEF/SU2_DOT and is not emitted.
ROLE_KEYS = {
    'design': None,
    'monitor': 'MARKER_MONITORING',
    'plot': 'MARKER_PLOTTING',
    'analyze': 'MARKER_ANALYZE',
    'deform': 'MARKER_DEFORM_MESH',
    'deform_sym': 'MARKER_DEFORM_MESH_SYM_PLANE',
}

#: SU2's fixed-lift family. Lift targeting is [trim] + [constraints.lift] with
#: alpha as a design variable; these keys may not reach a cfg.
FIXED_CL_KEYS = frozenset({'FIXED_CL_MODE', 'TARGET_CL', 'DCL_DALPHA',
                           'UPDATE_AOA_ITER_LIMIT', 'ITER_DCL_DALPHA', 'EVAL_DOF_DCX',
                           'DCD_DCL_VALUE', 'DCMX_DCL_VALUE', 'DCMY_DCL_VALUE',
                           'DCMZ_DCL_VALUE'})

#: Keys emitted from ``[[case.boundary]]``; passthrough may not name them.
BOUNDARY_KEYS = frozenset(f.key for f in BC_FAMILIES.values()) | frozenset(
    k for k in ROLE_KEYS.values() if k) | {'MARKER_DESIGNING'}


def bc_line(family, entries):
    """The cfg value for one bc family, from its entries in declaration order."""
    items = []
    for entry in entries:
        groups = [entry.markers] if family.paired else [[m] for m in entry.markers]
        for group in groups:
            items += list(group)
            for p in family.params:
                v = entry.params[p.name]
                items += [str(x) for x in v] if p.n > 1 else [str(v)]
    return marker_tuple(items)


# ------------------------------------------------------------ passthrough

def passthrough_blocked(key):
    """Why ``key`` may not appear in ``[solver.su2_extra]``, or '' if it may."""
    if key not in v85.OPTION_KEYS:
        return f'{key} is not an SU2 v8.5 option (commit {v85.SU2_COMMIT[:10]})'
    if key in RUNTIME_KEYS:
        return f'{key} is set by the pipeline at runtime'
    if key in FIXED_CL_KEYS:
        return (f'{key} belongs to SU2 fixed-CL mode, which the framework does not '
                f'use; target lift with [constraints.lift] and [trim]')
    if key in BOUNDARY_KEYS:
        return f'{key} is emitted from [[case.boundary]] bc/roles'
    typed = TYPED_KEYS.get(key)
    if typed:
        return f'{key} is typed; set {typed} instead'
    return ''


#: Keys that define the discretized problem. The discrete adjoint must see the
#: primal's values (primal/adjoint parity), so a side-specific passthrough may not name
#: them: flow conditions, gas and turbulence models, reference values, markers,
#: mesh, and the convective/gradient schemes it differentiates.
PHYSICS_KEY = re.compile(
    r'^(SOLVER|AOA|SIDESLIP_ANGLE|SYSTEM_MEASUREMENTS|INIT_OPTION|KIND_TURB_MODEL|'
    r'KIND_TRANS_MODEL|(SA|SST|LM)_OPTIONS|NUM_METHOD_GRAD|ENTROPY_FIX_COEFF)$|'
    r'^(FREESTREAM_|REYNOLDS_|MACH_|REF_|MARKER_|MESH_|FLUID_|GAMMA_|GAS_|MU_|'
    r'PRANDTL_|VISCOSITY_|CONDUCTIVITY_|TURBULENT_CONDUCTIVITY|SPECIFIC_HEAT|'
    r'MOLECULAR_WEIGHT|ACENTRIC_|CRITICAL_|SUTHERLAND_|INC_|LOW_MACH_|TIME_|'
    r'CONV_NUM_METHOD_|MUSCL_|SLOPE_LIMITER_|JST_|LAX_|VENKAT_|ROE_|'
    r'CENTRAL_|TIME_DISCRE_|GRID_MOVEMENT|MOTION_|ROTATION_|TRANSLATION_)')


def side_passthrough_blocked(key):
    """Why ``key`` may not go to one side only, or '' if it may."""
    if PHYSICS_KEY.match(key):
        return (f'{key} defines the discretized problem; primal and adjoint must '
                f'share it (primal/adjoint parity), so set it in [solver.su2_extra]')
    return ''


def fmt_passthrough(value):
    if isinstance(value, bool):
        return yesno(value)
    if isinstance(value, (list, tuple)):
        return marker_tuple(fmt_passthrough(v) for v in value)
    return str(value)


def enum_violations(key, value):
    """Tokens of an enum-valued cfg ``value`` that SU2 v8.5 does not accept."""
    enum = v85.ENUM_OF.get(key)
    if not enum:
        return []
    tokens = [t.strip() for t in str(value).strip('() ').split(',') if t.strip()]
    return [t for t in tokens if t not in v85.ENUMS[enum]]


# ------------------------------------------------------------------ views

#: key -> TOML path, for every key a TOML can move.
TYPED_KEYS = {o.key: o.toml_path for o in REGISTRY if o.toml_path}
TOML_REACHABLE_KEYS = {**TYPED_KEYS,
                       **{f.key: f'case.boundary[].bc = "{name}"'
                          for name, f in BC_FAMILIES.items()},
                       **{k: f'case.boundary[].roles = ["{r}"]'
                          for r, k in ROLE_KEYS.items() if k}}


def intentional_differences(extra_primal=(), extra_adjoint=()):
    """Keys allowed to differ between a pair: per-side records + side passthrough."""
    return frozenset(o.key for o in REGISTRY if o.parity == 'differ' or o.side != 'both') \
        | frozenset(extra_primal) | frozenset(extra_adjoint)


# ----------------------------------------------------------------- status

STATUS_VALUES = ('assumed', 'primal_checked', 'fd_validated', 'fd_failed', 'unsupported')
_NEEDS_EVIDENCE = ('primal_checked', 'fd_validated', 'fd_failed', 'unsupported')


@dataclass(frozen=True)
class Status:
    """Derivative evidence for one subject (``profile:``, ``response:``, ``bc:``)."""
    subject: str
    adjoint: str = 'assumed'
    primal: str = ''
    evidence: Tuple[str, ...] = ()

    def __post_init__(self):
        if self.adjoint not in STATUS_VALUES:
            raise ValueError(f'{self.subject}: status {self.adjoint!r} not in {STATUS_VALUES}')
        if self.adjoint in _NEEDS_EVIDENCE and not self.evidence:
            raise ValueError(f'{self.subject}: {self.adjoint} needs a cited evidence path')


_STATUS_DOC = 'docs/human-guide.pdf'
_VALIDATION = _STATUS_DOC

STATUS = (
    Status('profile:rans_sa', 'fd_validated', 'yes', (_STATUS_DOC,)),
    # Euler primal runs are accepted (ADODG1); the Euler FD record is per response.
    Status('profile:euler', 'primal_checked', 'yes', (_VALIDATION,)),
    Status('profile:rans_sst'),
    # E387 M 0.55 and NLF(1)-0416 M 0.25 cold primals; derivatives not FD-gated.
    Status('profile:rans_sa_bcm', 'primal_checked', 'yes',
           (_STATUS_DOC,)),
    Status('profile:rans_sa_lm', 'unsupported', 'yes',
           (_STATUS_DOC,)),
    Status('profile:rans_sst_lm', 'unsupported', 'yes',
           (_STATUS_DOC,)),
    Status('response:DRAG@rans_sa', 'fd_validated', 'yes', (_STATUS_DOC,)),
    Status('response:LIFT@rans_sa', 'fd_validated', 'yes', (_STATUS_DOC,)),
    Status('response:MOMENT_Z@rans_sa', 'fd_validated', 'yes', (_STATUS_DOC,)),
    # Alpha row and normal shape rows; three leading-edge rows excluded because their finite-difference values are unreliable.
    Status('response:EFFICIENCY@rans_sa', 'fd_validated', 'yes',
           (_STATUS_DOC,)),
    # TE-twist and alpha rows, ADODG Case 3 exact-LE mesh.
    Status('response:DRAG@euler', 'fd_validated', 'yes',
           (_STATUS_DOC,)),
    Status('response:LIFT@euler', 'fd_validated', 'yes',
           (_STATUS_DOC,)),
    Status('bc:heatflux', 'fd_validated', 'yes', (_STATUS_DOC,)),
    Status('bc:far', 'fd_validated', 'yes', (_STATUS_DOC,)),
    Status('bc:euler', 'primal_checked', 'yes', (_VALIDATION,)),
    Status('bc:sym', 'primal_checked', 'yes', (_VALIDATION,)),
)


def _check_registry():
    """Import-time contract: the records agree with the SU2 scan."""
    seen = set()
    for o in REGISTRY:
        if o.key in seen:
            raise ValueError(f'{o.key} is registered twice')
        seen.add(o.key)
        if o.key not in v85.OPTION_KEYS:
            raise ValueError(f'{o.key} is not an SU2 v8.5 option')
        if o.section not in SECTIONS or o.side not in SIDES or o.parity not in PARITIES:
            raise ValueError(f'{o.key}: bad section/side/parity')
        if o.enum and not o.values:
            raise ValueError(f'{o.key}: enum {o.enum} has no scanned values')
    for name, fam in BC_FAMILIES.items():
        if fam.key not in v85.MARKER_FAMILIES:
            raise ValueError(f'bc {name}: {fam.key} is not an SU2 v8.5 marker option')
        for p in fam.params:
            if p.enum and p.enum not in v85.ENUMS:
                raise ValueError(f'bc {name}: enum {p.enum} not scanned')


_check_registry()

__all__ = ['Ctx', 'Option', 'REGISTRY', 'SECTIONS', 'RUNTIME_KEYS', 'Param', 'BCFamily',
           'BC_FAMILIES', 'ROLE_KEYS', 'BOUNDARY_KEYS', 'FIXED_CL_KEYS', 'bc_line', 'TYPED_KEYS',
           'TOML_REACHABLE_KEYS', 'intentional_differences', 'passthrough_blocked',
           'side_passthrough_blocked', 'PHYSICS_KEY',
           'fmt_passthrough', 'enum_violations', 'Status', 'STATUS', 'STATUS_VALUES',
           'marker_tuple']
