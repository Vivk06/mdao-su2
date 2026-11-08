"""Typed SU2 turbulence, transition, and gas-model profile settings."""

from dataclasses import dataclass

from .errors import ConfigError


def _enums():
    from ..solver import su2_options_v85 as v85
    return v85.ENUMS


def _keys(data, allowed, ctx):
    unknown = sorted(set(data) - set(allowed))
    if unknown:
        raise ConfigError(f'{ctx}: unknown key(s) {unknown}. Allowed: {sorted(allowed)}')


def _options(value, enum, ctx):
    if value is None:
        return ()
    values = [value] if isinstance(value, str) else list(value)
    if not all(isinstance(item, str) for item in values):
        raise ConfigError(f'{ctx}: options must be strings or a list of strings')
    allowed = tuple(_enums()[enum])
    bad = [item for item in values if item not in allowed]
    if bad:
        raise ConfigError(f'{ctx}: {bad} must be chosen from {list(allowed)}')
    return tuple(values)


@dataclass(frozen=True)
class TurbulenceConfig:
    model: str = 'SA'
    options: tuple = ()

    @classmethod
    def from_dict(cls, data):
        _keys(data, ('model', 'options'), '[case.physics.turbulence]')
        model = data.get('model', 'SA')
        allowed = tuple(v for v in _enums()['Turb_Model_Map'] if v != 'NONE')
        if model not in allowed:
            raise ConfigError(f'[case.physics.turbulence] model={model!r} must be one of {list(allowed)}')
        return cls(model=model, options=_options(data.get('options'),
                                                   'SA_Options_Map' if model == 'SA'
                                                   else 'SST_Options_Map',
                                                   '[case.physics.turbulence].options'))


@dataclass(frozen=True)
class TransitionConfig:
    model: str = 'NONE'
    options: tuple = ()

    @classmethod
    def from_dict(cls, data):
        _keys(data, ('model', 'options'), '[case.physics.transition]')
        model = data.get('model', 'NONE')
        allowed = tuple(_enums()['Trans_Model_Map'])
        if model not in allowed:
            raise ConfigError(f'[case.physics.transition] model={model!r} must be one of {list(allowed)}')
        options = _options(data.get('options'), 'LM_Options_Map',
                           '[case.physics.transition].options')
        if model == 'NONE' and options:
            raise ConfigError('[case.physics.transition] options require model="LM"')
        return cls(model=model, options=options)


@dataclass(frozen=True)
class GasConfig:
    """Explicit gas values; ``None`` keeps SU2's STANDARD_AIR defaults implicit."""
    fluid_model: str = None
    gamma: float = None
    gas_constant: float = None
    acentric_factor: float = None
    specific_heat_cp: float = None
    molecular_weight: float = None
    viscosity_model: str = None
    mu_ref: float = None
    mu_t_ref: float = None
    sutherland_constant: float = None
    prandtl_lam: float = None
    prandtl_turb: float = None
    conductivity_model: str = None
    turbulent_conductivity_model: str = None
    init_option: str = None
    freestream_option: str = None

    @classmethod
    def from_dict(cls, data):
        allowed = ('fluid_model', 'gamma', 'gas_constant', 'acentric_factor',
                   'specific_heat_cp', 'molecular_weight', 'viscosity_model',
                   'mu_ref', 'mu_t_ref', 'sutherland_constant', 'prandtl_lam',
                   'prandtl_turb', 'conductivity_model',
                   'turbulent_conductivity_model', 'init_option',
                   'freestream_option')
        _keys(data, allowed, '[case.physics.gas]')
        d = dict(data)
        enums = {
            'fluid_model': 'FluidModel_Map',
            'viscosity_model': 'ViscosityModel_Map',
            'conductivity_model': 'ConductivityModel_Map',
            'turbulent_conductivity_model': 'TurbConductivityModel_Map',
            'init_option': 'InitOption_Map',
            'freestream_option': 'FreeStreamOption_Map',
        }
        for key, enum in enums.items():
            if key in d and d[key] not in _enums()[enum]:
                raise ConfigError(f'[case.physics.gas] {key}={d[key]!r} must be one of {list(_enums()[enum])}')
        for key in ('gamma', 'gas_constant', 'acentric_factor',
                    'specific_heat_cp', 'molecular_weight', 'mu_ref', 'mu_t_ref',
                    'sutherland_constant', 'prandtl_lam', 'prandtl_turb'):
            if key in d:
                d[key] = float(d[key])
        return cls(**d)


def profile_id(physics):
    """Return the stable profile key used by derivative status records."""
    solver = 'euler' if physics.inviscid else 'rans'
    if physics.inviscid:
        return 'euler'
    turb = physics.turbulence
    transition = physics.transition
    turb_options = ','.join(turb.options) or 'none'
    trans_options = ','.join(transition.options) or 'none'
    return f'{solver}:{turb.model.lower()}[{turb_options.lower()}]+{transition.model.lower()}[{trans_options.lower()}]'


__all__ = ['TurbulenceConfig', 'TransitionConfig', 'GasConfig', 'profile_id']
