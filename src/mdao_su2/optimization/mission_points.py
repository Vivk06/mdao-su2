"""Build mission operating points and linear constraints from run TOML."""

from dataclasses import dataclass, fields
import math
import tomllib

from mdao_su2.run_config.errors import ConfigError


G0 = 9.80665
R_AIR = 287.05287
GAMMA = 1.4
T0 = 288.15
P0 = 101325.0
LAPSE = 0.0065
MU0 = 1.716e-5
T_MU0 = 273.15      # temperature at which MU0 is defined (Sutherland reference)
SUTHERLAND = 110.4
MACH_TOL = 1e-9
LEVEL_SCALER = 1.0
BRACKET_SCALER = 100.0


@dataclass(frozen=True)
class AtmosphereConstants:
    gravity: float = G0
    gas_constant: float = R_AIR
    gamma: float = GAMMA
    sea_level_temperature: float = T0
    sea_level_pressure: float = P0
    lapse_rate: float = LAPSE
    reference_viscosity: float = MU0
    reference_temperature: float = T_MU0
    sutherland_constant: float = SUTHERLAND
    tropopause_altitude: float = 11000.0

    @classmethod
    def from_dict(cls, raw):
        if not isinstance(raw, dict):
            raise ConfigError('[mission.atmosphere] must be a table')
        allowed = {item.name for item in fields(cls)}
        unknown = set(raw) - allowed
        if unknown:
            raise ConfigError(f'[mission.atmosphere] unknown keys: {sorted(unknown)}')
        values = {}
        for key, value in raw.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ConfigError(f'[mission.atmosphere] {key} must be a positive number')
            value = float(value)
            if not math.isfinite(value) or value <= 0.0:
                raise ConfigError(f'[mission.atmosphere] {key} must be finite and positive')
            values[key] = value
        result = cls(**values)
        if result.sea_level_temperature <= result.lapse_rate * result.tropopause_altitude:
            raise ConfigError('[mission.atmosphere] temperature reaches zero before the tropopause')
        return result


def atmosphere(altitude, constants=None):
    """ISA troposphere and lower-stratosphere density, sound speed and viscosity."""
    constants = constants or AtmosphereConstants()
    if not math.isfinite(altitude) or altitude < 0.0:
        raise ValueError('altitude must be >= 0 m')
    if altitude <= constants.tropopause_altitude:
        temperature = constants.sea_level_temperature - constants.lapse_rate * altitude
        pressure = constants.sea_level_pressure * (
            temperature / constants.sea_level_temperature) ** (
                constants.gravity / (constants.gas_constant * constants.lapse_rate))
    else:
        temperature = constants.sea_level_temperature - (
            constants.lapse_rate * constants.tropopause_altitude)
        pressure_at_tropopause = constants.sea_level_pressure * (
            temperature / constants.sea_level_temperature) ** (
                constants.gravity / (constants.gas_constant * constants.lapse_rate))
        pressure = pressure_at_tropopause * math.exp(
            -constants.gravity * (altitude - constants.tropopause_altitude) /
            (constants.gas_constant * temperature))
    density = pressure / (constants.gas_constant * temperature)
    sound_speed = math.sqrt(constants.gamma * constants.gas_constant * temperature)
    viscosity = (constants.reference_viscosity *
                 (temperature / constants.reference_temperature) ** 1.5 *
                 (constants.reference_temperature + constants.sutherland_constant) /
                 (temperature + constants.sutherland_constant))
    return density, sound_speed, temperature, viscosity


def point_id(mach):
    """Scenario-safe point id: OpenMDAO rejects dots in subsystem names."""
    return f'mission_m{round(1000.0 * mach):04d}'


def cl_at_mach(cl_t, mach_t, mach):
    """Constant-loading lift coefficient: C_l(M) = C_l,t (M_t/M)^2."""
    return cl_t * (mach_t / mach) ** 2


def _find(mach_values, target, tolerance=MACH_TOL):
    for m in mach_values:
        if abs(m - target) <= tolerance:
            return m
    return None


def _terms(*items):
    body = ', '.join(f'{{ point = "{pid}", response = "{resp}", coeff = {coeff:.16g} }}'
                     for pid, resp, coeff in items)
    return f'terms = [{body}]'


def points_fragment(auw, sref, cref, altitude, mach_values, cl_at=None,
                    level_flight=False, bracket=None,
                    atmosphere_constants=None, level_scaler=LEVEL_SCALER,
                    bracket_scaler=BRACKET_SCALER, point_defaults=None,
                    mach_tolerance=MACH_TOL):
    """Return the TOML fragment text.

    The aircraft-data form passes ``auw`` and ``sref``; the lift-coefficient
    form passes ``cl_at = (M_t, C_l,t)`` and leaves them ``None``.
    """
    if cref <= 0:
        raise ValueError('cref must be a positive SI value')
    if cl_at is None:
        if auw is None or sref is None:
            raise ValueError('give either auw with sref, or cl_at')
        if auw <= 0 or sref <= 0:
            raise ValueError('auw and sref must be positive SI values')
        if level_flight or bracket is not None:
            raise ValueError('level_flight and bracket need cl_at')
    else:
        if auw is not None or sref is not None:
            raise ValueError('cl_at excludes auw and sref')
        mach_t, cl_t = cl_at
        if not (math.isfinite(mach_t) and mach_t > 0.0 and math.isfinite(cl_t) and cl_t > 0.0):
            raise ValueError('cl_at needs a positive finite Mach number and C_l')
    for mach in mach_values:
        if not math.isfinite(mach) or mach <= 0.0:
            raise ValueError(f'Mach values must be finite and positive, got {mach!r}')
    ids = [point_id(m) for m in mach_values]
    if len(set(ids)) != len(ids):
        raise ValueError('Mach values must differ by at least 0.0005')
    by_mach = dict(zip(mach_values, ids))
    mach_t_listed = None
    if cl_at is not None:
        mach_t_listed = _find(mach_values, cl_at[0], mach_tolerance)
        if level_flight and mach_t_listed is None:
            raise ValueError(f'level_flight needs M_t={cl_at[0]:g} among mach')
    if bracket is not None:
        if not math.isfinite(bracket) or bracket <= 0.0:
            raise ValueError('bracket must be a positive Mach increment')
        missing = [m for m in (cl_at[0] - bracket, cl_at[0], cl_at[0] + bracket)
                   if _find(mach_values, m, mach_tolerance) is None]
        if missing:
            raise ValueError('bracket needs M_t - delta, M_t and M_t + delta among '
                             f'mach; missing {[round(m, 12) for m in missing]}')

    rho, sound, temperature, mu = atmosphere(altitude, atmosphere_constants)
    if cl_at is None:
        rows = [f'# Mission inputs (SI): AUW={auw:g} N, Sref={sref:g} m^2, '
                f'cref={cref:g} m, altitude={altitude:g} m']
    else:
        rows = [f'# Mission inputs: C_l,t={cl_at[1]:g} at M_t={cl_at[0]:g}, '
                f'C_l(M) = C_l,t (M_t/M)^2, cref={cref:g} m, altitude={altitude:g} m']
        if level_flight:
            rows.append('# Level flight: start_cl is a trim aim; lift is held by the '
                        'level_* linear equalities')
        if bracket is not None:
            rows.append(f'# Drag bracket: delta={bracket:g}')
    rows.append(f'# ISA: T={temperature:.12g} K, rho={rho:.12g} kg/m^3')
    for mach, pid in zip(mach_values, ids):
        speed = mach * sound
        if cl_at is None:
            cl = 2.0 * auw / (rho * speed * speed * sref)
        else:
            cl = cl_at_mach(cl_at[1], cl_at[0], mach)
        reynolds = rho * speed * cref / mu
        key = 'start_cl' if level_flight else 'target_cl'
        rows.extend(('', '[[points]]',
                     f'id = "{pid}"', f'mach = {mach:.12g}',
                     f'{key} = {cl:.12g}'))
        for default_key, default_value in (point_defaults or {}).items():
            rows.append(f'{default_key} = {default_value!r}')
        rows.extend(('[points.freestream]',
                     f'temperature = {temperature:.12g}',
                     f'reynolds_number = "{reynolds:.12g}"',
                     f'reynolds_length = {cref:.12g}'))

    def point_of(mach):
        return by_mach[_find(mach_values, mach, mach_tolerance)]

    if level_flight:
        ref = point_of(cl_at[0])
        for mach, pid in zip(mach_values, ids):
            if pid == ref:
                continue
            rows.extend(('', '[[constraints.linear]]', f'name = "level_{pid}"',
                         _terms((pid, 'lift', 1.0),
                                (ref, 'lift', -(cl_at[0] / mach) ** 2)),
                         'equals = 0.0', f'scaler = {level_scaler:g}'))
    if bracket is not None:
        ref = point_of(cl_at[0])
        for name, mach in (('bracket_low', cl_at[0] - bracket),
                           ('bracket_high', cl_at[0] + bracket)):
            rows.extend(('', '[[constraints.linear]]', f'name = "{name}"',
                         _terms((point_of(mach), 'drag', 1.0),
                                (ref, 'drag', -cl_at[0] / mach)),
                         'lower = 0.0', f'scaler = {bracket_scaler:g}'))
    return '\n'.join(rows) + '\n'


def _number(value, name, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f'[mission] {name} must be a number')
    value = float(value)
    if not math.isfinite(value) or (positive and value <= 0.0):
        raise ConfigError(f'[mission] {name} must be finite'
                          + (' and positive' if positive else ''))
    return value


def mission_tables(raw):
    """Resolve a run TOML's ``[mission]`` into points and linear rows."""
    if not isinstance(raw, dict):
        raise ConfigError('[mission] must be a table')
    allowed = {'auw', 'sref', 'cl_at', 'cref', 'altitude', 'mach',
               'level_flight', 'bracket', 'level_scaler', 'bracket_scaler',
               'atmosphere', 'point_defaults', 'mach_tolerance'}
    unknown = set(raw) - allowed
    if unknown:
        raise ConfigError(f'[mission] unknown keys: {sorted(unknown)}')
    for key in ('cref', 'altitude', 'mach'):
        if key not in raw:
            raise ConfigError(f'[mission] missing {key}')
    mach = raw['mach']
    if not isinstance(mach, list) or not mach:
        raise ConfigError('[mission] mach must be a nonempty array')
    cl_at = raw.get('cl_at')
    if cl_at is not None and (not isinstance(cl_at, list) or len(cl_at) != 2):
        raise ConfigError('[mission] cl_at must contain [reference Mach, lift coefficient]')
    level_flight = raw.get('level_flight', False)
    if not isinstance(level_flight, bool):
        raise ConfigError('[mission] level_flight must be true or false')
    defaults = raw.get('point_defaults', {})
    if not isinstance(defaults, dict):
        raise ConfigError('[mission.point_defaults] must be a table')
    allowed_defaults = {'aoa_init', 'aoa_lower', 'aoa_upper', 'weight'}
    if set(defaults) - allowed_defaults:
        raise ConfigError('[mission.point_defaults] unknown keys: '
                          f'{sorted(set(defaults) - allowed_defaults)}')
    defaults = {key: _number(value, f'point_defaults.{key}', key == 'weight')
                for key, value in defaults.items()}
    mach_values = [_number(value, 'mach', True) for value in mach]
    mach_tolerance = _number(raw.get('mach_tolerance', MACH_TOL),
                             'mach_tolerance', True)
    if len(mach_values) > 1 and 2.0 * mach_tolerance >= min(
            abs(a - b) for index, a in enumerate(mach_values)
            for b in mach_values[index + 1:]):
        raise ConfigError('[mission] mach_tolerance must be less than half '
                          'the separation between any two Mach values')
    try:
        fragment = points_fragment(
            _number(raw['auw'], 'auw', True) if 'auw' in raw else None,
            _number(raw['sref'], 'sref', True) if 'sref' in raw else None,
            _number(raw['cref'], 'cref', True),
            _number(raw['altitude'], 'altitude'),
            mach_values,
            cl_at=tuple(_number(value, 'cl_at', True) for value in cl_at)
                  if cl_at is not None else None,
            level_flight=level_flight,
            bracket=_number(raw['bracket'], 'bracket', True)
                    if 'bracket' in raw else None,
            atmosphere_constants=AtmosphereConstants.from_dict(
                raw.get('atmosphere', {})),
            level_scaler=_number(raw.get('level_scaler', LEVEL_SCALER),
                                 'level_scaler', True),
            bracket_scaler=_number(raw.get('bracket_scaler', BRACKET_SCALER),
                                   'bracket_scaler', True),
            point_defaults=defaults,
            mach_tolerance=mach_tolerance)
    except ValueError as exc:
        raise ConfigError(f'[mission] {exc}') from None
    return tomllib.loads(fragment)
