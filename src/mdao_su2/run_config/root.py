"""Run configuration schema: frozen dataclasses, defaults, derived values."""

import math
import os
from dataclasses import asdict, dataclass, field, fields
from functools import cached_property
from typing import Any, Dict, List, Optional, Tuple

from .errors import ConfigError
from .physics import GasConfig, TransitionConfig, TurbulenceConfig

SCHEMA_VERSION = 1


# ---------------------------------------------------------------- helpers

def check_keys(cls, data, ctx):
    known = {f.name for f in fields(cls)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ConfigError(
            f'{ctx}: unknown key(s) {unknown}. Allowed: {sorted(known)}')


def _sub(data, key):
    """Nested table defaulting to empty; rejects scalar where table belongs."""
    val = data.get(key, {})
    if val is None:
        return {}
    if not isinstance(val, dict):
        raise ConfigError(f'{key!r} must be a table, got {type(val).__name__}')
    return val


def _names(value, ctx):
    """Normalize a marker name (str or list of str) to a tuple. Names are exact — no stripping/casing."""
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,) if value else ()
    if isinstance(value, (list, tuple)):
        out = []
        for v in value:
            if not isinstance(v, str) or not v:
                raise ConfigError(f'{ctx}: marker names must be non-empty '
                                  f'strings, got {v!r}')
            out.append(v)
        dupes = sorted({n for n in out if out.count(n) > 1})
        if dupes:
            raise ConfigError(f'{ctx}: marker(s) {dupes} listed twice')
        return tuple(out)
    raise ConfigError(f'{ctx}: expected a marker name or a list of them, '
                      f'got {type(value).__name__}')


def _floats(seq, ctx, n=None):
    try:
        out = [float(v) for v in seq]
    except (TypeError, ValueError):
        raise ConfigError(f'{ctx}: expected a list of numbers, got {seq!r}')
    if n is not None and len(out) != n:
        raise ConfigError(f'{ctx}: expected {n} numbers, got {len(out)}')
    return out


# ------------------------------------------------------------------ case

#: Named in errors for retired configuration keys.
MIGRATE_HINT = 'use the replacement keys named above'

@dataclass(frozen=True)
class FreestreamConfig:
    """Free-stream cfg block. ``reynolds_number`` is a string so the cfg keeps its exponent form."""
    temperature: float = 288.15
    reynolds_number: Optional[str] = None   # required for turbulent physics
    reynolds_length: float = 1.0
    #: FREESTREAM_PRESSURE. 0.0 = omit (SU2 default); inviscid cases must set explicitly.
    pressure: float = 0.0
    #: REF_DIMENSIONALIZATION. '' = omit (SU2 default DIMENSIONAL).
    dimensionalization: str = ''

    DIMENSIONALIZATIONS = ('', 'DIMENSIONAL', 'FREESTREAM_PRESS_EQ_ONE',
                           'FREESTREAM_VEL_EQ_MACH', 'FREESTREAM_VEL_EQ_ONE')

    @classmethod
    def from_dict(cls, data, ctx='[case.freestream]'):
        check_keys(cls, data, ctx)
        d = dict(data)
        for k in ('temperature', 'reynolds_length', 'pressure'):
            if k in d:
                d[k] = float(d[k])
        if 'reynolds_number' in d:
            d['reynolds_number'] = str(d['reynolds_number'])
        if d.get('dimensionalization', '') not in cls.DIMENSIONALIZATIONS:
            raise ConfigError(
                f'{ctx} dimensionalization='
                f'{d["dimensionalization"]!r} must be one of '
                f'{[x for x in cls.DIMENSIONALIZATIONS if x]} (or omitted)')
        return cls(**d)


@dataclass(frozen=True)
class ReferenceConfig:
    """REF_* normalisation quantities for C_L/C_D."""
    origin_moment: Tuple[float, float, float] = (0.25, 0.0, 0.0)
    length: float = 1.0
    #: REF_AREA. 0.0 = sentinel: SU2 auto-computes planform area from monitored markers.
    area: float = 1.0
    #: Figure axes (x, y or z); unset means the ndim default of :meth:`axes`.
    span_axis: Optional[str] = None
    lift_axis: Optional[str] = None

    #: Written into the cfg verbatim when ``area`` holds it.
    AUTO_AREA = 0.0

    def axes(self, ndim):
        """``(span_axis, lift_axis)``: the stated axes, else z/y in 2D and y/z in 3D."""
        span, lift = ('z', 'y') if ndim == 2 else ('y', 'z')
        return self.span_axis or span, self.lift_axis or lift

    @classmethod
    def from_dict(cls, data):
        check_keys(cls, data, '[case.reference]')
        d = dict(data)
        for k in ('span_axis', 'lift_axis'):
            if k in d and d[k] not in ('x', 'y', 'z'):
                raise ConfigError(f'[case.reference] {k} must be x, y or z')
        if d.get('span_axis') and d.get('span_axis') == d.get('lift_axis'):
            raise ConfigError('[case.reference] span_axis and lift_axis must differ')
        if 'origin_moment' in d:
            x, y, z = _floats(d['origin_moment'],
                              '[case.reference] origin_moment', 3)
            d['origin_moment'] = (x, y, z)
        for k in ('length', 'area'):
            if k in d:
                d[k] = float(d[k])
        return cls(**d)

    @property
    def area_is_auto(self):
        return self.area == self.AUTO_AREA


@dataclass(frozen=True)
class PhysicsConfig:
    """Flow equations and the typed SU2 physics profile."""
    kind: str = 'rans_sa'              # | 'euler'
    turbulence: TurbulenceConfig = field(default_factory=TurbulenceConfig)
    transition: TransitionConfig = field(default_factory=TransitionConfig)
    gas: GasConfig = field(default_factory=GasConfig)
    turbulence_intensity: float = None
    turb2lam_viscosity_ratio: float = None

    KINDS = ('rans_sa', 'euler')

    @classmethod
    def from_dict(cls, data):
        check_keys(cls, data, '[case.physics]')
        d = dict(data)
        if 'kind' in d and d['kind'] not in cls.KINDS:
            raise ConfigError(f'[case.physics] kind={d["kind"]!r} must be one '
                              f'of {list(cls.KINDS)}')
        d['turbulence'] = TurbulenceConfig.from_dict(_sub(d, 'turbulence'))
        d['transition'] = TransitionConfig.from_dict(_sub(d, 'transition'))
        d['gas'] = GasConfig.from_dict(_sub(d, 'gas'))
        for key in ('turbulence_intensity', 'turb2lam_viscosity_ratio'):
            if key in d:
                d[key] = float(d[key])
        return cls(**d)

    @property
    def inviscid(self):
        return self.kind == 'euler'

    @property
    def su2_solver(self):
        return 'EULER' if self.inviscid else 'RANS'

    @property
    def turbulent(self):
        """Whether turbulence keys apply."""
        return not self.inviscid


@dataclass(frozen=True)
class BoundaryConfig:
    """One ``[[case.boundary]]`` entry: markers, SU2 bc family, its parameters, roles."""
    markers: Tuple[str, ...]
    bc: str
    roles: Tuple[str, ...] = ()
    params: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data, idx):
        from ..solver.su2_options import BC_FAMILIES, ROLE_KEYS
        from ..solver.su2_options_v85 import ENUMS
        ctx = f'[[case.boundary]] #{idx}'
        d = dict(data)
        bc = d.pop('bc', None)
        family = BC_FAMILIES.get(bc)
        if family is None:
            raise ConfigError(f'{ctx}: bc={bc!r} must be one of {sorted(BC_FAMILIES)}')
        markers = _names(d.pop('markers', None), f'{ctx} markers')
        if not markers:
            raise ConfigError(f'{ctx}: markers must name at least one marker')
        if family.paired and len(markers) != 2:
            raise ConfigError(f'{ctx}: bc={bc!r} takes exactly two markers '
                              f'(inlet, outlet), got {list(markers)}')
        roles = tuple(d.pop('roles', ()))
        bad = sorted(set(roles) - set(ROLE_KEYS))
        if bad or len(set(roles)) != len(roles):
            raise ConfigError(f'{ctx}: roles {list(roles)} must be distinct names '
                              f'from {list(ROLE_KEYS)}')
        if 'design' in roles and not family.wall:
            raise ConfigError(f'{ctx}: role "design" needs a wall bc, not {bc!r}')
        if 'deform_sym' in roles and bc != 'sym':
            raise ConfigError(f'{ctx}: role "deform_sym" needs bc = "sym"; '
                              f'MARKER_DEFORM_MESH_SYM_PLANE takes its normal from '
                              f'a real plane')
        params = {}
        for p in family.params:
            if p.name not in d:
                raise ConfigError(f'{ctx}: bc={bc!r} needs parameter {p.name!r}')
            v = d.pop(p.name)
            if p.enum:
                if v not in ENUMS[p.enum]:
                    raise ConfigError(f'{ctx}: {p.name}={v!r} must be one of '
                                      f'{list(ENUMS[p.enum])}')
            elif p.n > 1:
                v = tuple(_floats(v, f'{ctx} {p.name}', p.n))
            else:
                v = float(v)
            params[p.name] = v
        if d:
            raise ConfigError(f'{ctx}: unknown key(s) {sorted(d)} for bc={bc!r}. '
                              f'Allowed: markers, bc, roles, '
                              f'{[p.name for p in family.params]}')
        return cls(markers=markers, bc=bc, roles=roles, params=params)


@dataclass(frozen=True)
class CaseConfig:
    """Case paths. ``root`` is repo-relative; sub-paths are root-relative.

    ``name``, ``root`` and ``mesh`` have no default: every TOML names its case
    must be supplied by the case. The empty strings only let the dataclass be built bare.
    """
    name: str = ''
    root: str = ''
    ndim: int = 2
    mesh: str = ''
    freestream: FreestreamConfig = field(default_factory=FreestreamConfig)
    reference: ReferenceConfig = field(default_factory=ReferenceConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    boundaries: Tuple[BoundaryConfig, ...] = ()

    SUBTABLES = ('freestream', 'reference', 'physics', 'boundary')
    REQUIRED = ('name', 'root', 'mesh')

    @classmethod
    def from_dict(cls, data):
        retired = sorted({'markers', 'wall_marker'} & set(data))
        if retired:
            raise ConfigError(
                f'[case] {retired} are no longer accepted: declare every marker '
                f'in [[case.boundary]] ({MIGRATE_HINT})')
        plain = {k: v for k, v in data.items() if k not in cls.SUBTABLES}
        check_keys(cls, plain, '[case]')
        physics = PhysicsConfig.from_dict(_sub(data, 'physics'))
        raw = data.get('boundary')
        if raw is None:
            raise ConfigError('[case] needs [[case.boundary]] entries naming '
                              'every mesh marker and its bc')
        if not isinstance(raw, list):
            raise ConfigError('[[case.boundary]] must be an array of tables')
        boundaries = tuple(BoundaryConfig.from_dict(b, i) for i, b in enumerate(raw))
        seen = {}
        for i, b in enumerate(boundaries):
            for name in b.markers:
                if name in seen:
                    raise ConfigError(f'[[case.boundary]] marker {name!r} is in '
                                      f'entries #{seen[name]} and #{i}; a marker '
                                      f'has one boundary condition')
                seen[name] = i
        unset = [k for k in cls.REQUIRED if not plain.get(k)]
        if unset:
            raise ConfigError(f'[case] {unset} must be set: no case is assumed '
                              f'(the case must specify it)')
        return cls(freestream=FreestreamConfig.from_dict(_sub(data, 'freestream')),
                   reference=ReferenceConfig.from_dict(_sub(data, 'reference')),
                   physics=physics, boundaries=boundaries, **plain)

    def markers_with(self, role):
        """Marker names carrying ``role``, in declaration order."""
        return tuple(n for b in self.boundaries if role in b.roles for n in b.markers)

    def markers_of(self, bc):
        """Marker names with boundary condition ``bc``, in declaration order."""
        return tuple(n for b in self.boundaries if b.bc == bc for n in b.markers)

    def all_markers(self):
        return tuple(n for b in self.boundaries for n in b.markers)

    @property
    def design_markers(self):
        """The wall seam surface: every marker with role ``design``."""
        return self.markers_with('design')

    @property
    def wall_single(self):
        """The sole design marker. Multi-marker walls run through MultiMarkerSU2AeroComp."""
        walls = self.design_markers
        if len(walls) != 1:
            raise ConfigError(
                f'[[case.boundary]] names {len(walls)} design markers {list(walls)}, '
                f'but this call site needs exactly one. The frozen single-marker '
                f'SU2AeroComp cannot serve a multi-marker wall; a case with several '
                f'design markers must run through MultiMarkerSU2AeroComp.')
        return walls[0]

    @property
    def is_multi_marker_wall(self):
        return len(self.design_markers) > 1


@dataclass(frozen=True)
class GeometryConfig:
    parameterization: str = 'ffd'
    #: Key into ``ffd_files``; required when an FFD shape block is on.
    ffd: str = ''
    zspan: float = 0.5                 # MUST match geometry/ffd_lattice.py ZSPAN
    ffd_files: Dict[str, str] = field(default_factory=dict)
    #: FFD spline basis: 'pygeo_default' (cubic), 'bezier' (order=nCtl), or explicit int kmax.
    ffd_order: Any = 'pygeo_default'
    #: CST-only: Selig-format airfoil .dat (case-root-relative).
    dat_file: str = ''
    #: CST-only: number of CST coefficients per surface (int or [n_upper, n_lower]).
    n_cst: Any = 8
    #: Hold wall nodes that lie on a bc = "sym" plane on that plane:
    #: their normal displacement and its derivative rows are zeroed.
    pin_symmetry: bool = False

    FFD_ORDERS = ('pygeo_default', 'bezier')

    @classmethod
    def from_dict(cls, data):
        check_keys(cls, data, '[geometry]')
        d = dict(data)
        d['ffd_files'] = dict(_sub(data, 'ffd_files'))
        order = d.get('ffd_order', 'pygeo_default')
        if isinstance(order, str):
            if order not in cls.FFD_ORDERS:
                raise ConfigError(
                    f'[geometry] ffd_order={order!r} must be one of '
                    f'{list(cls.FFD_ORDERS)} or an integer kmax')
        else:
            d['ffd_order'] = int(order)
            if d['ffd_order'] < 2:
                raise ConfigError('[geometry] ffd_order (kmax) must be >= 2')
        if 'n_cst' in d:
            n = d['n_cst']
            if isinstance(n, list):
                if len(n) != 2:
                    raise ConfigError(
                        '[geometry] n_cst must be an integer or a '
                        f'[n_upper, n_lower] pair, got {n!r}')
                d['n_cst'] = [int(v) for v in n]
            else:
                d['n_cst'] = int(n)
        if 'pin_symmetry' in d:
            if not isinstance(d['pin_symmetry'], bool):
                raise ConfigError(f'[geometry] pin_symmetry must be true or false, '
                                  f'got {d["pin_symmetry"]!r}')
        return cls(**d)

    def dvgeo_options(self, ffd_path):
        """kwargs for ``DVGeometry`` — ``{}`` or ``{'kmax': n}``."""
        if self.ffd_order == 'pygeo_default':
            return {}
        if self.ffd_order == 'bezier':
            with open(ffd_path) as fh:
                tokens = fh.read(4096).split()
            return {'kmax': max(int(t) for t in tokens[1:4])}
        return {'kmax': int(self.ffd_order)}

    @property
    def ffd_file(self):
        try:
            return self.ffd_files[self.ffd]
        except KeyError:
            raise ConfigError(
                f'[geometry] ffd={self.ffd!r} has no entry in '
                f'[geometry.ffd_files]; known: {sorted(self.ffd_files)}')


# ---------------------------------------------------------------- points

@dataclass(frozen=True)
class PointConfig:
    """One operating point. ``weight`` is relative — objective divides by sum."""
    id: str
    mach: float
    weight: float = 1.0
    # DEGREES, always. 0 is a placeholder: a point that sets none of the
    # three has alpha fixed at 0.
    aoa_init: float = 0.0
    aoa_lower: float = 0.0
    aoa_upper: float = 0.0
    aoa_scaler: float = 1.0
    freestream: Optional[Dict[str, Any]] = None
    target_cl: Optional[float] = None
    #: Lift coefficient the trim aims for at this point; not a constraint.
    start_cl: Optional[float] = None
    alpha: str = 'design'
    responses: Tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, data, idx):
        ctx = f'[[points]] #{idx}'
        check_keys(cls, data, ctx)
        d = dict(data)
        raw_freestream = d.pop('freestream', None)
        for req in ('id', 'mach'):
            if req not in d:
                raise ConfigError(f'{ctx}: missing required key {req!r}')
        d['id'] = str(d['id'])
        for k in ('mach', 'weight', 'aoa_init', 'aoa_lower', 'aoa_upper',
                  'aoa_scaler', 'target_cl', 'start_cl'):
            if k in d:
                d[k] = float(d[k])
        if raw_freestream is not None:
            if not isinstance(raw_freestream, dict):
                raise ConfigError(f'{ctx}.freestream must be a table')
            parsed = FreestreamConfig.from_dict(raw_freestream, f'{ctx}.freestream')
            d['freestream'] = {key: getattr(parsed, key) for key in raw_freestream}
        if 'responses' in d:
            if not isinstance(d['responses'], (list, tuple)):
                raise ConfigError(f'{ctx}.responses must be a list of catalog names')
            d['responses'] = tuple(str(name).lower() for name in d['responses'])
        return cls(**d)


# ------------------------------------------------------- design variables


def _unit_vector(val, ctx='[dv.shape] direction'):
    """Resolve 'x'/'y'/'z' or [x,y,z] to a unit 3-vector tuple."""
    import math as _math
    directions_map = {'x': (1.0, 0.0, 0.0), 'y': (0.0, 1.0, 0.0),
                      'z': (0.0, 0.0, 1.0)}
    if isinstance(val, str):
        if val not in directions_map:
            raise ConfigError(
                f'{ctx}={val!r} must be one of '
                f'{sorted(directions_map)} or an explicit [x, y, z]')
        return directions_map[val]
    vec = _floats(val, ctx, 3)
    norm = _math.sqrt(sum(v * v for v in vec))
    if norm == 0.0:
        raise ConfigError(f'{ctx} must not be the zero vector')
    return tuple(v / norm for v in vec)


SHAPE_PATTERNS = ('control_point_2d', 'bspline', 'ffd3d', 'thickness2d')


@dataclass(frozen=True)
class ShapeDVConfig:
    enabled: bool = False              # toggle dv.shape (run_config/toggles.py)
    pattern: str = 'control_point_2d'
    bound: float = 0.05                # singlepoint.py:63
    #: Lower bound 0 instead of -bound: every DV moves the surface outward when
    #: positive (thickness2d, or ffd3d on the upper row only). ADODG Case 1.
    outward_only: bool = False
    scaler: Any = 'auto'               # "auto" -> 1/bound (singlepoint.py:108)
    n_basis: int = 0                   # bspline only: 0 -> auto (match control_point_2d count)
    degree: int = 3                    # bspline only: cubic by default
    n_le: int = 0       # extra local DVs near LE; 0 = off
    free_te: bool = False  # free TE bodily y; see geometry/ffd._le_te_shapes

    direction: Any = 'y'  # 'x'|'y'|'z' or explicit 3-vector
    link_k: bool = True   # link k=0/k=1 planes (2D pattern); False for 3D
    #: Lattice indices held fixed per axis. Stand-in for SU2's FFD_CONTINUITY.
    freeze_i: Tuple[int, ...] = ()
    freeze_j: Tuple[int, ...] = ()
    freeze_k: Tuple[int, ...] = ()
    #: Individual fixed points as zero-based (i,j,k) triples. Use freeze_points_from_figure for 1-based.
    freeze_points: Tuple[Tuple[int, int, int], ...] = ()

    directions: Tuple[Any, ...] = ()  # per-direction overrides; () → (direction,)
    bounds: Tuple[float, ...] = ()     # per-direction step bounds; () → (bound,) broadcast

    @classmethod
    def from_dict(cls, data):
        check_keys(cls, data, '[dv.shape]')
        d = dict(data)
        if 'bound' in d:
            d['bound'] = float(d['bound'])
        for k in ('n_basis', 'degree', 'n_le'):
            if k in d:
                d[k] = int(d[k])
        if 'link_k' in d:
            d['link_k'] = bool(d['link_k'])
        if 'outward_only' in d:
            d['outward_only'] = bool(d['outward_only'])
        for k in ('freeze_i', 'freeze_j', 'freeze_k'):
            if k in d:
                try:
                    d[k] = tuple(int(v) for v in d[k])
                except (TypeError, ValueError):
                    raise ConfigError(f'[dv.shape] {k} must be a list of '
                                      f'lattice indices, got {d[k]!r}')
        if 'freeze_points' in d:
            pts = []
            for row in d['freeze_points']:
                try:
                    i, j, k = (int(v) for v in row)
                except (TypeError, ValueError):
                    raise ConfigError(
                        f'[dv.shape] freeze_points rows must be (i, j, k) '
                        f'triples, got {row!r}')
                pts.append((i, j, k))
            d['freeze_points'] = tuple(pts)
        if 'directions' in d:
            try:
                d['directions'] = tuple(d['directions'])
            except TypeError:
                raise ConfigError(f'[dv.shape] directions must be a list, '
                                  f'got {d["directions"]!r}')
        if 'bounds' in d:
            try:
                d['bounds'] = tuple(float(v) for v in d['bounds'])
            except (TypeError, ValueError):
                raise ConfigError(f'[dv.shape] bounds must be a list of '
                                  f'numbers, got {d["bounds"]!r}')
        cfg = cls(**d)
        if not cfg.enabled:             # an off block is not range-checked
            return cfg
        cfg.resolved_direction          # fail at load, not at model build
        cfg.resolved_directions         # fail at load for multi-direction too
        cfg.resolved_bounds
        cfg.resolved_scalers
        return cfg

    @property
    def resolved_direction(self):
        """The movement direction as a unit 3-vector (plain tuple)."""
        return _unit_vector(self.direction, '[dv.shape] direction')

    @property
    def resolved_directions(self):
        """All movement directions as a tuple of unit 3-vectors."""
        if not self.directions:
            return (self.resolved_direction,)
        return tuple(_unit_vector(d, f'[dv.shape] directions[{i}]')
                     for i, d in enumerate(self.directions))

    @property
    def resolved_bounds(self):
        """Per-direction step bounds; broadcasts scalar ``bound`` when empty."""
        dirs = self.resolved_directions
        if not self.bounds:
            return (self.bound,) * len(dirs)
        return tuple(self.bounds)

    @property
    def resolved_scalers(self):
        """Per-direction scalers; ``'auto'`` → ``1/bound_i`` per direction."""
        if isinstance(self.scaler, str):
            if self.scaler != 'auto':
                raise ConfigError(
                    f'[dv.shape] scaler must be a number or "auto", '
                    f'got {self.scaler!r}')
            return tuple(1.0 / b for b in self.resolved_bounds)
        s = float(self.scaler)
        return (s,) * len(self.resolved_directions)

    @property
    def resolved_scaler(self):
        if isinstance(self.scaler, str):
            if self.scaler != 'auto':
                raise ConfigError(
                    f'[dv.shape] scaler must be a number or "auto", '
                    f'got {self.scaler!r}')
            return 1.0 / self.bound
        return float(self.scaler)


@dataclass(frozen=True)
class HicksHenneDVConfig:
    """Hicks--Henne bump amplitudes configured under ``[dv.shape]``."""
    enabled: bool = False              # toggle dv.shape (run_config/toggles.py)
    n_dv: int = 20
    distribution: str = 'even'
    bump_width: int = 3
    bound: float = 0.05
    scaler: Any = 'auto'

    DISTRIBUTIONS = ('even', '1-cos')

    @classmethod
    def from_dict(cls, data):
        check_keys(cls, data, '[dv.shape]')
        d = dict(data)
        for key in ('n_dv', 'bump_width'):
            if key in d:
                d[key] = int(d[key])
        if 'bound' in d:
            d['bound'] = float(d['bound'])
        cfg = cls(**d)
        if cfg.enabled and cfg.bound > 0.0:
            cfg.resolved_scaler
        return cfg

    @property
    def resolved_scaler(self):
        if isinstance(self.scaler, str):
            if self.scaler != 'auto':
                raise ConfigError(
                    f'[dv.shape] scaler must be a number or "auto", '
                    f'got {self.scaler!r}')
            return 1.0 / self.bound
        return float(self.scaler)


def freeze_points_from_figure(triples):
    """Convert 1-based (i,j,k) lattice labels to 0-based for freeze_points."""
    return tuple((int(i) - 1, int(j) - 1, int(k) - 1) for i, j, k in triples)


@dataclass(frozen=True)
class AlphaDVConfig:
    per_point: bool = True

    @classmethod
    def from_dict(cls, data):
        check_keys(cls, data, '[dv.alpha]')
        return cls(**data)


@dataclass(frozen=True)
class TwistDVConfig:
    """``[dv.twist]``: TE twist of the wing FFD sections, in degrees, nose-up positive.

    Toggle ``dv.twist``. Off when absent; when on, every field
    below except ``enabled`` is required (``run_config/toggles.py``).
    """
    enabled: bool = False
    n: Optional[int] = None                 # FFD spanwise sections (reference-axis points)
    lower_deg: Optional[float] = None
    upper_deg: Optional[float] = None
    scaler: Optional[float] = None
    include_root: Optional[bool] = None     # twist the y = 0 section too (ADODG p. 23)
    axis_x_fraction: Optional[float] = None  # pyGeo xFraction of the TE axis in the box
    init_deg: Optional[float] = None

    @classmethod
    def from_dict(cls, data):
        check_keys(cls, data, '[dv.twist]')
        d = dict(data)
        if 'n' in d:
            d['n'] = int(d['n'])
        for k in ('lower_deg', 'upper_deg', 'scaler', 'axis_x_fraction', 'init_deg'):
            if k in d:
                d[k] = float(d[k])
        for k in ('enabled', 'include_root'):
            if k in d and not isinstance(d[k], bool):
                raise ConfigError(f'[dv.twist] {k} must be true or false, got {d[k]!r}')
        return cls(**d)

    @property
    def n_dv(self):
        """Twist DVs: one per section, less the root when it is held."""
        return self.n if self.include_root else self.n - 1


def shape_dv_bounds(shape, parameterization, n_shape, counts=None):
    """Lower and upper bounds of the shape array, scalar or per DV.

    ``shape`` is the ``[dv.shape]`` table; ``counts`` are the per-direction DV
    counts of the embedded geometry. A multi-direction pattern needs them and
    gives None without.
    """
    if parameterization == 'hicks_henne':
        lower = 0.0 if n_shape == shape.n_dv else -shape.bound
        return lower, shape.bound
    if counts is None and len(shape.resolved_directions) > 1:
        return None
    if counts is None or len(counts) == 1:
        # outward_only: every DV of the pattern moves the surface outward when
        # positive (validate.py), so the lower bound is 0 (paper Case 1).
        lower = 0.0 if shape.outward_only else -shape.bound
        return lower, shape.bound
    import numpy as np
    bounds = np.repeat(np.asarray(shape.resolved_bounds), counts)
    return -bounds, bounds


@dataclass(frozen=True)
class DVConfig:
    shape: Any = field(default_factory=ShapeDVConfig)
    twist: TwistDVConfig = field(default_factory=TwistDVConfig)
    alpha: AlphaDVConfig = field(default_factory=AlphaDVConfig)
    #: A recorded design to start from: a run id or ``<run>/evaluations/<name>``
    #: under the output root (``run_output/design_start.py``).
    start: str = ''

    @classmethod
    def from_dict(cls, data, parameterization='ffd'):
        check_keys(cls, data, '[dv]')
        shape_cls = (HicksHenneDVConfig
                     if parameterization == 'hicks_henne'
                     else ShapeDVConfig)
        start = data.get('start', '')
        if not isinstance(start, str):
            raise ConfigError(f'[dv] start must be a run id or an evaluation folder, '
                              f'got {start!r}')
        return cls(shape=shape_cls.from_dict(_sub(data, 'shape')),
                   twist=TwistDVConfig.from_dict(_sub(data, 'twist')),
                   alpha=AlphaDVConfig.from_dict(_sub(data, 'alpha')),
                   start=start)


# ------------------------------------------------- objective + constraints

#: Keys a legacy [objective] table used; the loader names the migration tool.
_RETIRED_OBJECTIVE_KEYS = ('kind', 'omega_d', 'omega_l', 'cd_target', 'cl_target')


@dataclass(frozen=True)
class ObjectiveConfig:
    """``[objective]``: an explicit expression and scaler."""
    expression: Any = None
    scaler: Any = None
    parameters: Dict[str, float] = field(default_factory=dict)

    @property
    def effective_scaler(self):
        """Scaler for add_objective; validate.py guarantees it is set."""
        return float(self.scaler)

    @classmethod
    def from_dict(cls, data):
        retired = sorted(set(data) & set(_RETIRED_OBJECTIVE_KEYS))
        if retired:
            raise ConfigError(
                f'[objective] {retired} are no longer accepted: state '
                f'expression, scaler and [objective.parameters] instead '
                f'({MIGRATE_HINT})')
        check_keys(cls, data, '[objective]')
        d = dict(data)
        if d.get('expression') is not None:
            d['expression'] = str(d['expression'])
        if d.get('scaler') is not None:
            d['scaler'] = float(d['scaler'])
        params = d.get('parameters') or {}
        if not isinstance(params, dict):
            raise ConfigError('[objective.parameters] must be a table')
        try:
            d['parameters'] = {str(k): float(v) for k, v in params.items()}
        except (TypeError, ValueError) as exc:
            raise ConfigError(f'[objective.parameters]: {exc}') from None
        return cls(**d)


#: Lift constraint forms (D1). Defined once and read by the accessors below.
LIFT_KINDS = ('equals', 'min', 'range')
MOMENT_KINDS = ('equals', 'min', 'max', 'range')
MOMENT_AXES = ('x', 'y', 'z')


def _point_ids(value, ctx):
    """``points`` of a constraint table: a list of point ids, or None when absent."""
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or not all(isinstance(v, str) for v in value):
        raise ConfigError(f'{ctx} points must be a list of point ids, got {value!r}')
    return tuple(value)


@dataclass(frozen=True)
class LiftConstraintConfig:
    """``[constraints.lift]``: off unless the TOML enables it with a kind and bound.

    No target is assumed: ``kind`` and the bound(s) it uses must be
    written in the TOML. ``None`` means unset; ``0.0`` is a real value.
    """
    enabled: bool = False
    kind: Optional[str] = None         # 'equals' | 'min' | 'range'  (D1)
    equals: Optional[float] = None     # kind='equals'
    lower: Optional[float] = None      # kind='min'/'range'
    upper: Optional[float] = None      # kind='range'
    scaler: float = 1.0
    #: Point ids the constraint applies to; None (absent) means every point.
    points: Optional[Tuple[str, ...]] = None

    def applies(self, point_id):
        """True when the constraint is on and applies to ``point_id``."""
        return self.enabled and (self.points is None or point_id in self.points)

    def _need(self, key):
        val = getattr(self, key)
        if val is None:
            raise ConfigError(
                f'[constraints.lift] kind={self.kind!r} needs {key}; the lift '
                f'constraint has no default target')
        return float(val)

    @property
    def trim_target(self):
        """The C_L target for trimming: ``equals`` or ``lower`` depending on kind."""
        if self.kind == 'equals':
            return self._need('equals')
        if self.kind in ('min', 'range'):
            return self._need('lower')
        raise ConfigError(
            f'[constraints.lift] kind={self.kind!r} must be one of '
            f'{list(LIFT_KINDS)}; an enabled lift constraint states it')

    @property
    def constraint_kwargs(self):
        """Keyword dict for ``add_constraint``. Raises ConfigError if bounds unset."""
        self.trim_target                # unset kind or active bound fails here
        if self.kind == 'equals':
            kw = {'equals': self._need('equals')}
        elif self.kind == 'min':
            kw = {'lower': self._need('lower')}
        else:
            kw = {'lower': self._need('lower'), 'upper': self._need('upper')}
        kw['scaler'] = float(self.scaler)
        return kw

    @classmethod
    def from_dict(cls, data):
        check_keys(cls, data, '[constraints.lift]')
        d = dict(data)
        d['points'] = _point_ids(d.get('points'), '[constraints.lift]')
        if d.get('kind') is not None:
            d['kind'] = str(d['kind'])
        for k in ('equals', 'lower', 'upper', 'scaler'):
            if k in d and d[k] is not None:
                d[k] = float(d[k])
        return cls(**d)


@dataclass(frozen=True)
class MomentConstraintConfig:
    """Moment constraint: axis, pivot, and bound mode (D11-D13)."""
    enabled: bool = False
    kind: str = 'min'
    equals: float = 0.0
    lower: float = 0.0
    upper: float = 0.0
    scaler: float = 1.0
    axis: str = 'z'
    pivot: Any = None
    #: Point ids the constraint applies to; None (absent) means every point.
    points: Optional[Tuple[str, ...]] = None
    _bound_keys_present: frozenset = field(default_factory=frozenset,
                                            repr=False, compare=False)

    def applies(self, point_id):
        """True when the constraint is on and applies to ``point_id``."""
        return self.enabled and (self.points is None or point_id in self.points)

    @property
    def objective_key(self):
        return f'MOMENT_{self.axis.upper()}'

    def resolved_pivot(self, case_origin):
        return tuple(case_origin if self.pivot is None else self.pivot)

    @property
    def constraint_kwargs(self):
        """Keyword dict for ``add_constraint``, mirroring the lift constraint."""
        present = self._bound_keys_present
        if self.kind == 'equals':
            if self.equals == 0.0 and 'equals' not in present:
                raise ConfigError(
                    "[constraints.moment] kind='equals' needs equals to be "
                    "set explicitly (C_M = 0 is valid, but must appear in the "
                    "TOML so 'unset' is distinguishable from 'zero')")
            kw = {'equals': float(self.equals)}
        elif self.kind == 'min':
            if self.lower == 0.0 and 'lower' not in present:
                raise ConfigError(
                    "[constraints.moment] kind='min' needs lower to be set")
            kw = {'lower': float(self.lower)}
        elif self.kind == 'max':
            if self.upper == 0.0 and 'upper' not in present:
                raise ConfigError(
                    "[constraints.moment] kind='max' needs upper to be set")
            kw = {'upper': float(self.upper)}
        else:
            if self.lower == 0.0 and 'lower' not in present:
                raise ConfigError(
                    "[constraints.moment] kind='range' needs lower to be set")
            if self.upper == 0.0 and 'upper' not in present:
                raise ConfigError(
                    "[constraints.moment] kind='range' needs upper to be set")
            kw = {'lower': float(self.lower), 'upper': float(self.upper)}
        kw['scaler'] = float(self.scaler)
        return kw

    @classmethod
    def from_dict(cls, data):
        check_keys(cls, data, '[constraints.moment]')
        d = dict(data)
        d['points'] = _point_ids(d.get('points'), '[constraints.moment]')
        bound_keys = frozenset(k for k in ('equals', 'lower', 'upper') if k in d)
        for key in ('kind', 'axis'):
            if key in d:
                d[key] = str(d[key]).lower()
        for key in ('equals', 'lower', 'upper', 'scaler'):
            if key in d and d[key] is not None:
                d[key] = float(d[key])
        if d.get('pivot') is not None:
            d['pivot'] = tuple(_floats(
                d['pivot'], '[constraints.moment] pivot', 3))
        d.pop('_bound_keys_present', None)
        return cls(_bound_keys_present=bound_keys, **d)


@dataclass(frozen=True)
class VolumeConstraintConfig:
    enabled: bool = False
    le_x: float = 0.01                 # geometry/ffd.py:96
    te_x: float = 0.99
    nSpan: int = 2
    nChord: int = 10
    lower: float = 1.0

    @classmethod
    def from_dict(cls, data):
        check_keys(cls, data, '[constraints.volume]')
        return cls(**data)


@dataclass(frozen=True)
class LERadiusConstraintConfig:
    enabled: bool = False
    le_x: float = 0.01                 # geometry/ffd.py:100
    nSpan: int = 2
    lower: float = 0.8
    axis: Tuple[float, float, float] = (0.0, 1.0, 0.0)
    chord_dir: Tuple[float, float, float] = (-1.0, 0.0, 0.0)

    @classmethod
    def from_dict(cls, data):
        check_keys(cls, data, '[constraints.le_radius]')
        d = dict(data)
        for k in ('axis', 'chord_dir'):
            if k in d:
                d[k] = tuple(_floats(d[k], f'[constraints.le_radius] {k}', 3))
        return cls(**d)


@dataclass(frozen=True)
class CurvatureConstraintConfig:
    """Surface-regularity rail. scaler critical: curvature gradient ~1e4× others."""
    enabled: bool = False
    start_x: float = 0.01              # curvature is NOT evaluated at the two
    end_x: float = 0.16                # end points, so the window overhangs
    n_pts: int = 17                    # spacing 0.009375: > mesh dx (0.0045),
    upper: float = 3.0                 # < FFD station interval (~0.02)
    scaler: float = 1.2e-3             # 1/815, the measured gradient norm
    curvature_type: str = 'mean'
    ks_coeff: float = 1.0

    @classmethod
    def from_dict(cls, data):
        check_keys(cls, data, '[constraints.curvature]')
        d = dict(data)
        for k in ('start_x', 'end_x', 'upper', 'scaler', 'ks_coeff'):
            if k in d:
                d[k] = float(d[k])
        if 'n_pts' in d:
            d['n_pts'] = int(d['n_pts'])
        return cls(**d)


AXES_CHORD_ZSPAN = 'chord_zspan'  # 2D: station = chord fraction, thickness in y
AXES_SPAN_Y = 'span_y'            # 3D: station = absolute y plane, thickness in z

_THICKNESS_AXES = (AXES_CHORD_ZSPAN, AXES_SPAN_Y)


@dataclass(frozen=True)
class ThicknessConstraintConfig:
    """Thickness rails. ``axes`` is explicit (not inferred from ndim) because the two modes measure in different directions."""
    enabled: bool = False
    #: "min": each station value is a lower bound; "max": an upper bound.
    kind: str = 'min'
    axes: str = AXES_CHORD_ZSPAN
    nSpan: int = 2
    nChord: int = 2
    scaled: bool = False
    te_offset: float = 0.05
    #: span_y only: chordwise probe x per station. Measure with postprocess/thickness.py.
    probe_x: List[float] = field(default_factory=list)
    #: span_y only: outboard band edges for pyGeo ruled surface (must avoid y=0 symmetry plane).
    span_offsets: Tuple[float, float] = (0.001, 0.006)
    stations: List[Tuple[float, float]] = field(
        default_factory=lambda: [(0.35, 0.1204), (0.96, 0.0050), (0.99, 0.0012)])

    @classmethod
    def from_dict(cls, data):
        check_keys(cls, data, '[constraints.thickness]')
        d = dict(data)
        if 'stations' in d:
            rows = []
            for row in d['stations']:
                pair = _floats(row, '[constraints.thickness] stations row', 2)
                rows.append((pair[0], pair[1]))
            d['stations'] = rows
        if 'probe_x' in d:
            d['probe_x'] = [float(v) for v in d['probe_x']]
        if 'span_offsets' in d:
            pair = _floats(d['span_offsets'],
                           '[constraints.thickness] span_offsets', 2)
            d['span_offsets'] = (pair[0], pair[1])
        cfg = cls(**d)
        if cfg.enabled:                 # an off block is not range-checked
            cfg._validate()
        return cfg

    def _validate(self):
        if self.kind not in ('min', 'max'):
            raise ConfigError(
                f'[constraints.thickness] kind={self.kind!r} is not one of '
                "['min', 'max']")
        if self.axes not in _THICKNESS_AXES:
            raise ConfigError(
                f'[constraints.thickness] axes={self.axes!r} is not one of '
                f'{list(_THICKNESS_AXES)}')
        if self.axes == AXES_SPAN_Y:
            if len(self.probe_x) != len(self.stations):
                raise ConfigError(
                    f'[constraints.thickness] axes="span_y" needs one probe_x '
                    f'per station: got {len(self.probe_x)} for '
                    f'{len(self.stations)} stations. Measure them with '
                    f'python -m mdao_su2.postprocess.thickness.')
            if self.span_offsets[0] >= self.span_offsets[1]:
                raise ConfigError(
                    '[constraints.thickness] span_offsets must be increasing; '
                    f'got {self.span_offsets}')
        elif self.probe_x:
            raise ConfigError(
                '[constraints.thickness] probe_x is meaningless under '
                f'axes="{AXES_CHORD_ZSPAN}" (the station IS the chord '
                'position); set axes="span_y" or drop probe_x.')


#: Keys of a ``[[constraints.linear]]`` entry and of one of its terms.
LINEAR_KEYS = ('name', 'terms', 'offset', 'equals', 'lower', 'upper', 'scaler')
LINEAR_TERM_KEYS = ('point', 'response', 'coeff')


def _float_or_none(value, ctx):
    if value is None:
        return None
    if isinstance(value, bool):
        raise ConfigError(f'{ctx} must be a number, got {value!r}')
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ConfigError(f'{ctx} must be a number, got {value!r}') from None


@dataclass(frozen=True)
class LinearTermConfig:
    """One term of a ``[[constraints.linear]]``: ``coeff * response(point)``."""
    point: str
    response: str
    coeff: float

    @classmethod
    def from_dict(cls, data, ctx):
        if not isinstance(data, dict):
            raise ConfigError(f'{ctx} must be a table {{point, response, coeff}}, '
                              f'got {data!r}')
        unknown = sorted(set(data) - set(LINEAR_TERM_KEYS))
        missing = [k for k in LINEAR_TERM_KEYS if k not in data]
        if unknown or missing:
            raise ConfigError(f'{ctx}: needs exactly {list(LINEAR_TERM_KEYS)}; '
                              f'unknown {unknown}, missing {missing}')
        for key in ('point', 'response'):
            if not isinstance(data[key], str) or not data[key]:
                raise ConfigError(f'{ctx}: {key} must be a non-empty string, '
                                  f'got {data[key]!r}')
        return cls(point=data['point'], response=data['response'],
                   coeff=_float_or_none(data['coeff'], f'{ctx} coeff'))


@dataclass(frozen=True)
class LinearConstraintConfig:
    """One ``[[constraints.linear]]``: ``sum(coeff * response(point)) + offset``,
    bounded by ``equals`` or by ``lower``, ``upper`` or both.

    The load-time checks are ``run_config/checks/points.py::check_points``.
    """
    name: str = ''
    terms: Tuple[LinearTermConfig, ...] = ()
    offset: float = 0.0
    equals: Optional[float] = None
    lower: Optional[float] = None
    upper: Optional[float] = None
    scaler: Optional[float] = None

    #: Prefix of the constraint's ``ExecComp`` in the model.
    COMPONENT_PREFIX = 'linear_'

    @property
    def component(self):
        """The model subsystem that computes the value."""
        return f'{self.COMPONENT_PREFIX}{self.name}'

    @property
    def path(self):
        """The driver response path of the value."""
        return f'{self.component}.value'

    @property
    def constraint_kwargs(self):
        """Keyword dict for ``add_constraint``: the stated bounds and the scaler."""
        kw = {k: getattr(self, k) for k in ('equals', 'lower', 'upper')
              if getattr(self, k) is not None}
        kw['scaler'] = self.scaler
        return kw

    @classmethod
    def from_dict(cls, data, index):
        ctx = f'[[constraints.linear]] {index}'
        if not isinstance(data, dict):
            raise ConfigError(f'{ctx} must be a table, got {data!r}')
        unknown = sorted(set(data) - set(LINEAR_KEYS))
        if unknown:
            raise ConfigError(f'{ctx}: unknown key(s) {unknown}. '
                              f'Allowed: {sorted(LINEAR_KEYS)}')
        name = data.get('name')
        if not isinstance(name, str) or not name:
            raise ConfigError(f'{ctx}: name must be a non-empty string, got {name!r}')
        ctx = f'[[constraints.linear]] {name!r}'
        terms = data.get('terms')
        if not isinstance(terms, list):
            raise ConfigError(f'{ctx}: terms must be a list of '
                              f'{{point, response, coeff}} tables, got {terms!r}')
        if 'scaler' not in data:
            raise ConfigError(f'{ctx}: scaler must be stated; no default is '
                              f'assumed')
        return cls(
            name=name,
            terms=tuple(LinearTermConfig.from_dict(t, f'{ctx} terms[{i}]')
                        for i, t in enumerate(terms)),
            offset=_float_or_none(data.get('offset', 0.0), f'{ctx} offset'),
            **{k: _float_or_none(data.get(k), f'{ctx} {k}')
               for k in ('equals', 'lower', 'upper', 'scaler')})


@dataclass(frozen=True)
class ConstraintsConfig:
    lift: LiftConstraintConfig = field(default_factory=LiftConstraintConfig)
    moment: MomentConstraintConfig = field(default_factory=MomentConstraintConfig)
    #: ``[[constraints.linear]]`` entries, in TOML order.
    linear: Tuple[LinearConstraintConfig, ...] = ()
    geometry: bool = False             # master switch; off when absent
    volume: VolumeConstraintConfig = field(default_factory=VolumeConstraintConfig)
    le_radius: LERadiusConstraintConfig = field(
        default_factory=LERadiusConstraintConfig)
    thickness: ThicknessConstraintConfig = field(
        default_factory=ThicknessConstraintConfig)
    curvature: CurvatureConstraintConfig = field(
        default_factory=CurvatureConstraintConfig)

    @classmethod
    def from_dict(cls, data):
        check_keys(cls, data, '[constraints]')
        geom = data.get('geometry', False)
        if isinstance(geom, dict):        # geometry = { enabled = true }
            extra = sorted(set(geom) - {'enabled'})
            if extra:
                raise ConfigError(
                    f'[constraints] geometry: unknown key(s) {extra}; the only '
                    f'key is "enabled" (or write geometry = true/false)')
            geom = geom.get('enabled', False)
        linear = data.get('linear', [])
        if not isinstance(linear, list):
            raise ConfigError('[[constraints.linear]] must be an array of tables, '
                              f'got {type(linear).__name__}')
        return cls(
            lift=LiftConstraintConfig.from_dict(_sub(data, 'lift')),
            moment=MomentConstraintConfig.from_dict(_sub(data, 'moment')),
            linear=tuple(LinearConstraintConfig.from_dict(item, i)
                         for i, item in enumerate(linear)),
            geometry=bool(geom),
            volume=VolumeConstraintConfig.from_dict(_sub(data, 'volume')),
            le_radius=LERadiusConstraintConfig.from_dict(_sub(data, 'le_radius')),
            thickness=ThicknessConstraintConfig.from_dict(
                _sub(data, 'thickness')),
            curvature=CurvatureConstraintConfig.from_dict(
                _sub(data, 'curvature')))


# ------------------------------------------------------------- optimizer

OPT_DEFAULTS = {
    'SLSQP': {'ACC': 1.0e-5, 'MAXIT': 60, 'IFILE': 'opt_SLSQP.txt'},
    'IPOPT': {
        'tol': 1.0e-5, 'constr_viol_tol': 1.0e-5, 'max_iter': 100,
        'print_level': 5, 'output_file': 'opt_IPOPT.txt',
        'mu_strategy': 'adaptive', 'limited_memory_max_history': 10,
        'nlp_scaling_method': 'none', 'alpha_for_y': 'full', 'recalc_y': 'yes',
    },
    'SNOPT': {
        'Major feasibility tolerance': 1.0e-5,
        'Major optimality tolerance': 1.0e-5,
        'Minor feasibility tolerance': 1.0e-5,
        'Verify level': -1, 'Function precision': 1.0e-5,
        'Major iterations limit': 100, 'Nonderivative linesearch': None,
        'Print file': 'opt_SNOPT_print.txt',
        'Summary file': 'opt_SNOPT_summary.txt',
    },
}
OPTIMIZERS = tuple(OPT_DEFAULTS)


@dataclass(frozen=True)
class OptimizerConfig:
    name: str = 'SLSQP'
    debug_print: List[str] = field(
        default_factory=lambda: ['nl_cons', 'objs', 'desvars'])
    print_opt_prob: bool = True
    hist_file: str = ''                # '' -> opt_<NAME>.hst
    settings: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data):
        plain = {k: v for k, v in data.items() if k not in OPT_DEFAULTS}
        check_keys(cls, plain, '[optimizer]')
        settings = {}
        for opt in OPT_DEFAULTS:
            merged = dict(OPT_DEFAULTS[opt])
            merged.update(_sub(data, opt))
            settings[opt] = merged
        plain.pop('settings', None)
        return cls(settings=settings, **plain)

    @property
    def opt_settings(self):
        try:
            return dict(self.settings[self.name])
        except KeyError:
            raise ConfigError(f'[optimizer] unknown optimizer {self.name!r}; '
                              f'known: {sorted(OPT_DEFAULTS)}')

    def resolved_hist_file(self):
        return self.hist_file or f'opt_{self.name}.hst'


# ------------------------------------------------------------------ trim

@dataclass(frozen=True)
class TrimConfig:
    """Secant trim on alpha for C_L = target, before the optimizer starts."""
    enabled: bool = False
    max_iter: int = 6                  # multipoint.py:416
    tol: float = 1.0e-4
    dcl_da_init: float = 0.11          # deg^-1 seed, multipoint.py:439
    alpha_clip: Tuple[float, float] = (0.0, 5.0)

    @classmethod
    def from_dict(cls, data):
        fixed = sorted(k for k in data if k.startswith('fixed_cl'))
        if fixed:
            raise ConfigError(
                f'[trim] {fixed}: SU2 FIXED_CL_MODE was removed. Its '
                f'primal-only alpha update left the adjoint linearizing at the input '
                f'alpha. Target lift with [constraints.lift] equals and [trim] '
                f'enabled = true (alpha as a design variable).')
        check_keys(cls, data, '[trim]')
        d = dict(data)
        if 'alpha_clip' in d:
            lo, hi = _floats(d['alpha_clip'], '[trim] alpha_clip', 2)
            d['alpha_clip'] = (lo, hi)
        return cls(**d)


# ------------------------------------------------------------------ polar

@dataclass(frozen=True)
class PolarConfig:
    """``[polar]``: the lift curve of the ``polar`` task, URANS where steady stops.

    ``urans`` holds the SU2 v8.5 time-domain keys of ``[polar.urans]``; they
    derive the URANS cfg and never enter a generated steady cfg. The switch
    for running URANS is ``run_urans`` because TOML cannot hold a key and a
    table of one name. An enabled block states every value it uses.
    """
    enabled: bool = False
    #: Angles in degrees, strictly increasing.
    alpha: Tuple[float, ...] = ()
    #: A URANS window whose ``C_L`` range is within this counts as settled and needs no
    #: ``urans_min_cycles``.
    cl_band: float = 0.0
    run_urans: bool = False
    #: Per point, the URANS angles run, in alpha order.
    urans_max_angles: int = 0
    #: Fewest oscillation cycles of ``C_L`` an unsteady averaging window must span.
    urans_min_cycles: int = 0
    #: Lowest Mach number the sweep covers (the case's range bound); several points need it.
    mach_floor: float = 0.0
    urans: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data):
        check_keys(cls, data, '[polar]')
        d = dict(data)
        if 'alpha' in d:
            d['alpha'] = tuple(_floats(d['alpha'], '[polar] alpha'))
        if 'cl_band' in d:
            d['cl_band'] = float(d['cl_band'])
        if 'urans_max_angles' in d:
            d['urans_max_angles'] = int(d['urans_max_angles'])
        if 'urans_min_cycles' in d:
            d['urans_min_cycles'] = int(d['urans_min_cycles'])
        if 'mach_floor' in d:
            d['mach_floor'] = float(d['mach_floor'])
        if 'urans' in d:
            table = d['urans']
            if not isinstance(table, dict):
                raise ConfigError('[polar.urans] must be a table of SU2 option keys')
            for key, value in table.items():
                items = value if isinstance(value, list) else [value]
                if not all(isinstance(v, (str, int, float, bool)) for v in items):
                    raise ConfigError(f'[polar.urans] {key}: values must be '
                                      f'strings, numbers, booleans, or lists of them')
            d['urans'] = dict(table)
        return cls(**d)


# ---------------------------------------------------------------- solver

@dataclass(frozen=True)
class SolverBlockConfig:
    """Per-side solver keys: iter cap, MG depth, CFL adapt, residual floor."""
    iter: int = 10000
    mglevel: int = 3
    cfl_adapt: bool = True             # multipoint.py:105
    cfl_adapt_param: str = '( 0.1, 2.0, 1.0, 100.0 )'
    conv_residual_minval: float = -9.0
    warm_start: bool = True            # False = cold start for finite-difference verification; required on
                                       # [solver.primal] (REQUIRED_KEYS)
    # Which keys the user actually wrote, as opposed to dataclass defaults.
    _explicit_keys: frozenset = field(default_factory=frozenset,
                                      repr=False, compare=False)

    @classmethod
    def from_dict(cls, data, ctx, explicit=None):
        check_keys(cls, data, ctx)
        d = dict(data)
        d.pop('_explicit_keys', None)
        if 'iter' in d:
            d['iter'] = int(d['iter'])
        if 'mglevel' in d:
            d['mglevel'] = int(d['mglevel'])
        if 'conv_residual_minval' in d:
            d['conv_residual_minval'] = float(d['conv_residual_minval'])
        return cls(_explicit_keys=frozenset(
            d if explicit is None else explicit), **d)


@dataclass(frozen=True)
class NumericsConfig:
    """Convective scheme/gradient reconstruction, shared by both sides: the discrete
    adjoint differentiates the primal discretization."""
    conv_method_flow: str = 'JST'
    muscl_flow: str = 'NO'
    slope_limiter_flow: str = 'VENKATAKRISHNAN'
    venkat_limiter_coeff: float = 0.03
    jst_sensor_coeff: str = '( 0.5, 0.02 )'
    time_discre_flow: str = 'EULER_IMPLICIT'
    conv_method_turb: str = 'SCALAR_UPWIND'
    muscl_turb: str = 'NO'
    slope_limiter_turb: str = 'VENKATAKRISHNAN'
    time_discre_turb: str = 'EULER_IMPLICIT'
    num_method_grad: str = 'WEIGHTED_LEAST_SQUARES'
    cfl_number: float = 4.0

    @classmethod
    def from_dict(cls, data):
        check_keys(cls, data, '[solver.numerics]')
        d = dict(data)
        for k in ('venkat_limiter_coeff', 'cfl_number'):
            if k in d:
                d[k] = float(d[k])
        return cls(**d)


@dataclass(frozen=True)
class LinearSolverConfig:
    """LINEAR_SOLVER* on both paths. ``error`` is a string to preserve cfg format."""
    solver: str = 'FGMRES'
    error: str = '1E-1'
    iter: int = 3
    prec: str = 'ILU'                  # SU2's default; emitted only when changed

    @classmethod
    def from_dict(cls, data):
        check_keys(cls, data, '[solver.linear]')
        d = dict(data)
        if 'error' in d:
            d['error'] = str(d['error'])
        if 'iter' in d:
            d['iter'] = int(d['iter'])
        return cls(**d)


@dataclass(frozen=True)
class MultigridConfig:
    """Multigrid cycle/smoothing. Counts are per-level; MG_*_SMOOTH derived from mglevel."""
    cycle: str = 'V_CYCLE'
    pre_smooth: int = 4
    post_smooth: int = 4
    correction_smooth: int = 1
    damp_restriction: float = 0.5
    damp_prolongation: float = 0.5

    @classmethod
    def from_dict(cls, data):
        check_keys(cls, data, '[solver.multigrid]')
        d = dict(data)
        for k in ('pre_smooth', 'post_smooth', 'correction_smooth'):
            if k in d:
                d[k] = int(d[k])
        for k in ('damp_restriction', 'damp_prolongation'):
            if k in d:
                d[k] = float(d[k])
        return cls(**d)


@dataclass(frozen=True)
class ConvergenceConfig:
    """Cauchy criterion and start iteration. Residual floor is per-side."""
    startiter: int = 10
    cauchy_elems: int = 100
    cauchy_eps: str = '1E-6'

    @classmethod
    def from_dict(cls, data):
        check_keys(cls, data, '[solver.convergence]')
        d = dict(data)
        for k in ('startiter', 'cauchy_elems'):
            if k in d:
                d[k] = int(d[k])
        if 'cauchy_eps' in d:
            d['cauchy_eps'] = str(d['cauchy_eps'])
        return cls(**d)


@dataclass(frozen=True)
class AdjointConfig:
    """Discrete-adjoint-only keys. FROZEN_VISC_CONT deliberately excluded."""
    frozen_visc_disc: bool = False
    frozen_limiter_disc: bool = False
    relaxation_factor: float = 1.0     # RELAXATION_FACTOR_ADJOINT, adjoint CFL behavior
    objective_function: str = 'DRAG'   # the driver switches this to LIFT

    @classmethod
    def from_dict(cls, data):
        check_keys(cls, data, '[solver.adjoint_settings]')
        d = dict(data)
        if 'relaxation_factor' in d:
            d['relaxation_factor'] = float(d['relaxation_factor'])
        return cls(**d)


@dataclass(frozen=True)
class SolverConfig:
    primal: SolverBlockConfig = field(default_factory=SolverBlockConfig)
    # Adjoint: MGLEVEL 0 always, validate.py enforces.
    adjoint: SolverBlockConfig = field(default_factory=lambda: SolverBlockConfig(
        iter=20000, mglevel=0, conv_residual_minval=-8.0))
    numerics: NumericsConfig = field(default_factory=NumericsConfig)
    linear: LinearSolverConfig = field(default_factory=LinearSolverConfig)
    multigrid: MultigridConfig = field(default_factory=MultigridConfig)
    convergence: ConvergenceConfig = field(default_factory=ConvergenceConfig)
    adjoint_settings: AdjointConfig = field(default_factory=AdjointConfig)
    #: ``[solver.su2_extra]`` passthrough: ``{'both'|'primal'|'adjoint':
    #: {SU2_KEY: value}}``. Keys are checked against the SU2 scan in validate.py.
    su2_extra: Dict[str, Dict[str, Any]] = field(
        default_factory=lambda: {'both': {}, 'primal': {}, 'adjoint': {}})

    @classmethod
    def from_dict(cls, data):
        check_keys(cls, data, '[solver]')
        extra = dict(_sub(data, 'su2_extra'))
        sides = {'primal': dict(_sub(extra, 'primal')),
                 'adjoint': dict(_sub(extra, 'adjoint'))}
        both = {k: v for k, v in extra.items() if k not in sides}
        for side, table in sides.items():
            clash = sorted(set(table) & set(both))
            if clash:
                raise ConfigError(f'[solver.su2_extra] {clash} set both shared and '
                                  f'under .{side}; a key has one home')
        for table in (both, *sides.values()):
            for key, value in table.items():
                items = value if isinstance(value, list) else [value]
                if not all(isinstance(v, (str, int, float, bool)) for v in items):
                    raise ConfigError(f'[solver.su2_extra] {key}: values must be '
                                      f'strings, numbers, booleans, or lists of them')
        primal = SolverBlockConfig.from_dict(_sub(data, 'primal'), '[solver.primal]')
        adj_data = _sub(data, 'adjoint')
        check_keys(SolverBlockConfig, adj_data, '[solver.adjoint]')
        adj_defaults = SolverBlockConfig(iter=20000, mglevel=0,
                                   conv_residual_minval=-8.0)
        merged = {k: v for k, v in asdict(adj_defaults).items()
                  if not k.startswith('_')}
        merged.update(adj_data)
        return cls(
            primal=primal,
            adjoint=SolverBlockConfig.from_dict(merged, '[solver.adjoint]',
                                          explicit=set(adj_data)),
            numerics=NumericsConfig.from_dict(_sub(data, 'numerics')),
            linear=LinearSolverConfig.from_dict(_sub(data, 'linear')),
            multigrid=MultigridConfig.from_dict(_sub(data, 'multigrid')),
            convergence=ConvergenceConfig.from_dict(_sub(data, 'convergence')),
            adjoint_settings=AdjointConfig.from_dict(
                _sub(data, 'adjoint_settings')),
            su2_extra={'both': both, **sides})


# ------------------------------------------------------- mesh deformation

@dataclass(frozen=True)
class DeformConfig:
    """SU2 in-memory mesh deformation. Defaults match SU2 v8.5 CConfig.cpp."""
    linear_solver: str = 'FGMRES'          # CConfig.cpp:2488
    linear_solver_prec: str = 'ILU'        # CConfig.cpp:2490
    linear_solver_error: float = 1e-14     # CConfig.cpp:2492
    linear_solver_iter: int = 1000         # CConfig.cpp:2494
    stiffness_type: str = 'WALL_DISTANCE'  # CConfig.cpp:2480 (SOLID_WALL_DISTANCE)
    nonlinear_iter: int = 1                # CConfig.cpp:2474
    console_output: bool = False           # CConfig.cpp:2472
    # Emitted only when user wrote [deform] in TOML.
    _explicit_keys: frozenset = field(default_factory=frozenset,
                                      repr=False, compare=False)

    @classmethod
    def from_dict(cls, data):
        check_keys(cls, data, '[deform]')
        d = dict(data)
        d.pop('_explicit_keys', None)
        if 'linear_solver_error' in d:
            d['linear_solver_error'] = float(d['linear_solver_error'])
        for k in ('linear_solver_iter', 'nonlinear_iter'):
            if k in d:
                d[k] = int(d[k])
        return cls(_explicit_keys=frozenset(d), **d)

    @property
    def emit(self):
        """True when [deform] was written in the TOML at all."""
        return bool(self._explicit_keys)


# --------------------------------------------------------------- remeshing

def _bl_thickness(first_cell_height, growth_ratio, n_layers):
    """Total BL depth: h1*(r**n - 1)/(r - 1)."""
    if growth_ratio == 1.0:
        return first_cell_height * n_layers
    return first_cell_height * (growth_ratio ** n_layers - 1) / (growth_ratio - 1)


def _layers_for_thickness(first_cell_height, growth_ratio, thickness):
    """Fewest geometric layers reaching ``thickness``. Rounded UP (safe direction)."""
    import math
    if thickness <= first_cell_height:
        return 1
    if growth_ratio == 1.0:
        return max(1, math.ceil(thickness / first_cell_height))
    ratio = 1.0 + thickness * (growth_ratio - 1.0) / first_cell_height
    return max(1, math.ceil(math.log(ratio) / math.log(growth_ratio)))


@dataclass(frozen=True)
class RemeshQualityConfig:
    """Hard gates on generated mesh quality; violation raises om.AnalysisError."""
    min_first_cell_height_ratio: float = 0.5     # measured / requested
    max_first_cell_height_ratio: float = 2.0
    min_bl_thickness_ratio: float = 0.5
    max_bl_thickness_ratio: float = 2.0
    max_skewness: float = 0.85
    # Gated on p99 (not max) because BL cells are naturally high-AR. Baseline p99=1075.
    max_aspect_ratio_p99: float = 2000.0

    @classmethod
    def from_dict(cls, data):
        check_keys(cls, data, '[remesh.quality]')
        return cls(**{k: float(v) for k, v in data.items()})


@dataclass(frozen=True)
class RemeshSolverConfig:
    """Overrides [solver] on remesh path: MG=0 (unstable on hybrid gmsh meshes)."""
    primal_iter: int = 50000
    primal_mglevel: int = 0
    adjoint_iter: int = 30000
    adjoint_mglevel: int = 0
    primal_conv_residual_minval: float = -9.0  # matches non-remesh path
    adjoint_conv_residual_minval: float = -6.0  # loose; adjoint accuracy bounded by field bias
    cfl_adapt_param: str = '( 0.1, 2.0, 1.0, 1000.0 )'
    adjoint_stall_delta: float = 0.1  # decades improvement threshold
    adjoint_stall_window: int = 2000  # sliding window for stall detection

    @classmethod
    def from_dict(cls, data):
        check_keys(cls, data, '[remesh.solver]')
        d = dict(data)
        for k in ('primal_iter', 'primal_mglevel', 'adjoint_iter',
                  'adjoint_mglevel', 'adjoint_stall_window'):
            if k in d:
                d[k] = int(d[k])
        for k in ('primal_conv_residual_minval',
                  'adjoint_conv_residual_minval',
                  'adjoint_stall_delta'):
            if k in d:
                d[k] = float(d[k])
        return cls(**d)


@dataclass(frozen=True)
class RemeshConfig:
    """Gmsh remeshing parameters. See mesh/mesh2d.py for the downstream consumer."""
    mode: str = 'none'                 # none -> SU2_DEF in-memory deformation
    first_cell_height: float = 2.0e-6  # chord units, y+~1 at Re=9e6
    growth_ratio: float = 1.2
    n_bl_layers: int = 25
    # >0 overrides n_bl_layers: layer count derived to reach this thickness.
    # Gmsh's default 0.01 once governed silently; the value is now explicit.
    bl_thickness: float = 0.0
    # Fan BL at sharp vertices (TE cusp). 0.0 disables; 60 selects only the TE.
    fan_angle_deg: float = 60.0
    fan_elements: int = 7
    farfield_radius: float = 100.0
    # Ceiling on element size; the slope below governs actual sizes.
    farfield_meshsize: float = 15.0
    # Size field: min(farfield_meshsize, max(near_meshsize, slope * d)).
    # Proportional growth matches baseline mesh h/d profile.
    near_meshsize: float = 0.01
    far_growth_slope: float = 0.07
    wall_meshsize: float = 0.0         # 0.0 -> None (auto from point spacing)
    # Optional wake wedge in the size field (mesh/mesh2d.py::size_field_expression);
    # wake_length = 0 is off, and a positive length needs the other three.
    wake_length: float = 0.0
    wake_angles_deg: tuple = ()        # [lo, hi] ray angles from +x, lo < hi
    wake_meshsize: float = 0.0         # size at the TE apex
    wake_stream_slope: float = 0.0     # size growth per unit length along the bisector

    # Extra mesh formats for inspection: 'msh' (gmsh), 'vtk' (ParaView). Empty by default.
    viewable_mesh: tuple = ()
    verbose: bool = False
    mesh_output: str = 'remesh/mesh_current_{id}.su2'
    solver: RemeshSolverConfig = field(default_factory=RemeshSolverConfig)
    quality: RemeshQualityConfig = field(default_factory=RemeshQualityConfig)

    @classmethod
    def from_dict(cls, data):
        plain = {k: v for k, v in data.items()
                 if k not in ('solver', 'quality')}
        check_keys(cls, plain, '[remesh]')
        for k in ('first_cell_height', 'growth_ratio', 'bl_thickness',
                  'fan_angle_deg', 'farfield_radius', 'farfield_meshsize',
                  'near_meshsize', 'far_growth_slope', 'wall_meshsize',
                  'wake_length', 'wake_meshsize', 'wake_stream_slope'):
            if k in plain:
                plain[k] = float(plain[k])
        if 'wake_angles_deg' in plain:
            plain['wake_angles_deg'] = tuple(float(a) for a in plain['wake_angles_deg'])
        for k in ('n_bl_layers', 'fan_elements'):
            if k in plain:
                plain[k] = int(plain[k])
        if 'viewable_mesh' in plain:
            # TOML gives a list; the dataclass is frozen, so it must be hashable.
            v = plain['viewable_mesh']
            plain['viewable_mesh'] = tuple(
                [v] if isinstance(v, str) else v)
        return cls(solver=RemeshSolverConfig.from_dict(_sub(data, 'solver')),
                   quality=RemeshQualityConfig.from_dict(_sub(data, 'quality')),
                   **plain)

    @property
    def enabled(self):
        return self.mode != 'none'

    def resolved_bl(self):
        """Return consistent (h1, ratio, n_layers, thickness) quadruple."""
        h, r = self.first_cell_height, self.growth_ratio
        if self.bl_thickness > 0.0:
            # Snap UP to whole layers so gmsh and the quality gate agree.
            n = _layers_for_thickness(h, r, self.bl_thickness)
            return h, r, n, _bl_thickness(h, r, n)
        n = int(self.n_bl_layers)
        return h, r, n, _bl_thickness(h, r, n)

    def gmsh_kwargs(self):
        """Kwargs for ``mesh.mesh2d.remesh_airfoil``. bl_thickness always explicit."""
        h, r, n, thickness = self.resolved_bl()
        wake = {}
        if self.wake_length > 0.0:
            wake = {'wake_length': self.wake_length,
                    'wake_angles_deg': tuple(self.wake_angles_deg),
                    'wake_meshsize': self.wake_meshsize,
                    'wake_stream_slope': self.wake_stream_slope}
        return {**wake,
            'first_cell_height': h,
            'growth_ratio': r,
            'n_bl_layers': n,
            'bl_thickness': thickness,
            'fan_angle_deg': self.fan_angle_deg,
            'fan_elements': self.fan_elements,
            'farfield_radius': self.farfield_radius,
            'farfield_meshsize': self.farfield_meshsize,
            'near_meshsize': self.near_meshsize,
            'far_growth_slope': self.far_growth_slope,
            'wall_meshsize': self.wall_meshsize or None,
            'viewable_formats': self.viewable_mesh,
            'verbose': self.verbose,
        }


# ------------------------------------------------- execution / verification

TASKS = ('run_model', 'trim', 'compute_totals', 'check_totals',
         'moment_pivot_check', 'run_driver', 'polar')
# Tasks that solve adjoints; the others never read the adjoint settings.
DERIVATIVE_TASKS = ('compute_totals', 'check_totals', 'moment_pivot_check',
                    'run_driver')


_DERIVATIVE_MODES = ('rev', 'fwd', 'auto')


@dataclass(frozen=True)
class ExecutionConfig:
    task: str = 'run_model'
    parallel_points: bool = False      # requires comm.size >= n_points
    adjoint_restart: bool = True       # explicit restart choice; required in every TOML (REQUIRED_KEYS)
    clean_restart: bool = True         # drop stale per-point solutions first
    derivative_mode: str = 'rev'       # OpenMDAO prob.setup(mode=…)

    @classmethod
    def from_dict(cls, data):
        check_keys(cls, data, '[execution]')
        d = dict(data)
        mode = d.get('derivative_mode', 'rev')
        if mode not in _DERIVATIVE_MODES:
            raise ConfigError(
                f'[execution] derivative_mode={mode!r} unknown; '
                f'allowed: {list(_DERIVATIVE_MODES)}')
        return cls(**d)


def normal_force_slope(dcl_dalpha, dcd_dalpha, cl, cd, alpha_deg):
    """Body-frame dC_N/dalpha per degree from the wind-frame slopes (per degree).

    C_N = C_L cos(a) + C_D sin(a); the product rule adds the frame-rotation
    term (C_D cos(a) - C_L sin(a)) * pi/180.
    """
    a = math.radians(alpha_deg)
    return (dcl_dalpha * math.cos(a) + dcd_dalpha * math.sin(a)
            + (-cl * math.sin(a) + cd * math.cos(a)) * math.radians(1.0))


@dataclass(frozen=True)
class MomentPivotConfig:
    """``task="moment_pivot_check"``: the exact moment-transfer identity as a gate.

    For any two pivots on the chord line the transfer is algebraically exact::

        dC_M(P)/dalpha = dC_M(ref)/dalpha + (x_ref - x_P)/c * dC_N/dalpha

    with the BODY-frame normal force derivative built from the WIND-frame force
    derivatives the FD check already verifies (6.4e-6 / 8.4e-6 relative)::

        dC_N/dalpha = dC_L/dalpha*cos(a) + dC_D/dalpha*sin(a)
                      + (-C_L*sin(a) + C_D*cos(a)) * pi/180

    That last term is the frame-rotation contribution -- O(C_L*pi/180), the same
    size as dC_L/dalpha itself.  It is what makes this check non-trivial: it ties
    the angle-of-attack force projection applied on the FORCE side to the NO-projection of the
    body-axis MOMENT (``solver/wall_seam.py::project_alpha``), so the identity holds only if both conventions are right
    together.  That intersection -- moment AND alpha -- is the one path the
    existing evidence never touches: the 17 C_M shape rows are moment-without-
    alpha, and the C_D/C_L alpha rows are alpha-without-moment.

    Cost is one primal plus three adjoints, no FD perturbation solves, because
    the FLOW does not depend on the moment reference point -- only the moment
    integration does.  Moving the pivot to the leading edge amplifies the signal
    74x (3.52e-2 against 4.75e-4), so the check is limited by the force
    adjoint's own 1.17e-6 absolute floor, i.e. 3.3e-5 relative -- 19x sharper
    than the cancellation-limited FD it replaces.
    """
    enabled: bool = False
    # Pivot the reference derivatives below were measured at, in chords.
    reference_pivot_x: float = 0.25
    # Reference-run measurements. All at THIS run's primal state, which the
    # pivot cannot change, so the two runs must otherwise be identical configs.
    dcm_dalpha_ref: float = 0.0
    dcl_dalpha: float = 0.0
    dcd_dalpha: float = 0.0
    cl: float = 0.0
    cd: float = 0.0
    alpha_deg: float = 0.0
    # Acceptance, in CHORDS of equivalent pivot error (same units as
    # VerificationConfig.alpha_band_moment_pivot, and the same rationale).
    band_pivot: float = 1.0e-4
    # Provenance of the reference numbers; printed and archived, never parsed.
    reference_run: str = ''

    @classmethod
    def from_dict(cls, data):
        check_keys(cls, data, '[verification.moment_pivot]')
        d = dict(data)
        if 'enabled' in d:
            d['enabled'] = bool(d['enabled'])
        for k in ('reference_pivot_x', 'dcm_dalpha_ref', 'dcl_dalpha',
                  'dcd_dalpha', 'cl', 'cd', 'alpha_deg', 'band_pivot'):
            if k in d:
                d[k] = float(d[k])
        if 'reference_run' in d:
            d['reference_run'] = str(d['reference_run'])
        return cls(**d)

    def dcn_dalpha(self):
        """Body-frame dC_N/dalpha per degree, from the verified wind-frame pair."""
        return normal_force_slope(self.dcl_dalpha, self.dcd_dalpha, self.cl,
                                  self.cd, self.alpha_deg)

    def predict(self, pivot_x):
        """dC_M/dalpha this run must reproduce at its own pivot ``pivot_x``."""
        return (self.dcm_dalpha_ref
                + (self.reference_pivot_x - pivot_x) * self.dcn_dalpha())


@dataclass(frozen=True)
class VerificationConfig:
    """check_totals FD settings. Shape 1e-6 avoids the finite-difference plateau; alpha uses 1e-3 in degrees."""
    shape_step: float = 1.0e-6
    alpha_step: float = 1.0e-3
    #: FD step for [dv.twist] rows, in degrees. No default: a twist gate states it.
    twist_step: Optional[float] = None
    form: str = 'central'
    step_calc: str = 'abs'
    # Acceptance bands. Unset by default: a run with no band reports
    # the differences without a verdict, and a run that states one must state
    # every band its rows use. See
    # The human guide explains the C_r evidence behind these recommendations.
    alpha_band: Optional[float] = None
    alpha_band_moment: float = 0.0
    # Normalized-absolute C_M alpha gate, in CHORDS of equivalent pivot error:
    # |analytic - fd| / |dC_N/dalpha| <= this.  Supersedes alpha_band_moment.
    # A relative band on dC_M/dalpha about the aerodynamic center is ill-posed --
    # the pivot makes the quantity ~0 by definition, so C_r is ~10^5 and the
    # FD reference is cancellation-limited at ~6e-4 relative no matter how far
    # the primal is converged (measured: 3.04e-7 absolute at primal -12, and
    # the best any h can reach is 1.01e-7).  The pivot-error form is well posed:
    # the denominator is dC_N/dalpha to 0.4%, is NOT cancellation-limited, and
    # the criterion reads as "the moment derivative is right to within a pivot
    # misplacement of X chords".  At 1e-4 chords the RAE 2822 row passes with
    # 45.6x margin and still fails on a 2.92% error in dC_M/dalpha.
    alpha_band_moment_pivot: float = 0.0
    # Per shape row: max(shape_band_floor, shape_band_slope * C_r[func][i]).
    shape_band_floor: Optional[float] = None
    shape_band_slope: Optional[float] = None
    # {functional: [C_r per shape DV]}. Empty by default; case-specific, not inherited.
    cancellation_cr: Dict[str, Tuple[float, ...]] = field(default_factory=dict)
    # Parameterization tag the C_r table was measured on;
    # the validator rejects a mismatch.
    cancellation_cr_param: str = ''
    # The FLAT fallback, for a functional with no measured C_r.
    shape_band: Optional[float] = None
    # Ceiling on the C_r-tiered band.  A band only gates if a plausible bug
    # exceeds it: a sign-flipped row scores rel 2.0, a zeroed or doubled row
    # 1.0, a 10% error 0.1.  So at C_r > 500 the tiered band passes a ZEROED
    # derivative, and past ~1000 it passes a SIGN FLIP -- the row is reported
    # as verified while nothing about it has been tested.  Rows whose band
    # would exceed this cap are NOT FD-verifiable and FAIL (a gate that
    # waves a row through is not a gate) unless the config names them in
    # shape_rows_unverifiable.  Measured C_r: 51 of 54 rows sit at C_r <= 50.
    shape_band_cap: float = 1.0
    # {functional: [row indices]} explicitly acknowledged as beyond the cap.
    # Naming a row here records that it is UNVERIFIED, never that it passed.
    shape_rows_unverifiable: Dict[str, Tuple[int, ...]] = field(default_factory=dict)
    # Extra scenario outputs checked only against alpha; moment checks use C_M here
    # without paying for unverified shape rows.
    alpha_outputs: Tuple[str, ...] = ()
    # Extra scenario outputs checked against shape DVs; empty preserves the
    # historical C_L/C_D shape gate. Moment checks use C_M here explicitly.
    shape_outputs: Tuple[str, ...] = ()
    # Diagnostic archive only; measured ratios never alter acceptance bands.
    record_shape_sensitivities: bool = False
    # The FD-free moment-transfer gate; see MomentPivotConfig.
    moment_pivot: 'MomentPivotConfig' = field(default_factory=lambda: MomentPivotConfig())

    # Alias: objective row → C_D cancellation table.
    OBJECTIVE_CR_KEY = 'C_D'

    @classmethod
    def from_dict(cls, data):
        d = dict(data)
        if 'cancellation_cr_ffd' in d:
            raise ConfigError('[verification] cancellation_cr_ffd is no longer '
                              'accepted; write cancellation_cr_param '
                              f'({MIGRATE_HINT})')
        check_keys(cls, d, '[verification]')
        for k in ('shape_step', 'alpha_step', 'twist_step', 'alpha_band',
                  'alpha_band_moment', 'alpha_band_moment_pivot', 'shape_band',
                  'shape_band_floor', 'shape_band_slope', 'shape_band_cap'):
            if k in d:
                d[k] = float(d[k])
        if 'shape_rows_unverifiable' in d:
            raw = d['shape_rows_unverifiable']
            if not isinstance(raw, dict):
                raise ConfigError(
                    '[verification] shape_rows_unverifiable must be a table of '
                    '{functional = [row index, ...]}, got '
                    f'{type(raw).__name__}')
            d['shape_rows_unverifiable'] = {
                str(k): tuple(int(v) for v in vals)
                for k, vals in raw.items()}
        if 'moment_pivot' in d:
            d['moment_pivot'] = MomentPivotConfig.from_dict(d['moment_pivot'])
        if 'cancellation_cr' in d:
            raw = d['cancellation_cr']
            if not isinstance(raw, dict):
                raise ConfigError(
                    '[verification] cancellation_cr must be a table of '
                    '{functional = [C_r, ...]}, got '
                    f'{type(raw).__name__}')
            d['cancellation_cr'] = {
                str(k): tuple(float(v) for v in vals)
                for k, vals in raw.items()}
        if 'cancellation_cr_param' in d:
            d['cancellation_cr_param'] = str(d['cancellation_cr_param'])
        if 'alpha_outputs' in d:
            d['alpha_outputs'] = tuple(str(v) for v in d['alpha_outputs'])
        if 'shape_outputs' in d:
            d['shape_outputs'] = tuple(str(v) for v in d['shape_outputs'])
        return cls(**d)

    #: The acceptance bands a run states or leaves unset.
    BAND_KEYS = ('alpha_band', 'shape_band', 'shape_band_floor', 'shape_band_slope')

    @property
    def stated_bands(self):
        """Band keys this config states; the moment overrides count when set."""
        stated = [k for k in self.BAND_KEYS if getattr(self, k) is not None]
        stated += [k for k in ('alpha_band_moment', 'alpha_band_moment_pivot')
                   if getattr(self, k) > 0]
        return tuple(stated)

    @property
    def judged(self):
        """True when check_totals gives verdicts; False when it only reports."""
        return bool(self.stated_bands)

    def shape_row_is_tiered(self, functional, param=None):
        """True when ``functional``'s shape rows take the C_r-tiered band."""
        if param is not None and self.cancellation_cr_param != param:
            return False
        return bool(self.cancellation_cr.get(functional))

    def alpha_row_band(self, functional):
        """Acceptance band for an alpha row. Uses alpha_band_moment for C_M when set."""
        if functional == 'C_M' and self.alpha_band_moment > 0:
            return self.alpha_band_moment
        return self.alpha_band

    def alpha_row_is_pivot_gated(self, functional):
        """True when this alpha row is judged in pivot-chords, not relative error."""
        return functional == 'C_M' and self.alpha_band_moment_pivot > 0

    def shape_row_band(self, functional, i, param=None):
        """Acceptance band for shape row i. Falls back to flat shape_band when no C_r measured."""
        if param is not None and self.cancellation_cr_param != param:
            return self.shape_band
        crs = self.cancellation_cr.get(functional)
        if not crs or i >= len(crs):
            return self.shape_band
        if self.shape_band_floor is None or self.shape_band_slope is None:
            return None
        return max(self.shape_band_floor, self.shape_band_slope * crs[i])

    def shape_row_is_verifiable(self, functional, i, param=None):
        """False when row i's band exceeds shape_band_cap -- the band has no teeth.

        The flat fallback is never capped: it is not derived from C_r, so a
        config that raises shape_band has said what it means. Only the TIERED
        band can silently inflate past the point where a bug would still pass.
        """
        if self.shape_band_cap <= 0:
            return True
        if param is not None and self.cancellation_cr_param != param:
            return True
        crs = self.cancellation_cr.get(functional)
        if not crs or i >= len(crs):
            return True
        band = self.shape_row_band(functional, i, param=param)
        return band is None or band <= self.shape_band_cap

    def shape_row_unverifiable_ack(self, functional, i):
        """True when the config explicitly records row i as not FD-verifiable."""
        return i in self.shape_rows_unverifiable.get(functional, ())


@dataclass(frozen=True)
class SU2OutputConfig:
    """SU2 output settings. History/screen outputs differ by side on purpose.
    SU2 silently drops unrecognised names; use group names (RMS_RES) for portability."""
    wrt_freq: int = 10000
    mesh_format: str = 'SU2'
    tabular_format: str = 'CSV'
    history_output_primal: str = 'ITER, RMS_RES, AOA, LIFT, DRAG, EFFICIENCY'
    history_output_adjoint: str = (
        'ITER, RMS_RES, SENS_AOA, LINSOL_RESIDUAL, LINSOL_ITER')
    screen_output_primal: str = 'INNER_ITER, RMS_RES, LIFT, DRAG'
    screen_output_adjoint: str = (
        'INNER_ITER, RMS_RES, SENS_AOA, LINSOL_RESIDUAL, LINSOL_ITER')
    #: VOLUME_OUTPUT: field groups or names of the volume and surface files
    #: (SU2 has no separate surface list); empty leaves SU2's default.
    volume_output: List[str] = field(default_factory=list)
    #: OUTPUT_FILES; empty leaves SU2's default. Must hold RESTART. The stage copy
    #: reads RESTART, PARAVIEW and SURFACE_PARAVIEW (``.dat`` and ``.vtu``
    #: files); SURFACE_CSV and the other v8.5 formats stay in su2_workdir/.
    output_files: List[str] = field(default_factory=list)
    #: RESTART_FILENAME / VOLUME_FILENAME / SURFACE_FILENAME stems (``_<point>`` is appended).
    restart_basename: str = 'restart_flow'
    volume_basename: str = 'flow'
    surface_basename: str = 'surface_flow'


    @classmethod
    def from_dict(cls, data):
        check_keys(cls, data, '[output.su2]')
        d = dict(data)
        if 'wrt_freq' in d:
            d['wrt_freq'] = int(d['wrt_freq'])
        for k in ('volume_output', 'output_files'):
            if k in d:
                d[k] = [str(v) for v in d[k]]
        if 'MESH' in d.get('output_files', ()):
            raise ConfigError('[output.su2] output_files may not hold MESH: SU2 v8.5\'s '
                              'MESH writer needs an undeclared boundary.dat')
        if d.get('output_files') and 'RESTART' not in d['output_files']:
            raise ConfigError('[output.su2] output_files needs RESTART: the adjoint '
                              'and warm starts read the primal restart')
        stems = [d.get(k, getattr(cls, k)) for k in
                 ('restart_basename', 'volume_basename', 'surface_basename')]
        if len(set(stems)) != 3 or not all(s and s.replace('_', '').isalnum() for s in stems):
            raise ConfigError('[output.su2] restart/volume/surface_basename must be '
                              'distinct names of letters, digits and underscores')
        return cls(**d)


@dataclass(frozen=True)
class SnapshotsConfig:
    """Saved OpenMDAO evaluations. Surface by default; volume is much larger."""
    # off / surface / volume
    mode: str = 'surface'
    every_n: int = 1
    # none / baseline_latest / all
    keep_volume: str = 'all'

    @classmethod
    def from_dict(cls, data):
        check_keys(cls, data, '[output.snapshots]')
        d = dict(data)
        if 'every_n' in d:
            d['every_n'] = int(d['every_n'])
        return cls(**d)


@dataclass(frozen=True)
class SpanLoadConfig:
    """``[postprocess.span_load]``: span load, e, e0 after the run.

    Toggle ``postprocess.span_load``. When on: ``n_stations``, ``span_axis``,
    ``lift_axis``, ``semispan`` (the b/2 of the elliptic reference) and
    ``aspect_ratio`` are required. ``cd0`` is optional: without it e0 is not
    reported (nothing is assumed for it).
    """
    enabled: bool = False
    n_stations: Optional[int] = None
    span_axis: Optional[str] = None
    lift_axis: Optional[str] = None
    semispan: Optional[float] = None
    aspect_ratio: Optional[float] = None
    cd0: Optional[float] = None

    @classmethod
    def from_dict(cls, data):
        check_keys(cls, data, '[postprocess.span_load]')
        d = dict(data)
        if 'n_stations' in d:
            d['n_stations'] = int(d['n_stations'])
        for k in ('semispan', 'aspect_ratio', 'cd0'):
            if k in d:
                d[k] = float(d[k])
        cfg = cls(**d)
        if cfg.enabled:
            for k in ('span_axis', 'lift_axis'):
                if getattr(cfg, k) not in ('x', 'y', 'z'):
                    raise ConfigError(f'[postprocess.span_load] {k} must be x, y or z')
            if cfg.span_axis == cfg.lift_axis:
                raise ConfigError('[postprocess.span_load] span_axis and lift_axis differ')
            if not (cfg.n_stations >= 4 and cfg.semispan > 0 and cfg.aspect_ratio > 0):
                raise ConfigError('[postprocess.span_load] n_stations >= 4, semispan > 0 '
                                  'and aspect_ratio > 0 are needed')
        return cfg


@dataclass(frozen=True)
class PostprocessConfig:
    span_load: SpanLoadConfig = field(default_factory=SpanLoadConfig)

    @classmethod
    def from_dict(cls, data):
        check_keys(cls, data, '[postprocess]')
        return cls(span_load=SpanLoadConfig.from_dict(_sub(data, 'span_load')))


#: ``[[output.seed]] kind``: the restart a seed supplies.
SEED_KINDS = ('flow', 'adjoint')


@dataclass(frozen=True)
class SeedConfig:
    """One ``[[output.seed]]``: restarts copied into ``su2_workdir/``.

    ``source`` is the TOML key ``from``: a run id (its ``final/``) or a solver
    directory under the output root. ``points`` maps a destination point id to
    the source point id it starts from; empty means every point from its own id.
    """
    kind: str = ''
    source: str = ''
    points: Tuple[Tuple[str, str], ...] = ()

    @classmethod
    def from_dict(cls, data, index):
        ctx = f'[[output.seed]] {index}'
        if not isinstance(data, dict):
            raise ConfigError(f'{ctx} must be a table, got {data!r}')
        unknown = sorted(set(data) - {'kind', 'from', 'points'})
        if unknown:
            raise ConfigError(f'{ctx}: unknown key(s) {unknown}. '
                              f"Allowed: ['from', 'kind', 'points']")
        kind, source = data.get('kind'), data.get('from')
        if kind not in SEED_KINDS:
            raise ConfigError(f'{ctx}: kind must be one of {list(SEED_KINDS)}, '
                              f'got {kind!r}')
        if not isinstance(source, str) or not source:
            raise ConfigError(f'{ctx}: from must name a run id or a solver '
                              f'directory, got {source!r}')
        points = data.get('points', {})
        if not isinstance(points, dict) or not all(
                isinstance(k, str) and isinstance(v, str) for k, v in points.items()):
            raise ConfigError(f'{ctx}: points must map destination point ids to '
                              f'source point ids, got {points!r}')
        return cls(kind=kind, source=source, points=tuple(points.items()))


@dataclass(frozen=True)
class OutputConfig:
    """Per-run output directory settings."""
    root: str = 'output'
    run_id: str = 'auto'
    # fail / resume / overwrite
    on_exists: str = 'fail'
    seed: Tuple[SeedConfig, ...] = ()
    # log keeps each solve's SU2 output in logs/solver.log (the last 30 lines
    # of a successful solve, all of a failed one); discard keeps only the last
    # solve's, in logs/solver_last.out.
    solver_stdout: str = 'log'
    su2: SU2OutputConfig = field(default_factory=SU2OutputConfig)
    snapshots: SnapshotsConfig = field(default_factory=SnapshotsConfig)

    @classmethod
    def from_dict(cls, data):
        if 'per_rank_solver_log' in data:
            raise ConfigError('[output] per_rank_solver_log was removed; use '
                              'solver_stdout = "discard" or "log"')
        if 'seed_from' in data:
            raise ConfigError('[output] seed_from was replaced by [[output.seed]] '
                              'entries with kind, from and points')
        for key in ('plots', 'probes'):
            if key in data:
                raise ConfigError(f'[output.{key}] was removed; figures are drawn '
                                  'from files with run/plot.py')
        plain = {k: v for k, v in data.items()
                 if k not in ('su2', 'snapshots', 'seed')}
        check_keys(cls, plain, '[output]')
        seeds = data.get('seed', [])
        if not isinstance(seeds, list):
            raise ConfigError('[output] seed must be an array of tables, '
                              '[[output.seed]]')
        out = cls(su2=SU2OutputConfig.from_dict(_sub(data, 'su2')),
                  snapshots=SnapshotsConfig.from_dict(_sub(data, 'snapshots')),
                  seed=tuple(SeedConfig.from_dict(e, i) for i, e in enumerate(seeds)),
                  **plain)
        if not isinstance(out.root, str) or not out.root.strip():
            raise ConfigError('[output] root must be a non-empty path')
        if (not isinstance(out.run_id, str) or not out.run_id
                or out.run_id in ('.', '..')
                or os.path.basename(out.run_id) != out.run_id
                or '\\' in out.run_id):
            raise ConfigError('[output] run_id must name one directory under root')
        if out.solver_stdout not in ('discard', 'log'):
            raise ConfigError('[output] solver_stdout must be "discard" or "log"')
        return out


# --------------------------------------------------------------- top level

#: Keys every run TOML states, with no default.
REQUIRED_KEYS = ('execution.adjoint_restart', 'solver.primal.warm_start')


@dataclass(frozen=True)
class RunConfig:
    schema_version: int = SCHEMA_VERSION
    name: str = 'run'
    case: CaseConfig = field(default_factory=CaseConfig)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    points: Tuple[PointConfig, ...] = ()
    mission: Optional[Dict[str, Any]] = None
    dv: DVConfig = field(default_factory=DVConfig)
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    constraints: ConstraintsConfig = field(default_factory=ConstraintsConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    trim: TrimConfig = field(default_factory=TrimConfig)
    polar: PolarConfig = field(default_factory=PolarConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    deform: DeformConfig = field(default_factory=DeformConfig)
    remesh: RemeshConfig = field(default_factory=RemeshConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    verification: VerificationConfig = field(default_factory=VerificationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    postprocess: PostprocessConfig = field(default_factory=PostprocessConfig)
    # Bookkeeping, not user-facing: filled in by the loader.
    source: str = ''
    overrides: Tuple[str, ...] = ()

    # ---------------------------------------------------------- construction

    @classmethod
    def from_dict(cls, data, source=''):
        data = dict(data)
        data.pop('include', None)       # consumed by the loader
        top = {k: v for k, v in data.items()
               if k in ('schema_version', 'name', 'source', 'overrides')}
        nested = set(data) - set(top) - {'points'}
        allowed_nested = {f.name for f in fields(cls)} - set(top)
        unknown = sorted(nested - allowed_nested)
        if unknown:
            raise ConfigError(f'unknown top-level table(s) {unknown}. '
                              f'Allowed: {sorted(allowed_nested)}')

        version = int(top.get('schema_version', SCHEMA_VERSION))
        if version != SCHEMA_VERSION:
            raise ConfigError(
                f'schema_version {version} is not supported by this code '
                f'(expects {SCHEMA_VERSION})')

        mission = data.get('mission')
        constraints_data = _sub(data, 'constraints')
        raw_points = data.get('points', [])
        if mission is not None:
            from ..optimization.mission_points import mission_tables
            generated = mission_tables(mission)
            raw_points = generated['points']
            generated_linear = generated.get('constraints', {}).get('linear', [])
            if generated_linear:
                constraints_data = dict(constraints_data)
                constraints_data['linear'] = (list(constraints_data.get('linear', []))
                                              + generated_linear)
        if not isinstance(raw_points, list) or not raw_points:
            raise ConfigError(
                'at least one [[points]] entry is required (a single-point run '
                'is the N=1 case, not a separate config shape)')

        from . import toggles
        unset = [k for k in REQUIRED_KEYS if not toggles.find_value(data, k)[0]]
        if unset:
            raise ConfigError(f'{unset} must be set explicitly: each selects a '
                              f'restart behavior with a recorded hazard (adjoint restart, '
                              f'cold-start finite differences)')
        toggles.check(data)
        points = tuple(PointConfig.from_dict(p, i)
                       for i, p in enumerate(raw_points))

        geometry = GeometryConfig.from_dict(_sub(data, 'geometry'))
        return cls(
            schema_version=version,
            name=str(top.get('name', 'run')),
            source=source or str(top.get('source', '')),
            overrides=tuple(top.get('overrides', ())),
            case=CaseConfig.from_dict(_sub(data, 'case')),
            geometry=geometry,
            points=points,
            mission=mission,
            dv=DVConfig.from_dict(
                _sub(data, 'dv'), parameterization=geometry.parameterization),
            objective=ObjectiveConfig.from_dict(_sub(data, 'objective')),
            constraints=ConstraintsConfig.from_dict(constraints_data),
            optimizer=OptimizerConfig.from_dict(_sub(data, 'optimizer')),
            trim=TrimConfig.from_dict(_sub(data, 'trim')),
            polar=PolarConfig.from_dict(_sub(data, 'polar')),
            solver=SolverConfig.from_dict(_sub(data, 'solver')),
            deform=DeformConfig.from_dict(_sub(data, 'deform')),
            remesh=RemeshConfig.from_dict(_sub(data, 'remesh')),
            execution=ExecutionConfig.from_dict(_sub(data, 'execution')),
            verification=VerificationConfig.from_dict(_sub(data, 'verification')),
            output=OutputConfig.from_dict(_sub(data, 'output')),
            postprocess=PostprocessConfig.from_dict(_sub(data, 'postprocess')),
        )

    # -------------------------------------------------------------- derived

    @property
    def n_points(self):
        return len(self.points)

    @property
    def point_ids(self):
        return [p.id for p in self.points]

    @property
    def weights(self):
        """Weights normalised to sum 1."""
        from .points import weight_total
        total = weight_total(self.points)
        return [p.weight / total for p in self.points]

    @cached_property
    def objective_compiled(self):
        """The compiled ``[objective] expression``."""
        from ..optimization.expression import ExpressionError, compile_expression
        from ..solver.response_catalog import lookup
        obj = self.objective
        if obj.expression is None:
            raise ConfigError('[objective] expression is required, e.g. '
                              'expression = "wsum(drag)"')
        prefix = 'scenarios.' if self.execution.parallel_points else ''
        try:
            return compile_expression(obj.expression, obj.parameters, lookup,
                                      self.points, prefix=prefix)
        except ExpressionError as exc:
            raise ConfigError(f'[objective] expression {obj.expression!r}: '
                              f'{exc}') from None

    @cached_property
    def linear_compiled(self):
        """``((constraint, Compiled), ...)`` for every ``[[constraints.linear]]``.
        Read only after the load-time checks have passed."""
        from ..optimization.expression import compile_linear
        from ..solver.response_catalog import lookup
        prefix = 'scenarios.' if self.execution.parallel_points else ''
        return tuple((c, compile_linear(c, lookup, self.points, prefix=prefix))
                     for c in self.constraints.linear)

    @property
    def geometry_dvs(self):
        """Geometry design-variable names in DV-vector order."""
        return tuple(name for name, on in (('shape', self.dv.shape.enabled),
                                           ('twist', self.dv.twist.enabled)) if on)

    @property
    def objective_formulation(self):
        """The ``formulation`` block of ``resolved_config.json``."""
        from ..solver.response_catalog import lookup
        from .toggles import states_of as toggles_states_of
        from .points import weight_total
        compiled = self.objective_compiled
        weighted = [p for p in self.points if p.weight > 0.0]
        total = weight_total(weighted)
        from .physics import profile_id
        profile = profile_id(self.case.physics)
        responses = []
        for name in compiled.responses:
            record = lookup(name)
            responses.append({'name': name, 'output': record.output,
                              'profile': profile,
                              'status': record.statuses.get(profile, 'assumed'),
                              'evidence': record.evidence.get(profile)})
        linear = []
        for constraint, source in self.linear_compiled:
            terms = []
            for term in constraint.terms:
                record = lookup(term.response)
                terms.append({'point': term.point, 'response': term.response,
                              'output': record.output, 'coeff': term.coeff,
                              'status': record.statuses.get(profile, 'assumed')})
            linear.append({'name': constraint.name, 'path': constraint.path,
                           'terms': terms, 'offset': constraint.offset,
                           **constraint.constraint_kwargs,
                           'rendered': source.rendered,
                           'execcomp_source': source.execcomp_src})
        return {
            'schema_version': 1,
            'expression': compiled.text,
            'scaler': self.objective.scaler,
            'parameters': dict(compiled.parameters),
            'weight_total': total,
            'points': [{'id': p.id, 'index': i, 'weight': p.weight,
                        'normalized_weight': p.weight / total if p.weight > 0 else 0.0}
                       for i, p in enumerate(self.points)],
            'base_functionals': {k: list(v) for k, v in
                                 compiled.base_functionals.items()},
            'connections': [list(c) for c in compiled.connections],
            'rendered': compiled.rendered,
            'execcomp_source': compiled.execcomp_src,
            'responses': responses,
            'linear_constraints': linear,
            'toggles': toggles_states_of(self),
        }

    @property
    def objective_expr(self):
        """``ExecComp`` source for ``obj_comb`` (weights enter once, through wsum)."""
        return self.objective_compiled.execcomp_src

    @property
    def objective_inputs(self):
        """``ExecComp`` input names, ``<output>_<point index>``."""
        return list(self.objective_compiled.inputs)

    @property
    def aero_functionals(self):
        """SU2AeroComp outputs that forward-mode derivatives must cover.

        Forward mode cannot see which outputs a response needs, so an output
        left out gets a silent zero derivative. This is a
        superset of every output a task differentiates.
        """
        task = self.execution.task
        names = {output for outputs in
                 self.objective_compiled.base_functionals.values()
                 for output in outputs}
        if (self.constraints.lift.enabled
                or task in ('compute_totals', 'check_totals')):
            names.add('C_L')
        if self.constraints.moment.enabled or task == 'moment_pivot_check':
            names.add('C_M')
        for _constraint, compiled in self.linear_compiled:
            names.update(output for outputs in compiled.base_functionals.values()
                         for output in outputs)
        names.update(self.verification.alpha_outputs)
        names.update(self.verification.shape_outputs)
        legacy = tuple(n for n in ('C_D', 'C_L', 'C_M') if n in names)
        from ..solver.response_catalog import lookup
        extra = tuple(lookup(name).output for point in self.points
                      for name in point.responses)
        return legacy + tuple(name for name in dict.fromkeys(extra)
                              if name not in legacy)

    @property
    def resolved_points(self):
        from .points import resolve_points
        return resolve_points(self)

    @property
    def solver_blocks(self):
        """(primal, adjoint) with remesh overrides merged (explicit keys win)."""
        if not self.remesh.enabled:
            return self.solver.primal, self.solver.adjoint
        r = self.remesh.solver

        def merged(base, remesh_values):
            values = {}
            for name, value in remesh_values.items():
                values[name] = (getattr(base, name)
                                if name in base._explicit_keys else value)
            for name in ('cfl_adapt', 'warm_start'):
                values[name] = getattr(base, name)   # no remesh counterpart
            return SolverBlockConfig(_explicit_keys=base._explicit_keys, **values)

        primal = merged(self.solver.primal, {
            'iter': r.primal_iter,
            'mglevel': r.primal_mglevel,
            'cfl_adapt_param': r.cfl_adapt_param,
            'conv_residual_minval': r.primal_conv_residual_minval})
        adjoint = merged(self.solver.adjoint, {
            'iter': r.adjoint_iter,
            'mglevel': r.adjoint_mglevel,
            'cfl_adapt_param': r.cfl_adapt_param,
            'conv_residual_minval': r.adjoint_conv_residual_minval})
        return primal, adjoint

    # ---------------------------------------------------------------- paths

    @property
    def case_root(self):
        from ..paths import REPO
        return os.path.join(REPO, self.case.root)

    def abs_path(self, rel):
        """Absolute path for a case-root-relative entry (absolute passes through)."""
        return rel if os.path.isabs(rel) else os.path.join(self.case_root, rel)

    @property
    def mesh_path(self):
        return self.abs_path(self.case.mesh)

    @property
    def ffd_path(self):
        return self.abs_path(self.geometry.ffd_file)

    @property
    def dat_path(self):
        return self.abs_path(self.geometry.dat_file) if self.geometry.dat_file else ''

    @property
    def thickness_stations(self):
        """``{x/c: min t/c}`` -- the shape ``geometry/ffd.py`` consumes."""
        return {x: t for x, t in self.constraints.thickness.stations}

    def to_dict(self):
        """Plain dict for provenance JSON. Strips private _explicit_keys fields."""
        def clean(value):
            if isinstance(value, dict):
                return {k: clean(v) for k, v in value.items()
                        if not k.startswith('_')}
            if isinstance(value, (list, tuple)):
                return type(value)(clean(v) for v in value)
            return value
        return clean(asdict(self))


__all__ = [
    'normal_force_slope', 'check_keys', 'OPT_DEFAULTS',
    'SCHEMA_VERSION', 'TASKS', 'OPTIMIZERS', 'SHAPE_PATTERNS', 'ConfigError',
    'RunConfig', 'CaseConfig', 'GeometryConfig', 'PointConfig',
    'DVConfig', 'ShapeDVConfig', 'AlphaDVConfig', 'ObjectiveConfig',
    'LIFT_KINDS',
    'MOMENT_KINDS', 'MOMENT_AXES',
    'ConstraintsConfig', 'LiftConstraintConfig', 'MomentConstraintConfig',
    'LinearConstraintConfig', 'LinearTermConfig',
    'VolumeConstraintConfig',
    'LERadiusConstraintConfig', 'ThicknessConstraintConfig',
    'CurvatureConstraintConfig',
    'OptimizerConfig', 'TrimConfig', 'PolarConfig', 'SolverConfig', 'SolverBlockConfig',
    'FreestreamConfig', 'ReferenceConfig', 'BoundaryConfig', 'PhysicsConfig',
    'TurbulenceConfig',
    'NumericsConfig', 'LinearSolverConfig', 'MultigridConfig',
    'ConvergenceConfig', 'AdjointConfig', 'DeformConfig',
    'SU2OutputConfig',
    'RemeshConfig', 'RemeshSolverConfig', 'RemeshQualityConfig',
    'ExecutionConfig', 'VerificationConfig', 'OutputConfig', 'SeedConfig',
    'SEED_KINDS', 'shape_dv_bounds',
]
