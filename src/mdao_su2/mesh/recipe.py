"""The ``[mesh]`` recipe and CLI-over-recipe precedence.

A recipe is a TOML file with one table per step (``[mesh.extract]``,
``[mesh.airfoil]``, ``[mesh.ffd]``, ...). Each step is a toggle in
``run_config/toggles.py``: it is off unless ``enabled = true`` (or
``run/mesh.py <step> --enable``), and an enabled step states every value it
reads. A step may hold named levels, ``[mesh.<step>.levels.<name>]``, whose
values replace the step's own for ``--level <name>``. Precedence: CLI flag,
then the level table, then the step table; a value missing from all three is
an error that names the recipe key and the flag. ``[mesh.outputs]`` names the
files ``run/mesh.py all`` writes (:func:`output_path`). Nothing here has a
default value.
"""

import dataclasses
import hashlib
import os
import tomllib
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from ..run_config import toggles


class RecipeError(ValueError):
    """A recipe or CLI value is missing, unknown, or inconsistent."""


@dataclass(frozen=True)
class ExtractStep:
    """``[mesh.extract]``: wall markers of an SU2 mesh (path relative to the recipe)."""
    mesh: str
    markers: Tuple[str, ...]


@dataclass(frozen=True)
class AirfoilStep:
    source: str
    n_points: int
    spacing: str
    te: str
    #: ``bigeometric`` end spacings (chord units) and growth ratio, stated together.
    le_spacing: Optional[float] = None
    te_spacing: Optional[float] = None
    spacing_ratio: Optional[float] = None


@dataclass(frozen=True)
class FFDStep:
    kind: str
    nx: int
    spacing: str
    zspan: Optional[float] = None
    x_bounds: Optional[Tuple[float, float]] = None
    y_bounds: Optional[Tuple[float, float]] = None
    z_bounds: Optional[Tuple[float, float]] = None
    x_margin: Optional[float] = None
    y_margin: Optional[float] = None
    le_te: Optional[Tuple[float, float]] = None
    n_span: Optional[int] = None
    nz: Optional[int] = None


@dataclass(frozen=True)
class Mesh2DStep:
    """``[mesh.mesh2d]``: ``RemeshConfig.gmsh_kwargs()`` names plus markers and gates."""
    first_cell_height: float
    growth_ratio: float
    fan_angle_deg: float
    fan_elements: int
    farfield_radius: float
    farfield_meshsize: float
    near_meshsize: float
    far_growth_slope: float
    wall_meshsize: float
    wall_marker: str
    farfield_marker: str
    farfield_center: Tuple[float, float]
    quality: Dict[str, float]
    n_bl_layers: Optional[int] = None
    bl_thickness: Optional[float] = None
    #: Generations allowed until the quality gates pass. gmsh's 2D mesher returns
    #: one of several meshes for the same input; each retry runs in a
    #: fresh process. Unset means one generation.
    attempts: Optional[int] = None
    #: Optional wake wedge in the size field; ``wake_length`` > 0 needs the other three.
    wake_length: Optional[float] = None
    wake_angles_deg: Optional[Tuple[float, float]] = None
    wake_meshsize: Optional[float] = None
    wake_stream_slope: Optional[float] = None

    def remesh_config(self):
        """The same values as a ``RemeshConfig``, for ``gmsh_kwargs`` and ``quality.check``."""
        from ..run_config.root import RemeshConfig
        d = {k: getattr(self, k) for k in (
            'first_cell_height', 'growth_ratio', 'fan_angle_deg', 'fan_elements',
            'farfield_radius', 'farfield_meshsize', 'near_meshsize',
            'far_growth_slope', 'wall_meshsize')}
        d['mode'] = 'gmsh'
        if self.bl_thickness is not None:
            d['bl_thickness'] = self.bl_thickness
        if self.n_bl_layers is not None:
            d['n_bl_layers'] = self.n_bl_layers
        for k in ('wake_length', 'wake_angles_deg', 'wake_meshsize', 'wake_stream_slope'):
            if getattr(self, k) is not None:
                d[k] = getattr(self, k)
        d['quality'] = dict(self.quality)
        return RemeshConfig.from_dict(d)

    def gmsh_kwargs(self):
        """Keyword arguments for ``mesh2d.remesh_airfoil``."""
        kw = self.remesh_config().gmsh_kwargs()
        kw.update(wall_marker=self.wall_marker, farfield_marker=self.farfield_marker,
                  farfield_center=tuple(self.farfield_center))
        return kw


@dataclass(frozen=True)
class Wing3DStep:
    """``[mesh.wing3d]``: half-wing geometry, sizes at level scale 1, markers, gates."""
    span: float
    tip: str
    farfield_radius: float
    farfield_center: Tuple[float, float, float]
    wall_size: float
    le_size: float
    te_size: float
    tip_size: float
    far_size: float
    size_growth: float
    edge_band: float
    te_band: float
    te_normal_scale: float
    edge_distance: str
    level_scale: Dict[str, float]
    wing_marker: str
    tip_marker: str
    symmetry_marker: str
    farfield_marker: str
    quality: Dict[str, float]

    TIPS = ('revolved',)

    @property
    def markers(self):
        return (self.wing_marker, self.tip_marker, self.symmetry_marker,
                self.farfield_marker)

    def generate_kwargs(self, level):
        if level not in self.level_scale:
            raise RecipeError(f'--level {level!r} is not in [mesh.wing3d] level_scale '
                              f'{sorted(self.level_scale)}')
        kw = {k: getattr(self, k) for k in (
            'span', 'farfield_radius', 'farfield_center', 'wall_size', 'le_size',
            'te_size', 'tip_size', 'far_size', 'size_growth', 'edge_band', 'te_band',
            'te_normal_scale', 'edge_distance',
            'wing_marker', 'tip_marker', 'symmetry_marker', 'farfield_marker')}
        kw['level_scale'] = float(self.level_scale[level])
        return kw


@dataclass(frozen=True)
class Section2DStep:
    """``[mesh.section2d]``: a wing3d span station in 2D -- sizes at level scale 1,
    D-domain, markers, gates (``mesh/section2d.py``)."""
    farfield_radius: float
    farfield_center: Tuple[float, float]
    wall_size: float
    le_size: float
    te_size: float
    far_size: float
    size_growth: float
    edge_band: float
    level_scale: Dict[str, float]
    wall_marker: str
    farfield_marker: str
    quality: Dict[str, float]

    def generate_kwargs(self, level):
        if level not in self.level_scale:
            raise RecipeError(f'--level {level!r} is not in [mesh.section2d] level_scale '
                              f'{sorted(self.level_scale)}')
        kw = {k: getattr(self, k) for k in (
            'farfield_radius', 'farfield_center', 'wall_size', 'le_size', 'te_size',
            'far_size', 'size_growth', 'edge_band', 'wall_marker', 'farfield_marker')}
        kw['level_scale'] = float(self.level_scale[level])
        return kw


STEPS = {'extract': ExtractStep, 'airfoil': AirfoilStep, 'ffd': FFDStep, 'mesh2d': Mesh2DStep,
         'wing3d': Wing3DStep, 'section2d': Section2DStep}
_PAIRS = ('x_bounds', 'y_bounds', 'z_bounds', 'le_te', 'wake_angles_deg')
_INTS = ('n_points', 'nx', 'n_span', 'nz', 'n_bl_layers', 'fan_elements', 'attempts')


@dataclass(frozen=True)
class Resolved:
    """One step's effective values and where each came from."""
    step: str
    values: Any                       # one of the STEPS dataclasses
    sources: Dict[str, str] = field(default_factory=dict)   # key -> 'cli' | 'recipe[:levels.<name>]'
    recipe: Optional[str] = None


def flag(key):
    if key.startswith('quality.'):
        return f'--quality {key.split(".", 1)[1]}=<value>'
    if key == 'level_scale':
        return '--level-scale <level>=<value>'
    if '.' in key:
        return 'recipe only'
    return '--' + key.replace('_', '-')


def load_recipe(path):
    """The ``[mesh]`` table of a recipe TOML (``{}`` for ``None``)."""
    if path is None:
        return {}
    with open(path, 'rb') as fh:
        data = tomllib.load(fh)
    extra = sorted(set(data) - {'mesh'})
    if extra:
        raise RecipeError(f'{path}: a recipe holds only [mesh.*] tables; found {extra}')
    mesh = data.get('mesh', {})
    unknown = sorted(set(mesh) - set(STEPS) - {'outputs'})
    if unknown:
        raise RecipeError(f'{path}: unknown recipe step(s) {unknown}')
    return mesh


def levels(recipe, steps=None):
    """Level names declared by ``[mesh.<step>.levels.*]`` tables, in recipe order."""
    names = []
    for step, table in recipe.items():
        if step in STEPS and (steps is None or step in steps):
            names += [n for n in table.get('levels', {}) if n not in names]
    return names


def _level_table(step, table, level, recipe_path):
    """``table`` with ``levels.<level>`` merged over it (nested tables merge per key)."""
    table = dict(table)
    tables = table.pop('levels', {})
    if level is None or not tables:
        return table
    if level not in tables:
        raise RecipeError(f'{recipe_path}: --level {level!r} is not in '
                          f'[mesh.{step}.levels] {sorted(tables)}')
    for key, value in tables[level].items():
        if isinstance(value, dict) and isinstance(table.get(key), dict):
            table[key] = {**table[key], **value}
        else:
            table[key] = value
    return table


def output_path(recipe, recipe_path, kind, level=None, base_dir=None):
    """The file ``[mesh.outputs] <kind>`` names, relative to ``base_dir`` (the recipe's).

    Patterns take ``{case}`` and ``{set}`` from ``[mesh.outputs]`` and ``{level}``
    from the run, e.g. ``surface = "../geom/{case}_surface_{set}_{level}.dat"``.
    """
    outputs = recipe.get('outputs', {})
    if kind not in outputs:
        raise RecipeError(f'{recipe_path}: [mesh.outputs] {kind} names the file this '
                          f'step writes; the recipe owns the names')
    fields = {k: v for k, v in outputs.items() if k in ('case', 'set')}
    try:
        name = outputs[kind].format(level=level, **fields)
    except KeyError as exc:
        raise RecipeError(f'{recipe_path}: [mesh.outputs] {kind} uses {exc}, which is '
                          f'neither case, set nor level') from None
    if '{level}' in outputs[kind] and level is None:
        raise RecipeError(f'{recipe_path}: [mesh.outputs] {kind} needs --level')
    base = base_dir or os.path.dirname(os.path.abspath(recipe_path))
    return os.path.normpath(os.path.join(base, name))


def resolve(step, recipe_path, cli, enable=False, level=None):
    """Merge ``cli`` (flag values; ``None`` = not given) over the recipe step.

    ``level`` selects ``[mesh.<step>.levels.<level>]``, which sits between
    the CLI and the step table. Raises :class:`RecipeError` when the step is
    off, a key or level is unknown, or an enabled step leaves a value it reads
    unset.
    """
    cls = STEPS[step]
    raw = load_recipe(recipe_path).get(step, {})
    level_keys = set(raw.get('levels', {}).get(level, {})) if level else set()
    table = _level_table(step, raw, level, recipe_path)
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = sorted(set(table) - known - {'enabled'})
    if unknown:
        raise RecipeError(f'[mesh.{step}] unknown key(s) {unknown}; allowed: '
                          f'{sorted(known)}')
    t = toggles.BY_BLOCK[f'mesh.{step}']
    on = table.get('enabled') is True or enable
    if not on:
        raise RecipeError(
            f'[mesh.{step}] is not enabled: set enabled = true in the recipe or '
            f'pass --enable to permit generation')
    merged, sources = {}, {}
    for key in known:
        if cli.get(key) is not None:
            merged[key], sources[key] = cli[key], 'cli'
            if isinstance(cli[key], dict) and isinstance(table.get(key), dict):
                merged[key] = {**table[key], **cli[key]}      # CLI per key over the recipe
        elif key in table:
            merged[key] = table[key]
            sources[key] = f'recipe:levels.{level}' if key in level_keys else 'recipe'
    probe = {'mesh': {step: dict(merged, enabled=True)}}
    gaps = [key[len(f'mesh.{step}.'):] for _, key in toggles.missing(probe, scope='recipe')
            if key.startswith(f'mesh.{step}.')]
    if gaps:
        lines = '\n'.join(f'  {k}  (recipe [mesh.{step}] {k}, or {flag(k)})' for k in gaps)
        raise RecipeError(f'[mesh.{step}] needs values it has no default for '
                          f' with these missing values:\n{lines}')
    for key in _PAIRS:
        if merged.get(key) is not None:
            pair = tuple(float(v) for v in merged[key])
            if len(pair) != 2 or not pair[0] < pair[1]:
                raise RecipeError(f'[mesh.{step}] {key} must be [low, high] with '
                                  f'low < high; got {merged[key]!r}')
            merged[key] = pair
    for key in ('le_spacing', 'te_spacing', 'spacing_ratio', 'wake_length',
                'wake_meshsize', 'wake_stream_slope'):
        if merged.get(key) is not None:
            merged[key] = float(merged[key])
    for key in _INTS:
        if merged.get(key) is not None:
            merged[key] = int(merged[key])
    if step == 'extract':
        markers = merged['markers']
        merged['markers'] = (markers,) if isinstance(markers, str) else tuple(markers)
    if step == 'ffd' and merged['kind'] == 'wing':
        (x_lo, x_hi), x_te = merged['x_bounds'], merged['le_te'][1]
        if not x_lo < x_te < x_hi:
            raise RecipeError(f'[mesh.ffd] the le_te trailing edge x = {x_te} must lie '
                              f'inside x_bounds {list(merged["x_bounds"])}')
    if step == 'airfoil':
        given = [k for k in ('le_spacing', 'te_spacing', 'spacing_ratio')
                 if merged.get(k) is not None]
        if merged['spacing'] != 'bigeometric' and given:
            raise RecipeError(
                f'[mesh.airfoil] {given} apply to spacing = "bigeometric" only: the '
                f'other spacings (cosine, conical, polynomial) have no end-spacing '
                f"or ratio parameters, so {merged['spacing']!r} would ignore them")
        if merged['spacing'] == 'bigeometric' and given:
            if len(given) < 3:
                raise RecipeError(
                    f'[mesh.airfoil] le_spacing, te_spacing and spacing_ratio are '
                    f'stated together; missing '
                    f'{sorted({"le_spacing", "te_spacing", "spacing_ratio"} - set(given))}')
            if not (merged['le_spacing'] > 0 and merged['te_spacing'] > 0
                    and merged['spacing_ratio'] > 1):
                raise RecipeError('[mesh.airfoil] le_spacing and te_spacing must be > 0 '
                                  'and spacing_ratio > 1')
    if step == 'mesh2d':
        if (merged.get('wake_length') or 0) > 0:
            gaps = [k for k in ('wake_angles_deg', 'wake_meshsize', 'wake_stream_slope')
                    if merged.get(k) is None]
            if gaps:
                raise RecipeError(f'[mesh.mesh2d] wake_length > 0 also needs {gaps}')
        merged['farfield_center'] = tuple(float(v) for v in merged['farfield_center'])
        extra = sorted(set(merged['quality']) - set(toggles.MESH2D_QUALITY))
        if extra:
            raise RecipeError(f'[mesh.mesh2d.quality] unknown key(s) {extra}')
        if merged.get('attempts') is not None and merged['attempts'] < 1:
            raise RecipeError(f"[mesh.mesh2d] attempts = {merged['attempts']} must be >= 1")
        values = cls(**merged)
        from ..run_config.checks.remesh import remesh_value_errors
        errors = remesh_value_errors(values.remesh_config(), '[mesh.mesh2d]')
        if errors:
            raise RecipeError('\n'.join(errors))
        return Resolved(step, values, sources, recipe_path)
    if step == 'wing3d':
        merged['farfield_center'] = tuple(float(v) for v in merged['farfield_center'])
        if len(merged['farfield_center']) != 3:
            raise RecipeError('[mesh.wing3d] farfield_center is [x, y, z]')
        from .wing3d import EDGE_DISTANCES
        if merged['edge_distance'] not in EDGE_DISTANCES:
            raise RecipeError(f"[mesh.wing3d] edge_distance={merged['edge_distance']!r} "
                              f'must be one of {list(EDGE_DISTANCES)}')
        if merged['tip'] not in Wing3DStep.TIPS:
            raise RecipeError(f"[mesh.wing3d] tip={merged['tip']!r} must be one of "
                              f'{list(Wing3DStep.TIPS)}')
        extra = sorted(set(merged['quality']) - set(toggles.WING3D_QUALITY))
        if extra:
            raise RecipeError(f'[mesh.wing3d.quality] unknown key(s) {extra}')
        sizes = ('span', 'farfield_radius', 'wall_size', 'le_size', 'te_size',
                 'tip_size', 'far_size', 'size_growth', 'edge_band', 'te_band',
                 'te_normal_scale')
        bad = [k for k in sizes if not float(merged[k]) > 0.0]
        bad += [f'level_scale.{k}' for k, v in merged['level_scale'].items()
                if not float(v) > 0.0]
        if bad:
            raise RecipeError(f'[mesh.wing3d] {bad} must be > 0')
        if not merged['wall_size'] <= merged['far_size']:
            raise RecipeError('[mesh.wing3d] wall_size must not exceed far_size')
        if merged['te_band'] < merged['edge_band']:
            raise RecipeError('[mesh.wing3d] te_band must be >= edge_band')
        return Resolved(step, cls(**merged), sources, recipe_path)
    if step == 'section2d':
        merged['farfield_center'] = tuple(float(v) for v in merged['farfield_center'])
        if len(merged['farfield_center']) != 2:
            raise RecipeError('[mesh.section2d] farfield_center is [x, y]')
        extra = sorted(set(merged['quality']) - set(toggles.SECTION2D_QUALITY))
        if extra:
            raise RecipeError(f'[mesh.section2d.quality] unknown key(s) {extra}')
        sizes = ('farfield_radius', 'wall_size', 'le_size', 'te_size', 'far_size',
                 'size_growth', 'edge_band')
        bad = [k for k in sizes if not float(merged[k]) > 0.0]
        bad += [f'level_scale.{k}' for k, v in merged['level_scale'].items()
                if not float(v) > 0.0]
        if bad:
            raise RecipeError(f'[mesh.section2d] {bad} must be > 0')
        if not merged['wall_size'] <= merged['far_size']:
            raise RecipeError('[mesh.section2d] wall_size must not exceed far_size')
        coarse = [k for k in ('le_size', 'te_size') if merged[k] > merged['wall_size']]
        if coarse:
            raise RecipeError(f'[mesh.section2d] {coarse} must not exceed wall_size: '
                              f'the edge fields grade from them up to wall_size')
        return Resolved(step, cls(**merged), sources, recipe_path)
    return Resolved(step, cls(**merged), sources, recipe_path)


def sha256(path):
    """Hex SHA-256 of a file; the run directory records it for its seed meshes."""
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()
