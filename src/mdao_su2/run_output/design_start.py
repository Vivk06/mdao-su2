"""Reader of ``[dv] start``: the recorded design a run starts from.

A source is a run id, whose ``results_run_driver.json`` holds
``optimizer.final_design`` (unscaled, by DV name), or an evaluation folder
``<run>/evaluations/<name>/`` with its ``geometry/design_variables.npy`` and
``call_summary.json``. Both resolve under the case's output root only. The
source run's ``run.toml`` must describe the same parameterization, so a vector
of the right length from another basis is rejected. Shape and twist values
enter the model unchanged; alpha maps by point id. The pyOptSparse history is
scaled and is not a source.
"""

import json
import os
import re
import tomllib
from dataclasses import asdict, dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

from ..mesh.recipe import sha256
from ..paths import REPO
from ..run_config.errors import ConfigError
from ..run_config.root import shape_dv_bounds

__all__ = ['DesignStart', 'read_start', 'start_record', 'start_lines',
           'check_shape_start']

#: ``[dv.shape]`` keys that bound or scale the DVs; every other key defines the basis.
_BOUND_KEYS = ('bound', 'bounds', 'scaler', 'outward_only')
#: ``[geometry]`` keys that define each shape basis: the CST order, the FFD DV order.
_GEOMETRY_BASIS = {'cst': ('n_cst',), 'ffd': ('ffd_order',)}
#: ``[dv.twist]`` keys that define the twist basis.
_TWIST_BASIS = ('n', 'include_root', 'axis_x_fraction')
_ALPHA_DV = re.compile(r'alpha_\d+$')


@dataclass(frozen=True)
class DesignStart:
    """One resolved ``[dv] start``."""
    source: str
    files: Tuple[str, ...]                     # absolute paths read for the values
    values: Dict[str, np.ndarray]              # geometry DV name -> unscaled values
    alphas: Dict[str, Tuple[str, float]]       # destination id -> (source id, degrees)
    kept: Tuple[str, ...]                      # design points keeping aoa_init
    fixed: Tuple[str, ...] = field(default=())  # fixed-alpha points, never mapped


def _resolve(cfg, source):
    """``(run root, evaluation folder or None)`` of ``source`` under the output root."""
    from .rundir import source_path
    path = source_path(cfg, source, '[dv] start')
    root = os.path.realpath(cfg.abs_path(cfg.output.root))
    parts = os.path.relpath(path, root).split(os.sep)
    run_root = os.path.join(root, parts[0])
    if len(parts) == 1:
        return run_root, None
    if len(parts) >= 3 and parts[1] == 'evaluations':
        return run_root, path
    raise ConfigError(f'[dv] start={source!r} names neither a run id nor an '
                      f'evaluation folder <run>/evaluations/<name>/')


def _source_config(run_root, source):
    """The source run's configuration, read from its ``run.toml``."""
    from ..run_config.root import RunConfig
    path = os.path.join(run_root, 'run.toml')
    if not os.path.isfile(path):
        raise ConfigError(f'[dv] start={source!r}: {path} is missing, so the source '
                          f'parameterization cannot be checked')
    try:
        with open(path, 'rb') as stream:
            return RunConfig.from_dict(tomllib.load(stream))
    except (ConfigError, tomllib.TOMLDecodeError, TypeError, ValueError) as exc:
        raise ConfigError(f'[dv] start={source!r}: {path} does not load: {exc}') from None


def _basis(shape_cfg):
    return {k: v for k, v in asdict(shape_cfg).items() if k not in _BOUND_KEYS}


def _parameterization_errors(cfg, src):
    """Why ``src``'s design cannot be applied to ``cfg``; empty when it can."""
    errors = []
    if src.geometry_dvs != cfg.geometry_dvs:
        errors.append(f'geometry DVs {list(src.geometry_dvs)} in the source, '
                      f'{list(cfg.geometry_dvs)} here')
    kind = cfg.geometry.parameterization
    if 'shape' in cfg.geometry_dvs:
        if src.geometry.parameterization != kind:
            errors.append(f'shape kind {src.geometry.parameterization!r} in the source, '
                          f'{kind!r} here')
        elif _basis(src.dv.shape) != _basis(cfg.dv.shape):
            diff = sorted(k for k, v in _basis(cfg.dv.shape).items()
                          if _basis(src.dv.shape).get(k) != v)
            errors.append(f'[dv.shape] {diff} differ from the source')
        diff = [k for k in _GEOMETRY_BASIS.get(kind, ())
                if getattr(src.geometry, k) != getattr(cfg.geometry, k)]
        if diff and src.geometry.parameterization == kind:
            errors.append(f'[geometry] {diff} differ from the source')
        files = {'ffd': ('FFD file', 'ffd_path'), 'cst': ('dat file', 'dat_path')}
        if kind in files and src.geometry.parameterization == kind:
            label, attr = files[kind]
            mine, theirs = getattr(cfg, attr), getattr(src, attr)
            if not (os.path.isfile(mine) and os.path.isfile(theirs)):
                errors.append(f'{label} missing: {theirs if os.path.isfile(mine) else mine}')
            elif sha256(mine) != sha256(theirs):
                errors.append(f'{label} {os.path.relpath(theirs, REPO)} differs from '
                              f'{os.path.relpath(mine, REPO)} (SHA-256)')
    if 'twist' in cfg.geometry_dvs and 'twist' in src.geometry_dvs:
        diff = [k for k in _TWIST_BASIS
                if getattr(src.dv.twist, k) != getattr(cfg.dv.twist, k)]
        if diff:
            errors.append(f'[dv.twist] {diff} differ from the source')
    return errors


def _from_run(run_root, cfg, source):
    path = os.path.join(run_root, 'results_run_driver.json')
    if not os.path.isfile(path):
        raise ConfigError(f'[dv] start={source!r}: the run has no '
                          f'results_run_driver.json; name an evaluation folder '
                          f'<run>/evaluations/<name>/ instead')
    with open(path) as stream:
        data = json.load(stream)
    design = (data.get('optimizer') or {}).get('final_design')
    if not isinstance(design, dict):
        raise ConfigError(f'[dv] start={source!r}: {path} has no '
                          f'optimizer.final_design')
    extra = sorted(k for k in design
                   if k not in cfg.geometry_dvs and not _ALPHA_DV.match(k))
    missing = sorted(set(cfg.geometry_dvs) - set(design))
    if extra or missing:
        raise ConfigError(f'[dv] start={source!r}: final_design names '
                          f'{sorted(design)}; missing {missing}, unknown {extra}')
    values = {name: design[name] for name in cfg.geometry_dvs}
    alphas = {p.get('id'): p.get('alpha') for p in data.get('points') or []}
    return (path,), values, alphas


def _from_evaluation(folder, cfg, source):
    summary_path = os.path.join(folder, 'call_summary.json')
    if not os.path.isfile(summary_path):
        raise ConfigError(f'[dv] start={source!r}: {summary_path} is missing')
    with open(summary_path) as stream:
        summary = json.load(stream)
    if summary.get('status') == 'failed':
        raise ConfigError(f'[dv] start={source!r}: the evaluation failed '
                          f'({summary.get("failure")})')
    files, values = [summary_path], {}
    if 'shape' in cfg.geometry_dvs:
        path = os.path.join(folder, 'geometry', 'design_variables.npy')
        if not os.path.isfile(path):
            raise ConfigError(f'[dv] start={source!r}: {path} is missing')
        try:
            values['shape'] = np.load(path, allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise ConfigError(f'[dv] start={source!r}: {path} is not a numeric '
                              f'.npy: {exc}') from None
        files.append(path)
    if 'twist' in cfg.geometry_dvs:
        record = (((summary.get('geometry_constraints') or {})
                   .get('design_variables') or {}).get('twist') or {})
        if 'value' not in record:
            raise ConfigError(f'[dv] start={source!r}: {summary_path} records no twist')
        values['twist'] = record['value']
    alphas = {p.get('id'): p.get('alpha') for p in summary.get('points') or []}
    return tuple(files), values, alphas


def _outside(value, lower, upper):
    """Indices of ``value`` outside ``[lower, upper]`` (scalar or per entry)."""
    lo = np.broadcast_to(np.asarray(lower, dtype=float), value.shape)
    hi = np.broadcast_to(np.asarray(upper, dtype=float), value.shape)
    tol = 1e-12 * np.maximum(1.0, np.abs(hi - lo))
    return np.nonzero((value < lo - tol) | (value > hi + tol))[0].tolist()


def check_shape_start(cfg, value, counts=None):
    """Errors for a start shape vector against this configuration's bounds.

    Without ``counts`` a multi-direction pattern cannot be bounded yet; the
    model repeats the check with the embedded geometry's counts.
    """
    bounds = shape_dv_bounds(cfg.dv.shape, cfg.geometry.parameterization, value.size, counts)
    if bounds is None:
        return []
    out = _outside(value, *bounds)
    return [f'shape DVs {out} lie outside their bounds'] if out else []


def read_start(cfg):
    """The resolved ``[dv] start`` of ``cfg``, or None when unset.

    Raises ``ConfigError`` listing every problem: another parameterization,
    missing or extra DVs, a wrong twist count, a non-finite value, or a value
    outside this configuration's bounds.
    """
    source = cfg.dv.start
    if not source:
        return None
    run_root, folder = _resolve(cfg, source)
    problems = _parameterization_errors(cfg, _source_config(run_root, source))
    if problems:
        raise ConfigError(f'[dv] start={source!r} has another parameterization: '
                          + '; '.join(problems))
    files, raw, source_alphas = (_from_run(run_root, cfg, source) if folder is None
                                 else _from_evaluation(folder, cfg, source))
    errors, values = [], {}
    for name, value in raw.items():
        array = np.asarray(value, dtype=float).ravel()
        if not np.all(np.isfinite(array)):
            errors.append(f'{name} holds non-finite values')
        values[name] = array
    if 'twist' in values:
        twist = cfg.dv.twist
        if values['twist'].size != twist.n_dv:
            errors.append(f'twist has {values["twist"].size} values; this '
                          f'configuration has {twist.n_dv}')
        else:
            out = _outside(values['twist'], twist.lower_deg, twist.upper_deg)
            if out:
                errors.append(f'twist DVs {out} lie outside [{twist.lower_deg}, '
                              f'{twist.upper_deg}] deg')
    if 'shape' in values and not errors:
        errors += check_shape_start(cfg, values['shape'])
    alphas, kept, fixed = {}, [], []
    for point in cfg.points:
        if point.alpha != 'design':
            fixed.append(point.id)
            continue
        value = source_alphas.get(point.id)
        if value is None:
            kept.append(point.id)
            continue
        value = float(value)
        if not np.isfinite(value):
            errors.append(f'alpha of point {point.id!r} is not finite')
        elif not point.aoa_lower - 1e-12 <= value <= point.aoa_upper + 1e-12:
            errors.append(f'alpha of point {point.id!r} is {value} deg, outside '
                          f'[{point.aoa_lower}, {point.aoa_upper}]')
        alphas[point.id] = (point.id, value)
    if errors:
        raise ConfigError(f'[dv] start={source!r}: ' + '; '.join(errors))
    return DesignStart(source=source, files=files, values=values, alphas=alphas,
                       kept=tuple(kept), fixed=tuple(fixed))


def start_record(start):
    """The ``design_start`` block of ``resolved_config.json``."""
    return {'source': start.source,
            'files': [{'path': os.path.relpath(path, REPO), 'sha256': sha256(path)}
                      for path in start.files],
            'values': {name: value.tolist() for name, value in start.values.items()},
            'alpha': {dst: {'from': src, 'deg': value}
                      for dst, (src, value) in start.alphas.items()},
            'alpha_kept': list(start.kept),
            'alpha_fixed': list(start.fixed)}


def start_lines(start):
    """Banner lines naming the source and the alpha mapping."""
    sizes = ', '.join(f'{name} {value.size}' for name, value in start.values.items())
    lines = [f'[start] design from {start.source} ({sizes or "no geometry DVs"})']
    if start.alphas:
        lines.append('[start] alpha from the source: ' + ', '.join(
            f'{dst} {value:.4f} deg' for dst, (_, value) in start.alphas.items()))
    if start.kept:
        lines.append(f'[start] alpha kept at aoa_init: {", ".join(start.kept)}')
    return lines
