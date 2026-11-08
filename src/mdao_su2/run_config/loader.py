"""How the text becomes a RunConfig: TOML, ``include``, and CLI overrides."""

import os
import tomllib

from .errors import ConfigError
from .root import RunConfig
from ..paths import REPO
from .checks.case import check_case
from .checks.execution import check_execution
from .checks.files import check_cfg_parity, check_input_files
from .checks.geometry import check_twist
from .checks.objective import check_dv_and_objective
from .checks.points import check_points
from .checks.polar import check_polar
from .checks.remesh import check_remesh
from .checks.solver import check_emitted_enums, check_solver, check_su2_extra
from .checks.task import check_optimizer_and_task, warn_derivative_fallback

MAX_INCLUDE_DEPTH = 8


def validate(cfg, comm=None, check_files=True):
    """Raise ``ConfigError`` after collecting every validation problem."""
    errors = []
    errors += check_points(cfg)
    errors += check_dv_and_objective(cfg)
    errors += check_twist(cfg)
    errors += check_optimizer_and_task(cfg)
    errors += check_polar(cfg)
    warn_derivative_fallback(cfg)
    errors += check_solver(cfg)
    errors += check_case(cfg)
    errors += check_su2_extra(cfg)
    if not errors:
        errors += check_emitted_enums(cfg)
    errors += check_remesh(cfg)
    errors += check_execution(cfg, comm)
    if check_files:
        errors += check_input_files(cfg)
        errors += check_cfg_parity(cfg)
    if errors:
        raise ConfigError(
            f'{len(errors)} problem(s) in {cfg.source or "config"}:\n  '
            + '\n  '.join(errors))
    return cfg


# ------------------------------------------------------------------- merge

def _deep_merge(base, over):
    """Recursively overlay ``over`` onto ``base``."""
    out = dict(base)
    for key, val in over.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = val
    return out


def _read_toml(path):
    if not os.path.isfile(path):
        raise ConfigError(f'config file not found: {path}')
    try:
        with open(path, 'rb') as f:
            return tomllib.load(f)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f'{path}: invalid TOML -- {exc}') from None


def _resolve_includes(path, depth=0, seen=None):
    """Load ``path`` with its ``include`` chain already merged underneath it."""
    if depth > MAX_INCLUDE_DEPTH:
        raise ConfigError(f'include chain deeper than {MAX_INCLUDE_DEPTH} '
                          f'levels at {path} -- probably a cycle')
    seen = seen or []
    abspath = os.path.abspath(path)
    if abspath in seen:
        raise ConfigError('include cycle: '
                          + ' -> '.join(seen + [abspath]))

    data = _read_toml(abspath)
    include = data.pop('include', None)
    if include is None:
        return data

    if isinstance(include, str):
        include = [include]
    merged = {}
    for inc in include:
        # Includes are relative to the file that names them.  This keeps an
        # include chain self-contained when its owning case is moved.
        cand = inc if os.path.isabs(inc) else os.path.join(os.path.dirname(abspath), inc)
        merged = _deep_merge(merged,
                             _resolve_includes(cand, depth + 1, seen + [abspath]))
    return _deep_merge(merged, data)


# --------------------------------------------------------------- overrides

def parse_set_override(text):
    """``"remesh.growth_ratio=1.15"`` -> ``('remesh.growth_ratio', 1.15)``."""
    if '=' not in text:
        raise ConfigError(f'--set expects dotted.key=value, got {text!r}')
    key, _, raw = text.partition('=')
    key, raw = key.strip(), raw.strip()
    if not key:
        raise ConfigError(f'--set has an empty key: {text!r}')
    try:
        value = tomllib.loads(f'v = {raw}')['v']
    except tomllib.TOMLDecodeError:
        # Bare words are the common case (--set optimizer.name=SLSQP); quoting
        # them at the shell is a papercut, so fall back to the literal string.
        value = raw
    return key, value


#: ``--band <name>=<value>`` -> the [verification] key it sets.
BAND_FLAGS = {'alpha': 'verification.alpha_band',
              'shape': 'verification.shape_band',
              'moment_pivot': 'verification.alpha_band_moment_pivot'}


def parse_band_override(text):
    """``"alpha=1e-4"`` -> ``('verification.alpha_band', 1e-4)``.

    A band is a ``--set`` of its [verification] key, so ``run.toml`` records
    it the same way under either route.
    """
    name, sep, raw = text.partition('=')
    name = name.strip()
    if not sep or name not in BAND_FLAGS:
        raise ConfigError(f'--band expects one of {sorted(BAND_FLAGS)}=<value>, '
                          f'got {text!r}')
    try:
        value = float(raw)
    except ValueError:
        raise ConfigError(f'--band {name}: {raw.strip()!r} is not a number') from None
    return BAND_FLAGS[name], value


def _apply_override(data, dotted, value):
    """Set ``dotted`` (``a.b.c``, with integer segments indexing arrays)."""
    parts = dotted.split('.')
    node = data
    for i, part in enumerate(parts[:-1]):
        nxt = parts[i + 1]
        if isinstance(node, list):
            node = node[_as_index(part, node, dotted)]
            continue
        if part not in node or not isinstance(node[part], (dict, list)):
            node[part] = [] if nxt.isdigit() else {}
        node = node[part]
    last = parts[-1]
    if isinstance(node, list):
        node[_as_index(last, node, dotted)] = value
    else:
        node[last] = value


def _as_index(part, seq, dotted):
    if not part.isdigit():
        raise ConfigError(f'--set {dotted}: {part!r} indexes a list, so it must '
                          f'be an integer')
    idx = int(part)
    if idx >= len(seq):
        raise ConfigError(f'--set {dotted}: index {idx} is out of range '
                          f'({len(seq)} entries)')
    return idx


# --------------------------------------------------------------- entry point

def resolve_dict(path, overrides=None):
    """The merged plain dict, before dataclass construction."""
    data = _resolve_includes(path)
    for dotted, value in (overrides or {}).items():
        _apply_override(data, dotted, value)
    return data


def load_config(path, overrides=None, comm=None, check_files=True):
    """Load, merge, override, construct and validate."""
    overrides = dict(overrides or {})
    # Read the leaf config before include resolution to track which [geometry]
    # keys the author explicitly set, as distinct from inherited ones.
    own_geometry_keys = frozenset(_read_toml(os.path.abspath(path))
                                 .get('geometry', {}).keys())
    data = resolve_dict(path, overrides)
    cfg = RunConfig.from_dict(data, source=os.path.relpath(
        os.path.abspath(path), REPO))
    object.__setattr__(cfg, 'overrides',
                       tuple(f'{k}={v!r}' for k, v in overrides.items()))
    object.__setattr__(cfg, '_own_geometry_keys', own_geometry_keys)
    validate(cfg, comm=comm, check_files=check_files)
    return cfg


__all__ = ['load_config', 'resolve_dict', 'parse_set_override',
           'parse_band_override', 'ConfigError']
