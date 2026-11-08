"""The SU2 configuration-file reader/writer.  One implementation, one place."""

import os
import re
import tempfile

__all__ = ['parse_cfg', 'parse_cfg_content', 'cfg_value', 'cfg_values', 'substitute', 'write_cfg',
           'rewrite_cfg', 'edit_cfg', 'derive_cfg', 'changed_keys',
           'PARITY_KEYS', 'PARITY_PREFIXES', 'cfg_parity_diff',
           'assert_cfg_parity', 'CfgParityError']


class CfgParityError(AssertionError):
    """A primal/adjoint cfg pair disagrees on a physics key."""


# --------------------------------------------------------------------- read

def parse_cfg_content(content):
    """Return ``{KEY: value}`` for SU2 cfg TEXT, comments stripped."""
    values = {}
    for line in content.splitlines():
        line = line.split('%')[0].strip()
        head, sep, val = line.partition('=')
        if sep:
            values.setdefault(head.strip(), val.strip())
    return values


def parse_cfg(cfg_path):
    """``{KEY: value}`` for the SU2 cfg at `cfg_path`.  See parse_cfg_content."""
    with open(cfg_path) as f:
        return parse_cfg_content(f.read())


def cfg_value(cfg_path, key, default=None):
    """Return the comment-stripped value string of `key` in a SU2 cfg, or None."""
    with open(cfg_path) as f:
        for line in f:
            line = line.split('%')[0].strip()
            head, sep, val = line.partition('=')
            if sep and head.strip() == key:
                return val.strip()
    return default


def cfg_values(cfg_path, keys):
    """``{key: value-or-None}`` for `keys`, in one pass over the file."""
    parsed = parse_cfg(cfg_path)
    return {k: parsed.get(k) for k in keys}


# -------------------------------------------------------------------- write

def substitute(content, replacements):
    """Return `content` with each ``KEY= value`` line replaced, appending any"""
    for key, value in replacements.items():
        content, n = re.subn(rf'^{key}\s*=[^\n]*', f'{key}= {value}',
                             content, flags=re.MULTILINE)
        if n == 0:
            content += f'\n{key}= {value}\n'
    return content


def write_cfg(path, content):
    """Write cfg text to `path` (parent directory must exist).  Returns `path`."""
    with open(path, 'w') as f:
        f.write(content)
    return path


def rewrite_cfg(cfg_path, replacements, dir=None, prefix='su2comp_'):
    """Write a temp copy of the cfg with `KEY= value` lines replaced; return its path."""
    with open(cfg_path) as f:
        content = f.read()
    content = substitute(content, replacements)
    tmp = tempfile.NamedTemporaryFile(
        mode='w', suffix='.cfg', dir=dir or os.path.dirname(cfg_path) or '.',
        delete=False, prefix=prefix)
    tmp.write(content)
    tmp.close()
    return tmp.name


def edit_cfg(cfg_path, replacements):
    """In-place key-value override on a SU2 cfg file.  Returns `cfg_path`."""
    with open(cfg_path) as f:
        content = f.read()
    return write_cfg(cfg_path, substitute(content, replacements))


def derive_cfg(src_cfg, dst_cfg, replacements):
    """Write a NEW cfg at `dst_cfg` derived from `src_cfg`.  Returns `dst_cfg`."""
    with open(src_cfg) as f:
        content = f.read()
    return write_cfg(dst_cfg, substitute(content, replacements))


# ------------------------------------------------------------------- verify

def changed_keys(reference_cfg, generated_cfg):
    """Keys whose value differs between two cfgs (including keys present in only"""
    a, b = parse_cfg(reference_cfg), parse_cfg(generated_cfg)
    return sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))


# Primal/adjoint parity: a pair must describe the SAME problem.  A mismatch
# here does not crash -- it silently produces sensitivities of a different flow
PARITY_KEYS = (
    'MESH_FILENAME', 'MESH_FORMAT',
    'MACH_NUMBER', 'AOA', 'SIDESLIP_ANGLE',
    'REYNOLDS_NUMBER', 'REYNOLDS_LENGTH',
    'FREESTREAM_TEMPERATURE', 'FREESTREAM_PRESSURE', 'FREESTREAM_DENSITY',
    'FREESTREAM_TURBULENCEINTENSITY', 'FREESTREAM_TURB2LAMVISCRATIO',
    'FLUID_MODEL', 'GAMMA_VALUE', 'GAS_CONSTANT',
    'SOLVER', 'KIND_TURB_MODEL', 'SA_OPTIONS',
    # The discrete adjoint differentiates the PRIMAL discretisation, so a
    # scheme that differs between the pair linearises a flow nobody solved.
    'CONV_NUM_METHOD_FLOW', 'SLOPE_LIMITER_FLOW', 'VENKAT_LIMITER_COEFF',
    'JST_SENSOR_COEFF', 'NUM_METHOD_GRAD',
    'CONV_NUM_METHOD_TURB', 'SLOPE_LIMITER_TURB',
)
# Every key starting with one of these is compared as well -- MARKER_* and REF_*
# are open-ended sets and enumerating them would silently miss a new one.
PARITY_PREFIXES = ('MARKER_', 'REF_')


def cfg_parity_diff(primal_cfg, adjoint_cfg, keys=PARITY_KEYS,
                    prefixes=PARITY_PREFIXES, as_text=False):
    """``{key: (primal_value, adjoint_value)}`` for every parity key that differs."""
    read = parse_cfg_content if as_text else parse_cfg
    p, a = read(primal_cfg), read(adjoint_cfg)
    watched = set(keys)
    watched.update(k for k in set(p) | set(a) if k.startswith(prefixes))
    return {k: (p.get(k), a.get(k)) for k in sorted(watched)
            if p.get(k) != a.get(k)}


def assert_cfg_parity(primal_cfg, adjoint_cfg, keys=PARITY_KEYS,
                      prefixes=PARITY_PREFIXES, as_text=False):
    """Raise CfgParityError unless the pair agrees on mesh, markers, references,
    freestream and turbulence."""
    diff = cfg_parity_diff(primal_cfg, adjoint_cfg, keys, prefixes, as_text)
    if diff:
        lines = '\n'.join(f'  {k}: primal={p!r}  adjoint={a!r}'
                          for k, (p, a) in diff.items())
        named = ('<generated primal>', '<generated adjoint>') if as_text \
            else (primal_cfg, adjoint_cfg)
        raise CfgParityError(
            f'primal/adjoint cfg parity violated\n'
            f'  primal : {named[0]}\n  adjoint: {named[1]}\n{lines}')
