"""Per-point SU2 cfg generation from RunConfig, as a walk over the option registry."""

import os

from .cfg_file import parse_cfg_content, write_cfg
from . import su2_options as reg
from ..mpi import on_root

TOML_REACHABLE_KEYS = reg.TOML_REACHABLE_KEYS


def intentional_differences(cfg):
    """Keys allowed to differ between a primal/adjoint ``cfg`` pair."""
    extra = cfg.solver.su2_extra
    return reg.intentional_differences(extra['primal'], extra['adjoint'])


def _boundary_lines(cfg):
    """``(key, value)`` for every bc family and role key, in declaration order."""
    case = cfg.case
    lines = []
    for name, family in reg.BC_FAMILIES.items():
        entries = [b for b in case.boundaries if b.bc == name]
        if entries:
            lines.append((family.key, reg.bc_line(family, entries)))
    for role, key in reg.ROLE_KEYS.items():
        names = case.markers_with(role)
        if key and names:
            lines.append((key, reg.marker_tuple(names)))
    return lines


def generate_cfg_content(cfg, point, is_adjoint):
    """Complete SU2 cfg text for one point and side. Keys whose predicate is off
    are omitted, so SU2's own default applies."""
    primal_block, adjoint_block = cfg.solver_blocks
    block = adjoint_block if is_adjoint else primal_block
    ctx = reg.Ctx(cfg=cfg, point=point, block=block, is_adjoint=is_adjoint,
                  n_levels=max(primal_block.mglevel, block.mglevel))
    L = [
        '%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%',
        f'% SU2 configuration -- GENERATED from {cfg.source or "a RunConfig"}',
        f'% case={cfg.case.name}  point={point.id}  '
        f'side={"DISCRETE_ADJOINT" if is_adjoint else "DIRECT"}',
        '% Do not edit: every value here comes from the TOML. Edit that.',
        '%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%',
    ]
    extra = cfg.solver.su2_extra
    passthrough = {**extra['both'], **extra['adjoint' if is_adjoint else 'primal']}
    for section in reg.SECTIONS:
        if section == 'BOUNDARY CONDITIONS':
            lines = [(k, v, 0) for k, v in _boundary_lines(cfg)]
        elif section == 'PASSTHROUGH':
            lines = [(k, reg.fmt_passthrough(v), 0) for k, v in passthrough.items()]
        else:
            lines = [(o.key, o.fmt(o.value(ctx)), o.group) for o in reg.REGISTRY
                     if o.section == section and o.applies(ctx)]
        if not lines:
            continue
        L += ['', f'% -------------------- {section} --------------------']
        group = lines[0][2]
        for key, value, g in lines:
            if g != group:
                L.append('')
                group = g
            L.append(f'{key}= {value}')
    return '\n'.join(L) + '\n'


def generate_point_cfgs(cfg, out_dir, comm=None, relative_to=None):
    """Write primal/adjoint cfgs for all points. Returns [(primal, adjoint), ...] relative paths."""
    relative_to = relative_to or out_dir

    def _write_all():
        os.makedirs(out_dir, exist_ok=True)
        for point in cfg.resolved_points:
            for is_adjoint in (False, True):
                out = os.path.join(out_dir, cfg_name(cfg, point, is_adjoint))
                write_cfg(out, generate_cfg_content(cfg, point, is_adjoint))

    if comm is not None:
        on_root(comm, _write_all, what='generate point cfgs')
    else:
        _write_all()

    # Paths are deterministic — every rank computes the same list.
    return [tuple(os.path.relpath(
                os.path.join(out_dir, cfg_name(cfg, point, is_adjoint)),
                relative_to)
                  for is_adjoint in (False, True))
            for point in cfg.resolved_points]


def write_unsuffixed_pair(cfg, out_dir, names=None):
    """Write cfg pair for first point with suffix stripped. Returns (primal_path, adjoint_path)."""
    if names is None:
        names = (f'{cfg.case.name}_primal.cfg', f'{cfg.case.name}_adjoint.cfg')
    point = cfg.resolved_points[0]
    suffix = f'_{point.id}'
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for name, is_adjoint in zip(names, (False, True)):
        text = generate_cfg_content(cfg, point, is_adjoint)
        for key, value in parse_cfg_content(text).items():
            if key.endswith('FILENAME') and value.endswith(suffix):
                text = text.replace(f'{key}= {value}',
                                    f'{key}= {value[:-len(suffix)]}')
        written.append(write_cfg(os.path.join(out_dir, name), text))
    return tuple(written)


def cfg_name(cfg, point, is_adjoint):
    """Generated cfg basename."""
    side = 'adjoint' if is_adjoint else 'primal'
    return f'{cfg.case.name}_{side}_{point.id}.cfg'


def describe(cfg, pairs):
    """One provenance line per point, for the head of a run log.

    A task that solves no adjoint names no adjoint cfg or setting.
    """
    from ..run_config.root import DERIVATIVE_TASKS
    adjoint_used = cfg.execution.task in DERIVATIVE_TASKS
    primal, adjoint = cfg.solver_blocks
    lines = []
    for point, (p, a) in zip(cfg.points, pairs):
        names = f'{p}  {a}' if adjoint_used else p
        lines.append(f'[cfg] {point.id} M={point.mach:.2f} '
                     f'alpha0={point.aoa_init:.2f} w={point.weight:g}: {names}')
    lines.append(f'[cfg] primal ITER={primal.iter} MGLEVEL={primal.mglevel} '
                 f'minval={primal.conv_residual_minval:g} '
                 f'cfl_adapt={primal.cfl_adapt}')
    if adjoint_used:
        lines.append(f'[cfg] adjoint ITER={adjoint.iter} MGLEVEL={adjoint.mglevel} '
                     f'minval={adjoint.conv_residual_minval:g}')
    return lines


def emitted_keys(cfg, point=None, is_adjoint=False):
    """The key SET of a generated cfg, without touching the filesystem."""
    point = point or cfg.points[0]
    return set(parse_cfg_content(generate_cfg_content(cfg, point, is_adjoint)))


def unreachable_keys(cfg):
    """Emitted keys that trace to no registry, runtime, bc, role, or passthrough key."""
    extra = cfg.solver.su2_extra
    known = (set(reg.TOML_REACHABLE_KEYS) | reg.RUNTIME_KEYS
             | {k for table in extra.values() for k in table})
    return sorted((emitted_keys(cfg, is_adjoint=False)
                   | emitted_keys(cfg, is_adjoint=True)) - known)


__all__ = ['TOML_REACHABLE_KEYS', 'intentional_differences', 'generate_point_cfgs',
           'cfg_name', 'write_unsuffixed_pair', 'generate_cfg_content', 'emitted_keys',
           'unreachable_keys', 'describe']
