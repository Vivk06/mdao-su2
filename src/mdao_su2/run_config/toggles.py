"""The toggle registry.

Every capability a TOML can switch on is one :class:`Toggle`. A toggle is off
when its block or its ``enabled`` key is absent, and an enabled toggle states
every value it reads: :func:`missing` lists the dotted TOML paths an enabled
block leaves unset, and the loader refuses the file. Safety checks (mesh
admission, the adjoint stall guard, derivative-status rejections) are never
registered here, so no TOML can switch them off.

    python -m mdao_su2.run_config.toggles --markdown   # the reference table
"""

import argparse
from dataclasses import dataclass
from typing import Callable, Optional, Tuple


def find_value(data, dotted):
    """``(found, value)`` for a dotted path in the raw TOML dict."""
    node = data
    for part in dotted.split('.'):
        if not isinstance(node, dict) or part not in node:
            return False, None
        node = node[part]
    return True, node


def _flag(data, dotted):
    found, value = find_value(data, dotted)
    return found and value is True


def _geometry_master(data):
    """``[constraints] geometry``: a bool or ``{ enabled = ... }``; off when absent."""
    found, value = find_value(data, 'constraints.geometry')
    if not found:
        return False
    if isinstance(value, dict):
        return value.get('enabled') is True
    return value is True


def _fixed(*keys):
    return lambda data, block: tuple(f'{block}.{k}' for k in keys)


def _bound_keys(kind):
    return {'equals': ('equals',), 'min': ('lower',), 'max': ('upper',),
            'range': ('lower', 'upper')}.get(kind, ())


def _lift_requires(data, block):
    kind = find_value(data, f'{block}.kind')[1]
    return tuple(f'{block}.{k}' for k in ('kind', *_bound_keys(kind), 'scaler'))


def _moment_requires(data, block):
    kind = find_value(data, f'{block}.kind')[1]
    return tuple(f'{block}.{k}'
                 for k in ('kind', *_bound_keys(kind), 'scaler', 'axis'))


def _thickness_requires(data, block):
    keys = ['axes', 'nSpan', 'nChord', 'scaled', 'te_offset', 'stations']
    if find_value(data, f'{block}.axes')[1] == 'span_y':
        keys += ['probe_x', 'span_offsets']
    return tuple(f'{block}.{k}' for k in keys)


def _extract_step_requires(data, block):
    return tuple(f'{block}.{k}' for k in ('mesh', 'markers'))


def _airfoil_step_requires(data, block):
    return tuple(f'{block}.{k}' for k in ('source', 'n_points', 'spacing', 'te'))


def _ffd_step_requires(data, block):
    kind = find_value(data, f'{block}.kind')[1]
    keys = ['kind', 'nx', 'spacing']
    if kind == 'wing':
        keys += ['n_span', 'nz', 'x_bounds', 'y_bounds', 'z_bounds', 'le_te']
    elif kind == 'airfoil':
        keys.append('zspan')
        if find_value(data, f'{block}.x_margin')[0] or find_value(data, f'{block}.y_margin')[0]:
            keys += ['x_margin', 'y_margin']
        else:
            keys += ['x_bounds', 'y_bounds']
        if find_value(data, f'{block}.spacing')[1] == 'cosine':
            keys.append('le_te')
    return tuple(f'{block}.{k}' for k in keys)


#: RemeshQualityConfig fields: every mesh2d gate threshold is stated.
MESH2D_QUALITY = ('min_first_cell_height_ratio', 'max_first_cell_height_ratio',
                  'min_bl_thickness_ratio', 'max_bl_thickness_ratio',
                  'max_skewness', 'max_aspect_ratio_p99')


def _mesh2d_step_requires(data, block):
    keys = ['first_cell_height', 'growth_ratio', 'fan_angle_deg', 'fan_elements',
            'farfield_radius', 'farfield_meshsize', 'near_meshsize',
            'far_growth_slope', 'wall_meshsize', 'wall_marker', 'farfield_marker',
            'farfield_center']
    # The BL depth is stated one way: a layer count, or a thickness.
    keys.append('bl_thickness' if find_value(data, f'{block}.bl_thickness')[0] else 'n_bl_layers')
    keys += [f'quality.{k}' for k in MESH2D_QUALITY]
    return tuple(f'{block}.{k}' for k in keys)


WING3D_QUALITY = ('min_dihedral_deg', 'max_dihedral_deg', 'max_aspect_ratio_p99')


def _wing3d_step_requires(data, block):
    keys = ['span', 'tip', 'farfield_radius', 'farfield_center', 'wall_size',
            'le_size', 'te_size', 'tip_size', 'far_size', 'size_growth', 'edge_band',
            'te_band', 'te_normal_scale', 'edge_distance',
            'level_scale', 'wing_marker', 'tip_marker', 'symmetry_marker',
            'farfield_marker']
    keys += [f'quality.{k}' for k in WING3D_QUALITY]
    return tuple(f'{block}.{k}' for k in keys)


SECTION2D_QUALITY = ('max_skewness', 'max_aspect_ratio_p99')


def _section2d_step_requires(data, block):
    keys = ['farfield_radius', 'farfield_center', 'wall_size', 'le_size', 'te_size',
            'far_size', 'size_growth', 'edge_band', 'level_scale', 'wall_marker',
            'farfield_marker']
    keys += [f'quality.{k}' for k in SECTION2D_QUALITY]
    return tuple(f'{block}.{k}' for k in keys)


def _shape_requires(data, block):
    param = find_value(data, 'geometry.parameterization')
    param = param[1] if param[0] else 'ffd'
    if param == 'hicks_henne':
        keys = ['n_dv', 'distribution', 'bump_width', 'bound', 'scaler']
        return tuple(f'{block}.{k}' for k in keys)
    out = [f'{block}.bound', f'{block}.scaler']
    if param == 'cst':
        return tuple(out + ['geometry.n_cst', 'geometry.dat_file'])
    out += [f'{block}.pattern', 'geometry.ffd']
    pattern = find_value(data, f'{block}.pattern')[1]
    if pattern == 'bspline':
        out += [f'{block}.n_basis', f'{block}.degree']
    if not find_value(data, f'{block}.directions')[0]:
        out.append(f'{block}.direction')
    return tuple(out)


def _polar_requires(data, block):
    keys = ['alpha', 'run_urans']
    if find_value(data, f'{block}.run_urans')[1] is True:
        keys += ['cl_band', 'urans_max_angles', 'urans_min_cycles', 'urans']
    if len(data.get('points') or ()) > 1:
        keys.append('mach_floor')
    return tuple(f'{block}.{k}' for k in keys)


@dataclass(frozen=True)
class Toggle:
    """One switch. ``block`` is the TOML table; the switch is ``<block>.enabled``."""
    block: str
    owner: str
    switches: str
    #: ``(raw data, block) -> dotted paths`` required while the toggle is on.
    requires: Callable
    #: Another toggle that must also be on for this one to act.
    parent: Optional[str] = None
    #: ``run``: a run TOML block. ``recipe``: a ``[mesh]`` recipe step (mesh/recipe.py).
    scope: str = 'run'

    @property
    def path(self):
        return self.block if self.block == 'constraints.geometry' \
            else f'{self.block}.enabled'

    def own_state(self, data):
        if self.block == 'constraints.geometry':
            return _geometry_master(data)
        return _flag(data, self.path)


REGISTRY: Tuple[Toggle, ...] = (
    Toggle('dv.shape', 'optimization/model.py', 'shape design variables of '
           'the selected parameterization', _shape_requires),
    Toggle('dv.twist', 'geometry/twist.py', 'trailing-edge twist DVs of the wing '
           'FFD sections (degrees, nose-up positive)',
           lambda data, block: _fixed('n', 'lower_deg', 'upper_deg', 'scaler',
                                      'include_root', 'axis_x_fraction',
                                      'init_deg')(data, block) + ('geometry.ffd',)),
    Toggle('trim', 'optimization/tasks.py', 'secant alpha trim to the lift '
           'target before the driver',
           _fixed('max_iter', 'tol', 'dcl_da_init', 'alpha_clip')),
    Toggle('polar', 'optimization/polar.py', 'the lift curve of the polar task, '
           'with URANS where the steady solve does not converge', _polar_requires),
    Toggle('constraints.geometry', 'optimization/model.py', 'the geometric '
           'constraints below (master switch)', _fixed()),
    Toggle('constraints.volume', 'geometry/ffd.py', 'volume constraint',
           _fixed('le_x', 'te_x', 'nSpan', 'nChord', 'lower'),
           parent='constraints.geometry'),
    Toggle('constraints.le_radius', 'geometry/ffd.py', 'leading-edge radius '
           'constraint', _fixed('le_x', 'nSpan', 'lower', 'axis', 'chord_dir'),
           parent='constraints.geometry'),
    Toggle('constraints.thickness', 'geometry/ffd.py', 'thickness rails',
           _thickness_requires, parent='constraints.geometry'),
    Toggle('constraints.curvature', 'geometry/ffd.py', 'curvature rail',
           _fixed('start_x', 'end_x', 'n_pts', 'upper', 'scaler',
                  'curvature_type', 'ks_coeff'),
           parent='constraints.geometry'),
    Toggle('constraints.lift', 'optimization/model.py', 'lift constraint',
           _lift_requires),
    Toggle('constraints.moment', 'optimization/model.py', 'moment constraint',
           _moment_requires),
    Toggle('verification.moment_pivot', 'optimization/tasks.py',
           'moment-pivot identity gate',
           _fixed('reference_pivot_x', 'dcm_dalpha_ref', 'dcl_dalpha',
                  'dcd_dalpha', 'cl', 'cd', 'alpha_deg', 'band_pivot',
                  'reference_run')),
    Toggle('postprocess.span_load', 'postprocess/span_load.py', 'span load, e, '
           'e0 and the Trefftz efficiency in the results file after the run',
           _fixed('n_stations', 'span_axis', 'lift_axis', 'semispan', 'aspect_ratio')),
    # --- [mesh] recipe steps (run/mesh.py); CLI flags count as set values.
    Toggle('mesh.extract', 'geometry/airfoil.py', 'recipe step: wall markers '
           'of a 2D SU2 mesh as a Selig .dat', _extract_step_requires,
           scope='recipe'),
    Toggle('mesh.airfoil', 'geometry/airfoil.py', 'recipe step: import '
           'and repanel an airfoil, write Selig .dat and geometry report',
           _airfoil_step_requires, scope='recipe'),
    Toggle('mesh.ffd', 'geometry/ffd_lattice.py', 'recipe step: write an airfoil or '
           'wing FFD box (Plot3D)', _ffd_step_requires, scope='recipe'),
    Toggle('mesh.mesh2d', 'mesh/mesh2d.py', 'recipe step: hybrid '
           'boundary-layer mesh around a 2D section, then its quality check',
           _mesh2d_step_requires, scope='recipe'),
    Toggle('mesh.wing3d', 'mesh/wing3d.py', 'recipe step: tetrahedral '
           'half-wing with a revolved tip, then its 3D admission check',
           _wing3d_step_requires, scope='recipe'),
    Toggle('mesh.section2d', 'mesh/section2d.py', 'recipe step: '
           'all-triangle 2D section mesh sized as a wing3d span station, then its '
           'check', _section2d_step_requires, scope='recipe'),
)

BY_BLOCK = {t.block: t for t in REGISTRY}
RUN = tuple(t for t in REGISTRY if t.scope == 'run')
RECIPE = tuple(t for t in REGISTRY if t.scope == 'recipe')


def state(data, toggle):
    """Whether ``toggle`` acts: its own switch and its parent's are on."""
    if not toggle.own_state(data):
        return False
    return toggle.parent is None or state(data, BY_BLOCK[toggle.parent])


def states(data):
    """``{path: bool}`` for every run toggle, from the raw run-TOML dict."""
    return {t.path: state(data, t) for t in RUN}


def states_of(cfg):
    """``{path: bool}`` from a built ``RunConfig`` (what ``formulation.json`` records)."""
    def own(t):
        node = cfg
        for part in t.path.split('.'):
            node = getattr(node, part)
        return bool(node)

    def acts(t):
        return own(t) and (t.parent is None or acts(BY_BLOCK[t.parent]))
    return {t.path: acts(t) for t in RUN}


def report(data, overrides=()):
    """``{path: {'on', 'source'}}`` for ``--print-config``; source is toml, override or absent."""
    overridden = {o.split('=', 1)[0] for o in overrides}
    out = {}
    for t in RUN:
        if t.path in overridden:
            source = 'override'
        else:
            source = 'toml' if find_value(data, t.path)[0] else 'absent'
        out[t.path] = {'on': state(data, t), 'source': source}
    return out


def missing(data, scope='run'):
    """``[(toggle path, dotted key)]`` an enabled toggle reads but the TOML leaves unset."""
    out = []
    for t in (RUN if scope == 'run' else RECIPE):
        if state(data, t):
            out += [(t.path, key) for key in t.requires(data, t.block)
                    if not find_value(data, key)[0]]
    return out


def check(data):
    """Raise ``ConfigError`` naming every value an enabled toggle leaves unset."""
    from .root import ConfigError
    gaps = missing(data)
    if gaps:
        lines = '\n'.join(f'  {key}  (needed by {path} = true)' for path, key in gaps)
        raise ConfigError('an enabled block must state every value it uses; no '
                          'default is assumed. Unset:\n' + lines)


def markdown():
    """Render the switch registry as a Markdown table."""
    rows = ['| Switch | Switches on | Required when on | Owner |',
            '|---|---|---|---|']
    for t in REGISTRY:
        probe = {}
        needs = ', '.join(f'`{k.split(t.block + ".", 1)[-1]}`'
                          for k in t.requires(probe, t.block)) or '—'
        if t.requires in (_lift_requires, _moment_requires):
            needs += ', and the bound(s) its `kind` uses'
        elif t.requires is _thickness_requires:
            needs += '; `probe_x`, `span_offsets` under `axes = "span_y"`'
        elif t.requires is _ffd_step_requires:
            needs = ('`kind`, `nx`, `spacing`; airfoil: `zspan` and `x_bounds`, '
                     '`y_bounds` (or `x_margin`, `y_margin`), `le_te` for '
                     '`cosine`; wing: `n_span`, `nz`, `x_bounds`, `y_bounds`, '
                     '`z_bounds`, `le_te` (its TE is the twist axis)')
        elif t.requires is _mesh2d_step_requires:
            needs = ('the `RemeshConfig.gmsh_kwargs()` names (`first_cell_height`, '
                     '`growth_ratio`, `n_bl_layers` or `bl_thickness`, '
                     '`fan_angle_deg`, `fan_elements`, `farfield_radius`, '
                     '`farfield_meshsize`, `near_meshsize`, `far_growth_slope`, '
                     '`wall_meshsize`), `wall_marker`, `farfield_marker`, '
                     '`farfield_center`, and `[mesh.mesh2d.quality]` (six bands)')
        elif t.requires is _wing3d_step_requires:
            needs = ('`span`, `tip`, `farfield_radius`, `farfield_center`, the sizes '
                     '(`wall_size`, `le_size`, `te_size`, `tip_size`, `far_size`, '
                     '`size_growth`, `edge_band`, `te_band`, `te_normal_scale`), '
                     '`edge_distance`, '
                     '`level_scale` (one factor per '
                     'level), the four marker names, and `[mesh.wing3d.quality]` '
                     '(dihedral bounds, aspect-ratio p99)')
        elif t.requires is _shape_requires:
            needs = ('`bound`, `scaler`; Hicks–Henne: `n_dv`, `distribution`, '
                     '`bump_width`; CST: `geometry.n_cst`, `geometry.dat_file`; '
                     'FFD: `pattern`, `geometry.ffd`, `direction` (or '
                     '`directions`), and `n_basis`, `degree` for `bspline`')
        switches = t.switches + (f' (needs `{BY_BLOCK[t.parent].path}`)'
                                 if t.parent else '')
        rows.append(f'| `{t.path}` | {switches} | {needs} | `{t.owner}` |')
    return '\n'.join(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--markdown', action='store_true',
                    help='print the registry as a markdown table')
    args = ap.parse_args(argv)
    if args.markdown:
        print(markdown())
    else:
        for t in REGISTRY:
            print(t.path)


if __name__ == '__main__':
    main()
