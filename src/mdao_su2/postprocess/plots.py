"""File-based figure entry points: explicit input files in, figures and ``manifest.json`` out.

Every entry point takes the files to draw (N of them: N surfaces overlaid, N
histories, N sweep curves), an explicit ``out_dir`` and a :class:`PlotOptions`.
None reads a run directory: the files carry the data, and what a file lacks
(freestream Mach, gamma) comes from an argument or a sibling SU2 cfg. A figure
that fails raises.

| Entry point | Inputs | Figures (``figures/__init__.py::FIGURES``) |
| --- | --- | --- |
| :func:`plot_surface` | surface ``.vtu``/``.csv`` x N (+ reference Cp files) | ``cp_chord``, ``field_surface``, ``surface_mach``, 3D panels, ``panel_delta_planform``, ``field_sections``, ``span_load``, ``section_geometry``, ``surface_3d_overlay`` |
| :func:`plot_shape` | surface x N (first is the base) | ``shape_overlay``, ``delta_y`` |
| :func:`plot_field` | volume ``.vtu`` x N | ``field_contour`` |
| :func:`plot_history` | SU2 history x N | ``residual_history``, ``coefficient_history``, ``history_columns``, ``adjoint_residuals`` |
| :func:`plot_stdout` | SU2 screen output x N | the history figures |
| :func:`plot_optimizer` | pyOptSparse ``.hst`` x N | ``optimizer_history``, ``constraint_history``, ``dv_history``, ``dv_profile`` |
| :func:`plot_sweep` | N groups of sample files | ``sweep`` |
| :func:`plot_iterations` | N groups of sample files | ``iteration_change``, ``linear_margins`` |
| :func:`plot_view` | surface (+ baseline) | ``paraview_view`` |
| :func:`plot_dv_map` | config | ``dv_map`` and the basis figures |
| :func:`plot_probes` | volume ``.vtu`` | ``probe_line``, ``probe_region`` |
"""
import hashlib
import json
import os
import subprocess
from dataclasses import dataclass

import numpy as np

from .readers import (ReaderError, cfg_float, read_history, read_hst, read_su2_cfg,
                      read_reference_cp, read_su2_stdout, read_surface, read_volume)

__all__ = [
    'PlotsError', 'PlotOptions', 'unmet', 'run_probes',
    'plot_surface', 'plot_shape', 'plot_field', 'plot_history', 'plot_stdout',
    'plot_optimizer', 'plot_sweep', 'plot_iterations', 'plot_view', 'plot_dv_map',
    'plot_probes',
]


class PlotsError(Exception):
    """A post-processing operation failed."""


@dataclass(frozen=True)
class PlotOptions:
    """Output and style options shared by every entry point.

    ``figsize`` is inches ``(w, h)`` (None: the figure's own size); ``colors``
    and ``styles`` are per input in input order (``style.series_styles``, more
    inputs than colours raises); ``xlim`` applies to every panel and ``ylim``
    to a figure with one panel; ``legend_out`` also writes a standalone legend
    image; ``cmd`` is stamped in the figure footer and the manifest.
    """
    fmt: str = 'png'
    dpi: int = None
    figsize: tuple = None
    colors: tuple = None
    styles: tuple = None
    palette: str = 'default'
    xlim: tuple = None
    ylim: tuple = None
    legend_out: bool = False
    cmd: str = None


# ---- provenance ---------------------------------------------------------------

def _git_sha():
    try:
        return subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip() or None
    except Exception:
        return None


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def _write_manifest(out_dir, inputs, figures, skipped, options, extra=None):
    """Merge this call into ``out_dir/manifest.json``: inputs with SHA-256, figures, skips."""
    path = os.path.join(out_dir, 'manifest.json')
    manifest = {'inputs': [], 'figures': {}, 'skipped': {}}
    if os.path.isfile(path):
        with open(path) as stream:
            manifest.update(json.load(stream))
    seen = {(i['role'], i['path']) for i in manifest['inputs']}
    for role, file in inputs:
        file = os.path.abspath(file)
        if (role, file) not in seen:
            seen.add((role, file))
            manifest['inputs'].append({'role': role, 'path': file, 'sha256': _sha256(file)})
    manifest['figures'].update({k: os.path.basename(v) for k, v in figures.items()})
    manifest['skipped'].update(skipped)
    manifest['git'] = _git_sha()
    if options.cmd:
        manifest['command'] = options.cmd
    manifest.update(extra or {})
    with open(path, 'w') as stream:
        json.dump(manifest, stream, indent=2)
        stream.write('\n')


# ---- requirements --------------------------------------------------------------

def unmet(requires, ctx):
    """Why a figure cannot be drawn from these inputs, or None.

    ``ctx`` holds ``kind`` (the input kind given), ``n_inputs``, ``ndim`` (or
    None), ``fields`` (point-data names present in every input),
    ``parameterization`` and ``probes`` (the probe kinds given).
    """
    if ctx['kind'] not in requires['inputs']:
        return f"needs {' or '.join(requires['inputs'])} inputs, got {ctx['kind']}"
    if ctx['n_inputs'] < requires['min_inputs']:
        return f"needs at least {requires['min_inputs']} inputs"
    if requires['ndim'] is not None and ctx.get('ndim') is not None \
            and requires['ndim'] != ctx['ndim']:
        return f"needs ndim = {requires['ndim']}"
    if requires['parameterization'] and \
            ctx.get('parameterization') not in requires['parameterization']:
        return f"needs parameterization in {list(requires['parameterization'])}"
    missing = [f for f in requires['fields'] if f not in ctx.get('fields', ())]
    if missing:
        return f'the input lacks {missing}'
    if requires['probe'] and requires['probe'] not in ctx.get('probes', ()):
        return f"needs a {requires['probe']} probe"
    return None


# ---- drawing ------------------------------------------------------------------

class _Sheet:
    """Figures of one entry-point call: draws, saves, records skips, writes the manifest."""

    def __init__(self, out_dir, options, inputs, entry, ctx):
        import matplotlib
        matplotlib.use('Agg')
        from .style import DPI
        if out_dir is None:
            raise PlotsError('give an output directory (out_dir)')
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir, self.options, self.inputs = out_dir, options, list(inputs)
        self.entry, self.ctx = entry, ctx
        self.dpi = options.dpi or DPI
        self.sha = _git_sha()
        self.written, self.skipped = {}, {}

    def select(self, figures, extra_default=()):
        """The figure names to draw: the named ones (unmet raises) or the defaults."""
        from .figures import FIGURES
        if figures:
            for name in figures:
                if name not in FIGURES or FIGURES[name][1]['entry'] != self.entry:
                    mine = sorted(n for n, (_, r) in FIGURES.items() if r['entry'] == self.entry)
                    raise PlotsError(f'{name!r} is not a {self.entry} figure. Available: '
                                     f'{", ".join(mine)}')
                reason = unmet(FIGURES[name][1], self.ctx)
                if reason:
                    raise PlotsError(f'{name}: {reason}')
            return list(figures)
        names = []
        for name, (_, req) in FIGURES.items():
            if req['entry'] != self.entry or not (req['default'] or name in extra_default):
                continue
            reason = unmet(req, self.ctx)
            if reason:
                self.skipped[name] = reason
                print(f'  skip {name}: {reason}')
            else:
                names.append(name)
        return names

    def draw(self, stem, build, size=(8.0, 6.0)):
        """Save ``stem.<fmt>``; ``build(fig)`` draws (and may resize) the figure."""
        import matplotlib.pyplot as plt
        from .style import footer
        fig = plt.figure(figsize=self.options.figsize or size)
        try:
            build(fig)
            _apply_limits(fig, stem, self.options.xlim, self.options.ylim)
            footer(fig, sha=self.sha, cmd=self.options.cmd)
            path = os.path.join(self.out_dir, f'{stem}.{self.options.fmt}')
            fig.savefig(path, dpi=self.dpi, bbox_inches='tight', facecolor='white')
        except Exception as exc:
            raise PlotsError(f'Figure {stem!r} failed to render: {exc}') from exc
        finally:
            plt.close(fig)
        self.written[stem] = path
        return path

    def axes(self, stem, build, size=(8.0, 6.0)):
        """Like :meth:`draw` with one Axes: ``build(ax)``."""
        return self.draw(stem, lambda fig: build(fig.subplots()), size)

    def legend(self, labels, colors):
        if self.options.legend_out:
            from .figures.comparison import save_legend
            path = os.path.join(self.out_dir, f'legend.{self.options.fmt}')
            save_legend(labels, path, dpi=self.dpi, colors=colors)
            self.written['legend'] = path

    def finish(self, extra=None):
        _write_manifest(self.out_dir, self.inputs, self.written, self.skipped, self.options,
                        extra)
        return dict(self.written)


def _apply_limits(fig, name, xlim=None, ylim=None):
    """``xlim`` on every panel; ``ylim`` only on a figure with a single axes.

    An inverted axis stays inverted.
    """
    axes = [ax for ax in fig.axes if ax.axison and ax.get_label() != '<colorbar>']
    for ax in axes if xlim else ():
        ax.set_xlim(*(sorted(xlim, reverse=ax.xaxis_inverted())))
    if ylim:
        if len(axes) == 1:
            axes[0].set_ylim(*sorted(ylim, reverse=axes[0].yaxis_inverted()))
        else:
            print(f'  note: ylim not applied to {name} (it has {len(axes)} panels)')


def _labels(paths, labels):
    """One label per path: the given ones, else the file names (parent folder when they repeat)."""
    if labels:
        if len(labels) != len(paths):
            raise PlotsError(f'{len(paths)} inputs but {len(labels)} labels')
        return list(labels)
    stems = [os.path.splitext(os.path.basename(p))[0] for p in paths]
    if len(set(stems)) == len(stems):
        return stems
    return [os.path.join(os.path.basename(os.path.dirname(os.path.abspath(p))), s)
            for p, s in zip(paths, stems)]


def _slug(label):
    return ''.join(c if c.isalnum() else '_' for c in label).strip('_').lower() or 'entry'


def _styles(count, options):
    from .style import series_styles
    try:
        return series_styles(count, options.palette, options.colors, options.styles)
    except ValueError as exc:
        raise PlotsError(str(exc)) from exc


def _read(reader, paths, what):
    try:
        return [reader(p) for p in paths]
    except ReaderError as exc:
        raise PlotsError(str(exc)) from exc


def _paths(paths):
    out = [paths] if isinstance(paths, (str, os.PathLike)) else list(paths)
    if not out:
        raise PlotsError('no input files')
    return [os.fspath(p) for p in out]


def _per_input(value, cfgs, key, count, default=None):
    """A per-input scalar: ``value`` (one number or one per input), else the cfgs' ``key``."""
    if value is not None:
        values = [value] * count if np.isscalar(value) else list(value)
        if len(values) != count:
            raise PlotsError(f'{count} inputs but {len(values)} values for {key}')
        return [float(v) for v in values]
    if cfgs:
        cfgs = [cfgs] if isinstance(cfgs, (str, os.PathLike)) else list(cfgs)
        cfgs = cfgs * count if len(cfgs) == 1 else cfgs
        if len(cfgs) != count:
            raise PlotsError(f'{count} inputs but {len(cfgs)} cfg files')
        out = [cfg_float(_read_cfg(c), key, default) for c in cfgs]
        if None in out:
            raise PlotsError(f'{key} is not in every cfg file')
        return out
    return None


def _read_cfg(path):
    try:
        return read_su2_cfg(path)
    except ReaderError as exc:
        raise PlotsError(str(exc)) from exc


def _common_fields(datas):
    sets = [set(d['point_data']) for d in datas]
    return set.intersection(*sets) if sets else set()


# ---- surfaces ------------------------------------------------------------------

def plot_surface(paths, out_dir, labels=None, fields=None, figures=None, mach=None,
                 gamma=None, cfgs=None, stations=None, span_axis='y', lift_axis='z',
                 options=None, references=None, reference_labels=None, aoa=None,
                 ref_area=None, semispan=None, n_stations=60, quantities=None,
                 displacement_scale=1.0):
    """Surface figures of N surface files, written to ``out_dir``. Returns ``{stem: path}``.

    Default figures: ``cp_chord`` and ``field_surface`` for a 2D wall; the
    planform, sections, root section and 3D views for a 3D one. ``surface_mach``
    is drawn when named, from ``mach`` (one number or one per file) and
    ``gamma``, else from ``MACH_NUMBER`` and ``GAMMA_VALUE`` of ``cfgs`` (one
    SU2 cfg per file, or one for all). ``fields`` are the point-data names the
    field figures draw (default ``Pressure_Coefficient``); ``stations`` the
    span positions of ``panel_sections`` (default 10, 50 and 90 % of the
    first surface's span). Planform and 3D views are drawn per file; the other
    figures overlay all files. ``span_axis`` and ``lift_axis`` are the wing's.

    ``references`` are reference Cp files drawn as markers on ``cp_chord``
    (each ``'FILE'`` or ``'FILE:point=N'``, ``readers.read_reference_cp``),
    labelled by ``reference_labels`` (default the file stem, plus
    ``@point N`` when a point is selected). They join the manifest ``inputs``
    (role ``reference``) and the selected ``ALPHA`` of each is recorded under
    ``cp_references``.

    Named 3D figures (not drawn by default). ``field_sections`` draws each ``fields`` entry along the wall section at
    each of ``stations`` (default 10, 50 and 90 % of the span), one panel per
    station. ``panel_delta_planform`` draws,
    for each later file, its field minus the first file's on the first file's
    mesh (equal point counts required). ``span_load`` draws the sectional lift
    of every file in ``n_stations`` strips with its elliptic distribution and
    Trefftz efficiency; it needs the angle of attack ``aoa`` (degrees, one
    value or one per file), the reference area ``ref_area`` of the modelled
    wing and its ``semispan``, each given or (``aoa``, ``ref_area``) read as
    ``AOA`` and ``REF_AREA`` from ``cfgs``; a missing one raises.
    ``section_geometry`` draws twist (relative to the first file), chord and
    thickness-to-chord at ``stations`` (default 24 evenly spaced from 2 to
    96 % of the first surface's span); ``quantities`` picks the panels
    (``twist``, ``chord``, ``thickness``; default all three).
    ``surface_3d_overlay`` draws all files as semi-transparent 3D surfaces on
    one axes as wall sections (the first grey, the others in their entry
    colour), with no field colouring; the points of
    each later file move to the first file's points plus ``displacement_scale``
    times their displacement (equal point counts required when it is not 1).
    """
    options = options or PlotOptions()
    paths = _paths(paths)
    labels = _labels(paths, labels)
    surfaces = _read(read_surface, paths, 'surface')
    refs, ref_inputs, ref_record = _read_cp_references(references, reference_labels)
    from .extract import surface_ndim
    ndims = {surface_ndim(s) for s in surfaces}
    if len(ndims) != 1:
        raise PlotsError('the surface files mix 2D and 3D walls')
    ctx = {'kind': 'surface', 'n_inputs': len(paths), 'ndim': ndims.pop(),
           'fields': _common_fields(surfaces)}
    inputs = ([('surface', p) for p in paths] + [('cfg', c) for c in _listed(cfgs)]
              + ref_inputs)
    sheet = _Sheet(out_dir, options, inputs, 'plot_surface', ctx)
    colors, styles = _styles(len(paths) + len(refs), options)
    ref_colors = colors[len(paths):]
    colors, styles = colors[:len(paths)], styles[:len(paths)]
    entries = list(zip(labels, surfaces))
    fields = list(fields) if fields else ['Pressure_Coefficient']
    wanted = sheet.select(figures)
    from .figures import FIGURES

    def fn(name):
        return FIGURES[name][0]

    def field_stem(name, field):
        return name if len(fields) == 1 else f'{name}_{field}'

    for name in wanted:
        if name == 'cp_chord':
            sheet.axes(name, lambda ax: fn(name)(ax, entries, colors, styles,
                                                 references=refs, reference_colors=ref_colors))
        elif name == 'field_surface':
            for field in fields:
                sheet.axes(field_stem(name, field),
                           lambda ax, f=field: fn(name)(ax, entries, colors, styles, field=f))
        elif name == 'surface_mach':
            machs = _per_input(mach, cfgs, 'MACH_NUMBER', len(paths))
            # A cfg without GAMMA_VALUE runs at SU2's default, 1.4.
            gammas = _per_input(gamma, cfgs, 'GAMMA_VALUE', len(paths), default=1.4)
            if machs is None or gammas is None:
                raise PlotsError('surface_mach needs mach and gamma, or cfgs holding '
                                 'MACH_NUMBER and GAMMA_VALUE')
            if len(set(gammas)) != 1:
                raise PlotsError('surface_mach needs one gamma for all inputs')
            sheet.axes(name, lambda ax: fn(name)(ax, entries, machs, gammas[0], colors, styles),
                       size=(6.0, 3.6))
        elif name in ('panel_field_planform', 'surface_3d'):
            _surface_3d_figures(sheet, name, fn(name), entries, fields, field_stem, lift_axis,
                                span_axis)
        elif name == 'panel_delta_planform':
            _delta_planform_figures(sheet, fn(name), entries, fields, field_stem, lift_axis,
                                    span_axis)
        elif name == 'field_sections':
            _field_sections_figures(sheet, fn(name), entries, fields, field_stem, stations,
                                    colors, styles, span_axis, lift_axis)
        elif name == 'span_load':
            _span_load_figure(sheet, fn(name), entries, colors, styles, aoa, ref_area,
                              semispan, cfgs, n_stations, span_axis, lift_axis)
        elif name == 'section_geometry':
            _section_geometry_figure(sheet, fn(name), entries, colors, styles, stations,
                                     span_axis, lift_axis, quantities)
        elif name == 'surface_3d_overlay':
            _surface_3d_overlay_figure(sheet, fn(name), entries, colors, span_axis,
                                       lift_axis, displacement_scale)
        elif name == 'panel_sections':
            lo_hi = _span_extent(surfaces[0], span_axis)
            use = list(stations) if stations else [
                lo_hi[0] + f * (lo_hi[1] - lo_hi[0]) for f in (0.1, 0.5, 0.9)]
            sheet.axes(name, lambda ax: fn(name)(ax, entries, use, colors, styles,
                                                 span_axis=span_axis, lift_axis=lift_axis))
        elif name == 'panel_root_section':
            root = _span_extent(surfaces[0], span_axis)[0]
            sheet.axes(name, lambda ax: fn(name)(ax, entries, root, colors, styles,
                                                 span_axis=span_axis, lift_axis=lift_axis))
    sheet.legend(labels, colors)
    return sheet.finish({'cp_references': ref_record} if ref_record else None)


def _read_cp_references(references, labels):
    """``(entries, inputs, record)`` of reference Cp specs ``FILE[:point=N]``.

    ``entries`` are ``[(label, reference)]`` (``read_reference_cp`` results),
    ``inputs`` the ``('reference', path)`` pairs and ``record`` the
    ``{label: {'path', 'point', 'ALPHA'}}`` of the manifest.
    """
    refs = _listed(references)
    labels = list(labels) if labels else []
    if labels and len(labels) != len(refs):
        raise PlotsError(f'{len(refs)} references but {len(labels)} reference labels')
    entries, inputs, record = [], [], {}
    for i, text in enumerate(refs):
        text = os.fspath(text)
        path, point = text, None
        head, sep, tail = text.rpartition(':point=')
        if sep:
            path, point = head, tail
        try:
            data = read_reference_cp(path, point)
        except ReaderError as exc:
            raise PlotsError(str(exc)) from exc
        stem = os.path.splitext(os.path.basename(path))[0]
        label = labels[i] if labels else stem + (f' @point {point}' if sep else '')
        entries.append((label, data))
        inputs.append(('reference', path))
        record[label] = {'path': os.path.abspath(path), 'point': data['point'],
                         'ALPHA': data['alpha']}
    return entries, inputs, record


def _listed(value):
    if value is None:
        return []
    return [os.fspath(value)] if isinstance(value, (str, os.PathLike)) else list(value)


def _span_extent(surface, span_axis):
    from .figures.wing import AXIS
    column = surface['points'][:, AXIS[span_axis]]
    return float(column.min()), float(column.max())


def _field_sections_figures(sheet, fn, entries, fields, field_stem, stations, colors, styles,
                            span_axis, lift_axis):
    """``field_sections[_<field>].png``: the field along the wall section at each station."""
    lo, hi = _span_extent(entries[0][1], span_axis)
    use = list(stations) if stations else [lo + f * (hi - lo) for f in (0.1, 0.5, 0.9)]
    for field in fields:
        for label, surface in entries:
            if field not in surface['point_data']:
                raise PlotsError(f'field_sections: field {field!r} not in {label}')
        values = [surface['point_data'][field].astype(float) for _, surface in entries]
        sheet.draw(field_stem('field_sections', field),
                   lambda fig, v=values, f=field: fn(fig, entries, use, v, f, colors, styles,
                                                     span_axis, lift_axis))


def _need(value, what):
    if value is None:
        raise PlotsError(f'span_load needs {what}')
    return value


def _span_load_figure(sheet, fn, entries, colors, styles, aoa, ref_area, semispan, cfgs,
                      n_stations, span_axis, lift_axis):
    """``span_load.png``: every entry's strip load, elliptic load and Trefftz efficiency."""
    from .figures.span_distributions import load_record
    count = len(entries)
    alphas = _need(_per_input(aoa, cfgs, 'AOA', count),
                   'the angle of attack: give aoa (--aoa) or a cfg with AOA (--cfg)')
    areas = _need(_per_input(ref_area, cfgs, 'REF_AREA', count),
                  'the reference area: give ref_area (--ref-area) or a cfg with REF_AREA (--cfg)')
    semispan = _need(semispan, 'the semispan: give semispan (--semispan)')
    for label, surface in entries:
        if 'Pressure_Coefficient' not in surface['point_data']:
            raise PlotsError(f'span_load: {label} has no Pressure_Coefficient')
    records = [(label, load_record(surface, alpha, area, semispan, n_stations, span_axis,
                                   lift_axis))
               for (label, surface), alpha, area in zip(entries, alphas, areas)]
    sheet.draw('span_load', lambda fig: fn(fig, records, semispan, colors, styles),
               size=(7.0, 6.0))


def _section_geometry_figure(sheet, fn, entries, colors, styles, stations, span_axis,
                             lift_axis, quantities):
    """``section_geometry.png``: the chosen panels of every entry at common stations."""
    from .figures.span_distributions import section_record
    lo, hi = _span_extent(entries[0][1], span_axis)
    use = list(stations) if stations else list(np.linspace(lo + 0.02 * (hi - lo),
                                                           lo + 0.96 * (hi - lo), 24))
    records = [(label, section_record(surface, use, span_axis, lift_axis))
               for label, surface in entries]
    picked = tuple(quantities) if quantities else ('twist', 'chord', 'thickness')
    sheet.draw('section_geometry',
               lambda fig: fn(fig, records, colors, styles, picked),
               size=(5.5, 3.7) if len(picked) == 1 else (7.0, 7.0))


def _surface_3d_overlay_figure(sheet, fn, entries, colors, span_axis, lift_axis, scale):
    """``surface_3d_overlay.png``: every entry's surface on one 3D axes."""
    from .figures.wing import surface_mesh
    meshes = [surface_mesh(surface)[:2] for _, surface in entries]
    base_pts = meshes[0][0]
    if scale != 1.0:
        for (label, _), (pts, _) in zip(entries[1:], meshes[1:]):
            if len(pts) != len(base_pts):
                raise PlotsError(f'surface_3d_overlay: {label} has {len(pts)} points but '
                                 f'{entries[0][0]} has {len(base_pts)}; a displacement scale '
                                 f'needs equal meshes')
        meshes = [meshes[0]] + [(base_pts + scale * (pts - base_pts), tris)
                                for pts, tris in meshes[1:]]
    labels = [label if i == 0 or scale == 1.0 else f'{label} (displacement x{scale:g})'
              for i, (label, _) in enumerate(entries)]

    def build(fig):
        ax = fig.add_subplot(111, projection='3d')
        fn(ax, [(lab, pts, tris) for lab, (pts, tris) in zip(labels, meshes)], colors,
           span_axis=span_axis, lift_axis=lift_axis)

    sheet.draw('surface_3d_overlay', build, size=(7.0, 4.6))


def _delta_planform_figures(sheet, fn, entries, fields, field_stem, lift_axis, span_axis):
    """One planform per later entry and field: its field minus the first entry's."""
    from .figures.wing import split_surfaces, surface_mesh
    base_label, base = entries[0]
    pts, tris, base_data = surface_mesh(base)
    upper, _ = split_surfaces(pts, tris, lift_axis)
    for label, surface in entries[1:]:
        if len(surface['points']) != len(pts):
            raise PlotsError(f'panel_delta_planform: {label} has {len(surface["points"])} '
                             f'points but {base_label} has {len(pts)}; the meshes differ')
        for field in fields:
            if field not in surface['point_data'] or field not in base_data:
                raise PlotsError(f'panel_delta_planform: field {field!r} missing in '
                                 f'{label} or {base_label}')
            delta = (surface['point_data'][field].astype(float)
                     - base_data[field].astype(float))

            def build(fig, d=delta, f=field, lab=label):
                ax = fig.subplots()
                col = fn(ax, pts, upper, d, span_axis=span_axis)
                fig.colorbar(col, ax=ax, label=f'{f}: {lab} minus {base_label}')
            sheet.draw(f'{field_stem("panel_delta_planform", field)}_{_slug(label)}', build,
                       size=(4.0, 6.0))


def _surface_3d_figures(sheet, name, fn, entries, fields, field_stem, lift_axis, span_axis):
    """The planform or 3D view of each surface and field, one figure per pair."""
    from .figures.wing import split_surfaces, surface_mesh
    for label, surface in entries:
        pts, tris, point_data = surface_mesh(surface)
        for field in fields:
            if field not in point_data:
                raise PlotsError(f'{name}: field {field!r} not in {label}; '
                                 f'available {sorted(point_data)}')
            values = point_data[field].astype(float)
            stem = field_stem(name, field)
            if len(entries) > 1:
                stem = f'{stem}_{_slug(label)}'
            if name == 'surface_3d':
                def build(fig, v=values, f=field):
                    ax = fig.add_subplot(111, projection='3d')
                    fn(ax, pts, tris, v, f, lift_axis=lift_axis)
            else:
                upper, _ = split_surfaces(pts, tris, lift_axis)

                def build(fig, v=values, up=upper, f=field):
                    ax = fig.subplots()
                    col = fn(ax, pts, up, v, float(v.min()), float(v.max()),
                             span_axis=span_axis)
                    fig.colorbar(col, ax=ax, label=f)
            sheet.draw(stem, build, size=(8.0, 6.0) if name == 'surface_3d' else (4.0, 6.0))


def plot_shape(paths, out_dir, labels=None, figures=None, options=None):
    """Wall shape figures of N 2D surface files; the first file is the base.

    ``shape_overlay`` overlays the outlines; ``delta_y`` (two or more files)
    draws each later wall's displacement from the first by x/c.
    """
    options = options or PlotOptions()
    paths = _paths(paths)
    labels = _labels(paths, labels)
    surfaces = _read(read_surface, paths, 'surface')
    from .extract import surface_ndim
    ctx = {'kind': 'surface', 'n_inputs': len(paths),
           'ndim': 2 if all(surface_ndim(s) == 2 for s in surfaces) else 3}
    sheet = _Sheet(out_dir, options, [('surface', p) for p in paths], 'plot_shape', ctx)
    colors, styles = _styles(len(paths), options)
    entries = list(zip(labels, surfaces))
    from .figures import FIGURES
    for name in sheet.select(figures):
        fn = FIGURES[name][0]
        if name == 'shape_overlay':
            sheet.axes(name, lambda ax: fn(ax, entries, colors, styles), size=(8.0, 4.0))
        else:
            sheet.draw(name, lambda fig: fn(fig, entries, colors, styles))
    sheet.legend(labels, colors)
    return sheet.finish()


def plot_view(surface, out_dir, field=None, baseline=None, cfg=None, dvs=None, lift_axis='z',
              options=None):
    """The isometric wing view ``paraview_view.png`` of a 3D surface file.

    ``baseline`` is an optional second surface drawn faint. With ``cfg`` (a
    run config whose parameterization is FFD) the baseline and deformed FFD
    lattices are drawn too, deformed by ``dvs`` (an array or a ``.npy`` path);
    ``lift_axis`` then comes from the config. ParaView renders it when
    ``$PARAVIEW_PVPYTHON`` names ``pvpython``, else matplotlib does.
    """
    options = options or PlotOptions()
    inputs = [('surface', surface)] + ([('surface', baseline)] if baseline else [])
    lattices = None
    if cfg is not None:
        if cfg.geometry.parameterization == 'ffd':
            from ..run_output.geometry_artifacts import ffd_lattice_pair
            values, dv_inputs = _load_dvs(dvs)
            inputs += dv_inputs
            lattices = ffd_lattice_pair(cfg, values)
        _, lift_axis = cfg.case.reference.axes(cfg.case.ndim)
    sheet = _Sheet(out_dir, options, inputs, 'plot_view',
                   {'kind': 'surface', 'n_inputs': 1, 'ndim': 3})
    from .figures import FIGURES
    sheet.select(['paraview_view'])
    try:
        entry = FIGURES['paraview_view'][0](surface, out_dir, field=field,
                                            baseline_surface=baseline, lattices=lattices,
                                            lift_axis=lift_axis)
    except Exception as exc:
        raise PlotsError(f"Figure 'paraview_view' failed to render: {exc}") from exc
    sheet.written['paraview_view'] = entry['path']
    return sheet.finish({'paraview_view': {k: v for k, v in entry.items() if k != 'path'}})


def _load_dvs(dvs):
    """``(array or None, [(role, path)])`` of design variables given as an array or ``.npy`` path."""
    if dvs is None:
        return None, []
    if isinstance(dvs, (str, os.PathLike)):
        if not os.path.isfile(dvs):
            raise PlotsError(f'Design-variable file not found: {dvs}')
        return np.load(dvs), [('design_variables', os.fspath(dvs))]
    return np.asarray(dvs, dtype=float), []


# ---- volumes -------------------------------------------------------------------

def plot_field(paths, out_dir, labels=None, fields=('Mach',), walls=None, length=1.0,
               margin=0.3, levels=28, options=None):
    """One contour figure per volume file and field, on one colour scale per field.

    ``walls`` are optional surface files of the same cases (one per volume)
    that set the window to the wall's bounding box plus ``margin`` chords;
    coordinates are divided by ``length``. With several volumes a standalone
    colour bar ``field_colorbar_<field>`` is written and the figures carry
    none; with one the bar is in the figure.
    """
    options = options or PlotOptions()
    paths = _paths(paths)
    labels = _labels(paths, labels)
    volumes = _read(read_volume, paths, 'volume')
    walls = _listed(walls)
    if walls and len(walls) != len(paths):
        raise PlotsError(f'{len(paths)} volumes but {len(walls)} wall files')
    wall_data = _read(read_surface, walls, 'surface') if walls else [None] * len(paths)
    ctx = {'kind': 'volume', 'n_inputs': len(paths), 'ndim': 2,
           'fields': _common_fields(volumes)}
    sheet = _Sheet(out_dir, options, [('volume', p) for p in paths]
                   + [('surface', w) for w in walls], 'plot_field', ctx)
    sheet.select(['field_contour'])
    from .extract import ExtractError
    from .figures.airfoil_figures import field_contour, field_range
    from .figures.comparison import save_colorbar
    size = options.figsize or ((8.0, 6.0) if len(paths) == 1 else (4.0, 2.25))
    for field in fields:
        try:
            vmin, vmax = field_range(volumes, field)
        except ExtractError as exc:
            raise PlotsError(str(exc)) from exc
        for (label, volume), wall in zip(zip(labels, volumes), wall_data):
            stem = 'field_contour' + (f'_{field}' if len(fields) > 1 else '')
            if len(paths) > 1:
                stem += f'_{_slug(label)}'

            def build(fig, volume=volume, wall=wall, title=label):
                ax = fig.subplots()
                cf = field_contour(ax, volume, wall, field=field, margin=margin,
                                   levels=levels, length=length, vmin=vmin, vmax=vmax,
                                   title=title if len(paths) > 1 else None)
                if len(paths) == 1:
                    fig.colorbar(cf, ax=ax, label=field)
                else:
                    fig.tight_layout(pad=0.3)
            sheet.draw(stem, build, size=size)
        if len(paths) > 1:
            bar = os.path.join(out_dir, f'field_colorbar_{_slug(field)}.{options.fmt}')
            save_colorbar(bar, vmin, vmax, field, dpi=sheet.dpi,
                          figsize=(size[0], max(0.35, size[1] * 0.2)))
            sheet.written[f'field_colorbar_{_slug(field)}'] = bar
    sheet.legend(labels, _styles(len(paths), options)[0])
    return sheet.finish()


# ---- solver histories ------------------------------------------------------------

def _plot_tables(kind, reader, paths, out_dir, labels, figures, columns, x, floor, floor_cfg,
                 adjoint_column, options):
    options = options or PlotOptions()
    paths = _paths(paths)
    labels = _labels(paths, labels)
    tables = _read(reader, paths, kind)
    ctx = {'kind': kind, 'n_inputs': len(paths)}
    inputs = [(kind, p) for p in paths] + ([('cfg', floor_cfg)] if floor_cfg else [])
    sheet = _Sheet(out_dir, options, inputs, 'plot_history', ctx)
    colors, styles = _styles(len(paths), options)
    entries = list(zip(labels, tables))
    from .figures import FIGURES
    wanted = sheet.select(figures, extra_default=('history_columns',) if columns else ())
    if floor is None and floor_cfg:
        floor = cfg_float(_read_cfg(floor_cfg), 'CONV_RESIDUAL_MINVAL')
    for name in wanted:
        fn = FIGURES[name][0]
        if name in ('residual_history', 'coefficient_history'):
            sheet.axes(name, lambda ax: fn(ax, entries, colors, styles, x=x))
        elif name == 'history_columns':
            if not columns:
                raise PlotsError('history_columns needs columns (expressions)')
            sheet.axes(name, lambda ax: fn(ax, entries, list(columns), colors, styles, x=x))
        elif name == 'adjoint_residuals':
            sheet.axes(name, lambda ax: fn(ax, entries, adjoint_column, floor, colors),
                       size=(8.0, 4.0))
    sheet.legend(labels, colors)
    return sheet.finish()


def plot_history(paths, out_dir, labels=None, figures=None, columns=None, x=None, floor=None,
                 floor_cfg=None, adjoint_column='rms[A_Rho]', options=None):
    """History figures of N SU2 history files (``.csv`` or tecplot ``.dat``).

    Default: ``residual_history`` and ``coefficient_history`` (plus
    ``history_columns`` when ``columns`` are given: expressions over the
    columns, ``series.evaluate`` syntax, e.g. ``['rms_Rho', 'CL/CD']``). ``x``
    is an expression for the x axis (default the row index). ``adjoint_residuals``
    (named) draws the last ``adjoint_column`` value of each file against
    ``floor``, or ``CONV_RESIDUAL_MINVAL`` of ``floor_cfg``.
    """
    return _plot_tables('history', read_history, paths, out_dir, labels, figures, columns, x,
                        floor, floor_cfg, adjoint_column, options)


def plot_stdout(paths, out_dir, labels=None, figures=None, columns=None, x=None, options=None):
    """The history figures of N SU2 screen-output files (their convergence tables)."""
    return _plot_tables('stdout', read_su2_stdout, paths, out_dir, labels, figures, columns,
                        x, None, None, 'rms[A_Rho]', options)


# ---- optimizer -----------------------------------------------------------------

def plot_optimizer(paths, out_dir, labels=None, figures=None, constraints=None, dv=None,
                   crest=None, bounds=(None, None), unit=1e-3, iteration=-1, options=None):
    """Optimizer figures of N pyOptSparse ``.hst`` files.

    Default: ``optimizer_history`` (objective by major iteration),
    ``constraint_history`` and ``dv_history`` (every design-variable element
    by major iteration, ``bounds`` dotted). ``constraints`` is ``{name: {'target': value or
    None, 'kind': 'eq' or 'ineq'}}`` (default: every scalar constraint of the
    first file, no target line). ``dv_profile`` (drawn when ``crest`` is given
    or it is named) plots design variable ``dv`` (default the only one) of
    major iteration ``iteration`` against the bump ``crest`` positions, with
    ``bounds`` ``(lower, upper)`` dotted, in units of ``unit``.
    """
    options = options or PlotOptions()
    paths = _paths(paths)
    labels = _labels(paths, labels)
    hsts = _read(read_hst, paths, 'hst')
    ctx = {'kind': 'hst', 'n_inputs': len(paths)}
    sheet = _Sheet(out_dir, options, [('hst', p) for p in paths], 'plot_optimizer', ctx)
    colors, styles = _styles(len(paths), options)
    entries = list(zip(labels, hsts))
    from .figures import FIGURES
    wanted = sheet.select(figures, extra_default=('dv_profile',) if crest is not None else ())
    for name in wanted:
        fn = FIGURES[name][0]
        if name == 'optimizer_history':
            sheet.axes(name, lambda ax: fn(ax, entries, colors, styles))
        elif name == 'constraint_history':
            specs = constraints or {
                k: {'target': None} for k, v in hsts[0]['constraints'].items()
                if np.ndim(v) == 1}
            if not specs:
                skip = 'no scalar constraint in the history'
                sheet.skipped[name] = skip
                print(f'  skip {name}: {skip}')
                continue

            def build(fig, specs=specs):
                axes = fig.subplots(len(specs), 1, squeeze=False)[:, 0]
                fig.set_size_inches(8, 2.5 * len(specs))
                fn(axes, entries, specs, colors, styles)
            sheet.draw(name, build)
        elif name == 'dv_history':
            if not any(h['dvs'] for h in hsts):
                skip = 'no design variable in the history'
                sheet.skipped[name] = skip
                print(f'  skip {name}: {skip}')
                continue

            def build(fig):
                ax = fig.subplots()
                fig.subplots_adjust(bottom=0.2)
                fn(ax, entries, *bounds)
            sheet.draw(name, build, size=(8.0, 4.5))
        elif name == 'dv_profile':
            if crest is None:
                raise PlotsError('dv_profile needs the bump crest positions (crest)')
            vectors = []
            for label, hst in entries:
                names = list(hst['dvs'])
                key = dv or (names[0] if len(names) == 1 else None)
                if key not in hst['dvs']:
                    raise PlotsError(f'{label}: name the design variable (dv) from {names}')
                table = np.asarray(hst['dvs'][key])
                vectors.append((label, table.reshape(len(table), -1)[iteration]))
            lower, upper = bounds
            sheet.draw(name, lambda fig: fn(fig, vectors, np.asarray(crest, dtype=float),
                                            colors, styles, lower=lower, upper=upper,
                                            unit=unit))
    sheet.legend(labels, colors)
    return sheet.finish()


# ---- sample series ----------------------------------------------------------------

def _load_groups(groups):
    """``[[sample dict]]`` and the ``(role, path)`` inputs of N groups of ``SampleSpec``.

    A spec gives one sample, or many (``polar``, ``table``); a group holds the
    samples of its specs in order.
    """
    from .series import SampleSpec, SeriesError, load_samples
    samples, inputs = [], []
    try:
        for group in groups:
            specs = [group] if isinstance(group, SampleSpec) else list(group)
            samples.append([sample for spec in specs for sample in load_samples(spec)])
            for spec in specs:
                inputs += spec.inputs()
    except (ReaderError, SeriesError) as exc:
        raise PlotsError(str(exc)) from exc
    return samples, inputs


def _unique_mean(x, y):
    """``x`` and ``y`` sorted by ``x`` with duplicate ``x`` replaced by the mean ``y``."""
    ux, inverse = np.unique(x, return_inverse=True)
    return ux, np.bincount(inverse, weights=y) / np.bincount(inverse)


def _comparison(sim, ref, expr):
    """``[{x, S, D, E}]`` at each reference sample: ``S`` interpolated, ``D`` the reference.

    ``S`` is the simulation y interpolated linearly in x over its unique x
    (duplicates averaged); a reference x outside the simulation x range gives
    ``None`` for ``S`` and ``E``. ``E = S - D``.
    """
    sx, sy = _unique_mean(sim['x'], sim['y'][expr])
    rows = []
    for x, d in zip(ref['x'], ref['y'][expr]):
        inside = len(sx) > 0 and sx[0] <= x <= sx[-1] and np.isfinite(x)
        value = float(np.interp(x, sx, sy)) if inside else None
        value = value if value is None or np.isfinite(value) else None
        rows.append({'x': float(x), 'S': value, 'D': float(d),
                     'E': None if value is None else value - float(d)})
    return rows


def _print_comparison(key, rows):
    def cell(v):
        return '      n/a' if v is None else f'{v:9.4g}'
    print(f'  comparison {key}')
    print('        x         S         D         E')
    for r in rows:
        print('  ' + ' '.join(cell(r[k]) for k in ('x', 'S', 'D', 'E')))


def plot_sweep(groups, out_dir, x, y, labels=None, peak=(), log_y=(), y_labels=None,
               x_label=None, options=None, reference=None):
    """One sweep figure: each expression of ``y`` against ``x``, N curves overlaid.

    ``groups`` is a list of N groups, each a list of ``series.SampleSpec``
    (a file gives one sample, or many for the ``polar`` and ``table`` kinds).
    ``x`` and ``y`` are expressions over the sample names
    (``series.evaluate``): ``plot_sweep(groups, out_dir, 'MACH',
    ['sqrt(CL)/CD', 'CD'], peak=['sqrt(CL)/CD'])``. Each group is sorted by
    ``x``. ``peak`` expressions get a maximum marker and the parabola vertex,
    ``log_y`` a log axis, ``y_labels`` maps an expression to its axis label.

    ``reference`` is one bool per group; a True group is a reference drawn as
    markers only, without peak markers, and samples lacking a needed column
    are dropped. For every simulation group, reference group and y expression
    the comparison ``E = S - D`` at each reference sample is printed and
    written to ``manifest.json`` under ``sweep.comparison`` (``S`` the
    simulation y linearly interpolated at the reference x over unique x with
    duplicates averaged, ``None`` outside the simulation x range, ``D`` the
    reference y).
    """
    from .series import SeriesError, build_series
    options = options or PlotOptions()
    y = [y] if isinstance(y, str) else list(y)
    groups = list(groups)
    if not groups:
        raise PlotsError('no sample groups')
    flags = [bool(f) for f in reference] if reference else [False] * len(groups)
    if len(flags) != len(groups):
        raise PlotsError(f'{len(groups)} groups but {len(flags)} reference flags')
    sample_sets, inputs = _load_groups(groups)
    labels = labels or [f'series {i + 1}' for i in range(len(groups))]
    if len(labels) != len(groups):
        raise PlotsError(f'{len(groups)} groups but {len(labels)} labels')
    try:
        entries = [(label, build_series(samples, x, y, skip_incomplete=is_ref))
                   for label, samples, is_ref in zip(labels, sample_sets, flags)]
    except SeriesError as exc:
        raise PlotsError(str(exc)) from exc
    ctx = {'kind': 'series', 'n_inputs': len(groups)}
    sheet = _Sheet(out_dir, options, inputs, 'plot_sweep', ctx)
    sheet.select(['sweep'])
    colors, styles = _styles(len(groups), options)
    from .figures.sweep import sweep
    sheet.draw('sweep', lambda fig: sweep(fig, entries, colors, styles, x_label or x,
                                          y_labels, tuple(peak), tuple(log_y),
                                          reference=flags),
               size=(11.0, 3.4))
    sheet.legend(labels, colors)
    extra = {'sweep': {'x': x, 'y': y, 'peaks': {
        f'{label}: {e}': dict(zip(('x_max', 'y_max', 'x_vertex', 'y_vertex'), _peak(data, e)))
        for (label, data), is_ref in zip(entries, flags) if not is_ref for e in peak}}}
    comparison = {}
    for sim_label, sim, sim_ref in ((l, d, f) for (l, d), f in zip(entries, flags)):
        if sim_ref:
            continue
        for (ref_label, ref), is_ref in zip(entries, flags):
            if not is_ref:
                continue
            for expr in y:
                key = f'{sim_label} vs {ref_label}: {expr}'
                comparison[key] = _comparison(sim, ref, expr)
                _print_comparison(key, comparison[key])
    if comparison:
        extra['sweep']['comparison'] = comparison
    return sheet.finish(extra)


def _peak(data, expr):
    from .series import fit_peak
    return fit_peak(data['x'], data['y'][expr])


def plot_iterations(groups, out_dir, exprs=(), rows=(), labels=None, percent=True,
                    options=None, y_labels=None):
    """Per-iteration change and constraint-margin figures of N groups of sample files.

    Each group is a list of ``series.SampleSpec``, one per iteration in order
    (an ``iteration`` value in the samples, if any, is the x axis). ``exprs``
    gives ``iteration_change`` (one panel per expression, change from the
    first iteration); ``rows`` gives ``linear_margins`` (dicts ``name``,
    ``terms`` = ``[(column, coeff)]``, ``offset``, ``lower``/``upper``/``equals``;
    ``series.linear_terms`` makes the terms of a linear expression). Columns
    are addressed by name or by ``series.alias`` (``x[3]`` is ``x_3``).
    ``y_labels`` maps an expression of ``exprs`` to its y axis label.
    """
    from .series import SeriesError, alias
    options = options or PlotOptions()
    groups = list(groups)
    sample_sets, inputs = _load_groups(groups)
    labels = labels or [f'series {i + 1}' for i in range(len(groups))]
    if len(labels) != len(groups):
        raise PlotsError(f'{len(groups)} groups but {len(labels)} labels')
    tables = []
    for samples in sample_sets:
        names = [n for n in samples[0] if all(n in s for s in samples)]
        table = {n: np.array([s[n] for s in samples], dtype=float) for n in names}
        for n in names:
            table.setdefault(alias(n), table[n])
        tables.append(table)
    entries = list(zip(labels, tables))
    ctx = {'kind': 'series', 'n_inputs': len(groups)}
    sheet = _Sheet(out_dir, options, inputs, 'plot_iterations', ctx)
    colors, styles = _styles(len(groups), options)
    from .figures import FIGURES
    names = (['iteration_change'] if exprs else []) + (['linear_margins'] if rows else [])
    sheet.select(names)
    try:
        for name in names:
            fn = FIGURES[name][0]
            if name == 'iteration_change':
                sheet.draw(name, lambda fig: fn(fig, entries, list(exprs), colors, styles,
                                                percent=percent, y_labels=y_labels))
            else:
                sheet.draw(name, lambda fig: fn(fig, entries, list(rows), colors, styles))
    except SeriesError as exc:
        raise PlotsError(str(exc)) from exc
    sheet.legend(labels, colors)
    return sheet.finish()


# ---- configuration figures ---------------------------------------------------------

def plot_dv_map(cfg, out_dir, figures=None, wall=None, dvs=None, options=None):
    """The design-variable figures of a run config: the DV map and the basis figures.

    Default ``dv_map``. ``wall`` is an optional surface file drawn with the
    FFD box; ``dvs`` (an array or ``.npy`` path) the shape design variables the
    FFD influence panel uses (unit values when None).
    """
    options = options or PlotOptions()
    from .dvmap import dv_map_panels, load_lattice
    from .figures import FIGURES
    param = cfg.geometry.parameterization
    wall_points = None
    inputs = []
    if getattr(cfg, 'source', None) and os.path.isfile(cfg.source):
        inputs.append(('config', cfg.source))
    if wall:
        wall_points = _read(read_surface, [wall], 'surface')[0]['points'].astype(float)
        inputs.append(('surface', wall))
    values, dv_inputs = _load_dvs(dvs)
    inputs += dv_inputs
    ctx = {'kind': 'config', 'n_inputs': 1, 'ndim': cfg.case.ndim, 'parameterization': param}
    sheet = _Sheet(out_dir, options, inputs, 'plot_dv_map', ctx)
    lattice = load_lattice(cfg.ffd_path) if param == 'ffd' else None
    for name in sheet.select(figures):
        fn = FIGURES[name][0]
        if name == 'dv_map':
            sheet.draw(name, lambda fig: dv_map_panels(fig, cfg, lattice, dvs=values,
                                                       wall=wall_points))
        elif name == 'ffd_box':
            sheet.axes(name, lambda ax: fn(ax, lattice, wall=wall_points))
        elif name in ('dv_influence', 'cst_dv_influence', 'hh_dv_influence'):
            use = None if param in ('cst', 'hicks_henne') else values
            sheet.axes(name, lambda ax: fn(ax, cfg, lattice, use))
        else:
            sheet.axes(name, lambda ax: fn(ax, cfg))
    return sheet.finish()


# ---- probes ------------------------------------------------------------------------

def run_probes(volume_path, probes, out_dir, ndim=2, keep=False):
    """Sample ``probes`` (spec dicts) on one volume file.

    Writes ``<out_dir>/probes/<name>/probe.{vtp|vtu,csv,json}``. Returns the
    metadata dicts, or ``(meta, probe)`` pairs with ``keep``. A box probe
    samples one z layer, so it needs ``ndim`` 2.
    """
    from .probe import (
        probe_line, probe_box, probe_plane,
        write_vtp, write_vtu, write_csv, ProbeError,
    )
    source = _read(read_volume, [volume_path], 'volume')[0]
    results = []
    for spec in probes:
        name, kind = spec['name'], spec['kind']
        probe_dir = os.path.join(out_dir, 'probes', name)
        os.makedirs(probe_dir, exist_ok=True)
        probe_fields = spec.get('fields') or None
        n = spec.get('n', 200)
        if kind != 'line':
            n = tuple(n) if isinstance(n, (list, tuple)) else (n, n)
        try:
            if kind == 'line':
                where = f"line from {list(spec['p0'])} to {list(spec['p1'])}"
                probe = probe_line(source, spec['p0'], spec['p1'], n=n, fields=probe_fields)
            elif kind == 'box':
                if ndim == 3:
                    raise PlotsError(f'Probe {name!r}: a box probe samples one z layer; '
                                     f'3D boxes are not implemented, use a plane')
                lo, hi = spec.get('lo') or spec.get('p0'), spec.get('hi') or spec.get('p1')
                where = f'box from {list(lo)} to {list(hi)}'
                probe = probe_box(source, lo=lo, hi=hi, n=n, fields=probe_fields)
            elif kind == 'plane':
                where = f"plane at {list(spec['origin'])} normal {list(spec['normal'])}"
                probe = probe_plane(source, spec['origin'], spec['normal'], n=n,
                                    fields=probe_fields)
            else:
                raise PlotsError(f'Probe {name!r}: unknown kind {kind!r}. '
                                 f'Use "line", "box" or "plane".')
        except ProbeError as exc:
            raise PlotsError(f'Probe {name!r} failed: {exc}') from exc
        if probe.n_missed == probe.n_total:
            raise PlotsError(f'Probe {name!r}: all {probe.n_total} samples are NaN; '
                             f'the {where} missed the mesh.')
        if kind == 'line':
            write_vtp(probe, os.path.join(probe_dir, 'probe.vtp'))
        else:
            write_vtu(probe, os.path.join(probe_dir, 'probe.vtu'))
        write_csv(probe, os.path.join(probe_dir, 'probe.csv'))

        meta = {'name': name, 'kind': kind, 'source': os.path.abspath(volume_path),
                'n_missed': probe.n_missed, 'n_total': probe.n_total,
                'fields': list(probe.fields.keys())}
        for key in {'line': ('p0', 'p1'), 'box': ('lo', 'hi'),
                    'plane': ('origin', 'normal')}[kind]:
            value = spec.get(key) or spec.get({'lo': 'p0', 'hi': 'p1'}.get(key, key))
            meta[key] = [float(v) for v in value]
        with open(os.path.join(probe_dir, 'probe.json'), 'w') as f:
            json.dump(meta, f, indent=2)
            f.write('\n')
        results.append((meta, probe) if keep else meta)
        if 0 < probe.n_missed < probe.n_total:
            print(f'  probe {name!r}: {probe.n_missed}/{probe.n_total} '
                  f'samples outside the mesh (NaN)')
    return results


def plot_probes(volume_path, probes, out_dir, ndim=2, fields=None, options=None):
    """Sample ``probes`` on one volume file and draw one figure per probe.

    ``probe_line`` for a line probe, ``probe_region`` for a box or plane (the
    first of ``fields``, else the first field). Data goes under
    ``<out_dir>/probes/``. Returns ``{stem: path}``.
    """
    options = options or PlotOptions()
    os.makedirs(out_dir, exist_ok=True)
    results = run_probes(volume_path, probes, out_dir, ndim=ndim, keep=True)
    kinds = {('line' if meta['kind'] == 'line' else 'region') for meta, _ in results}
    ctx = {'kind': 'probe', 'n_inputs': len(results), 'ndim': ndim, 'probes': kinds}
    sheet = _Sheet(out_dir, options, [('volume', volume_path)], 'plot_probes', ctx)
    from .figures import FIGURES
    for meta, probe in results:
        name = 'probe_line' if meta['kind'] == 'line' else 'probe_region'
        reason = unmet(FIGURES[name][1], ctx)
        if reason:
            raise PlotsError(f'{name}: {reason}')
        fn = FIGURES[name][0]

        def build(ax, probe=probe, meta=meta, name=name):
            if name == 'probe_line':
                fn(ax, probe, fields=fields)
            else:
                fn(ax, probe, (fields or list(probe.fields))[0])
            ax.set_title(f"probe {meta['name']}", fontsize=9)
        sheet.axes(f"probe_{meta['name']}", build)
    extra = {'probes': [meta for meta, _ in results]}
    return sheet.finish(extra)
