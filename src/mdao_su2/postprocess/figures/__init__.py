"""Figure registry: every figure with the input kinds and capabilities it needs."""
import numpy as np

from .airfoil_figures import cp_chord, shape_overlay, field_contour, field_surface
from .wing import (
    panel_field_planform, panel_delta_planform, panel_sections, field_sections,
    panel_root_section, surface_3d, surface_3d_overlay,
)
from .span_distributions import span_load_figure, section_geometry
from .convergence import (
    residual_history, coefficient_history, history_columns, adjoint_residuals,
    optimizer_history, constraint_history,
)
from ..dvmap import (
    ffd_box, dv_map, dv_influence, basis_functions, cst_classification,
)
from ..paraview_view import render_view
from .comparison import save_legend, save_colorbar
from .sweep import sweep
from .optimization import (
    iteration_change, linear_margins, dv_profile, dv_history, delta_y_figure,
)
from .surface_mach import surface_mach

__all__ = ['FIGURES', 'probe_line_figure', 'probe_region_figure', 'save_legend',
           'save_colorbar']


def probe_line_figure(ax, probe, fields=None, **kwargs):
    """Value vs arc length for a LineProbe.  N fields on one axis or stacked."""
    from ..style import TEXT, GRID
    names = fields or list(probe.fields.keys())
    for i, name in enumerate(names):
        arr = probe.fields[name]
        if arr.ndim == 2:
            mag = np.linalg.norm(arr, axis=1)
            ax.plot(probe.arc_length, mag, label=f'|{name}|',
                    color=f'C{i}', **kwargs)
        else:
            ax.plot(probe.arc_length, arr, label=name,
                    color=f'C{i}', **kwargs)
    ax.set_xlabel('Arc length', fontsize=9, color=TEXT)
    ax.set_ylabel('Value', fontsize=9, color=TEXT)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.25, color=GRID)
    ax.tick_params(labelsize=8, colors=TEXT)


def probe_region_figure(ax, probe, field, **kwargs):
    """A field on a sampled region as a pcolormesh."""
    from ..style import TEXT, GRID
    arr = probe.fields[field]
    if arr.ndim == 3:
        arr = np.linalg.norm(arr, axis=2)
    pts = probe.points
    x = pts[:, :, 0] if pts.ndim == 3 else pts[:, 0].reshape(probe.shape)
    y = pts[:, :, 1] if pts.ndim == 3 else pts[:, 1].reshape(probe.shape)

    cmap = kwargs.pop('cmap', 'coolwarm')
    cm = ax.pcolormesh(x, y, arr, cmap=cmap, shading='auto', **kwargs)
    ax.set_aspect('equal')
    ax.set_xlabel('x', fontsize=9, color=TEXT)
    ax.set_ylabel('y', fontsize=9, color=TEXT)
    ax.tick_params(labelsize=8, colors=TEXT)
    return cm


def _req(entry, inputs, ndim=None, parameterization=None, fields=(), probe=None,
         default=True, min_inputs=1):
    """What a figure needs.

    ``entry`` is the ``postprocess/plots.py`` function that draws it and
    ``inputs`` the input kinds it accepts (``surface``, ``volume``,
    ``history``, ``stdout``, ``hst``, ``series``, ``config``, ``probe``); one
    of them suffices. ``min_inputs`` is the least number of files, ``fields``
    the surface or volume point-data names that must exist, ``default`` whether
    the entry point draws it when no figure is named.
    """
    return {'entry': entry, 'inputs': tuple(inputs), 'ndim': ndim,
            'parameterization': parameterization, 'fields': tuple(fields),
            'probe': probe, 'default': default, 'min_inputs': min_inputs}


_SOLVER = ('history', 'stdout')

#: name -> (callable, requires). An entry point draws a figure only when its
#: inputs meet ``requires`` (``postprocess/plots.py::unmet``); a named figure
#: that does not raises, a default one is listed as skipped, with the reason,
#: in ``manifest.json``.
FIGURES = {
    'cp_chord':              (cp_chord, _req('plot_surface', ('surface',), 2,
                                             fields=('Pressure_Coefficient',))),
    'field_surface':         (field_surface, _req('plot_surface', ('surface',), 2)),
    'surface_mach':          (surface_mach, _req('plot_surface', ('surface',), 2,
                                                 fields=('Pressure_Coefficient',),
                                                 default=False)),
    'panel_field_planform':  (panel_field_planform, _req('plot_surface', ('surface',), 3)),
    'panel_sections':        (panel_sections, _req('plot_surface', ('surface',), 3)),
    'panel_root_section':    (panel_root_section, _req('plot_surface', ('surface',), 3)),
    'surface_3d':            (surface_3d, _req('plot_surface', ('surface',), 3)),
    'surface_3d_overlay':    (surface_3d_overlay, _req('plot_surface', ('surface',), 3,
                                                     default=False, min_inputs=2)),
    'panel_delta_planform':  (panel_delta_planform, _req('plot_surface', ('surface',), 3,
                                                         default=False, min_inputs=2)),
    'field_sections':        (field_sections, _req('plot_surface', ('surface',), 3,
                                                   default=False)),
    'span_load':             (span_load_figure, _req('plot_surface', ('surface',), 3,
                                                     fields=('Pressure_Coefficient',),
                                                     default=False)),
    'section_geometry':      (section_geometry, _req('plot_surface', ('surface',), 3,
                                                     default=False)),
    'shape_overlay':         (shape_overlay, _req('plot_shape', ('surface',), 2)),
    'delta_y':               (delta_y_figure, _req('plot_shape', ('surface',), 2, min_inputs=2)),
    'paraview_view':         (render_view, _req('plot_view', ('surface',), 3, default=False)),
    'field_contour':         (field_contour, _req('plot_field', ('volume',), 2)),
    'residual_history':      (residual_history, _req('plot_history', _SOLVER)),
    'coefficient_history':   (coefficient_history, _req('plot_history', _SOLVER)),
    'history_columns':       (history_columns, _req('plot_history', _SOLVER, default=False)),
    'adjoint_residuals':     (adjoint_residuals, _req('plot_history', ('history',),
                                                      default=False)),
    'optimizer_history':     (optimizer_history, _req('plot_optimizer', ('hst',))),
    'constraint_history':    (constraint_history, _req('plot_optimizer', ('hst',))),
    'dv_history':            (dv_history, _req('plot_optimizer', ('hst',))),
    'dv_profile':            (dv_profile, _req('plot_optimizer', ('hst',), default=False)),
    'sweep':                 (sweep, _req('plot_sweep', ('series',), default=False)),
    'iteration_change':      (iteration_change, _req('plot_iterations', ('series',),
                                                     default=False)),
    'linear_margins':        (linear_margins, _req('plot_iterations', ('series',),
                                                   default=False)),
    'probe_line':            (probe_line_figure, _req('plot_probes', ('probe',), probe='line')),
    'probe_region':          (probe_region_figure, _req('plot_probes', ('probe',), 2,
                                                        probe='region')),
    'dv_map':                (dv_map, _req('plot_dv_map', ('config',))),
    'ffd_box':               (ffd_box, _req('plot_dv_map', ('config',), parameterization=('ffd',),
                                            default=False)),
    'dv_influence':          (dv_influence, _req('plot_dv_map', ('config',),
                                                 parameterization=('ffd',), default=False)),
    'basis_functions':       (basis_functions, _req('plot_dv_map', ('config',),
                                                    parameterization=('ffd',), default=False)),
    'cst_basis_functions':   (basis_functions, _req('plot_dv_map', ('config',), 2,
                                                    parameterization=('cst',), default=False)),
    'cst_dv_influence':      (dv_influence, _req('plot_dv_map', ('config',), 2,
                                                 parameterization=('cst',), default=False)),
    'cst_classification':    (cst_classification, _req('plot_dv_map', ('config',), 2,
                                                       parameterization=('cst',),
                                                       default=False)),
    'hh_basis_functions':    (basis_functions, _req('plot_dv_map', ('config',), 2,
                                                    parameterization=('hicks_henne',),
                                                    default=False)),
    'hh_dv_influence':       (dv_influence, _req('plot_dv_map', ('config',), 2,
                                                 parameterization=('hicks_henne',),
                                                 default=False)),
}
