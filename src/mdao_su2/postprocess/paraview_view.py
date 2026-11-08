"""Isometric 3D view of the FFD lattice and the wing surface (ONERA M6 tutorial style).

``render_view`` writes the view's inputs (baseline and deformed FFD lattice,
baseline and final surface) beside the figure, then renders them with
ParaView's ``pvpython`` when ``$PARAVIEW_PVPYTHON`` names it, else with a
matplotlib fallback. The renderer used is returned and goes into the plot
manifest. No ParaView path is assumed.
"""
import os
import shutil
import subprocess

import numpy as np

__all__ = ['render_view', 'pvpython_script', 'PVPYTHON_ENV']

#: Environment variable holding the user's pvpython executable.
PVPYTHON_ENV = 'PARAVIEW_PVPYTHON'

_AXIS = {'x': 0, 'y': 1, 'z': 2}


def _camera(bounds_lo, bounds_hi, lift_axis):
    """Focal point, position and view-up of an isometric camera with lift up."""
    centre = 0.5 * (bounds_lo + bounds_hi)
    size = float(np.linalg.norm(bounds_hi - bounds_lo))
    up = np.zeros(3)
    up[_AXIS[lift_axis]] = 1.0
    others = [a for a in range(3) if a != _AXIS[lift_axis]]
    direction = up.copy()
    direction[others[0]], direction[others[1]] = 1.0, -1.0
    direction /= np.linalg.norm(direction)
    return centre, centre + 2.5 * size * direction, up


def pvpython_script(inputs, png, field, camera, size=(1600, 1000)):
    """The ``paraview.simple`` script that renders ``inputs`` to ``png``.

    ``inputs`` maps ``surface``, ``baseline_surface``, ``ffd`` and
    ``baseline_ffd`` to VTU paths; missing entries are left out.
    """
    focal, position, up = (list(map(float, v)) for v in camera)
    lines = [
        'from paraview.simple import *',
        'view = CreateView("RenderView")',
        f'view.ViewSize = {list(size)}',
        'view.Background = [1.0, 1.0, 1.0]',
        'view.OrientationAxesVisibility = 1',
        'def show(path, rgb=None, field=None, opacity=1.0, wire=False):',
        '    r = XMLUnstructuredGridReader(FileName=[path])',
        '    d = Show(r, view)',
        '    if field:',
        '        ColorBy(d, ("POINTS", field))',
        '        d.RescaleTransferFunctionToDataRange(True)',
        '        d.SetScalarBarVisibility(view, True)',
        '    else:',
        '        ColorBy(d, None)',
        '        d.AmbientColor = rgb',
        '        d.DiffuseColor = rgb',
        '    d.Opacity = opacity',
        '    if wire:',
        '        d.SetRepresentationType("Wireframe")',
        '        d.LineWidth = 2.0',
        '    return d',
    ]
    if 'baseline_surface' in inputs:
        lines.append(f'show({inputs["baseline_surface"]!r}, rgb=[0.6, 0.6, 0.6], opacity=0.25)')
    if 'surface' in inputs:
        lines.append(f'show({inputs["surface"]!r}, rgb=[0.75, 0.75, 0.8], field={field!r})')
    if 'baseline_ffd' in inputs:
        lines.append(f'show({inputs["baseline_ffd"]!r}, rgb=[0.4, 0.4, 0.4], wire=True)')
    if 'ffd' in inputs:
        lines.append(f'show({inputs["ffd"]!r}, rgb=[0.85, 0.2, 0.15], wire=True)')
    lines += [
        f'view.CameraFocalPoint = {focal}',
        f'view.CameraPosition = {position}',
        f'view.CameraViewUp = {up}',
        'view.ResetCamera(False)',
        f'SaveScreenshot({png!r}, view, ImageResolution={list(size)})',
    ]
    return '\n'.join(lines) + '\n'


def _fallback(png, surface, baseline_surface, lattices, field, lift_axis, camera_elev=30.0,
              camera_azim=-45.0):
    """Matplotlib rendering of the same inputs."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    from .figures.wing import surface_3d, surface_mesh
    from .readers import read_surface

    fig = plt.figure(figsize=(11, 7))
    ax = fig.add_subplot(111, projection='3d')
    pts, tris, pd = surface_mesh(read_surface(surface))
    values = pd[field].astype(float) if field and field in pd else pts[:, _AXIS[lift_axis]]
    surface_3d(ax, pts, tris, values, field or lift_axis, lift_axis=lift_axis,
               elev=camera_elev, azim=camera_azim)
    h = _AXIS[lift_axis]
    order = [a for a in range(3) if a != h] + [h]
    for lattice, colour in lattices:
        ni, nj, nk = lattice.shape[:3]
        segs = []
        for i in range(ni):
            for j in range(nj):
                for k in range(nk):
                    for di, dj, dk in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
                        a, b, c = i + di, j + dj, k + dk
                        if a < ni and b < nj and c < nk:
                            segs.append([lattice[i, j, k][order], lattice[a, b, c][order]])
        ax.add_collection3d(Line3DCollection(segs, colors=colour, linewidths=0.8))
    fig.savefig(png, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def render_view(surface, out_dir, field=None, baseline_surface=None, lattices=None,
                lift_axis='z'):
    """Write the view inputs and render ``paraview_view.png``; returns a manifest entry.

    ``surface`` and ``baseline_surface`` are surface ``.vtu`` paths.
    ``lattices`` is None or ``(baseline, deformed)`` FFD point arrays
    (``run_output/geometry_artifacts.py::ffd_lattice_pair``), written beside the
    figure as VTU files for ParaView.
    """
    from ..run_output.geometry_artifacts import write_lattice_vtu
    from .figures.wing import surface_mesh
    from .readers import read_surface

    os.makedirs(out_dir, exist_ok=True)
    inputs = {'surface': surface}
    if baseline_surface:
        inputs['baseline_surface'] = baseline_surface
    drawn = []
    if lattices is not None:
        base, deformed = lattices
        inputs['baseline_ffd'] = write_lattice_vtu(
            os.path.join(out_dir, 'paraview_ffd_baseline.vtu'), base)
        inputs['ffd'] = write_lattice_vtu(os.path.join(out_dir, 'paraview_ffd.vtu'), deformed)
        drawn = [(base, '0.4'), (deformed, 'tab:red')]
    pts = np.vstack([surface_mesh(read_surface(surface))[0]]
                    + [lat.reshape(-1, 3) for lat, _ in drawn])
    camera = _camera(pts.min(0), pts.max(0), lift_axis)
    png = os.path.join(out_dir, 'paraview_view.png')
    pvpython = os.environ.get(PVPYTHON_ENV)
    if pvpython and (os.path.isfile(pvpython) or shutil.which(pvpython)):
        script = os.path.join(out_dir, 'paraview_view.py')
        with open(script, 'w') as fh:
            fh.write(pvpython_script(inputs, png, field, camera))
        subprocess.run([pvpython, script], check=True, capture_output=True, text=True)
        return {'path': png, 'renderer': 'paraview', 'pvpython': pvpython, 'script': script}
    _fallback(png, surface, baseline_surface, drawn, field, lift_axis)
    return {'path': png, 'renderer': 'matplotlib',
            'reason': f'${PVPYTHON_ENV} is not set to a pvpython executable'}
