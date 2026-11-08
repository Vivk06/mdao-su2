"""Spanwise figures of a 3D wing from its surface files: section load and section geometry.

``entries`` are ``[(label, record)]``; colours and line styles are per entry.
The load comes from ``span_load.span_load`` (pressure force per strip) and the
geometry from ``wing.section`` cuts, so neither figure reads anything but the
surface triangles, the surface Cp and the scalars passed in.
"""
import numpy as np

from ..span_load import elliptic_load, span_load, trefftz_efficiency
from ..style import TEXT
from .panels import resolve_styles, style_axes
from .wing import section, surface_mesh

__all__ = ['load_record', 'section_record', 'span_load_figure', 'section_geometry']

#: Elliptic loads below this fraction of their peak are left out of the ratio panel.
RATIO_FLOOR = 0.25


def load_record(surface, alpha_deg, s_ref, semispan, n_stations, span_axis='y',
                lift_axis='z'):
    """Strip load, elliptic load of the same ``C_L`` and Trefftz efficiency of one surface.

    ``alpha_deg`` is the angle of attack, ``s_ref`` the reference area of the
    modelled (half) wing and ``semispan`` its span coordinate extent that the
    elliptic load and the sine fit use. The surface needs ``Pressure_Coefficient``.
    """
    pts, tris, data = surface_mesh(surface)
    load = span_load(pts, tris, data['Pressure_Coefficient'], alpha_deg, s_ref, n_stations,
                     span_axis, lift_axis)
    e_tr, _ = trefftz_efficiency(load['y'], load['lprime'], semispan)
    return {'y': load['y'], 'lprime': load['lprime'], 'CL': load['CL'], 'e_trefftz': e_tr,
            'elliptic': elliptic_load(load['y'], load['CL'], s_ref, semispan)}


def span_load_figure(fig, entries, semispan, colors=None, styles=None):
    """Sectional load ``l'(y)`` and its ratio to the elliptic load of the same lift.

    ``entries`` are ``[(label, load_record)]``. The top panel draws each load
    solid and its elliptic distribution dotted in the same colour, with the
    Trefftz efficiency ``e`` of the sine fit in the legend; the bottom panel
    draws ``l' / l'_elliptic`` where the elliptic load exceeds
    ``RATIO_FLOOR`` of its peak, with 1 as the reference line.
    """
    colors, styles = resolve_styles(len(entries), colors, styles)
    top, bottom = fig.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3, 2]})
    for i, (label, rec) in enumerate(entries):
        top.plot(rec['y'], rec['lprime'], color=colors[i], ls=styles[i], lw=1.4,
                 label=f"{label}: C_L = {rec['CL']:.4f}, e = {rec['e_trefftz']:.4f}")
        top.plot(rec['y'], rec['elliptic'], color=colors[i], ls=':', lw=1.0)
        keep = rec['elliptic'] > RATIO_FLOOR * rec['elliptic'].max()
        bottom.plot(rec['y'][keep], rec['lprime'][keep] / rec['elliptic'][keep],
                    color=colors[i], ls=styles[i], lw=1.4)
    top.plot([], [], color='grey', ls=':', lw=1.0, label='elliptic, same C_L')
    bottom.axhline(1.0, color='black', lw=0.8)
    style_axes(top, ylabel="sectional lift l' = c_l c (per unit q)")
    style_axes(bottom, xlabel='span coordinate', ylabel="l' / l' elliptic")
    top.set_xlim(0.0, semispan)
    top.legend(fontsize=7, framealpha=0.9)
    fig.tight_layout(rect=(0, 0.03, 1, 1))


def _vertical_extent(loop, xs):
    """Largest spread of the loop's lift coordinate on each vertical line ``x = xs[k]``."""
    a, b = loop[:-1], loop[1:]
    out = np.zeros(len(xs))
    for k, x in enumerate(xs):
        cross = (a[:, 0] - x) * (b[:, 0] - x) < 0.0
        if cross.sum() < 2:
            continue
        t = (x - a[cross, 0]) / (b[cross, 0] - a[cross, 0])
        h = a[cross, 1] + t * (b[cross, 1] - a[cross, 1])
        out[k] = h.max() - h.min()
    return out


def section_record(surface, stations, span_axis='y', lift_axis='z'):
    """``{y, chord, twist_deg, t_c}`` of the wall sections at ``stations``.

    The chord is the leading-to-trailing-edge distance (the extreme x points of
    the section), ``twist_deg`` the chord-line angle, positive when the leading
    edge is above the trailing edge (nose up), and ``t_c`` the largest
    vertical extent of the section, measured in the frame rotated by the
    twist, over the chord. A station with no section is NaN.
    """
    pts, tris, _ = surface_mesh(surface)
    rows = []
    for y in stations:
        loop = section(pts, tris, y, span_axis, lift_axis)
        if loop is None or len(loop) < 4:
            rows.append((np.nan, np.nan, np.nan))
            continue
        le, te = loop[np.argmin(loop[:, 0])], loop[np.argmax(loop[:, 0])]
        d = te - le
        chord, twist = float(np.hypot(*d)), float(np.arctan2(-d[1], d[0]))
        c, s = np.cos(twist), np.sin(twist)
        local = np.column_stack([c * (loop[:, 0] - le[0]) - s * (loop[:, 1] - le[1]),
                                 s * (loop[:, 0] - le[0]) + c * (loop[:, 1] - le[1])])
        thickness = _vertical_extent(local, np.linspace(0.0, chord, 201)[1:-1]).max()
        rows.append((chord, np.degrees(twist), thickness / chord))
    chord, twist, t_c = np.array(rows).T
    return {'y': np.asarray(stations, dtype=float), 'chord': chord, 'twist_deg': twist,
            't_c': t_c}


def _show_absolute(ax, span=0.1):
    """Plain tick labels, and at least ``+-span`` (relative) of room around flat data."""
    ax.ticklabel_format(useOffset=False)
    lo, hi = ax.get_ylim()
    mid = 0.5 * (lo + hi)
    if hi - lo < 2.0 * span * abs(mid):
        ax.set_ylim(mid * (1.0 - span), mid * (1.0 + span))


#: Panels of :func:`section_geometry` in drawing order: key -> (record key, y label).
GEOMETRY_QUANTITIES = {
    'twist': ('twist_deg', 'twist relative to first entry (deg, nose up +)'),
    'chord': ('chord', 'chord'),
    'thickness': ('t_c', 'thickness / chord'),
}


def section_geometry(fig, entries, colors=None, styles=None,
                     quantities=('twist', 'chord', 'thickness')):
    """Twist, chord and thickness-to-chord against span, N entries overlaid.

    ``entries`` are ``[(label, section_record)]`` on one set of stations.
    ``quantities`` picks the panels (keys of ``GEOMETRY_QUANTITIES``), drawn in
    that order. The twist panel is relative to the first entry's twist at the
    same station (the first entry draws as zero); chord and t/c are absolute.
    """
    unknown = [q for q in quantities if q not in GEOMETRY_QUANTITIES]
    if unknown or not quantities:
        raise ValueError(f'section_geometry quantities {list(quantities)} must be a non-empty '
                         f'subset of {sorted(GEOMETRY_QUANTITIES)}')
    colors, styles = resolve_styles(len(entries), colors, styles)
    axes = fig.subplots(len(quantities), 1, sharex=True, squeeze=False)[:, 0]
    first = entries[0][1]
    for i, (label, rec) in enumerate(entries):
        kw = {'color': colors[i], 'ls': styles[i], 'lw': 1.4, 'marker': 'o', 'ms': 2.5,
              'mew': 0}
        for ax, q in zip(axes, quantities):
            key = GEOMETRY_QUANTITIES[q][0]
            y = rec[key] - first[key] if q == 'twist' else rec[key]
            ax.plot(rec['y'], y, label=label if ax is axes[0] else None, **kw)
    for ax, q in zip(axes, quantities):
        style_axes(ax, ylabel=GEOMETRY_QUANTITIES[q][1])
        if q == 'twist':
            ax.axhline(0.0, color='black', lw=0.8)
        else:
            _show_absolute(ax)
    axes[0].legend(fontsize=7, framealpha=0.9)
    axes[-1].set_xlabel('span coordinate', fontsize=8, color=TEXT)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
