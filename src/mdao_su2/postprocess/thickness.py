#!/usr/bin/env python
"""Measure baseline max thickness at spanwise stations from an SU2 mesh."""

import argparse
import json
import sys

import numpy as np

from mdao_su2.mesh.su2_mesh_file import read_wall_surface
from mdao_su2.postprocess.span_load import section_cut


def max_thickness(section, n_probe=2001):
    """Max thickness via SU2's Compute_MaxThickness. Returns (thickness, x_at_max, chord, aoa_deg)."""
    x, z = section[:, 0], section[:, 1]

    # TE = largest x; LE = farthest from it (matches SU2's ordered-list convention).
    i_max = int(np.argmax(x))
    d = np.hypot(x - x[i_max], z - z[i_max])
    i_le = int(np.argmax(d))
    xte, zte = x[i_max], z[i_max]
    xle, zle = x[i_le], z[i_le]
    chord = float(np.hypot(xte - xle, zte - zle))
    aoa = np.arctan2(zle - zte, xte - xle)          # SU2's sign convention

    # Translate to the trailing edge, then rotate the section flat.
    xs, zs = x - xte, z - zte
    c, s = np.cos(aoa), np.sin(aoa)
    xr = xs * c - zs * s
    zr = zs * c + xs * s

    # Upper/lower split in the de-rotated frame; the chord line is z=0 there.
    up, lo = zr >= 0.0, zr <= 0.0
    if up.sum() < 2 or lo.sum() < 2:
        raise ValueError('section did not split into two sides')

    def interp(mask):
        o = np.argsort(xr[mask])
        xu, zu = xr[mask][o], zr[mask][o]
        xu, idx = np.unique(xu, return_index=True)
        return xu, zu[idx]

    xu, zu = interp(up)
    xl, zl = interp(lo)
    lo_x = max(xu.min(), xl.min())
    hi_x = min(xu.max(), xl.max())
    xp = np.linspace(lo_x, hi_x, n_probe)
    t = np.interp(xp, xu, zu) - np.interp(xp, xl, zl)
    k = int(np.argmax(t))

    # Back to the original frame so the value is usable as a probe station.
    xb = xp[k] * c + 0.0 * s + xte
    return float(t[k]), float(xb), chord, float(np.degrees(aoa))


def main(argv=None):
    ap = argparse.ArgumentParser(description='Measure baseline max thickness at spanwise stations.',
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('mesh')
    ap.add_argument('--wall', action='append', required=True,
                    help='wall marker (repeatable)')
    ap.add_argument('--station', action='append', type=float, required=True,
                    help='absolute spanwise (y) station (repeatable)')
    ap.add_argument('--bound', action='append', type=float, default=None,
                    help='absolute minimum thickness for the matching station')
    ap.add_argument('--json', help='write the table here')
    args = ap.parse_args(argv)

    bounds = args.bound or []
    if bounds and len(bounds) != len(args.station):
        ap.error('--bound must be given once per --station, or not at all')

    points, faces = read_wall_surface(args.mesh, args.wall)
    print(f'{len(points)} points, {len(faces)} wall faces on '
          f'{", ".join(args.wall)}\n')

    rows = []
    print(f'{"station y":>10} {"t_max":>10} {"x@t_max":>10} {"chord":>9} '
          f'{"t/c":>8} {"incid":>7}   bound     margin')
    ok = True
    for k, y in enumerate(args.station):
        t, xb, chord, aoa = max_thickness(section_cut(points, faces, y))
        row = {'station': y, 'thickness': t, 'x_at_max': xb,
               'chord': chord, 'toc': t / chord, 'incidence_deg': aoa}
        line = (f'{y:10.4f} {t:10.6f} {xb:10.6f} {chord:9.6f} '
                f'{t / chord:8.5f} {aoa:7.3f}')
        if bounds:
            b = bounds[k]
            row['bound'] = b
            row['margin_pct'] = 100.0 * (t - b) / b
            row['feasible'] = t >= b
            ok &= row['feasible']
            line += f'   {b:.4f}  {row["margin_pct"]:+7.2f}%'
        print(line)
        rows.append(row)

    if bounds:
        print('\nBaseline is ' + ('FEASIBLE' if ok else 'INFEASIBLE')
              + ' against every bound.')
    if args.json:
        with open(args.json, 'w') as fh:
            json.dump({'mesh': args.mesh, 'walls': args.wall, 'stations': rows},
                      fh, indent=2)
        print(f'wrote {args.json}')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
