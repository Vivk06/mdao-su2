"""Trailing-edge twist DVs on a wing FFD.

A pyGeo reference axis runs spanwise (FFD index j) through each section's
control points at ``axis_x_fraction`` of the box chord and at the box
mid-height; with the Case 3 box that is the wing TE line (x = 1, z = 0). The
``twist`` global DV sets ``rot_y`` at each axis point, so each FFD section
turns rigidly about its TE point. Between sections the FFD spline (pyGeo's
default order) blends the rotations.

Units and sign: degrees, positive nose-up (LE toward +z). pyGeo's ``rot_y`` is
already nose-up positive for flow along +x and span along +y
(``tests/regression.py`` checks it). DV order: root to tip, the root
first when ``include_root`` is on.
"""

import numpy as np

AXIS = 'twist_axis'
DV_NAME = 'twist'
#: rot_y coefficient per degree of nose-up twist (checked by the regression suite).
SIGN = 1.0


def twist_function(n_axis, include_root):
    """The pyGeo global-DV callback: section ``i`` gets ``val[i - first]`` degrees."""
    first = 0 if include_root else 1

    def twist(val, geo):
        for i in range(first, n_axis):
            geo.rot_y[AXIS].coef[i] = SIGN * val[i - first]
    return twist


def add_twist_axis(dvgeo, twist_cfg):
    """Add the TE reference axis to a ``DVGeometry``; returns its point count."""
    n_axis = dvgeo.addRefAxis(AXIS, xFraction=twist_cfg.axis_x_fraction, alignIndex='j')
    if n_axis != twist_cfg.n:
        from ..run_config import ConfigError
        raise ConfigError(f'[dv.twist] n = {twist_cfg.n}, but the FFD has {n_axis} '
                          f'spanwise sections')
    return n_axis


def add_twist_dvs(geometry, twist_cfg):
    """Register the twist DV on an ``OM_DVGEOCOMP``; returns the DV count."""
    n_axis = add_twist_axis(geometry.nom_getDVGeo(), twist_cfg)
    n_dv = twist_cfg.n_dv
    geometry.nom_addGlobalDV(DV_NAME, value=np.full(n_dv, twist_cfg.init_deg),
                             func=twist_function(n_axis, twist_cfg.include_root))
    return n_dv


def add_twist_dvs_dvgeo(dvgeo, twist_cfg):
    """The same on a bare ``DVGeometry`` (tests, dvmap); returns the DV count."""
    n_axis = add_twist_axis(dvgeo, twist_cfg)
    dvgeo.addGlobalDV(DV_NAME, np.full(twist_cfg.n_dv, twist_cfg.init_deg),
                      twist_function(n_axis, twist_cfg.include_root))
    return twist_cfg.n_dv


def station_angles(points, n_stations):
    """Nose-up angle (deg) of the LE->TE chord of each station.

    ``points`` holds, per station, an LE point then a TE point (rows 2s, 2s+1).
    """
    p = np.asarray(points, dtype=float).reshape(n_stations, 2, 3)
    le, te = p[:, 0], p[:, 1]
    return np.degrees(np.arctan2(le[:, 2] - te[:, 2], te[:, 0] - le[:, 0]))
