"""Airfoil intake: read, generate, repanel, report, and write 2D sections.

A section is held as an ``(n, 2)`` Selig loop without a closing repeat: TE
(upper) -> upper surface -> LE -> lower surface -> TE (lower). For a sharp TE
the first and last rows coincide; for an open TE they differ by the TE gap.
:func:`write_selig` adds the repeated TE row that ``prefoil`` needs on an open
TE. Every airfoil datum comes from ``prefoil`` or from a closed-form
equation; there is no XFOIL.

No function names a case, a path, or a marker by default.
"""

import os

import numpy as np

#: The x**4 coefficient of the NACA 4-digit thickness form. ``open`` is the
#: standard section (TE gap 0.00252 t/0.12); ``closed`` gives a zero-thickness
#: TE (ADODG paper Eq. 8, p. 7).
NACA4_A4 = {'open': -0.1015, 'closed': -0.1036}


def bigeometric_spacing(start, end, n, a1, b1, ra, rb):
    """``n`` parametric stations on ``[start, end]``: geometric growth ``ra`` from the
    start (first panel ``a1``), geometric growth ``rb`` from the end (last panel
    ``b1``), and a constant middle run; ``a1``, ``b1`` are fractions of the interval.

    The same distribution as ``prefoil.sampling.bigeometric``. The run lengths
    ``na`` (start side) and ``nb`` (end side) are the integers that make the middle
    panel closest to the last start-side panel; prefoil's own search brackets
    ``na`` up to ``0.49 n``, where the runs overlap, and fails for a few hundred
    points per side.
    """
    best = None
    for na in range(1, n - 3):
        a_na = a1 * ra ** na
        nb = int(round(np.log(a_na / b1) / np.log(rb)))
        nc = n - (2 + na + nb)
        if nb < 1 or nc < 1:
            continue
        da = a1 * (1.0 - ra ** na) / (1.0 - ra)
        db = b1 * (1.0 - rb ** nb) / (1.0 - rb)
        dc = (1.0 - db) - da
        if dc <= 0.0:
            continue
        score = abs(dc / (nc + 1) / a_na - 1.0)
        if best is None or score < best[0]:
            best = (score, na, nb, nc, dc / (nc + 1))
    if best is None:
        raise ValueError(f'bigeometric: {n} points cannot hold runs of ratio {ra} from '
                         f'spacings {a1:.3g} and {b1:.3g} (fractions of the side); '
                         f'lower the spacings or the ratio, or raise n_points')
    _, na, nb, nc, deltac = best
    u = np.empty(n)
    u[0], u[-1] = 0.0, 1.0
    for i in range(1, n - 1):
        if i <= na:
            u[i] = u[i - 1] + a1 * ra ** (i - 1)
        elif i <= na + nc:
            u[i] = u[i - 1] + deltac
        else:
            u[i] = 1.0 - b1 * (1.0 - rb ** (n - i - 1)) / (1.0 - rb)
    return u * (end - start) + start


def _spacing_funcs():
    from prefoil import sampling
    return {'cosine': sampling.cosine, 'conical': sampling.conical,
            'polynomial': sampling.polynomial, 'bigeometric': sampling.bigeometric}


SPACINGS = ('cosine', 'conical', 'polynomial', 'bigeometric')
TE_FORMS = ('keep', 'sharp', 'blunt', 'round')


# ------------------------------------------------------------------ loops

def read_dat(path):
    """``(n, 2)`` loop from a two-column ``.dat``/``.csv``; non-numeric lines are skipped."""
    rows = []
    with open(path) as fh:
        for line in fh:
            parts = line.replace(',', ' ').split()
            if len(parts) < 2:
                continue
            try:
                rows.append((float(parts[0]), float(parts[1])))
            except ValueError:
                continue                      # a header line
    if len(rows) < 5:
        raise ValueError(f'{path}: fewer than 5 coordinate rows')
    c = np.array(rows)
    # A written Selig file repeats the TE row; the loop does not (module doc).
    if np.array_equal(c[-1], c[0]) and not np.array_equal(c[-2], c[0]):
        lower_te_gap = abs(c[-2, 1] - c[0, 1])
        if lower_te_gap > 0.0 and np.isclose(c[-2, 0], c[0, 0], rtol=0, atol=1e-9):
            c = c[:-1]                        # open TE: drop the repeated endpoint
    return c


def naca4(code, n, spacing, te):
    """NACA 4-digit section, ``n`` points per surface (LE and TE included).

    ``spacing``: ``cosine`` (x = (1 - cos t)/2) or ``uniform``. ``te``: ``open``
    or ``closed`` (:data:`NACA4_A4`). Returns the Selig loop (2n - 1 rows).
    """
    code = str(code)
    if len(code) != 4 or not code.isdigit():
        raise ValueError(f'naca4 code {code!r} must be four digits')
    if te not in NACA4_A4:
        raise ValueError(f'naca4 te={te!r} must be one of {sorted(NACA4_A4)}')
    if n < 3:
        raise ValueError('naca4 needs n >= 3 points per surface')
    m, p, t = int(code[0]) / 100.0, int(code[1]) / 10.0, int(code[2:]) / 100.0
    if spacing == 'cosine':
        x = (1.0 - np.cos(np.linspace(0.0, np.pi, n))) / 2.0
    elif spacing == 'uniform':
        x = np.linspace(0.0, 1.0, n)
    else:
        raise ValueError(f'naca4 spacing={spacing!r} must be cosine or uniform')
    yt = 5.0 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x ** 2
                    + 0.2843 * x ** 3 + NACA4_A4[te] * x ** 4)
    if te == 'closed':
        yt[-1] = 0.0          # the coefficients sum to 0; round-off leaves ~1e-17
    if m == 0.0 or p == 0.0:
        yc = np.zeros_like(x)
        theta = np.zeros_like(x)
    else:
        fwd = x < p
        yc = np.where(fwd, m / p ** 2 * (2 * p * x - x ** 2),
                      m / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * x - x ** 2))
        dyc = np.where(fwd, 2 * m / p ** 2 * (p - x), 2 * m / (1 - p) ** 2 * (p - x))
        theta = np.arctan(dyc)
    xu, yu = x - yt * np.sin(theta), yc + yt * np.cos(theta)
    xl, yl = x + yt * np.sin(theta), yc - yt * np.cos(theta)
    upper = np.c_[xu, yu][::-1]
    lower = np.c_[xl, yl][1:]
    return np.vstack([upper, lower])


def selig_from_mesh(mesh_path, markers):
    """Wall nodes of one or more 2D line markers as one Selig loop.

    Several markers (for example an upper and a lower surface) are joined at
    the nodes they share. The walk starts at the largest-x node and goes to the
    upper surface first (``mesh/su2_mesh_file.get_loop_order``).
    """
    from ..mesh.su2_mesh_file import (collect_line_elements, extract_baseline_surface,
                                      get_loop_order, parse_su2_mesh)
    markers = [markers] if isinstance(markers, str) else list(markers)
    if not markers:
        raise ValueError('selig_from_mesh needs at least one marker')
    coords = extract_baseline_surface(mesh_path, markers)
    _, loop = get_loop_order(mesh_path, markers)
    n_segments = len(collect_line_elements(parse_su2_mesh(mesh_path)[2], markers))
    if len(loop) != len(coords) or n_segments != len(coords):
        raise ValueError(
            f'markers {markers} in {mesh_path}: the loop walk visited {len(loop)} '
            f'of {len(coords)} nodes -- the wall is not one closed loop')
    section = np.asarray(coords)[loop]

    # DVGeometryCST expects the closed loop orientation used by its internal
    # upper/lower split.  Mesh marker connectivity is allowed to use either
    # orientation, so normalize the walk before writing a Selig file.  Without
    # this, a valid clockwise mesh loop can make one CST slice empty.
    if len(section) >= 3:
        edge_prev = section[0] - section[-1]
        edge_next = section[1] - section[0]
        if np.cross(np.r_[edge_prev, 0.0], np.r_[edge_next, 0.0])[2] < 0.0:
            section = section[::-1]
        # Reversal changes the starting node.  Selig/DVGeometryCST expects the
        # walk to start at the trailing edge; choose the upper TE node on a
        # blunt edge so the split is deterministic for both orientations.
        x_max = np.max(section[:, 0])
        te = np.flatnonzero(np.isclose(section[:, 0], x_max, rtol=0.0, atol=1e-14))
        start = te[np.argmax(section[te, 1])]
        section = np.roll(section, -int(start), axis=0)
    return section


def read_surface(spec, base_dir=None):
    """A section from a source spec.

    - ``naca4:<code>:<open|closed>[:<n per surface>]`` -- closed form, cosine
      spacing; ``n`` defaults to nothing, so it must be given (2001 is enough to
      repanel from).
    - ``mesh:<path>:<marker>[,<marker>...]`` -- the wall of an SU2 mesh.
    - any other string -- a two-column ``.dat``/``.csv`` path.

    Relative paths resolve against ``base_dir`` when given.
    """
    def _path(p):
        return p if os.path.isabs(p) or base_dir is None else os.path.join(base_dir, p)
    if spec.startswith('naca4:'):
        parts = spec.split(':')
        if len(parts) != 4:
            raise ValueError(f'source {spec!r}: write naca4:<code>:<open|closed>:<n>')
        return naca4(parts[1], int(parts[3]), 'cosine', parts[2])
    if spec.startswith('mesh:'):
        _, rest = spec.split(':', 1)
        path, _, names = rest.rpartition(':')
        if not path or not names:
            raise ValueError(f'source {spec!r}: write mesh:<path>:<marker>[,<marker>]')
        return selig_from_mesh(_path(path), [m for m in names.split(',') if m])
    return read_dat(_path(spec))


# ---------------------------------------------------------------- prefoil

def _airfoil(coords):
    from prefoil import Airfoil
    return Airfoil(np.asarray(coords, dtype=float))


def parse_te(te):
    """``keep`` | ``sharp:<x_cut>`` | ``blunt:<x_cut>`` | ``round:<x_cut>`` -> (form, value).

    ``x_cut`` is ``prefoil``'s own parameter: the chord station where the TE edit
    starts. No value is assumed.
    """
    form, _, value = str(te).partition(':')
    if form not in TE_FORMS:
        raise ValueError(f'te={te!r}: form must be one of {list(TE_FORMS)}')
    if form == 'keep':
        if value:
            raise ValueError(f'te={te!r}: keep takes no value')
        return form, None
    if not value:
        raise ValueError(f'te={te!r}: {form} needs a value ({form}:<number>); '
                         f'none is assumed')
    return form, float(value)


def _with_te(coords, te):
    """A ``prefoil.Airfoil`` of ``coords`` with the TE edit applied."""
    form, value = parse_te(te)
    af = _airfoil(coords)
    edit = {'sharp': af.sharpenTE, 'round': af.roundTE, 'blunt': af.makeBluntTE}.get(form)
    if edit is not None:
        try:
            edit(xCut=value)
        except RuntimeError as exc:
            raise ValueError(f'te={te!r}: prefoil could not apply it ({exc}); a '
                             f'section whose TE is already closed takes te = "keep" '
                             f'or "blunt:<x_cut>"') from None
    return af


def bigeometric_func_args(af, le_spacing, te_spacing, spacing_ratio):
    """Per-side ``func_args`` for ``prefoil.sampling.bigeometric`` from chord-unit spacings.

    ``bigeometric`` takes ``a1``/``b1`` as fractions of the side's parametric
    interval, and the side runs over ``[0, s_LE]`` (TE to LE) or ``[s_LE, 1]``
    (LE to TE). A chord-unit spacing becomes ``spacing / (|dX/ds| * interval)``,
    with the spline speed ``|dX/ds|`` taken at the end the spacing belongs to,
    so the first panel of each geometric run has the stated length.
    """
    s_le = float(af.s_LE)
    speed = lambda s: float(np.linalg.norm(af.spline.getDerivative(s)))
    le_u = le_spacing / (speed(s_le))
    te_u = te_spacing / (speed(0.0))
    te_u_end = te_spacing / (speed(1.0))
    ratio = float(spacing_ratio)
    return [dict(a1=te_u / s_le, b1=le_u / s_le, ra=ratio, rb=ratio),
            dict(a1=le_u / (1.0 - s_le), b1=te_u_end / (1.0 - s_le), ra=ratio, rb=ratio)]


def repanel(coords, n, spacing, te, le_spacing=None, te_spacing=None, spacing_ratio=None):
    """Resample a section to ``n`` points with a ``prefoil`` spacing and TE edit.

    ``le_spacing``, ``te_spacing`` (chord units) and ``spacing_ratio`` set the
    end spacings and growth of ``bigeometric``; they are stated together and
    belong to no other spacing, which has no such parameters.
    """
    funcs = _spacing_funcs()
    if spacing not in funcs:
        raise ValueError(f'spacing={spacing!r} must be one of {list(funcs)}')
    given = [v is not None for v in (le_spacing, te_spacing, spacing_ratio)]
    if any(given) and not all(given):
        raise ValueError('le_spacing, te_spacing and spacing_ratio are stated together')
    if any(given) and spacing != 'bigeometric':
        raise ValueError(f'le_spacing/te_spacing/spacing_ratio apply to spacing = '
                         f'"bigeometric" only; spacing={spacing!r} has no end-spacing '
                         f'parameters')
    af = _with_te(coords, te)
    func_args = (bigeometric_func_args(af, le_spacing, te_spacing, spacing_ratio)
                 if all(given) else None)
    pts = af.getSampledPts(
        int(n), spacingFunc=bigeometric_spacing if all(given) else funcs[spacing],
        func_args=func_args)
    if not af.closedCurve:
        pts = pts[:-1]                    # prefoil appends the closing repeat
    return np.asarray(pts)


def geometry_report(coords):
    """``prefoil`` geometry data for a section, as plain floats."""
    af = _airfoil(coords)
    xa, ta = af.getMaxThickness('american')
    xb, tb = af.getMaxThickness('british')
    xc, cm = af.getMaxCamber()
    le = int(np.argmin(np.asarray(coords)[:, 0]))
    return {
        'n_points': int(len(coords)),
        'chord': float(af.getChord()),
        'le': [float(v) for v in np.asarray(coords)[le]],
        'max_thickness': {'x': float(xa), 't': float(ta), 'type': 'american'},
        'max_thickness_british': {'x': float(xb), 't': float(tb)},
        'max_camber': {'x': float(xc), 'c': float(cm)},
        'le_radius': float(af.getLERadius()),
        'te_angle_deg': float(af.getTEAngle()),
        'te_thickness': float(af.getTEThickness()),
        'prefoil_version': _prefoil_version(),
    }


def _prefoil_version():
    import prefoil
    return getattr(prefoil, '__version__', 'unknown')


def polygon_area(coords):
    """Enclosed area of the closed polygon through ``coords`` (shoelace)."""
    x, y = np.asarray(coords, dtype=float).T
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


# ------------------------------------------------------------------ write

def selig_rows(coords):
    """The rows a Selig file holds: the loop, plus the TE row again on an open TE."""
    c = np.asarray(coords, dtype=float)
    if np.array_equal(c[-1], c[0]):
        return c                          # sharp TE: the lower surface ends there
    return np.vstack([c, c[0]])


def format_selig(coords):
    """One ``x y`` pair per line, 12 decimals, no header (``DVGeoCST.readCoordFile``)."""
    return ''.join(f'{x:.12f}  {y:.12f}\n' for x, y in selig_rows(coords))


def write_selig(path, coords):
    """Write the Selig file; returns the number of rows written."""
    rows = selig_rows(coords)
    with open(path, 'w') as fh:
        fh.write(format_selig(coords))
    return len(rows)
