"""Measure a generated mesh, then decide whether the solver may use it."""

import numpy as np

from .su2_mesh_file import parse_su2_mesh

__all__ = ['measure', 'check', 'enforce', 'report_lines', 'json_safe',
           'TRI', 'QUAD', 'MeshQualityError']

# SU2 VTK-style element type codes.
TRI = 5
QUAD = 9


class MeshQualityError(RuntimeError):
    """A generated mesh failed its gates.  Converted to om.AnalysisError by"""


# ------------------------------------------------------------ measurement

def _wall_nodes_and_edges(markers, wall_marker):
    edges = [(e[1], e[2]) for e in markers.get(wall_marker, ())]
    nodes = {n for e in edges for n in e}
    return nodes, edges


def _edge_to_quads(elements):
    """``frozenset(edge) -> [quad, ...]``, for walking outward from the wall."""
    index = {}
    for e in elements:
        if e[0] != QUAD:
            continue
        nodes = e[1:5]
        for i in range(4):
            key = frozenset((nodes[i], nodes[(i + 1) % 4]))
            index.setdefault(key, []).append(nodes)
    return index


def _walk_column(edge, edge_quads, max_layers=500):
    """Layer heights for one wall segment, from the wall outward."""
    heights = []
    current = edge
    seen = set()
    for _ in range(max_layers):
        key = frozenset(current)
        candidates = [q for q in edge_quads.get(key, ())
                      if tuple(sorted(q)) not in seen]
        if not candidates:
            break
        quad = candidates[0]
        seen.add(tuple(sorted(quad)))
        # The opposite edge of a quad (n0,n1,n2,n3) across edge (n0,n1) is
        # (n2,n3); find where `current` sits in the cyclic order first.
        try:
            i = next(k for k in range(4)
                     if {quad[k], quad[(k + 1) % 4]} == set(current))
        except StopIteration:                       # pragma: no cover
            break
        outer = (quad[(k := (i + 2) % 4)], quad[(k + 1) % 4])
        heights.append((current, outer))
        current = outer
    return heights


def _midpoint_distances(columns, coords):
    """(n_columns, ) depths and a ragged list of per-layer heights."""
    per_column = []
    for pairs in columns:
        hs = []
        for inner, outer in pairs:
            a = 0.5 * (coords[inner[0]] + coords[inner[1]])
            b = 0.5 * (coords[outer[0]] + coords[outer[1]])
            hs.append(float(np.linalg.norm(b - a)))
        per_column.append(hs)
    return per_column


def _equiangle_skew(poly):
    """Equiangle skewness in [0, 1]: 0 is a perfect element, 1 is degenerate."""
    n = len(poly)
    ideal = 60.0 if n == 3 else 90.0
    angles = []
    for i in range(n):
        a, b, c = poly[i - 1], poly[i], poly[(i + 1) % n]
        v1, v2 = a - b, c - b
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 == 0.0 or n2 == 0.0:
            return 1.0
        cos = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
        angles.append(np.degrees(np.arccos(cos)))
    hi, lo = max(angles), min(angles)
    return float(max((hi - ideal) / (180.0 - ideal), (ideal - lo) / ideal))


def measure(mesh_path, wall_marker='AIRFOIL'):
    """Everything ``check`` needs, plus what a human wants in a report."""
    points, elements, markers = parse_su2_mesh(mesh_path)
    n = len(points)
    coords = np.array([points[i] for i in range(n)])

    wall_nodes, wall_edges = _wall_nodes_and_edges(markers, wall_marker)
    n_tri = sum(1 for e in elements if e[0] == TRI)
    n_quad = sum(1 for e in elements if e[0] == QUAD)

    out = {
        'mesh_path': mesh_path,
        'n_nodes': n,
        'n_tri': n_tri,
        'n_quad': n_quad,
        'n_elements': n_tri + n_quad,
        'markers': sorted(markers),
        'n_wall_nodes': len(wall_nodes),
        'n_wall_elements': len(wall_edges),
    }
    if not wall_edges:
        out['error'] = f'marker {wall_marker!r} is missing or empty'
        return out

    edge_quads = _edge_to_quads(elements)
    columns = [_walk_column(e, edge_quads) for e in wall_edges]
    per_column = _midpoint_distances(columns, coords)

    depths = np.array([len(hs) for hs in per_column])
    first = np.array([hs[0] for hs in per_column if hs])

    out.update({
        # Per-column, so a BL that collapsed over part of the surface is visible.
        # min is the number that matters: one bare wall segment is enough to
        'bl_layers_min': int(depths.min()),
        'bl_layers_max': int(depths.max()),
        'bl_layers_median': int(np.median(depths)),
        'n_columns_short_of_max': int((depths < depths.max()).sum()),
    })
    if first.size:
        out.update({
            'first_cell_height_min': float(first.min()),
            'first_cell_height_max': float(first.max()),
            'first_cell_height_median': float(np.median(first)),
        })

    # Growth ratio, measured: median over every consecutive layer pair in every
    # column.  The configured value is what gmsh was asked for; this is what the
    ratios = [b / a for hs in per_column for a, b in zip(hs, hs[1:]) if a > 0.0]
    if ratios:
        out['growth_ratio_median'] = float(np.median(ratios))
        out['growth_ratio_max'] = float(np.max(ratios))
    if per_column:
        out['bl_thickness_median'] = float(np.median(
            [sum(hs) for hs in per_column if hs] or [0.0]))

    # Skewness and degeneracy over every element.  A negative/zero area is not a
    # quality complaint, it is an invalid mesh.
    skews, areas, quad_skews, ratios_ar = [], [], [], []
    for e in elements:
        poly = coords[list(e[1:4])] if e[0] == TRI else coords[list(e[1:5])]
        areas.append(0.5 * abs(np.dot(poly[:, 0], np.roll(poly[:, 1], -1))
                               - np.dot(poly[:, 1], np.roll(poly[:, 0], -1))))
        s = _equiangle_skew(poly)
        skews.append(s)
        if e[0] == QUAD:
            quad_skews.append(s)
        # Edge-length aspect ratio.  Measured because it is what stalls a solve:
        # the local time step is set by an element's SMALLEST dimension, so a
        edges = [np.linalg.norm(poly[(k + 1) % len(poly)] - poly[k])
                 for k in range(len(poly))]
        lo = min(edges)
        if lo > 0.0:
            ratios_ar.append(max(edges) / lo)
    skews = np.asarray(skews)
    areas = np.asarray(areas)
    aspect = np.asarray(ratios_ar) if ratios_ar else np.zeros(1)
    worst = int(np.argmax(skews))
    e = elements[worst]
    centroid = coords[list(e[1:4] if e[0] == TRI else e[1:5])].mean(axis=0)

    out.update({
        'max_skewness': float(skews.max()),
        'max_skewness_quads': float(max(quad_skews)) if quad_skews else 0.0,
        'max_skewness_at': [float(centroid[0]), float(centroid[1])],
        'skew_p99': float(np.percentile(skews, 99)),
        'skew_p999': float(np.percentile(skews, 99.9)),
        'aspect_ratio_median': float(np.median(aspect)),
        'aspect_ratio_p99': float(np.percentile(aspect, 99)),
        'aspect_ratio_max': float(aspect.max()),
        'min_area': float(areas.min()),
        'degenerate_cells': int((areas <= 1e-20).sum()),
        '_skews': skews,          # for check(); dropped before JSON
    })
    return out


# ----------------------------------------------------------------- gates

def check(measured, remesh, wall_marker='AIRFOIL'):
    """``[(name, ok, detail), ...]`` -- every gate, passing ones included."""
    q = remesh.quality
    # The RESOLVED triple, not the raw config fields: if bl_thickness is in
    # charge, n_bl_layers was derived, and gating against the un-derived field
    want_h, want_r, want_n, want_t = remesh.resolved_bl()
    results = []

    def add(name, ok, detail):
        results.append((name, bool(ok), detail))

    if 'error' in measured:
        add('mesh_structure', False, measured['error'])
        return results
    add('mesh_structure', True,
        f"{measured['n_nodes']} nodes, {measured['n_tri']} tri + "
        f"{measured['n_quad']} quad, markers {measured['markers']}")

    add('no_degenerate_cells', measured['degenerate_cells'] == 0,
        f"{measured['degenerate_cells']} cells with area <= 1e-20 "
        f"(min area {measured['min_area']:.3e})")

    # The layer count is REPORTED against the request, and gated only on being
    # UNIFORM.  Two deliberate choices:
    got_min, got_max = measured['bl_layers_min'], measured['bl_layers_max']
    short = measured['n_columns_short_of_max']
    add('bl_layers_uniform', got_min == got_max,
        f'requested {want_n}, measured min {got_min} / median '
        f"{measured['bl_layers_median']} / max {got_max}"
        + (f', {short} of '
           f"{measured['n_wall_elements']} columns short of the deepest"
           if short else ' (every column the same depth)'))

    # y+ lives or dies on the first cell height, so it gets a band rather than
    # an equality: gmsh's extrusion does not land on the request exactly.
    lo, hi = q.min_first_cell_height_ratio, q.max_first_cell_height_ratio
    h_min = measured.get('first_cell_height_min')
    h_max = measured.get('first_cell_height_max')
    if h_min is None:
        add('first_cell_height', False, 'no boundary layer was extruded at all')
    else:
        add('first_cell_height', lo * want_h <= h_min and h_max <= hi * want_h,
            f'requested {want_h:.3e}, measured {h_min:.3e}..{h_max:.3e} '
            f'(ratio {h_min / want_h:.2f}..{h_max / want_h:.2f}, '
            f'allowed {lo:g}..{hi:g})')

    # Depth, gated separately from the first cell: a BL deep enough with the
    # wrong first cell (or the reverse) fails for a different physical reason --
    t_lo, t_hi = q.min_bl_thickness_ratio, q.max_bl_thickness_ratio
    got_t = measured.get('bl_thickness_median')
    if got_t is None:
        add('bl_thickness', False, 'no boundary layer was extruded at all')
    else:
        add('bl_thickness', t_lo * want_t <= got_t <= t_hi * want_t,
            f'requested {want_t:.4e}, measured {got_t:.4e} '
            f'(ratio {got_t / want_t:.2f}, allowed {t_lo:g}..{t_hi:g})')

    skews = measured.get('_skews')
    n_over = int((skews > q.max_skewness).sum()) if skews is not None else None
    where = measured['max_skewness_at']
    add('skewness', measured['max_skewness'] <= q.max_skewness,
        f"max {measured['max_skewness']:.3f} at "
        f'(x={where[0]:.4f}, y={where[1]:.4f}), p99 '
        f"{measured['skew_p99']:.3f}, allowed {q.max_skewness:g}"
        + (f"; {n_over} of {measured['n_elements']} elements over"
           if n_over else ''))

    add('aspect_ratio', measured['aspect_ratio_p99'] <= q.max_aspect_ratio_p99,
        f"p99 {measured['aspect_ratio_p99']:.0f}, max "
        f"{measured['aspect_ratio_max']:.0f}, median "
        f"{measured['aspect_ratio_median']:.1f}; p99 allowed "
        f'{q.max_aspect_ratio_p99:g}')

    if 'growth_ratio_median' in measured:
        # Reported, NOT gated: the growth ratio is an input to the BL thickness,
        # and the thing that has to hold is the first height and the layer count.
        add('growth_ratio (reported)', True,
            f'requested {want_r:g}, measured median '
            f"{measured['growth_ratio_median']:.4f} / max "
            f"{measured['growth_ratio_max']:.4f}")
    return results


def json_safe(measured):
    """The measurement without the raw per-element array, for a results file."""
    return {k: v for k, v in measured.items() if not k.startswith('_')}


def enforce(mesh_path, remesh, wall_marker='AIRFOIL', as_analysis_error=True):
    """Measure, gate, and raise on failure.  Returns the measurement."""
    measured = measure(mesh_path, wall_marker)
    results = check(measured, remesh, wall_marker)
    failures = [f'{name}: {detail}' for name, ok, detail in results if not ok]
    if failures:
        message = ('generated mesh failed its [remesh.quality] gates:\n  '
                   + '\n  '.join(failures)
                   + f'\n  mesh: {mesh_path}')
        if as_analysis_error:
            import openmdao.api as om
            raise om.AnalysisError(message)
        raise MeshQualityError(message)
    return measured


def report_lines(measured, results):
    """Human-readable mesh-quality verdict block."""
    lines = [f"=== mesh quality: {measured['mesh_path']} ==="]
    for name, ok, detail in results:
        lines.append(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    failed = sum(1 for _, ok, _ in results if not ok)
    lines.append(f"  VERDICT: {'PASS' if not failed else f'FAIL ({failed})'}")
    return lines
