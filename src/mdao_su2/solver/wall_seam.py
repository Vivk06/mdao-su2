"""The wall seam: marker lookup, reference surface, displacement scatter, per-node
sensitivity, rank reduction, and alpha projection.

Pure functions over a pysu2/pysu2ad driver and a ``gid -> x_aero position`` map, so
stub drivers can test them and the remesh path can pass its own new-mesh map.
``seam_comp.py`` delegates here. The frozen
``aero_comp.py`` keeps its own copy, and ``bidir_comp.py`` inherits that copy.

x_aero layout: wall nodes sorted by mesh-global index, coordinates interleaved
``[x0, y0, (z0), x1, ...]``, identical on every rank and independent of partitioning.
"""

import math

import numpy as np

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

#: SENS_AOA is per radian; alpha is carried in degrees.
DEG = math.pi / 180.0


def local_markers(driver, names):
    """``[(marker_id, n_vertex), ...]`` for the named markers this rank holds.

    Marker indices are rank-local, and absence on a rank is normal.
    """
    indices = driver.GetMarkerIndices()
    out = []
    for name in names:
        marker_id = indices.get(name, -1)
        if marker_id < 0:
            continue
        n_vertex = driver.GetNumberMarkerNodes(marker_id)
        if n_vertex:
            out.append((marker_id, n_vertex))
    return out


def reference_surface(driver, comm, names, source):
    """``(x_ref, gid_order, ndim)`` from a pristine primal driver, which is finalized.

    Call once at setup, before any pysu2ad driver has run in the process.
    Halo vertices are left to their owner; a node reached through two
    markers lands in one slot. Raises when a marker resolves on no rank, because
    that would otherwise be an empty surface with no design variables.
    """
    ndim = driver.GetNumberDimensions()
    indices = driver.GetMarkerIndices()
    local, found_local = {}, set()
    for name in names:
        marker_id = indices.get(name, -1)
        if marker_id < 0:
            continue
        found_local.add(name)
        n_vertex = driver.GetNumberMarkerNodes(marker_id)
        if n_vertex == 0:
            continue
        ini = driver.MarkerInitialCoordinates(marker_id)
        for iv in range(n_vertex):
            i_point = driver.GetMarkerNode(marker_id, iv)
            if not driver.GetNodeDomain(i_point):
                continue
            local[driver.GetNodeGlobalIndex(i_point)] = tuple(ini.Get(iv))
    all_markers = sorted(indices)
    driver.Finalize()

    merged = {}
    for rank_nodes in comm.allgather(local):
        merged.update(rank_nodes)
    found = set()
    for rank_found in comm.allgather(found_local):
        found |= rank_found

    missing = [n for n in names if n not in found]
    if missing:
        raise RuntimeError(
            f'wall marker(s) {missing} resolve on no rank in {source}. '
            f'The mesh has {all_markers}. A marker that resolves nowhere yields '
            f'an empty surface rather than an error, so this raises instead.')

    gid_order = sorted(merged)
    x_ref = np.array([c for gid in gid_order for c in merged[gid]])
    return x_ref, gid_order, ndim


def apply_displacements(driver, markers, x_aero, x_ref, gid_pos, ndim):
    """Set every rank-local vertex (halos included) to ``x_aero - x_ref``.

    The reference is the setup-captured surface, never this driver's
    ``MarkerInitialCoordinates``.
    """
    for marker_id, n_vertex in markers:
        for iv in range(n_vertex):
            gid = driver.GetNodeGlobalIndex(driver.GetMarkerNode(marker_id, iv))
            pos = gid_pos[gid]
            disp = tuple(float(x_aero[ndim * pos + d] - x_ref[ndim * pos + d])
                         for d in range(ndim))
            driver.SetMarkerCustomDisplacement(marker_id, iv, disp)


def read_sensitivities(driver, markers, gid_pos, ndim, size, skip_unmapped=False):
    """Rank-local ``dJ/dx_aero``: owned vertices only, ASSIGNED per node.

    ``GetMarkerDisplacementSensitivity`` is per node, so a node shared by two
    markers is written twice with the same value, never summed. A gid missing
    from ``gid_pos`` raises ``KeyError`` unless ``skip_unmapped`` (the remesh
    path, whose new-mesh map covers only the nodes it matched).
    """
    djdx = np.zeros(size)
    for marker_id, n_vertex in markers:
        for iv in range(n_vertex):
            i_point = driver.GetMarkerNode(marker_id, iv)
            if not driver.GetNodeDomain(i_point):
                continue
            gid = driver.GetNodeGlobalIndex(i_point)
            if skip_unmapped and gid not in gid_pos:
                continue
            pos = gid_pos[gid]
            sens = driver.GetMarkerDisplacementSensitivity(marker_id, iv)
            for d in range(ndim):
                djdx[ndim * pos + d] = sens[d]
    return djdx


def reduce(comm, local):
    """Sum the owned-vertex rows across ranks, in place; identical on every rank."""
    comm.Allreduce(MPI.IN_PLACE, local, op=MPI.SUM)
    return local


def project_alpha(objective, sens_aoa, cl, cd, moment_key):
    """Total ``dJ/dalpha`` (per degree) from SU2's flow-path partial.

    Moments are integrated in body axes and take no projection term; drag adds
    ``C_L`` and lift subtracts ``C_D``, each times the degree factor. Efficiency
    ``E = C_L/C_D`` takes the quotient rule over both: ``-(1 + E**2)``.
    """
    if moment_key is not None and objective == moment_key:
        return sens_aoa
    if objective == 'DRAG':
        return sens_aoa + cl * DEG
    if objective == 'EFFICIENCY':
        return sens_aoa - (1.0 + (cl / cd) ** 2) * DEG
    return sens_aoa - cd * DEG


__all__ = ['local_markers', 'reference_surface', 'apply_displacements',
           'read_sensitivities', 'reduce', 'project_alpha', 'DEG']
