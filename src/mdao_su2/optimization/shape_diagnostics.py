"""Opt-in shape cancellation diagnostics; no solver calls or band changes."""
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from ..mpi import on_root
from ..solver.aero_comp import SU2AeroComp
from ..solver.variable_names import SURF


def _tangential_source(x_ref, sensitivity, local_jac, ndim):
    """Split each shape row into normal- and tangential-sourced parts (2D only).

    Tangential surface motion is a reparameterization, not a shape change, so it
    lies in the nullspace of the continuous shape derivative and the discrete
    adjoint carries no reliable information about it. A row drawing
    much of its value from the tangential part is therefore not FD-verifiable,
    however well converged the primal is.
    """
    if ndim != 2:
        return None
    pts = x_ref.reshape(-1, 2)
    if len(pts) < 3:
        return None
    d2 = ((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    neighbours = np.argsort(d2, axis=1)[:, :2]
    tangent = pts[neighbours[:, 0]] - pts[neighbours[:, 1]]
    length = np.linalg.norm(tangent, axis=1, keepdims=True)
    if not np.all(length > 0):
        return None
    tangent = tangent / length
    # A per-node sign flip of the frame cancels in both products below.
    normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)
    field = sensitivity.reshape(-1, 2)
    motion = local_jac.reshape(len(pts), 2, -1)
    tangential = ((field * tangent).sum(1)[:, None]
                  * np.einsum('nd,ndk->nk', tangent, motion)).sum(0)
    normal_part = ((field * normal).sum(1)[:, None]
                   * np.einsum('nd,ndk->nk', normal, motion)).sum(0)
    total = np.abs(tangential) + np.abs(normal_part)
    fraction = np.divide(np.abs(tangential), total,
                         out=np.zeros_like(total), where=total > 0)
    return tangential, normal_part, fraction


@contextmanager
def capture_shape_sensitivities(prob, cfg, rundir):
    """Temporarily intercept the already-replicated, unseeded adjoint field."""
    if (rundir is None or cfg.execution.parallel_points
            or cfg.geometry.parameterization != 'ffd' or cfg.remesh.enabled
            or cfg.execution.task != 'check_totals'):
        raise ValueError('Shape diagnostics require a run directory and '
                         'fixed-mesh, nonparallel FFD check_totals')
    fields, restore = {}, []
    try:
        for point in cfg.points:
            comps = [c for c in prob.model.system_iter(typ=SU2AeroComp)
                     if c.pathname.startswith(point.id + '.')]
            if len(comps) != 1:
                raise ValueError(f'{point.id}: expected exactly one SU2 aero component')
            comp = comps[0]
            original = comp._run_adjoint
            # Preserve an existing instance override, or remove ours to restore
            # ordinary class-method lookup (including multi-marker subclasses).
            restore.append((comp, '_run_adjoint' in comp.__dict__, original))

            def capture(objective, inputs, *, _original=original,
                        _comp=comp, _point=point.id):
                result = _original(objective, inputs)
                key = (_point, objective)
                sample = {
                    'sensitivities': np.array(result[0], dtype=np.float64, copy=True),
                    'dalpha': np.array(result[1], dtype=np.float64),
                    'x_ref': np.array(_comp.x_aero0, dtype=np.float64, copy=True),
                    'x_current': np.array(inputs['x_aero'], dtype=np.float64, copy=True),
                    'alpha': np.array(inputs['alpha'], dtype=np.float64, copy=True),
                    'gid': np.array(_comp._gid_order, dtype=np.int64),
                    'ndim': _comp._ndim,
                    'component': _comp.pathname,
                    'output': {'DRAG': 'C_D', 'LIFT': 'C_L',
                               _comp.options['moment_key']: 'C_M'}[objective],
                }
                if key in fields:
                    for name in ('sensitivities', 'x_current', 'alpha'):
                        np.testing.assert_allclose(fields[key][name], sample[name],
                                                   rtol=1e-12, atol=1e-14)
                fields[key] = sample
                return result

            comp._run_adjoint = capture
        yield fields
    finally:
        for comp, had_override, original in reversed(restore):
            if had_override:
                comp._run_adjoint = original
            else:
                del comp._run_adjoint


def write_shape_sensitivities(prob, cfg, rundir, fields, shape_data):
    """Contract captured fields with the refreshed geometry-only Jacobian.

    Geometry emits gid-ordered xyz; SU2 uses xy in 2D and xyz in 3D.
    All ranks participate in compute_totals/get_val. Root alone writes the
    replicated field, without another reduction or multiplication by MPI size.
    """
    if prob.model._owns_approx_jac:
        raise ValueError('Shape diagnostics require analytic geometry totals')
    coords = prob.model.get_source(f'geometry.{SURF.Geometry.COORDINATES_OUTPUT}')
    jac = prob.compute_totals(of=[coords], wrt=['shape'])[(coords, 'shape')]
    xyz = prob.get_val(coords, get_remote=True).reshape(-1, 3)
    shape = prob.get_val('shape', get_remote=True).copy()
    # C_D and C_L exist at every point; a catalog output only where the
    # objective reads it (its points request the response).
    wanted = {(p.id, name) for p in cfg.points for name in ('C_D', 'C_L')}
    wanted |= set(cfg.objective_compiled.sources.values())
    values = {}
    for pid, name in sorted(wanted):
        values.setdefault(pid, {})[name] = float(
            prob.get_val(f'{pid}.{name}', get_remote=True)[0])

    def write():
        payload = {'geometry_jacobian': np.asarray(jac, dtype=np.float64),
                   'geometry_coordinates': np.asarray(xyz, dtype=np.float64),
                   'shape': np.asarray(shape, dtype=np.float64)}
        report = {'geometry_output': coords, 'diagnostic_only': True,
                  'fields': [], 'comparisons': []}
        contractions = {}
        if not fields:
            raise ValueError('No shape adjoint fields captured')
        for (point, objective), sample in fields.items():
            prefix = f'{point}__{objective}'
            ndim, gids = sample['ndim'], sample['gid']
            if ndim not in (2, 3) or len(gids) != len(xyz) or np.any(np.diff(gids) <= 0):
                raise ValueError(f'{prefix}: invalid gid-sorted coordinate layout')
            x = xyz[:, :ndim].reshape(-1)
            np.testing.assert_allclose(sample['x_current'], x, rtol=0, atol=1e-12)
            local_jac = np.asarray(jac).reshape(len(gids), 3, -1)[:, :ndim, :]
            local_jac = local_jac.reshape(len(x), -1)
            sensitivity = sample['sensitivities'].reshape(-1)
            if sensitivity.shape != x.shape or not np.all(np.isfinite(sensitivity)):
                raise ValueError(f'{prefix}: invalid sensitivity field')
            contributions = sensitivity[:, None] * local_jac
            contraction = contributions.sum(axis=0)
            l1 = np.abs(contributions).sum(axis=0)
            cr = np.divide(l1, np.abs(contraction), out=np.full_like(l1, np.inf),
                           where=contraction != 0)
            for name in ('sensitivities', 'dalpha', 'x_ref', 'x_current', 'alpha', 'gid'):
                payload[f'{prefix}__{name}'] = sample[name]
            for name, arr in (('contraction', contraction), ('l1', l1), ('cr', cr)):
                payload[f'{prefix}__{name}'] = arr
            contractions[(point, sample['output'])] = contraction
            entry = {
                'point': point, 'objective': objective, 'output': sample['output'],
                'component': sample['component'], 'npz_prefix': prefix, 'ndim': ndim,
                'contraction': contraction.tolist(), 'l1': l1.tolist(),
                # Infinite C_r means an exactly zero signed sum, not a pass.
                'cr': [float(c) if np.isfinite(c) else None for c in cr],
                'zero_sum': (contraction == 0).tolist()}
            split = _tangential_source(sample['x_ref'], sensitivity, local_jac, ndim)
            if split is not None:
                tangential, normal_part, fraction = split
                np.testing.assert_allclose(tangential + normal_part, contraction,
                                           rtol=1e-9, atol=1e-14)
                for name, arr in (('tangential_source', tangential),
                                  ('normal_source', normal_part),
                                  ('tangential_fraction', fraction)):
                    payload[f'{prefix}__{name}'] = arr
                    entry[name] = arr.tolist()
            report['fields'].append(entry)

        for (of, wrt), row in shape_data.items():
            if wrt != 'shape':
                continue
            if of.split('.')[-1] in ('obj', 'obj_out'):
                # d obj / d output from the compiled objective.
                compiled = cfg.objective_compiled
                factors = compiled.partials(
                    {name: values[pid][output]
                     for name, (pid, output) in compiled.sources.items()})
                contracted = np.zeros(shape.size)
                for name, factor in factors.items():
                    if factor:
                        contracted += factor * contractions[compiled.sources[name]]
            else:
                point, output = of.split('.')
                contracted = contractions[(point, output)]
            analytic = np.asarray(row['J_rev']).reshape(-1)
            passed = bool(np.all(np.isfinite(analytic))
                          and np.all(np.isfinite(contracted))
                          and np.allclose(contracted, analytic, rtol=1e-10, atol=1e-12))
            payload[f'{of}__analytic_shape'] = analytic.astype(np.float64)
            report['comparisons'].append({
                'of': of, 'pass': passed, 'analytic': analytic.tolist(),
                'contraction': contracted.tolist(),
                'max_abs_error': float(np.max(np.abs(contracted - analytic)))})
        report['contraction_pass'] = all(r['pass'] for r in report['comparisons'])
        report['fields_archived'] = len(fields)
        np.savez(Path(rundir.root) / 'shape_sensitivities.npz', **payload)
        return report

    report = on_root(prob.comm, write, what='archive shape sensitivities')
    if prob.comm.rank == 0 and report['contraction_pass']:
        print(f'Shape diagnostics: {report["fields_archived"]} fields archived; '
              'contractions verified. Measured C_r is diagnostic only; '
              'acceptance bands unchanged.')
    return report
