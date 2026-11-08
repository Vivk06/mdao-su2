"""One adjoint per (point, functional, design) across ``check_totals``,
``compute_totals`` and ``run_driver``.

OpenMDAO seeds each response separately in reverse mode, so every seeded
response runs its own adjoint even when two responses share a functional (the
objective and a linear row both ask for a point's drag), and ``check_totals``
repeats the analytic totals for every group. The
vector-Jacobian product is linear in the seed, so one (dJ/dx_aero,
dJ/dalpha) per functional and design serves every seeded response. A result
at the same inputs is reused; each component keeps one design at a time. FD
primals still run, and reuse changes no total.
"""

import hashlib
from contextlib import contextmanager
from types import MethodType

import numpy as np

from ..run_output.solve_hooks import solver_components


def _design_key(inputs):
    digest = hashlib.sha1()
    for name in ('x_aero', 'alpha'):
        digest.update(np.ascontiguousarray(inputs[name], dtype=np.float64).tobytes())
    return digest.hexdigest()


def _agree(comm, hit):
    """Every rank of the component reuses, or none does."""
    if comm is None or getattr(comm, 'size', 1) == 1:
        return hit
    return bool(comm.allreduce(int(hit), op=_mpi_min()))


def _mpi_min():
    from mpi4py import MPI
    return MPI.MIN


@contextmanager
def reuse_adjoints(model):
    """Yield ``{'solved': n, 'reused': m}``, counted while the block runs."""
    counts = {'solved': 0, 'reused': 0}
    installed = []
    systems = solver_components(model) if hasattr(model, 'system_iter') else ()
    for component in systems:
        cache = {}
        names = ['_run_adjoint']
        if hasattr(component, '_run_response_adjoint'):
            names.append('_run_response_adjoint')
        for name in names:
            original = getattr(component, name)

            def cached(self, functional, inputs, _original=original, _name=name,
                       _cache=cache):
                tag = getattr(functional, 'su2_objective', functional)
                design = _design_key(inputs)
                if _cache.get('design') != design:
                    _cache.clear()
                    _cache['design'] = design
                key = (_name, str(tag), design)
                if _agree(self.comm, key in _cache):
                    counts['reused'] += 1
                    djdx, djdalpha = _cache[key]
                    return np.array(djdx, copy=True), djdalpha
                result = _original(functional, inputs)
                counts['solved'] += 1
                _cache[key] = (np.array(result[0], copy=True), result[1])
                return result

            had = name in component.__dict__
            installed.append((component, name, had, component.__dict__.get(name)))
            setattr(component, name, MethodType(cached, component))
    try:
        yield counts
    finally:
        for component, name, had, previous in reversed(installed):
            if had:
                setattr(component, name, previous)
            else:
                component.__dict__.pop(name, None)
