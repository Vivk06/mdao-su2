"""Per-instance output capture around the SU2 component interface."""

import hashlib
import os
import shutil
from types import MethodType

import numpy as np
import openmdao.api as om
from openmdao.drivers.pyoptsparse_driver import UserRequestedException

from ..vtu import vtu_to_csv, write_unstructured
from .layout import ARTIFACTS


class DriverStopError(UserRequestedException):
    """The optimizer must stop: its first design failed or it proposed NaN.

    Raised inside the model, OpenMDAO returns it as ``fail = 2``. From then
    on the driver wrapper answers every optimizer request at once with NaN
    and ``fail = 2``, without running the model or SU2: IPOPT and SNOPT stop
    at their next callback, and SLSQP, which ignores the flag, runs out its
    iterations in milliseconds. The optimizer returns normally, so it closes
    its history and releases the other ranks; OpenMDAO's final re-run then
    raises the stop on every rank, and ``run_driver`` reports it in place of
    anything OpenMDAO stored on the way. ``reported`` and ``exit_code`` let
    the console and ``run/optimize.py`` print its one-line message without a
    traceback.
    """

    reported = True
    exit_code = 1


def solver_components(model):
    """Every SU2 component under ``model``: it runs adjoints and names its cfgs."""
    for component in model.system_iter(include_self=True, recurse=True):
        if not hasattr(component, '_run_adjoint') or not hasattr(component, 'options'):
            continue
        try:
            component.options['adjoint_cfg']
            component.options['primal_cfg']
        except (KeyError, AttributeError):
            continue
        yield component


def component_point_id(component, cfg):
    """The point a component solves, from its generated cfg file name.

    ``cfg_name`` ends in ``_<point>.cfg``; the longest matching ID wins, so
    ``cruise_hi`` is not read as ``cruise`` or ``hi``.
    """
    name = os.path.basename(component.options['primal_cfg'])
    suffixed = [point.id for point in cfg.points if name.endswith(f'_{point.id}.cfg')]
    if suffixed:
        return max(suffixed, key=len)
    for point in cfg.points:
        if f'_{point.id}.' in name or f'_{point.id}_' in name:
            return point.id
    return cfg.points[0].id


def _stage(snapshotter, objective, point, cfg):
    from .layout import adjoint_stage_name
    from .results import stage_name
    return os.path.join(snapshotter.rundir.evaluations, snapshotter.current,
                        stage_name(adjoint_stage_name(objective), point, cfg))


def _copy(source, destination):
    if os.path.isfile(source):
        shutil.copy2(source, destination)
        return True
    return False


#: SU2 objective -> the suffix SU2 appends to its adjoint restart name
#: (``CConfig::GetObjFunc_Extension``, SU2 v8.5).
RESTART_TAG = {
    'DRAG': 'cd', 'LIFT': 'cl', 'SIDEFORCE': 'csf', 'EFFICIENCY': 'eff',
    'MOMENT_X': 'cmx', 'MOMENT_Y': 'cmy', 'MOMENT_Z': 'cmz',
    'FORCE_X': 'cfx', 'FORCE_Y': 'cfy', 'FORCE_Z': 'cfz',
    'INVERSE_DESIGN_PRESSURE': 'invpress', 'INVERSE_DESIGN_HEATFLUX': 'invheat',
    'EQUIVALENT_AREA': 'ea', 'NEARFIELD_PRESSURE': 'nfp',
    'THRUST': 'ct', 'TORQUE': 'cq', 'TOTAL_HEATFLUX': 'totheat',
    'MAXIMUM_HEATFLUX': 'maxheat', 'AVG_TEMPERATURE': 'avtp',
    'FIGURE_OF_MERIT': 'merit', 'BUFFET': 'buffet',
}


def _wrap_driver(driver, snapshotter, state):
    """Mark the optimizer's function and gradient requests for the snapshotter.

    Instance overrides of ``_objfunc`` and ``_gradfunc``: pyOptSparse calls
    them through ``WeakMethodWrapper``, which looks the method up by name.
    Once ``state['stop']`` is set they return NaN with ``fail = 2`` at once
    (see ``DriverStopError``). Returns the restore function.
    """
    if driver is None or not hasattr(driver, '_objfunc'):
        return lambda: None
    objfunc, gradfunc = driver._objfunc, driver._gradfunc
    template = {}         # the shape of the last function values returned

    def wrapped_objfunc(dv_dict):
        if state['stop'] is not None and template:
            return {name: np.full_like(value, np.nan) for name, value in template.items()}, 2
        snapshotter.start_call(dv_dict)
        try:
            funcs, fail = objfunc(dv_dict)
        finally:
            snapshotter.end_call()
        template.clear()
        template.update({name: np.array(value, dtype=float) for name, value in funcs.items()})
        return funcs, fail

    def wrapped_gradfunc(dv_dict, func_dict):
        if state['stop'] is not None:
            return {name: {dv: np.full((np.size(value), np.size(dv_value)), np.nan)
                           for dv, dv_value in dv_dict.items()}
                    for name, value in func_dict.items()}, 2
        snapshotter.gradient_requested(dv_dict)
        return gradfunc(dv_dict, func_dict)

    driver._objfunc = wrapped_objfunc
    driver._gradfunc = wrapped_gradfunc

    def restore():
        driver.__dict__.pop('_objfunc', None)
        driver.__dict__.pop('_gradfunc', None)
    return restore


def _wrap_analysis_errors(driver, snapshotter):
    """Record any ``AnalysisError`` of the model as a failure of the current call.

    pyOptSparse's ``_objfunc`` catches it around ``_run_solve_nonlinear`` and
    returns ``fail = 1``. An error outside the SU2 components (objective
    domain, geometry, mesh) would otherwise leave the call marked ``ok``, so
    the call is marked failed before the recorder's ``take()`` runs.
    ``call_failed`` keeps the first failure. Returns the restore function.
    """
    if driver is None or not hasattr(driver, '_run_solve_nonlinear'):
        return lambda: None
    run = driver._run_solve_nonlinear

    def wrapped_run():
        try:
            return run()
        except om.AnalysisError as exc:
            snapshotter.call_failed(str(exc))
            raise

    driver._run_solve_nonlinear = wrapped_run

    def restore():
        driver.__dict__.pop('_run_solve_nonlinear', None)
    return restore


def install_solve_hooks(prob, cfg, rundir, snapshotter):
    """Capture each adjoint before shared SU2 names are overwritten.

    Instance overrides keep ``aero_comp.py`` and its derived components unchanged.
    """
    saved = []
    state = {'stop': None}
    driver = getattr(prob, 'driver', None)
    restore_driver = _wrap_driver(driver, snapshotter, state)
    restore_analysis = _wrap_analysis_errors(driver, snapshotter)

    def refuse_non_finite(component, inputs):
        """Stop before SU2 sees a non-finite design: it aborts the MPI job."""
        if all(np.all(np.isfinite(inputs[name]))
               for name in ('x_aero', 'alpha') if name in inputs):
            return
        point = component_point_id(component, cfg)
        if state['stop'] is None:
            state['stop'] = DriverStopError(
                f'The optimizer proposed a non-finite design at {point}; SU2 was '
                f'not called. An earlier evaluation failed: see logs/progress.log.')
            snapshotter.call_failed(state['stop'])
        raise state['stop']

    for component in solver_components(prob.model):
        original = component._run_adjoint
        original_compute = component.compute
        original_response = getattr(component, '_run_response_adjoint', None)
        solved = [False]      # this point has had one successful primal

        def wrapped_compute(self, inputs, outputs, _original=original_compute,
                            _solved=solved):
            point = component_point_id(self, cfg)
            if state['stop'] is not None:
                raise state['stop']        # OpenMDAO's final re-run after a stop
            refuse_non_finite(self, inputs)
            try:
                result = _original(inputs, outputs)
            except om.AnalysisError as exc:
                snapshotter.call_failed(f'{point}: {exc}')
                if _solved[0]:
                    raise          # a line-search step failed; the optimizer backs off
                state['stop'] = DriverStopError(
                    f'The first primal at {point} failed ({exc}); the optimizer cannot '
                    f'start from a failed design. Converge it (for example a larger '
                    f'[solver.primal] iter).')
                raise state['stop'] from exc
            _solved[0] = True
            return result

        def wrapped(self, objective, inputs, _original=original):
            refuse_non_finite(self, inputs)
            result = _original(objective, inputs)
            capture(self, objective, inputs, result)
            return result

        def wrapped_response(self, record, inputs, _original=original_response):
            refuse_non_finite(self, inputs)
            result = _original(record, inputs)
            capture(self, record.su2_objective, inputs, result)
            return result

        component._run_adjoint = MethodType(wrapped, component)
        component.compute = MethodType(wrapped_compute, component)
        if original_response is not None:
            component._run_response_adjoint = MethodType(wrapped_response, component)
        saved.append((component, original, original_compute, original_response))

    def capture(self, objective, inputs, result):
        """Rank 0: copy one adjoint's files into the current call folder."""
        if self.comm.rank != 0 or snapshotter.current is None:
            return
        point = component_point_id(self, cfg)
        directory = _stage(snapshotter, objective, point, cfg)
        os.makedirs(directory, exist_ok=True)
        suffix = f'_{point}'
        work = rundir.workdir
        _copy(os.path.join(work, f'history_adj{suffix}.csv'),
              os.path.join(directory, ARTIFACTS['history']))
        surface = os.path.join(work, f'surface_adjoint{suffix}.vtu')
        if _copy(surface, os.path.join(directory, ARTIFACTS['surface'])):
            vtu_to_csv(os.path.join(directory, ARTIFACTS['surface']),
                       os.path.join(directory, 'surface.csv'))
        from .layout import keep_volume
        if keep_volume(snapshotter.snapshots):
            _copy(os.path.join(work, f'adjoint{suffix}.vtu'),
                  os.path.join(directory, ARTIFACTS['volume']))
        tag = RESTART_TAG.get(str(objective).upper(), str(objective).lower())
        for source in (f'restart_adj{suffix}_{tag}.dat',
                       f'restart_adj{suffix}.dat'):
            if _copy(os.path.join(work, source), os.path.join(directory, ARTIFACTS['restart'])):
                break
        from ..solver.cfg_file import derive_cfg
        derive_cfg(self.options['adjoint_cfg'],
                   os.path.join(directory, ARTIFACTS['config']),
                   {'OBJECTIVE_FUNCTION': objective})
        djdx, djdalpha = result
        coords = np.asarray(inputs['x_aero']).reshape(-1, self._ndim)
        sens = np.asarray(djdx).reshape(-1, self._ndim)
        write_unstructured(os.path.join(directory, 'sensitivity.vtu'), coords,
                           [], np.empty(0, dtype=int),
                           point_data={'Sensitivity': np.pad(
                               sens, ((0, 0), (0, 3 - self._ndim)))})
        vtu_to_csv(os.path.join(directory, 'sensitivity.vtu'),
                   os.path.join(directory, 'sensitivity.csv'))
        digest = hashlib.sha1(np.asarray(inputs['x_aero']).tobytes()).hexdigest()
        snapshotter.adjoint_done(objective, point, djdalpha, digest)

    def restore():
        for component, original, original_compute, original_response in saved:
            component._run_adjoint = original
            component.compute = original_compute
            if original_response is not None:
                component._run_response_adjoint = original_response
        restore_driver()
        restore_analysis()
    restore.state = state      # run_driver re-raises state['stop'] after the driver
    return restore
