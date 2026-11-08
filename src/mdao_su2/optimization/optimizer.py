"""pyOptSparse driver setup from a RunConfig. Defaults live in run_config/root.py::OPT_DEFAULTS."""

import os
import sys
from contextlib import contextmanager

import openmdao.api as om

__all__ = ['setup_driver', 'OPT_FILE_KEYS', 'OPTIMIZER_LOG', 'optimizer_output',
           'describe_outcome', 'outcome']

#: pyOptSparse's problem and result tables and OpenMDAO's ``debug_print``
#: output, in ``opt_hst/``; the terminal keeps the progress lines.
OPTIMIZER_LOG = 'optimizer.log'

# Per-optimizer settings keys whose value is an output filename.
OPT_FILE_KEYS = ('IFILE',            # SLSQP
                 'output_file',      # IPOPT
                 'Print file', 'Summary file')   # SNOPT


def setup_driver(prob, cfg, rundir=None):
    """Attach a pyOptSparse driver configured by ``cfg.optimizer``."""
    opt = cfg.optimizer
    settings = opt.opt_settings
    hist_file = opt.resolved_hist_file()
    if rundir is not None:
        hist_file = os.path.join(rundir.opt, os.path.basename(hist_file))
        settings = {
            k: (os.path.join(rundir.opt, os.path.basename(v))
                if k in OPT_FILE_KEYS and isinstance(v, str) else v)
            for k, v in settings.items()}

    # A failed point raises om.AnalysisError; OpenMDAO returns fail = 1 and, for
    # optimizers outside its respects_fail_flag table (SLSQP, IPOPT), NaN values.
    # SLSQP ignores the flag; the NaN fails its line-search test and the step is
    # cut before the solve budget is exhausted.
    prob.driver = om.pyOptSparseDriver()
    prob.driver.options['optimizer'] = opt.name
    prob.driver.options['debug_print'] = list(opt.debug_print)
    prob.driver.options['print_opt_prob'] = opt.print_opt_prob
    prob.driver.options['hist_file'] = hist_file
    if rundir is not None:
        # OpenMDAO overwrites IFILE / output_file / Print file unconditionally
        # with `<output_dir>/<OPT>.out` (pyoptsparse_driver.py), discarding the
        # OPT_FILE_KEYS rewrite above. Without this the SLSQP log lands in
        # OpenMDAO's default output directory, outside the documented run layout.
        prob.driver.options['output_dir'] = rundir.opt
    # `opt_settings` resolves the per-optimizer block, so switching optimizer is
    # a one-key change rather than an if/elif chain that must be kept in sync.
    prob.driver.opt_settings = settings
    return prob.driver


@contextmanager
def optimizer_output(opt_dir, rank0=True):
    """Send Python's stdout to ``<opt_dir>/optimizer.log`` while the driver runs.

    pyOptSparse and OpenMDAO print their tables through ``print``; the
    progress lines write to the console's terminal stream directly, so they
    stay on the terminal and in ``logs/progress.log``.
    """
    if not rank0 or opt_dir is None:
        yield None
        return
    os.makedirs(opt_dir, exist_ok=True)
    saved = sys.stdout
    with open(os.path.join(opt_dir, OPTIMIZER_LOG), 'a') as stream:
        sys.stdout = stream
        try:
            yield stream
        finally:
            sys.stdout.flush()
            sys.stdout = saved


def outcome(driver, calls, final_design=None):
    """The ``optimizer`` block of ``results_run_driver.json``.

    ``calls`` are ``IterationSnapshotter`` call records (``status``,
    ``major_iteration``, ``has_gradient``, ``objective``); ``final_design``
    maps each design variable to its final value. ``major_iterations``
    counts the major iterations after the baseline (major 0).
    """
    # pyOptSparse 2.16: a SolutionInform (value, message); older: a dict.
    inform = getattr(getattr(driver, 'pyopt_solution', None), 'optInform', None)
    code = getattr(inform, 'value', None)
    text = getattr(inform, 'message', None)
    if isinstance(inform, dict):
        code, text = inform.get('value'), inform.get('text')
    majors = [c for c in calls if c['major_iteration'] is not None]
    return {
        'name': driver.options['optimizer'],
        'exit_code': None if code is None else int(code),
        'exit_message': str(text or '').strip().rstrip('.') or None,
        'major_iterations': max((c['major_iteration'] for c in majors), default=0),
        'calls': len(calls),
        'failed_calls': sum(1 for c in calls if c['status'] == 'failed'),
        'gradient_calls': sum(1 for c in calls if c.get('has_gradient')),
        'final_design': final_design,
    }


def describe_outcome(driver, calls):
    """The closing line of ``run_driver``: exit message, counts, objective change.

    ``calls`` are ``IterationSnapshotter`` call records (``status``,
    ``major_iteration``, ``objective``).
    """
    block = outcome(driver, calls)
    name, code = block['name'], block['exit_code']
    exit_text = block['exit_message'] or 'no exit message'
    head = f'{name} exit {code}: {exit_text}' if code is not None else f'{name}: {exit_text}'
    majors = [c for c in calls if c['major_iteration'] is not None]
    text = (f'{head}; {block["major_iterations"]} major iteration(s) after the baseline, '
            f'{block["calls"]} call(s) ({block["failed_calls"]} failed)')
    if majors and majors[0]['objective'] is not None and majors[-1]['objective'] is not None:
        first, last = majors[0]['objective'], majors[-1]['objective']
        change = f' ({(last / first - 1.0) * 100:+.2f}%)' if first else ''
        text += f'; objective {first:.6e} -> {last:.6e}{change}'
    return text
