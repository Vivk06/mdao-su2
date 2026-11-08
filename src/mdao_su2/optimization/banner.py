"""The startup banner: the SU2 MDO logo above what this run is.

Modelled on the box SU2's ``shape_optimization.py`` prints. Rank 0 prints it
once, first, from ``tasks.prepare``; every row is ``WIDTH`` characters wide
with aligned borders, and a value too long for its row is shortened in the
middle.
"""

import os
from datetime import datetime, timezone

__all__ = ['banner_lines', 'run_banner', 'WIDTH']

#: Width of every banner row, borders included (SU2's box width).
WIDTH = 73
#: Room for text between ``| `` and `` |``.
_INNER = WIDTH - 4
#: Column of the right-hand field in a two-field row.
_RIGHT = 32
#: Width of a row label (``Task``, ``Config``, ...).
_LABEL = 8

_LOGO = (
    r"  ___ _   _ ___      __  __ ___   ___",
    r" / __| | | |_  )    |  \/  |   \ / _ \    SU2 MDO",
    r" \__ \ |_| |/ /     | |\/| | |) | (_) |   Adjoint shape optimization",
    r" |___/\___//___|    |_|  |_|___/ \___/    pysu2ad + OpenMDAO/MPhys",
    '',
)


def _shorten(text, width):
    """``text``, or its two ends joined by ``…`` when it is wider than ``width``."""
    text = str(text)
    if len(text) <= width:
        return text
    keep = width - 1
    return f'{text[:keep - keep // 2]}…{text[len(text) - keep // 2:]}'


def _field(label, value, width):
    return f'{label:<{_LABEL}}{_shorten(value, width - _LABEL)}'


def _row(text):
    return f'| {text:<{_INNER}} |'


def _pair(left_label, left, right_label, right):
    """Two labelled fields; the second label starts at column ``_RIGHT``."""
    return _row(_field(left_label, left, _RIGHT - 2).ljust(_RIGHT)
                + _field(right_label, right, _INNER - _RIGHT))


def banner_lines(task, case, config, run, ranks, su2_commit, code_sha, dirty,
                 start):
    """The banner rows; ``dirty`` is True, False or None (unknown)."""
    rule = '-' * WIDTH
    state = {True: 'dirty', False: 'clean'}.get(dirty)
    code = f'{str(code_sha)[:7]} ({state})' if state else str(code_sha)[:7]
    return [rule, *(_row(line) for line in _LOGO), rule,
            _pair('Task', task, 'Case', case),
            _row(_field('Config', config, _INNER)),
            _row(_field('Run', run, _INNER)),
            _pair('Ranks', ranks, 'SU2', f'v8.5 ({str(su2_commit)[:10]})'),
            _pair('Code', code, 'Start', start),
            rule]


def run_banner(cfg, rd, ranks):
    """``banner_lines`` for a configured run in ``rd``."""
    from ..paths import REPO
    from ..run_output.rundir import git_head
    from ..solver.su2_options_v85 import SU2_COMMIT
    sha, dirty = git_head(REPO)
    root = os.path.abspath(rd.root)
    run = os.path.relpath(root, REPO) if root.startswith(REPO + os.sep) else root
    start = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    return banner_lines(cfg.execution.task, cfg.case.name, cfg.source, run, ranks,
                        SU2_COMMIT, sha, dirty, start)
