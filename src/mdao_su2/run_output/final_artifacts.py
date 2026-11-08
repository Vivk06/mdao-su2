"""``<run>/final/``: each point's last solver artifacts with plain names.

On ``DONE`` rank 0 copies what ``su2_workdir/`` holds for every point into
``final/<point>/`` and moves ``run_driver``'s surface and DV arrays into
``final/``; ``rundir.finish`` then removes ``su2_workdir/``. A failed run keeps
``su2_workdir/`` and gets no ``final/``.

SU2 overwrites one adjoint history, surface and volume file per point, so
those belong to the functional solved last (the newest adjoint restart);
every functional keeps its restart and its cfg.
"""

import glob
import os
import shutil

from ..mpi import on_root
from ..optimization.progress import adjoint_label
from ..solver.cfg_file import cfg_value, derive_cfg
from .solve_hooks import RESTART_TAG
from .layout import final_artifact, source_artifact

__all__ = ['keep_final', 'seed_files', 'DRIVER_ARRAYS']

#: ``run_driver``'s arrays in ``su2_workdir/``, moved to ``final/``.
DRIVER_ARRAYS = ('baseline_surface.npy', 'optimized_*.npy')

#: SU2 restart tag -> SU2 objective (``cd`` -> ``DRAG``).
_TAG_OBJECTIVE = {tag: objective for objective, tag in RESTART_TAG.items()}


def _copy(source, target):
    if source is None or not os.path.isfile(source):
        return False
    shutil.copy2(source, target)
    return True


def _cfg_path(cfg_dir, cfg, point, is_adjoint):
    """The generated cfg of ``point``, or None when its name cannot be formed."""
    from ..solver.point_cfg_writer import cfg_name
    try:
        return os.path.join(cfg_dir, cfg_name(cfg, point, is_adjoint))
    except (AttributeError, TypeError):
        return None


def _primal(work, cfg_dir, cfg, point, target):
    _copy(_cfg_path(cfg_dir, cfg, point, False),
          os.path.join(target, final_artifact('primal', 'config')))
    su2 = cfg.output.su2
    for stem, kind in (('history', 'history'), (su2.restart_basename, 'restart'),
                       (su2.volume_basename, 'volume'), (su2.surface_basename, 'surface')):
        name = final_artifact('primal', kind)
        extension = os.path.splitext(name)[1]
        _copy(source_artifact(work, stem, point, extension), os.path.join(target, name))


def _adjoints(work, cfg_dir, cfg, point, target):
    """Per functional: the cfg and the restart; the last one also its history and fields."""
    adjoint_cfg = _cfg_path(cfg_dir, cfg, point, True)
    prefix = os.path.join(work, f'restart_adj_{point.id}_')
    restarts = sorted(glob.glob(f'{prefix}*.dat'), key=os.path.getmtime)
    # Only SU2's own suffixes: another point whose ID starts with this one's
    # (``a`` and ``a_b``) also matches the glob.
    restarts = [path for path in restarts
                if path[len(prefix):-len('.dat')] in _TAG_OBJECTIVE]
    for index, restart in enumerate(restarts):
        tag = restart[len(prefix):-len('.dat')]
        objective = _TAG_OBJECTIVE[tag]
        stem = f'adjoint_{adjoint_label(objective)}'
        if adjoint_cfg is not None and os.path.isfile(adjoint_cfg):
            derive_cfg(adjoint_cfg, os.path.join(target, final_artifact(stem, 'config')),
                       {'OBJECTIVE_FUNCTION': objective})
        _copy(restart, os.path.join(target, final_artifact(stem, 'restart')))
        if index == len(restarts) - 1:
            for source, kind in ((f'history_adj_{point.id}.csv', 'history'),
                                 (f'surface_adjoint_{point.id}.vtu', 'surface'),
                                 (f'adjoint_{point.id}.vtu', 'volume')):
                _copy(os.path.join(work, source),
                      os.path.join(target, final_artifact(stem, kind)))


def _write(cfg, rd):
    work = rd.workdir
    for point in cfg.points:
        target = os.path.join(rd.final, point.id)
        os.makedirs(target, exist_ok=True)
        _primal(work, rd.su2_cfg, cfg, point, target)
        _adjoints(work, rd.su2_cfg, cfg, point, target)
        if not os.listdir(target):
            os.rmdir(target)
    for pattern in DRIVER_ARRAYS:
        for path in sorted(glob.glob(os.path.join(work, pattern))):
            os.makedirs(rd.final, exist_ok=True)
            shutil.move(path, os.path.join(rd.final, os.path.basename(path)))
    return True


def keep_final(cfg, rd, comm=None):
    """Write ``final/`` on rank 0; collective when ``comm`` spans several ranks."""
    if comm is None or getattr(comm, 'size', 1) == 1:
        return _write(cfg, rd)
    return on_root(comm, _write, cfg, rd, what='keep final artifacts')


def seed_files(directory, cfg, kind, destination, source):
    """``[(path, su2_workdir name)]`` of one seed ``kind`` for one point pair.

    ``directory`` is a run's ``final/`` or a solver directory; ``source`` is
    the point id there and ``destination`` the point id here. A flow seed is
    the primal restart, renamed ``solution_flow_<destination>.dat``; an
    adjoint seed is every adjoint restart, renamed
    ``restart_adj_<destination>_<tag>.dat`` with the tag of its functional.
    """
    folder = os.path.join(directory, source)
    if os.path.isdir(folder):
        if kind == 'flow':
            restart = os.path.join(folder, final_artifact('primal', 'restart'))
            return ([(restart, f'solution_flow_{destination}.dat')]
                    if os.path.isfile(restart) else [])
        files = []
        for adjoint_cfg in sorted(glob.glob(os.path.join(folder, 'adjoint_*.cfg'))):
            stem = os.path.splitext(adjoint_cfg)[0]
            objective = cfg_value(adjoint_cfg, 'OBJECTIVE_FUNCTION')
            restart = f'{stem}_restart.dat'
            if objective and os.path.isfile(restart):
                tag = RESTART_TAG.get(objective.upper(), objective.lower())
                files.append((restart, f'restart_adj_{destination}_{tag}.dat'))
        return files
    # A solver directory keeps SU2's point-suffixed names.
    if kind == 'flow':
        for name in (f'{cfg.output.su2.restart_basename}_{source}.dat',
                     f'solution_flow_{source}.dat'):
            path = os.path.join(directory, name)
            if os.path.isfile(path):
                return [(path, f'solution_flow_{destination}.dat')]
        return []
    prefix = os.path.join(directory, f'restart_adj_{source}_')
    return [(path, f'restart_adj_{destination}_{path[len(prefix):-len(".dat")]}.dat')
            for path in sorted(glob.glob(f'{prefix}*.dat'))
            if path[len(prefix):-len('.dat')] in _TAG_OBJECTIVE]
