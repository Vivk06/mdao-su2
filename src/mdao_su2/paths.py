"""Repo-root and case paths, computed once. REPO = checkout root (this file is <REPO>/src/mdao_su2/paths.py)."""

import os

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CASES = os.path.join(REPO, 'cases')


def case_dir(name):
    """Absolute path to ``cases/<name>``. No default — caller must specify."""
    return os.path.join(CASES, name)


__all__ = ['REPO', 'CASES', 'case_dir']
