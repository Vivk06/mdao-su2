"""Collective wrappers for rank-0-only operations. Prevents MPI deadlocks by ensuring all ranks raise together."""

import traceback

__all__ = ['on_root', 'raise_together']


def on_root(comm, fn, *args, what='operation', error=None):
    """Run fn(*args) on rank 0, broadcast result or raise on ALL ranks."""
    error = error or RuntimeError
    payload = None  # (True, result) or (False, traceback_str)

    if comm.rank == 0:
        try:
            result = fn(*args)
            payload = (True, result)
        except Exception:
            payload = (False, traceback.format_exc())

    payload = comm.bcast(payload, root=0)

    ok, value = payload
    if not ok:
        raise error(f'{what} failed on rank 0:\n{value}')
    return value


def raise_together(comm, failure, error=None):
    """If ANY rank has a non-None failure, EVERY rank raises."""
    error = error or RuntimeError
    # Gather all failures; any non-None triggers everyone.
    all_failures = comm.allgather(failure)
    bad = [f for f in all_failures if f is not None]
    if bad:
        # Use the first failure message (they're usually identical).
        raise error(bad[0])


# Rank-0 sites in builder.py/aero_comp.py — safe because they run at setup().
# Rank 0 computes, then broadcasts.
