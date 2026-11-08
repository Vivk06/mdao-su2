"""Solver stdout kept out of the terminal, and the run's progress log.

Inside the console, file descriptor 1 (where SU2 writes) goes to
``<logdir>/solver.log`` on rank 0 (``solver_stdout = "log"``) or to
``<logdir>/solver_last.out`` (``"discard"``, truncated before each solve), and
to ``/dev/null`` on other ranks. Python's ``sys.stdout`` on rank 0 is a tee to
the terminal and ``<logdir>/progress.log``, so task reports and progress lines
reach both while SU2's residuals reach neither.

Under ``"log"``, a solve that succeeds keeps its header line and its last
``KEEP_LINES`` lines; a failed solve keeps all of its output. Between solves,
once ``solver.log`` passes ``ROTATE_BYTES``, it is gzipped to
``solver.<NNN>.log.gz`` and a new ``solver.log`` is opened.
"""

import ctypes
import gzip
import os
import shutil
import sys
import warnings
from collections import deque
from contextlib import contextmanager

from ..paths import REPO

#: Lines a successful solve keeps in ``solver.log`` after its header line.
KEEP_LINES = 30
#: ``solver.log`` size past which it is gzipped and reopened between solves.
ROTATE_BYTES = 50 * 1024 * 1024


def show_warnings_once(comm):
    """Python and OpenMDAO warnings: shown once on rank 0, ignored elsewhere.

    One copy per rank flooded the console at 48 ranks. OpenMDAO sets its own
    ``always`` filters at import, so this must run after that import and
    before the problem is built. An invalid ``check_partials`` option stays an
    error on every rank, so ranks do not diverge. OpenMDAO's derivative mode
    warning compares row counts, which do not measure cost here: geometry rows
    never reach SU2 and a gradient takes one adjoint per functional.
    """
    from openmdao.utils.om_warnings import (DerivativesWarning,
                                            OMInvalidCheckDerivativesOptionsWarning)
    warnings.simplefilter('once' if comm is None or comm.rank == 0 else 'ignore')
    warnings.filterwarnings('error',
                            category=OMInvalidCheckDerivativesOptionsWarning)
    warnings.filterwarnings('ignore', message='Inefficient choice of derivative mode',
                            category=DerivativesWarning)


class _Tee:
    """A text stream writing to several streams; ``fileno`` is the first's."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def fileno(self):
        return self.streams[0].fileno()

    def isatty(self):
        return self.streams[0].isatty()

    def writable(self):
        return True

    @property
    def encoding(self):
        return getattr(self.streams[0], 'encoding', 'utf-8')


def _flush_c_stdout():
    """Flush C stdio so a header line follows the previous solve's output."""
    try:
        ctypes.CDLL(None).fflush(None)
    except (OSError, AttributeError):
        pass


class SolverConsole:
    """Route SU2 stdout to the run's logs; keep Python output on the terminal."""

    def __init__(self, comm, logdir=None, quiet=True, solver_stdout='log',
                 **_ignored):
        self.comm = comm
        # quiet=False leaves rank 0's SU2 output on the terminal (diagnosis only).
        self.quiet = bool(quiet)
        self.logdir = logdir or os.path.join(REPO, 'logs')
        self.solver_stdout = solver_stdout
        self.terminal = None
        self._raw_terminal = None
        self._progress = None
        self._saved_fd = None
        self._saved_stdout = None
        self._fh = None
        self._solve_start = None
        self._rotations = 0

    @property
    def rank(self):
        return 0 if self.comm is None else self.comm.rank

    @property
    def last_output(self):
        return os.path.join(self.logdir, 'solver_last.out')

    @property
    def solver_log(self):
        return os.path.join(self.logdir, 'solver.log')

    @property
    def progress_log(self):
        return os.path.join(self.logdir, 'progress.log')

    def truncate_last_output(self):
        if (self.rank == 0 and self.solver_stdout == 'discard' and self.quiet
                and self._saved_fd is not None):
            _flush_c_stdout()
            os.ftruncate(1, 0)
            os.lseek(1, 0, os.SEEK_SET)

    @property
    def _logging(self):
        return (self.rank == 0 and self._saved_fd is not None and self.quiet
                and self.solver_stdout == 'log')

    def solver_note(self, line):
        """Write ``line`` into the solver output, between two solves."""
        if self.rank != 0 or self._saved_fd is None or not self.quiet:
            return
        _flush_c_stdout()
        os.write(1, f'{line}\n'.encode())

    def solve_started(self, header):
        """Write a solve's header line; the solve's output follows it."""
        if self._logging:
            _flush_c_stdout()
            if os.fstat(1).st_size > ROTATE_BYTES:
                self._rotate()
        self.solver_note(header)
        self._solve_start = os.fstat(1).st_size if self._logging else None

    def solve_finished(self, ok):
        """Keep the last ``KEEP_LINES`` lines of a successful solve, all of a failed one."""
        start, self._solve_start = self._solve_start, None
        if not self._logging or start is None or not ok:
            return
        _flush_c_stdout()
        with open(self.solver_log, 'rb') as stream:
            stream.seek(start)
            tail = deque(stream, maxlen=KEEP_LINES)
        os.ftruncate(1, start)
        if tail:
            os.write(1, b''.join(tail))

    def _rotate(self):
        """Gzip ``solver.log`` to ``solver.<NNN>.log.gz`` and open a new one."""
        self._rotations += 1
        stem = os.path.join(self.logdir, f'solver.{self._rotations:03d}.log')
        os.rename(self.solver_log, stem)
        fh = open(self.solver_log, 'a')
        os.dup2(fh.fileno(), 1)
        self._fh.close()
        self._fh = fh
        # Append: a resumed run restarts the count and must not overwrite.
        with open(stem, 'rb') as source, gzip.open(f'{stem}.gz', 'ab') as target:
            shutil.copyfileobj(source, target)
        os.remove(stem)

    @contextmanager
    def solver_output(self):
        """Send Python's ``print`` to the solver output (library text such as prefoil's)."""
        if self.rank != 0 or self._saved_fd is None or not self.quiet:
            yield
            return
        saved = sys.stdout
        saved.flush()
        stream = open(1, 'w', closefd=False)
        sys.stdout = stream
        try:
            yield
        finally:
            stream.flush()
            sys.stdout = saved

    def __enter__(self):
        os.makedirs(self.logdir, exist_ok=True)
        sys.stdout.flush()
        self._saved_fd = os.dup(1)
        if self.rank == 0:
            path = self.solver_log if self.solver_stdout == 'log' else self.last_output
            self._fh = open(path, 'a' if self.solver_stdout == 'log' else 'w')
            self._raw_terminal = os.fdopen(os.dup(self._saved_fd), 'w')
            self._progress = open(self.progress_log, 'a')
            self.terminal = _Tee(self._raw_terminal, self._progress)
            self._saved_stdout = sys.stdout
            sys.stdout = self.terminal
        else:
            self._fh = open(os.devnull, 'w')
        if self.quiet or self.rank != 0:
            os.dup2(self._fh.fileno(), 1)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # A gate that printed its own report (GateError.reported) needs no
        # traceback on every rank; its report already says what went wrong.
        reported = getattr(exc_val, 'reported', False)
        if exc_type is not None and not reported:
            import traceback
            traceback.print_exception(exc_type, exc_val, exc_tb)
        if exc_type is not None and self.rank == 0 and not reported:
            line = f'[rank {self.rank}] {exc_type.__name__}: {exc_val}'
            print(line, file=self.terminal, flush=True)
        if self._saved_stdout is not None:
            self.terminal.flush()
            sys.stdout = self._saved_stdout
        else:
            sys.stdout.flush()
        _flush_c_stdout()
        os.dup2(self._saved_fd, 1)
        os.close(self._saved_fd)
        self._saved_fd = None
        self._fh.close()
        if self._progress is not None:
            self._progress.close()
            self._raw_terminal.close()
        self.terminal = None
        return False
