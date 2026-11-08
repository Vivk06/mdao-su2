#!/usr/bin/env bash
# Activate the SU2 MDO environment. Source before any run/test: source activate.sh

# Must be sourced, not executed — exports would be lost in a child shell.
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    echo "ERROR: activate.sh must be SOURCED, not executed — nothing would persist."
    echo "       Use:  source ${BASH_SOURCE[0]}"
    exit 1
fi

if [ -n "${CONDA_PREFIX:-}" ]; then
    echo "ERROR: An active conda environment ($CONDA_PREFIX) is not supported: its compilers, flags and MPI leak into the builds. Run 'conda deactivate' (until no environment is active) and re-run." >&2
    return 1 2>/dev/null || exit 1
fi

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Source shared library ────────────────────────────────────────────
source "$REPO/shell_helpers/env_common.sh"

# A failed source leaves the helpers undefined; continuing from here would
# half-configure the shell and still print "environment activated".
if [ "${_ENV_COMMON_LOADED:-}" != "1" ]; then
    echo "ERROR: failed to load $REPO/shell_helpers/env_common.sh (see the errors above)."
    echo "       The environment has NOT been activated."
    return 1 2>/dev/null || exit 1
fi

# Idempotent PATH helper — removes existing entry before prepending.
_prepend_path() {
    local var="$1" dir="$2" cur out entry
    cur=":${!var:-}:"
    out=""
    while IFS= read -r entry || [ -n "$entry" ]; do
        [ -z "$entry" ] && continue
        [ "$entry" = "$dir" ] && continue
        out="${out:+$out:}$entry"
    done < <(printf '%s' "${cur:1:${#cur}-2}" | tr ':' '\n')
    export "$var=$dir${out:+:$out}"
}

# ── Apply it: read .env_local, resolve prefixes, export paths, activate venv ──
# Saved paths from .env_local (written by setup.sh). They override values the
# terminal already exports: activation applies exactly what setup.sh resolved.
load_env_file "$REPO/.env_local" --file-wins
# An explicit venv selection is useful for isolated installation trials.
[ -z "${MDAO_SU2_VENV_DIR:-}" ] || VENV_DIR="$MDAO_SU2_VENV_DIR"

# Defaults derived from SHAPEOPT_ROOT
: "${SHAPEOPT_ROOT:=$(cd "$REPO/.." && pwd)}"
: "${SU2_PREFIX:=$SHAPEOPT_ROOT/SU2v8.5/Binaries}"
: "${OPENMPI_PREFIX:=$SHAPEOPT_ROOT/openmpi}"
: "${VENV_DIR:=$SHAPEOPT_ROOT/venvs/mdo}"
export SHAPEOPT_ROOT SU2_PREFIX OPENMPI_PREFIX VENV_DIR

# Validate
if [ ! -d "$SU2_PREFIX/bin" ]; then
    echo "ERROR: SU2_PREFIX=$SU2_PREFIX/bin does not exist."
    echo "       The environment has NOT been activated."
    return 1 2>/dev/null || exit 1
fi
if [ ! -d "$OPENMPI_PREFIX/bin" ]; then
    echo "ERROR: OPENMPI_PREFIX=$OPENMPI_PREFIX/bin does not exist."
    echo "       The environment has NOT been activated."
    return 1 2>/dev/null || exit 1
fi
# -I ignores PYTHONPATH/PYTHONHOME, so this tests the venv itself.
if ! "$VENV_DIR/bin/python" -I -c '' &>/dev/null; then
    echo "ERROR: no working venv interpreter at $VENV_DIR/bin/python. Run setup.sh first."
    echo "       The environment has NOT been activated."
    return 1 2>/dev/null || exit 1
fi

# The checks passed; PATH, library and Python settings change from here on.

# Deactivate any venv the terminal session may have restored.
if [ -n "${VIRTUAL_ENV:-}" ] && type deactivate &>/dev/null; then
    deactivate
fi

# Drop library directories that replace system libraries (a bundled OpenSSL,
# libgfortran) or add another MPI; see filter_ld_library_path.
filter_ld_library_path "$SHAPEOPT_ROOT" "$OPENMPI_PREFIX" "${IPOPT_PREFIX:-}" "$SU2_PREFIX"
_ld_dropped=${#LD_LIBRARY_PATH_DROPPED[@]}
unset LD_LIBRARY_PATH_DROPPED

# Python imports only from the venv, src/ and $SU2_RUN. Every other PYTHONPATH
# entry, PYTHONHOME, and python/pip aliases from shell startup files are removed
# for this terminal (see filter_python_path, drop_python_shims).
filter_python_path "$REPO/src" "$SU2_PREFIX/bin"
drop_python_shims
_py_dropped=$(( PYTHON_PATH_DROPPED + PYTHON_SHIMS_DROPPED ))
unset PYTHON_PATH_DROPPED PYTHON_SHIMS_DROPPED

# Open MPI (lib64 on some distributions)
_ompi_lib="$OPENMPI_PREFIX/lib"
ls "$OPENMPI_PREFIX"/lib64/libmpi.so* &>/dev/null && _ompi_lib="$OPENMPI_PREFIX/lib64"
_prepend_path PATH "$OPENMPI_PREFIX/bin"
_prepend_path LD_LIBRARY_PATH "$_ompi_lib"
unset _ompi_lib
# mpirun uses OPAL/PRTE_PREFIX (not PATH) for MCA/PRRTE — foreign exports break it.
export OPAL_PREFIX="$OPENMPI_PREFIX"
export PRTE_PREFIX="$OPENMPI_PREFIX"

# SU2
export SU2_RUN="$SU2_PREFIX/bin"
export SU2_HOME="$SU2_PREFIX"
_prepend_path PATH "$SU2_RUN"

# IPOPT (optional)
if [ -n "${IPOPT_PREFIX:-}" ] && [ -d "$IPOPT_PREFIX/lib" ]; then
    _prepend_path PATH "$IPOPT_PREFIX/bin"
    _prepend_path LD_LIBRARY_PATH "$IPOPT_PREFIX/lib"
    _prepend_path PKG_CONFIG_PATH "$IPOPT_PREFIX/lib/pkgconfig"
    export IPOPT_LIBRARY_PATH="$IPOPT_PREFIX/lib"
fi

# Python venv (checked above)
source "$VENV_DIR/bin/activate"

# mdao_su2 is imported from the source tree, not installed; pysu2 from $SU2_RUN.
export PYTHONPATH="$REPO/src:$SU2_RUN"
# Tells setup.sh that this terminal's prefixes came from .env_local, not the user.
export MDAO_SU2_ACTIVATED_REPO="$REPO"

# Matplotlib's default user cache may be read-only in managed environments.
# Keep its writable cache with the other gitignored test baselines.
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/mdao-su2-matplotlib}"
mkdir -p "$MPLCONFIGDIR"

# Path checks — mpirun must resolve inside $OPENMPI_PREFIX.
_mpirun="$(command -v mpirun 2>/dev/null || true)"
if [ -z "$_mpirun" ]; then
    warn "mpirun not found on PATH."
elif [ "${_mpirun#$OPENMPI_PREFIX/}" = "$_mpirun" ]; then
    warn "mpirun resolves to $_mpirun, OUTSIDE \$OPENMPI_PREFIX ($OPENMPI_PREFIX)."
    warn "Open a fresh shell and source activate.sh before any MPI run."
fi

# python3 must be the venv's interpreter.
if [ "$(command -v python3)" != "$VENV_DIR/bin/python3" ]; then
    warn "python3 resolves to $(command -v python3), not the venv's $VENV_DIR/bin/python3."
fi

# mdao_su2 imports from $REPO/src
if ! python3 -c "import mdao_su2" &>/dev/null; then
    warn "mdao_su2 not importable from $REPO/src."
fi

# Warn when any installed binding or SU2 wrapper resolves a foreign libmpi.
_foreign_mpi=""
for _lib in "$VENV_DIR"/lib/python*/site-packages/mpi4py/MPI*.so \
            "$VENV_DIR"/lib/python*/site-packages/petsc/lib/libpetsc.so* \
            "$VENV_DIR"/lib/python*/site-packages/petsc4py/lib/PETSc*.so \
            "$SU2_RUN"/_pysu2*.so; do
    [ -f "$_lib" ] || continue
    _mpi_so=$(env -u LD_LIBRARY_PATH ldd "$_lib" 2>/dev/null | awk '/libmpi\.so/ && /=>/ {print $3; exit}' || true)
    [ -n "$_mpi_so" ] || continue
    case "$(realpath -m "$_mpi_so")" in
        "$(realpath "$OPENMPI_PREFIX")"/*) ;;
        *) _foreign_mpi="${_foreign_mpi:+$_foreign_mpi, }${_lib##*/} -> $_mpi_so" ;;
    esac
done
if [ -n "$_foreign_mpi" ]; then
    warn "Foreign libmpi linked: $_foreign_mpi"
    warn "Ensure \$OPENMPI_PREFIX/lib ($OPENMPI_PREFIX/lib) is first on LD_LIBRARY_PATH."
fi
unset _mpirun _foreign_mpi _lib _mpi_so

# ── Summary ──────────────────────────────────────────────────────────
echo "SU2 MDO environment activated"
echo "  REPO:           $REPO"
echo "  SHAPEOPT_ROOT:  ${SHAPEOPT_ROOT:-<not set>}"
echo "  SU2_PREFIX:     ${SU2_PREFIX:-<not set>}"
echo "  SU2_RUN:        ${SU2_RUN:-<not set>}"
echo "  OPENMPI_PREFIX: ${OPENMPI_PREFIX:-<not set>}"
echo "  IPOPT_PREFIX:   ${IPOPT_PREFIX:-<not set>}"
echo "  VENV:           $VENV_DIR"
echo "  Python:         $(python3 --version 2>/dev/null) at $(command -v python3 2>/dev/null)"
echo "  PYTHONPATH:     $PYTHONPATH"
[ "$_py_dropped" -eq 0 ] || \
echo "                  (previous Python paths removed for this terminal; only these are used)"
[ "$_ld_dropped" -eq 0 ] || \
echo "  LD_LIBRARY_PATH: inherited entries removed ($_ld_dropped: another MPI or replaced system libraries)"
echo "  mpirun:         $(command -v mpirun 2>/dev/null || echo '<not found>')"
unset -f _prepend_path
unset _py_dropped _ld_dropped
