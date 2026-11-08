#!/usr/bin/env bash
# One-time environment setup. Default: --with-python only; native builds opt-in.
set -euo pipefail

# The environment this script was started with, unchanged. The activation gate
# at the end replays it, so it sees exactly what a new terminal sees.
mapfile -d '' -t _ORIG_ENV < <(env -0)

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "$REPO/shell_helpers/env_common.sh"
source "$REPO/shell_helpers/preflight.sh"

# Started from a terminal with an active venv (e.g. after `source activate.sh`):
# its python3.14 would come first on PATH and build the new venv from it.
if [ -n "${VIRTUAL_ENV:-}" ]; then
    _p=""
    while IFS= read -r -d: _e; do
        [ "$_e" = "$VIRTUAL_ENV/bin" ] || _p="${_p:+$_p:}$_e"
    done < <(printf '%s:' "$PATH")
    export PATH="$_p"
    unset VIRTUAL_ENV _p _e
fi

# Precedence: --cli > environment > .env_local > derived from SHAPEOPT_ROOT.
_PATH_VARS=(SHAPEOPT_ROOT VENV_DIR OPENMPI_PREFIX SU2_PREFIX SU2_SRC SU2_HOME IPOPT_PREFIX USER_PACKAGES_DIR)
declare -A PATH_SRC=()
for _v in "${_PATH_VARS[@]}"; do
    [ -n "${!_v+x}" ] && PATH_SRC[$_v]="env"
done

WITH_OPENMPI=0; WITH_SU2=0; WITH_IPOPT=0; WITH_PYTHON=1
FORCE_OPENMPI=0; FORCE_SU2=0; FORCE_IPOPT=0; FORCE_PYTHON=0
RECREATE_VENV=0
CLEAN=0
_DRY_RUN=0
SKIP_VERIFY=0
BINDINGS_ONLY=0
JOBS_MAX=8
JOBS=$JOBS_MAX
_ROOT_EXPLICIT=0
# Precedence: --python > environment > .env_local > python3.14 (set after .env_local).
PYTHON_EXE="${PYTHON_EXE:-}"
# ParaView's pvpython for run/plot.py's paraview_view; saved in .env_local, which
# activate.sh exports. No default path.
PARAVIEW_ARG=""
OPENMPI_VERSION="${OPENMPI_VERSION:-5.0.10}"
SU2_REF="${SU2_REF:-v8.5.0}"
# Python packages pinned to the versions the verification lanes passed with
# (2026-10-03). OpenMDAO's finite-difference order and relevance, which the
# progress console's FD plan follows, were measured on 3.45.0.
NUMPY_PIN="numpy==2.5.4"
CORE_PINS=("$NUMPY_PIN" "scipy==1.18.1" "matplotlib==3.11.1")
OPENMDAO_PIN="openmdao==3.45.0"
MPHYS_PIN="mphys==2.0.0"
GMSH_PIN="gmsh==4.15.2"
PETSC_VERSION="3.25.4"
VERBOSE=0
YES=0

usage() {
    cat <<'EOF'
setup.sh [OPTIONS]

COMPONENTS (default: --with-python only)
  --with-openmpi  --with-su2  --with-ipopt  --with-python  --all
  --no-<component>                     overrides --all

REBUILD
  --force[-<component>]                rebuild in place
  --recreate-venv                      [DESTRUCTIVE]
  --yes

CLEANUP
  --clean                              list setup-created paths and sizes
  --clean --yes                        delete listed paths and this root's pip cache

PATHS/VERSIONS
  --root DIR [checkout parent]         --venv DIR      --python EXE [saved, else python3.14]
  --paraview PVPYTHON                  save ParaView's pvpython as PARAVIEW_PVPYTHON in .env_local
  --su2-ref REF [v8.5.0]              --openmpi-version V [5.0.10]

TUNING
  --jobs N [8]                         hard cap 8; never scaled from nproc

OUTPUT
  --dry-run   --skip-verify   -q/-v   -h
  --bindings-only                     install/check MPI bindings, then stop
EOF
    exit "${1:-0}"
}

while [ $# -gt 0 ]; do
    case "$1" in
        --with-openmpi)   WITH_OPENMPI=1 ;;
        --with-su2)       WITH_SU2=1 ;;
        --with-ipopt)     WITH_IPOPT=1 ;;
        --with-python)    WITH_PYTHON=1 ;;
        --no-openmpi)     WITH_OPENMPI=0 ;;
        --no-su2)         WITH_SU2=0 ;;
        --no-ipopt)       WITH_IPOPT=0 ;;
        --no-python)      WITH_PYTHON=0 ;;
        --all)            WITH_OPENMPI=1; WITH_SU2=1; WITH_IPOPT=1; WITH_PYTHON=1 ;;
        --force)          FORCE_OPENMPI=1; FORCE_SU2=1; FORCE_IPOPT=1; FORCE_PYTHON=1 ;;
        --force-openmpi)  FORCE_OPENMPI=1; WITH_OPENMPI=1 ;;
        --force-su2)      FORCE_SU2=1; WITH_SU2=1 ;;
        --force-ipopt)    FORCE_IPOPT=1; WITH_IPOPT=1 ;;
        --force-python)   FORCE_PYTHON=1; WITH_PYTHON=1 ;;
        --recreate-venv)  RECREATE_VENV=1 ;;
        --clean)          CLEAN=1 ;;
        --yes)            YES=1 ;;
        --root)           shift; SHAPEOPT_ROOT="$1"; PATH_SRC[SHAPEOPT_ROOT]="cli"; _ROOT_EXPLICIT=1 ;;
        --venv)           shift; VENV_DIR="$1"; PATH_SRC[VENV_DIR]="cli" ;;
        --python)         shift; PYTHON_EXE="$1" ;;
        --paraview)       shift; PARAVIEW_ARG="$1" ;;
        --su2-ref)        shift; SU2_REF="$1" ;;
        --openmpi-version) shift; OPENMPI_VERSION="$1" ;;
        --jobs)           shift; JOBS="$1" ;;
        --dry-run)        _DRY_RUN=1 ;;
        --skip-verify)    SKIP_VERIFY=1 ;;
        --bindings-only) BINDINGS_ONLY=1 ;;
        -v)               VERBOSE=1 ;;
        -q)               VERBOSE=0 ;;
        -h|--help)        usage 0 ;;
        *)                echo "Unknown option: $1"; usage 1 ;;
    esac
    shift
done

export _DRY_RUN

# Hard cap on build parallelism.
if ! [[ "$JOBS" =~ ^[0-9]+$ ]] || [ "$JOBS" -lt 1 ]; then
    warn "--jobs must be a positive integer; using $JOBS_MAX"
    JOBS=$JOBS_MAX
fi
[ "$JOBS" -gt "$JOBS_MAX" ] && JOBS=$JOBS_MAX

# Load saved paths; re-passing the same --root must not discard them.
parse_env_file "$REPO/.env_local"
_FILE_ROOT="${ENV_FILE_VALUES[SHAPEOPT_ROOT]:-}"

# A terminal that ran `source activate.sh` exports the .env_local values. They
# are not user overrides: a new --root must still re-derive them. A prefix the
# user exports in a terminal that was never activated stays an override.
if [ "${MDAO_SU2_ACTIVATED_REPO:-}" = "$REPO" ]; then
    for _v in "${_PATH_VARS[@]}"; do
        if [ "${PATH_SRC[$_v]:-}" = "env" ] && [ -n "${ENV_FILE_VALUES[$_v]+x}" ] \
                && [ "${!_v}" = "${ENV_FILE_VALUES[$_v]}" ]; then
            unset "PATH_SRC[$_v]"
        fi
    done
fi

load_env_file "$REPO/.env_local"

for _v in "${_PATH_VARS[@]}"; do
    [ -n "${!_v+x}" ] && [ -z "${PATH_SRC[$_v]:-}" ] && PATH_SRC[$_v]=".env_local"
done

# --root with a new root: clear .env_local-sourced paths so they re-derive.
if [ "$_ROOT_EXPLICIT" -eq 1 ] && [ "$SHAPEOPT_ROOT" != "$_FILE_ROOT" ]; then
    for _v in VENV_DIR OPENMPI_PREFIX SU2_PREFIX SU2_SRC SU2_HOME IPOPT_PREFIX USER_PACKAGES_DIR; do
        case "${PATH_SRC[$_v]:-}" in
            cli|env) ;;
            *)       unset "$_v" 2>/dev/null || true; unset "PATH_SRC[$_v]" 2>/dev/null || true ;;
        esac
    done
fi

# ── Resolve paths ──
: "${SHAPEOPT_ROOT:=$(cd "$REPO/.." && pwd)}"
: "${VENV_DIR:=$SHAPEOPT_ROOT/venvs/mdo}"
: "${OPENMPI_PREFIX:=$SHAPEOPT_ROOT/openmpi}"
: "${SU2_PREFIX:=$SHAPEOPT_ROOT/SU2v8.5/Binaries}"
: "${SU2_SRC:=$SHAPEOPT_ROOT/SU2v8.5/SU2}"
: "${SU2_HOME:=$SU2_PREFIX}"
: "${IPOPT_PREFIX:=$SHAPEOPT_ROOT/coin-or/Optimise}"
: "${USER_PACKAGES_DIR:=$SHAPEOPT_ROOT/User_Packages}"
: "${PYTHON_EXE:=python3.14}"
# Generated dependencies belong outside the checkout, including when a saved
# path or explicit override is used. Keep .env_local as the sole repo-local
# setup artifact.
_repo_real="$(realpath -m -- "$REPO")"
for _v in SHAPEOPT_ROOT VENV_DIR OPENMPI_PREFIX SU2_PREFIX SU2_SRC SU2_HOME IPOPT_PREFIX USER_PACKAGES_DIR; do
    _path_real="$(realpath -m -- "${!_v}")"
    case "$_path_real" in
        "$_repo_real"|"$_repo_real"/*)
            die "$_v=${!_v} resolves inside the checkout. Select a path outside $REPO."
            ;;
    esac
done
_root_real="$(realpath -m -- "$SHAPEOPT_ROOT")"
for _path in "$SHAPEOPT_ROOT/logs" "$SHAPEOPT_ROOT/tmp" "$SHAPEOPT_ROOT/coin-or" "$SHAPEOPT_ROOT/SU2v8.5"; do
    _path_real="$(realpath -m -- "$_path")"
    case "$_path_real" in
        "$_root_real"/*) ;;
        *) die "Generated path $_path resolves outside the selected root $SHAPEOPT_ROOT." ;;
    esac
    case "$_path_real" in
        "$_repo_real"|"$_repo_real"/*)
            die "Generated path $_path resolves inside the checkout $REPO."
            ;;
    esac
done
unset _repo_real _root_real _path_real _path _v
if [ -n "$PARAVIEW_ARG" ]; then
    _pv="$(command -v "$PARAVIEW_ARG" 2>/dev/null || true)"
    [ -n "$_pv" ] && [ -x "$_pv" ] || die "--paraview $PARAVIEW_ARG is not an executable pvpython."
    PARAVIEW_PVPYTHON="$(readlink -f "$_pv")"
    unset _pv
fi
PYTHON_EXE="$(command -v "$PYTHON_EXE" 2>/dev/null || echo "$PYTHON_EXE")"
# A pyenv shim is a script that picks an interpreter per directory; build from,
# and save, the interpreter it resolves to now.
if [[ "$PYTHON_EXE" == */shims/* ]] && command -v pyenv &>/dev/null; then
    _real="$(pyenv which "$(basename "$PYTHON_EXE")" 2>/dev/null || true)"
    [ -n "$_real" ] || die "pyenv shim $PYTHON_EXE resolves to no interpreter here. Select one (pyenv shell/local) or pass --python /path/to/python3.14."
    PYTHON_EXE="$_real"
    unset _real
fi

# Conda is not supported: an active environment puts its own compilers, flags
# and MPI first on PATH, and the native builds would mix them with the system
# toolchain.
if [ -n "${CONDA_PREFIX:-}" ]; then
    die "An active conda environment ($CONDA_PREFIX) is not supported: its compilers, flags and MPI leak into the builds. Run 'conda deactivate' (until no environment is active) and re-run."
fi

MANIFEST="$SHAPEOPT_ROOT/logs/mdao-su2-setup-manifest"
PIP_CACHE="$SHAPEOPT_ROOT/tmp/pip-cache"
export PIP_CACHE_DIR="$PIP_CACHE"

# Register only new artifacts. The entry precedes the operation so an
# interrupted download or build still appears in the next clean dry run.
record_new() {
    local path="$1"
    [ "$_DRY_RUN" -eq 0 ] || return 0
    [ ! -e "$path" ] && [ ! -L "$path" ] || return 0
    case "$path" in *$'\n'*) die "Newline in setup path: $path" ;; esac
    mkdir -p "$(dirname "$MANIFEST")"
    touch "$MANIFEST"
    grep -Fxq -- "$path" "$MANIFEST" || printf '%s\n' "$path" >> "$MANIFEST"
}

clean_target_ok() {
    local path="$1" resolved root_real repo_real
    [ ! -L "$path" ] || return 1
    resolved="$(realpath -m -- "$path")"
    root_real="$(realpath -m -- "$SHAPEOPT_ROOT")"
    repo_real="$(realpath -m -- "$REPO")"
    if [ "$path" = "$REPO/.env_local" ]; then
        [ "$resolved" = "$repo_real/.env_local" ]; return
    fi
    case "$resolved" in "$root_real"/*) ;; *) return 1 ;; esac
    case "$resolved" in "$repo_real"|"$repo_real"/*) return 1 ;; esac
    case "$repo_real" in "$resolved"|"$resolved"/*) return 1 ;; esac
    case "$path" in
        "$VENV_DIR"|"$OPENMPI_PREFIX"|"$SU2_PREFIX"|"$SU2_SRC"|\
        "$IPOPT_PREFIX"|"$SU2_SRC/build"|\
        "$PIP_CACHE"|\
        "$USER_PACKAGES_DIR/openmpi-$OPENMPI_VERSION.tar.bz2"|\
        "$USER_PACKAGES_DIR/openmpi-$OPENMPI_VERSION"|\
        "$SHAPEOPT_ROOT/coin-or"|"$SHAPEOPT_ROOT/coin-or/coinbrew"|\
        "$REPO/.env_local") return 0 ;;
    esac
    return 1
}

if [ "$CLEAN" -eq 1 ]; then
    [ ! -L "$MANIFEST" ] || die "Refusing symlinked setup manifest: $MANIFEST"
    targets=()
    if [ -f "$MANIFEST" ]; then
        while IFS= read -r path || [ -n "$path" ]; do
            [ -n "$path" ] || continue
            clean_target_ok "$path" || die "Unsafe manifest path: $path"
            targets+=("$path")
        done < "$MANIFEST"
    fi
    clean_target_ok "$PIP_CACHE" || die "Unsafe pip cache path: $PIP_CACHE"
    targets+=("$PIP_CACHE")
    for path in "${targets[@]}"; do
        if [ -e "$path" ]; then
            printf '%s  %s\n' "$(du -sh -- "$path" | cut -f1)" "$path"
        else
            printf '0  %s (absent)\n' "$path"
        fi
    done
    if [ "$YES" -eq 1 ]; then
        for path in "${targets[@]}"; do rm -rf -- "$path"; done
        [ ! -f "$MANIFEST" ] || rm -f -- "$MANIFEST"
    else
        echo 'Dry run: pass --clean --yes to delete these paths.'
    fi
    exit 0
fi


for _v in "${_PATH_VARS[@]}"; do
    [ -z "${PATH_SRC[$_v]:-}" ] && PATH_SRC[$_v]="derived"
done

# Keep compiler, pip, and source-build temporary files under the selected root.
mkdir -p "$SHAPEOPT_ROOT/tmp"
export TMPDIR="$SHAPEOPT_ROOT/tmp"

# print_prefixes [TITLE] — resolved prefixes and their provenance.
print_prefixes() {
    [ -n "${1:-}" ] && echo "  --- $1 ---"
    local v
    for v in SHAPEOPT_ROOT VENV_DIR OPENMPI_PREFIX SU2_PREFIX IPOPT_PREFIX; do
        printf "  %-16s %s  (%s)\n" "$v:" "${!v}" "${PATH_SRC[$v]:-derived}"
    done
}

# ── Library search path ──
# Every program this script starts (wget, git, compilers, pip builds) loads
# libraries through LD_LIBRARY_PATH; drop entries that replace system libraries
# or add another MPI. The project's own prefixes are added back below.
filter_ld_library_path "$SHAPEOPT_ROOT" "$OPENMPI_PREFIX" "$IPOPT_PREFIX" "$SU2_PREFIX"
_ld_dropped=${#LD_LIBRARY_PATH_DROPPED[@]}
# Another application's PYTHONPATH (e.g. a Python 2.7 site-packages) comes before
# the venv's site-packages and replaces its setuptools, numpy or mpi4py. This
# run uses none of it; activate.sh sets the entries the project needs.
filter_python_path
_py_dropped=$PYTHON_PATH_DROPPED
unset LD_LIBRARY_PATH_DROPPED PYTHON_PATH_DROPPED

echo "=== SU2 MDO Environment Setup ==="
echo "  Repo:            $REPO"
echo "  Python:          $PYTHON_EXE"
print_prefixes
echo "  Components:      python=$(( WITH_PYTHON )) openmpi=$(( WITH_OPENMPI )) su2=$(( WITH_SU2 )) ipopt=$(( WITH_IPOPT ))"
echo "  Jobs:            $JOBS (hard cap $JOBS_MAX)"
echo "  Dry-run:         $(( _DRY_RUN ))"
[ "$_py_dropped" -eq 0 ] || \
echo "  PYTHONPATH:      previous Python paths removed for this run; the venv is used alone"
[ "$_ld_dropped" -eq 0 ] || \
echo "  LD_LIBRARY_PATH: inherited entries removed ($_ld_dropped: another MPI or replaced system libraries)"
echo ""

# ── Preflight ──
echo "=== Preflight checks ==="
preflight_base
[ "$WITH_SU2" -eq 1 ] && preflight_su2
[ "$WITH_IPOPT" -eq 1 ] && preflight_ipopt
if ! preflight_report; then
    exit 2
fi
# SU2's pywrapper embeds Python (meson python.dependency(embed: true)); a
# static-only libpython fails late in the SU2 configure.
if [ "$WITH_SU2" -eq 1 ] && command -v "$PYTHON_EXE" &>/dev/null \
        && ! "$PYTHON_EXE" -c "import sys, sysconfig; sys.exit(not sysconfig.get_config_var('Py_ENABLE_SHARED'))" 2>/dev/null; then
    die "$PYTHON_EXE has no shared libpython, which SU2's Python wrapper needs. With pyenv: PYTHON_CONFIGURE_OPTS=--enable-shared pyenv install <version>, then pass --python with that interpreter."
fi
# Native builds download Open MPI, SU2 and COIN-OR sources over HTTPS.
if [ "$_DRY_RUN" -eq 0 ] && [ $(( WITH_OPENMPI + WITH_SU2 + WITH_IPOPT )) -gt 0 ]; then
    check_https "https://github.com" || die "HTTPS certificate verification fails; see above."
fi

# Validate interpreter early — absent $PYTHON_EXE would die inside `run`.
if [ "$WITH_PYTHON" -eq 1 ]; then
    if ! command -v "$PYTHON_EXE" &>/dev/null; then
        echo ""
        echo "  Python interpreter not found: $PYTHON_EXE"
        echo "  Install it, or point setup at one you have:"
        echo "      bash setup.sh --python /path/to/python3.14 ..."
        echo "  Interpreters on this machine:"
        compgen -c 'python3' 2>/dev/null | grep -xE 'python3(\.[0-9]+)?' | sort -u -V \
            | sed 's/^/      /' | head -10
        die "A Python interpreter is required for --with-python."
    fi
    # venv/ensurepip are separate packages on Debian; without them python -m venv fails.
    if ! "$PYTHON_EXE" -c "import venv, ensurepip" 2>/dev/null; then
        echo ""
        echo "  $PYTHON_EXE cannot create virtual environments (venv/ensurepip missing)."
        _py_tag="$("$PYTHON_EXE" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo 3)"
        echo "  Install them with:"
        echo "      sudo apt-get install -y python${_py_tag}-venv"
        die "python -m venv is unusable."
    fi
    # ABI boundary: mpi4py, petsc4py and SU2 pywrapper are compiled against 3.14.
    _py_ver="$("$PYTHON_EXE" -c 'import sys; print("%d.%d.%d" % sys.version_info[:3])' 2>/dev/null || echo unknown)"
    case "$_py_ver" in
        3.14.4) : ;;
        3.14.*) warn "Python $_py_ver — the stack is built and tested on 3.14.4." ;;
        *)      echo ""
                echo "  Python $_py_ver is not supported; this project requires 3.14.x."
                echo "  mpi4py, petsc4py and SU2's pywrapper are compiled against the"
                echo "  venv interpreter, so a different minor breaks the ABI at import."
                echo "  Point setup at a 3.14 interpreter:"
                echo "      bash setup.sh --python /path/to/python3.14 ..."
                die "Python 3.14.x required (found $_py_ver)."
                ;;
    esac
fi

# logs/ must exist before any component block.
mkdir -p "$SHAPEOPT_ROOT/logs"
# Advisory only.
check_disk_gb 12 "$SHAPEOPT_ROOT" || true
echo "  All prerequisites found."
echo ""

# persist_env — write .env_local.
persist_env() {
    [ "$_DRY_RUN" -eq 0 ] || return 0
    record_new "$REPO/.env_local"
    local resolved_python
    resolved_python="$(readlink -f "$(command -v "$PYTHON_EXE" 2>/dev/null || echo "$PYTHON_EXE")")"
    write_env_file "$REPO/.env_local" \
        "SHAPEOPT_ROOT=\"$SHAPEOPT_ROOT\"" \
        "PYTHON_EXE=\"$resolved_python\"" \
        "VENV_DIR=\"$VENV_DIR\"" \
        "USER_PACKAGES_DIR=\"$USER_PACKAGES_DIR\"" \
        "OPENMPI_PREFIX=\"$OPENMPI_PREFIX\"" \
        "OPENMPI_VERSION=\"$OPENMPI_VERSION\"" \
        "SU2_PREFIX=\"$SU2_PREFIX\"" \
        "SU2_SRC=\"$SU2_SRC\"" \
        "SU2_HOME=\"$SU2_HOME\"" \
        "SU2_REF=\"$SU2_REF\"" \
        "IPOPT_PREFIX=\"$IPOPT_PREFIX\"" \
        ${PARAVIEW_PVPYTHON:+"PARAVIEW_PVPYTHON=\"$PARAVIEW_PVPYTHON\""} \
        "SETUP_TIMESTAMP=\"$(date -Iseconds)\"" \
        "SETUP_REVISION=\"$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown)\""
    echo "  Saved paths to $REPO/.env_local"
}

# Every install goes through the venv's own interpreter, never a `pip` or
# `python3` found on PATH or defined as an alias.
VENV_PY="$VENV_DIR/bin/python"
PIP=("$VENV_PY" -m pip)

# venv_has PKG — is PKG importable by the venv interpreter?
venv_has() {
    [ -x "$VENV_DIR/bin/python" ] || return 1
    "$VENV_DIR/bin/python" -c "import $1" 2>/dev/null
}


# Each predicate sets <COMPONENT>_SKIP_REASON for diagnostics.

openmpi_installed() {
    _OPENMPI_SKIP_REASON=""
    local libdir=""
    local d
    for d in "$OPENMPI_PREFIX/lib" "$OPENMPI_PREFIX/lib64"; do
        if ls "$d"/libmpi.so* &>/dev/null; then libdir="$d"; break; fi
    done
    local f
    for f in mpirun mpicc mpicxx ompi_info; do
        if [ ! -x "$OPENMPI_PREFIX/bin/$f" ]; then
            _OPENMPI_SKIP_REASON="$f not executable in $OPENMPI_PREFIX/bin"
            return 1
        fi
    done
    if [ -z "$libdir" ]; then
        _OPENMPI_SKIP_REASON="libmpi.so not found in $OPENMPI_PREFIX/{lib,lib64}"
        return 1
    fi
    if [ ! -f "$OPENMPI_PREFIX/include/mpi.h" ]; then
        _OPENMPI_SKIP_REASON="mpi.h not found in $OPENMPI_PREFIX/include"
        return 1
    fi
    # Version probe; both tools need libmpi on the loader path.
    local found
    found=$(LD_LIBRARY_PATH="$libdir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
            "$OPENMPI_PREFIX/bin/ompi_info" --parsable 2>/dev/null \
            | sed -n 's/^ompi:version:full://p' | head -1)
    if [ -z "$found" ]; then
        found=$(LD_LIBRARY_PATH="$libdir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
                "$OPENMPI_PREFIX/bin/mpirun" --version 2>/dev/null \
                | sed -n '1s/.*) \([0-9][0-9.]*\).*/\1/p')
    fi
    if [ -z "$found" ]; then
        _OPENMPI_SKIP_REASON="neither ompi_info nor mpirun could report a version (broken install)"
        return 1
    fi
    if [ "$found" != "$OPENMPI_VERSION" ]; then
        _OPENMPI_SKIP_REASON="version mismatch: found $found, wanted $OPENMPI_VERSION"
        return 1
    fi
    _OPENMPI_FOUND_VERSION="$found"
    _OPENMPI_LIBDIR="$libdir"
    return 0
}

su2_python_abi() {
    [ -x "$VENV_DIR/bin/python" ] || return 0
    "$VENV_DIR/bin/python" -c 'import sysconfig; print(sysconfig.get_config_var("SOABI"))' 2>/dev/null || true
}

su2_installed() {
    _SU2_SKIP_REASON=""
    local f
    for f in bin/SU2_CFD bin/SU2_CFD_AD bin/SU2_DOT_AD \
             bin/_pysu2.so bin/_pysu2ad.so bin/pysu2ad.py; do
        if [ ! -e "$SU2_PREFIX/$f" ]; then
            _SU2_SKIP_REASON="missing $f"
            return 1
        fi
    done
    return 0
}

ipopt_installed() {
    _IPOPT_SKIP_REASON=""
    local f
    for f in lib/libipopt.so lib/pkgconfig/ipopt.pc bin/ipopt; do
        if [ ! -e "$IPOPT_PREFIX/$f" ]; then
            _IPOPT_SKIP_REASON="missing $f"
            return 1
        fi
    done

    if ! ls "$IPOPT_PREFIX/lib"/libcoinmumps.so* &>/dev/null && \
       ! ls "$IPOPT_PREFIX/lib"/libcoinhsl.so* &>/dev/null; then
        _IPOPT_SKIP_REASON="no linear solver (libcoinmumps/libcoinhsl) in $IPOPT_PREFIX/lib"
        return 1
    fi
    return 0
}

# ── Open MPI ──

if [ "$WITH_OPENMPI" -eq 1 ]; then
    echo "=== Open MPI ==="
    echo "  OPENMPI_PREFIX:  $OPENMPI_PREFIX  (${PATH_SRC[OPENMPI_PREFIX]:-derived})"
    if openmpi_installed && [ "$FORCE_OPENMPI" -eq 0 ]; then
        echo "  SKIPPED — Open MPI ${_OPENMPI_FOUND_VERSION} verified at $OPENMPI_PREFIX"
    else
        if [ "$FORCE_OPENMPI" -eq 1 ]; then
            echo "  REBUILD — --force-openmpi requested"
        else
            echo "  NOT USABLE — ${_OPENMPI_SKIP_REASON:-unknown}"
            echo "  To adopt an Open MPI $OPENMPI_VERSION that is already on this machine,"
            echo "  re-run with it exported (an explicit prefix survives --root):"
            echo "      export OPENMPI_PREFIX=/path/to/openmpi"
        fi
        echo "  Building Open MPI $OPENMPI_VERSION at $OPENMPI_PREFIX"
        run mkdir -p "$USER_PACKAGES_DIR"
        OMPI_TARBALL="$USER_PACKAGES_DIR/openmpi-${OPENMPI_VERSION}.tar.bz2"
        record_new "$OMPI_TARBALL"
        OMPI_URL="https://download.open-mpi.org/release/open-mpi/v${OPENMPI_VERSION%.*}/openmpi-${OPENMPI_VERSION}.tar.bz2"
        # Reuse a cached tarball only if it is a complete archive.
        if [ -f "$OMPI_TARBALL" ] && tar tjf "$OMPI_TARBALL" &>/dev/null; then
            echo "  Using cached tarball: $OMPI_TARBALL"
        else
            [ -f "$OMPI_TARBALL" ] && echo "  Cached tarball is incomplete — re-downloading."
            echo "  Downloading $OMPI_URL ..."
            # Download to .part; rename only on success.
            if [ "$_DRY_RUN" -eq 0 ]; then
                rm -f "$OMPI_TARBALL" "$OMPI_TARBALL.part"
                if ! wget -q -O "$OMPI_TARBALL.part" "$OMPI_URL" && \
                   ! curl -fsSL -o "$OMPI_TARBALL.part" "$OMPI_URL"; then
                    rm -f "$OMPI_TARBALL.part"
                    die "Failed to download $OMPI_URL"
                fi
                tar tjf "$OMPI_TARBALL.part" &>/dev/null \
                    || { rm -f "$OMPI_TARBALL.part"; die "Downloaded archive is corrupt: $OMPI_URL"; }
                mv -f "$OMPI_TARBALL.part" "$OMPI_TARBALL"
            fi
        fi
        OMPI_BUILD="$USER_PACKAGES_DIR/openmpi-${OPENMPI_VERSION}"
        record_new "$OMPI_BUILD"
        if [ ! -d "$OMPI_BUILD" ]; then
            run tar xjf "$OMPI_TARBALL" -C "$USER_PACKAGES_DIR"
        fi
        echo "  Configuring..."
        run_in "$OMPI_BUILD" ./configure \
            --prefix="$OPENMPI_PREFIX" \
            --with-pmix=internal --with-prrte=internal \
            --with-hwloc=internal --with-libevent=internal \
            2>&1 | tee "$SHAPEOPT_ROOT/logs/openmpi_configure.log" | tail -5
        echo "  Building (jobs=$JOBS)..."
        run_in "$OMPI_BUILD" make -j"$JOBS" \
            2>&1 | tee "$SHAPEOPT_ROOT/logs/openmpi_build.log" | tail -3
        echo "  Installing (serial)..."
        record_new "$OPENMPI_PREFIX"
        run_in "$OMPI_BUILD" make install \
            2>&1 | tee "$SHAPEOPT_ROOT/logs/openmpi_install.log" | tail -3
        echo "  Open MPI $OPENMPI_VERSION installed at $OPENMPI_PREFIX"
    fi
    if [ "$_DRY_RUN" -eq 0 ]; then
        openmpi_installed || die "Open MPI install is not usable: ${_OPENMPI_SKIP_REASON:-unknown}; retaining build inputs."
        [ ! -L "$USER_PACKAGES_DIR" ] || die "Refusing symlinked package directory cleanup: $USER_PACKAGES_DIR"
        OMPI_TARBALL="$USER_PACKAGES_DIR/openmpi-${OPENMPI_VERSION}.tar.bz2"
        OMPI_BUILD="$USER_PACKAGES_DIR/openmpi-${OPENMPI_VERSION}"
        for _artifact in "$OMPI_TARBALL" "$OMPI_BUILD"; do
            clean_target_ok "$_artifact" || die "Refusing unsafe build-artifact cleanup: $_artifact"
            if [ -e "$_artifact" ]; then
                rm -rf -- "$_artifact"
                echo "  Removed temporary build input: $_artifact"
            fi
        done
        if [ -d "$USER_PACKAGES_DIR" ] && rmdir -- "$USER_PACKAGES_DIR" 2>/dev/null; then
            echo "  Removed empty package directory: $USER_PACKAGES_DIR"
        fi
        unset _artifact
    fi
    echo ""
fi

# ── IPOPT ──


if [ "$WITH_IPOPT" -eq 1 ]; then
    echo "=== IPOPT ==="
    echo "  IPOPT_PREFIX:    $IPOPT_PREFIX  (${PATH_SRC[IPOPT_PREFIX]:-derived})"
    if ipopt_installed && [ "$FORCE_IPOPT" -eq 0 ]; then
        echo "  SKIPPED — IPOPT already installed at $IPOPT_PREFIX"
    else
        [ "$FORCE_IPOPT" -eq 1 ] && echo "  REBUILD — --force-ipopt requested" \
                                 || echo "  NOT USABLE — ${_IPOPT_SKIP_REASON:-unknown}"
        echo "  Building IPOPT at $IPOPT_PREFIX"
        record_new "$SHAPEOPT_ROOT/coin-or"
        run mkdir -p "$SHAPEOPT_ROOT/coin-or"
        COINBREW="$SHAPEOPT_ROOT/coin-or/coinbrew"
        record_new "$COINBREW"
        COINBREW_URL="https://raw.githubusercontent.com/coin-or/coinbrew/master/coinbrew"
        # An interrupted or failed download leaves an empty file; fetch again.
        if [ ! -s "$COINBREW" ]; then
            echo "  Downloading $COINBREW_URL ..."
            # Download to .part; rename only on success. Time out instead of hanging.
            if [ "$_DRY_RUN" -eq 0 ]; then
                rm -f "$COINBREW" "$COINBREW.part"
                if ! wget -q --timeout=30 --tries=3 -O "$COINBREW.part" "$COINBREW_URL" && \
                   ! curl -fsSL --connect-timeout 30 --retry 2 -o "$COINBREW.part" "$COINBREW_URL"; then
                    rm -f "$COINBREW.part"
                    die "Failed to download $COINBREW_URL (check network access to raw.githubusercontent.com)"
                fi
                head -c 2 "$COINBREW.part" | grep -q '^#!' \
                    || { rm -f "$COINBREW.part"; die "Downloaded coinbrew is not a script: $COINBREW_URL"; }
                mv -f "$COINBREW.part" "$COINBREW"
            fi
        fi
        run chmod u+x "$COINBREW"
        echo "  Fetching Ipopt..."
        run_in "$SHAPEOPT_ROOT/coin-or" "$COINBREW" fetch Ipopt --no-prompt
        echo "  Building (jobs=$JOBS); full output in $SHAPEOPT_ROOT/logs/ipopt.log ..."
        record_new "$IPOPT_PREFIX"
        run_in "$SHAPEOPT_ROOT/coin-or" "$COINBREW" build Ipopt \
            --prefix="$IPOPT_PREFIX" --no-prompt -j "$JOBS" --test \
            2>&1 | tee "$SHAPEOPT_ROOT/logs/ipopt.log" | tail -5
        echo "  IPOPT installed at $IPOPT_PREFIX"
    fi
    echo ""
fi

# ── Export paths ──
export OPENMPI_PREFIX SU2_PREFIX IPOPT_PREFIX
export PATH="$OPENMPI_PREFIX/bin:$SU2_PREFIX/bin:${IPOPT_PREFIX:+$IPOPT_PREFIX/bin:}$PATH"
export LD_LIBRARY_PATH="${_OPENMPI_LIBDIR:-$OPENMPI_PREFIX/lib}${IPOPT_PREFIX:+:$IPOPT_PREFIX/lib}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PKG_CONFIG_PATH="${IPOPT_PREFIX:+$IPOPT_PREFIX/lib/pkgconfig}${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
# SU2 needs mpi.h in both C and C++ include paths.
export CPLUS_INCLUDE_PATH="$OPENMPI_PREFIX/include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
export C_INCLUDE_PATH="$OPENMPI_PREFIX/include${C_INCLUDE_PATH:+:$C_INCLUDE_PATH}"

# ── Python stack ──

if [ "$WITH_PYTHON" -eq 1 ]; then

    if [ "$_DRY_RUN" -eq 0 ] && ! openmpi_installed; then
        echo ""
        echo "  Unusable OpenMPI at OPENMPI_PREFIX=$OPENMPI_PREFIX (${PATH_SRC[OPENMPI_PREFIX]:-derived})"
        echo "    reason: ${_OPENMPI_SKIP_REASON:-unknown}"
        _root_arg=""
        [ "$_ROOT_EXPLICIT" -eq 1 ] && _root_arg=" --root $SHAPEOPT_ROOT"
        echo "  The Python stack requires a working Open MPI $OPENMPI_VERSION. Either:"
        echo "    - point setup at an existing install (survives --root):"
        echo "        export OPENMPI_PREFIX=/path/to/openmpi"
        echo "        bash setup.sh$_root_arg"
        echo "    - or build it into the root:"
        echo "        bash setup.sh$_root_arg --with-openmpi"
        die "OpenMPI is required for the Python stack."
    fi

    echo "=== Python venv ==="
    echo "  VENV_DIR:        $VENV_DIR  (${PATH_SRC[VENV_DIR]:-derived})"

    if [ "$RECREATE_VENV" -eq 1 ] && [ -d "$VENV_DIR" ]; then
        if [ "$YES" -eq 0 ] && [ "$_DRY_RUN" -eq 0 ]; then
            echo "  --recreate-venv will DELETE $VENV_DIR. Pass --yes to confirm."
            exit 1
        fi
        run rm -rf "$VENV_DIR"
    fi

    if [ -d "$VENV_DIR" ]; then
        # The compiled bindings need the venv to be isolated and on Python 3.14.
        # -I ignores PYTHON* variables, so these test the venv itself.
        _venv_problem=""
        if ! "$VENV_DIR/bin/python" -I -c '' &>/dev/null; then
            _venv_problem="its interpreter does not run (was its base Python removed or upgraded?)"
        elif ! grep -qiE '^[[:space:]]*include-system-site-packages[[:space:]]*=[[:space:]]*false' "$VENV_DIR/pyvenv.cfg" 2>/dev/null; then
            _venv_problem="it includes the system site-packages, which shadow the pinned packages"
        else
            _venv_ver="$("$VENV_DIR/bin/python" -I -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
            [ "$_venv_ver" = "3.14" ] || _venv_problem="it runs Python $_venv_ver, not 3.14"
        fi
        if [ -n "$_venv_problem" ]; then
            _root_arg=""
            [ "$_ROOT_EXPLICIT" -eq 1 ] && _root_arg=" --root $SHAPEOPT_ROOT"
            [ "${PATH_SRC[VENV_DIR]:-}" = "cli" ] && _root_arg+=" --venv $VENV_DIR"
            echo "  The venv at $VENV_DIR cannot be used: $_venv_problem."
            echo "  Recreate it (this deletes $VENV_DIR):"
            echo "      bash setup.sh$_root_arg --recreate-venv --yes"
            die "Unusable venv at $VENV_DIR."
        fi
        echo "  Venv exists — activating."
    else
        record_new "$VENV_DIR"
        run "$PYTHON_EXE" -m venv "$VENV_DIR"
    fi
    if [ "$_DRY_RUN" -eq 0 ]; then
        source "$VENV_DIR/bin/activate"
    fi
    run "${PIP[@]}" install --no-cache-dir -q --upgrade pip setuptools wheel
    echo "  Python: $("$VENV_PY" --version 2>/dev/null) at $VENV_PY"
    echo ""

    # Build the three MPI bindings with this venv and this Open MPI prefix.
    echo "=== MPI & PETSc Python bindings ==="
    echo "  OPENMPI_PREFIX:  $OPENMPI_PREFIX  (${PATH_SRC[OPENMPI_PREFIX]:-derived})"
    echo "  Interpreter:     $VENV_PY"
    _MPI_LIBDIR="${_OPENMPI_LIBDIR:-$OPENMPI_PREFIX/lib}"
    [ -x "$OPENMPI_PREFIX/bin/mpicc" ] || die "No mpicc at $OPENMPI_PREFIX/bin/mpicc"
    [ -x "$OPENMPI_PREFIX/bin/mpicxx" ] || die "No mpicxx at $OPENMPI_PREFIX/bin/mpicxx"
    [ -x "$OPENMPI_PREFIX/bin/mpifort" ] || die "No mpifort at $OPENMPI_PREFIX/bin/mpifort"
    export PATH="$OPENMPI_PREFIX/bin:$PATH"
    [ "$(command -v mpicc)" = "$OPENMPI_PREFIX/bin/mpicc" ] \
        || die "mpicc does not resolve to $OPENMPI_PREFIX/bin/mpicc"
    export LD_LIBRARY_PATH="$_MPI_LIBDIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export OPAL_PREFIX="$OPENMPI_PREFIX" PRTE_PREFIX="$OPENMPI_PREFIX"
    export MPICC="$OPENMPI_PREFIX/bin/mpicc"
    export MPICXX="$OPENMPI_PREFIX/bin/mpicxx"
    export MPIFORT="$OPENMPI_PREFIX/bin/mpifort"
    export MPI4PY_BUILD_MPICC="$MPICC"
    unset PETSC_CONFIGURE_OPTIONS
    export CFLAGS="-I$OPENMPI_PREFIX/include"
    export LDFLAGS="-L$_MPI_LIBDIR -Wl,-rpath,$_MPI_LIBDIR"
    unset PETSC_DIR PETSC_ARCH PYTHONHOME

    binding_version() {
        "$VENV_PY" -I -c 'from importlib.metadata import version; import sys; print(version(sys.argv[1]))' "$1" 2>/dev/null || true
    }
    _mpi_ver="$(binding_version mpi4py)"
    _petsc_ver="$(binding_version petsc)"
    _petsc4py_ver="$(binding_version petsc4py)"
    _rebuild_bindings=$FORCE_PYTHON
    if [ "$_DRY_RUN" -eq 0 ] && [ -n "$_mpi_ver$_petsc_ver$_petsc4py_ver" ]; then
        if ! OPENMPI_PREFIX="$OPENMPI_PREFIX" SU2_RUN="$SU2_PREFIX/bin" \
                "$VENV_PY" "$REPO/tools/check_env.py" --abi-only; then
            _rebuild_bindings=1
            echo "  Existing binding ABI is wrong; rebuilding against $OPENMPI_PREFIX."
        fi
    fi
    _force_arg=()
    [ "$_rebuild_bindings" -eq 0 ] || _force_arg=(--force-reinstall)
    if [ "$_mpi_ver" != 4.1.2 ] || [ "$_rebuild_bindings" -eq 1 ]; then
        echo "  Installing mpi4py 4.1.2 from source..."
        run "${PIP[@]}" install --no-cache-dir --no-binary=mpi4py "${_force_arg[@]}" "mpi4py==4.1.2"
    else
        echo "  mpi4py $_mpi_ver already installed."
    fi
    _petsc_changed=0
    if [ "$_petsc_ver" != "$PETSC_VERSION" ] || [ "$_rebuild_bindings" -eq 1 ]; then
        _petsc_changed=1
        echo "  Installing PETSc $PETSC_VERSION for $OPENMPI_PREFIX..."
        run "${PIP[@]}" install --no-cache-dir "${_force_arg[@]}" "petsc==$PETSC_VERSION"
    else
        echo "  PETSc $_petsc_ver already installed."
    fi
    if [ "$_DRY_RUN" -eq 0 ]; then
        export PETSC_DIR="$("$VENV_PY" -I -c 'import petsc; print(petsc.__path__[0])')"
        [ -d "$PETSC_DIR" ] || die "PETSc directory missing: $PETSC_DIR"
    fi
    if [ "$_petsc4py_ver" != "$PETSC_VERSION" ] || [ "$_petsc_changed" -eq 1 ] || [ "$_rebuild_bindings" -eq 1 ]; then
        echo "  Installing petsc4py $PETSC_VERSION..."
        run "${PIP[@]}" install --no-cache-dir 'cython<3.1' "$NUMPY_PIN"
        _petsc4py_force=()
        [ "$_petsc_changed" -eq 0 ] && [ "$_rebuild_bindings" -eq 0 ] || _petsc4py_force=(--force-reinstall)
        run "${PIP[@]}" install --no-cache-dir --no-build-isolation --no-deps "${_petsc4py_force[@]}" "petsc4py==$PETSC_VERSION"
    else
        echo "  petsc4py $_petsc4py_ver already installed."
    fi
    if [ "$_DRY_RUN" -eq 0 ]; then
        OPENMPI_PREFIX="$OPENMPI_PREFIX" SU2_RUN="$SU2_PREFIX/bin" \
            "$VENV_PY" "$REPO/tools/check_env.py" --abi-only \
            || die "MPI ABI mismatch after binding installation."
    fi
    echo ""
    # Four-layer mpi4py assertion via the venv interpreter.
    if [ "$_DRY_RUN" -eq 0 ] && venv_has mpi4py; then
        echo "  Verifying mpi4py binding..."
        OPENMPI_VERSION="$OPENMPI_VERSION" SU2_RUN="$SU2_PREFIX/bin" "$VENV_DIR/bin/python" -c "
import sys, os, subprocess, re

# 1. mpi4py.__file__ is under sys.prefix (catches the no-op)
import mpi4py
if not mpi4py.__file__.startswith(sys.prefix):
    print(f'  FAIL: mpi4py at {mpi4py.__file__} is outside venv {sys.prefix}')
    print(f'  This is caused by a foreign PYTHONPATH entry shadowing the venv copy.')
    print(f'  Check: echo \$PYTHONPATH')
    print(f'  Fix: remove entries containing other Python versions from PYTHONPATH,')
    print(f'       then re-run setup.sh --with-python')
    sys.exit(1)
print('  [1/4] mpi4py.__file__ is inside the venv')

# 2. Exact version check, against the version this setup targets
from mpi4py import MPI
want = os.environ['OPENMPI_VERSION']
lib_ver = MPI.Get_library_version()
if want not in lib_ver:
    print(f'  FAIL: mpi4py is linked to {lib_ver.splitlines()[0]}, not Open MPI {want}')
    print('  Re-run setup.sh with the selected venv and Open MPI prefix.')
    sys.exit(1)
print(f'  [2/4] MPI library: {lib_ver.splitlines()[0]}')

# 3. ldd with IPOPT_LIBRARY_PATH cleared resolves inside OPENMPI_PREFIX
mpi_so = os.path.join(os.path.dirname(mpi4py.__file__), 'MPI.cpython*.so')
import glob
mpi_sos = glob.glob(mpi_so)
assert mpi_sos, f'Cannot find MPI.*.so at {mpi_so}'
env = dict(os.environ)
env.pop('IPOPT_LIBRARY_PATH', None)
env.pop('LD_LIBRARY_PATH', None)
ldd_out = subprocess.check_output(['ldd', mpi_sos[0]], env=env, text=True)
ompi_prefix = os.environ['OPENMPI_PREFIX'].rstrip('/') + '/'
libmpi_lines = [l for l in ldd_out.splitlines() if 'libmpi.so' in l and '=>' in l]
assert libmpi_lines, f'ldd shows no libmpi.so link'
resolved = libmpi_lines[0].split('=>')[1].split('(')[0].strip()
assert resolved.startswith(ompi_prefix), (
    f'libmpi.so resolves to {resolved}, not inside {ompi_prefix}')
print(f'  [3/4] libmpi.so RPATH resolves inside \$OPENMPI_PREFIX (no env var needed)')

# 4. mpi4py and _pysu2.so share the same libmpi realpath
su2_run = os.environ.get('SU2_RUN', os.environ.get('SU2_PREFIX', '') + '/bin')
pysu2_so_glob = glob.glob(os.path.join(su2_run, '_pysu2*.so'))
if pysu2_so_glob:
    ldd_su2 = subprocess.check_output(['ldd', pysu2_so_glob[0]], env=env, text=True)
    su2_libmpi = [l for l in ldd_su2.splitlines() if 'libmpi.so' in l and '=>' in l]
    if su2_libmpi:
        su2_resolved = os.path.realpath(su2_libmpi[0].split('=>')[1].split('(')[0].strip())
        mpi4py_resolved = os.path.realpath(resolved)
        assert su2_resolved == mpi4py_resolved, (
            f'ABI mismatch: mpi4py -> {mpi4py_resolved}, _pysu2.so -> {su2_resolved}')
        print(f'  [4/4] mpi4py and _pysu2.so share the same libmpi: {mpi4py_resolved}')
    else:
        print('  [4/4] SKIP — _pysu2.so has no libmpi link to check')
else:
    print('  [4/4] SKIP — _pysu2.so not found (SU2 not installed yet)')

print('  mpi4py binding verified.')
"
    fi
    echo ""

    if [ "$BINDINGS_ONLY" -eq 1 ]; then
        echo "  Binding installation complete for $VENV_DIR."
        exit 0
    fi

    # ── Core scientific stack ──
    echo "=== Core scientific stack ==="
    run "${PIP[@]}" install --no-cache-dir -q "${CORE_PINS[@]}"
    echo ""

    # ── MDO stack ──
    echo "=== MDO Python stack ==="

    echo "  openmdao..."
    run "${PIP[@]}" install --no-cache-dir -q "$OPENMDAO_PIN" "$NUMPY_PIN"

    echo "  mphys..."
    run "${PIP[@]}" install --no-cache-dir -q "$MPHYS_PIN" "$NUMPY_PIN"

    if venv_has pyspline; then
        echo "  pyspline already installed — skipping."
    else
        echo "  pyspline (Fortran source build)..."
        if [ "$_DRY_RUN" -eq 1 ]; then
            echo "  [dry-run] git clone --depth 1 --branch v1.5.4 https://github.com/mdolab/pyspline.git <tmpdir>"
        else
            PYSPLINE_DIR=$(mktemp -d "$TMPDIR/pyspline.XXXXXX")
            trap 'rm -rf -- "$PYSPLINE_DIR"' EXIT
            git clone --depth 1 --branch v1.5.4 https://github.com/mdolab/pyspline.git "$PYSPLINE_DIR" 2>/dev/null
            pushd "$PYSPLINE_DIR" > /dev/null
            cp config/defaults/config.LINUX_GFORTRAN.mk config/config.mk
            sed -i "s|^F2PY = f2py|F2PY = $VENV_PY -m numpy.f2py|" config/config.mk
            make -s
            "${PIP[@]}" install --no-cache-dir -q .
            popd > /dev/null
            rm -rf "$PYSPLINE_DIR"
            trap - EXIT
        fi
    fi

    if venv_has pygeo; then
        echo "  pygeo already installed — skipping."
    else
        echo "  pygeo (from mdolab GitHub)..."
        run "${PIP[@]}" install --no-cache-dir -q "pygeo @ git+https://github.com/mdolab/pygeo.git@v1.17.0"
    fi

    if venv_has pyoptsparse; then
        echo "  pyoptsparse already installed — skipping."
    else
        echo "  pyoptsparse (from mdolab GitHub)..."
        run "${PIP[@]}" install --no-cache-dir -q "pyoptsparse @ git+https://github.com/mdolab/pyoptsparse.git@v2.16.0"
    fi

    # prefoil (required): run/mesh.py reads airfoil geometry through it, and
    # pygeo's DVGeometryCST imports it. Not on PyPI at all -- GitHub only.
    if venv_has prefoil; then
        echo "  prefoil already installed — skipping."
    else
        echo "  prefoil (from mdolab GitHub)..."
        run "${PIP[@]}" install --no-cache-dir -q "prefoil @ git+https://github.com/mdolab/prefoil.git@v2.0.2"
    fi

    echo "  gmsh..."
    run "${PIP[@]}" install --no-cache-dir -q "$GMSH_PIN"

    # ── cyipopt ──
    if [ -n "${IPOPT_PREFIX:-}" ] && [ -d "$IPOPT_PREFIX/lib" ]; then
        if venv_has cyipopt && [ "$FORCE_PYTHON" -eq 0 ]; then
            echo ""
            echo "  cyipopt already installed — skipping."
        else
            echo ""
            echo "=== cyipopt (source build with RPATH against $IPOPT_PREFIX) ==="
            run env \
                LDFLAGS="-Wl,-rpath,$IPOPT_PREFIX/lib${LDFLAGS:+ $LDFLAGS}" \
                PKG_CONFIG_PATH="$IPOPT_PREFIX/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}" \
                "${PIP[@]}" install --no-cache-dir --ignore-installed --no-binary=cyipopt cyipopt

            if [ "$_DRY_RUN" -eq 0 ]; then
                echo "  Verifying cyipopt RPATH..."
                "$VENV_PY" -c "
import subprocess, os, glob, sysconfig

site = sysconfig.get_paths()['purelib']
cyipopt_sos = glob.glob(os.path.join(site, 'cyipopt', '*.so'))
if not cyipopt_sos:
    cyipopt_sos = glob.glob(os.path.join(site, 'cyipopt*.so'))
if cyipopt_sos:
    env = dict(os.environ)
    env.pop('IPOPT_LIBRARY_PATH', None)
    env.pop('LD_LIBRARY_PATH', None)
    ldd = subprocess.check_output(['ldd', cyipopt_sos[0]], env=env, text=True)
    ipopt_prefix = os.environ['IPOPT_PREFIX'].rstrip('/') + '/'
    ipopt_lines = [l for l in ldd.splitlines() if 'libipopt' in l and '=>' in l]
    if ipopt_lines:
        resolved = ipopt_lines[0].split('=>')[1].split('(')[0].strip()
        assert resolved.startswith(ipopt_prefix), (
            f'cyipopt libipopt resolves to {resolved}, not inside {ipopt_prefix}')
        print(f'  cyipopt RPATH verified: libipopt -> {resolved}')
    else:
        print('  WARNING: ldd shows no libipopt link')
else:
    print('  WARNING: cyipopt .so not found for RPATH check')
"
            fi
        fi
    else
        echo ""
        echo "  Skipping cyipopt (IPOPT_PREFIX not set or lib/ missing). SLSQP still works."
    fi
    echo ""
fi

# ── SU2 ──

if [ "$WITH_SU2" -eq 1 ]; then
    echo "=== SU2 ==="
    echo "  SU2_PREFIX:      $SU2_PREFIX  (${PATH_SRC[SU2_PREFIX]:-derived})"
    echo "  SU2_SRC:         $SU2_SRC  (${PATH_SRC[SU2_SRC]:-derived})"
    if su2_installed && [ "$FORCE_SU2" -eq 0 ]; then
        echo "  SKIPPED — SU2 already installed at $SU2_PREFIX"
    else
        [ "$FORCE_SU2" -eq 1 ] && echo "  REBUILD — --force-su2 requested" \
                               || echo "  NOT USABLE — ${_SU2_SKIP_REASON:-unknown}"
        echo "  Building SU2 ($SU2_REF) at $SU2_PREFIX"
        # SU2 meson build requires mpi4py from the venv.
        if [ "$_DRY_RUN" -eq 0 ]; then
            [ -x "$VENV_DIR/bin/python" ] \
                || die "No venv at $VENV_DIR. SU2's meson build imports mpi4py from it — run setup.sh --with-python first."
            venv_has mpi4py \
                || die "mpi4py is not importable by $VENV_DIR/bin/python. SU2's pywrapper build requires it — run setup.sh --with-python first."
        fi
        run mkdir -p "$SHAPEOPT_ROOT/SU2v8.5"
        # Pinned shallow clone; meson handles submodule init. SU2_REF is a tag
        # or a branch of origin (git clone --branch takes either).
        if [ ! -d "$SU2_SRC/.git" ]; then
            echo "  Cloning SU2 $SU2_REF (shallow, pinned)..."
            record_new "$SU2_SRC"
            run git clone --depth 1 --branch "$SU2_REF" \
                https://github.com/su2code/SU2.git "$SU2_SRC"
        else
            # Never move a checkout that has local changes to tracked files.
            if [ -n "$(git -C "$SU2_SRC" status --porcelain --untracked-files=no 2>/dev/null)" ]; then
                die "SU2_SRC ($SU2_SRC) has local changes to tracked files. Commit, stash or discard them, then re-run."
            fi
            # Ensure existing checkout is on the correct ref.
            _have="$(git -C "$SU2_SRC" rev-parse --verify -q "refs/tags/$SU2_REF^{commit}" 2>/dev/null || true)"
            _head="$(git -C "$SU2_SRC" rev-parse HEAD 2>/dev/null || echo unknown)"
            if [ -n "$_have" ] && [ "$_have" = "$_head" ]; then
                echo "  SU2_SRC already at $SU2_REF."
            elif git -C "$SU2_SRC" ls-remote --exit-code --tags origin "refs/tags/$SU2_REF" &>/dev/null; then
                echo "  SU2_SRC is not at tag $SU2_REF — fetching and checking out."
                run git -C "$SU2_SRC" fetch --depth 1 origin "refs/tags/$SU2_REF:refs/tags/$SU2_REF"
                run git -C "$SU2_SRC" checkout --detach "refs/tags/$SU2_REF"
            elif git -C "$SU2_SRC" ls-remote --exit-code --heads origin "refs/heads/$SU2_REF" &>/dev/null; then
                echo "  Fetching branch $SU2_REF and checking out its head."
                run git -C "$SU2_SRC" fetch --depth 1 origin "refs/heads/$SU2_REF"
                run git -C "$SU2_SRC" checkout --detach FETCH_HEAD
            else
                die "SU2_REF=$SU2_REF is neither a tag nor a branch of $(git -C "$SU2_SRC" remote get-url origin 2>/dev/null || echo origin)."
            fi
        fi
        if [ "$_DRY_RUN" -eq 0 ]; then
            source "$VENV_DIR/bin/activate"
        fi
        echo "  Configuring..."
        SU2_MESON_OPTS=(-Denable-autodiff=true -Denable-pywrapper=true -Dwith-mpi=enabled
                        -Denable-cgns=true "--prefix=$SU2_PREFIX")
        record_new "$SU2_SRC/build"
        if [ -d "$SU2_SRC/build" ]; then
            # An existing build directory: reconfigure in place, else wipe it.
            if ! run_in "$SU2_SRC" "$VENV_DIR/bin/python" ./meson.py setup build --reconfigure \
                    "${SU2_MESON_OPTS[@]}" \
                    2>&1 | tee "$SHAPEOPT_ROOT/logs/su2_configure.log" | tail -10; then
                echo "  Reconfigure failed — wiping the build directory (log kept)."
                run_in "$SU2_SRC" "$VENV_DIR/bin/python" ./meson.py setup build --wipe \
                    "${SU2_MESON_OPTS[@]}" \
                    2>&1 | tee -a "$SHAPEOPT_ROOT/logs/su2_configure.log" | tail -10
            fi
        else
            run_in "$SU2_SRC" "$VENV_DIR/bin/python" ./meson.py setup build \
                "${SU2_MESON_OPTS[@]}" \
                2>&1 | tee "$SHAPEOPT_ROOT/logs/su2_configure.log" | tail -10
        fi
        echo "  Building + installing..."
        record_new "$SU2_PREFIX"
        run_in "$SU2_SRC" ./ninja -C build install \
            2>&1 | tee "$SHAPEOPT_ROOT/logs/su2_install.log" | tail -5

        if [ "$_DRY_RUN" -eq 0 ]; then
            su2_commit=$(git -C "$SU2_SRC" rev-parse HEAD 2>/dev/null || echo "unknown")

            su2_describe=$(git -C "$SU2_SRC" describe --tags --always 2>/dev/null || echo "unknown")
            cat > "$SU2_PREFIX/.mdao-su2-stamp" <<STAMP
su2_ref=$SU2_REF
su2_commit=$su2_commit
su2_describe=$su2_describe
python=$("$VENV_DIR/bin/python" --version 2>/dev/null)
python_abi=$(su2_python_abi)
meson_options=${SU2_MESON_OPTS[*]}
STAMP
        fi
        echo "  SU2 installed at $SU2_PREFIX"
    fi
    echo ""
fi

# ── Write .env_local ──
echo "=== Writing .env_local ==="
persist_env
echo ""

# ── Verification ──
if [ "$SKIP_VERIFY" -eq 0 ] && [ "$_DRY_RUN" -eq 0 ] && [ "$WITH_PYTHON" -eq 1 ]; then
    echo "=== Verifying installed packages ==="

    "$VENV_DIR/bin/python" -c "
import importlib, sys
pkgs = ['numpy', 'scipy', 'matplotlib', 'mpi4py', 'petsc', 'petsc4py',
        'openmdao', 'mphys', 'pygeo', 'pyspline', 'pyoptsparse', 'gmsh',
        'prefoil']
ok = True
for p in pkgs:
    try:
        m = importlib.import_module(p)
        v = getattr(m, '__version__', '?')
        loc = getattr(m, '__file__', '?')
        print(f'  {p:16s} {v:12s}  {loc}')
    except ImportError as e:
        print(f'  {p:16s} MISSING  ({e})')
        ok = False
if not ok:
    sys.exit(1)
print()
print('  All packages installed and importable.')
"
    echo ""
fi

# ── Activation gate ──
# The check above runs in the environment this script cleaned, so it passes even
# when the user's own terminal would load another application's Python packages.
# Repeat it the way every run starts: the environment setup.sh was started with,
# then source activate.sh.
if [ "$SKIP_VERIFY" -eq 0 ] && [ "$_DRY_RUN" -eq 0 ]; then
    echo "=== Verifying imports through activate.sh ==="
    if [ ! -x "$VENV_PY" ] || [ ! -d "$SU2_PREFIX/bin" ] || [ ! -d "$OPENMPI_PREFIX/bin" ]; then
        echo "  SKIPPED — activate.sh needs the venv, SU2 and Open MPI installed."
    elif ! env -i "${_ORIG_ENV[@]}" bash -c \
            'source "$1/activate.sh" > /dev/null || exit 1
             python "$1/tools/check_env.py" --isolation-only' _ "$REPO"; then
        echo ""
        echo "  The packages import from the venv only in the environment this script"
        echo "  cleans; after 'source activate.sh' in your terminal they load from"
        echo "  elsewhere. The table above names the foreign path or module."
        die "Activated environment does not import from $VENV_DIR."
    fi
    echo ""
fi

echo "=== Setup complete ==="
print_prefixes "final prefixes"
echo ""
echo "  Next steps:"
echo "    source activate.sh          # activate environment"
echo "    bash verify.sh [NP]         # verify at NP cores"
echo ""
echo "  The venv and all prefixes are persisted in .env_local."
echo "  'source activate.sh' restores them in any new shell."
