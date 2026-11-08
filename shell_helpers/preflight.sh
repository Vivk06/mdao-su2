#!/usr/bin/env bash
# Sudo-free prerequisite checks for setup.sh. Source only; requires env_common.sh.

[ "${_ENV_COMMON_LOADED:-}" = "1" ] || { echo "ERROR: source env_common.sh first" >&2; exit 1; }

# check_tool NAME [PACKAGE_HINT]
# Returns 0 if the tool is on PATH, 1 otherwise. Accumulates into _MISSING_TOOLS.
_MISSING_TOOLS=()
_MISSING_PKGS=()
# Every prerequisite checked, for the full list in preflight_report.
_ALL_DEPS=()

check_tool() {
    local tool="$1"
    local pkg="${2:-$1}"
    _ALL_DEPS+=("$tool")
    if ! command -v "$tool" &>/dev/null; then
        _MISSING_TOOLS+=("$tool")
        _MISSING_PKGS+=("$pkg")
        return 1
    fi
    return 0
}

# check_header NAME [PACKAGE_HINT]
# Checks that the C compiler can resolve a header through its include path.
check_header() {
    local header="$1"
    local pkg="${2:-}"
    _ALL_DEPS+=("$header")
    if ! printf '#include <%s>\n' "$header" | gcc -E -x c - >/dev/null 2>&1; then
        if [ -n "$pkg" ]; then
            _MISSING_TOOLS+=("$header")
            _MISSING_PKGS+=("$pkg")
        fi
        return 1
    fi
    return 0
}

# preflight_base — tools needed for every setup.sh invocation.
preflight_base() {
    check_tool gcc
    check_tool g++
    check_tool gfortran
    check_tool make
    check_tool git
    check_tool pkg-config pkg-config
    check_tool tar
    check_tool patch
    check_tool ldd libc-bin
    # wget or curl — at least one
    _ALL_DEPS+=("wget|curl")
    if ! command -v wget &>/dev/null && ! command -v curl &>/dev/null; then
        _MISSING_TOOLS+=("wget|curl")
        _MISSING_PKGS+=("wget")
    fi
    check_tool bzip2
}

# preflight_ipopt — system libraries the IPOPT build (MUMPS) links for
# --with-ipopt: LAPACK, BLAS and METIS. Presence only, not versions.
preflight_ipopt() {
    local entry lib pkg cache
    # Read the cache once: grep -q closing a pipe early fails it under pipefail.
    cache="$("$(command -v ldconfig 2>/dev/null || echo /sbin/ldconfig)" -p 2>/dev/null || true)"
    for entry in lapack:liblapack-dev blas:libblas-dev metis:libmetis-dev; do
        lib="${entry%%:*}"; pkg="${entry#*:}"
        _ALL_DEPS+=("lib$lib.so")
        if ! grep -q "lib$lib\.so " <<< "$cache"; then
            _MISSING_TOOLS+=("lib$lib.so")
            _MISSING_PKGS+=("$pkg")
        fi
    done
    check_header metis.h libmetis-dev || true
}

# preflight_su2 — additional tools for --with-su2.
preflight_su2() {
    check_tool swig swig
    # swig version >= 3.0.11
    if command -v swig &>/dev/null; then
        local swig_ver
        swig_ver=$(swig -version 2>/dev/null | grep -oP 'SWIG Version \K[0-9.]+' || echo "0")
        local major minor patch
        IFS='.' read -r major minor patch <<< "$swig_ver"
        major=${major:-0}; minor=${minor:-0}; patch=${patch:-0}
        if [ "$major" -lt 3 ] || { [ "$major" -eq 3 ] && [ "$minor" -eq 0 ] && [ "${patch}" -lt 11 ]; }; then
            warn "swig $swig_ver < 3.0.11 — SU2's pywrapper build requires >= 3.0.11"
        fi
    fi
    # Python.h via sysconfig
    local python_inc
    python_inc=$("${PYTHON_EXE:-python3}" -c "import sysconfig; print(sysconfig.get_paths()['include'])" 2>/dev/null || echo "")
    if [ -n "$python_inc" ] && [ ! -f "$python_inc/Python.h" ]; then
        warn "Python.h not found at $python_inc/Python.h — SU2 build needs python-dev headers"
    fi
    # shared libpython for SU2's embed dependency
    if ! "${PYTHON_EXE:-python3}" -c "import sysconfig; assert sysconfig.get_config_var('Py_ENABLE_SHARED')" 2>/dev/null; then
        warn "Python was not built with --enable-shared — SU2's meson 'python.dependency(embed: true)' may fail"
    fi
    return 0
}

# preflight_report — print actionable summary. Returns 1 if anything is missing.
preflight_report() {
    if [ "${#_MISSING_TOOLS[@]}" -eq 0 ]; then
        return 0
    fi
    local d
    local -A missing
    for d in "${_MISSING_TOOLS[@]}"; do missing[$d]=1; done
    echo ""
    echo "All prerequisites checked:"
    for d in "${_ALL_DEPS[@]}"; do
        printf "  %-28s %s\n" "$d" "$([ -n "${missing[$d]+x}" ] && echo MISSING || echo found)"
    done
    echo ""
    echo "Missing prerequisite tools:"
    local i
    for i in "${!_MISSING_TOOLS[@]}"; do
        echo "  - ${_MISSING_TOOLS[$i]}"
    done
    # Deduplicate package names
    local -A seen
    local pkgs=()
    for p in "${_MISSING_PKGS[@]}"; do
        if [ -z "${seen[$p]+x}" ]; then
            seen[$p]=1
            pkgs+=("$p")
        fi
    done
    echo ""
    echo "Install them with:"
    echo "  sudo apt-get install -y ${pkgs[*]}"
    echo ""
    return 1
}

# check_https URL
# Returns 1 when wget (or curl) reaches URL but cannot verify its certificate, the
# failure every native build's download would hit later. Network errors return 0;
# the download itself reports those. Prints the OpenSSL the tool loaded.
check_https() {
    local url="$1" tool rc=0 ssl
    if command -v wget &>/dev/null; then
        tool=wget
        wget -q --spider --timeout=20 --tries=1 "$url" &>/dev/null || rc=$?
        [ "$rc" -eq 5 ] || return 0          # 5: SSL verification failure
    else
        tool=curl
        curl -fsI --connect-timeout 20 -o /dev/null "$url" &>/dev/null || rc=$?
        [ "$rc" -eq 60 ] || [ "$rc" -eq 77 ] || return 0   # certificate / CA file
    fi
    ssl=$(ldd "$(command -v "$tool")" 2>/dev/null | awk '/libssl/ {print $3; exit}')
    echo ""
    echo "  $tool cannot verify the certificate of $url"
    echo "  $tool loads libssl from: ${ssl:-<unknown>}"
    echo "  If that is not a system path, a directory on LD_LIBRARY_PATH supplies it;"
    echo "  remove that directory where your shell startup files set it."
    echo "  Otherwise update the CA store: sudo apt-get install --reinstall ca-certificates"
    return 1
}

# check_disk_gb MIN_GB [PATH]
# Warns if fewer than MIN_GB are free at PATH (default /).
check_disk_gb() {
    local min_gb="$1"
    local mount="${2:-/}"
    local avail_kb
    avail_kb=$(df --output=avail "$mount" 2>/dev/null | tail -1 | tr -d '[:space:]')
    if [ -n "$avail_kb" ]; then
        local avail_gb=$(( avail_kb / 1048576 ))
        if [ "$avail_gb" -lt "$min_gb" ]; then
            warn "Only ${avail_gb} GB free at $mount (recommend ≥${min_gb} GB for a full build)"
            return 1
        fi
    fi
    return 0
}
