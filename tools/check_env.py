"""Verify all MDO stack imports, MPI runtime, and optimizer access."""
# Must be set BEFORE OpenMDAO import to prevent report artifacts in CWD.
import os
os.environ.setdefault('OPENMDAO_REPORTS', '0')
import sys
import importlib

REQUIRED_MODULES = [
    ("mpi4py", "MPI runtime bindings"),
    ("pysu2", "SU2 primal Python wrapper"),
    ("pysu2ad", "SU2 discrete-adjoint Python wrapper"),
    ("openmdao", "OpenMDAO framework"),
    ("mphys", "MPhys multiphysics framework"),
    ("pygeo", "pyGeo geometry parameterization"),
    ("pyoptsparse", "pyOptSparse optimizer interface"),
    ("prefoil", "airfoil geometry for run/mesh.py and DVGeometryCST"),
]

VERSION_FLOORS = {
    "pygeo": ("1.15", "PyPI 'pygeo' is a squatted package (0.1.3). "
              "Install the real one: pip install 'pygeo @ git+https://github.com/mdolab/pygeo.git@v1.17.0'"),
    "pyspline": ("1.5", "PyPI 'pyspline' is a squatted package (0.0.1). "
                 "Install the real one: clone github.com/mdolab/pyspline, pip install ."),
}

EXPECTED_OPENMPI_VERSION = "5.0.10"


# Packages that must load from the venv; mdao_su2 and pysu2 have their own homes.
ISOLATED_MODULES = ["numpy", "scipy", "mpi4py", "petsc4py", "openmdao", "mphys",
                    "pygeo", "pyspline", "pyoptsparse", "prefoil"]


def _under(path, root):
    path, root = os.path.realpath(path), os.path.realpath(root)
    return path == root or path.startswith(root + os.sep)


def check_import_isolation():
    """sys.path holds only the venv, the base stdlib, $SU2_RUN and src/.

    A PYTHONPATH entry exported by a shell startup file (another application's
    Python 2.7 site-packages, say) precedes the venv's site-packages and replaces
    its scipy or numpy; the import then fails with an unrelated-looking
    undefined-symbol error. activate.sh removes such entries.
    """
    repo_src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
    su2_run = os.environ.get("SU2_RUN", "")
    in_venv = sys.prefix != sys.base_prefix
    results = [("python.venv", "PASS" if in_venv else "FAIL",
                f"{sys.executable} (prefix {sys.prefix})" if in_venv else
                f"{sys.executable} is not a venv interpreter; source activate.sh")]

    allowed = [sys.prefix, repo_src] + ([su2_run] if su2_run else [])
    foreign = []
    # sys.path[0] is the script directory (or '' for -c); PYTHONPATH follows it.
    for entry in sys.path[1:]:
        if any(_under(entry, a) for a in allowed):
            continue
        base_lib = os.path.join(sys.base_prefix, "lib")
        if _under(entry, base_lib) and "-packages" not in entry:
            continue  # the base interpreter's standard library
        foreign.append(entry)
    for var in ("PYTHONHOME",):
        if os.environ.get(var):
            foreign.append(f"{var}={os.environ[var]}")
    if foreign:
        results.append(("sys.path", "FAIL",
                        "foreign entries " + ", ".join(foreign)
                        + " — set by PYTHONPATH/PYTHONHOME in a shell startup file;"
                          " open a new shell and source activate.sh"))
    else:
        results.append(("sys.path", "PASS", "venv, base stdlib, $SU2_RUN and src/ only"))

    for name in ISOLATED_MODULES:
        try:
            mod = importlib.import_module(name)
        except Exception as e:  # a foreign extension module fails with ImportError or worse
            results.append((name, "FAIL", f"{type(e).__name__}: {e}"))
            continue
        loc = getattr(mod, "__file__", None) or "?"
        if _under(loc, sys.prefix):
            results.append((name, "PASS", loc))
        else:
            results.append((name, "FAIL", f"loaded from {loc}, outside the venv {sys.prefix}"))
    return results


def version_tuple(v):
    return tuple(int(x) for x in v.split(".")[:2])


def check_imports():
    results = []
    for mod_name, desc in REQUIRED_MODULES:
        try:
            importlib.import_module(mod_name)
            results.append((mod_name, desc, "PASS", ""))
        except ImportError as e:
            results.append((mod_name, desc, "FAIL", str(e)))
    return results


def check_versions():
    results = []
    for mod_name, (floor, msg) in VERSION_FLOORS.items():
        try:
            mod = importlib.import_module(mod_name)
            ver = getattr(mod, "__version__", None)
            if ver is None:
                results.append((mod_name, "WARN", f"no __version__ attribute"))
                continue
            if version_tuple(ver) >= version_tuple(floor):
                results.append((mod_name, "PASS", f"v{ver} >= {floor}"))
            else:
                results.append((mod_name, "FAIL", f"v{ver} < {floor} — {msg}"))
        except ImportError:
            results.append((mod_name, "SKIP", "import failed (reported above)"))
    return results


def check_functional_probes():
    results = []

    try:
        from pygeo import DVGeometry  # noqa: F401
        results.append(("pygeo.DVGeometry", "PASS", ""))
    except (ImportError, AttributeError) as e:
        results.append(("pygeo.DVGeometry", "FAIL", str(e)))

    try:
        from pyoptsparse import OPT
        OPT("SLSQP")
        results.append(("pyoptsparse.SLSQP", "PASS", ""))
    except Exception as e:
        results.append(("pyoptsparse.SLSQP", "FAIL", str(e)))

    try:
        from pyoptsparse import OPT
        OPT("IPOPT")
        results.append(("pyoptsparse.IPOPT", "PASS", "available"))
    except Exception:
        results.append(("pyoptsparse.IPOPT", "WARN",
                        "IPOPT not available — SLSQP will still work. "
                        "To enable: set IPOPT_PREFIX and run "
                        "pip install --no-binary cyipopt cyipopt"))

    return results


def check_mpi_runtime():
    from mpi4py import MPI

    lib_version = MPI.Get_library_version()
    if "Open MPI" not in lib_version:
        return [("MPI runtime", "FAIL",
                 f"Expected Open MPI, got: {lib_version.splitlines()[0]}. "
                 "Check that OPENMPI_PREFIX/lib is on LD_LIBRARY_PATH.")]
    version_line = lib_version.splitlines()[0]

    if EXPECTED_OPENMPI_VERSION not in lib_version:
        return [("MPI runtime", "FAIL",
                 f"Expected Open MPI {EXPECTED_OPENMPI_VERSION}, "
                 f"got: {version_line}")]

    return [("MPI runtime", "PASS", version_line)]


def check_mpi4py_binding():
    """Four-layer mpi4py ABI verification."""
    import subprocess
    import glob

    results = []

    # 1. mpi4py.__file__ is under sys.prefix
    import mpi4py
    if mpi4py.__file__.startswith(sys.prefix):
        results.append(("mpi4py.location", "PASS", "inside venv"))
    else:
        results.append(("mpi4py.location", "FAIL",
                        f"{mpi4py.__file__} is outside venv {sys.prefix} — "
                        "the base copy is shadowing the venv copy"))

    # 2. Exact version (already checked in check_mpi_runtime, but confirm here)
    from mpi4py import MPI
    lib_ver = MPI.Get_library_version()
    if EXPECTED_OPENMPI_VERSION in lib_ver:
        results.append(("mpi4py.version", "PASS",
                        f"linked to Open MPI {EXPECTED_OPENMPI_VERSION}"))
    else:
        results.append(("mpi4py.version", "FAIL",
                        f"linked to {lib_ver.splitlines()[0]}, "
                        f"expected {EXPECTED_OPENMPI_VERSION}"))

    # 3. ldd with IPOPT_LIBRARY_PATH cleared resolves inside $OPENMPI_PREFIX
    openmpi_prefix = os.environ.get("OPENMPI_PREFIX", "")
    mpi_so_pattern = os.path.join(os.path.dirname(mpi4py.__file__), "MPI.cpython*.so")
    mpi_sos = glob.glob(mpi_so_pattern)
    if mpi_sos and openmpi_prefix:
        env = dict(os.environ)
        env.pop("IPOPT_LIBRARY_PATH", None)
        try:
            ldd_out = subprocess.check_output(
                ["ldd", mpi_sos[0]], env=env, text=True, stderr=subprocess.DEVNULL)
            libmpi_lines = [l for l in ldd_out.splitlines()
                            if "libmpi.so" in l and "=>" in l]
            if libmpi_lines:
                resolved = libmpi_lines[0].split("=>")[1].split("(")[0].strip()
                if resolved.startswith(openmpi_prefix):
                    results.append(("mpi4py.rpath", "PASS",
                                    f"libmpi.so -> {resolved}"))
                else:
                    results.append(("mpi4py.rpath", "FAIL",
                                    f"libmpi.so -> {resolved}, "
                                    f"expected inside {openmpi_prefix}"))
            else:
                results.append(("mpi4py.rpath", "WARN", "no libmpi.so in ldd output"))
        except (subprocess.CalledProcessError, FileNotFoundError):
            results.append(("mpi4py.rpath", "WARN", "ldd check failed"))
    else:
        results.append(("mpi4py.rpath", "SKIP",
                        "OPENMPI_PREFIX not set or MPI .so not found"))

    # 4. mpi4py and _pysu2.so share the same libmpi realpath
    su2_run = os.environ.get("SU2_RUN", "")
    if su2_run and mpi_sos and openmpi_prefix:
        pysu2_sos = glob.glob(os.path.join(su2_run, "_pysu2*.so"))
        if pysu2_sos:
            env = dict(os.environ)
            env.pop("IPOPT_LIBRARY_PATH", None)
            try:
                ldd_m = subprocess.check_output(
                    ["ldd", mpi_sos[0]], env=env, text=True, stderr=subprocess.DEVNULL)
                ldd_s = subprocess.check_output(
                    ["ldd", pysu2_sos[0]], env=env, text=True, stderr=subprocess.DEVNULL)
                m_lines = [l for l in ldd_m.splitlines()
                           if "libmpi.so" in l and "=>" in l]
                s_lines = [l for l in ldd_s.splitlines()
                           if "libmpi.so" in l and "=>" in l]
                if m_lines and s_lines:
                    m_real = os.path.realpath(
                        m_lines[0].split("=>")[1].split("(")[0].strip())
                    s_real = os.path.realpath(
                        s_lines[0].split("=>")[1].split("(")[0].strip())
                    if m_real == s_real:
                        results.append(("mpi4py._pysu2.coresidency", "PASS",
                                        f"both -> {m_real}"))
                    else:
                        results.append(("mpi4py._pysu2.coresidency", "FAIL",
                                        f"mpi4py -> {m_real}, _pysu2.so -> {s_real}"))
                else:
                    results.append(("mpi4py._pysu2.coresidency", "WARN",
                                    "libmpi.so not in ldd output"))
            except (subprocess.CalledProcessError, FileNotFoundError):
                results.append(("mpi4py._pysu2.coresidency", "WARN",
                                "ldd check failed"))
        else:
            results.append(("mpi4py._pysu2.coresidency", "SKIP",
                            "_pysu2.so not found"))
    else:
        results.append(("mpi4py._pysu2.coresidency", "SKIP",
                        "SU2_RUN not set"))

    return results


def check_cyipopt_binding():
    """Verify cyipopt resolves libipopt via RPATH, not env vars."""
    import subprocess
    import glob
    import sysconfig

    results = []
    ipopt_prefix = os.environ.get("IPOPT_PREFIX", "")
    if not ipopt_prefix:
        results.append(("cyipopt.rpath", "SKIP", "IPOPT_PREFIX not set"))
        return results

    site = sysconfig.get_paths()["purelib"]
    cyipopt_sos = glob.glob(os.path.join(site, "cyipopt", "*.so"))
    if not cyipopt_sos:
        cyipopt_sos = glob.glob(os.path.join(site, "cyipopt*.so"))
    if not cyipopt_sos:
        results.append(("cyipopt.rpath", "SKIP", "cyipopt .so not found"))
        return results

    env = dict(os.environ)
    env.pop("IPOPT_LIBRARY_PATH", None)
    env.pop("LD_LIBRARY_PATH", None)
    try:
        ldd_out = subprocess.check_output(
            ["ldd", cyipopt_sos[0]], env=env, text=True, stderr=subprocess.DEVNULL)
        ipopt_lines = [l for l in ldd_out.splitlines()
                       if "libipopt" in l and "=>" in l]
        if ipopt_lines:
            resolved = ipopt_lines[0].split("=>")[1].split("(")[0].strip()
            if resolved.startswith(ipopt_prefix):
                results.append(("cyipopt.rpath", "PASS",
                                f"libipopt -> {resolved}"))
            else:
                results.append(("cyipopt.rpath", "FAIL",
                                f"libipopt -> {resolved}, "
                                f"expected inside {ipopt_prefix}"))
        else:
            results.append(("cyipopt.rpath", "WARN",
                            "no libipopt in ldd output"))
    except (subprocess.CalledProcessError, FileNotFoundError):
        results.append(("cyipopt.rpath", "WARN", "ldd check failed"))

    return results


def check_mpi_abi_preflight():
    """Pre-import ldd check: all MPI-linked .so must resolve to $OPENMPI_PREFIX's libmpi."""
    import subprocess
    import glob
    import importlib.util

    results = []
    openmpi_prefix = os.environ.get("OPENMPI_PREFIX", "")
    if not openmpi_prefix:
        return [("mpi.abi.preflight", "SKIP", "OPENMPI_PREFIX not set")]

    def module_libs(mod_name):
        """Shared objects shipped by a module, located WITHOUT importing it."""
        try:
            spec = importlib.util.find_spec(mod_name)
        except (ImportError, ValueError):
            return []
        if spec is None or not spec.origin:
            return []
        root = os.path.dirname(spec.origin)
        return (glob.glob(os.path.join(root, "*.so"))
                + glob.glob(os.path.join(root, "lib", "*.so*")))

    targets = []
    for mod in ("mpi4py", "petsc4py", "petsc"):
        targets.extend((mod, so) for so in module_libs(mod))
    su2_run = os.environ.get("SU2_RUN", "")
    if su2_run:
        targets.extend(("SU2", so) for so in glob.glob(os.path.join(su2_run, "_pysu2*.so")))

    env = dict(os.environ)
    env.pop("IPOPT_LIBRARY_PATH", None)
    env.pop("LD_LIBRARY_PATH", None)
    bad = []
    checked = 0
    for owner, so in targets:
        try:
            ldd_out = subprocess.check_output(
                ["ldd", so], env=env, text=True, stderr=subprocess.DEVNULL)
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue
        for line in ldd_out.splitlines():
            if "libmpi.so" not in line or "=>" not in line:
                continue
            resolved = line.split("=>")[1].split("(")[0].strip()
            if not resolved or resolved == "not found":
                bad.append((owner, os.path.basename(so), "not found"))
                break
            checked += 1
            if not _under(resolved, openmpi_prefix):
                bad.append((owner, os.path.basename(so), resolved))
            break

    if bad:
        for owner, so, resolved in bad:
            results.append((f"mpi.abi.{owner}", "FAIL",
                            f"{so} -> {resolved} (outside {openmpi_prefix})"))
    elif checked:
        results.append(("mpi.abi.preflight", "PASS",
                        f"{checked} MPI-linked libraries all use {openmpi_prefix}"))
    else:
        results.append(("mpi.abi.preflight", "WARN", "no MPI-linked libraries found"))
    return results


def check_mpi_allreduce():
    from mpi4py import MPI
    import numpy as np

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    local = np.array([float(rank + 1)])
    total = np.zeros(1)
    comm.Allreduce(local, total, op=MPI.SUM)
    expected = size * (size + 1) / 2.0

    if abs(total[0] - expected) < 1e-12:
        return [("MPI allreduce", "PASS", f"rank={rank}, size={size}, sum={total[0]}")]
    else:
        return [("MPI allreduce", "FAIL", f"expected {expected}, got {total[0]}")]


def print_table(title, rows):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")
    for row in rows:
        name = row[0]
        status = row[1] if len(row) > 1 else ""
        detail = row[2] if len(row) > 2 else ""
        extra = row[3] if len(row) > 3 else ""
        if extra:
            print(f"  [{status:4s}] {name}: {extra}")
        elif detail:
            print(f"  [{status:4s}] {name}: {detail}")
        else:
            print(f"  [{status:4s}] {name}")
    print()


def abi_only():
    """ABI gate only — imports nothing MPI-linked, safe for setup.sh."""
    results = check_mpi_abi_preflight()
    print_table("MPI ABI Preflight", results)
    if any(s == "FAIL" for _, s, _ in results):
        print("  Rebuild the offending package against "
              f"$OPENMPI_PREFIX ({os.environ.get('OPENMPI_PREFIX', '?')}).")
        return 1
    return 0


def isolation_only():
    """Import isolation gate only — setup.sh runs it inside a fresh activation."""
    results = check_import_isolation()
    print_table("Python Import Isolation", results)
    return 1 if any(s == "FAIL" for _, s, _ in results) else 0


def main():
    if "--abi-only" in sys.argv:
        sys.exit(abi_only())
    if "--isolation-only" in sys.argv:
        sys.exit(isolation_only())

    from mpi4py import MPI
    rank = MPI.COMM_WORLD.Get_rank()

    all_results = []

    # MPI ABI preflight — must run before any MPI-linked import (SIGABRT on mismatch).
    abi_results = check_mpi_abi_preflight()
    if rank == 0:
        print_table("MPI ABI Preflight", abi_results)
    all_results.extend([(n, "", s, d) for n, s, d in abi_results])
    if any(s == "FAIL" for _, s, _ in abi_results):
        if rank == 0:
            print("  A library is built against a different Open MPI than "
                  f"{os.environ.get('OPENMPI_PREFIX', '?')}.")
            print("  Importing it would abort this rank, so the remaining checks "
                  "are skipped.")
            print("  Rebuild the offending package against $OPENMPI_PREFIX — for "
                  "petsc/petsc4py:")
            print("      pip uninstall -y petsc petsc4py")
            print("      export PETSC_CONFIGURE_OPTIONS=\"--with-mpi-dir=$OPENMPI_PREFIX\"")
            print("      pip install --no-cache-dir petsc && \\")
            print("        PETSC_DIR=$(python3 -c 'import petsc,os;print(os.path.dirname(petsc.__file__))') \\")
            print("        pip install --no-cache-dir petsc4py")
            print(f"\n{'=' * 60}")
            print("  RESULT: FAIL (MPI ABI mismatch)")
            print(f"{'=' * 60}")
        sys.exit(1)

    # 0. Import isolation: a foreign sys.path entry explains every import failure below.
    iso_results = check_import_isolation()
    if rank == 0:
        print_table("Python Import Isolation", iso_results)
    all_results.extend([(n, "", s, d) for n, s, d in iso_results])

    # 1. Module imports
    import_results = check_imports()
    if rank == 0:
        print_table("Module Imports", [(n, s, d) for n, _, s, d in import_results])
    all_results.extend(import_results)

    # 2. Anti-squatter version checks
    ver_results = check_versions()
    if rank == 0:
        print_table("Version Checks (anti-squatter)", ver_results)
    all_results.extend([(n, "", s, d) for n, s, d in ver_results])

    # 3. Functional probes
    func_results = check_functional_probes()
    if rank == 0:
        print_table("Functional Probes", func_results)
    all_results.extend([(n, "", s, d) for n, s, d in func_results])

    # 4. MPI runtime identity (exact version)
    mpi_results = check_mpi_runtime()
    if rank == 0:
        print_table("MPI Runtime", mpi_results)
    all_results.extend([(n, "", s, d) for n, s, d in mpi_results])

    # 5. mpi4py ABI binding (four-layer)
    mpi4py_results = check_mpi4py_binding()
    if rank == 0:
        print_table("mpi4py ABI Binding", mpi4py_results)
    all_results.extend([(n, "", s, d) for n, s, d in mpi4py_results])

    # 6. cyipopt RPATH
    cyipopt_results = check_cyipopt_binding()
    if rank == 0:
        print_table("cyipopt RPATH", cyipopt_results)
    all_results.extend([(n, "", s, d) for n, s, d in cyipopt_results])

    # 7. MPI allreduce
    ar_results = check_mpi_allreduce()
    if rank == 0:
        print_table("MPI Allreduce", ar_results)
    all_results.extend([(n, "", s, d) for n, s, d in ar_results])

    # Summary
    statuses = [r[2] if len(r) > 2 else r[1] for r in all_results]
    failures = sum(1 for s in statuses if s == "FAIL")
    warnings = sum(1 for s in statuses if s == "WARN")

    if rank == 0:
        print(f"{'=' * 60}")
        if failures == 0:
            print(f"  RESULT: PASS ({warnings} warning(s))")
        else:
            print(f"  RESULT: FAIL ({failures} failure(s), {warnings} warning(s))")
        print(f"{'=' * 60}")

    sys.exit(1 if failures > 0 else 0)


if __name__ == "__main__":
    main()
