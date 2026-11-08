# Code manual

The package turns a validated run configuration into geometry, SU2 flow solves, derivatives, optimization tasks, and run records. Start with the selected case TOML and `run/optimize.py`; follow its imports into `src/mdao_su2/`. The [reference manual](reference_manual.md) lists accepted configuration keys and command flags.

## Package map and data flow

| Owner | Responsibility | Output to the next owner |
| --- | --- | --- |
| `src/mdao_su2/run_config/` | Resolve includes and CLI overrides, construct typed values, check cross-field rules, and reject unknown keys. | A validated `RunConfig`. |
| `src/mdao_su2/geometry/` | Construct FFD, CST, Hicks–Henne, and twist motion; map design variables to wall coordinates and reverse sensitivities. | Global-ID-sorted wall motion and its Jacobian. |
| `src/mdao_su2/mesh/` | Read, generate, and admit meshes and geometry recipes. | A mesh with named markers and quality evidence. |
| `src/mdao_su2/solver/` | Emit SU2 configurations, couple `pysu2` and `pysu2ad`, and expose responses. | Flow responses and surface sensitivities. |
| `src/mdao_su2/optimization/` | Compile expressions, assemble the OpenMDAO model, route derivatives, and dispatch tasks. | Task results and optimizer evaluations. |
| `src/mdao_su2/run_output/` | Manage run directories, status, artifacts, and archives. | Inspectable run records. |
| `src/mdao_su2/postprocess/` | Read recorded output and make figures. | Figures and plot manifests. |

`run/optimize.py` calls `run_config/loader.py`. The loader merges included TOMLs, applies `--set`, constructs `RunConfig` in `root.py`, and calls `validate` in `loader.py`. Validation delegates to `checks/`. The `toggles.py` registry requires an enabled feature to state all the values it reads. `run/mesh.py`, `run/plot.py`, and `run/archive.py` provide separate mesh, plot, and archive commands.

The solver path starts with the chosen case's mesh, boundary roles, physics, points, and numerical settings. `solver/point_cfg_writer.py` writes primal and adjoint cfg files. `solver/su2_options.py` owns typed options, passthrough policy, and derivative evidence. `solver/response_catalog.py` defines responses. `solver/aero_comp.py`, `solver/builder.py`, and `solver/standalone/` are stable coupling interfaces: preserve their MPI, primal, and adjoint contracts when editing callers.

### Module lookup

| Question | Current owner |
| --- | --- |
| How are includes, overrides, and errors handled? | `run_config/loader.py` and `run_config/errors.py`. |
| Where are accepted fields and defaults declared? | `run_config/root.py` and `run_config/physics.py`. |
| Which combinations are refused? | `run_config/checks/`; `checks/task.py` owns derivative routing. |
| Which values must an enabled block state? | `run_config/toggles.py`. |
| How are multipoint values resolved? | `run_config/points.py` and `optimization/mission_points.py`. |
| Who builds and drives the OpenMDAO problem? | `optimization/model.py`, `optimization/tasks.py`, and `optimization/optimizer.py`. |
| Who compiles objective expressions and checks totals? | `optimization/expression.py` and `optimization/check_totals.py`. |
| Who emits SU2 options and cfg files? | `solver/su2_options.py`, `solver/su2_options_v85.py`, and `solver/point_cfg_writer.py`. |
| Who selects a response and couples the wall? | `solver/response_catalog.py`, `solver/builder.py`, `solver/seam_comp.py`, and `solver/aero_comp.py`. |
| Who owns each shape family? | `geometry/factory.py`, `geometry/ffd.py`, `geometry/cst.py`, `geometry/hicks_henne.py`, and `geometry/twist.py`. |
| Who admits a mesh and reads its recipe? | `mesh/admission.py`, `mesh/quality2d.py`, and `mesh/recipe.py`. |
| Who names artifacts and archives runs? | `run_output/layout.py`, `run_output/rundir.py`, and `run_output/archive.py`. |
| Who reads files and draws plots? | `postprocess/readers.py`, `postprocess/plots.py`, and `postprocess/figures/`. |

The loader validates point, objective, geometry, solver, remesh, execution, and file contracts before a task starts. It resolves includes in listed order and then applies the leaf file and CLI overrides. A user TOML can include other files; the shipped runnable TOMLs are self-contained. The loader limits an include chain to eight levels. It rejects unknown fields instead of silently ignoring them.

## Geometry, solver, and optimizer contracts

The design vector orders enabled shape variables, twist variables, and selected angle variables. Geometry computes wall motion from the undeformed reference surface. SU2 deforms the interior mesh. The reverse geometry Jacobian maps SU2 surface sensitivities back to the design vector. The global surface uses global-ID ordering; each MPI rank translates to its local marker indices. A node shared by two markers contributes one node sensitivity.

The primal uses `pysu2`; the discrete adjoint uses `pysu2ad`. Use one adjoint per functional and `MGLEVEL=0` for the adjoint. Primal and adjoint cfg files must agree on the mesh, marker roles, free stream, and relevant numerics. `run_config/checks/files.py` checks declared cfg parity. Cache pristine reference coordinates before an in-process adjoint, because later coordinate accessors can be invalid.

Hicks–Henne lower-surface positive amplitude moves outward, with the opposite signed y displacement from the upper surface. A half-model basis that reaches the symmetry plane pins symmetry nodes. Preserve bounds, scalers, and design-variable order when changing a parameterization. For angle derivatives, add the wind-axis projection to the flow-path partial. Transfer moment pivots before comparing a moment derivative with finite differences.

`optimization/tasks.py` dispatches `run_model`, `trim`, `compute_totals`, `check_totals`, `moment_pivot_check`, `run_driver`, and `polar`. `optimization/expression.py` compiles objective expressions; `optimization/model.py` assembles the model. A driver success requires final unscaled constraints and convergence evidence, not merely an optimizer exit code.

## Derivative routing and evidence

`run_config/checks/task.py` resolves `[execution] derivative_mode` before OpenMDAO setup. `rev` uses reverse products. A 2D fixed-mesh case with one design marker and no catalog responses can use `fwd` or `auto`. The forward product reuses cached SU2 adjoint rows; it is not a direct SU2 state-sensitivity solve. `auto` lets OpenMDAO choose the product direction on this path.

| Condition | Routed mode and component |
| --- | --- |
| A 2D fixed mesh, one design marker, no catalog responses, no parallel points | `rev` uses `SU2Builder`; `fwd` or `auto` uses `BidirSU2Builder`. FFD uses `FFDGeoBidirComp` for forward products. |
| A 3D wall, multiple design markers, or catalog responses | Reverse only; `ResponseSU2Builder` handles the response and marker seam. |
| Remeshing or `parallel_points` | Reverse only; a `fwd` or `auto` request warns and falls back to `rev`. |

The resolved configuration and task results record the requested mode, resolved mode, and fallback reason. A fallback changes product cost, not the mathematical total. One SU2 adjoint still runs per seeded functional per point in either OpenMDAO mode. The caller must preserve the geometry vector order and rank-independent surface layout on both routes.

The selected task, response, physics profile, and boundary family determine whether the model can use an adjoint. The registry status `fd_validated` applies only to recorded rows and conditions. `primal_checked` covers the primal; `assumed` does not pass a derivative gate. `fd_failed` and `unsupported` block an adjoint task. The registry requires cited evidence for any status beyond `assumed`. `check_totals` uses a cold finite-difference comparison with the selected TOML's steps and bands. Mostly tangential 2D wall motion is outside the reliable shape-derivative contract. Read excluded rows and the step sweep before accepting a derivative.

## Add a case, parameterization, or response

To add a case, create `cases/<case>/case.toml` and immutable files under `input/`, including a self-contained runnable file under `input/configs/`. Record mesh sources in its recipe. Run `python run/optimize.py --config <file> --print-config` to check resolution before any solve. Keep case values in TOML; shared Python owns behavior, not case-specific constants.

To add a parameterization, implement forward wall motion and its reverse derivative in `geometry/`, register it through `geometry/factory.py`, and state its ordering, sign, bounds, scaling, and symmetry behavior. To add a response, register it in `solver/response_catalog.py`, connect its objective expression, and state the derivative status in `solver/su2_options.py`. A validated derivative claim needs its own cold finite-difference gate with a stated band and exclusions.

Add a typed SU2 option through `solver/su2_options.py` and the cfg writer so primal and adjoint emission remains visible. Do not hide a runtime-owned option in a passthrough map. Add a task through the typed execution schema and `optimization/tasks.py`; add a new CLI flag at the owning `run/*.py` command.

## Placement and working rules

Use `snake_case` Python module names under the package that owns the behavior. Put runnable case TOMLs under `cases/<case>/input/configs/` and mesh recipes under `input/mesh/`. Plotting implementations belong in `postprocess/`; figures included in the guide belong in `docs/figures/`. Run records belong under a chosen output root. Inspect the selected TOML and `--help` before changing a command.

Source `activate.sh` before running Python commands. Use `bash verify.sh --lane fast` for bounded local checks and `tests/regression.py` for the regression script that lane invokes. The `solver` lane starts SU2 work; `all` includes it. Agents start solves only when asked. `setup.sh --clean` previews paths and requires `--yes` for deletion. `run/plot.py convert` converts a SU2 mesh for inspection. The [human guide](../human-guide.pdf) explains the current workflows and evidence limits.
