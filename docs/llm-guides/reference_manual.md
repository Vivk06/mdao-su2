# Reference manual

Use the selected TOML and command `--help` as the executable authority. This manual inventories the loader schema and CLI flags at this checkout, then gives the output, SU2, case, and derivative contracts. The [code manual](code_manual.md) explains ownership and extension points.

## Run configuration keys

The next table is generated from `run_config/root.py`, `physics.py`, the mission parser, boundary-family parameters, point freestream overrides, and `include` consumed by the loader. `validate` lives in `loader.py` and delegates to `checks/`; `toggles.py` adds conditions for enabled blocks. `source` and `overrides` are loader bookkeeping that the top-level parser also accepts; the loader normally supplies them. Defaults marked “see schema” use a factory or expression. Dynamic maps such as SU2 extra options, mission terms, FFD names, and recipe steps need their owning parser.

Generated key rows: **405**. The count is from the code-derived inventory used to build this table.

| TOML table | Key | Default or source |
| --- | --- | --- |
| `(top level)` | `case` | see schema |
| `(top level)` | `constraints` | see schema |
| `(top level)` | `deform` | see schema |
| `(top level)` | `dv` | see schema |
| `(top level)` | `execution` | see schema |
| `(top level)` | `geometry` | see schema |
| `(top level)` | `include` | optional; depth at most 8 |
| `(top level)` | `mission` | see schema |
| `(top level)` | `name` | run |
| `(top level)` | `objective` | see schema |
| `(top level)` | `optimizer` | see schema |
| `(top level)` | `output` | see schema |
| `(top level)` | `overrides` | () |
| `(top level)` | `points` | () |
| `(top level)` | `polar` | see schema |
| `(top level)` | `postprocess` | see schema |
| `(top level)` | `remesh` | see schema |
| `(top level)` | `schema_version` | see schema |
| `(top level)` | `solver` | see schema |
| `(top level)` | `source` |  |
| `(top level)` | `trim` | see schema |
| `(top level)` | `verification` | see schema |
| `case` | `boundary` | array of boundary tables |
| `case` | `freestream` | see schema |
| `case` | `mesh` |  |
| `case` | `name` |  |
| `case` | `ndim` | 2 |
| `case` | `physics` | see schema |
| `case` | `reference` | see schema |
| `case` | `root` |  |
| `case.boundary` | `bc` | see schema |
| `case.boundary` | `coefficient` | required by selected bc family |
| `case.boundary` | `direction` | required by selected bc family |
| `case.boundary` | `kind` | required by selected bc family |
| `case.boundary` | `markers` | see schema |
| `case.boundary` | `pressure` | required by selected bc family |
| `case.boundary` | `roles` | () |
| `case.boundary` | `temperature` | required by selected bc family |
| `case.boundary` | `total_pressure` | required by selected bc family |
| `case.boundary` | `total_temperature` | required by selected bc family |
| `case.boundary` | `value` | required by selected bc family |
| `case.boundary` | `values` | required by selected bc family |
| `case.boundary` | `velocity` | required by selected bc family |
| `case.freestream` | `dimensionalization` |  |
| `case.freestream` | `pressure` | 0.0 |
| `case.freestream` | `reynolds_length` | 1.0 |
| `case.freestream` | `reynolds_number` | see schema |
| `case.freestream` | `temperature` | 288.15 |
| `case.physics` | `gas` | see schema |
| `case.physics` | `kind` | rans_sa |
| `case.physics` | `transition` | see schema |
| `case.physics` | `turb2lam_viscosity_ratio` | see schema |
| `case.physics` | `turbulence` | see schema |
| `case.physics` | `turbulence_intensity` | see schema |
| `case.physics.gas` | `acentric_factor` | see schema |
| `case.physics.gas` | `conductivity_model` | see schema |
| `case.physics.gas` | `fluid_model` | see schema |
| `case.physics.gas` | `freestream_option` | see schema |
| `case.physics.gas` | `gamma` | see schema |
| `case.physics.gas` | `gas_constant` | see schema |
| `case.physics.gas` | `init_option` | see schema |
| `case.physics.gas` | `molecular_weight` | see schema |
| `case.physics.gas` | `mu_ref` | see schema |
| `case.physics.gas` | `mu_t_ref` | see schema |
| `case.physics.gas` | `prandtl_lam` | see schema |
| `case.physics.gas` | `prandtl_turb` | see schema |
| `case.physics.gas` | `specific_heat_cp` | see schema |
| `case.physics.gas` | `sutherland_constant` | see schema |
| `case.physics.gas` | `turbulent_conductivity_model` | see schema |
| `case.physics.gas` | `viscosity_model` | see schema |
| `case.physics.transition` | `model` | NONE |
| `case.physics.transition` | `options` | () |
| `case.physics.turbulence` | `model` | SA |
| `case.physics.turbulence` | `options` | () |
| `case.reference` | `area` | 1.0 |
| `case.reference` | `length` | 1.0 |
| `case.reference` | `lift_axis` | see schema |
| `case.reference` | `origin_moment` | (0.25, 0.0, 0.0) |
| `case.reference` | `span_axis` | see schema |
| `constraints` | `curvature` | see schema |
| `constraints` | `geometry` | False |
| `constraints` | `le_radius` | see schema |
| `constraints` | `lift` | see schema |
| `constraints` | `linear` | () |
| `constraints` | `moment` | see schema |
| `constraints` | `thickness` | see schema |
| `constraints` | `volume` | see schema |
| `constraints.curvature` | `curvature_type` | mean |
| `constraints.curvature` | `enabled` | False |
| `constraints.curvature` | `end_x` | 0.16 |
| `constraints.curvature` | `ks_coeff` | 1.0 |
| `constraints.curvature` | `n_pts` | 17 |
| `constraints.curvature` | `scaler` | 0.0012 |
| `constraints.curvature` | `start_x` | 0.01 |
| `constraints.curvature` | `upper` | 3.0 |
| `constraints.le_radius` | `axis` | (0.0, 1.0, 0.0) |
| `constraints.le_radius` | `chord_dir` | (-1.0, 0.0, 0.0) |
| `constraints.le_radius` | `enabled` | False |
| `constraints.le_radius` | `le_x` | 0.01 |
| `constraints.le_radius` | `lower` | 0.8 |
| `constraints.le_radius` | `nSpan` | 2 |
| `constraints.lift` | `enabled` | False |
| `constraints.lift` | `equals` | see schema |
| `constraints.lift` | `kind` | see schema |
| `constraints.lift` | `lower` | see schema |
| `constraints.lift` | `points` | see schema |
| `constraints.lift` | `scaler` | 1.0 |
| `constraints.lift` | `upper` | see schema |
| `constraints.linear` | `equals` | see schema |
| `constraints.linear` | `lower` | see schema |
| `constraints.linear` | `name` |  |
| `constraints.linear` | `offset` | 0.0 |
| `constraints.linear` | `scaler` | see schema |
| `constraints.linear` | `terms` | () |
| `constraints.linear` | `upper` | see schema |
| `constraints.linear.terms` | `coeff` | see schema |
| `constraints.linear.terms` | `point` | see schema |
| `constraints.linear.terms` | `response` | see schema |
| `constraints.moment` | `axis` | z |
| `constraints.moment` | `enabled` | False |
| `constraints.moment` | `equals` | 0.0 |
| `constraints.moment` | `kind` | min |
| `constraints.moment` | `lower` | 0.0 |
| `constraints.moment` | `pivot` | see schema |
| `constraints.moment` | `points` | see schema |
| `constraints.moment` | `scaler` | 1.0 |
| `constraints.moment` | `upper` | 0.0 |
| `constraints.thickness` | `axes` | see schema |
| `constraints.thickness` | `enabled` | False |
| `constraints.thickness` | `kind` | min |
| `constraints.thickness` | `nChord` | 2 |
| `constraints.thickness` | `nSpan` | 2 |
| `constraints.thickness` | `probe_x` | see schema |
| `constraints.thickness` | `scaled` | False |
| `constraints.thickness` | `span_offsets` | (0.001, 0.006) |
| `constraints.thickness` | `stations` | see schema |
| `constraints.thickness` | `te_offset` | 0.05 |
| `constraints.volume` | `enabled` | False |
| `constraints.volume` | `le_x` | 0.01 |
| `constraints.volume` | `lower` | 1.0 |
| `constraints.volume` | `nChord` | 10 |
| `constraints.volume` | `nSpan` | 2 |
| `constraints.volume` | `te_x` | 0.99 |
| `deform` | `console_output` | False |
| `deform` | `linear_solver` | FGMRES |
| `deform` | `linear_solver_error` | 1e-14 |
| `deform` | `linear_solver_iter` | 1000 |
| `deform` | `linear_solver_prec` | ILU |
| `deform` | `nonlinear_iter` | 1 |
| `deform` | `stiffness_type` | WALL_DISTANCE |
| `dv` | `alpha` | see schema |
| `dv` | `shape` | see schema |
| `dv` | `start` |  |
| `dv` | `twist` | see schema |
| `dv.alpha` | `per_point` | True |
| `dv.shape` | `bound` | 0.05 |
| `dv.shape` | `bound` | 0.05 |
| `dv.shape` | `bounds` | () |
| `dv.shape` | `bump_width` | 3 |
| `dv.shape` | `degree` | 3 |
| `dv.shape` | `direction` | y |
| `dv.shape` | `directions` | () |
| `dv.shape` | `distribution` | even |
| `dv.shape` | `enabled` | False |
| `dv.shape` | `enabled` | False |
| `dv.shape` | `free_te` | False |
| `dv.shape` | `freeze_i` | () |
| `dv.shape` | `freeze_j` | () |
| `dv.shape` | `freeze_k` | () |
| `dv.shape` | `freeze_points` | () |
| `dv.shape` | `link_k` | True |
| `dv.shape` | `n_basis` | 0 |
| `dv.shape` | `n_dv` | 20 |
| `dv.shape` | `n_le` | 0 |
| `dv.shape` | `outward_only` | False |
| `dv.shape` | `pattern` | control_point_2d |
| `dv.shape` | `scaler` | auto |
| `dv.shape` | `scaler` | auto |
| `dv.twist` | `axis_x_fraction` | see schema |
| `dv.twist` | `enabled` | False |
| `dv.twist` | `include_root` | see schema |
| `dv.twist` | `init_deg` | see schema |
| `dv.twist` | `lower_deg` | see schema |
| `dv.twist` | `n` | see schema |
| `dv.twist` | `scaler` | see schema |
| `dv.twist` | `upper_deg` | see schema |
| `execution` | `adjoint_restart` | True |
| `execution` | `clean_restart` | True |
| `execution` | `derivative_mode` | rev |
| `execution` | `parallel_points` | False |
| `execution` | `task` | run_model |
| `geometry` | `dat_file` |  |
| `geometry` | `ffd` |  |
| `geometry` | `ffd_files` | see schema |
| `geometry` | `ffd_order` | pygeo_default |
| `geometry` | `n_cst` | 8 |
| `geometry` | `parameterization` | ffd |
| `geometry` | `pin_symmetry` | False |
| `geometry` | `zspan` | 0.5 |
| `mission` | `altitude` | see mission_points.py |
| `mission` | `atmosphere` | see mission_points.py |
| `mission` | `auw` | see mission_points.py |
| `mission` | `bracket` | see mission_points.py |
| `mission` | `bracket_scaler` | see mission_points.py |
| `mission` | `cl_at` | see mission_points.py |
| `mission` | `cref` | see mission_points.py |
| `mission` | `level_flight` | see mission_points.py |
| `mission` | `level_scaler` | see mission_points.py |
| `mission` | `mach` | see mission_points.py |
| `mission` | `mach_tolerance` | see mission_points.py |
| `mission` | `point_defaults` | see mission_points.py |
| `mission` | `sref` | see mission_points.py |
| `mission.atmosphere` | `gamma` | GAMMA |
| `mission.atmosphere` | `gas_constant` | R_AIR |
| `mission.atmosphere` | `gravity` | G0 |
| `mission.atmosphere` | `lapse_rate` | LAPSE |
| `mission.atmosphere` | `reference_temperature` | T_MU0 |
| `mission.atmosphere` | `reference_viscosity` | MU0 |
| `mission.atmosphere` | `sea_level_pressure` | P0 |
| `mission.atmosphere` | `sea_level_temperature` | T0 |
| `mission.atmosphere` | `sutherland_constant` | SUTHERLAND |
| `mission.atmosphere` | `tropopause_altitude` | 11000.0 |
| `mission.point_defaults` | `aoa_init` | see mission_points.py |
| `mission.point_defaults` | `aoa_lower` | see mission_points.py |
| `mission.point_defaults` | `aoa_upper` | see mission_points.py |
| `mission.point_defaults` | `weight` | see mission_points.py |
| `objective` | `expression` | see schema |
| `objective` | `parameters` | see schema |
| `objective` | `scaler` | see schema |
| `optimizer` | `debug_print` | see schema |
| `optimizer` | `hist_file` |  |
| `optimizer` | `name` | SLSQP |
| `optimizer` | `print_opt_prob` | True |
| `optimizer` | `settings` | see schema |
| `output` | `on_exists` | fail |
| `output` | `root` | output |
| `output` | `run_id` | auto |
| `output` | `seed` | () |
| `output` | `snapshots` | see schema |
| `output` | `solver_stdout` | log |
| `output` | `su2` | see schema |
| `output.seed` | `from` |  |
| `output.seed` | `kind` |  |
| `output.seed` | `points` | () |
| `output.snapshots` | `every_n` | 1 |
| `output.snapshots` | `keep_volume` | all |
| `output.snapshots` | `mode` | surface |
| `output.su2` | `history_output_adjoint` | ITER, RMS_RES, SENS_AOA, LINSOL_RESIDUAL, LINSOL_ITER |
| `output.su2` | `history_output_primal` | ITER, RMS_RES, AOA, LIFT, DRAG, EFFICIENCY |
| `output.su2` | `mesh_format` | SU2 |
| `output.su2` | `output_files` | see schema |
| `output.su2` | `restart_basename` | restart_flow |
| `output.su2` | `screen_output_adjoint` | INNER_ITER, RMS_RES, SENS_AOA, LINSOL_RESIDUAL, LINSOL_ITER |
| `output.su2` | `screen_output_primal` | INNER_ITER, RMS_RES, LIFT, DRAG |
| `output.su2` | `surface_basename` | surface_flow |
| `output.su2` | `tabular_format` | CSV |
| `output.su2` | `volume_basename` | flow |
| `output.su2` | `volume_output` | see schema |
| `output.su2` | `wrt_freq` | 10000 |
| `points` | `alpha` | design |
| `points` | `aoa_init` | 0.0 |
| `points` | `aoa_lower` | 0.0 |
| `points` | `aoa_scaler` | 1.0 |
| `points` | `aoa_upper` | 0.0 |
| `points` | `freestream` | see schema |
| `points` | `id` | see schema |
| `points` | `mach` | see schema |
| `points` | `responses` | () |
| `points` | `start_cl` | see schema |
| `points` | `target_cl` | see schema |
| `points` | `weight` | 1.0 |
| `points.freestream` | `dimensionalization` | overrides case.freestream |
| `points.freestream` | `pressure` | overrides case.freestream |
| `points.freestream` | `reynolds_length` | overrides case.freestream |
| `points.freestream` | `reynolds_number` | overrides case.freestream |
| `points.freestream` | `temperature` | overrides case.freestream |
| `polar` | `alpha` | () |
| `polar` | `cl_band` | 0.0 |
| `polar` | `enabled` | False |
| `polar` | `mach_floor` | 0.0 |
| `polar` | `run_urans` | False |
| `polar` | `urans` | see schema |
| `polar` | `urans_max_angles` | 0 |
| `polar` | `urans_min_cycles` | 0 |
| `postprocess` | `span_load` | see schema |
| `postprocess.span_load` | `aspect_ratio` | see schema |
| `postprocess.span_load` | `cd0` | see schema |
| `postprocess.span_load` | `enabled` | False |
| `postprocess.span_load` | `lift_axis` | see schema |
| `postprocess.span_load` | `n_stations` | see schema |
| `postprocess.span_load` | `semispan` | see schema |
| `postprocess.span_load` | `span_axis` | see schema |
| `remesh` | `bl_thickness` | 0.0 |
| `remesh` | `fan_angle_deg` | 60.0 |
| `remesh` | `fan_elements` | 7 |
| `remesh` | `far_growth_slope` | 0.07 |
| `remesh` | `farfield_meshsize` | 15.0 |
| `remesh` | `farfield_radius` | 100.0 |
| `remesh` | `first_cell_height` | 2e-06 |
| `remesh` | `growth_ratio` | 1.2 |
| `remesh` | `mesh_output` | remesh/mesh_current_{id}.su2 |
| `remesh` | `mode` | none |
| `remesh` | `n_bl_layers` | 25 |
| `remesh` | `near_meshsize` | 0.01 |
| `remesh` | `quality` | see schema |
| `remesh` | `solver` | see schema |
| `remesh` | `verbose` | False |
| `remesh` | `viewable_mesh` | () |
| `remesh` | `wake_angles_deg` | () |
| `remesh` | `wake_length` | 0.0 |
| `remesh` | `wake_meshsize` | 0.0 |
| `remesh` | `wake_stream_slope` | 0.0 |
| `remesh` | `wall_meshsize` | 0.0 |
| `remesh.quality` | `max_aspect_ratio_p99` | 2000.0 |
| `remesh.quality` | `max_bl_thickness_ratio` | 2.0 |
| `remesh.quality` | `max_first_cell_height_ratio` | 2.0 |
| `remesh.quality` | `max_skewness` | 0.85 |
| `remesh.quality` | `min_bl_thickness_ratio` | 0.5 |
| `remesh.quality` | `min_first_cell_height_ratio` | 0.5 |
| `solver` | `adjoint` | see schema |
| `solver` | `adjoint_conv_residual_minval` | -6.0 |
| `solver` | `adjoint_iter` | 30000 |
| `solver` | `adjoint_mglevel` | 0 |
| `solver` | `adjoint_settings` | see schema |
| `solver` | `adjoint_stall_delta` | 0.1 |
| `solver` | `adjoint_stall_window` | 2000 |
| `solver` | `cfl_adapt_param` | ( 0.1, 2.0, 1.0, 1000.0 ) |
| `solver` | `convergence` | see schema |
| `solver` | `linear` | see schema |
| `solver` | `multigrid` | see schema |
| `solver` | `numerics` | see schema |
| `solver` | `primal` | see schema |
| `solver` | `primal_conv_residual_minval` | -9.0 |
| `solver` | `primal_iter` | 50000 |
| `solver` | `primal_mglevel` | 0 |
| `solver` | `su2_extra` | see schema |
| `solver.adjoint_settings` | `frozen_limiter_disc` | False |
| `solver.adjoint_settings` | `frozen_visc_disc` | False |
| `solver.adjoint_settings` | `objective_function` | DRAG |
| `solver.adjoint_settings` | `relaxation_factor` | 1.0 |
| `solver.convergence` | `cauchy_elems` | 100 |
| `solver.convergence` | `cauchy_eps` | 1E-6 |
| `solver.convergence` | `startiter` | 10 |
| `solver.linear` | `error` | 1E-1 |
| `solver.linear` | `iter` | 3 |
| `solver.linear` | `prec` | ILU |
| `solver.linear` | `solver` | FGMRES |
| `solver.multigrid` | `correction_smooth` | 1 |
| `solver.multigrid` | `cycle` | V_CYCLE |
| `solver.multigrid` | `damp_prolongation` | 0.5 |
| `solver.multigrid` | `damp_restriction` | 0.5 |
| `solver.multigrid` | `post_smooth` | 4 |
| `solver.multigrid` | `pre_smooth` | 4 |
| `solver.numerics` | `cfl_number` | 4.0 |
| `solver.numerics` | `conv_method_flow` | JST |
| `solver.numerics` | `conv_method_turb` | SCALAR_UPWIND |
| `solver.numerics` | `jst_sensor_coeff` | ( 0.5, 0.02 ) |
| `solver.numerics` | `muscl_flow` | NO |
| `solver.numerics` | `muscl_turb` | NO |
| `solver.numerics` | `num_method_grad` | WEIGHTED_LEAST_SQUARES |
| `solver.numerics` | `slope_limiter_flow` | VENKATAKRISHNAN |
| `solver.numerics` | `slope_limiter_turb` | VENKATAKRISHNAN |
| `solver.numerics` | `time_discre_flow` | EULER_IMPLICIT |
| `solver.numerics` | `time_discre_turb` | EULER_IMPLICIT |
| `solver.numerics` | `venkat_limiter_coeff` | 0.03 |
| `solver.primal or solver.adjoint` | `cfl_adapt` | True |
| `solver.primal or solver.adjoint` | `cfl_adapt_param` | ( 0.1, 2.0, 1.0, 100.0 ) |
| `solver.primal or solver.adjoint` | `conv_residual_minval` | -9.0 |
| `solver.primal or solver.adjoint` | `iter` | 10000 |
| `solver.primal or solver.adjoint` | `mglevel` | 3 |
| `solver.primal or solver.adjoint` | `warm_start` | True |
| `trim` | `alpha_clip` | (0.0, 5.0) |
| `trim` | `dcl_da_init` | 0.11 |
| `trim` | `enabled` | False |
| `trim` | `max_iter` | 6 |
| `trim` | `tol` | 0.0001 |
| `verification` | `alpha_band` | see schema |
| `verification` | `alpha_band_moment` | 0.0 |
| `verification` | `alpha_band_moment_pivot` | 0.0 |
| `verification` | `alpha_outputs` | () |
| `verification` | `alpha_step` | 0.001 |
| `verification` | `cancellation_cr` | see schema |
| `verification` | `cancellation_cr_param` |  |
| `verification` | `form` | central |
| `verification` | `moment_pivot` | see schema |
| `verification` | `record_shape_sensitivities` | False |
| `verification` | `shape_band` | see schema |
| `verification` | `shape_band_cap` | 1.0 |
| `verification` | `shape_band_floor` | see schema |
| `verification` | `shape_band_slope` | see schema |
| `verification` | `shape_outputs` | () |
| `verification` | `shape_rows_unverifiable` | see schema |
| `verification` | `shape_step` | 1e-06 |
| `verification` | `step_calc` | abs |
| `verification` | `twist_step` | see schema |
| `verification.moment_pivot` | `alpha_deg` | 0.0 |
| `verification.moment_pivot` | `band_pivot` | 0.0001 |
| `verification.moment_pivot` | `cd` | 0.0 |
| `verification.moment_pivot` | `cl` | 0.0 |
| `verification.moment_pivot` | `dcd_dalpha` | 0.0 |
| `verification.moment_pivot` | `dcl_dalpha` | 0.0 |
| `verification.moment_pivot` | `dcm_dalpha_ref` | 0.0 |
| `verification.moment_pivot` | `enabled` | False |
| `verification.moment_pivot` | `reference_pivot_x` | 0.25 |
| `verification.moment_pivot` | `reference_run` |  |

### Toggle requirements

A missing toggle block or `enabled = false` leaves that feature off. When enabled, `toggles.py` requires every field read by that feature. The geometry constraint switch accepts a boolean or an `enabled` table. Mesh recipe steps have their own toggle scope. Safety checks for mesh admission, adjoint stalls, and derivative status have no off switch. `schema_version` must match the version in `root.py`; the loader rejects unknown keys. `include` has an eight-level depth limit and merges tables before CLI overrides.

### Dynamic configuration entries

The 405 generated rows are fixed schema fields. Some accepted entries use names supplied by a case or by another parser:

| TOML location | Accepted entries and owner |
| --- | --- |
| `geometry.ffd_files` | FFD box names map to file paths; `run_config/checks/geometry.py` validates the selected geometry. |
| `optimizer.options` | Optimizer-specific option names pass to the selected optimizer; `run_config/checks/task.py` checks the optimizer choice. |
| `solver.primal`, `solver.adjoint` | Each accepts the common controls shown once in the generated table as `solver.primal or solver.adjoint`. Extra SU2 option names are checked by `solver/su2_options.py`. |
| `verification.cancellation_cr`, `verification.shape_rows_unverifiable` | Parameterization and response-specific row maps are read by the derivative check. The selected verification TOML supplies values. |
| `mission` terms and `remesh.recipe` steps | Term and recipe parsers own their entries, with per-step toggle conditions. |

`[[output.seed]]` uses the TOML key `from`; the loader stores it in the `SeedConfig.source` field. `points` maps destination point IDs to source point IDs.

## CLI flags

The following **175** argparse actions are generated from `add_argument` calls in `run/*.py` and `tools/*.py`. `run/mesh.py` also constructs three axis-bound flags in a loop; those are included. The same spelling can appear in several subcommands. Use the command’s `--help` for positionals, required combinations, choices, and examples.

| Command source | Flags |
| --- | --- |
| `run/archive.py:25` | `--run` |
| `run/archive.py:27` | `--verify-only` |
| `run/archive.py:29` | `--output`, `-o` |
| `run/archive.py:31` | `--include-work` |
| `run/mesh.py:64` | `--config` |
| `run/mesh.py:66` | `--enable` |
| `run/mesh.py:69` | `--level` |
| `run/mesh.py:70` | `-o`, `--output` |
| `run/mesh.py:74` | `--quality` |
| `run/mesh.py:79` | `--level-scale` |
| `run/mesh.py:90` | `--mesh` |
| `run/mesh.py:92` | `--markers` |
| `run/mesh.py:97` | `--source` |
| `run/mesh.py:98` | `--n-points` |
| `run/mesh.py:99` | `--spacing` |
| `run/mesh.py:100` | `--te` |
| `run/mesh.py:101` | `--le-spacing` |
| `run/mesh.py:102` | `--te-spacing` |
| `run/mesh.py:103` | `--spacing-ratio` |
| `run/mesh.py:108` | `-o`, `--output` |
| `run/mesh.py:109` | `--box` |
| `run/mesh.py:110` | `--verify` |
| `run/mesh.py:112` | `--tol` |
| `run/mesh.py:116` | `--wall` |
| `run/mesh.py:117` | `--continuity` |
| `run/mesh.py:119` | `--symmetry-plane` |
| `run/mesh.py:120` | `--box` |
| `run/mesh.py:121` | `--json` |
| `run/mesh.py:125` | `--ffd` |
| `run/mesh.py:126` | `--box` |
| `run/mesh.py:127` | `--tol` |
| `run/mesh.py:129` | `--kmax` |
| `run/mesh.py:130` | `--enclosure` |
| `run/mesh.py:131` | `--json` |
| `run/mesh.py:134` | `--kind` |
| `run/mesh.py:135` | `--nx` |
| `run/mesh.py:136` | `--spacing` |
| `run/mesh.py:137` | `--zspan` |
| `run/mesh.py:139` | `--x-bounds` |
| `run/mesh.py:139` | `--y-bounds` |
| `run/mesh.py:139` | `--z-bounds` |
| `run/mesh.py:140` | `--x-margin` |
| `run/mesh.py:141` | `--y-margin` |
| `run/mesh.py:142` | `--le-te` |
| `run/mesh.py:143` | `--n-span` |
| `run/mesh.py:144` | `--nz` |
| `run/mesh.py:145` | `--surface` |
| `run/mesh.py:149` | `--surface` |
| `run/mesh.py:160` | `--wake-angles-deg` |
| `run/mesh.py:161` | `--farfield-center` |
| `run/mesh.py:164` | `--view` |
| `run/mesh.py:166` | `--verbose` |
| `run/mesh.py:167` | `--json` |
| `run/mesh.py:171` | `--surface` |
| `run/mesh.py:172` | `--level` |
| `run/mesh.py:176` | `--edge-distance` |
| `run/mesh.py:178` | `--tip` |
| `run/mesh.py:179` | `--farfield-center` |
| `run/mesh.py:184` | `--threads` |
| `run/mesh.py:186` | `--verbose` |
| `run/mesh.py:187` | `--json` |
| `run/mesh.py:192` | `--surface` |
| `run/mesh.py:193` | `--level` |
| `run/mesh.py:196` | `--farfield-center` |
| `run/mesh.py:197` | `--wall-marker` |
| `run/mesh.py:198` | `--farfield-marker` |
| `run/mesh.py:201` | `--verbose` |
| `run/mesh.py:202` | `--json` |
| `run/mesh.py:207` | `-o`, `--output` |
| `run/mesh.py:208` | `--boundary` |
| `run/mesh.py:211` | `--markers` |
| `run/mesh.py:213` | `--json` |
| `run/mesh.py:216` | `--config` |
| `run/mesh.py:217` | `--output-dir` |
| `run/mesh.py:219` | `--surface` |
| `run/mesh.py:220` | `--level` |
| `run/mesh.py:222` | `--view` |
| `run/optimize.py:28` | `--config` |
| `run/optimize.py:29` | `--task` |
| `run/optimize.py:30` | `--optimizer` |
| `run/optimize.py:31` | `--cl-target` |
| `run/optimize.py:32` | `--remesh` |
| `run/optimize.py:33` | `--ffd` |
| `run/optimize.py:34` | `--run-id` |
| `run/optimize.py:35` | `--output-root` |
| `run/optimize.py:36` | `--on-exists` |
| `run/optimize.py:38` | `--seed-from` |
| `run/optimize.py:43` | `--parallel-points` |
| `run/optimize.py:44` | `--no-trim` |
| `run/optimize.py:45` | `--no-constraints` |
| `run/optimize.py:46` | `--set` |
| `run/optimize.py:50` | `--band` |
| `run/optimize.py:55` | `--print-config` |
| `run/optimize.py:58` | `--archive` |
| `run/plot.py:66` | `-o`, `--out` |
| `run/plot.py:67` | `--format` |
| `run/plot.py:69` | `--dpi` |
| `run/plot.py:70` | `--figsize` |
| `run/plot.py:72` | `--colors` |
| `run/plot.py:73` | `--styles` |
| `run/plot.py:75` | `--palette` |
| `run/plot.py:77` | `--legend-out` |
| `run/plot.py:78` | `--xlim` |
| `run/plot.py:79` | `--ylim` |
| `run/plot.py:81` | `--figures` |
| `run/plot.py:86` | `--file` |
| `run/plot.py:88` | `--label` |
| `run/plot.py:94` | `--mach` |
| `run/plot.py:95` | `--gamma` |
| `run/plot.py:96` | `--cfg` |
| `run/plot.py:115` | `--curve` |
| `run/plot.py:117` | `--sample` |
| `run/plot.py:122` | `--cfg` |
| `run/plot.py:124` | `--row` |
| `run/plot.py:224` | `--reference` |
| `run/plot.py:227` | `--reference-label` |
| `run/plot.py:229` | `--field` |
| `run/plot.py:230` | `--stations` |
| `run/plot.py:231` | `--span-axis` |
| `run/plot.py:232` | `--lift-axis` |
| `run/plot.py:233` | `--aoa` |
| `run/plot.py:236` | `--ref-area` |
| `run/plot.py:238` | `--semispan` |
| `run/plot.py:240` | `--n-stations` |
| `run/plot.py:241` | `--quantities` |
| `run/plot.py:243` | `--displacement-scale` |
| `run/plot.py:253` | `--baseline` |
| `run/plot.py:254` | `--field` |
| `run/plot.py:255` | `--lift-axis` |
| `run/plot.py:259` | `--field` |
| `run/plot.py:260` | `--wall` |
| `run/plot.py:262` | `--length` |
| `run/plot.py:263` | `--margin` |
| `run/plot.py:264` | `--levels` |
| `run/plot.py:270` | `--columns` |
| `run/plot.py:273` | `--x` |
| `run/plot.py:275` | `--floor` |
| `run/plot.py:276` | `--floor-cfg` |
| `run/plot.py:277` | `--adjoint-column` |
| `run/plot.py:282` | `--constraints` |
| `run/plot.py:284` | `--dv` |
| `run/plot.py:285` | `--crest` |
| `run/plot.py:286` | `--n-dv` |
| `run/plot.py:287` | `--distribution` |
| `run/plot.py:288` | `--bounds` |
| `run/plot.py:289` | `--unit` |
| `run/plot.py:290` | `--iteration` |
| `run/plot.py:294` | `--reference` |
| `run/plot.py:298` | `--label` |
| `run/plot.py:301` | `--x` |
| `run/plot.py:302` | `--y` |
| `run/plot.py:304` | `--peak` |
| `run/plot.py:306` | `--log-y` |
| `run/plot.py:308` | `--x-label` |
| `run/plot.py:309` | `--y-label` |
| `run/plot.py:314` | `--label` |
| `run/plot.py:316` | `--expr` |
| `run/plot.py:318` | `--margin` |
| `run/plot.py:321` | `--absolute` |
| `run/plot.py:322` | `--y-label` |
| `run/plot.py:327` | `--volume` |
| `run/plot.py:328` | `--field` |
| `run/plot.py:329` | `--ndim` |
| `run/plot.py:330` | `--probe-line` |
| `run/plot.py:331` | `--probe-box` |
| `run/plot.py:332` | `--probe-plane` |
| `run/plot.py:333` | `--normal` |
| `run/plot.py:334` | `--n` |
| `run/plot.py:335` | `--name` |
| `run/plot.py:336` | `--data-only` |
| `run/plot.py:339` | `--config` |
| `run/plot.py:340` | `--wall` |
| `run/plot.py:341` | `--dvs` |
| `run/plot.py:345` | `-o`, `--out` |
| `run/plot.py:347` | `--marker` |

## Run output layout

The default root is `cases/<case>/output/<run_id>/`; `--output-root` or `[output] root` redirects it. A run writes `STATUS`, `run.toml`, `resolved_config.json`, `logs/`, `results_<task>.json`, and `final/` artifacts. It uses `su2_workdir/` while solving and may remove that directory after retaining final artifacts. A failed run can keep it for diagnosis. A `run_driver` task also writes `opt_hst/` and `evaluations/`: a baseline, calls under `evaluations/calls/call_NNNN/`, iterations under `evaluations/iterations/iter_NNN/`, and a final evaluation. Stage folders use `primal` or an adjoint response name, with a point suffix for multipoint runs. `run_output/layout.py` defines artifact names: `history.csv`, `surface.vtu`, `volume.vtu`, `restart.dat`, and `config.cfg`; the final primal uses `primal_history.csv` and `primal.cfg`. Volume retention depends on `[output.snapshots]`.

`run/archive.py` creates a ZIP and SHA-256 manifest. Verify it with `--verify-only` before removing a run. `run/plot.py` reads explicit files and writes figures and a manifest; `convert` creates VTU from a SU2 mesh without a solve. A `DONE` status says the task completed, not that its mesh or physical result was validated.

## SU2 options and physics

`solver/su2_options.py` declares typed options, passthrough limits, boundary families, and derivative evidence. `solver/su2_options_v85.py` holds generated SU2 v8.5 enumerations and defaults. `solver/point_cfg_writer.py` emits primal and adjoint cfg files from one resolved configuration; `run_config/checks/files.py` checks allowed differences. The adjoint uses `pysu2ad`, one functional per solve, and `MGLEVEL=0`. The primal and adjoint share mesh, marker roles, free stream, and relevant numerical settings.

The shipped physics includes Euler and RANS with Spalart–Allmaras turbulence. SA–BCM transition has recorded primal and analytic totals but no finite-difference derivative validation. Langtry–Menter transition is primal-only on pinned SU2 v8.5 because its discrete-adjoint setup lacks a transition solver slot. A response status belongs to a physics profile, response, and boundary family. `fd_validated` covers recorded finite-difference rows; `primal_checked` covers a primal; `assumed` is not a derivative gate. `fd_failed` and `unsupported` block an adjoint task.

### Option ownership and cfg parity

The generated SU2 v8.5 registry contains 736 parsed option keys, 53 `MARKER_*` families, and 107 enum maps. Those counts come from `OPTION_KEYS`, `MARKER_FAMILIES`, and `ENUMS` in `solver/su2_options_v85.py`. `solver/su2_options.py` maps typed keys to TOML paths and assigns each one a side and parity rule. The writer sets runtime-owned keys, including mesh and point conditions, from the resolved run. `[solver.su2_extra]` admits declared passthrough keys; physics-changing passthrough values must agree on both sides. `run_config/checks/files.py` compares emitted primal and adjoint cfg files before a solve. A cfg example proves accepted syntax, not a derivative gate.

`[[case.boundary]]` gives every mesh marker one boundary condition. Its `bc` selects a family from `BC_FAMILIES`; that family determines any required parameter keys. A design role needs a wall boundary. A `deform_sym` role needs a symmetry boundary. The loader rejects duplicate markers and unknown family parameters. The boundary-family parameter names appear in the generated key table above.

### Flow and sensitivity contracts

At each operating point, SU2 solves a discrete residual `R(U, X(d), alpha) = 0`. The response `J` depends on the flow state `U`, wall coordinates `X`, and possibly angle `alpha`. Euler and RANS choices, turbulence and transition options, freestream values, boundary roles, reference quantities, numerics, and solver caps come from the run TOML. The cfg writer must send the same discretized problem to primal and adjoint. The adjoint uses one functional per solve and `MGLEVEL=0`; the primal may use multigrid.

The reverse total derivative uses `R_U^T psi = J_U^T` and `dJ/dd = J_d - psi^T R_d`. Geometry then maps the wall sensitivity through `dX/dd`. A forward OpenMDAO product uses cached adjoint rows on the supported 2D fixed-mesh path; it does not solve a direct SU2 state-sensitivity equation. A 3D wall, multiple design markers, catalog responses, remeshing, or parallel points route `fwd` and `auto` requests to `rev` with a warning. `run_config/checks/task.py` owns that routing. The resolved configuration records requested and routed modes and the reason.

Angle-of-attack totals add wind-axis projection to SU2's flow-path partial. For drag with angle measured in degrees, the projection includes approximately `C_L*pi/180`. Moment comparisons must use the same pivot. A two-dimensional wall motion dominated by tangential displacement changes point labels more than shape, so its adjoint shape row is outside the reliable finite-difference contract.

### Finite-difference bands and limits

`check_totals` runs cold perturbed primals. A central difference uses two perturbations per design variable. Its useful step lies on a stable adjacent-step plateau: small steps expose solver noise, while large steps expose truncation error. The default `shape_step` is `1e-6`; the default `alpha_step` is `1e-3` degrees. A gate TOML, not those defaults, supplies the acceptance bands and any row exclusions.

For per-node derivative contributions `Gamma_n`, the cancellation ratio is `C_r = sum(abs(Gamma_n)) / abs(sum(Gamma_n))`. When positive and negative node terms nearly cancel, a small per-node error can become a large relative row error. The recommended shape band is `max(1e-3, 2e-3*C_r)` with a cap of `1.0`; the recommended angle band is `1e-4`. A band at or above the cap cannot distinguish a zeroed or doubled row and is not a useful gate. `verification.cancellation_cr_param` ties a measured ratio table to one parameterization. Twist rows use their own stated band unless a matching ratio table exists.

The row metric is `abs(adjoint - FD) / max(abs(FD), 1e-30)`. A row outside its band can be inconclusive if the finite-difference resolution from the final primal drift is too large. A converged primal does not guarantee adjoint convergence: the adjoint runs on a single grid and must reach its own residual floor. An optimizer exit does not establish grid convergence, physical validation, or final unscaled feasibility.

### Current numeric checks and their conditions

| Check | Condition and owner |
| --- | --- |
| Point Mach number | `run_config/checks/points.py` requires `0 < mach < 1.5` for each configured point. |
| CST surface match | `run_config/checks/files.py` requires each matching wall node to lie within `1e-6` times `[case.reference] length` of the selected `.dat` surface. This applies when CST uses a surface file. |
| Default two-dimensional remesh quality | `RemeshQualityConfig` sets first-cell-height and boundary-layer-thickness ratios to `0.5` through `2.0`, maximum skewness to `0.85`, and the 99th-percentile aspect-ratio limit to `2000`. An enabled recipe must state its own values; the admitted mesh file must pass them. |
| RAE 2822 shape and angle gate | `cases/rae2822_rans_sa_hh/input/configs/fd_check_totals.toml` states shape floor `0.001`, slope `0.002*C_r`, and angle band `0.00012`. Its comment records a measured `dC_L/dalpha` relative error of `0.000118` at four ranks, which exceeds the general `0.0001` angle recommendation. The gate applies to that file's mesh, physics, points, and rows. |
| ONERA M6 shape gate | `cases/oneram6_euler_ffd/input/configs/fd_check_totals.toml` states the same shape floor and slope, an angle band of `0.0001`, and a `wing` cancellation-ratio table. Its first drag row has `C_r = 239.3`, giving a band of `0.4786`; this does not set a tolerance for another parameterization. |

The selected TOML can tighten or widen a recorded gate. A recommended value does not override it. The numerical limits above describe input admission and derivative comparisons, not validation against experimental data.

## Shipped case dossiers

The case directory and selected TOML own exact executable values. There are seven case directories and **33** runnable TOMLs at this checkout. The [three-case ADODG study](../validation-study.md) reports the benchmark comparisons and their validation limits.

| Case directory | Runnable TOMLs | Evidence boundary |
| --- | --- | --- |
| `adodg1_naca0012_euler_ffd` | `run_sp.toml`, `run_half_sp.toml`, `opt_ffd_cp.toml`, `opt_ffd_thickness.toml`, `opt_ffd_thickness_half.toml`, `opt_hh_1cos_n20.toml`, `opt_hh_even_n20.toml` | Euler NACA 0012 full and half models. The half model pins symmetry nodes. Grid sensitivity for the shipped meshes is not measured; another Euler derivative gate does not cover every shape row. |
| `adodg2_rae2822_rans_sa_ffd` | `run_sp.toml`, `opt_ffd_cp.toml`, `opt_hh_1cos_n20.toml`, `opt_hh_even_n20.toml` | RAE 2822 transonic RANS-SA. The selected remesh has no recorded admission gate or original source; no finite-difference TOML is shipped for this mesh. |
| `adodg3_naca0012_euler_ffd` | `run_alpha0_sp.toml`, `run_trim_le_exact.toml`, `fd_twist_n5.toml`, `opt_ffd_twist_n5.toml` | Euler wing on the exact-leading-edge mesh. The five-variable twist gate and optimization are recorded; local grid uncertainty is unverified. |
| `rae2822_rans_sa_hh` | `run_sp.toml`, `run_mp4.toml`, `run_mission.toml`, `opt_rans_sa_hh.toml`, `fd_check_totals.toml`, `fd_cst8.toml`, `fd_moment.toml`, `fd_moment_pivot_check.toml` | RAE 2822 RANS-SA includes baseline, multipoint, mission, optimization, and several derivative checks. Each gate's own TOML states its band and exclusions. |
| `oneram6_euler_ffd` | `run_sp.toml`, `opt_euler_ffd.toml`, `opt_opt.toml`, `fd_check_totals.toml`, `fd_moment.toml` | ONERA M6 Euler wing. Multiple wall markers and the symmetry plane are part of the derivative contract. An 18-row finite-difference record passed at 36 ranks. |
| `e387_rans_sa_bcm` | `run_su2_standard_sa_bcm.toml`, `run_e387_jst_sa_bcm.toml`, `fd_su2_standard_sa_bcm.toml` | E387 transition studies. SA-BCM primals and finite analytic totals were recorded, but no SA-BCM finite-difference gate establishes derivative accuracy. |
| `nlf0416_rans_sa_bcm` | `run_nlf0416_roe_sa_bcm_m025.toml`, `run_nlf0416_roe_sa_bcm.toml` | NLF(1)-0416 transition studies. Mach 0.25 primals were accepted; the shipped Mach 0.40 case has no accepted run in the record. Derivatives are assumed. |

The list totals 7 + 4 + 4 + 8 + 5 + 3 + 2 = 33 files. Paths are relative
to each case's `input/configs/`. All shipped runnable TOMLs are
self-contained. `case.toml`, mesh recipes, geometry files, meshes, and FFD
boxes are inputs rather than runnable TOMLs. A CLI override changes a run
without changing the source TOML; read the resolved configuration from the
run directory before interpreting it.


## Derivative status and accepted evidence

The registry in `src/mdao_su2/solver/su2_options.py` scopes a status by
physics profile, response, and boundary family. `fd_validated` means the
listed derivative rows passed a recorded finite-difference gate.
`primal_checked` covers only a primal. `assumed` is not a gate.
`unsupported` prevents an adjoint task.

The [human guide](../human-guide.pdf) summarizes the recorded gates and their limits.

| Recorded object | Result and limitation |
| --- | --- |
| Standalone RAE 2822 derivative | At Mach 0.74, the recorded finite-difference agreement was 0.00 to 0.07 percent. The result is specific to its selected rows and conditions. |
| RANS-SA efficiency derivative | The angle row had relative error 1.7e-5; 5 of 8 shape rows passed. Three leading-edge rows were excluded and are not validated by that record. |
| ONERA M6 multi-marker path | A 36-rank seam comparison was bit-identical, and 18 finite-difference rows were in band. |
| ADODG Case 3 twist | The exact-leading-edge mesh passed 12 of 12 cold-start twist and angle rows at 60 ranks. Twist band: 1e-3. Angle band: 1e-4. This does not validate normal shape rows on other Euler meshes. |
| E387 SA-BCM | A Mach 0.55 primal reached residual -11; analytic totals were finite after an adjoint at -8. A finite-difference derivative gate has not been established. |
| Langtry-Menter transition | SU2 v8.5 does not supply the transition solver slot required by its discrete-adjoint setup. The supported scope is primal-only. |

Mostly tangential 2D wall motion is outside the reliable shape-derivative
contract. A cold finite-difference sweep needs a stable adjacent-step
plateau and a band stated by the selected gate TOML. The generally
recommended shape band is `max(1e-3, 2e-3*C_r)`, capped at 1.0, with
`C_r` the measured cancellation ratio. The generally recommended angle
band is 1e-4.

Use the actual gate TOML when it states a different band. A converged
primal does not establish adjoint convergence or physical validation.

## Recorded baselines and physical limits

| Case and condition | Recorded value | Interpretation |
| --- | --- | --- |
| RAE 2822 at Mach 0.74, angle 2.70 degrees | Lift 0.791755; drag 0.020697 | Historical values; solver references in `tests/regression.py` remain unset until a user run is reviewed. |
| RAE 2822 four-point trim | Weighted drag 0.020637 | Historical value; inspect each point before adopting a new-run baseline. |
| ADODG Case 3 exact-leading-edge mesh | Zero-lift drag -14.22 counts; trim angle 4.263906770 degrees at lift 0.375 | Absolute drag has a mesh-and-scheme offset. Local grid uncertainty is unverified. |
| ADODG Case 3 five-variable twist optimum | Drag 65.487 to 64.542 counts; span efficiency 0.9360 to 0.9472 | The accepted result is the drag change of -0.945 count, not raw absolute drag. The near-field to Trefftz offset changed by 0.41 count. |
| NLF(1)-0416 at Mach 0.25 | At 0 and 4 degrees: lift 0.4748 and 0.9183; drag 0.020717 and 0.031434 | These primals were accepted. A Mach 0.75 trial was rejected after a residual plateau; it is not a shipped accepted condition. |

A 12-rank NACA 0012 Euler benchmark stopped at density residual -12.781
against a -13 target; that rank count is not validated for the benchmark.
The 2D Gmsh generator can return distinct interior meshes for identical
inputs. One E387 L3 repeat failed a 0.85 skewness gate with 0.908. Admit
and compare each mesh file, including a regenerated one. A successful
optimizer exit still requires converged primal and adjoint solves and
final unscaled constraints.

## Input provenance and publication

The bundled tutorial meshes, FFD boxes, and native SU2 comparison inputs
have unresolved per-input provenance or license evidence. The release
inventory outside the shipped repository tracks 53 such rows. The
`cases/adodg2_rae2822_rans_sa_ffd` intermediate Gmsh file is not
shipped; its selected SU2 mesh remains. Numerical checks do not establish
redistribution rights. Do not claim physical validation from a published
comparison without stated numerical, input, and experimental
uncertainties.

## Reading a new run

1. Resolve the selected TOML with `python run/optimize.py --config <toml>
   --print-config`, or read the resolved configuration in the run root.
2. Confirm the mesh file and its admission, marker roles, point conditions,
   objective, bounds, scalers, and output root.
3. Read the run status, residual history, task JSON, and call summaries.
   A failed line-search call can be recorded inside an otherwise completed
   driver run.
4. Judge only the derivative rows whose bands and exclusions are stated
   in the selected finite-difference gate TOML.
5. Check the final unscaled constraints and distinguish derivative
   verification from physical validation.
