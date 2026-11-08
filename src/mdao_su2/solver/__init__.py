"""Solver modules:
aero_comp.py: SU2 aerodynamic component.
bidir_comp.py: bidirectional solver adapter.
builder.py: SU2 MPhys builder.
cfg_file.py: SU2 cfg parsing and writing.
point_cfg_writer.py: primal and adjoint cfg generation.
response_catalog.py: response support status.
seam_comp.py: multi-marker solver and adjoint stall check.
su2_options.py: SU2 option registry.
su2_options_v85.py: SU2 v8.5 option data.
unsteady_primal.py: unsteady primal runs.
variable_names.py: MPhys variable names.
standalone/: standalone solver commands.
"""

from .aero_comp import SU2AeroComp, get_reference_surface
