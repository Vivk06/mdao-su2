"""Canonical names for the MPhys surface variables used across the pipeline."""

from mphys import MPhysVariables

SURF = MPhysVariables.Aerodynamics.Surface

# Adapter output feeding SU2AeroComp's (2D, non-distributed) x_aero -- an internal
# connection name, not an MPhys-standard one.  Mirrors builder._X_AERO_2D.
X_AERO_2D = 'x_aero_2d'

__all__ = ['SURF', 'X_AERO_2D']
