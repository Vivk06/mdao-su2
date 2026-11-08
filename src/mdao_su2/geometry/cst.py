"""OpenMDAO wrapper for pyGeo's direct-surface CST parameterization."""

import os

os.environ.setdefault('OPENMDAO_REPORTS', '0')

import numpy as np
import openmdao.api as om
from pygeo import DVConstraints, DVGeometryCST


__all__ = ['CSTGeoComp']


class CSTGeoComp(om.ExplicitComponent):
    """Expose a ``DVGeometryCST`` through the MPhys geometry interface.

    The public ``shape`` input is a zero-based perturbation from the fitted CST
    coefficients. This preserves the existing model contract in which a zero
    shape vector is the baseline geometry.
    """

    def initialize(self):
        self.options.declare('dat_file', types=str)
        self.options.declare('n_cst', types=(int, list))
        self.options.declare('idx_chord', default=0, types=int)
        self.options.declare('idx_vertical', default=1, types=int)
        self._point_names = []
        self._reference_points = {}
        self._shape_name = None
        self._constraint_names = []
        self._constraint_sensitivities = None

    def setup(self):
        self._dvgeo = DVGeometryCST(
            self.options['dat_file'],
            numCST=self.options['n_cst'],
            idxChord=self.options['idx_chord'],
            idxVertical=self.options['idx_vertical'],
            comm=self.comm,
        )
        self._dvcon = DVConstraints()
        self._dvcon.setDVGeo(self._dvgeo)

    def nom_add_discipline_coords(self, discipline, points):
        """Register one distributed MPhys coordinate input/output point set."""
        input_name = discipline.COORDINATES_INPUT
        output_name = discipline.COORDINATES_OUTPUT
        reference = np.asarray(points).reshape(-1, 3).copy()
        self._reference_points[output_name] = reference.copy()

        self._dvgeo.addPointSet(reference.copy(), output_name)
        if not hasattr(self._dvgeo, 'ptSetNames'):
            self._dvgeo.ptSetNames = []
        if output_name not in self._dvgeo.ptSetNames:
            self._dvgeo.ptSetNames.append(output_name)
        if output_name not in self._point_names:
            self._point_names.append(output_name)

        flat = reference.reshape(-1)
        self.add_input(input_name, distributed=True, val=flat)
        self.add_output(output_name, distributed=True, val=flat)

    def add_shape_dv(self, name='shape'):
        """Add upper/lower CST coefficients as one concatenated perturbation."""
        if self._shape_name is not None:
            raise RuntimeError('CST shape design variables have already been added')

        n_upper = self._dvgeo.addDV(dvName='upper', dvType='upper')
        n_lower = self._dvgeo.addDV(dvName='lower', dvType='lower')
        self._baseline_upper = self._dvgeo.DVs['upper'].value.copy()
        self._baseline_lower = self._dvgeo.DVs['lower'].value.copy()
        self._n_upper = n_upper
        self._n_lower = n_lower
        self._shape_name = name
        self.add_input(name, val=np.zeros(n_upper + n_lower), distributed=False)
        n_total = n_upper + n_lower
        return n_total, (n_total,)

    def nom_setConstraintSurface(
            self, surface, name='default', addToDVGeo=False,
            DVGeoName='default', surfFormat='point-vector'):
        """Set the triangulated surface used by ``DVConstraints``."""
        self._dvcon.setSurface(
            surface, name=name, addToDVGeo=addToDVGeo,
            DVGeoName=DVGeoName, surfFormat=surfFormat)

    def _assert_thickness_straddles(self, name):
        """Reject thickness probes whose endpoints land on the same CST surface."""
        point_set = self._dvgeo.points[name]
        points = point_set['points']
        upper = set(point_set['upper'])
        lower = set(point_set['lower'])
        for index in range(0, len(points), 2):
            labels = {
                (row in upper, row in lower)
                for row in (index, index + 1)
            }
            if labels != {(True, False), (False, True)}:
                x = float(np.real(points[index, self.options['idx_chord']]))
                raise ValueError(
                    f'{name}: thickness probe pair at x={x:.6g} does not '
                    'straddle the CST upper and lower surfaces')

    def nom_addThicknessConstraints2D(
            self, name, leList, teList, nSpan, nChord, scaled=True,
            addToPyOpt=True, surfaceName='default', DVGeoName='default',
            compNames=None, projected=False):
        self._dvcon.addThicknessConstraints2D(
            leList, teList, nSpan, nChord, name=name, scaled=scaled,
            addToPyOpt=addToPyOpt, surfaceName=surfaceName,
            DVGeoName=DVGeoName, compNames=compNames, projected=projected)
        self._assert_thickness_straddles(name)
        self._constraint_names.append(name)
        self.add_output(name, distributed=False,
                        val=np.ones(nSpan * nChord))

    def nom_addVolumeConstraint(
            self, name, leList, teList, nSpan=10, nChord=10, scaled=True,
            addToPyOpt=True, surfaceName='default', DVGeoName='default',
            compNames=None):
        self._dvcon.addVolumeConstraint(
            leList, teList, nSpan=nSpan, nChord=nChord, name=name,
            scaled=scaled, addToPyOpt=addToPyOpt, surfaceName=surfaceName,
            DVGeoName=DVGeoName, compNames=compNames)
        if scaled:
            constraint = self._dvcon.constraints['volCon'][name]
            constraint.coords = self._dvgeo.update(name)
            constraint.V0 = constraint.evalVolume()
        self._constraint_names.append(name)
        self.add_output(name, distributed=False, val=1.0)

    def nom_addLERadiusConstraints(
            self, name, leList, nSpan, axis, chordDir, scaled=True,
            addToPyOpt=True, surfaceName='default', DVGeoName='default',
            compNames=None):
        self._dvcon.addLERadiusConstraints(
            leList=leList, nSpan=nSpan, axis=axis, chordDir=chordDir,
            name=name, scaled=scaled, addToPyOpt=addToPyOpt,
            surfaceName=surfaceName, DVGeoName=DVGeoName,
            compNames=compNames)
        self._constraint_names.append(name)
        self.add_output(name, distributed=False, val=np.ones(nSpan))

    def nom_addCurvatureConstraint1D(
            self, name, start, end, nPts, axis, curvatureType='mean',
            scaled=True, KSCoeff=1.0, addToPyOpt=True,
            surfaceName='default', DVGeoName='default', compNames=None):
        self._dvcon.addCurvatureConstraint1D(
            start=start, end=end, nPts=nPts, axis=axis, name=name,
            curvatureType=curvatureType, scaled=scaled, KSCoeff=KSCoeff,
            addToPyOpt=addToPyOpt, surfaceName=surfaceName,
            DVGeoName=DVGeoName, compNames=compNames)
        self._constraint_names.append(name)
        self.add_output(name, distributed=False, val=1.0)

    def compute(self, inputs, outputs):
        shape = np.asarray(inputs[self._shape_name])
        split = self._n_upper
        self._dvgeo.setDesignVars({
            'upper': self._baseline_upper + shape[:split],
            'lower': self._baseline_lower + shape[split:],
        })
        for point_name in self._point_names:
            outputs[point_name] = self._dvgeo.update(point_name).reshape(-1)

        functions = {}
        self._dvcon.evalFunctions(functions, includeLinear=True)
        if functions.pop('fail', False):
            raise om.AnalysisError('Analysis error in CST geometric constraints')
        for name, value in functions.items():
            outputs[name] = value
        self._constraint_sensitivities = None

    def _constraint_jacobian(self, name):
        """d(constraint)/d(shape), shape (m, n_upper + n_lower); once per compute."""
        if self._constraint_sensitivities is None:
            self._constraint_sensitivities = {}
            self._dvcon.evalFunctionsSens(
                self._constraint_sensitivities, includeLinear=True)
        sensitivity = self._constraint_sensitivities[name]
        return np.concatenate([
            np.atleast_2d(sensitivity['upper']),
            np.atleast_2d(sensitivity['lower']),
        ], axis=-1)

    def _jacvec_fwd(self, d_inputs, d_outputs):
        """Forward product: the transpose of the reverse branch."""
        shape_seed = np.asarray(d_inputs[self._shape_name])
        for name in self._constraint_names:
            if name in d_outputs:
                d_outputs[name] += self._constraint_jacobian(name) @ shape_seed

        split = self._n_upper
        seeds = {'upper': shape_seed[:split], 'lower': shape_seed[split:]}
        for point_name in self._point_names:
            if point_name in d_outputs:
                # Local rows only; shape is replicated, so no collective.
                d_outputs[point_name] += np.asarray(
                    self._dvgeo.totalSensitivityProd(seeds, point_name)).reshape(-1)

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        if self._shape_name not in d_inputs:
            return
        if mode == 'fwd':
            self._jacvec_fwd(d_inputs, d_outputs)
            return

        for name in self._constraint_names:
            if name in d_outputs:
                seed = np.asarray(d_outputs[name]).reshape(-1)
                d_inputs[self._shape_name] += seed @ self._constraint_jacobian(name)

        for point_name in self._point_names:
            if point_name not in d_outputs:
                continue
            seed = np.asarray(d_outputs[point_name]).reshape(-1, 3)
            sensitivity = self._dvgeo.totalSensitivity(
                seed, point_name, comm=self.comm)
            gradient = np.concatenate([
                np.asarray(sensitivity['upper']).reshape(-1),
                np.asarray(sensitivity['lower']).reshape(-1),
            ])
            d_inputs[self._shape_name] += gradient
