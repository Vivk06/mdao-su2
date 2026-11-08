"""Matrix-free OpenMDAO Hicks--Henne surface parameterization."""

import os

os.environ.setdefault('OPENMDAO_REPORTS', '0')

import numpy as np
import openmdao.api as om
from pygeo import DVConstraints


__all__ = ['HicksHenneGeoComp']


def compute_bump_locations(n_dv, distribution):
    indices = np.arange(1, n_dv + 1, dtype=float)
    if distribution == 'even':
        return indices / (n_dv + 1.0)
    if distribution == '1-cos':
        return 0.5 * (1.0 - np.cos(np.pi * indices / (n_dv + 1.0)))
    raise ValueError(
        f'distribution={distribution!r} must be "even" or "1-cos"')


def build_basis_matrix(x, locations, bump_width):
    """Return one Hicks--Henne basis column per bump location."""
    x = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
    powers = np.log(0.5) / np.log(np.asarray(locations, dtype=float))
    basis = np.sin(np.pi * x[:, None] ** powers[None, :]) ** bump_width
    basis[(x == 0.0) | (x == 1.0), :] = 0.0
    return basis


def _upper_surface_mask(local_points, all_points, segments=None):
    """Classify the two branches of a connected airfoil surface."""
    x_all = all_points[:, 0]
    x_min = float(np.min(x_all))
    x_max = float(np.max(x_all))
    chord = x_max - x_min
    if chord <= 0.0:
        raise ValueError('Hicks-Henne surface has zero chord length')
    tol = max(1.0, abs(x_min), abs(x_max)) * 1e-12
    xhat = np.clip((local_points[:, 0] - x_min) / chord, 0.0, 1.0)
    if segments is not None and len(segments) > 0:
        endpoints = set(np.flatnonzero(
            (np.abs(x_all - x_min) <= tol)
            | (np.abs(x_all - x_max) <= tol)))
        adjacency = {i: set() for i in range(len(all_points))}
        for a, b in segments:
            adjacency[a].add(b)
            adjacency[b].add(a)
        components = []
        unseen = set(adjacency) - endpoints
        while unseen:
            stack = [unseen.pop()]
            component = set(stack)
            while stack:
                for neighbour in adjacency[stack.pop()] - endpoints:
                    if neighbour in unseen:
                        unseen.remove(neighbour)
                        component.add(neighbour)
                        stack.append(neighbour)
            components.append(component)
        if len(components) == 2:
            upper_component = max(
                components,
                key=lambda indices: np.mean(all_points[list(indices), 1]))
            upper = np.zeros(len(all_points), dtype=bool)
            upper[list(upper_component)] = True
            return xhat, upper[:len(local_points)]

    y_le = float(np.mean(all_points[np.abs(x_all - x_min) <= tol, 1]))
    y_te = float(np.mean(all_points[np.abs(x_all - x_max) <= tol, 1]))
    chord_y = y_le + xhat * (y_te - y_le)
    return xhat, local_points[:, 1] >= chord_y


def _curve_xy(points, mask, endpoints):
    selected = np.flatnonzero(mask | endpoints)
    order = np.argsort(points[selected, 0])
    x = points[selected[order], 0]
    y = points[selected[order], 1]
    unique_x, inverse = np.unique(x, return_inverse=True)
    mean_y = np.zeros_like(unique_x)
    counts = np.zeros_like(unique_x)
    np.add.at(mean_y, inverse, y)
    np.add.at(counts, inverse, 1.0)
    return unique_x, mean_y / counts


class _HicksHenneDVGeo:
    """The subset of the DVGeometry contract consumed by DVConstraints."""

    def __init__(self, surface, segments, locations, bump_width, upper_only):
        self._surface = np.asarray(surface, dtype=float)
        self._locations = locations
        self._bump_width = bump_width
        self._upper_only = upper_only
        self._n_dv = len(locations)
        self._n_total = self._n_dv if upper_only else 2 * self._n_dv
        self._values = np.zeros(self._n_total)
        self._point_sets = {}

        x = self._surface[:, 0]
        self._x_min = float(np.min(x))
        self._chord = float(np.max(x) - self._x_min)
        _, upper = _upper_surface_mask(
            self._surface, self._surface, segments)
        tol = max(1.0, abs(self._x_min), abs(np.max(x))) * 1e-12
        endpoints = ((np.abs(x - self._x_min) <= tol)
                     | (np.abs(x - np.max(x)) <= tol))
        self._upper_curve = _curve_xy(self._surface, upper, endpoints)
        self._lower_curve = _curve_xy(self._surface, ~upper, endpoints)

    def getVarNames(self):
        return ['shape']

    def getNDV(self):
        return self._n_total

    def setDesignVars(self, values):
        self._values = np.asarray(values['shape']).copy()

    def _coordinates(self, points):
        points = np.asarray(points)
        xhat = np.clip((points[:, 0] - self._x_min) / self._chord,
                       0.0, 1.0)
        upper_y = np.interp(points[:, 0], *self._upper_curve)
        lower_y = np.interp(points[:, 0], *self._lower_curve)
        upper = points[:, 1] >= 0.5 * (upper_y + lower_y)
        basis = build_basis_matrix(xhat, self._locations, self._bump_width)
        return basis, upper

    def addPointSet(self, points, name, compNames=None, **kwargs):
        reference = np.asarray(points).reshape(-1, 3).copy()
        basis, upper = self._coordinates(reference)
        self._point_sets[name] = (reference, basis, upper)

    def update(self, name, config=None):
        reference, basis, upper = self._point_sets[name]
        deformed = reference.copy()
        deformed[upper, 1] += basis[upper] @ self._values[:self._n_dv]
        if not self._upper_only:
            lower = ~upper
            deformed[lower, 1] -= basis[lower] @ self._values[self._n_dv:]
        return deformed

    def totalSensitivity(self, seed, name, comm=None, config=None):
        _, basis, upper = self._point_sets[name]
        seed = np.asarray(seed)
        single = seed.ndim == 2
        if single:
            seed = seed[None, ...]
        gradient = np.zeros((seed.shape[0], self._n_total))
        gradient[:, :self._n_dv] = seed[:, upper, 1] @ basis[upper]
        if not self._upper_only:
            lower = ~upper
            gradient[:, self._n_dv:] = -(seed[:, lower, 1] @ basis[lower])
        return {'shape': gradient[0] if single else gradient}


class HicksHenneGeoComp(om.ExplicitComponent):
    """Apply Hicks--Henne vertical bumps to an MPhys surface point set."""

    def initialize(self):
        self.options.declare('n_dv', types=int)
        self.options.declare('distribution', types=str)
        self.options.declare('bump_width', types=int)
        self.options.declare('upper_only', default=False, types=bool)
        self._point_sets = {}
        self._shape_name = None
        self._surface_segments = None
        self._constraint_names = []
        self._constraint_sensitivities = None

    def setup(self):
        self._dvcon = DVConstraints()

    def nom_set_surface_segments(self, segments):
        self._surface_segments = segments

    def nom_add_discipline_coords(self, discipline, points):
        input_name = discipline.COORDINATES_INPUT
        output_name = discipline.COORDINATES_OUTPUT
        reference = np.asarray(points, dtype=float).reshape(-1, 3).copy()
        gathered = (self.comm.allgather(reference)
                    if hasattr(self.comm, 'allgather') else [reference])
        all_points = np.concatenate(gathered, axis=0)
        xhat, upper = _upper_surface_mask(
            reference, all_points, self._surface_segments)
        locations = compute_bump_locations(
            self.options['n_dv'], self.options['distribution'])
        basis = build_basis_matrix(
            xhat, locations, self.options['bump_width'])
        self._point_sets[output_name] = {
            'input': input_name,
            'basis': basis,
            'upper': upper,
        }
        if not hasattr(self, '_dvgeo'):
            self._dvgeo = _HicksHenneDVGeo(
                all_points, self._surface_segments, locations,
                self.options['bump_width'], self.options['upper_only'])
            self._dvcon.setDVGeo(self._dvgeo)
        self._dvgeo.addPointSet(reference, output_name)
        flat = reference.reshape(-1)
        self.add_input(input_name, distributed=True, val=flat)
        self.add_output(output_name, distributed=True, val=flat)

    def nom_pin_points(self, output_name, rows, axis):
        """Hold ``rows`` of a point set on their reference ``axis`` coordinate.

        The bumps move y only, so a y pin zeroes those basis rows. The output and
        both derivative products all use the basis, so they follow. A pin along x
        or z holds nothing the bumps move. Wall/symmetry corner nodes at x̂ = 0 or 1
        already have zero rows; the pin makes that hold for any row.
        """
        if axis != 1:
            return
        rows = np.asarray(rows, dtype=int)
        self._point_sets[output_name]['basis'][rows] = 0.0
        self._dvgeo._point_sets[output_name][1][rows] = 0.0

    def add_shape_dv(self, name='shape'):
        if self._shape_name is not None:
            raise RuntimeError(
                'Hicks-Henne shape design variables have already been added')
        self._shape_name = name
        n_dv = self.options['n_dv']
        n_total = n_dv if self.options['upper_only'] else 2 * n_dv
        self.add_input(name, val=np.zeros(n_total), distributed=False)
        return n_total, (n_total,)

    def nom_setConstraintSurface(
            self, surface, name='default', addToDVGeo=False,
            DVGeoName='default', surfFormat='point-vector'):
        self._dvcon.setSurface(
            surface, name=name, addToDVGeo=addToDVGeo,
            DVGeoName=DVGeoName, surfFormat=surfFormat)

    def nom_addThicknessConstraints2D(
            self, name, leList, teList, nSpan, nChord, scaled=True,
            addToPyOpt=True, surfaceName='default', DVGeoName='default',
            compNames=None, projected=False):
        self._dvcon.addThicknessConstraints2D(
            leList, teList, nSpan, nChord, name=name, scaled=scaled,
            addToPyOpt=addToPyOpt, surfaceName=surfaceName,
            DVGeoName=DVGeoName, compNames=compNames, projected=projected)
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
        self._dvgeo.setDesignVars({'shape': shape})
        n_dv = self.options['n_dv']
        for output_name, point_set in self._point_sets.items():
            coordinates = np.asarray(inputs[point_set['input']]).reshape(-1, 3)
            deformed = coordinates.copy()
            basis = point_set['basis']
            upper = point_set['upper']
            deformed[upper, 1] += basis[upper] @ shape[:n_dv]
            if not self.options['upper_only']:
                lower = ~upper
                deformed[lower, 1] -= basis[lower] @ shape[n_dv:]
            outputs[output_name] = deformed.reshape(-1)

        functions = {}
        self._dvcon.evalFunctions(functions, includeLinear=True)
        if functions.pop('fail', False):
            raise om.AnalysisError(
                'Analysis error in Hicks-Henne geometric constraints')
        for name, value in functions.items():
            outputs[name] = value
        self._constraint_sensitivities = None

    def _constraint_jacobian(self, name):
        """d(constraint)/d(shape), shape (m, n_shape); evaluated once per compute."""
        if self._constraint_sensitivities is None:
            self._constraint_sensitivities = {}
            self._dvcon.evalFunctionsSens(
                self._constraint_sensitivities, includeLinear=True)
        return np.atleast_2d(self._constraint_sensitivities[name]['shape'])

    def _jacvec_fwd(self, d_inputs, d_outputs):
        """Forward product: the transpose of the reverse branch."""
        n_dv = self.options['n_dv']
        shape_seed = (np.asarray(d_inputs[self._shape_name])
                      if self._shape_name in d_inputs else None)
        if shape_seed is not None:
            for name in self._constraint_names:
                if name in d_outputs:
                    d_outputs[name] += self._constraint_jacobian(name) @ shape_seed

        for output_name, point_set in self._point_sets.items():
            if output_name not in d_outputs:
                continue
            input_name = point_set['input']
            if input_name in d_inputs:
                d_outputs[output_name] += d_inputs[input_name]
            if shape_seed is None:
                continue
            # Shape is replicated, so each rank fills its own local surface
            # rows and no collective is needed (unlike the reverse allreduce).
            basis = point_set['basis']
            upper = point_set['upper']
            delta = np.zeros((len(upper), 3))
            delta[upper, 1] = basis[upper] @ shape_seed[:n_dv]
            if not self.options['upper_only']:
                lower = ~upper
                delta[lower, 1] = -(basis[lower] @ shape_seed[n_dv:])
            d_outputs[output_name] += delta.reshape(-1)

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        if mode == 'fwd':
            self._jacvec_fwd(d_inputs, d_outputs)
            return

        n_dv = self.options['n_dv']
        if self._shape_name in d_inputs:
            for name in self._constraint_names:
                if name in d_outputs:
                    seed = np.asarray(d_outputs[name]).reshape(-1)
                    d_inputs[self._shape_name] += (
                        seed @ self._constraint_jacobian(name))

        for output_name, point_set in self._point_sets.items():
            gradient = np.zeros(
                n_dv if self.options['upper_only'] else 2 * n_dv)
            if output_name in d_outputs:
                seed = np.asarray(d_outputs[output_name]).reshape(-1, 3)
                input_name = point_set['input']
                if input_name in d_inputs:
                    d_inputs[input_name] += seed.reshape(-1)
                if self._shape_name in d_inputs:
                    basis = point_set['basis']
                    upper = point_set['upper']
                    gradient[:n_dv] = basis[upper].T @ seed[upper, 1]
                    if not self.options['upper_only']:
                        lower = ~upper
                        gradient[n_dv:] = -(
                            basis[lower].T @ seed[lower, 1])
            if self._shape_name in d_inputs:
                # Shape is replicated while surface coordinates are
                # distributed, so every rank must contribute its local seed.
                gradient = self.comm.allreduce(gradient)
                d_inputs[self._shape_name] += gradient
