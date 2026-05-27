"""Optimization scaffolding (constraint, cost function, end criteria, method, problem).

# C++ parity: ql/math/optimization/* (v1.42.1).

L1-D scaffolding (closed in Phase 1):

- ``Constraint`` base + ``NoConstraint``, ``PositiveConstraint``,
  ``BoundaryConstraint``
- ``CostFunction`` abstract
- ``EndCriteria`` + ``EndCriteria.Type`` IntEnum
- ``OptimizationMethod`` abstract
- ``Problem`` (cost + constraint + state bundle)

L4-A concretizations (this batch, closing Phase 1 carry-overs):

- ``LevenbergMarquardt`` (scipy-backed ``least_squares(method='lm')``)
- ``Simplex`` (scipy-backed ``minimize(method='Nelder-Mead')``)

Carve-outs (still deferred to follow-up clusters): Bfgs,
ConjugateGradient, SimulatedAnnealing, DifferentialEvolution,
LineSearch + subclasses, CompositeConstraint,
NonhomogeneousBoundaryConstraint, ParametersTransformation,
SimpleCostFunction.
"""
