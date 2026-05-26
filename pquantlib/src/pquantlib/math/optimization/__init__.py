"""Optimization scaffolding (constraint, cost function, end criteria, method, problem).

# C++ parity: ql/math/optimization/* (v1.42.1).

L1-D cluster scope (this batch):

- ``Constraint`` base + ``NoConstraint``, ``PositiveConstraint``,
  ``BoundaryConstraint``
- ``CostFunction`` abstract
- ``EndCriteria`` + ``EndCriteria.Type`` IntEnum
- ``OptimizationMethod`` abstract
- ``Problem`` (cost + constraint + state bundle)

Carve-outs (deferred to follow-up clusters): LevenbergMarquardt, Bfgs,
ConjugateGradient, Simplex, SimulatedAnnealing, DifferentialEvolution,
LineSearch + subclasses, CompositeConstraint, NonhomogeneousBoundaryConstraint,
ParametersTransformation, SimpleCostFunction.
"""
