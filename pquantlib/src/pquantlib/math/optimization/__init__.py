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

v1.43 additions:

- ``LBFGSB`` — the limited-memory bound-constrained quasi-Newton method,
  ported line for line from ``ql/math/optimization/lbfgsb.{hpp,cpp}``, plus
  the prerequisites it reaches for: ``NonhomogeneousBoundaryConstraint``,
  ``CostFunction.value_and_gradient`` and the ``EndCriteria`` max-iterations
  / zero-gradient-norm checks.

Carve-outs (still deferred to follow-up clusters): Bfgs,
ConjugateGradient, SimulatedAnnealing, DifferentialEvolution,
LineSearch + subclasses, CompositeConstraint,
ParametersTransformation, SimpleCostFunction, and the ``EndCriteria``
checkers that carry the ``statStateIterations`` in-out counter
(``checkStationaryPoint``, ``checkStationaryFunctionValue``,
``checkStationaryFunctionAccuracy``, ``operator()``).
"""
