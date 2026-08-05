"""Optimization: constraints, cost functions, end criteria, methods, problems.

# C++ parity: ql/math/optimization/* (v1.43).

Scaffolding
-----------
- ``Constraint`` base + ``NoConstraint``, ``PositiveConstraint``,
  ``BoundaryConstraint``, ``NonhomogeneousBoundaryConstraint``,
  ``CompositeConstraint``, plus ``Constraint.update``
- ``CostFunction`` abstract + ``SimpleCostFunction``,
  ``ParametersTransformation``
- ``EndCriteria`` + ``EndCriteria.Type`` IntEnum, with every checker
  including the ones that carry the ``statStateIterations`` in-out counter
- ``OptimizationMethod`` abstract
- ``Problem`` (cost + constraint + state bundle)
- ``Projection``, ``ProjectedConstraint``, ``ProjectedCostFunction``
- ``LeastSquareProblem``, ``LeastSquareFunction``, ``NonLinearLeastSquare``

Methods
-------
- ``LBFGSB`` — limited-memory bound-constrained quasi-Newton
- ``Simplex`` — Nelder-Mead, transcribed from simplex.cpp
- ``LineSearch`` + ``ArmijoLineSearch``, ``GoldsteinLineSearch``
- ``LineSearchBasedMethod`` + ``SteepestDescent``, ``ConjugateGradient``,
  ``BFGS``
- ``SimulatedAnnealing`` — annealed simplex, seeded from QuantLib's own
  Mersenne twister
- ``DifferentialEvolution`` (+ ``Candidate``, ``Configuration``) — 7
  strategies x 3 crossover types, likewise seeded
- ``SphereCylinderOptimizer``
- ``LevenbergMarquardt`` — **still a scipy delegation, not a port.** See
  ``tests/math/optimization/test_levenberg_marquardt_cpp_parity.py`` for
  the measured divergences against C++ v1.43; porting MINPACK's ``lmdif``
  (ql/math/optimization/lmdif.cpp) is the outstanding work.
"""
