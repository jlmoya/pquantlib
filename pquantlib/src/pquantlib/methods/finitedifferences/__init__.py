"""Finite-difference framework.

# C++ parity: ql/methods/finitedifferences/ (v1.42.1).

L5-D ports the **minimal 1-D Black-Scholes** subset:

- ``meshers/`` — index layout + 1-D / multi-D grids (log-spot mesh
  anchored at the strike).
- ``operators/`` — sparse banded first/second derivative operators
  and the BSM operator ``L = -0.5 sigma^2 S^2 d^2/dS^2 - (r-q) S d/dS
  + r`` (built in log-spot coordinates).
- ``schemes/`` — ImplicitEuler / ExplicitEuler / CrankNicolson +
  the scheme descriptor enum.
- ``step_conditions/`` — composite + American early-exercise.
- ``solvers/`` — solver-desc DTO + ``FdmBackwardSolver`` that
  back-propagates from maturity to t=0.

Multi-asset FD (Heston/G2/Bates/...), additional operator
splittings, full BC support, and Concentrating/Exponential meshers
are deferred to Phase 6.
"""
