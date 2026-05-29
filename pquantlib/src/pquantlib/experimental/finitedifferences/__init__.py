"""Experimental finite-difference operators, solvers, and inner-value calculators.

# C++ parity: ql/experimental/finitedifferences/* (v1.42.1).

W5-A houses the ExtOU + Kluge FD infrastructure used to price energy
swing / VPP / storage / spread options:

* ``Glued1dMesher`` — composite 1-D mesher (left + right ⨁ optional
  common point).
* ``FdmExtendedOrnsteinUhlenbeckOp`` — 1-D FD operator for the ExtOU
  SDE.
* ``FdmExtOUJumpOp`` — 2-D FD operator for the Kluge ExtOU + jump
  process.
* ``FdmKlugeExtOUOp`` — 3-D FD operator for the correlated Kluge +
  ExtOU process.
* ``FdmExtOUJumpSolver`` / ``FdmKlugeExtOUSolver`` /
  ``FdmSimple2dExtOUSolver`` / ``FdmSimple3dExtOUJumpSolver`` — thin
  ``valueAt(...)`` wrappers over the C++ ``Fdm{N}DimSolver``
  templates.
* ``FdmExpExtOUInnerValueCalculator`` — payoff on ``exp(x)`` at a
  grid node.
* ``FdmSpreadPayoffInnerValue`` — 2-asset spread payoff calculator.
"""
