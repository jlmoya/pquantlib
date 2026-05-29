"""Experimental finite-difference operators + engines.

# C++ parity: ``ql/experimental/finitedifferences/`` (v1.42.1).

Phase 11 W5-C scope:
* ``FdmDupire1dOp`` — 1-D Dupire local-volatility operator.
* ``FdmZabrOp`` — 2-D ZABR pricing operator.
* ``FdmOrnsteinUhlenbeckOp`` — 1-D OU operator (lives in core C++
  ``ql/methods/finitedifferences/operators`` but ported here since
  it's only used by the W5-C experimental engine).
* ``FdOrnsteinUhlenbeckVanillaEngine`` — FD engine for vanilla
  options under OU dynamics.
* ``FdmSimpleProcess1dMesher`` — 1-D mesher driven by process
  expectations + variance.
* ``FdmExtOUJumpModelInnerValue`` — inner-value calculator for
  ExtOU+Jump models (used by deferred FdExtOUJumpVanillaEngine +
  FdKlugeExtOUSpreadEngine — see W5-C carve-outs).
"""
