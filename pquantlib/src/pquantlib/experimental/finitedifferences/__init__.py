"""Experimental finite-difference operators, solvers, engines, and step conditions.

# C++ parity: ql/experimental/finitedifferences/* (v1.42.1).

Phase 11 W5 ports the experimental finite-difference tree across three
cluster contributions:

**W5-A — ExtOU + Kluge FD infrastructure** (energy swing / VPP / storage /
spread option machinery):

* ``Glued1dMesher`` — composite 1-D mesher (left + right ⨁ optional
  common point).
* ``FdmExtendedOrnsteinUhlenbeckOp`` — 1-D FD operator for the ExtOU SDE.
* ``FdmExtOUJumpOp`` — 2-D FD operator for the Kluge ExtOU + jump process.
* ``FdmKlugeExtOUOp`` — 3-D FD operator for the correlated Kluge + ExtOU
  process.
* ``FdmExtOUJumpSolver`` / ``FdmKlugeExtOUSolver`` /
  ``FdmSimple2dExtOUSolver`` / ``FdmSimple3dExtOUJumpSolver`` — thin
  ``valueAt(...)`` wrappers over the C++ ``Fdm{N}DimSolver`` templates.
* ``FdmExpExtOUInnerValueCalculator`` — payoff on ``exp(x)`` at a grid node.
* ``FdmSpreadPayoffInnerValue`` — 2-asset spread payoff calculator.

**W5-B — VPP / swing / storage instruments + step conditions:**

* :class:`SwingExercise` — Bermudan exercise variant with per-date
  intraday-second offsets, used by VPP and swing options.
* :class:`VanillaVPPOption` — vanilla virtual-power-plant option.
* :class:`FdmVPPStepCondition` / :class:`FdmVPPStartLimitStepCondition`
  / :class:`FdmVPPStepConditionFactory` — backward-induction step
  conditions used by the dynamic-programming intrinsic engine.
* :class:`DynProgVPPIntrinsicValueEngine` — closed-form NPV via dynamic
  programming against fuel + power price arrays.
* :class:`FdSimpleExtOUStorageEngine` / :class:`FdSimpleKlugeExtOUVPPEngine`
  / :class:`FdSimpleExtOUJumpSwingEngine` — FD engines under the Kluge +
  Extended-Ornstein-Uhlenbeck family. Structural scaffolds; full pricing
  depends on the multi-D backward FDM framework (deferred).

**W5-C — ZABR / Dupire / OU FD operators + engine:**

* ``FdmDupire1dOp`` — 1-D Dupire local-volatility operator.
* ``FdmZabrOp`` — 2-D ZABR pricing operator (closes the W2-A ZABR-FD
  carve-out at the operator level).
* ``FdmExtOUJumpModelInnerValue`` — inner-value calculator for ExtOU+Jump
  models.
* ``FdOrnsteinUhlenbeckVanillaEngine`` — FD engine for vanilla options
  under OU dynamics (with its ``FdmOrnsteinUhlenbeckOp`` +
  ``FdmSimpleProcess1dMesher`` in ``ql/methods/finitedifferences``).
"""
