"""Experimental finite-differences engines and step conditions.

# C++ parity: ql/experimental/finitedifferences/ (v1.42.1).

Phase 11 W5-B cluster:

* :class:`SwingExercise` — Bermudan exercise variant with per-date
  intraday-second offsets, used by VPP and swing options.
* :class:`VanillaVPPOption` — vanilla virtual-power-plant option.
* :class:`FdmVPPStepCondition` / :class:`FdmVPPStartLimitStepCondition`
  / :class:`FdmVPPStepConditionFactory` — backward-induction step
  conditions used by the dynamic-programming intrinsic engine.
* :class:`DynProgVPPIntrinsicValueEngine` — closed-form NPV via
  dynamic programming against fuel + power price arrays.
* :class:`FdSimpleExtOUStorageEngine` /
  :class:`FdSimpleKlugeExtOUVPPEngine` /
  :class:`FdSimpleExtOUJumpSwingEngine` — FD engines under the
  Kluge + Extended-Ornstein-Uhlenbeck family. Structural scaffolds
  only at W5-B; full pricing depends on the ExtOU/Kluge FD operator
  cluster (W5-A) and is deferred until that lands.
"""
