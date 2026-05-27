"""Interest-rate model hierarchy — abstract bases + Parameter + calibration glue.

# C++ parity: ql/models/* (v1.42.1).

L4-A pilot scope (this phase):

- ``parameter.py`` — Parameter hierarchy (Null/Constant/PiecewiseConstant/
  TermStructureFittedParameter), plus the nested ``Parameter.Impl``
  strategy.
- ``model.py`` — abstract bases (``Model``, ``TermStructureConsistentModel``,
  ``CalibratedModel`` with the calibrate() orchestration).
- ``calibration_helper.py`` — abstract ``CalibrationHelper`` +
  ``BlackCalibrationHelper`` bases.
- ``protocols.py`` — runtime-checkable Protocols for cross-cluster typing
  (``ModelProtocol``, ``CalibrationHelperProtocol``,
  ``ShortRateModelProtocol``).

Concrete short-rate / market / equity / volatility models land in L4-B/C/D/E.
"""
