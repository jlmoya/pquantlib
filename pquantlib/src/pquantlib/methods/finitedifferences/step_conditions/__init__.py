"""Step conditions applied at each time step (early exercise, etc.).

# C++ parity: ql/methods/finitedifferences/stepconditions/ (v1.43).

One module per C++ header:

* ``step_condition`` — the ``StepCondition<Array>`` ABC.
* ``fdm_american_step_condition`` — American early exercise (every step).
* ``fdm_bermudan_step_condition`` — early exercise on a date schedule.
* ``fdm_arithmetic_average_condition`` — Asian fixing-date average update.
* ``fdm_simple_storage_condition`` — gas-storage inject/withdraw decision.
* ``fdm_simple_swing_condition`` — swing-option exercise decision.
* ``fdm_snapshot_condition`` — capture the solution vector at one instant.
* ``fdm_step_condition_composite`` — sequence of conditions + stopping times.

``ZeroCondition`` lives one level up, at
``pquantlib.methods.finitedifferences.zero_condition``, mirroring the C++
header ``ql/methods/finitedifferences/zerocondition.hpp``.

``inner_value_calculator_protocol`` is not a port: it is the structural
contract the conditions above need from ``FdmInnerValueCalculator``.
"""
