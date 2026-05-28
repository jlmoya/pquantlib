"""FdmStepConditionComposite — list of step conditions applied in sequence.

# C++ parity: ql/methods/finitedifferences/stepconditions/fdmstepconditioncomposite.{hpp,cpp}
# (v1.42.1).

Carries a list of step conditions and a list of *stopping times* (the
times at which any of the conditions wants to be invoked). The
backward solver consults ``stopping_times()`` to refine the step
schedule, then calls ``apply_to`` at each step.

The L5-D scope uses the trivial composite (single American or empty)
exclusively. ``vanilla_composite`` is a convenience builder that the
C++ engine uses — Python keeps it for parity but populates only the
American branch (dividends + Bermudan deferred).
"""

from __future__ import annotations

from typing import final

from pquantlib.math.array import Array
from pquantlib.methods.finitedifferences.step_conditions.step_condition import (
    StepCondition,
)


@final
class FdmStepConditionComposite(StepCondition):
    """Composite of step conditions applied in sequence.

    # C++ parity: ``class FdmStepConditionComposite : public StepCondition<Array>``.
    """

    def __init__(
        self,
        stopping_times: list[list[float]],
        conditions: list[StepCondition],
    ) -> None:
        self._conditions: list[StepCondition] = list(conditions)
        # Flatten + dedupe + sort stopping times.
        all_times: set[float] = set()
        for ts in stopping_times:
            all_times.update(ts)
        self._stopping_times: list[float] = sorted(all_times)

    def conditions(self) -> list[StepCondition]:
        return list(self._conditions)

    def stopping_times(self) -> list[float]:
        return list(self._stopping_times)

    def apply_to(self, a: Array, t: float) -> None:
        for c in self._conditions:
            c.apply_to(a, t)


__all__ = ["FdmStepConditionComposite"]
