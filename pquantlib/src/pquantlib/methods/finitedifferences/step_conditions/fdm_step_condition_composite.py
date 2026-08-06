"""FdmStepConditionComposite — list of step conditions applied in sequence.

# C++ parity: ql/methods/finitedifferences/stepconditions/fdmstepconditioncomposite.{hpp,cpp}
# (v1.42.1).

Carries a list of step conditions and a list of *stopping times* (the
times at which any of the conditions wants to be invoked). The
backward solver consults ``stopping_times()`` to refine the step
schedule, then calls ``apply_to`` at each step.

C++ declares two static builders on this class. ``join_conditions``
(C++ ``joinConditions``) is ported below. The other, ``vanillaComposite``,
is **not** ported yet: it needs ``Exercise``-type dispatch, a
``DividendSchedule`` filter feeding ``FdmDividendHandler``, and an
``FdmAmericanStepCondition`` whose second argument is an
``FdmInnerValueCalculator`` rather than a bare ``Payoff`` — the Python
``FdmAmericanStepCondition`` still takes the payoff.
"""

from __future__ import annotations

from typing import final

from pquantlib.math.array import Array
from pquantlib.methods.finitedifferences.step_conditions.fdm_snapshot_condition import (
    FdmSnapshotCondition,
)
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

    @staticmethod
    def join_conditions(
        c1: FdmSnapshotCondition,
        c2: FdmStepConditionComposite,
    ) -> FdmStepConditionComposite:
        """Glue a snapshot condition onto an existing composite.

        # C++ parity: ``FdmStepConditionComposite::joinConditions``.

        The snapshot's own time joins the stopping times, so the solver is
        forced to land exactly on it — which is what makes
        ``FdmSnapshotCondition``'s exact ``t == t_`` test fire. Order matters:
        ``c2`` runs first, then the snapshot, so the recorded values are the
        *post-condition* ones.
        """
        return FdmStepConditionComposite(
            [c2.stopping_times(), [c1.get_time()]],
            [c2, c1],
        )


__all__ = ["FdmStepConditionComposite"]
