"""Cross-validation tests for ``FdmStepConditionComposite.join_conditions``.

# C++ parity: ql/methods/finitedifferences/stepconditions/fdmstepconditioncomposite.{hpp,cpp}
# @ v1.43 — the ``joinConditions`` static.

``joinConditions`` is the library's only consumer of ``FdmSnapshotCondition``,
so it is pinned in this package alongside it. Reference:
``migration-harness/references/v143/methods/stepconditions.json``, ``join*``
keys.

The second case is the one that carries information: the snapshot time is set
to an exercise time, so the recorded values can only match C++ if the wrapped
composite runs *before* the snapshot.
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.math.array import Array
from pquantlib.methods.finitedifferences.step_conditions.fdm_bermudan_step_condition import (
    FdmBermudanStepCondition,
)
from pquantlib.methods.finitedifferences.step_conditions.fdm_snapshot_condition import (
    FdmSnapshotCondition,
)
from pquantlib.methods.finitedifferences.step_conditions.fdm_step_condition_composite import (
    FdmStepConditionComposite,
)
from pquantlib.testing.tolerance import tight
from pquantlib.time.date import Date, Month
from tests.methods.finitedifferences.step_conditions import _fixtures

_REF_DATE = Date.from_ymd(15, Month.June, 2025)
_EXERCISE_DATES = [
    Date.from_ymd(15, Month.December, 2025),
    Date.from_ymd(15, Month.June, 2026),
    Date.from_ymd(15, Month.December, 2026),
]


@pytest.fixture(scope="module")
def reference() -> dict[str, Any]:
    return _fixtures.load()


def _bermudan_composite() -> tuple[FdmStepConditionComposite, FdmBermudanStepCondition]:
    mesher = _fixtures.bermudan_mesher()
    bermudan = FdmBermudanStepCondition(
        _EXERCISE_DATES,
        _REF_DATE,
        Actual365Fixed(),
        mesher,
        _fixtures.put_100(mesher),
    )
    composite = FdmStepConditionComposite([bermudan.exercise_times()], [bermudan])
    return composite, bermudan


def _assert_times(actual: list[float], reference: dict[str, Any], key: str) -> None:
    expected: list[float] = reference[key]
    assert len(actual) == len(expected)
    for i, exp in enumerate(expected):
        tight(actual[i], float(exp), reason=f"{key}[{i}]")


def test_wrapped_composite_stopping_times(reference: dict[str, Any]) -> None:
    composite, _ = _bermudan_composite()
    _assert_times(composite.stopping_times(), reference, "join_c2_stopping_times")
    assert len(composite.conditions()) == reference["join_c2_num_conditions"]


def test_join_adds_the_snapshot_time(reference: dict[str, Any]) -> None:
    composite, _ = _bermudan_composite()
    joined = FdmStepConditionComposite.join_conditions(FdmSnapshotCondition(0.75), composite)
    _assert_times(joined.stopping_times(), reference, "join_stopping_times")
    assert len(joined.conditions()) == reference["join_num_conditions"]


def test_join_at_a_non_exercise_snapshot_time(reference: dict[str, Any]) -> None:
    """At 0.75 the Bermudan condition is inert, so the snapshot sees the seed."""
    composite, _ = _bermudan_composite()
    snapshot = FdmSnapshotCondition(0.75)
    joined = FdmStepConditionComposite.join_conditions(snapshot, composite)
    a: Array = _fixtures.ref_array(reference, "join_seed")
    joined.apply_to(a, 0.75)
    _fixtures.assert_array_tight(a, reference, "join_out_at_snapshot")
    _fixtures.assert_array_tight(snapshot.get_values(), reference, "join_snapshot_values")


def test_snapshot_records_post_condition_values(reference: dict[str, Any]) -> None:
    """Snapshot time == exercise time: the recorded values must be exercised.

    C++ ``joinConditions`` pushes ``c2`` then ``c1``, so the snapshot is last.
    A port that reversed the order would record the seed here instead.
    """
    composite, bermudan = _bermudan_composite()
    ex0 = bermudan.exercise_times()[0]
    snapshot = FdmSnapshotCondition(ex0)
    joined = FdmStepConditionComposite.join_conditions(snapshot, composite)
    _assert_times(joined.stopping_times(), reference, "join2_stopping_times")

    a: Array = _fixtures.ref_array(reference, "join_seed")
    joined.apply_to(a, ex0)
    _fixtures.assert_array_tight(a, reference, "join2_out_at_exercise")
    _fixtures.assert_array_tight(snapshot.get_values(), reference, "join2_snapshot_values")
    # ...and that really is different from the seed.
    seed: list[float] = reference["join_seed"]
    post: list[float] = reference["join2_snapshot_values"]
    assert seed != post
