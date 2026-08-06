"""Cross-validation tests for ``FdmBermudanStepCondition``.

# C++ parity: ql/methods/finitedifferences/stepconditions/fdmbermudanstepcondition.{hpp,cpp}
# @ v1.43.

Reference: ``migration-harness/references/v143/methods/stepconditions.json``,
``berm_*`` keys.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exceptions import LibraryException
from pquantlib.math.array import Array
from pquantlib.methods.finitedifferences.step_conditions.fdm_bermudan_step_condition import (
    FdmBermudanStepCondition,
)
from pquantlib.methods.finitedifferences.step_conditions.step_condition import (
    StepCondition,
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


def _condition() -> FdmBermudanStepCondition:
    mesher = _fixtures.bermudan_mesher()
    return FdmBermudanStepCondition(
        _EXERCISE_DATES,
        _REF_DATE,
        Actual365Fixed(),
        mesher,
        _fixtures.put_100(mesher),
    )


def test_is_a_step_condition() -> None:
    assert isinstance(_condition(), StepCondition)


def test_mesher_layout_matches_probe(reference: dict[str, Any]) -> None:
    mesher = _fixtures.bermudan_mesher()
    assert mesher.layout().size() == reference["berm_size"]
    assert list(mesher.layout().dim()) == reference["berm_dim"]
    _fixtures.assert_array_tight(mesher.locations(0), reference, "berm_locations_0")
    _fixtures.assert_array_tight(mesher.locations(1), reference, "berm_locations_1")


def test_inner_value_stub_matches_cpp_fdm_log_inner_value(reference: dict[str, Any]) -> None:
    """The test's calculator stub is itself cross-validated before anything uses it."""
    mesher = _fixtures.bermudan_mesher()
    values = _fixtures.inner_values(mesher, _fixtures.put_100(mesher), 0.5)
    _fixtures.assert_array_tight(values, reference, "berm_inner")


def test_exercise_times_from_dates(reference: dict[str, Any]) -> None:
    """TIGHT: ``Actual365Fixed`` year fractions are a single integer division."""
    times = _condition().exercise_times()
    expected: list[float] = reference["berm_exercise_times"]
    assert len(times) == len(expected)
    for i, exp in enumerate(expected):
        tight(times[i], float(exp), reason=f"berm_exercise_times[{i}]")


def test_exercise_times_returns_a_copy() -> None:
    cond = _condition()
    times = cond.exercise_times()
    times[0] = -1.0
    assert cond.exercise_times()[0] != -1.0


def test_no_op_away_from_an_exercise_time(reference: dict[str, Any]) -> None:
    a: Array = _fixtures.ref_array(reference, "berm_seed")
    _condition().apply_to(a, 0.3)
    _fixtures.assert_array_tight(a, reference, "berm_out_no_exercise")


@pytest.mark.parametrize(
    ("index", "key"),
    [(0, "berm_out_ex0"), (1, "berm_out_ex1"), (2, "berm_out_ex2")],
)
def test_floors_at_inner_value_on_each_exercise_time(reference: dict[str, Any], index: int, key: str) -> None:
    cond = _condition()
    a: Array = _fixtures.ref_array(reference, "berm_seed")
    cond.apply_to(a, cond.exercise_times()[index])
    _fixtures.assert_array_tight(a, reference, key)


def test_idempotent_at_an_exercise_time(reference: dict[str, Any]) -> None:
    cond = _condition()
    a: Array = _fixtures.ref_array(reference, "berm_seed")
    cond.apply_to(a, cond.exercise_times()[0])
    cond.apply_to(a, cond.exercise_times()[0])
    _fixtures.assert_array_tight(a, reference, "berm_out_ex0_twice")


def test_time_test_is_exact_not_tolerant(reference: dict[str, Any]) -> None:
    """C++ ``std::find`` uses ``==``: one ULP off an exercise time is a no-op."""
    cond = _condition()
    a: Array = _fixtures.ref_array(reference, "berm_seed")
    cond.apply_to(a, math.nextafter(cond.exercise_times()[0], 1.0))
    _fixtures.assert_array_tight(a, reference, "berm_out_nextafter_ex0")


def test_inconsistent_array_dimensions_raises() -> None:
    """C++ ``QL_REQUIRE(mesher_->layout()->size() == a.size(), ...)``."""
    cond = _condition()
    bad: Array = np.zeros(3, dtype=np.float64)
    with pytest.raises(LibraryException):
        cond.apply_to(bad, cond.exercise_times()[0])


def test_empty_schedule_never_fires() -> None:
    mesher = _fixtures.bermudan_mesher()
    cond = FdmBermudanStepCondition([], _REF_DATE, Actual365Fixed(), mesher, _fixtures.put_100(mesher))
    assert cond.exercise_times() == []
    a: Array = np.zeros(mesher.layout().size(), dtype=np.float64)
    cond.apply_to(a, 0.0)
    assert not a.any()
