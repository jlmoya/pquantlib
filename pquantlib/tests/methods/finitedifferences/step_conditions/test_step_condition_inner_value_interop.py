"""The step conditions drive the real ``FdmInnerValueCalculator`` hierarchy.

# C++ parity: the step conditions in
# ql/methods/finitedifferences/stepconditions/ take an
# ``ext::shared_ptr<FdmInnerValueCalculator>`` @ v1.43.

The other test modules in this package deliberately use a local calculator
stub, so a condition's own arithmetic can be pinned without dragging in the
utilities cluster. This module closes the loop: it re-runs one reference case
per condition with the **real**
``pquantlib.methods.finitedifferences.utilities.fdm_inner_value_calculator.FdmLogInnerValue``
and asserts the same C++ reference values, proving that

  * ``FdmLogInnerValue`` satisfies the ``FdmInnerValueCalculatorLike``
    protocol the conditions declare, and
  * it reproduces C++ ``FdmLogInnerValue::innerValue`` node for node.
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.math.array import Array
from pquantlib.methods.finitedifferences.meshers.fdm_mesher import FdmMesher
from pquantlib.methods.finitedifferences.step_conditions.fdm_bermudan_step_condition import (
    FdmBermudanStepCondition,
)
from pquantlib.methods.finitedifferences.step_conditions.fdm_simple_storage_condition import (
    FdmSimpleStorageCondition,
)
from pquantlib.methods.finitedifferences.step_conditions.fdm_simple_swing_condition import (
    FdmSimpleSwingCondition,
)
from pquantlib.methods.finitedifferences.step_conditions.inner_value_calculator_protocol import (
    FdmInnerValueCalculatorLike,
)
from pquantlib.methods.finitedifferences.utilities.fdm_inner_value_calculator import (
    FdmLogInnerValue,
)
from pquantlib.payoffs import OptionType, PlainVanillaPayoff
from pquantlib.time.date import Date, Month
from tests.methods.finitedifferences.step_conditions import _fixtures


@pytest.fixture(scope="module")
def reference() -> dict[str, Any]:
    return _fixtures.load()


def _log_inner(mesher: FdmMesher, option_type: OptionType, strike: float) -> FdmLogInnerValue:
    return FdmLogInnerValue(PlainVanillaPayoff(option_type, strike), mesher, 0)


def test_fdm_log_inner_value_satisfies_the_protocol() -> None:
    mesher = _fixtures.bermudan_mesher()
    calculator = _log_inner(mesher, OptionType.Put, 100.0)
    assert isinstance(calculator, FdmInnerValueCalculatorLike)


def test_bermudan_with_real_calculator(reference: dict[str, Any]) -> None:
    mesher = _fixtures.bermudan_mesher()
    cond = FdmBermudanStepCondition(
        [
            Date.from_ymd(15, Month.December, 2025),
            Date.from_ymd(15, Month.June, 2026),
            Date.from_ymd(15, Month.December, 2026),
        ],
        Date.from_ymd(15, Month.June, 2025),
        Actual365Fixed(),
        mesher,
        _log_inner(mesher, OptionType.Put, 100.0),
    )
    a: Array = _fixtures.ref_array(reference, "berm_seed")
    cond.apply_to(a, cond.exercise_times()[0])
    _fixtures.assert_array_tight(a, reference, "berm_out_ex0")


def test_swing_with_real_calculator(reference: dict[str, Any]) -> None:
    mesher = _fixtures.swing_mesher()
    cond = FdmSimpleSwingCondition(
        [0.25, 0.5, 0.75], mesher, _log_inner(mesher, OptionType.Call, 100.0), 1, 0
    )
    a: Array = _fixtures.ref_array(reference, "swing_seed")
    cond.apply_to(a, 0.5)
    _fixtures.assert_array_tight(a, reference, "swing_min0_t050")


def test_storage_with_real_calculator(reference: dict[str, Any]) -> None:
    mesher = _fixtures.storage_mesher()
    cond = FdmSimpleStorageCondition([0.25, 0.5], mesher, _log_inner(mesher, OptionType.Call, 0.0), 1.5)
    a: Array = _fixtures.ref_array(reference, "storage_seed")
    cond.apply_to(a, 0.25)
    _fixtures.assert_array_tight(a, reference, "storage_rate15_t025")
