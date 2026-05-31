"""Behavioural tests for the W11-A composite / callability framework.

Covers ``MultiProductComposite``, ``SingleProductComposite``,
``ExerciseAdapter`` (+ its ``MarketModelExerciseValue`` Protocol), and
``CallSpecifiedMultiProduct`` (+ its ``ExerciseStrategy`` Protocol) using minimal
in-test stubs that satisfy the W11-C Protocol shapes. These are
self-consistent structural / behavioural checks (the C++ ``compositeproduct`` is
already validated end-to-end by the canonical BGM test; here we pin the
remaining composite-leaf wiring + the callability mechanics).

C++ parity:
  ql/models/marketmodels/products/{multiproductcomposite,singleproductcomposite}
  ql/models/marketmodels/products/multistep/{exerciseadapter,
    callspecifiedmultiproduct}
  @ v1.42.1 (099987f0).
"""

from __future__ import annotations

import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.models.marketmodels.curve_state import CurveState
from pquantlib.models.marketmodels.curvestates.lmm_curve_state import LMMCurveState
from pquantlib.models.marketmodels.evolution_description import EvolutionDescription
from pquantlib.models.marketmodels.multi_product import (
    CashFlow,
    MarketModelMultiProduct,
)
from pquantlib.models.marketmodels.products import (
    CallSpecifiedMultiProduct,
    ExerciseAdapter,
    ExerciseStrategy,
    MarketModelExerciseValue,
    MultiProductComposite,
    MultiStepForwards,
    SingleProductComposite,
)

_RATE_TIMES = [0.5, 1.0, 1.5, 2.0]
_FLAT = 0.05
_ACCRUALS = [0.5, 0.5, 0.5]
_PAY_TIMES = [1.0, 1.5, 2.0]
_STRIKES = [0.04, 0.045, 0.05]


def _flat_state() -> LMMCurveState:
    cs = LMMCurveState(_RATE_TIMES)
    cs.set_on_forward_rates([_FLAT, _FLAT, _FLAT])
    return cs


def _drive(product: MarketModelMultiProduct) -> list[float]:
    """Run a product to completion over a flat state; return per-product totals."""
    np_ = product.number_of_products()
    max_flows = product.max_number_of_cash_flows_per_product_per_step()
    n = [0] * np_
    gen = [[CashFlow() for _ in range(max_flows)] for _ in range(np_)]
    totals = [0.0] * np_
    product.reset()
    state = _flat_state()
    done = False
    guard = 0
    while not done and guard < 100:
        done = product.next_time_step(state, n, gen)
        for i in range(np_):
            for j in range(n[i]):
                totals[i] += gen[i][j].amount
        guard += 1
    return totals


# --- MultiProductComposite: union of two forwards products -------------------
def test_multiproduct_composite_union() -> None:
    a = MultiStepForwards(_RATE_TIMES, _ACCRUALS, _PAY_TIMES, _STRIKES)
    b = MultiStepForwards(_RATE_TIMES, _ACCRUALS, _PAY_TIMES, _STRIKES)
    comp = MultiProductComposite()
    comp.add(a)
    comp.add(b, multiplier=2.0)
    comp.finalize()
    assert comp.number_of_products() == 6  # 3 + 3
    assert comp.size() == 2
    assert comp.multiplier(1) == 2.0

    totals = _drive(comp)
    standalone = _drive(MultiStepForwards(_RATE_TIMES, _ACCRUALS, _PAY_TIMES, _STRIKES))
    # first 3 products == standalone; next 3 == 2x standalone.
    for i in range(3):
        assert abs(totals[i] - standalone[i]) < 1e-14
        assert abs(totals[i + 3] - 2.0 * standalone[i]) < 1e-14


def test_composite_requires_finalize() -> None:
    comp = MultiProductComposite()
    comp.add(MultiStepForwards(_RATE_TIMES, _ACCRUALS, _PAY_TIMES, _STRIKES))
    # evolution / cash-flow times require finalization first.
    with pytest.raises(LibraryException):
        comp.evolution()
    comp.finalize()
    assert comp.evolution().number_of_steps() == 3


def test_composite_clone_independent() -> None:
    comp = MultiProductComposite()
    comp.add(MultiStepForwards(_RATE_TIMES, _ACCRUALS, _PAY_TIMES, _STRIKES))
    comp.finalize()
    clone = comp.clone()
    assert clone.number_of_products() == comp.number_of_products()
    # driving the clone does not disturb the original.
    _drive(clone)
    assert abs(_drive(comp)[0] - _drive(clone)[0]) < 1e-14


# --- SingleProductComposite: collapse to one product -------------------------
def test_single_product_composite_collapses() -> None:
    a = MultiStepForwards(_RATE_TIMES, _ACCRUALS, _PAY_TIMES, _STRIKES)
    comp = SingleProductComposite()
    comp.add(a)
    comp.finalize()
    assert comp.number_of_products() == 1
    # the single product's total == sum of the 3 forwards' totals.
    standalone = _drive(MultiStepForwards(_RATE_TIMES, _ACCRUALS, _PAY_TIMES, _STRIKES))
    total = _drive(comp)[0]
    assert abs(total - sum(standalone)) < 1e-14


# --- Stub ExerciseValue / ExerciseStrategy (the W11-C Protocol shapes) --------
class _StubExerciseValue:
    """Never-exercise exercise value: 1 exercise opportunity, value 0.

    Demonstrates the ``MarketModelExerciseValue`` Protocol that W11-C concretes
    must satisfy.
    """

    def __init__(self) -> None:
        # single evolution time at the next-to-last rate time -> exercisable.
        self._evolution = EvolutionDescription(_RATE_TIMES, [_RATE_TIMES[-2]])

    def number_of_exercises(self) -> int:
        return 1

    def evolution(self) -> EvolutionDescription:
        return self._evolution

    def possible_cash_flow_times(self) -> list[float]:
        return [_RATE_TIMES[-2]]

    def next_step(self, current_state: CurveState) -> None:
        pass

    def reset(self) -> None:
        pass

    def is_exercise_time(self) -> list[bool]:
        return [True]

    def value(self, current_state: CurveState) -> CashFlow:
        return CashFlow(time_index=0, amount=0.0)

    def clone(self) -> _StubExerciseValue:
        return _StubExerciseValue()


class _AlwaysExerciseStrategy:
    """Exercise as soon as the first exercise time is reached.

    Demonstrates the ``ExerciseStrategy`` Protocol that W11-C concretes must
    satisfy.
    """

    def __init__(self, exercise: bool) -> None:
        self._exercise = exercise

    def exercise_times(self) -> list[float]:
        return [_RATE_TIMES[0]]

    def relevant_times(self) -> list[float]:
        return [_RATE_TIMES[0]]

    def reset(self) -> None:
        pass

    def exercise(self, current_state: CurveState) -> bool:
        return self._exercise

    def next_step(self, current_state: CurveState) -> None:
        pass

    def clone(self) -> _AlwaysExerciseStrategy:
        return _AlwaysExerciseStrategy(self._exercise)


def test_protocols_are_satisfied() -> None:
    # The stubs structurally satisfy the forward-declared W11-C Protocols.
    assert isinstance(_StubExerciseValue(), MarketModelExerciseValue)
    assert isinstance(_AlwaysExerciseStrategy(True), ExerciseStrategy)


# --- ExerciseAdapter ---------------------------------------------------------
def test_exercise_adapter() -> None:
    adapter = ExerciseAdapter(_StubExerciseValue())
    assert adapter.number_of_products() == 1
    assert adapter.max_number_of_cash_flows_per_product_per_step() == 1
    assert adapter.possible_cash_flow_times() == [_RATE_TIMES[-2]]
    # at the (single) exercise time it pays the exercise value (0 here) + done.
    totals = _drive(adapter)
    assert abs(totals[0]) < 1e-14
    # the adapter exposes its exercise value.
    assert isinstance(adapter.exercise_value(), MarketModelExerciseValue)


# --- CallSpecifiedMultiProduct ----------------------------------------------
def test_call_specified_not_exercised_pays_underlying() -> None:
    underlying = MultiStepForwards(_RATE_TIMES, _ACCRUALS, _PAY_TIMES, _STRIKES)
    # strategy never exercises -> underlying cash flows flow through unchanged.
    callable_product = CallSpecifiedMultiProduct(
        underlying, _AlwaysExerciseStrategy(exercise=False)
    )
    assert callable_product.number_of_products() == 3
    totals = _drive(callable_product)
    standalone = _drive(MultiStepForwards(_RATE_TIMES, _ACCRUALS, _PAY_TIMES, _STRIKES))
    for i in range(3):
        assert abs(totals[i] - standalone[i]) < 1e-14


def test_call_specified_exercised_pays_rebate() -> None:
    underlying = MultiStepForwards(_RATE_TIMES, _ACCRUALS, _PAY_TIMES, _STRIKES)
    # strategy exercises immediately -> default zero rebate, underlying killed.
    callable_product = CallSpecifiedMultiProduct(
        underlying, _AlwaysExerciseStrategy(exercise=True)
    )
    totals = _drive(callable_product)
    # default rebate amounts are all zero -> nothing paid.
    assert all(abs(t) < 1e-14 for t in totals)


def test_call_specified_disable_callability() -> None:
    underlying = MultiStepForwards(_RATE_TIMES, _ACCRUALS, _PAY_TIMES, _STRIKES)
    callable_product = CallSpecifiedMultiProduct(
        underlying, _AlwaysExerciseStrategy(exercise=True)
    )
    # disabling callability -> behaves like the underlying despite the strategy.
    callable_product.disable_callability()
    totals = _drive(callable_product)
    standalone = _drive(MultiStepForwards(_RATE_TIMES, _ACCRUALS, _PAY_TIMES, _STRIKES))
    for i in range(3):
        assert abs(totals[i] - standalone[i]) < 1e-14
    callable_product.enable_callability()


def test_call_specified_inspectors() -> None:
    underlying = MultiStepForwards(_RATE_TIMES, _ACCRUALS, _PAY_TIMES, _STRIKES)
    strategy = _AlwaysExerciseStrategy(exercise=False)
    callable_product = CallSpecifiedMultiProduct(underlying, strategy)
    assert callable_product.underlying().number_of_products() == 3
    assert isinstance(callable_product.strategy(), ExerciseStrategy)
    assert callable_product.rebate().number_of_products() == 3
    # the combined cash-flow-time vector includes the rebate's times (offset).
    assert len(callable_product.possible_cash_flow_times()) > len(_PAY_TIMES)
