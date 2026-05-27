"""Tests for the abstract Swap class + result carriers.

# C++ parity: ql/instruments/swap.{hpp,cpp} (v1.42.1).
"""

from __future__ import annotations

import pytest

from pquantlib.cashflows.simple_cash_flow import SimpleCashFlow
from pquantlib.exceptions import LibraryException
from pquantlib.instruments.swap import (
    Swap,
    SwapArguments,
    SwapResults,
    SwapType,
    leg_maturity_date,
    leg_start_date,
)
from pquantlib.time.date import Date
from pquantlib.time.month import Month


def _date(d: int, m: Month, y: int) -> Date:
    return Date.from_ymd(d, m, y)


# --- SwapType -----------------------------------------------------------


def test_swap_type_int_values() -> None:
    assert int(SwapType.Receiver) == -1
    assert int(SwapType.Payer) == 1


# --- SwapArguments validate ---------------------------------------------


def test_swap_arguments_validate_ok() -> None:
    args = SwapArguments()
    args.legs = [[], []]
    args.payer = [-1.0, 1.0]
    args.validate()  # no raise


def test_swap_arguments_validate_size_mismatch_raises() -> None:
    args = SwapArguments()
    args.legs = [[], []]
    args.payer = [1.0]
    with pytest.raises(LibraryException, match="number of legs and multipliers differ"):
        args.validate()


# --- SwapResults.reset ---------------------------------------------------


def test_swap_results_reset_clears_lists() -> None:
    r = SwapResults()
    r.value = 1.0
    r.leg_npv = [1.0, 2.0]
    r.leg_bps = [3.0]
    r.start_discounts = [0.99]
    r.end_discounts = [0.95]
    r.npv_date_discount = 1.0
    r.reset()
    assert r.value is None
    assert r.leg_npv == []
    assert r.leg_bps == []
    assert r.start_discounts == []
    assert r.end_discounts == []
    assert r.npv_date_discount is None


# --- leg_start_date / leg_maturity_date ----------------------------------


def test_leg_start_maturity_for_simple_cashflows() -> None:
    cf1 = SimpleCashFlow(100.0, _date(15, Month.January, 2024))
    cf2 = SimpleCashFlow(200.0, _date(15, Month.July, 2024))
    leg = [cf1, cf2]
    assert leg_start_date(leg) == _date(15, Month.January, 2024)
    assert leg_maturity_date(leg) == _date(15, Month.July, 2024)


def test_leg_helpers_empty_leg_raises() -> None:
    with pytest.raises(LibraryException, match="empty leg"):
        leg_start_date([])
    with pytest.raises(LibraryException, match="empty leg"):
        leg_maturity_date([])


# --- Swap.from_legs / from_multi ----------------------------------------


def _build_two_leg_swap() -> Swap:
    """Construct a minimal Swap (paid 100 on Jan 24, received 105 on Jul 24)."""
    leg1 = [SimpleCashFlow(100.0, _date(15, Month.January, 2024))]
    leg2 = [SimpleCashFlow(105.0, _date(15, Month.July, 2024))]
    return Swap.from_legs(leg1, leg2)


def test_from_legs_sets_payer_signs() -> None:
    s = _build_two_leg_swap()
    assert s.number_of_legs() == 2
    # First leg paid (sign -1), second received (sign +1).
    assert s.payer(0)
    assert not s.payer(1)


def test_from_multi_with_explicit_payer_flags() -> None:
    leg_a = [SimpleCashFlow(50.0, _date(15, Month.January, 2024))]
    leg_b = [SimpleCashFlow(60.0, _date(15, Month.April, 2024))]
    leg_c = [SimpleCashFlow(70.0, _date(15, Month.July, 2024))]
    s = Swap.from_multi([leg_a, leg_b, leg_c], [True, False, False])
    assert s.number_of_legs() == 3
    assert s.payer(0)
    assert not s.payer(1)
    assert not s.payer(2)


def test_from_multi_size_mismatch_raises() -> None:
    leg = [SimpleCashFlow(50.0, _date(15, Month.January, 2024))]
    with pytest.raises(LibraryException, match="size mismatch"):
        Swap.from_multi([leg, leg], [True])


def test_start_and_maturity_date() -> None:
    s = _build_two_leg_swap()
    assert s.start_date() == _date(15, Month.January, 2024)
    assert s.maturity_date() == _date(15, Month.July, 2024)


def test_leg_index_out_of_range_raises() -> None:
    s = _build_two_leg_swap()
    with pytest.raises(LibraryException, match="doesn't exist"):
        s.leg(2)
    with pytest.raises(LibraryException, match="doesn't exist"):
        s.payer(2)
