"""Tests for the basis-swap rate helpers.

Cross-validates against ``migration-harness/references/cluster/w8d.json``.

C++ parity: ql/experimental/termstructures/basisswapratehelpers.{hpp,cpp}
@ v1.42.1.
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exceptions import LibraryException
from pquantlib.experimental.termstructures.basis_swap_rate_helpers import (
    IborIborBasisSwapRateHelper,
    OvernightIborBasisSwapRateHelper,
)
from pquantlib.indexes.ibor.euribor import Euribor
from pquantlib.indexes.ibor.sofr import Sofr
from pquantlib.indexes.ibor.usd_libor import USDLibor
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing.reference_reader import load as load_reference
from pquantlib.testing.tolerance import loose
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.calendars.united_states import UnitedStates
from pquantlib.time.date import Date
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit

_REF = Date.from_ymd(15, Month.January, 2024)


@pytest.fixture
def ref() -> dict[str, Any]:
    return load_reference("cluster/w8d")


@pytest.fixture(autouse=True)
def _set_eval_date() -> None:  # pyright: ignore[reportUnusedFunction]
    ObservableSettings().evaluation_date = _REF


def _curve(rate: float) -> FlatForward:
    return FlatForward(_REF, SimpleQuote(rate), Actual365Fixed())


def test_ibor_ibor_basis_implied_quote(ref: dict[str, Any]) -> None:
    """ibor-ibor basis helper reproduces the C++ implied basis."""
    base_idx = Euribor(Period(3, TimeUnit.Months), _curve(0.032))
    other_fwd = _curve(0.035)
    other_idx = Euribor(Period(6, TimeUnit.Months), other_fwd)
    h = IborIborBasisSwapRateHelper(
        SimpleQuote(0.0010),
        Period(5, TimeUnit.Years),
        2,
        TARGET(),
        BusinessDayConvention.ModifiedFollowing,
        False,
        base_idx,
        other_idx,
        _curve(0.030),
        False,
    )
    h.set_term_structure(other_fwd)
    loose(h.implied_quote(), ref["iibasis_implied_quote"])


def test_overnight_ibor_basis_implied_quote(ref: dict[str, Any]) -> None:
    """overnight-ibor basis helper reproduces the C++ implied basis."""
    cal = UnitedStates(UnitedStates.Market.GovernmentBond)
    on_idx = Sofr(_curve(0.031))
    ibor_fwd = _curve(0.034)
    ibor_idx = USDLibor(Period(3, TimeUnit.Months), ibor_fwd)
    h = OvernightIborBasisSwapRateHelper(
        SimpleQuote(0.0015),
        Period(5, TimeUnit.Years),
        2,
        cal,
        BusinessDayConvention.ModifiedFollowing,
        False,
        on_idx,
        ibor_idx,
        _curve(0.030),
    )
    h.set_term_structure(ibor_fwd)
    loose(h.implied_quote(), ref["onibasis_implied_quote"])


def test_swap_accessor() -> None:
    """``swap()`` returns the underlying two-leg basis swap."""
    base_idx = Euribor(Period(3, TimeUnit.Months), _curve(0.032))
    other_idx = Euribor(Period(6, TimeUnit.Months), _curve(0.035))
    h = IborIborBasisSwapRateHelper(
        SimpleQuote(0.0010),
        Period(5, TimeUnit.Years),
        2,
        TARGET(),
        BusinessDayConvention.ModifiedFollowing,
        False,
        base_idx,
        other_idx,
        _curve(0.030),
        False,
    )
    swap = h.swap()
    assert len(swap.legs()) == 2


def test_term_structure_required() -> None:
    """``implied_quote`` raises before the term structure is set."""
    base_idx = Euribor(Period(3, TimeUnit.Months), _curve(0.032))
    other_idx = Euribor(Period(6, TimeUnit.Months), _curve(0.035))
    h = IborIborBasisSwapRateHelper(
        SimpleQuote(0.0010),
        Period(5, TimeUnit.Years),
        2,
        TARGET(),
        BusinessDayConvention.ModifiedFollowing,
        False,
        base_idx,
        other_idx,
        _curve(0.030),
        False,
    )
    with pytest.raises(LibraryException):
        h.implied_quote()
