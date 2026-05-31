"""Tests for the cross-currency basis-swap rate helpers.

Cross-validates against ``migration-harness/references/cluster/w8d.json``.

C++ parity: ql/experimental/termstructures/crosscurrencyratehelpers.{hpp,cpp}
@ v1.42.1.
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.daycounters.thirty_360 import Convention, Thirty360
from pquantlib.exceptions import LibraryException
from pquantlib.experimental.termstructures.cross_currency_rate_helpers import (
    ConstNotionalCrossCurrencyBasisSwapRateHelper,
    ConstNotionalCrossCurrencySwapRateHelper,
    MtMCrossCurrencyBasisSwapRateHelper,
)
from pquantlib.indexes.ibor.euribor import Euribor
from pquantlib.indexes.ibor.usd_libor import USDLibor
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing.reference_reader import load as load_reference
from pquantlib.testing.tolerance import loose
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.date import Date
from pquantlib.time.frequency import Frequency
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit

_REF = Date.from_ymd(15, Month.January, 2024)
_MF = BusinessDayConvention.ModifiedFollowing


@pytest.fixture
def ref() -> dict[str, Any]:
    return load_reference("cluster/w8d")


@pytest.fixture(autouse=True)
def _set_eval_date() -> None:  # pyright: ignore[reportUnusedFunction]
    ObservableSettings().evaluation_date = _REF


def _curve(rate: float) -> FlatForward:
    return FlatForward(_REF, SimpleQuote(rate), Actual365Fixed())


def _eur_idx() -> Euribor:
    return Euribor(Period(3, TimeUnit.Months), _curve(0.025))


def _usd_idx() -> USDLibor:
    return USDLibor(Period(3, TimeUnit.Months), _curve(0.040))


def test_const_notional_basis_implied_quote(ref: dict[str, Any]) -> None:
    eur_fwd = _curve(0.025)
    eur_idx = Euribor(Period(3, TimeUnit.Months), eur_fwd)
    cn = ConstNotionalCrossCurrencyBasisSwapRateHelper(
        SimpleQuote(-0.0020),
        Period(5, TimeUnit.Years),
        2,
        TARGET(),
        _MF,
        False,
        eur_idx,
        _usd_idx(),
        _curve(0.038),
        False,
        True,
        Frequency.Quarterly,
        0,
    )
    cn.set_term_structure(eur_fwd)
    loose(cn.implied_quote(), ref["xccy_const_implied_quote"])


def test_mtm_basis_implied_quote(ref: dict[str, Any]) -> None:
    eur_fwd = _curve(0.025)
    eur_idx = Euribor(Period(3, TimeUnit.Months), eur_fwd)
    mtm = MtMCrossCurrencyBasisSwapRateHelper(
        SimpleQuote(-0.0020),
        Period(5, TimeUnit.Years),
        2,
        TARGET(),
        _MF,
        False,
        eur_idx,
        _usd_idx(),
        _curve(0.038),
        False,
        True,
        True,
        Frequency.Quarterly,
        0,
    )
    mtm.set_term_structure(eur_fwd)
    loose(mtm.implied_quote(), ref["xccy_mtm_implied_quote"])


def test_fixed_float_par_implied_quote(ref: dict[str, Any]) -> None:
    usd_fwd = _curve(0.040)
    usd_idx = USDLibor(Period(3, TimeUnit.Months), usd_fwd)
    ff = ConstNotionalCrossCurrencySwapRateHelper(
        SimpleQuote(0.02),
        Period(5, TimeUnit.Years),
        2,
        TARGET(),
        _MF,
        False,
        Frequency.Annual,
        Thirty360(Convention.BondBasis),
        usd_idx,
        _curve(0.038),
        True,
        0,
    )
    ff.set_term_structure(usd_fwd)
    loose(ff.implied_quote(), ref["xccy_fixfloat_implied_quote"])


def test_term_structure_required() -> None:
    """``implied_quote`` raises before the term structure is set."""
    cn = ConstNotionalCrossCurrencyBasisSwapRateHelper(
        SimpleQuote(-0.0020),
        Period(5, TimeUnit.Years),
        2,
        TARGET(),
        _MF,
        False,
        _eur_idx(),
        _usd_idx(),
        _curve(0.038),
        False,
        True,
        Frequency.Quarterly,
        0,
    )
    with pytest.raises(LibraryException):
        cn.implied_quote()
