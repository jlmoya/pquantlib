"""Smoke tests for the experimental volatility surface abstract bases.

# C++ parity: ql/experimental/volatility/{blackatmvolcurve, blackvolsurface,
#             equityfxvolsurface, interestratevolsurface, volcube}.* (v1.42.1).

The bases are abstract, so each test drives a *minimal concrete
subclass* that returns a constant FlatSmileSection. This exercises the
``smile_section`` -> ``atm_vol`` / ``atm_variance`` dispatch chain and
the forward-vol / index-date-conversion logic.
"""

from __future__ import annotations

import math

import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exceptions import LibraryException
from pquantlib.experimental.volatility.black_atm_vol_curve import BlackAtmVolCurve
from pquantlib.experimental.volatility.black_vol_surface import BlackVolSurface
from pquantlib.experimental.volatility.equity_fx_vol_surface import EquityFXVolSurface
from pquantlib.experimental.volatility.interest_rate_vol_surface import (
    InterestRateVolSurface,
)
from pquantlib.experimental.volatility.vol_cube import VolatilityCube
from pquantlib.indexes.ibor.euribor import Euribor
from pquantlib.termstructures.volatility.flat_smile_section import FlatSmileSection
from pquantlib.termstructures.volatility.smile_section import SmileSection
from pquantlib.testing import tolerance
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.date import Date
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit

_REF_DATE = Date.from_ymd(15, Month.May, 2026)
_DC = Actual365Fixed()
_FLAT_VOL = 0.20
_ATM_LEVEL = 100.0


# --- minimal concretes -------------------------------------------------


class _ConstAtmCurve(BlackAtmVolCurve):
    """ATM curve returning a flat vol (variance = vol^2 * t)."""

    def __init__(self) -> None:
        super().__init__(
            reference_date=_REF_DATE,
            calendar=TARGET(),
            day_counter=_DC,
        )

    def max_date(self) -> Date:
        return Date.max_date()

    def min_strike(self) -> float:
        return -math.inf

    def max_strike(self) -> float:
        return math.inf

    def _atm_vol_impl(self, t: float) -> float:
        return _FLAT_VOL

    def _atm_variance_impl(self, t: float) -> float:
        return _FLAT_VOL * _FLAT_VOL * t


class _ConstSmileSurface(BlackVolSurface):
    """BlackVolSurface returning a FlatSmileSection per expiry."""

    def __init__(self) -> None:
        super().__init__(
            business_day_convention=BusinessDayConvention.Following,
            reference_date=_REF_DATE,
            calendar=TARGET(),
            day_counter=_DC,
        )

    def max_date(self) -> Date:
        return Date.max_date()

    def min_strike(self) -> float:
        return -math.inf

    def max_strike(self) -> float:
        return math.inf

    def _smile_section_impl(self, t: float) -> SmileSection:
        return FlatSmileSection(
            volatility=_FLAT_VOL,
            exercise_time=t,
            atm_level=_ATM_LEVEL,
        )


class _ConstEquityFxSurface(EquityFXVolSurface):
    """EquityFXVolSurface returning a FlatSmileSection per expiry."""

    def __init__(self) -> None:
        super().__init__(
            business_day_convention=BusinessDayConvention.Following,
            reference_date=_REF_DATE,
            calendar=TARGET(),
            day_counter=_DC,
        )

    def max_date(self) -> Date:
        return Date.max_date()

    def min_strike(self) -> float:
        return -math.inf

    def max_strike(self) -> float:
        return math.inf

    def _smile_section_impl(self, t: float) -> SmileSection:
        return FlatSmileSection(
            volatility=_FLAT_VOL,
            exercise_time=t,
            atm_level=_ATM_LEVEL,
        )


class _ConstIrSurface(InterestRateVolSurface):
    """InterestRateVolSurface returning a FlatSmileSection per expiry."""

    def __init__(self) -> None:
        super().__init__(
            Euribor.six_months(),
            business_day_convention=BusinessDayConvention.Following,
            reference_date=_REF_DATE,
            calendar=TARGET(),
            day_counter=_DC,
        )

    def max_date(self) -> Date:
        return Date.max_date()

    def min_strike(self) -> float:
        return -math.inf

    def max_strike(self) -> float:
        return math.inf

    def _smile_section_impl(self, t: float) -> SmileSection:
        return FlatSmileSection(
            volatility=_FLAT_VOL,
            exercise_time=t,
            atm_level=_ATM_LEVEL,
        )


# --- BlackAtmVolCurve --------------------------------------------------


def test_black_atm_vol_curve_atm_vol_and_variance() -> None:
    curve = _ConstAtmCurve()
    d = _REF_DATE + Period(1, TimeUnit.Years)
    t = _DC.year_fraction(_REF_DATE, d)
    # by date.
    tolerance.tight(curve.atm_vol_at_date(d), _FLAT_VOL)
    tolerance.tight(curve.atm_variance_at_date(d), _FLAT_VOL * _FLAT_VOL * t)
    # by time.
    tolerance.tight(curve.atm_vol_at_time(t), _FLAT_VOL)
    tolerance.tight(curve.atm_variance_at_time(t), _FLAT_VOL * _FLAT_VOL * t)
    # by tenor.
    tolerance.tight(curve.atm_vol(Period(1, TimeUnit.Years)), _FLAT_VOL)


def test_black_atm_vol_curve_range_check_raises() -> None:
    curve = _ConstAtmCurve()
    # 1Y beyond... actually the curve allows up to maxDate (Date.max);
    # negative time triggers the range check instead.
    with pytest.raises(LibraryException):
        curve.atm_vol_at_time(-1.0, False)


# --- BlackVolSurface ---------------------------------------------------


def test_black_vol_surface_smile_section_dispatch() -> None:
    surface = _ConstSmileSurface()
    d = _REF_DATE + Period(2, TimeUnit.Years)
    t = _DC.year_fraction(_REF_DATE, d)
    smile = surface.smile_section_at_date(d, True)
    tolerance.tight(smile.volatility(123.0), _FLAT_VOL)
    tolerance.tight(smile.atm_level(), _ATM_LEVEL)
    # The ATM vol/variance derive from the smile at its ATM level.
    tolerance.tight(surface.atm_vol_at_date(d), _FLAT_VOL)
    tolerance.tight(surface.atm_variance_at_date(d), _FLAT_VOL * _FLAT_VOL * t)


def test_black_vol_surface_smile_section_by_tenor_and_time() -> None:
    surface = _ConstSmileSurface()
    smile_tenor = surface.smile_section(Period(3, TimeUnit.Years), True)
    tolerance.tight(smile_tenor.volatility(99.0), _FLAT_VOL)
    smile_time = surface.smile_section_at_time(1.5, True)
    tolerance.tight(smile_time.exercise_time(), 1.5)


# --- EquityFXVolSurface ------------------------------------------------


def test_equity_fx_surface_atm_forward_vol_and_variance() -> None:
    surface = _ConstEquityFxSurface()
    d1 = _REF_DATE + Period(1, TimeUnit.Years)
    d2 = _REF_DATE + Period(2, TimeUnit.Years)
    t1 = _DC.year_fraction(_REF_DATE, d1)
    t2 = _DC.year_fraction(_REF_DATE, d2)
    # forward variance = vol^2 * (t2 - t1); forward vol = sqrt(fwdVar/(t2-t1)).
    fwd_var = surface.atm_forward_variance(d1, d2)
    tolerance.tight(fwd_var, _FLAT_VOL * _FLAT_VOL * (t2 - t1))
    tolerance.tight(surface.atm_forward_vol(d1, d2), _FLAT_VOL)
    # time-anchored variants agree.
    tolerance.tight(surface.atm_forward_variance_at_time(t1, t2), fwd_var)
    tolerance.tight(surface.atm_forward_vol_at_time(t1, t2), _FLAT_VOL)


def test_equity_fx_surface_wrong_dates_raise() -> None:
    surface = _ConstEquityFxSurface()
    d1 = _REF_DATE + Period(2, TimeUnit.Years)
    d2 = _REF_DATE + Period(1, TimeUnit.Years)
    with pytest.raises(LibraryException):
        surface.atm_forward_vol(d1, d2)


# --- InterestRateVolSurface --------------------------------------------


def test_ir_surface_index_and_smile() -> None:
    surface = _ConstIrSurface()
    assert surface.index().family_name() == "Euribor"
    # optionlet-style date conversion is monotone in tenor.
    d1 = surface.option_date_from_tenor(Period(1, TimeUnit.Years))
    d5 = surface.option_date_from_tenor(Period(5, TimeUnit.Years))
    assert d5 > d1
    smile = surface.smile_section(Period(1, TimeUnit.Years), True)
    tolerance.tight(smile.volatility(0.03), _FLAT_VOL)


# --- VolatilityCube ----------------------------------------------------


def test_vol_cube_aggregation_and_ref_date_check() -> None:
    s1 = _ConstIrSurface()
    s2 = _ConstIrSurface()
    cube = VolatilityCube([s1, s2], [])
    assert len(cube.surfaces()) == 2
    assert cube.curves() == []
    assert cube.surfaces()[0].reference_date() == _REF_DATE


def test_vol_cube_requires_two_surfaces() -> None:
    with pytest.raises(LibraryException):
        VolatilityCube([_ConstIrSurface()], [])
