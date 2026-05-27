"""Tests for BlackConstantVol — cross-validated against L2-E probe."""

from __future__ import annotations

import math

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.volatility.equity_fx.black_constant_vol import BlackConstantVol
from pquantlib.testing import reference_reader, tolerance
from pquantlib.time.calendars.null_calendar import NullCalendar
from pquantlib.time.date import Date
from pquantlib.time.month import Month

_REF = reference_reader.load("cluster/l2e")
_BCV = _REF["black_constant_vol"]
_BCV_Q = _REF["black_constant_vol_via_quote"]


def _new_bcv(vol: float = 0.20) -> BlackConstantVol:
    return BlackConstantVol(
        reference_date=Date.from_ymd(15, Month.June, 2026),
        calendar=NullCalendar(),
        day_counter=Actual365Fixed(),
        volatility=vol,
    )


def test_reference_date_round_trip() -> None:
    bcv = _new_bcv()
    assert bcv.reference_date().serial == _BCV["reference_date_serial"]


def test_max_date_is_unbounded() -> None:
    bcv = _new_bcv()
    assert bcv.max_date() == Date.max_date()


def test_min_max_strike_are_infinite() -> None:
    bcv = _new_bcv()
    assert bcv.min_strike() == -math.inf
    assert bcv.max_strike() == math.inf


def test_black_vol_constant_regardless_of_time_or_strike() -> None:
    bcv = _new_bcv()
    tolerance.exact(bcv.black_vol_at_time(1.0, 100.0), _BCV["vol_t1"])
    tolerance.exact(bcv.black_vol_at_time(2.0, 90.0), _BCV["vol_t2"])


def test_black_variance_constant_in_strike_linear_in_t() -> None:
    bcv = _new_bcv()
    tolerance.tight(bcv.black_variance_at_time(1.0, 100.0), _BCV["variance_t1"])
    tolerance.tight(bcv.black_variance_at_time(2.0, 90.0), _BCV["variance_t2"])


def test_forward_vol_equals_spot_vol_for_constant() -> None:
    bcv = _new_bcv()
    tolerance.exact(
        bcv.black_forward_vol_at_time(1.0, 2.0, 100.0), _BCV["forward_vol_t1_t2"]
    )


def test_forward_variance_equals_diff_for_constant() -> None:
    bcv = _new_bcv()
    tolerance.tight(
        bcv.black_forward_variance_at_time(1.0, 2.0, 100.0),
        _BCV["forward_variance_t1_t2"],
    )


# --- Quote-driven constructor ------------------------------------------------


def test_quote_driven_initial_vol() -> None:
    q = SimpleQuote(0.25)
    bcv = BlackConstantVol(
        reference_date=Date.from_ymd(15, Month.June, 2026),
        calendar=NullCalendar(),
        day_counter=Actual365Fixed(),
        volatility=q,
    )
    tolerance.exact(bcv.black_vol_at_time(1.0, 100.0), _BCV_Q["initial_vol"])


def test_quote_driven_update_propagates() -> None:
    q = SimpleQuote(0.25)
    bcv = BlackConstantVol(
        reference_date=Date.from_ymd(15, Month.June, 2026),
        calendar=NullCalendar(),
        day_counter=Actual365Fixed(),
        volatility=q,
    )
    q.set_value(0.30)
    tolerance.exact(bcv.black_vol_at_time(1.0, 100.0), _BCV_Q["after_update_vol"])


def test_quote_change_notifies_term_structure_observers() -> None:
    q = SimpleQuote(0.25)
    bcv = BlackConstantVol(
        reference_date=Date.from_ymd(15, Month.June, 2026),
        calendar=NullCalendar(),
        day_counter=Actual365Fixed(),
        volatility=q,
    )

    counts = [0]

    class _Counter:
        def update(self) -> None:
            counts[0] += 1

    obs = _Counter()
    bcv.register_with(obs)
    q.set_value(0.30)
    assert counts[0] == 1
