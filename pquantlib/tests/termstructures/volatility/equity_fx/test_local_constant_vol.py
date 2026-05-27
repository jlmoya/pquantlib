"""Tests for LocalConstantVol — cross-validated against L2-E probe."""

from __future__ import annotations

import math

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.volatility.equity_fx.local_constant_vol import (
    LocalConstantVol,
)
from pquantlib.testing import reference_reader, tolerance
from pquantlib.time.date import Date
from pquantlib.time.month import Month

_REF = reference_reader.load("cluster/l2e")
_LCV = _REF["local_constant_vol"]


def _new() -> LocalConstantVol:
    return LocalConstantVol(
        reference_date=Date.from_ymd(15, Month.June, 2026),
        volatility=0.20,
        day_counter=Actual365Fixed(),
    )


def test_max_date_is_unbounded() -> None:
    lcv = _new()
    assert lcv.max_date() == Date.max_date()


def test_min_max_strike_are_infinite() -> None:
    lcv = _new()
    assert lcv.min_strike() == -math.inf
    assert lcv.max_strike() == math.inf


def test_local_vol_constant_regardless_of_time_or_strike() -> None:
    lcv = _new()
    tolerance.exact(lcv.local_vol_at_time(1.0, 100.0), _LCV["vol_t1_s100"])
    tolerance.exact(lcv.local_vol_at_time(2.0, 90.0), _LCV["vol_t2_s90"])


def test_quote_driven_construction_and_update() -> None:
    q = SimpleQuote(0.20)
    lcv = LocalConstantVol(
        reference_date=Date.from_ymd(15, Month.June, 2026),
        volatility=q,
        day_counter=Actual365Fixed(),
    )
    tolerance.exact(lcv.local_vol_at_time(1.0, 100.0), 0.20)
    q.set_value(0.30)
    tolerance.exact(lcv.local_vol_at_time(1.0, 100.0), 0.30)


def test_quote_update_notifies_term_structure_observers() -> None:
    q = SimpleQuote(0.20)
    lcv = LocalConstantVol(
        reference_date=Date.from_ymd(15, Month.June, 2026),
        volatility=q,
        day_counter=Actual365Fixed(),
    )
    counts = [0]

    class _Counter:
        def update(self) -> None:
            counts[0] += 1

    obs = _Counter()
    lcv.register_with(obs)
    q.set_value(0.30)
    assert counts[0] == 1
