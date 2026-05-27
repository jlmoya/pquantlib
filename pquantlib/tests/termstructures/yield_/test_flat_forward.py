"""Cross-validate :class:`pquantlib.termstructures.yield_.FlatForward` against C++.

Probe source: migration-harness/cpp/probes/cluster_l2b/probe.cpp
Reference:    migration-harness/references/cluster/l2b.json (key: "flat_forward")
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing import reference_reader, tolerance
from pquantlib.time.compounding import Compounding
from pquantlib.time.date import Date
from pquantlib.time.frequency import Frequency
from pquantlib.time.month import Month


@pytest.fixture(scope="module")
def cpp_ref() -> dict[str, Any]:
    return reference_reader.load("cluster/l2b")["flat_forward"]


@pytest.fixture(scope="module")
def ref_date() -> Date:
    return Date.from_ymd(15, Month.June, 2026)


def test_continuous_actual360_5pct_discount_t0(cpp_ref: dict[str, Any], ref_date: Date) -> None:
    ff = FlatForward.from_rate(ref_date, 0.05, Actual360(), Compounding.Continuous, Frequency.Annual)
    # discount(t=0) — tight: closed-form exp(0)=1.0 is bit-exact.
    tolerance.exact(ff.discount(0.0), cpp_ref["cont_a360_05"]["discount_t0"])


def test_continuous_actual360_5pct_discounts(cpp_ref: dict[str, Any], ref_date: Date) -> None:
    ff = FlatForward.from_rate(ref_date, 0.05, Actual360(), Compounding.Continuous, Frequency.Annual)
    # TIGHT: closed-form exp() applied to identical inputs in both C++ and Python.
    # libm exp() may differ across platforms by 1 ulp; TIGHT is safe.
    tolerance.tight(ff.discount(1.0), cpp_ref["cont_a360_05"]["discount_t1"])
    tolerance.tight(ff.discount(2.0), cpp_ref["cont_a360_05"]["discount_t2"])
    tolerance.tight(ff.discount(0.5), cpp_ref["cont_a360_05"]["discount_t05"])


def test_continuous_actual360_5pct_zero_rate(cpp_ref: dict[str, Any], ref_date: Date) -> None:
    ff = FlatForward.from_rate(ref_date, 0.05, Actual360(), Compounding.Continuous, Frequency.Annual)
    rate = ff.zero_rate(1.0, Compounding.Continuous, Frequency.Annual).rate()
    tolerance.tight(rate, cpp_ref["cont_a360_05"]["zero_rate_t1"])


def test_continuous_actual360_5pct_forward_rate(cpp_ref: dict[str, Any], ref_date: Date) -> None:
    ff = FlatForward.from_rate(ref_date, 0.05, Actual360(), Compounding.Continuous, Frequency.Annual)
    rate = ff.forward_rate(1.0, 2.0, Compounding.Continuous, Frequency.Annual).rate()
    tolerance.tight(rate, cpp_ref["cont_a360_05"]["fwd_rate_t1_t2"])


def test_semiannual_compounded(cpp_ref: dict[str, Any], ref_date: Date) -> None:
    ff = FlatForward.from_rate(
        ref_date, 0.05, Actual360(), Compounding.Compounded, Frequency.Semiannual
    )
    tolerance.tight(ff.discount(1.0), cpp_ref["semi_a360_05"]["discount_t1"])
    tolerance.tight(ff.discount(2.0), cpp_ref["semi_a360_05"]["discount_t2"])
    rate = ff.zero_rate(1.0, Compounding.Compounded, Frequency.Semiannual).rate()
    tolerance.tight(rate, cpp_ref["semi_a360_05"]["zero_rate_t1"])


def test_quote_based_continuous_actual365fixed(cpp_ref: dict[str, Any], ref_date: Date) -> None:
    q = SimpleQuote(0.03)
    ff = FlatForward(ref_date, q, Actual365Fixed(), Compounding.Continuous, Frequency.Annual)
    tolerance.tight(ff.discount(1.0), cpp_ref["quote_a365_03"]["discount_t1"])
    tolerance.tight(ff.discount(3.0), cpp_ref["quote_a365_03"]["discount_t3"])


def test_simple_compounding(cpp_ref: dict[str, Any], ref_date: Date) -> None:
    ff = FlatForward.from_rate(ref_date, 0.05, Actual360(), Compounding.Simple, Frequency.Annual)
    tolerance.tight(ff.discount(1.0), cpp_ref["simple_a360_05"]["discount_t1"])
    tolerance.tight(ff.discount(0.5), cpp_ref["simple_a360_05"]["discount_t05"])


def test_date_overload(cpp_ref: dict[str, Any], ref_date: Date) -> None:
    q = SimpleQuote(0.03)
    ff = FlatForward(ref_date, q, Actual365Fixed(), Compounding.Continuous, Frequency.Annual)
    # ref_date + 365 days at Actual365Fixed = t = 1.0; same value as discount(1.0).
    d = ref_date + 365
    tolerance.tight(ff.discount(d), cpp_ref["date_discount"]["d365_actual365"])


def test_quote_observation_invalidates(ref_date: Date) -> None:
    """Quote change triggers LazyObject cache invalidation."""
    q = SimpleQuote(0.03)
    ff = FlatForward(ref_date, q, Actual365Fixed(), Compounding.Continuous, Frequency.Annual)
    d1 = ff.discount(1.0)  # cached
    q.set_value(0.05)  # observer.update propagates → cache invalidated
    d2 = ff.discount(1.0)
    assert d1 != d2  # different rate → different discount
