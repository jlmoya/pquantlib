"""Cross-validate ``ForwardSpreadedTermStructure`` vs C++.

Probe key: "forward_spreaded".
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.termstructures.yield_.forward_spreaded_term_structure import (
    ForwardSpreadedTermStructure,
)
from pquantlib.testing import reference_reader, tolerance
from pquantlib.time.compounding import Compounding
from pquantlib.time.date import Date
from pquantlib.time.frequency import Frequency
from pquantlib.time.month import Month


@pytest.fixture(scope="module")
def cpp_ref() -> dict[str, Any]:
    return reference_reader.load("cluster/l2b")["forward_spreaded"]


@pytest.fixture(scope="module")
def ref_date() -> Date:
    return Date.from_ymd(15, Month.June, 2026)


def test_base_plus_constant_spread(cpp_ref: dict[str, Any], ref_date: Date) -> None:
    base = FlatForward.from_rate(
        ref_date, 0.05, Actual360(), Compounding.Continuous, Frequency.Annual
    )
    spread = SimpleQuote(0.01)
    fst = ForwardSpreadedTermStructure(base, spread)

    rate_t1 = fst.zero_rate(1.0, Compounding.Continuous, Frequency.Annual).rate()
    tolerance.tight(rate_t1, cpp_ref["zero_rate_t1"])
    rate_t2 = fst.zero_rate(2.0, Compounding.Continuous, Frequency.Annual).rate()
    tolerance.tight(rate_t2, cpp_ref["zero_rate_t2"])
    tolerance.tight(fst.discount(1.0), cpp_ref["discount_t1"])
    tolerance.tight(fst.discount(2.0), cpp_ref["discount_t2"])


def test_spread_bump_reflected(cpp_ref: dict[str, Any], ref_date: Date) -> None:
    base = FlatForward.from_rate(
        ref_date, 0.05, Actual360(), Compounding.Continuous, Frequency.Annual
    )
    spread = SimpleQuote(0.01)
    fst = ForwardSpreadedTermStructure(base, spread)
    _ = fst.discount(1.0)  # warm any caches
    spread.set_value(0.02)
    rate = fst.zero_rate(1.0, Compounding.Continuous, Frequency.Annual).rate()
    tolerance.tight(rate, cpp_ref["zero_rate_t1_bumped"])
    tolerance.tight(fst.discount(1.0), cpp_ref["discount_t1_bumped"])
