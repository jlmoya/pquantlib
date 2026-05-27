"""Cross-validate ``ZeroSpreadedTermStructure`` vs C++.

Probe key: "zero_spreaded".
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.termstructures.yield_.zero_spreaded_term_structure import (
    ZeroSpreadedTermStructure,
)
from pquantlib.testing import reference_reader, tolerance
from pquantlib.time.compounding import Compounding
from pquantlib.time.date import Date
from pquantlib.time.frequency import Frequency
from pquantlib.time.month import Month


@pytest.fixture(scope="module")
def cpp_ref() -> dict[str, Any]:
    return reference_reader.load("cluster/l2b")["zero_spreaded"]


@pytest.fixture(scope="module")
def ref_date() -> Date:
    return Date.from_ymd(15, Month.June, 2026)


def test_continuous_default(cpp_ref: dict[str, Any], ref_date: Date) -> None:
    # Default ctor uses (Continuous, NoFrequency) — see C++ header.
    base = FlatForward.from_rate(
        ref_date, 0.05, Actual360(), Compounding.Continuous, Frequency.Annual
    )
    spread = SimpleQuote(0.01)
    zst = ZeroSpreadedTermStructure(base, spread)
    rate_t1 = zst.zero_rate(1.0, Compounding.Continuous, Frequency.Annual).rate()
    tolerance.tight(rate_t1, cpp_ref["continuous_zero_rate_t1"])
    tolerance.tight(zst.discount(1.0), cpp_ref["continuous_discount_t1"])
    tolerance.tight(zst.discount(2.0), cpp_ref["continuous_discount_t2"])


def test_compounded_semiannual(cpp_ref: dict[str, Any], ref_date: Date) -> None:
    base = FlatForward.from_rate(
        ref_date, 0.05, Actual360(), Compounding.Continuous, Frequency.Annual
    )
    spread = SimpleQuote(0.01)
    zst = ZeroSpreadedTermStructure(base, spread, Compounding.Compounded, Frequency.Semiannual)
    rate_t1 = zst.zero_rate(1.0, Compounding.Continuous, Frequency.Annual).rate()
    tolerance.tight(rate_t1, cpp_ref["semi_zero_rate_t1"])
    tolerance.tight(zst.discount(1.0), cpp_ref["semi_discount_t1"])
