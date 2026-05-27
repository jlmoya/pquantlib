"""Cross-validate ``ImpliedTermStructure`` vs C++.

Probe key: "implied".
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.termstructures.yield_.implied_term_structure import ImpliedTermStructure
from pquantlib.testing import reference_reader, tolerance
from pquantlib.time.compounding import Compounding
from pquantlib.time.date import Date
from pquantlib.time.frequency import Frequency
from pquantlib.time.month import Month


@pytest.fixture(scope="module")
def cpp_ref() -> dict[str, Any]:
    return reference_reader.load("cluster/l2b")["implied"]


@pytest.fixture(scope="module")
def ref_date() -> Date:
    return Date.from_ymd(15, Month.June, 2026)


def test_discount_at_future_reference(cpp_ref: dict[str, Any], ref_date: Date) -> None:
    base = FlatForward.from_rate(
        ref_date, 0.05, Actual365Fixed(), Compounding.Continuous, Frequency.Annual
    )
    future_ref = ref_date + 365
    its = ImpliedTermStructure(base, future_ref)

    # discount(t=0) = 1.0 since it's evaluated at our reference date.
    tolerance.exact(its.discount(0.0), cpp_ref["discount_t0"])
    # discount(t=1.0) — flat-fwd: exp(-0.05*1.0) ≈ 0.951229...
    tolerance.tight(its.discount(1.0), cpp_ref["discount_t1"])


def test_reference_date_is_future_ref(ref_date: Date) -> None:
    base = FlatForward.from_rate(
        ref_date, 0.05, Actual365Fixed(), Compounding.Continuous, Frequency.Annual
    )
    future_ref = ref_date + 365
    its = ImpliedTermStructure(base, future_ref)
    assert its.reference_date() == future_ref


def test_max_date_forwards_to_base(ref_date: Date) -> None:
    base = FlatForward.from_rate(
        ref_date, 0.05, Actual365Fixed(), Compounding.Continuous, Frequency.Annual
    )
    future_ref = ref_date + 365
    its = ImpliedTermStructure(base, future_ref)
    assert its.max_date() == base.max_date()
