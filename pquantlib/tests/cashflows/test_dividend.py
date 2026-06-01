"""Tests for the Dividend cash-flow family.

Probe source: migration-harness/cpp/probes/cluster_w12c/probe.cpp
Reference:    migration-harness/references/cluster/w12c.json

Covers FixedDividend / FractionalDividend amount() + amount(underlying) and
the dividend_vector helper. Cross-validated against C++ v1.42.1 (099987f0).
"""

from __future__ import annotations

import pytest

from pquantlib.cashflows.cash_flow import CashFlow
from pquantlib.cashflows.dividend import (
    Dividend,
    FixedDividend,
    FractionalDividend,
    dividend_vector,
)
from pquantlib.exceptions import LibraryException
from pquantlib.testing import reference_reader, tolerance
from pquantlib.time.date import Date
from pquantlib.time.month import Month

_DATE = Date.from_ymd(15, Month.June, 2024)


@pytest.fixture(scope="module")
def ref() -> dict[str, float]:
    return reference_reader.load("cluster/w12c")


# --- FixedDividend ---------------------------------------------------------


def test_fixed_dividend_amount(ref: dict[str, float]) -> None:
    d = FixedDividend(2.5, _DATE)
    tolerance.exact(d.amount(), ref["fixed_amount"])


def test_fixed_dividend_amount_ignores_underlying(ref: dict[str, float]) -> None:
    d = FixedDividend(2.5, _DATE)
    tolerance.exact(d.amount_with_underlying(100.0), ref["fixed_amount_with_underlying"])


def test_fixed_dividend_date() -> None:
    d = FixedDividend(2.5, _DATE)
    assert d.date() == _DATE


def test_fixed_dividend_is_cashflow() -> None:
    d = FixedDividend(2.5, _DATE)
    assert isinstance(d, Dividend)
    assert isinstance(d, CashFlow)


# --- FractionalDividend ----------------------------------------------------


def test_fractional_dividend_inspectors(ref: dict[str, float]) -> None:
    d = FractionalDividend(0.03, _DATE, nominal=200.0)
    tolerance.tight(d.rate(), ref["frac_rate"])
    assert d.nominal() == ref["frac_nominal"]


def test_fractional_dividend_amount_uses_nominal(ref: dict[str, float]) -> None:
    d = FractionalDividend(0.03, _DATE, nominal=200.0)
    tolerance.tight(d.amount(), ref["frac_amount"])  # rate * nominal == 6.0


def test_fractional_dividend_amount_with_underlying(ref: dict[str, float]) -> None:
    d = FractionalDividend(0.03, _DATE, nominal=200.0)
    tolerance.tight(d.amount_with_underlying(150.0), ref["frac_amount_underlying"])  # 4.5


def test_fractional_dividend_no_nominal_raises() -> None:
    d = FractionalDividend(0.03, _DATE)
    assert d.nominal() is None
    with pytest.raises(LibraryException, match="no nominal given"):
        d.amount()
    # amount_with_underlying still works without a nominal
    tolerance.tight(d.amount_with_underlying(150.0), 4.5)


# --- dividend_vector helper ------------------------------------------------


def test_dividend_vector(ref: dict[str, float]) -> None:
    dates = [Date.from_ymd(15, Month.March, 2024), Date.from_ymd(15, Month.June, 2024)]
    amounts = [1.0, 2.0]
    vec = dividend_vector(dates, amounts)
    assert len(vec) == int(ref["divvec_size"])
    tolerance.exact(vec[0].amount(), ref["divvec_0_amount"])
    tolerance.exact(vec[1].amount(), ref["divvec_1_amount"])
    assert all(isinstance(x, FixedDividend) for x in vec)
    assert vec[0].date() == dates[0]


def test_dividend_vector_size_mismatch() -> None:
    with pytest.raises(LibraryException, match="size mismatch"):
        dividend_vector([_DATE], [1.0, 2.0])
