"""Tests for pquantlib.cashflows.interest_rate (and Compounding enum)."""

from __future__ import annotations

import math

import pytest

from pquantlib.cashflows.compounding import Compounding
from pquantlib.cashflows.interest_rate import InterestRate
from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exceptions import LibraryException
from pquantlib.testing import reference_reader, tolerance
from pquantlib.time.frequency import Frequency

# --- module-scoped reference cache -----------------------------------------


@pytest.fixture(scope="module")
def ref() -> dict[str, float]:
    blob = reference_reader.load("cluster/l2d")
    return blob["interest_rate"]


# --- Compounding -----------------------------------------------------------


def test_compounding_values_match_cpp() -> None:
    # C++ parity: ql/compounding.hpp:32-37 — explicit ordering Simple=0...
    assert int(Compounding.Simple) == 0
    assert int(Compounding.Compounded) == 1
    assert int(Compounding.Continuous) == 2
    assert int(Compounding.SimpleThenCompounded) == 3
    assert int(Compounding.CompoundedThenSimple) == 4


# --- InterestRate.compound_factor ------------------------------------------


def test_compound_factor_simple(ref: dict[str, float]) -> None:
    ir = InterestRate(ref["r"], Actual365Fixed(), Compounding.Simple, Frequency.Annual)
    tolerance.tight(ir.compound_factor(ref["t"]), ref["cf_simple"])


def test_compound_factor_compounded_annual(ref: dict[str, float]) -> None:
    ir = InterestRate(ref["r"], Actual365Fixed(), Compounding.Compounded, Frequency.Annual)
    tolerance.tight(ir.compound_factor(ref["t"]), ref["cf_annual"])


def test_compound_factor_compounded_semi(ref: dict[str, float]) -> None:
    ir = InterestRate(ref["r"], Actual365Fixed(), Compounding.Compounded, Frequency.Semiannual)
    tolerance.tight(ir.compound_factor(ref["t"]), ref["cf_semi"])


def test_compound_factor_continuous(ref: dict[str, float]) -> None:
    ir = InterestRate(ref["r"], Actual365Fixed(), Compounding.Continuous, Frequency.Annual)
    tolerance.tight(ir.compound_factor(ref["t"]), ref["cf_continuous"])


# --- InterestRate.discount_factor ------------------------------------------


def test_discount_factor_simple(ref: dict[str, float]) -> None:
    ir = InterestRate(ref["r"], Actual365Fixed(), Compounding.Simple, Frequency.Annual)
    tolerance.tight(ir.discount_factor(ref["t"]), ref["df_simple"])


def test_discount_factor_continuous(ref: dict[str, float]) -> None:
    ir = InterestRate(ref["r"], Actual365Fixed(), Compounding.Continuous, Frequency.Annual)
    tolerance.tight(ir.discount_factor(ref["t"]), ref["df_continuous"])


# --- InterestRate.equivalent_rate roundtrip -------------------------------


def test_equivalent_rate_roundtrip(ref: dict[str, float]) -> None:
    ir = InterestRate(ref["r"], Actual365Fixed(), Compounding.Compounded, Frequency.Annual)
    ir_semi = ir.equivalent_rate(Compounding.Compounded, Frequency.Semiannual, ref["t"])
    tolerance.tight(ir_semi.rate, ref["eq_annual_to_semi"])
    ir_back = ir_semi.equivalent_rate(Compounding.Compounded, Frequency.Annual, ref["t"])
    tolerance.tight(ir_back.rate, ref["eq_back_to_annual"])


# --- InterestRate.implied_rate ---------------------------------------------


def test_implied_rate_recovers_5pct(ref: dict[str, float]) -> None:
    ir = InterestRate.implied_rate(
        1.05, Actual365Fixed(), Compounding.Compounded, Frequency.Annual, 1.0
    )
    tolerance.tight(ir.rate, ref["implied_5pct"])


# --- guard rails -----------------------------------------------------------


def test_compounded_requires_real_frequency() -> None:
    # C++ parity: ql/interestrate.cpp:38-40 — QL_REQUIRE freq!=Once && freq!=NoFrequency
    with pytest.raises(LibraryException, match="frequency not allowed"):
        InterestRate(0.05, Actual365Fixed(), Compounding.Compounded, Frequency.NoFrequency)
    with pytest.raises(LibraryException, match="frequency not allowed"):
        InterestRate(0.05, Actual365Fixed(), Compounding.Compounded, Frequency.Once)


def test_negative_time_rejected() -> None:
    ir = InterestRate(0.05, Actual365Fixed(), Compounding.Simple, Frequency.Annual)
    with pytest.raises(LibraryException, match="negative time"):
        ir.compound_factor(-1.0)


def test_implied_rate_requires_positive_compound() -> None:
    with pytest.raises(LibraryException, match="positive compound factor"):
        InterestRate.implied_rate(0.0, Actual365Fixed(), Compounding.Simple, Frequency.Annual, 1.0)


def test_frozen_dataclass_cannot_mutate() -> None:
    ir = InterestRate(0.05, Actual365Fixed(), Compounding.Simple, Frequency.Annual)
    with pytest.raises(AttributeError):
        ir.rate = 0.06  # type: ignore[misc]


def test_freq_makes_sense() -> None:
    simple = InterestRate(0.05, Actual365Fixed(), Compounding.Simple, Frequency.Annual)
    cont = InterestRate(0.05, Actual365Fixed(), Compounding.Continuous, Frequency.Annual)
    comp = InterestRate(0.05, Actual365Fixed(), Compounding.Compounded, Frequency.Annual)
    assert not simple.freq_makes_sense
    assert not cont.freq_makes_sense
    assert comp.freq_makes_sense


# --- algebraic sanity ------------------------------------------------------


def test_simple_and_continuous_recover_known_values() -> None:
    # 1+0.05*1 == 1.05
    ir_s = InterestRate(0.05, Actual365Fixed(), Compounding.Simple, Frequency.Annual)
    tolerance.exact(ir_s.compound_factor(1.0), 1.05)

    # exp(0.05) == math.exp(0.05) — derived from same primitive
    ir_c = InterestRate(0.05, Actual365Fixed(), Compounding.Continuous, Frequency.Annual)
    tolerance.exact(ir_c.compound_factor(1.0), math.exp(0.05))
