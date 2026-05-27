"""Unit tests for :class:`pquantlib.interest_rate.InterestRate`.

# C++ parity: ql/interestrate.cpp (v1.42.1)

Compounding/discount formulas are verified analytically (no probe JSON
needed — closed-form double-precision math is bit-exact under TIGHT
tolerance for the simple cases we test). The round-trip ``implied_rate``
→ ``rate`` is exercised via the FlatForward probe values.
"""

from __future__ import annotations

import math

import pytest

from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exceptions import LibraryException
from pquantlib.interest_rate import InterestRate
from pquantlib.testing import tolerance
from pquantlib.time.compounding import Compounding
from pquantlib.time.frequency import Frequency


def test_continuous_compound_factor() -> None:
    # exp(0.05 * 1.0) = 1.0512710963760241
    ir = InterestRate(0.05, Actual360(), Compounding.Continuous, Frequency.Annual)
    tolerance.tight(ir.compound_factor(1.0), math.exp(0.05))
    tolerance.tight(ir.discount_factor(1.0), math.exp(-0.05))


def test_simple_compound_factor() -> None:
    # 1 + 0.05 * 0.5 = 1.025
    ir = InterestRate(0.05, Actual360(), Compounding.Simple, Frequency.NoFrequency)
    tolerance.tight(ir.compound_factor(0.5), 1.025)
    tolerance.tight(ir.discount_factor(0.5), 1.0 / 1.025)


def test_compounded_factor_semiannual() -> None:
    # (1 + 0.05/2)^(2*1) = 1.050625
    ir = InterestRate(0.05, Actual360(), Compounding.Compounded, Frequency.Semiannual)
    tolerance.tight(ir.compound_factor(1.0), 1.025 * 1.025)
    tolerance.tight(ir.discount_factor(1.0), 1.0 / (1.025 * 1.025))


def test_compounded_requires_frequency() -> None:
    with pytest.raises(LibraryException):  # LibraryException is RuntimeError subclass
        InterestRate(0.05, Actual360(), Compounding.Compounded, Frequency.NoFrequency)
    with pytest.raises(LibraryException):
        InterestRate(0.05, Actual360(), Compounding.Compounded, Frequency.Once)


def test_implied_rate_continuous_roundtrip() -> None:
    # compound = exp(0.05*2) → implied rate = 0.05
    compound = math.exp(0.10)
    ir = InterestRate.implied_rate(
        compound, Actual360(), Compounding.Continuous, Frequency.NoFrequency, 2.0
    )
    tolerance.tight(ir.rate(), 0.05)


def test_implied_rate_compounded_roundtrip() -> None:
    # (1 + 0.05/2)^(2*1) → implied = 0.05
    compound = (1.025) ** 2
    ir = InterestRate.implied_rate(
        compound, Actual360(), Compounding.Compounded, Frequency.Semiannual, 1.0
    )
    tolerance.tight(ir.rate(), 0.05)


def test_equivalent_rate_continuous_to_compounded() -> None:
    # Continuous 0.05 over t=1.0 → compound factor exp(0.05)
    # Equivalent compounded-semiannual: r = 2*(exp(0.05/2) - 1) ≈ 0.05063
    ir_cont = InterestRate(0.05, Actual360(), Compounding.Continuous, Frequency.NoFrequency)
    ir_eq = ir_cont.equivalent_rate(Compounding.Compounded, Frequency.Semiannual, 1.0)
    expected_r = 2.0 * (math.exp(0.05 / 2.0) - 1.0)
    tolerance.tight(ir_eq.rate(), expected_r)
    assert ir_eq.compounding() == Compounding.Compounded
    assert ir_eq.frequency() == Frequency.Semiannual


def test_frequency_inspector_returns_no_frequency_for_simple() -> None:
    ir = InterestRate(0.05, Actual360(), Compounding.Simple, Frequency.Annual)
    # Simple compounding doesn't care about frequency; inspector returns NoFrequency.
    assert ir.frequency() == Frequency.NoFrequency


def test_null_rate() -> None:
    null_ir = InterestRate.null()
    assert null_ir.is_null()
    with pytest.raises(LibraryException):
        null_ir.compound_factor(1.0)


def test_negative_time_disallowed() -> None:
    ir = InterestRate(0.05, Actual360(), Compounding.Continuous, Frequency.NoFrequency)
    with pytest.raises(LibraryException):
        ir.compound_factor(-1.0)


def test_implied_rate_compound_eq_1_requires_nonneg_time() -> None:
    # compound = 1.0 → rate is 0.0 (C++ behavior); requires t >= 0
    ir = InterestRate.implied_rate(
        1.0, Actual360(), Compounding.Continuous, Frequency.NoFrequency, 1.0
    )
    tolerance.exact(ir.rate(), 0.0)
    with pytest.raises(LibraryException):
        InterestRate.implied_rate(
            1.0, Actual360(), Compounding.Continuous, Frequency.NoFrequency, -1.0
        )


def test_implied_rate_compound_lt_1_negative_rate() -> None:
    # compound < 1 with Continuous → rate = log(compound)/t (negative)
    ir = InterestRate.implied_rate(
        0.95, Actual365Fixed(), Compounding.Continuous, Frequency.NoFrequency, 1.0
    )
    tolerance.tight(ir.rate(), math.log(0.95))
