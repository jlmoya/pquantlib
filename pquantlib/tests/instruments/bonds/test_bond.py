"""Tests for the abstract Bond base class.

Covers the protected ``_calculate_notionals_from_cashflows`` and
``_add_redemptions_to_cashflows`` paths via a minimal subclass, plus
public inspectors (notional, settlement_date, accrued_amount).
"""

from __future__ import annotations

import pytest

from pquantlib.cashflows.cash_flow import CashFlow
from pquantlib.cashflows.fixed_rate_coupon import FixedRateCoupon
from pquantlib.cashflows.simple_cash_flow import AmortizingPayment, Redemption
from pquantlib.daycounters.thirty_360 import Convention, Thirty360
from pquantlib.exceptions import LibraryException
from pquantlib.instruments.bond import (
    Bond,
    BondArguments,
    BondPrice,
    BondPriceType,
    BondResults,
)
from pquantlib.interest_rate import InterestRate
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.testing import tolerance
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.compounding import Compounding
from pquantlib.time.date import Date
from pquantlib.time.frequency import Frequency
from pquantlib.time.month import Month


def _dc() -> Thirty360:
    return Thirty360(Convention.BondBasis)


def _ir(rate: float) -> InterestRate:
    return InterestRate(rate, _dc(), Compounding.Simple, Frequency.Annual)


def _coupons_3y(face: float) -> list[CashFlow]:
    """Build 3 annual coupons starting Jan 15, 2025."""
    dates = [
        Date.from_ymd(15, Month.January, 2025),
        Date.from_ymd(15, Month.January, 2026),
        Date.from_ymd(15, Month.January, 2027),
        Date.from_ymd(15, Month.January, 2028),
    ]
    leg: list[CashFlow] = []
    for i in range(3):
        leg.append(
            FixedRateCoupon(dates[i + 1], face, _ir(0.05), dates[i], dates[i + 1])
        )
    return leg


# --- BondPrice + BondPriceType --------------------------------------------


def test_bond_price_default_is_clean_and_invalid() -> None:
    p = BondPrice()
    assert p.type() == BondPriceType.Clean
    assert p.is_valid() is False


def test_bond_price_amount_requires_value() -> None:
    with pytest.raises(LibraryException, match="no amount given"):
        BondPrice().amount()


def test_bond_price_full_constructor() -> None:
    p = BondPrice(99.5, BondPriceType.Dirty)
    assert p.amount() == 99.5
    assert p.type() == BondPriceType.Dirty
    assert p.is_valid() is True


# --- BondArguments + BondResults ------------------------------------------


def test_bond_arguments_validate_requires_settlement_and_cashflows() -> None:
    args = BondArguments()
    with pytest.raises(LibraryException, match="no settlement date"):
        args.validate()
    args.settlement_date = Date.from_ymd(15, Month.January, 2025)
    with pytest.raises(LibraryException, match="no cash flow provided"):
        args.validate()
    coupons = _coupons_3y(100.0)
    args.cashflows = coupons
    args.validate()  # no raise


def test_bond_results_reset_clears_fields() -> None:
    r = BondResults()
    r.value = 100.0
    r.settlement_value = 99.5
    r.reset()
    assert r.value is None
    assert r.settlement_value is None


# --- Bond subclassing via the modern constructor --------------------------


class _NeverExpired(Bond):
    """Concrete Bond shell used for testing the base-class logic."""

    def is_expired(self) -> bool:
        return False


def test_modern_ctor_builds_notional_schedule_and_redemption() -> None:
    """100 face, 3y constant notional → 1 redemption of 100, 4 cashflows."""
    coupons = _coupons_3y(100.0)
    bond = _NeverExpired(2, TARGET(), Date.from_ymd(1, Month.January, 2025), coupons)
    # 3 coupons + 1 redemption
    assert len(bond.cashflows()) == 4
    assert len(bond.redemptions()) == 1
    assert isinstance(bond.redemption(), Redemption)
    # Single-redemption bond: notional schedule = [Null, maturity]; notionals = [100, 0]
    assert bond.notionals() == [100.0, 0.0]


def test_modern_ctor_rejects_invalid_issue_date() -> None:
    """Issue date must be strictly earlier than the first cashflow."""
    coupons = _coupons_3y(100.0)
    with pytest.raises(LibraryException, match="issue date"):
        _NeverExpired(
            2,
            TARGET(),
            # issue after the first coupon — invalid
            Date.from_ymd(15, Month.June, 2026),
            coupons,
        )


def test_legacy_face_amount_ctor() -> None:
    """``with_face_amount`` factory: last cashflow is the redemption."""
    coupons = _coupons_3y(100.0)
    redemption_cf = Redemption(100.0, Date.from_ymd(15, Month.January, 2028))
    cashflows = [*coupons, redemption_cf]
    bond = _NeverExpired.with_face_amount(
        2,
        TARGET(),
        100.0,
        Date.from_ymd(15, Month.January, 2028),
        Date.from_ymd(1, Month.January, 2025),
        cashflows,
    )
    assert len(bond.cashflows()) == 4
    assert bond.redemption() is redemption_cf


# --- settlement_date logic -----------------------------------------------


def test_settlement_date_max_of_advance_and_issue() -> None:
    """Settlement = max(eval+2BD, issue_date)."""
    settings = ObservableSettings()
    saved = settings.evaluation_date
    try:
        settings.evaluation_date = Date.from_ymd(1, Month.January, 2025)
        coupons = _coupons_3y(100.0)
        bond = _NeverExpired(
            2, TARGET(), Date.from_ymd(1, Month.January, 2025), coupons
        )
        # eval = Jan 1 (Wed), +2BD = Jan 3 (Fri) on TARGET
        # issue = Jan 1, 2025 — both are equal in this test setup, so
        # settle = max(Jan 3, Jan 1) = Jan 3.
        settle = bond.settlement_date()
        assert settle == Date.from_ymd(3, Month.January, 2025)
    finally:
        settings.evaluation_date = saved


def test_notional_zero_after_maturity() -> None:
    coupons = _coupons_3y(100.0)
    bond = _NeverExpired(2, TARGET(), Date.from_ymd(1, Month.January, 2025), coupons)
    # Maturity is Jan 15, 2028; one day after = 0.
    assert bond.notional(Date.from_ymd(16, Month.January, 2028)) == 0.0


# --- _add_redemptions amortising path -------------------------------------


def test_amortising_redemptions_created_at_step_downs() -> None:
    """Decreasing notionals → AmortizingPayments at intermediate steps."""
    # Build 3 coupons with notionals 100, 75, 50 — each AmortizingPayment
    # is 25, final Redemption is 50 (for full step-down).
    dates = [
        Date.from_ymd(15, Month.January, 2025),
        Date.from_ymd(15, Month.January, 2026),
        Date.from_ymd(15, Month.January, 2027),
        Date.from_ymd(15, Month.January, 2028),
    ]
    leg: list[CashFlow] = []
    for i, nom in enumerate([100.0, 75.0, 50.0]):
        leg.append(
            FixedRateCoupon(dates[i + 1], nom, _ir(0.05), dates[i], dates[i + 1])
        )

    bond = _NeverExpired(2, TARGET(), Date.from_ymd(1, Month.January, 2025), leg)
    # 3 coupons + 2 AmortizingPayments + 1 Redemption = 6
    assert len(bond.cashflows()) == 6
    amortising = [cf for cf in bond.cashflows() if isinstance(cf, AmortizingPayment) and not isinstance(cf, Redemption)]
    redemptions = [cf for cf in bond.cashflows() if isinstance(cf, Redemption)]
    assert len(amortising) == 2
    assert len(redemptions) == 1
    # The starting face was 100, ends at 50, so the last redemption is 50
    # (and the two intermediate AmortizingPayments are 25 each).
    tolerance.tight(redemptions[0].amount(), 50.0)
    for cf in amortising:
        tolerance.tight(cf.amount(), 25.0)


# --- accrued_amount mid-period -------------------------------------------


def test_accrued_amount_mid_first_period() -> None:
    """Accrued at the mid-point of the first coupon."""
    coupons = _coupons_3y(100.0)
    bond = _NeverExpired(2, TARGET(), Date.from_ymd(1, Month.January, 2025), coupons)
    # 6 months into period 0: 30/360 → 0.5 year.
    # 5% rate, 100 nominal → accrued = 100 * 0.05 * 0.5 = 2.5
    # accrued_amount returns per-100 quote, so still 2.5.
    mid = Date.from_ymd(15, Month.July, 2025)
    accrued = bond.accrued_amount(mid)
    # Thirty360 BondBasis: 6 months → 0.5
    tolerance.tight(accrued, 2.5)
