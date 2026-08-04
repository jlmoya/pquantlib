"""Cross-validate the constant-notional cross-currency swap family (C++ v1.43).

Probe: ``v143/xccy/swaps``. Covers ``ConstNotionalCrossCurrencySwap``,
``ConstNotionalCrossCurrencyBasisSwap``,
``ConstNotionalCrossCurrencyFixedVsFloatingSwap`` and
``DiscountingConstNotionalCrossCurrencySwapEngine``.

Every input is built explicitly from literal discount-factor tables and inline
index definitions, matching the probe exactly, so a failure localises to the
swap or the engine rather than to a curve or index definition shared with other
tests.

Besides the NPV the test pins each leg's NPV, BPS, in-currency NPV/BPS and the
three discount factors, plus the full cashflow listing. The listing is the part
that actually catches leg-construction bugs: an NPV alone can match while two
errors cancel, and notional-exchange placement is easy to get subtly wrong.

Tolerance: LOOSE tier (1e-8 relative, 1e-8 absolute floor). These are 5Y swaps
on 125M notionals discounted through interpolated curves, so the absolute
figures are ~1e7 and only a relative comparison is meaningful.
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.cashflows.coupon import Coupon
from pquantlib.cashflows.fixed_rate_leg import fixed_rate_leg
from pquantlib.cashflows.rate_averaging import RateAveraging
from pquantlib.cashflows.simple_cash_flow import SimpleCashFlow
from pquantlib.currencies.america import USDCurrency
from pquantlib.currencies.europe import EURCurrency
from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.indexes.ibor_index import IborIndex
from pquantlib.indexes.overnight_index import OvernightIndex
from pquantlib.instruments.const_notional_cross_currency_basis_swap import (
    ConstNotionalCrossCurrencyBasisSwap,
)
from pquantlib.instruments.const_notional_cross_currency_fixed_vs_floating_swap import (
    ConstNotionalCrossCurrencyFixedVsFloatingSwap,
)
from pquantlib.instruments.const_notional_cross_currency_swap import (
    ConstNotionalCrossCurrencySwap,
)
from pquantlib.instruments.swap import Leg, SwapType
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.pricingengines.swap.discounting_const_notional_cross_currency_swap_engine import (
    DiscountingConstNotionalCrossCurrencySwapEngine,
)
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.yield_.interpolated_discount_curve import (
    InterpolatedDiscountCurve,
)
from pquantlib.testing import reference_reader
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.date import Date
from pquantlib.time.date_generation import DateGeneration
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.schedule import Schedule
from pquantlib.time.time_unit import TimeUnit

# LOOSE tier: 1e-8 relative, 1e-8 absolute floor for quantities that legitimately
# pass through zero.
_REL_TOL = 1.0e-8
_ABS_TOL = 1.0e-8

_TODAY = Date.from_ymd(11, Month.September, 2018)
_USD_NOMINAL = 125_000_000.0
_SPOT_FX = 1.22  # USD per EUR


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("v143/xccy/swaps")


@pytest.fixture(autouse=True)
def _pinned_evaluation_date():  # type: ignore[no-untyped-def]
    """The probe runs with ``Settings::evaluationDate() = 2018-09-11``."""
    settings = ObservableSettings()
    saved = settings.evaluation_date
    settings.evaluation_date = _TODAY
    yield
    settings.evaluation_date = saved


def _assert_close(what: str, expected: float, actual: float) -> None:
    tol = max(_ABS_TOL, _REL_TOL * abs(expected))
    assert abs(actual - expected) <= tol, (
        f"{what}: expected {expected!r}, got {actual!r} (tol {tol!r})"
    )


# --- market data — mirrors the probe's tables exactly ------------------------


def _curve_dates() -> list[Date]:
    return [
        Date.from_ymd(11, Month.September, 2018),
        Date.from_ymd(11, Month.December, 2018),
        Date.from_ymd(11, Month.March, 2019),
        Date.from_ymd(11, Month.September, 2019),
        Date.from_ymd(11, Month.September, 2020),
        Date.from_ymd(13, Month.September, 2021),
        Date.from_ymd(12, Month.September, 2022),
        Date.from_ymd(11, Month.September, 2023),
        Date.from_ymd(11, Month.September, 2028),
    ]


def _discount_curve(dfs: list[float]) -> InterpolatedDiscountCurve:
    # C++ DiscountCurve == InterpolatedDiscountCurve<LogLinear>, which is
    # PQuantLib's default interpolator here too.
    return InterpolatedDiscountCurve(_curve_dates(), dfs, Actual365Fixed())


def _usd_discount() -> InterpolatedDiscountCurve:
    return _discount_curve([1.0, 0.9941, 0.9888, 0.9757, 0.9486, 0.9228, 0.8983, 0.8747, 0.7630])


def _eur_discount() -> InterpolatedDiscountCurve:
    return _discount_curve([1.0, 0.9998, 0.9995, 0.9986, 0.9955, 0.9910, 0.9850, 0.9775, 0.9210])


def _usd_projection() -> InterpolatedDiscountCurve:
    return _discount_curve([1.0, 0.9935, 0.9871, 0.9727, 0.9433, 0.9148, 0.8876, 0.8615, 0.7386])


def _eur_projection() -> InterpolatedDiscountCurve:
    return _discount_curve([1.0, 0.9996, 0.9991, 0.9978, 0.9938, 0.9881, 0.9808, 0.9720, 0.9040])


def _usd_ibor_3m() -> IborIndex:
    return IborIndex(
        "USD-XCCY-3M",
        Period(3, TimeUnit.Months),
        2,
        USDCurrency(),
        TARGET(),
        BusinessDayConvention.ModifiedFollowing,
        False,
        Actual360(),
        _usd_projection(),
    )


def _eur_ibor_3m() -> IborIndex:
    return IborIndex(
        "EUR-XCCY-3M",
        Period(3, TimeUnit.Months),
        2,
        EURCurrency(),
        TARGET(),
        BusinessDayConvention.ModifiedFollowing,
        False,
        Actual360(),
        _eur_projection(),
    )


def _usd_on() -> OvernightIndex:
    return OvernightIndex(
        "USD-XCCY-ON", 0, USDCurrency(), TARGET(), Actual360(), _usd_projection()
    )


def _eur_on() -> OvernightIndex:
    return OvernightIndex(
        "EUR-XCCY-ON", 0, EURCurrency(), TARGET(), Actual360(), _eur_projection()
    )


def _schedule(start: Date, end: Date, tenor: Period) -> Schedule:
    return Schedule.from_rule(
        effective_date=start,
        termination_date=end,
        tenor=tenor,
        calendar=TARGET(),
        convention=BusinessDayConvention.ModifiedFollowing,
        termination_date_convention=BusinessDayConvention.ModifiedFollowing,
        rule=DateGeneration.Forward,
        end_of_month=False,
    )


def _quarterly(start: Date, end: Date) -> Schedule:
    return _schedule(start, end, Period(3, TimeUnit.Months))


def _annual(start: Date, end: Date) -> Schedule:
    return _schedule(start, end, Period(1, TimeUnit.Years))


def _start_date() -> Date:
    return TARGET().advance(_TODAY, 2, TimeUnit.Days)


def _end_date() -> Date:
    return TARGET().advance(_TODAY, 5, TimeUnit.Years)


def _engine(
    npv_date: Date | None = None, spot_fx_settle_date: Date | None = None
) -> DiscountingConstNotionalCrossCurrencySwapEngine:
    # _SPOT_FX is quoted as USD per EUR; the engine wants domestic per foreign,
    # and the domestic currency here is USD.
    return DiscountingConstNotionalCrossCurrencySwapEngine(
        USDCurrency(),
        _usd_discount(),
        EURCurrency(),
        _eur_discount(),
        SimpleQuote(_SPOT_FX),
        npv_date=npv_date,
        spot_fx_settle_date=spot_fx_settle_date,
    )


# --- instrument builders — mirror the probe's builders exactly ---------------


def _fixed_leg_with_exchanges(nominal: float, rate: float, sched: Schedule) -> Leg:
    """A fixed leg with the notional exchanges attached by hand.

    Mirrors the probe (and the upstream C++ test) rather than going through
    ``add_notional_exchanges_to_leg``: the initial exchange lands on the
    adjusted first schedule date and the final one on the last coupon's own
    payment date.
    """
    cal = TARGET()
    leg = fixed_rate_leg(
        sched,
        nominals=[nominal],
        rates=[rate],
        day_counter=Actual365Fixed(),
        payment_adjustment=BusinessDayConvention.ModifiedFollowing,
        payment_calendar=cal,
    )
    first = cal.adjust(sched.date(0), BusinessDayConvention.ModifiedFollowing)
    leg.insert(0, SimpleCashFlow(-nominal, first))
    leg.append(SimpleCashFlow(nominal, leg[-1].date()))
    return leg


def _make_fix_fix() -> ConstNotionalCrossCurrencySwap:
    sched = _quarterly(_start_date(), _end_date())
    eur_nominal = _USD_NOMINAL / _SPOT_FX
    usd_leg = _fixed_leg_with_exchanges(_USD_NOMINAL, 0.0575, sched)
    eur_leg = _fixed_leg_with_exchanges(eur_nominal, 0.0201, sched)
    return ConstNotionalCrossCurrencySwap.from_legs_and_currencies(
        usd_leg, USDCurrency(), eur_leg, EURCurrency()
    )


def _make_basis(*, overnight: bool) -> ConstNotionalCrossCurrencyBasisSwap:
    sched = _quarterly(_start_date(), _end_date())
    eur_nominal = _USD_NOMINAL / _SPOT_FX
    if overnight:
        return ConstNotionalCrossCurrencyBasisSwap(
            _USD_NOMINAL,
            USDCurrency(),
            sched,
            _usd_on(),
            0.0010,
            1.0,
            eur_nominal,
            EURCurrency(),
            sched,
            _eur_on(),
            0.0025,
            1.0,
            pay_payment_lag=2,
            rec_payment_lag=2,
        )
    return ConstNotionalCrossCurrencyBasisSwap(
        _USD_NOMINAL,
        USDCurrency(),
        sched,
        _usd_ibor_3m(),
        0.0010,
        1.0,
        eur_nominal,
        EURCurrency(),
        sched,
        _eur_ibor_3m(),
        0.0025,
        1.0,
    )


def _make_fix_float(
    swap_type: SwapType,
) -> ConstNotionalCrossCurrencyFixedVsFloatingSwap:
    cal = TARGET()
    return ConstNotionalCrossCurrencyFixedVsFloatingSwap(
        swap_type,
        _USD_NOMINAL,
        USDCurrency(),
        _annual(_start_date(), _end_date()),
        0.0325,
        Actual365Fixed(),
        BusinessDayConvention.ModifiedFollowing,
        0,
        cal,
        _USD_NOMINAL / _SPOT_FX,
        EURCurrency(),
        _quarterly(_start_date(), _end_date()),
        _eur_ibor_3m(),
        0.0015,
        BusinessDayConvention.ModifiedFollowing,
        0,
        cal,
    )


# --- shared assertions -------------------------------------------------------


def _check_cashflows(tag: str, expected_flows: list[dict[str, Any]], leg: Leg) -> None:
    assert len(leg) == len(expected_flows), f"{tag}: cashflow count"
    for k, (fe, cf) in enumerate(zip(expected_flows, leg, strict=True)):
        ftag = f"{tag} flow {k}"
        assert cf.date().serial_number() == fe["date_serial"], f"{ftag}: date"
        _assert_close(f"{ftag} amount", float(fe["amount"]), cf.amount())

        is_coupon = bool(fe["is_coupon"])
        assert isinstance(cf, Coupon) is is_coupon, f"{ftag}: is_coupon"
        if is_coupon:
            assert isinstance(cf, Coupon)
            _assert_close(f"{ftag} nominal", float(fe["nominal"]), cf.nominal())
            assert (
                cf.accrual_start_date().serial_number() == fe["accrual_start_serial"]
            ), f"{ftag}: accrual start"
            assert cf.accrual_end_date().serial_number() == fe["accrual_end_serial"], (
                f"{ftag}: accrual end"
            )
            _assert_close(
                f"{ftag} accrual period",
                float(fe["accrual_period"]),
                cf.accrual_period(),
            )
            _assert_close(f"{ftag} rate", float(fe["rate"]), cf.rate())


def _check_swap(
    case: str, swap: ConstNotionalCrossCurrencySwap, expected: dict[str, Any]
) -> None:
    _assert_close(f"{case}: NPV", float(expected["npv"]), swap.npv())
    assert swap.valuation_date().serial_number() == expected["valuation_date_serial"]
    assert swap.start_date().serial_number() == expected["start_date_serial"]
    assert swap.maturity_date().serial_number() == expected["maturity_date_serial"]

    legs: list[dict[str, Any]] = expected["legs"]
    for i, le in enumerate(legs):
        tag = f"{case}: leg {i}"
        assert swap.leg_currency(i).code == le["currency"], f"{tag}: currency"
        _assert_close(f"{tag} leg_npv", float(le["leg_npv"]), swap.leg_npv(i))
        _assert_close(f"{tag} leg_bps", float(le["leg_bps"]), swap.leg_bps(i))
        _assert_close(
            f"{tag} in_ccy_leg_npv",
            float(le["in_ccy_leg_npv"]),
            swap.in_ccy_leg_npv(i),
        )
        _assert_close(
            f"{tag} in_ccy_leg_bps",
            float(le["in_ccy_leg_bps"]),
            swap.in_ccy_leg_bps(i),
        )
        _assert_close(
            f"{tag} npv_date_discounts",
            float(le["npv_date_discounts"]),
            swap.npv_date_discounts(i),
        )
        _assert_close(
            f"{tag} start_discounts",
            float(le["start_discounts"]),
            swap.start_discounts(i),
        )
        _assert_close(
            f"{tag} end_discounts", float(le["end_discounts"]), swap.end_discounts(i)
        )
        _check_cashflows(tag, le["cashflows"], swap.leg(i))


# --- tests -------------------------------------------------------------------


def test_fixed_vs_fixed(cpp: dict[str, Any]) -> None:
    swap = _make_fix_fix()
    swap.set_pricing_engine(_engine())
    _check_swap("fix_fix", swap, cpp["fix_fix"])


def test_fixed_vs_fixed_with_forward_fx_settlement(cpp: dict[str, Any]) -> None:
    """spot_fx_settle_date != reference date switches on the forward-FX adjustment."""
    swap = _make_fix_fix()
    swap.set_pricing_engine(
        _engine(spot_fx_settle_date=Date.from_ymd(11, Month.September, 2019))
    )
    _check_swap("fix_fix_fwd_fx_settle", swap, cpp["fix_fix_fwd_fx_settle"])


def test_fixed_vs_fixed_with_forward_npv_date(cpp: dict[str, Any]) -> None:
    """npv_date != reference date rebases the NPV off the reference date."""
    swap = _make_fix_fix()
    swap.set_pricing_engine(_engine(npv_date=Date.from_ymd(11, Month.March, 2019)))
    _check_swap("fix_fix_forward_npv_date", swap, cpp["fix_fix_forward_npv_date"])


def test_basis_swap_with_ibor_legs(cpp: dict[str, Any]) -> None:
    swap = _make_basis(overnight=False)
    swap.set_pricing_engine(_engine())
    expected = cpp["basis_ibor"]
    _check_swap("basis_ibor", swap, expected)
    _assert_close(
        "basis_ibor: fair_pay_spread",
        float(expected["fair_pay_spread"]),
        swap.fair_pay_spread(),
    )
    _assert_close(
        "basis_ibor: fair_rec_spread",
        float(expected["fair_rec_spread"]),
        swap.fair_rec_spread(),
    )


def test_basis_swap_with_overnight_legs(cpp: dict[str, Any]) -> None:
    """Overnight legs, both with a two-business-day payment lag."""
    swap = _make_basis(overnight=True)
    swap.set_pricing_engine(_engine())
    expected = cpp["basis_overnight"]
    _check_swap("basis_overnight", swap, expected)
    _assert_close(
        "basis_overnight: fair_pay_spread",
        float(expected["fair_pay_spread"]),
        swap.fair_pay_spread(),
    )
    _assert_close(
        "basis_overnight: fair_rec_spread",
        float(expected["fair_rec_spread"]),
        swap.fair_rec_spread(),
    )


@pytest.mark.parametrize(
    ("label", "swap_type"),
    [("payer", SwapType.Payer), ("receiver", SwapType.Receiver)],
)
def test_fixed_vs_floating(
    cpp: dict[str, Any], label: str, swap_type: SwapType
) -> None:
    swap = _make_fix_float(swap_type)
    swap.set_pricing_engine(_engine())
    case = f"fixed_vs_floating_{label}"
    expected = cpp[case]
    _check_swap(case, swap, expected)
    _assert_close(f"{case}: fair_rate", float(expected["fair_rate"]), swap.fair_rate())
    _assert_close(
        f"{case}: fair_spread", float(expected["fair_spread"]), swap.fair_spread()
    )


def test_multi_leg_constructor_matches_two_leg_constructor(cpp: dict[str, Any]) -> None:
    """The multi-leg constructor is not reachable through the probe cases above.

    Building the same fixed/fixed swap through it must give an identical price,
    otherwise the payer-flag or currency wiring differs between the two entry
    points.
    """
    reference = _make_fix_fix()
    multi_leg = ConstNotionalCrossCurrencySwap.from_multi_and_currencies(
        [reference.leg(0), reference.leg(1)],
        [True, False],
        [USDCurrency(), EURCurrency()],
    )
    multi_leg.set_pricing_engine(_engine())
    _check_swap("fix_fix", multi_leg, cpp["fix_fix"])


def test_overnight_modifiers_are_rejected_rather_than_ignored() -> None:
    """PQuantLib's OvernightIndexedCoupon supports none of the C++ modifiers.

    They must raise instead of being silently priced as if unset — otherwise
    the swap would answer a different question than the caller asked.
    """
    sched = _quarterly(_start_date(), _end_date())
    eur_nominal = _USD_NOMINAL / _SPOT_FX
    with pytest.raises(NotImplementedError, match="pay_compound_spread=True"):
        ConstNotionalCrossCurrencyBasisSwap(
            _USD_NOMINAL,
            USDCurrency(),
            sched,
            _usd_on(),
            0.0010,
            1.0,
            eur_nominal,
            EURCurrency(),
            sched,
            _eur_on(),
            0.0025,
            1.0,
            pay_compound_spread=True,
        )
    with pytest.raises(NotImplementedError, match="rec_lockout_days=5"):
        ConstNotionalCrossCurrencyBasisSwap(
            _USD_NOMINAL,
            USDCurrency(),
            sched,
            _usd_ibor_3m(),
            0.0010,
            1.0,
            eur_nominal,
            EURCurrency(),
            sched,
            _eur_on(),
            0.0025,
            1.0,
            rec_lockout_days=5,
        )
    with pytest.raises(NotImplementedError, match="float_averaging_method=Simple"):
        ConstNotionalCrossCurrencyFixedVsFloatingSwap(
            SwapType.Payer,
            _USD_NOMINAL,
            USDCurrency(),
            _annual(_start_date(), _end_date()),
            0.0325,
            Actual365Fixed(),
            BusinessDayConvention.ModifiedFollowing,
            0,
            TARGET(),
            eur_nominal,
            EURCurrency(),
            sched,
            _eur_on(),
            0.0015,
            BusinessDayConvention.ModifiedFollowing,
            0,
            TARGET(),
            float_averaging_method=RateAveraging.Simple,
        )


def test_ibor_legs_ignore_the_overnight_modifiers() -> None:
    """The guard only fires on an overnight index — C++ ignores them otherwise."""
    sched = _quarterly(_start_date(), _end_date())
    swap = ConstNotionalCrossCurrencyBasisSwap(
        _USD_NOMINAL,
        USDCurrency(),
        sched,
        _usd_ibor_3m(),
        0.0010,
        1.0,
        _USD_NOMINAL / _SPOT_FX,
        EURCurrency(),
        sched,
        _eur_ibor_3m(),
        0.0025,
        1.0,
        pay_compound_spread=True,
        rec_lockout_days=7,
    )
    swap.set_pricing_engine(_engine())
    assert swap.npv() != 0.0
