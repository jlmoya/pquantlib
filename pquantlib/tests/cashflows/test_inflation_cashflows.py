"""Tests for inflation cashflows + coupons (L7-C).

Cross-validates against ``migration-harness/references/cluster/l7c.json``
emitted by ``cluster_l7c_probe``.

Covers:

* ``CPI.lagged_fixing`` (Flat, AsIndex, Linear modes against a ramped EUHICP).
* ``IndexedCashFlow`` amount (bond-style + swap-style).
* ``ZeroInflationCashFlow`` base/index/amount (lag-adjusted CPI lookup).
* ``CPICashFlow`` with explicit base fixing override.
* ``CPICoupon`` ``indexRatio`` + ``amount``.
* ``YoYInflationCoupon`` ``indexFixing`` + ``amount``.
* ``CappedFlooredYoYInflationCoupon`` rate at known cap/floor placements
  (without vol surface — intrinsic-only).
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from pquantlib.cashflows.capflored_inflation_coupon import (
    CappedFlooredCPICoupon,
    CappedFlooredInflationCoupon,
    CappedFlooredYoYInflationCoupon,
)
from pquantlib.cashflows.cpi_coupon import CPICashFlow, CPICoupon
from pquantlib.cashflows.cpi_coupon_pricer import (
    BlackCPICouponPricer,
    CPICouponPricer,
)
from pquantlib.cashflows.indexed_cashflow import IndexedCashFlow
from pquantlib.cashflows.inflation_coupon import InflationCoupon
from pquantlib.cashflows.yoy_inflation_coupon import YoYInflationCoupon
from pquantlib.cashflows.yoy_inflation_coupon_pricer import (
    YoYInflationCouponPricer,
)
from pquantlib.cashflows.zero_inflation_cashflow import ZeroInflationCashFlow
from pquantlib.daycounters.thirty_360 import Convention, Thirty360
from pquantlib.exceptions import LibraryException
from pquantlib.indexes.inflation.cpi import (
    InterpolationType,
    lagged_fixing,
    lagged_yoy_rate,
)
from pquantlib.indexes.inflation.eu_hicp import EUHICP, YoYEUHICP
from pquantlib.testing import reference_reader, tolerance
from pquantlib.time.date import Date
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit

# ---- Reference + helpers ---------------------------------------------------


@pytest.fixture(scope="module")
def reference() -> dict[str, Any]:
    return reference_reader.load("cluster/l7c")


def _ramp_fixing(year: int, month: int) -> float:
    """Match the probe's ``ramp_fixing(y, m)``."""
    return 100.0 + 0.5 * (12 * (year - 2020) + (month - 1))


@pytest.fixture
def seeded_eu_hicp() -> Iterator[EUHICP]:
    """EUHICP with deterministic ramp history 2020-01..2022-12.

    The ``IndexManager`` is global state, so we clear before seeding and
    clear again on teardown to avoid leaking ramp fixings into other
    test modules (notably ``tests/indexes/inflation/test_inflation_index.py``
    which also stores fixings on EUHICP at overlapping dates).
    """
    idx = EUHICP()
    idx.clear_fixings()
    for y in range(2020, 2023):
        for m in range(1, 13):
            idx.add_fixing(Date.from_ymd(1, Month(m), y), _ramp_fixing(y, m), True)
    yield idx
    idx.clear_fixings()


@pytest.fixture
def seeded_yoy_eu_hicp() -> Iterator[YoYEUHICP]:
    """YYEUHICP with a deterministic 2-year history.

    Each (year, month) gets ``0.020 + 0.005 * (m - 1)`` to mirror the probe.
    Teardown clears the IndexManager to avoid cross-test pollution.
    """
    idx = YoYEUHICP(interpolated=False)
    idx.clear_fixings()
    for y in range(2021, 2023):
        for m in range(1, 13):
            idx.add_fixing(
                Date.from_ymd(1, Month(m), y), 0.020 + 0.005 * (m - 1), True
            )
    yield idx
    idx.clear_fixings()


# ---- CPI.lagged_fixing -----------------------------------------------------


def test_lagged_fixing_flat_2021_06_15(
    seeded_eu_hicp: EUHICP, reference: dict[str, Any]
) -> None:
    """Flat lag = period-start fixing of the lag-adjusted date.

    # Justification (TIGHT): closed-form table lookup; numerically equal
    # to the probe's ``ramp_fixing(2021, 3) = 107.0`` modulo IEEE-754.
    """
    expected = reference["lagged_fixing"]["date_2021_06_15"]
    d = Date.from_ymd(15, Month.June, 2021)
    lag = Period(3, TimeUnit.Months)
    result_flat = lagged_fixing(seeded_eu_hicp, d, lag, InterpolationType.Flat)
    result_as = lagged_fixing(seeded_eu_hicp, d, lag, InterpolationType.AsIndex)
    tolerance.tight(result_flat, expected["flat"])
    tolerance.tight(result_as, expected["as_index"])


def test_lagged_fixing_linear_2021_06_15(
    seeded_eu_hicp: EUHICP, reference: dict[str, Any]
) -> None:
    """Linear branch interpolates between bracketing fixings.

    # Justification (TIGHT): closed-form linear blend; only IEEE-754
    # rounding separates Python and the C++ probe.
    """
    expected = reference["lagged_fixing"]["date_2021_06_15"]["linear"]
    d = Date.from_ymd(15, Month.June, 2021)
    lag = Period(3, TimeUnit.Months)
    result = lagged_fixing(seeded_eu_hicp, d, lag, InterpolationType.Linear)
    tolerance.tight(result, expected)


def test_lagged_fixing_flat_2022_08_20(
    seeded_eu_hicp: EUHICP, reference: dict[str, Any]
) -> None:
    expected = reference["lagged_fixing"]["date_2022_08_20"]
    d = Date.from_ymd(20, Month.August, 2022)
    lag = Period(3, TimeUnit.Months)
    result_flat = lagged_fixing(seeded_eu_hicp, d, lag, InterpolationType.Flat)
    result_linear = lagged_fixing(seeded_eu_hicp, d, lag, InterpolationType.Linear)
    tolerance.tight(result_flat, expected["flat"])
    tolerance.tight(result_linear, expected["linear"])


def test_lagged_fixing_linear_fast_path(
    seeded_eu_hicp: EUHICP, reference: dict[str, Any]
) -> None:
    """When the date sits exactly on a period start, no interpolation
    happens — return I0 directly (avoids needing the period-end fixing).

    # Justification (TIGHT): the fast-path returns the stored ramp
    # value bit-equal modulo IEEE-754.
    """
    expected = reference["lagged_fixing"]["date_2020_04_01_lag0_linear_eq_start"]
    d = Date.from_ymd(1, Month.April, 2020)
    lag = Period(0, TimeUnit.Months)
    result = lagged_fixing(seeded_eu_hicp, d, lag, InterpolationType.Linear)
    tolerance.tight(result, expected)


def test_lagged_fixing_invalid_interpolation_raises(seeded_eu_hicp: EUHICP) -> None:
    """Unknown ``InterpolationType`` values raise ``LibraryException``.

    Python's IntEnum rejects unknown ints at construction, so we cast a
    raw int through the type to force the qassert.fail path in
    ``lagged_fixing``.
    """
    d = Date.from_ymd(1, Month.April, 2020)
    bad: InterpolationType = 99  # type: ignore[assignment]
    with pytest.raises(LibraryException):
        lagged_fixing(
            seeded_eu_hicp,
            d,
            Period(0, TimeUnit.Months),
            bad,
        )


def test_lagged_yoy_rate_flat_quoted(
    seeded_yoy_eu_hicp: YoYEUHICP,
) -> None:
    """Flat YoY rate = period-start YoY fixing of the lag-adjusted date.

    # Justification (TIGHT): closed-form table lookup; the probe's
    # YoY coupon at end=2022-06-01, lag=3M, Flat reads YoY(2022-03-01)
    # = 0.020 + 0.005 * 2 = 0.030.
    """
    d = Date.from_ymd(1, Month.June, 2022)
    lag = Period(3, TimeUnit.Months)
    result = lagged_yoy_rate(seeded_yoy_eu_hicp, d, lag, InterpolationType.Flat)
    tolerance.tight(result, 0.030)


def test_lagged_yoy_rate_invalid_interpolation_raises(
    seeded_yoy_eu_hicp: YoYEUHICP,
) -> None:
    """Unknown ``InterpolationType`` raises ``LibraryException`` on YoY too."""
    d = Date.from_ymd(1, Month.June, 2022)
    bad: InterpolationType = 99  # type: ignore[assignment]
    with pytest.raises(LibraryException):
        lagged_yoy_rate(
            seeded_yoy_eu_hicp,
            d,
            Period(3, TimeUnit.Months),
            bad,
        )


# ---- IndexedCashFlow -------------------------------------------------------


def test_indexed_cashflow_amount(
    seeded_eu_hicp: EUHICP, reference: dict[str, Any]
) -> None:
    """``notional * I1/I0`` (bond-style) and ``notional * (I1/I0 - 1)`` (swap-style).

    # Justification (TIGHT): direct ratio of stored ramp values; bit-equal
    # modulo IEEE-754.
    """
    expected = reference["indexed_cashflow"]
    cf_bond = IndexedCashFlow(
        notional=expected["notional"],
        index=seeded_eu_hicp,
        base_date=Date.from_ymd(1, Month.June, 2020),
        fixing_date=Date.from_ymd(1, Month.June, 2022),
        payment_date=Date.from_ymd(15, Month.June, 2022),
        growth_only=False,
    )
    cf_swap = IndexedCashFlow(
        notional=expected["notional"],
        index=seeded_eu_hicp,
        base_date=Date.from_ymd(1, Month.June, 2020),
        fixing_date=Date.from_ymd(1, Month.June, 2022),
        payment_date=Date.from_ymd(15, Month.June, 2022),
        growth_only=True,
    )
    tolerance.tight(cf_bond.base_fixing(), expected["base_fixing"])
    tolerance.tight(cf_bond.index_fixing(), expected["index_fixing"])
    tolerance.tight(cf_bond.amount(), expected["bond_style_amount"])
    tolerance.tight(cf_swap.amount(), expected["swap_style_amount"])
    assert cf_bond.date() == Date.from_ymd(15, Month.June, 2022)
    assert cf_bond.notional() == expected["notional"]
    assert cf_bond.growth_only() is False
    assert cf_swap.growth_only() is True


# ---- ZeroInflationCashFlow ------------------------------------------------


def test_zero_inflation_cashflow_amount(
    seeded_eu_hicp: EUHICP, reference: dict[str, Any]
) -> None:
    """Closed-form ZeroInflationCashFlow base/index/amount.

    # Justification (TIGHT): the lag-adjusted CPI lookup pulls integer
    # ramp values; growth amount is ``N * (113/107 - 1)``.
    """
    expected = reference["zero_inflation_cashflow"]
    start = Date.from_ymd(1, Month.June, 2021)
    end = Date.from_ymd(1, Month.June, 2022)
    pay = Date.from_ymd(15, Month.June, 2022)
    lag = Period(3, TimeUnit.Months)
    cf_growth = ZeroInflationCashFlow(
        notional=expected["notional"],
        index=seeded_eu_hicp,
        observation_interpolation=InterpolationType.Flat,
        start_date=start,
        end_date=end,
        observation_lag=lag,
        payment_date=pay,
        growth_only=True,
    )
    cf_bond = ZeroInflationCashFlow(
        notional=expected["notional"],
        index=seeded_eu_hicp,
        observation_interpolation=InterpolationType.Flat,
        start_date=start,
        end_date=end,
        observation_lag=lag,
        payment_date=pay,
        growth_only=False,
    )
    tolerance.tight(cf_growth.base_fixing(), expected["base_fixing"])
    tolerance.tight(cf_growth.index_fixing(), expected["index_fixing"])
    tolerance.tight(cf_growth.amount(), expected["growth_only_amount"])
    tolerance.tight(cf_bond.amount(), expected["bond_style_amount"])
    # Accessors:
    assert cf_growth.zero_inflation_index() is seeded_eu_hicp
    assert cf_growth.observation_interpolation() == InterpolationType.Flat


# ---- CPICashFlow ----------------------------------------------------------


def test_cpi_cashflow_explicit_base(
    seeded_eu_hicp: EUHICP, reference: dict[str, Any]
) -> None:
    """``CPICashFlow`` with explicit base fixing — base_fixing returns it as-is.

    # Justification (TIGHT): direct division ``113/100 - 1 == 0.13``.
    """
    expected = reference["cpi_cashflow"]
    cf = CPICashFlow(
        notional=expected["notional"],
        index=seeded_eu_hicp,
        base_date=Date.from_ymd(1, Month.March, 2020),
        base_fixing=100.0,
        observation_date=Date.from_ymd(1, Month.June, 2022),
        observation_lag=Period(3, TimeUnit.Months),
        interpolation=InterpolationType.Flat,
        payment_date=Date.from_ymd(15, Month.June, 2022),
        growth_only=True,
    )
    tolerance.tight(cf.base_fixing(), expected["base_fixing"])
    tolerance.tight(cf.index_fixing(), expected["index_fixing"])
    tolerance.tight(cf.amount(), expected["growth_only_amount"])


def test_cpi_cashflow_invalid_neither_base_raises(seeded_eu_hicp: EUHICP) -> None:
    """Constructing CPICashFlow with no base fixing AND no base date raises."""
    with pytest.raises(LibraryException):
        CPICashFlow(
            notional=1000.0,
            index=seeded_eu_hicp,
            base_date=Date(),  # null
            base_fixing=None,
            observation_date=Date.from_ymd(1, Month.June, 2022),
            observation_lag=Period(3, TimeUnit.Months),
            interpolation=InterpolationType.Flat,
            payment_date=Date.from_ymd(15, Month.June, 2022),
        )


def test_cpi_cashflow_zero_base_raises(seeded_eu_hicp: EUHICP) -> None:
    """Numerical guard: |baseCPI| < 1e-16 raises."""
    with pytest.raises(LibraryException):
        CPICashFlow(
            notional=1000.0,
            index=seeded_eu_hicp,
            base_date=Date.from_ymd(1, Month.March, 2020),
            base_fixing=0.0,
            observation_date=Date.from_ymd(1, Month.June, 2022),
            observation_lag=Period(3, TimeUnit.Months),
            interpolation=InterpolationType.Flat,
            payment_date=Date.from_ymd(15, Month.June, 2022),
        )


def test_cpi_cashflow_base_date_lookup(seeded_eu_hicp: EUHICP) -> None:
    """When base_fixing is None, the base date is looked up via lagged_fixing.

    # Justification (TIGHT): with lag=0 the lookup returns the ramp fixing
    # at the base date itself.
    """
    cf = CPICashFlow(
        notional=1000.0,
        index=seeded_eu_hicp,
        base_date=Date.from_ymd(1, Month.March, 2020),
        base_fixing=None,
        observation_date=Date.from_ymd(1, Month.June, 2022),
        observation_lag=Period(3, TimeUnit.Months),
        interpolation=InterpolationType.Flat,
        payment_date=Date.from_ymd(15, Month.June, 2022),
        growth_only=False,
    )
    # base_fixing should be ramp(2020, 3) = 101.0
    tolerance.tight(cf.base_fixing(), 101.0)


# ---- CPICoupon ------------------------------------------------------------


def test_cpi_coupon_index_ratio_and_amount(
    seeded_eu_hicp: EUHICP, reference: dict[str, Any]
) -> None:
    """End-to-end CPICoupon roundtrip via CPICouponPricer.

    # Justification (TIGHT): the rate is ``fixedRate * indexRatio``; the
    # amount = ``nominal * rate * accrual_period``. All operands are
    # closed-form ramped values + a Thirty360 year fraction.
    """
    expected = reference["cpi_coupon"]
    dc = Thirty360(Convention.BondBasis)
    cpn = CPICoupon(
        payment_date=Date.from_ymd(15, Month.June, 2022),
        nominal=expected["nominal"],
        accrual_start_date=Date.from_ymd(1, Month.June, 2021),
        accrual_end_date=Date.from_ymd(1, Month.June, 2022),
        index=seeded_eu_hicp,
        observation_lag=Period(3, TimeUnit.Months),
        observation_interpolation=InterpolationType.Flat,
        day_counter=dc,
        fixed_rate=expected["fixed_rate"],
        base_cpi=expected["base_cpi"],
    )
    cpn.set_pricer(CPICouponPricer())
    tolerance.tight(cpn.accrual_period(), expected["accrual_period"])
    tolerance.tight(cpn.index_fixing(), expected["index_fixing"])
    tolerance.tight(
        cpn.index_ratio(cpn.accrual_end_date()), expected["index_ratio_at_end"]
    )
    tolerance.tight(cpn.rate(), expected["rate"])
    tolerance.tight(cpn.amount(), expected["amount"])
    # Inspectors
    assert cpn.fixed_rate() == expected["fixed_rate"]
    assert cpn.base_cpi() == expected["base_cpi"]
    assert cpn.observation_interpolation() == InterpolationType.Flat
    assert cpn.cpi_index() is seeded_eu_hicp


def test_cpi_coupon_invalid_neither_base_raises(seeded_eu_hicp: EUHICP) -> None:
    """At least one of ``base_cpi`` / ``base_date`` must be supplied."""
    with pytest.raises(LibraryException):
        CPICoupon(
            payment_date=Date.from_ymd(15, Month.June, 2022),
            nominal=100000.0,
            accrual_start_date=Date.from_ymd(1, Month.June, 2021),
            accrual_end_date=Date.from_ymd(1, Month.June, 2022),
            index=seeded_eu_hicp,
            observation_lag=Period(3, TimeUnit.Months),
            observation_interpolation=InterpolationType.Flat,
            day_counter=Thirty360(Convention.BondBasis),
            fixed_rate=0.025,
            base_cpi=None,
            base_date=None,
        )


def test_cpi_coupon_index_ratio_via_base_date(seeded_eu_hicp: EUHICP) -> None:
    """When only ``base_date`` is given, ``index_ratio`` looks up the base
    via ``lagged_fixing(base_date + lag, lag)``.

    # Justification (TIGHT): with base_date=2020-03-01, lag=3M, this looks
    # up CPI(2020-03-01) = ramp(2020,3) = 101.0. End fixing = 113.0.
    # Ratio = 113/101.
    """
    cpn = CPICoupon(
        payment_date=Date.from_ymd(15, Month.June, 2022),
        nominal=100000.0,
        accrual_start_date=Date.from_ymd(1, Month.June, 2021),
        accrual_end_date=Date.from_ymd(1, Month.June, 2022),
        index=seeded_eu_hicp,
        observation_lag=Period(3, TimeUnit.Months),
        observation_interpolation=InterpolationType.Flat,
        day_counter=Thirty360(Convention.BondBasis),
        fixed_rate=0.025,
        base_cpi=None,
        base_date=Date.from_ymd(1, Month.March, 2020),
    )
    ratio = cpn.index_ratio(cpn.accrual_end_date())
    tolerance.tight(ratio, 113.0 / 101.0)


def test_cpi_coupon_adjusted_index_growth(
    seeded_eu_hicp: EUHICP, reference: dict[str, Any]
) -> None:
    """``adjusted_index_growth`` == ``rate / fixedRate`` == indexRatio (no adjustment).

    # Justification (TIGHT): identity in the plain CPICouponPricer.
    """
    expected = reference["cpi_coupon"]
    cpn = CPICoupon(
        payment_date=Date.from_ymd(15, Month.June, 2022),
        nominal=expected["nominal"],
        accrual_start_date=Date.from_ymd(1, Month.June, 2021),
        accrual_end_date=Date.from_ymd(1, Month.June, 2022),
        index=seeded_eu_hicp,
        observation_lag=Period(3, TimeUnit.Months),
        observation_interpolation=InterpolationType.Flat,
        day_counter=Thirty360(Convention.BondBasis),
        fixed_rate=expected["fixed_rate"],
        base_cpi=expected["base_cpi"],
    )
    cpn.set_pricer(CPICouponPricer())
    tolerance.tight(cpn.adjusted_index_growth(), expected["index_ratio_at_end"])


def test_cpi_coupon_wrong_pricer_type_raises(
    seeded_eu_hicp: EUHICP, seeded_yoy_eu_hicp: YoYEUHICP
) -> None:
    """Attaching a YoY pricer to a CPICoupon raises."""
    cpn = CPICoupon(
        payment_date=Date.from_ymd(15, Month.June, 2022),
        nominal=100000.0,
        accrual_start_date=Date.from_ymd(1, Month.June, 2021),
        accrual_end_date=Date.from_ymd(1, Month.June, 2022),
        index=seeded_eu_hicp,
        observation_lag=Period(3, TimeUnit.Months),
        observation_interpolation=InterpolationType.Flat,
        day_counter=Thirty360(Convention.BondBasis),
        fixed_rate=0.025,
        base_cpi=100.0,
    )
    del seeded_yoy_eu_hicp
    with pytest.raises(LibraryException):
        cpn.set_pricer(YoYInflationCouponPricer())


# ---- YoYInflationCoupon ---------------------------------------------------


def test_yoy_inflation_coupon_amount(
    seeded_yoy_eu_hicp: YoYEUHICP, reference: dict[str, Any]
) -> None:
    """Closed-form YoY coupon amount.

    # Justification (TIGHT): rate = gearing * yoy_fixing + spread, amount =
    # nominal * rate * accrual_period. All values are exact ramp lookups.
    """
    expected = reference["yoy_inflation_coupon"]
    cpn = YoYInflationCoupon(
        payment_date=Date.from_ymd(15, Month.June, 2022),
        nominal=expected["nominal"],
        accrual_start_date=Date.from_ymd(1, Month.June, 2021),
        accrual_end_date=Date.from_ymd(1, Month.June, 2022),
        fixing_days=0,
        index=seeded_yoy_eu_hicp,
        observation_lag=Period(3, TimeUnit.Months),
        interpolation=InterpolationType.Flat,
        day_counter=Thirty360(Convention.BondBasis),
        gearing=expected["gearing"],
        spread=expected["spread"],
    )
    cpn.set_pricer(YoYInflationCouponPricer())
    tolerance.tight(cpn.accrual_period(), expected["accrual_period"])
    tolerance.tight(cpn.index_fixing(), expected["index_fixing"])
    tolerance.tight(cpn.rate(), expected["rate"])
    tolerance.tight(cpn.amount(), expected["amount"])
    assert cpn.gearing() == expected["gearing"]
    assert cpn.spread() == expected["spread"]
    assert cpn.yoy_index() is seeded_yoy_eu_hicp
    assert cpn.interpolation() == InterpolationType.Flat


def test_yoy_inflation_coupon_adjusted_fixing(
    seeded_yoy_eu_hicp: YoYEUHICP,
) -> None:
    """``adjusted_fixing`` == ``(rate - spread) / gearing`` — identity in the
    plain pricer.

    # Justification (TIGHT): closed-form identity.
    """
    cpn = YoYInflationCoupon(
        payment_date=Date.from_ymd(15, Month.June, 2022),
        nominal=100000.0,
        accrual_start_date=Date.from_ymd(1, Month.June, 2021),
        accrual_end_date=Date.from_ymd(1, Month.June, 2022),
        fixing_days=0,
        index=seeded_yoy_eu_hicp,
        observation_lag=Period(3, TimeUnit.Months),
        interpolation=InterpolationType.Flat,
        day_counter=Thirty360(Convention.BondBasis),
        gearing=1.5,
        spread=0.005,
    )
    cpn.set_pricer(YoYInflationCouponPricer())
    # adjusted_fixing = (rate - spread) / gearing == index_fixing in the plain pricer.
    tolerance.tight(cpn.adjusted_fixing(), cpn.index_fixing())


def test_yoy_inflation_coupon_wrong_pricer_type_raises(
    seeded_yoy_eu_hicp: YoYEUHICP,
) -> None:
    """A CPI pricer on a YoY coupon raises."""
    cpn = YoYInflationCoupon(
        payment_date=Date.from_ymd(15, Month.June, 2022),
        nominal=100000.0,
        accrual_start_date=Date.from_ymd(1, Month.June, 2021),
        accrual_end_date=Date.from_ymd(1, Month.June, 2022),
        fixing_days=0,
        index=seeded_yoy_eu_hicp,
        observation_lag=Period(3, TimeUnit.Months),
        interpolation=InterpolationType.Flat,
        day_counter=Thirty360(Convention.BondBasis),
    )
    with pytest.raises(LibraryException):
        cpn.set_pricer(CPICouponPricer())


# ---- CappedFlooredYoYInflationCoupon --------------------------------------


def test_capped_floored_aliases() -> None:
    """Python-side aliases redirect to the YoY variant.

    # C++ parity: no analogous class; documented Python convenience.
    """
    assert CappedFlooredCPICoupon is CappedFlooredYoYInflationCoupon
    assert CappedFlooredInflationCoupon is CappedFlooredYoYInflationCoupon


def test_capped_floored_unbounded_rate_matches_base(
    seeded_yoy_eu_hicp: YoYEUHICP,
) -> None:
    """No cap, no floor — rate == base swaplet rate.

    # Justification (TIGHT): the capped/floored wrapper degenerates to the
    # base ``YoYInflationCoupon.rate`` when neither cap nor floor is set.
    """
    base_cpn = YoYInflationCoupon(
        payment_date=Date.from_ymd(15, Month.June, 2022),
        nominal=100000.0,
        accrual_start_date=Date.from_ymd(1, Month.June, 2021),
        accrual_end_date=Date.from_ymd(1, Month.June, 2022),
        fixing_days=0,
        index=seeded_yoy_eu_hicp,
        observation_lag=Period(3, TimeUnit.Months),
        interpolation=InterpolationType.Flat,
        day_counter=Thirty360(Convention.BondBasis),
        gearing=1.0,
        spread=0.0,
    )
    base_cpn.set_pricer(YoYInflationCouponPricer())
    capped = CappedFlooredYoYInflationCoupon(
        payment_date=Date.from_ymd(15, Month.June, 2022),
        nominal=100000.0,
        accrual_start_date=Date.from_ymd(1, Month.June, 2021),
        accrual_end_date=Date.from_ymd(1, Month.June, 2022),
        fixing_days=0,
        index=seeded_yoy_eu_hicp,
        observation_lag=Period(3, TimeUnit.Months),
        interpolation=InterpolationType.Flat,
        day_counter=Thirty360(Convention.BondBasis),
    )
    capped.set_pricer(YoYInflationCouponPricer())
    assert not capped.is_capped()
    assert not capped.is_floored()
    tolerance.tight(capped.rate(), base_cpn.rate())


def test_capped_floored_cap_floor_sign_aware(
    seeded_yoy_eu_hicp: YoYEUHICP,
) -> None:
    """Positive gearing: ``cap()`` returns the user cap; ``floor()`` the floor.

    # C++ parity: capflooredinflationcoupon.cpp:108-123.
    """
    capped = CappedFlooredYoYInflationCoupon(
        payment_date=Date.from_ymd(15, Month.June, 2022),
        nominal=100000.0,
        accrual_start_date=Date.from_ymd(1, Month.June, 2021),
        accrual_end_date=Date.from_ymd(1, Month.June, 2022),
        fixing_days=0,
        index=seeded_yoy_eu_hicp,
        observation_lag=Period(3, TimeUnit.Months),
        interpolation=InterpolationType.Flat,
        day_counter=Thirty360(Convention.BondBasis),
        gearing=1.0,
        spread=0.0,
        cap=0.04,
        floor=0.01,
    )
    assert capped.cap() == 0.04
    assert capped.floor() == 0.01
    assert capped.is_capped()
    assert capped.is_floored()
    tolerance.tight(capped.effective_cap(), 0.04)
    tolerance.tight(capped.effective_floor(), 0.01)


def test_capped_floored_cap_below_floor_raises(
    seeded_yoy_eu_hicp: YoYEUHICP,
) -> None:
    """Cap < floor in a collar configuration raises."""
    with pytest.raises(LibraryException):
        CappedFlooredYoYInflationCoupon(
            payment_date=Date.from_ymd(15, Month.June, 2022),
            nominal=100000.0,
            accrual_start_date=Date.from_ymd(1, Month.June, 2021),
            accrual_end_date=Date.from_ymd(1, Month.June, 2022),
            fixing_days=0,
            index=seeded_yoy_eu_hicp,
            observation_lag=Period(3, TimeUnit.Months),
            interpolation=InterpolationType.Flat,
            day_counter=Thirty360(Convention.BondBasis),
            cap=0.01,
            floor=0.05,
        )


def test_capped_floored_underlying_rate_propagates(
    seeded_yoy_eu_hicp: YoYEUHICP,
) -> None:
    """``from_underlying`` wraps an existing YoY coupon; setting a pricer
    propagates to both wrapper and underlying.

    # Justification: behavioural — verify the pricer fan-out logic.
    """
    underlying = YoYInflationCoupon(
        payment_date=Date.from_ymd(15, Month.June, 2022),
        nominal=100000.0,
        accrual_start_date=Date.from_ymd(1, Month.June, 2021),
        accrual_end_date=Date.from_ymd(1, Month.June, 2022),
        fixing_days=0,
        index=seeded_yoy_eu_hicp,
        observation_lag=Period(3, TimeUnit.Months),
        interpolation=InterpolationType.Flat,
        day_counter=Thirty360(Convention.BondBasis),
    )
    capped = CappedFlooredYoYInflationCoupon.from_underlying(underlying)
    capped.set_pricer(YoYInflationCouponPricer())
    assert underlying.pricer() is not None
    assert capped.pricer() is not None


# ---- InflationCoupon abstract ---------------------------------------------


def test_inflation_coupon_cannot_be_instantiated_directly() -> None:
    """Direct instantiation of the abstract base raises (Python's ABC machinery)."""
    with pytest.raises(TypeError):
        InflationCoupon(  # type: ignore[abstract]
            payment_date=Date.from_ymd(15, Month.June, 2022),
            nominal=1.0,
            accrual_start_date=Date.from_ymd(1, Month.June, 2021),
            accrual_end_date=Date.from_ymd(1, Month.June, 2022),
            fixing_days=0,
            index=EUHICP(),
            observation_lag=Period(3, TimeUnit.Months),
            day_counter=Thirty360(Convention.BondBasis),
        )


def test_inflation_coupon_rate_without_pricer_raises(seeded_eu_hicp: EUHICP) -> None:
    """Calling ``rate()`` before ``set_pricer`` raises."""
    cpn = CPICoupon(
        payment_date=Date.from_ymd(15, Month.June, 2022),
        nominal=100000.0,
        accrual_start_date=Date.from_ymd(1, Month.June, 2021),
        accrual_end_date=Date.from_ymd(1, Month.June, 2022),
        index=seeded_eu_hicp,
        observation_lag=Period(3, TimeUnit.Months),
        observation_interpolation=InterpolationType.Flat,
        day_counter=Thirty360(Convention.BondBasis),
        fixed_rate=0.025,
        base_cpi=100.0,
    )
    with pytest.raises(LibraryException):
        cpn.rate()


def test_inflation_coupon_accrued_amount_outside_window(
    seeded_eu_hicp: EUHICP,
) -> None:
    """``accrued_amount`` returns 0 outside the accrual window."""
    cpn = CPICoupon(
        payment_date=Date.from_ymd(15, Month.June, 2022),
        nominal=100000.0,
        accrual_start_date=Date.from_ymd(1, Month.June, 2021),
        accrual_end_date=Date.from_ymd(1, Month.June, 2022),
        index=seeded_eu_hicp,
        observation_lag=Period(3, TimeUnit.Months),
        observation_interpolation=InterpolationType.Flat,
        day_counter=Thirty360(Convention.BondBasis),
        fixed_rate=0.025,
        base_cpi=100.0,
    )
    cpn.set_pricer(CPICouponPricer())
    # Before accrual start.
    assert cpn.accrued_amount(Date.from_ymd(1, Month.May, 2021)) == 0.0
    # After payment date.
    assert cpn.accrued_amount(Date.from_ymd(15, Month.July, 2022)) == 0.0


def test_inflation_coupon_alias_class_exposes_black_pricer() -> None:
    """``BlackCPICouponPricer`` is a CPICouponPricer subclass — sanity check."""
    pricer = BlackCPICouponPricer()
    assert isinstance(pricer, CPICouponPricer)
