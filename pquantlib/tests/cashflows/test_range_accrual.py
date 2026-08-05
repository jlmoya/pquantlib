"""Cross-validate the range-accrual family against C++ QuantLib v1.43.

Probe: ``v143/cf/rangeaccrual`` — RangeAccrualFloatersCoupon /
RangeAccrualPricer / RangeAccrualPricerByBgm / RangeAccrualLeg.

The market (zero curve, Euribor6M, schedules) is rebuilt here exactly as the
probe builds it; every number below comes from running C++ v1.43.

Tolerance tiers used:

- TIGHT for everything structural (dates, accruals, times, triggers), for the
  analytic digital price, and for the smile-corrected price — all short
  arithmetic chains over identical inputs, measured at <= 4e-16 relative.
- LOOSE only for ``callSpreadPrice``, where the digital is recovered by
  dividing a difference of two nearly equal Black prices by
  ``eps_ = 1e-8``; the derivation of that bound is inline at the call site.
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.cashflows.cash_flows import CashFlows
from pquantlib.cashflows.coupon_pricer import set_coupon_pricer
from pquantlib.cashflows.fixed_rate_coupon import FixedRateCoupon
from pquantlib.cashflows.range_accrual import (
    RangeAccrualFloatersCoupon,
    RangeAccrualLeg,
    RangeAccrualPricer,
    RangeAccrualPricerByBgm,
)
from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exceptions import LibraryException
from pquantlib.indexes.ibor.euribor import Euribor
from pquantlib.termstructures.volatility.flat_smile_section import FlatSmileSection
from pquantlib.termstructures.volatility.smile_section import SmileSection
from pquantlib.termstructures.yield_.interpolated_zero_curve import InterpolatedZeroCurve
from pquantlib.testing import reference_reader, tolerance
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.date import Date
from pquantlib.time.date_generation import DateGeneration
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.schedule import Schedule
from pquantlib.time.time_unit import TimeUnit

MF = BusinessDayConvention.ModifiedFollowing
FOLLOWING = BusinessDayConvention.Following
PRECEDING = BusinessDayConvention.Preceding
UNADJUSTED = BusinessDayConvention.Unadjusted

EVAL_DATE = Date.from_ymd(15, Month.February, 2024)
CORRELATION = 0.7


# --- market ---------------------------------------------------------------


# C++ ``ZeroCurve`` is ``InterpolatedZeroCurve<Linear>``; PQuantLib exposes the
# same class with LinearInterpolation as its default interpolator.
def _curve() -> InterpolatedZeroCurve:
    dates = [
        Date.from_ymd(15, Month.February, 2024),
        Date.from_ymd(15, Month.August, 2024),
        Date.from_ymd(15, Month.February, 2025),
        Date.from_ymd(15, Month.February, 2026),
        Date.from_ymd(15, Month.February, 2027),
        Date.from_ymd(15, Month.February, 2029),
        Date.from_ymd(15, Month.February, 2034),
    ]
    zeros = [0.0300, 0.0320, 0.0345, 0.0375, 0.0395, 0.0420, 0.0450]
    return InterpolatedZeroCurve(dates, zeros, Actual365Fixed())


def _index(curve: InterpolatedZeroCurve) -> Euribor:
    return Euribor.six_months(curve)


# --- smile fixtures -------------------------------------------------------


class _AffineSmileSection(SmileSection):
    """``vol(K) = base + slope * (K - anchor)`` — mirrors the probe's fixture.

    A flat smile makes ``smileCorrection`` identically zero, so the
    ``with_smile=True / by_call_spread=False`` path needs a section with a
    non-zero, exactly-known dSigma/dK.
    """

    def __init__(
        self, *, exercise_time: float, base: float, slope: float, anchor: float, atm: float
    ) -> None:
        super().__init__(exercise_time=exercise_time, day_counter=Actual365Fixed())
        self._base = base
        self._slope = slope
        self._anchor = anchor
        self._atm = atm

    def min_strike(self) -> float:
        return -1.0

    def max_strike(self) -> float:
        return 10.0

    def atm_level(self) -> float:
        return self._atm

    def _volatility_impl(self, strike: float) -> float:
        dk = strike - self._anchor
        scaled = self._slope * dk
        return self._base + scaled


def _affine_expiry() -> _AffineSmileSection:
    return _AffineSmileSection(exercise_time=1.5, base=0.25, slope=-2.0, anchor=0.04, atm=0.04)


def _affine_payment() -> _AffineSmileSection:
    return _AffineSmileSection(exercise_time=2.0, base=0.22, slope=-1.5, anchor=0.04, atm=0.04)


def _flat_expiry() -> FlatSmileSection:
    return FlatSmileSection(
        volatility=0.25, exercise_time=1.5, day_counter=Actual365Fixed(), atm_level=0.04
    )


def _flat_payment() -> FlatSmileSection:
    return FlatSmileSection(
        volatility=0.22, exercise_time=2.0, day_counter=Actual365Fixed(), atm_level=0.04
    )


def _pricer(name: str) -> RangeAccrualPricerByBgm:
    """The probe's seven (smile fixture, with_smile, by_call_spread) configs."""
    configs: dict[str, tuple[bool, bool, bool]] = {
        # name -> (affine smile?, with_smile, by_call_spread)
        "affine_nosmile_nocs": (True, False, False),
        "affine_nosmile_cs": (True, False, True),
        "affine_smile_nocs": (True, True, False),
        "affine_smile_cs": (True, True, True),
        "flat_nosmile_nocs": (False, False, False),
        "flat_smile_nocs": (False, True, False),
        "flat_smile_cs": (False, True, True),
    }
    affine, with_smile, by_call_spread = configs[name]
    on_expiry = _affine_expiry() if affine else _flat_expiry()
    on_payment = _affine_payment() if affine else _flat_payment()
    return RangeAccrualPricerByBgm(
        CORRELATION, on_expiry, on_payment, with_smile, by_call_spread
    )


# --- schedules ------------------------------------------------------------


def _schedule(start: Date, end: Date, convention: BusinessDayConvention) -> Schedule:
    return Schedule.from_rule(
        start,
        end,
        Period(6, TimeUnit.Months),
        TARGET(),
        convention,
        convention,
        DateGeneration.Forward,
        False,
    )


def _schedule_main() -> Schedule:
    return _schedule(Date.from_ymd(17, Month.March, 2025), Date.from_ymd(17, Month.March, 2027), MF)


def _schedule_vectors() -> Schedule:
    return _schedule(
        Date.from_ymd(16, Month.June, 2025), Date.from_ymd(16, Month.December, 2026), MF
    )


def _schedule_fixed() -> Schedule:
    return _schedule(
        Date.from_ymd(15, Month.September, 2025), Date.from_ymd(15, Month.September, 2026), MF
    )


def _schedule_unadjusted() -> Schedule:
    """Raw schedule whose 2026-05-17 termination is a Sunday."""
    return _schedule(
        Date.from_ymd(17, Month.May, 2025), Date.from_ymd(17, Month.May, 2026), UNADJUSTED
    )


# --- legs -----------------------------------------------------------------


def _leg_main(index: Euribor) -> list[Any]:
    return (
        RangeAccrualLeg(_schedule_main(), index)
        .with_notionals([1000000.0, 900000.0, 800000.0])
        .with_payment_day_counter(Actual360())
        .with_payment_adjustment(MF)
        .with_fixing_days([3, 1])
        .with_gearings(0.75)
        .with_spreads(0.0025)
        .with_lower_triggers(0.030)
        .with_upper_triggers(0.055)
        .with_observation_tenor(Period(1, TimeUnit.Months))
        .with_observation_convention(MF)
        .build()
    )


def _leg_vectors(
    index: Euribor,
    *,
    observation_tenor: Period | None = None,
    observation_convention: BusinessDayConvention = UNADJUSTED,
) -> list[Any]:
    tenor = observation_tenor if observation_tenor is not None else Period(2, TimeUnit.Months)
    return (
        RangeAccrualLeg(_schedule_vectors(), index)
        .with_notionals(2000000.0)
        .with_payment_day_counter(Actual365Fixed())
        .with_payment_adjustment(FOLLOWING)
        .with_fixing_days(0)
        .with_gearings([1.25, 0.5])
        .with_spreads([-0.001, 0.002])
        .with_lower_triggers([0.025, 0.028, 0.031])
        .with_upper_triggers([0.060, 0.058, 0.056])
        .with_observation_tenor(tenor)
        .with_observation_convention(observation_convention)
        .build()
    )


def _leg_fixed(index: Euribor) -> list[Any]:
    return (
        RangeAccrualLeg(_schedule_fixed(), index)
        .with_notionals(500000.0)
        .with_payment_day_counter(Actual360())
        .with_gearings(0.0)
        .with_spreads(0.035)
        .with_lower_triggers(0.030)
        .with_upper_triggers(0.055)
        .with_observation_tenor(Period(1, TimeUnit.Months))
        .build()
    )


def _leg_unadjusted(index: Euribor, payment_adjustment: BusinessDayConvention) -> list[Any]:
    return (
        RangeAccrualLeg(_schedule_unadjusted(), index)
        .with_notionals(1000000.0)
        .with_payment_day_counter(Actual360())
        .with_payment_adjustment(payment_adjustment)
        .with_gearings(1.0)
        .with_lower_triggers(0.030)
        .with_upper_triggers(0.055)
        .with_observation_tenor(Period(3, TimeUnit.Months))
        # Unadjusted is mandatory: the coupon requires the observation schedule
        # to start / end exactly on the (weekend) accrual dates.
        .with_observation_convention(UNADJUSTED)
        .build()
    )


# --- fixtures -------------------------------------------------------------


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("v143/cf/rangeaccrual")


@pytest.fixture
def index() -> Euribor:
    return _index(_curve())


# --- setup sanity ---------------------------------------------------------


def test_setup_matches_probe(cpp: dict[str, Any], index: Euribor) -> None:
    setup = cpp["setup"]
    assert EVAL_DATE.serial_number() == setup["evaluation_date"]
    assert index.name() == setup["index_name"]
    curve = _curve()
    assert [d.serial_number() for d in curve.dates()] == setup["curve_dates"]
    tolerance.tight(CORRELATION, setup["correlation"])
    for section, key, strike in (
        (_affine_expiry(), "affine_expiry", 0.03),
        (_affine_payment(), "affine_payment", 0.03),
    ):
        ref = setup[key]
        expected = ref["base"] + ref["slope"] * (strike - ref["anchor"])
        tolerance.tight(section.volatility(strike), expected)
    tolerance.tight(_flat_expiry().volatility(0.03), setup["flat_expiry_vol"])
    tolerance.tight(_flat_payment().volatility(0.09), setup["flat_payment_vol"])


# --- coupon structure -----------------------------------------------------


def _check_coupon_structure(coupon: RangeAccrualFloatersCoupon, ref: dict[str, Any]) -> None:
    assert coupon.date().serial_number() == ref["payment_date"]
    tolerance.tight(coupon.nominal(), ref["nominal"])
    assert coupon.accrual_start_date().serial_number() == ref["accrual_start_date"]
    assert coupon.accrual_end_date().serial_number() == ref["accrual_end_date"]
    tolerance.tight(coupon.accrual_period(), ref["accrual_period"])
    assert coupon.day_counter().name() == ref["day_counter"]
    assert coupon.fixing_days() == ref["fixing_days"]
    assert coupon.fixing_date().serial_number() == ref["fixing_date"]
    tolerance.tight(coupon.index_fixing(), ref["index_fixing"])
    tolerance.tight(coupon.gearing(), ref["gearing"])
    tolerance.tight(coupon.spread(), ref["spread"])
    tolerance.tight(coupon.start_time(), ref["start_time"])
    tolerance.tight(coupon.end_time(), ref["end_time"])
    tolerance.tight(coupon.lower_trigger(), ref["lower_trigger"])
    tolerance.tight(coupon.upper_trigger(), ref["upper_trigger"])
    assert coupon.observations_no() == ref["observations_no"]
    assert [d.serial_number() for d in coupon.observation_schedule().dates] == ref[
        "observation_schedule_dates"
    ]
    assert [d.serial_number() for d in coupon.observation_dates()] == ref["observation_dates"]
    assert len(coupon.observation_times()) == len(ref["observation_times"])
    for actual, expected in zip(
        coupon.observation_times(), ref["observation_times"], strict=True
    ):
        tolerance.tight(actual, expected)


def _check_leg_structure(leg: list[Any], ref: dict[str, Any], schedule: Schedule) -> None:
    assert [d.serial_number() for d in schedule.dates] == ref["schedule_dates"]
    assert len(leg) == ref["coupon_count"]
    # The C++ builder returns 2n entries whose first n are null (see the
    # range_accrual module docstring); build() returns the n real coupons.
    assert ref["raw_leg_size"] == 2 * ref["coupon_count"]
    assert ref["null_prefix"] == ref["coupon_count"]
    for coupon, cref in zip(leg, ref["coupons"], strict=True):
        assert isinstance(coupon, RangeAccrualFloatersCoupon)
        _check_coupon_structure(coupon, cref)


def test_leg_main_structure(cpp: dict[str, Any], index: Euribor) -> None:
    _check_leg_structure(_leg_main(index), cpp["leg_main"], _schedule_main())


def test_leg_vectors_structure(cpp: dict[str, Any], index: Euribor) -> None:
    _check_leg_structure(_leg_vectors(index), cpp["leg_vectors"], _schedule_vectors())


@pytest.mark.parametrize("leg_key", ["leg_main", "leg_vectors"])
def test_price_without_optionality(cpp: dict[str, Any], index: Euribor, leg_key: str) -> None:
    curve = _curve()
    idx = _index(curve)
    leg = _leg_main(idx) if leg_key == "leg_main" else _leg_vectors(idx)
    del index
    for coupon, cref in zip(leg, cpp[leg_key]["coupons"], strict=True):
        assert isinstance(coupon, RangeAccrualFloatersCoupon)
        # TIGHT: accrualPeriod * (gearing * fixing + spread) * nominal * df —
        # a short arithmetic chain over identical curve inputs, no solver.
        tolerance.tight(
            coupon.price_without_optionality(curve), cref["price_without_optionality"]
        )


# --- BGM pricer -----------------------------------------------------------

#: TIGHT paths.
#:
#: - ``*_nosmile_*``: ``with_smile`` is False, so ``digitalPriceWithoutSmile``
#:   is the whole computation — closed form, no finite difference anywhere.
#: - ``flat_smile_nocs``: ``smileCorrection`` runs, but a flat smile makes its
#:   ``eps_ = 1e-8`` difference quotient exactly 0/1e-8, so the correction is
#:   0.0 on both sides.
#: - ``affine_smile_nocs``: ``smileCorrection`` runs on the affine fixture. Its
#:   ``eps_`` step never amplifies here, because (vol(K+eps/2) - vol(K-eps/2))
#:   is computed from the same operands and the same operations in both
#:   languages, so the quotient agrees to the last bit; ``smileCorrection``
#:   itself is closed form (no solver, despite the FD step). Measured C++ vs
#:   Python: 3.4e-16 relative, four orders inside TIGHT. A *curved* smile
#:   fixture would need LOOSE — the affine one deliberately does not.
_TIGHT_PRICERS = [
    "affine_nosmile_nocs",
    "affine_nosmile_cs",
    "affine_smile_nocs",
    "flat_nosmile_nocs",
]

#: LOOSE paths — the call-spread replication, where 1e-8 genuinely amplifies.
_LOOSE_PRICERS = ["affine_smile_cs", "flat_smile_cs"]


def _check_priced_leg(
    leg: list[Any], curve: InterpolatedZeroCurve, ref: dict[str, Any], *, tight: bool
) -> None:
    def check(actual: float, expected: float) -> None:
        if tight:
            tolerance.tight(actual, expected)
        else:
            # LOOSE: ``callSpreadPrice`` recovers the digital as
            # (C(K-eps/2) - C(K+eps/2)) / eps with eps = 1e-8. Each Black price
            # carries ~1e-16 relative rounding — and the C++
            # ErrorFunction-based CumulativeNormalDistribution is not
            # bit-identical to Python's math.erf — while the numerator is only
            # ~1e-8 * dC/dK. The quotient therefore inherits roughly
            # 1e-16 / 1e-8 * C/(dC/dK) ~ 1e-9 relative error. Measured worst
            # case here: 1.2e-9, an order inside LOOSE. Pure conditioning of
            # the replication, not a modelling difference; it cannot be
            # tightened without changing the C++ eps_.
            tolerance.loose(actual, expected)

    for coupon, rate, amount, price in zip(
        leg, ref["rates"], ref["amounts"], ref["prices"], strict=True
    ):
        assert isinstance(coupon, RangeAccrualFloatersCoupon)
        check(coupon.rate(), rate)
        check(coupon.amount(), amount)
        check(coupon.price(curve), price)
    check(CashFlows.npv_curve(leg, curve, False, EVAL_DATE), ref["npv"])


@pytest.mark.parametrize("name", _TIGHT_PRICERS)
def test_leg_main_pricers_tight(cpp: dict[str, Any], name: str) -> None:
    curve = _curve()
    leg = _leg_main(_index(curve))
    set_coupon_pricer(leg, _pricer(name))
    _check_priced_leg(leg, curve, cpp["leg_main_pricers"][name], tight=True)


@pytest.mark.parametrize("name", _LOOSE_PRICERS)
def test_leg_main_pricers_loose(cpp: dict[str, Any], name: str) -> None:
    curve = _curve()
    leg = _leg_main(_index(curve))
    set_coupon_pricer(leg, _pricer(name))
    _check_priced_leg(leg, curve, cpp["leg_main_pricers"][name], tight=False)


def test_by_call_spread_is_inert_without_smile(cpp: dict[str, Any]) -> None:
    """``by_call_spread`` selects nothing when ``with_smile`` is False."""
    pricers = cpp["leg_main_pricers"]
    assert pricers["affine_nosmile_nocs"]["rates"] == pricers["affine_nosmile_cs"]["rates"]

    curve = _curve()
    rates: list[list[float]] = []
    for name in ("affine_nosmile_nocs", "affine_nosmile_cs"):
        leg = _leg_main(_index(curve))
        set_coupon_pricer(leg, _pricer(name))
        rates.append([c.rate() for c in leg])
    assert rates[0] == rates[1]


def test_flat_smile_correction_vanishes(cpp: dict[str, Any]) -> None:
    """A flat smile has dSigma/dK == 0, so smileCorrection must be exactly 0."""
    pricers = cpp["leg_main_pricers"]
    assert pricers["flat_nosmile_nocs"]["rates"] == pricers["flat_smile_nocs"]["rates"]

    curve = _curve()
    rates: list[list[float]] = []
    for name in ("flat_nosmile_nocs", "flat_smile_nocs"):
        leg = _leg_main(_index(curve))
        set_coupon_pricer(leg, _pricer(name))
        rates.append([c.rate() for c in leg])
    assert rates[0] == rates[1]


def test_smile_paths_differ(cpp: dict[str, Any]) -> None:
    """The four combinations must not collapse onto one another."""
    pricers = cpp["leg_main_pricers"]
    assert pricers["affine_smile_nocs"]["rates"] != pricers["affine_nosmile_nocs"]["rates"]
    assert pricers["affine_smile_cs"]["rates"] != pricers["affine_smile_nocs"]["rates"]


def test_leg_vectors_priced(cpp: dict[str, Any]) -> None:
    curve = _curve()
    leg = _leg_vectors(_index(curve))
    set_coupon_pricer(leg, _pricer("affine_smile_cs"))
    _check_priced_leg(leg, curve, cpp["leg_vectors_pricers"]["affine_smile_cs"], tight=False)


# --- builder arguments: each at a non-default value ------------------------


def test_with_notionals_vector_and_scalar(cpp: dict[str, Any], index: Euribor) -> None:
    """Vector overload, shorter than the leg: the last value repeats (detail::get)."""
    nominals = [c.nominal() for c in _leg_main(index)]
    assert nominals == [c["nominal"] for c in cpp["leg_main"]["coupons"]]
    assert nominals == [1000000.0, 900000.0, 800000.0, 800000.0]
    # scalar overload
    assert [c.nominal() for c in _leg_vectors(index)] == [2000000.0] * 3


def test_with_payment_day_counter(cpp: dict[str, Any], index: Euribor) -> None:
    """Act/360 vs Act/365F produce different accrual periods, not just labels."""
    main = _leg_main(index)
    vectors = _leg_vectors(index)
    assert {c.day_counter().name() for c in main} == {"Actual/360"}
    assert {c.day_counter().name() for c in vectors} == {"Actual/365 (Fixed)"}
    tolerance.tight(main[0].accrual_period(), cpp["leg_main"]["coupons"][0]["accrual_period"])
    tolerance.tight(
        vectors[0].accrual_period(), cpp["leg_vectors"]["coupons"][0]["accrual_period"]
    )
    assert main[0].accrual_period() != vectors[0].accrual_period()


def test_with_payment_adjustment(cpp: dict[str, Any], index: Euribor) -> None:
    """A/B over an unadjusted schedule whose last date is a Sunday."""
    ref = cpp["variant_payment_adjustment"]
    assert [d.serial_number() for d in _schedule_unadjusted().dates] == ref["schedule_dates"]
    for key, convention in (
        ("preceding", PRECEDING),
        ("following", FOLLOWING),
        ("modified_following", MF),
    ):
        leg = _leg_unadjusted(index, convention)
        assert [c.date().serial_number() for c in leg] == ref[key]
    # the setter must actually change something
    assert ref["preceding"] != ref["following"]


def test_with_fixing_days_vector_and_scalar(cpp: dict[str, Any], index: Euribor) -> None:
    """Non-default fixing days move the fixing date and the pricer's fixings."""
    main = _leg_main(index)
    assert [c.fixing_days() for c in main] == [3, 1, 1, 1]
    assert [c.fixing_days() for c in main] == [c["fixing_days"] for c in cpp["leg_main"]["coupons"]]
    assert [c.fixing_date().serial_number() for c in main] == [
        c["fixing_date"] for c in cpp["leg_main"]["coupons"]
    ]
    # scalar overload, 0 instead of the C++ default of 2
    assert [c.fixing_days() for c in _leg_vectors(index)] == [0, 0, 0]


def test_with_gearings_scalar_and_vector(cpp: dict[str, Any], index: Euribor) -> None:
    assert [c.gearing() for c in _leg_main(index)] == [0.75] * 4
    assert [c.gearing() for c in _leg_vectors(index)] == [1.25, 0.5, 0.5]
    assert [c.gearing() for c in _leg_vectors(index)] == [
        c["gearing"] for c in cpp["leg_vectors"]["coupons"]
    ]


def test_with_spreads_scalar_and_vector(cpp: dict[str, Any], index: Euribor) -> None:
    assert [c.spread() for c in _leg_main(index)] == [0.0025] * 4
    assert [c.spread() for c in _leg_vectors(index)] == [
        c["spread"] for c in cpp["leg_vectors"]["coupons"]
    ]


def test_with_triggers_scalar_and_vector(cpp: dict[str, Any], index: Euribor) -> None:
    main = _leg_main(index)
    assert [c.lower_trigger() for c in main] == [0.030] * 4
    assert [c.upper_trigger() for c in main] == [0.055] * 4
    vectors = _leg_vectors(index)
    assert [c.lower_trigger() for c in vectors] == [
        c["lower_trigger"] for c in cpp["leg_vectors"]["coupons"]
    ]
    assert [c.upper_trigger() for c in vectors] == [
        c["upper_trigger"] for c in cpp["leg_vectors"]["coupons"]
    ]


def test_with_observation_tenor_and_convention(cpp: dict[str, Any], index: Euribor) -> None:
    """A/B over one schedule: the tenor sets the count, the convention the dates."""
    ref = cpp["variant_observation"]
    variants = (
        ("tenor_2m_unadjusted", Period(2, TimeUnit.Months), UNADJUSTED),
        ("tenor_2m_modified_following", Period(2, TimeUnit.Months), MF),
        ("tenor_3m_unadjusted", Period(3, TimeUnit.Months), UNADJUSTED),
    )
    for key, tenor, convention in variants:
        leg = _leg_vectors(index, observation_tenor=tenor, observation_convention=convention)
        assert [c.observations_no() for c in leg] == ref[key]["observations_no"]
        assert [
            [d.serial_number() for d in c.observation_dates()] for c in leg
        ] == ref[key]["observation_dates"]
    # both setters must actually change something
    assert (
        ref["tenor_2m_unadjusted"]["observation_dates"]
        != ref["tenor_2m_modified_following"]["observation_dates"]
    )
    assert ref["tenor_2m_unadjusted"]["observations_no"] != ref["tenor_3m_unadjusted"][
        "observations_no"
    ]


def test_zero_gearing_builds_fixed_coupons(cpp: dict[str, Any], index: Euribor) -> None:
    """``with_gearings(0.0)`` selects the FixedRateCoupon branch of operator Leg()."""
    ref = cpp["leg_zero_gearing"]
    leg = _leg_fixed(index)
    assert len(leg) == ref["coupon_count"]
    for coupon, cref in zip(leg, ref["coupons"], strict=True):
        assert isinstance(coupon, FixedRateCoupon)
        assert coupon.date().serial_number() == cref["payment_date"]
        tolerance.tight(coupon.nominal(), cref["nominal"])
        assert coupon.accrual_start_date().serial_number() == cref["accrual_start_date"]
        assert coupon.accrual_end_date().serial_number() == cref["accrual_end_date"]
        tolerance.tight(coupon.accrual_period(), cref["accrual_period"])
        # the fixed rate is the period's spread
        tolerance.tight(coupon.rate(), cref["rate"])
        tolerance.tight(coupon.amount(), cref["amount"])
    tolerance.tight(CashFlows.npv_curve(leg, _curve(), False, EVAL_DATE), ref["npv"])


# --- failure modes --------------------------------------------------------


def test_build_requires_notionals(index: Euribor) -> None:
    with pytest.raises(LibraryException, match="no notional given"):
        RangeAccrualLeg(_schedule_main(), index).with_payment_day_counter(Actual360()).build()


def test_build_requires_payment_day_counter(index: Euribor) -> None:
    with pytest.raises(LibraryException, match="no payment day counter given"):
        RangeAccrualLeg(_schedule_main(), index).with_notionals(1.0).build()


def test_build_rejects_too_many_gearings(index: Euribor) -> None:
    with pytest.raises(LibraryException, match="too many gearings"):
        (
            RangeAccrualLeg(_schedule_main(), index)
            .with_notionals(1.0)
            .with_payment_day_counter(Actual360())
            .with_gearings([1.0] * 5)
            .build()
        )


def test_unset_triggers_fail(index: Euribor) -> None:
    """C++ passes Null<Rate>(); the lower < upper check then fails."""
    with pytest.raises(LibraryException, match="lowerTrigger"):
        (
            RangeAccrualLeg(_schedule_main(), index)
            .with_notionals(1.0)
            .with_payment_day_counter(Actual360())
            .with_observation_tenor(Period(1, TimeUnit.Months))
            .build()
        )


def test_coupon_rejects_inconsistent_observation_schedule(index: Euribor) -> None:
    start = Date.from_ymd(17, Month.March, 2025)
    end = Date.from_ymd(17, Month.September, 2025)
    wrong = Schedule.from_rule(
        start,
        Date.from_ymd(17, Month.December, 2025),
        Period(1, TimeUnit.Months),
        TARGET(),
        MF,
        MF,
        DateGeneration.Forward,
        False,
    )
    with pytest.raises(LibraryException, match="incompatible end date"):
        RangeAccrualFloatersCoupon(
            end, 1.0, index, start, end, 2, Actual360(), 1.0, 0.0, None, None, wrong, 0.03, 0.05
        )


def test_range_accrual_pricer_is_abstract() -> None:
    """C++ leaves swapletPrice pure virtual on RangeAccrualPricer."""
    with pytest.raises(TypeError):
        RangeAccrualPricer()  # type: ignore[abstract]


def test_caplet_and_floorlet_are_not_implemented(index: Euribor) -> None:
    pricer = _pricer("affine_smile_cs")
    for call in (pricer.caplet_price, pricer.caplet_rate, pricer.floorlet_price, pricer.floorlet_rate):
        with pytest.raises(LibraryException, match="not implemented"):
            call(0.04)
    del index
