"""Cross-validate ``LinearTsrPricer`` against C++ QuantLib v1.43.

# C++ parity:
#   ql/cashflows/lineartsrpricer.hpp + .cpp   (LinearTsrPricer, Settings,
#                                              VegaRatioHelper, PriceHelper)
#   ql/math/integrals/kronrodintegral.hpp/.cpp (GaussKronrodNonAdaptive)

Probe: ``v143/cf/lineartsr``.

The reference sweeps EIGHTEEN configurations of the pricer over FOUR CMS
coupons (in-advance, in-arrears with gearing + spread, capped, floored),
pinning for each the structural coupon fields, the market inputs that feed the
model (swap rate, annuity, ATM vol, payment discount, coupon-discount ratio),
and the pricer's own output (swaplet / caplet / floorlet rate and price, the
isolated convexity adjustment, plus the coupon-level ``rate()`` / ``amount()``).

``test_every_configuration_changes_the_answer`` then closes the loop the shared
brief asks for: every optional argument is additionally asserted to *move* the
result relative to the default configuration, so an argument that were accepted
and dropped would fail even if the reference numbers were regenerated.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import pytest

from pquantlib.cashflows.capped_floored_coupon import (
    CappedFlooredCmsCoupon,
    CappedFlooredCoupon,
)
from pquantlib.cashflows.cms_coupon import CmsCoupon
from pquantlib.cashflows.floating_rate_coupon import FloatingRateCoupon
from pquantlib.cashflows.ibor_coupon import IborCoupon
from pquantlib.cashflows.linear_tsr_pricer import (
    DEFAULT_LOWER_BOUND,
    DEFAULT_UPPER_BOUND,
    LinearTsrPricer,
    PriceHelper,
    Settings,
    Strategy,
    VegaRatioHelper,
)
from pquantlib.currencies.europe import EURCurrency
from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.daycounters.thirty_360 import Convention as Thirty360Convention
from pquantlib.daycounters.thirty_360 import Thirty360
from pquantlib.exceptions import LibraryException
from pquantlib.indexes.ibor.euribor import Euribor
from pquantlib.indexes.swap_index import SwapIndex
from pquantlib.math.integrals.integrator import Integrator
from pquantlib.math.integrals.kronrod import GaussKronrodNonAdaptive
from pquantlib.math.integrals.segment import SegmentIntegral
from pquantlib.math.solvers1d.brent import Brent
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.payoffs import OptionType
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.volatility.atm_smile_section import AtmSmileSection
from pquantlib.termstructures.volatility.smile_section import SmileSection
from pquantlib.termstructures.volatility.swaption.swaption_constant_vol import (
    SwaptionConstantVolatility,
)
from pquantlib.termstructures.volatility.swaption.swaption_volatility_structure import (
    SwaptionVolatilityStructure,
)
from pquantlib.termstructures.volatility.volatility_type import VolatilityType
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing.reference_reader import load as load_reference
from pquantlib.testing.tolerance import exact, loose, tight
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.date import Date
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit

_TODAY = Date.from_ymd(15, Month.January, 2024)
_FIXED_DC = Thirty360(Thirty360Convention.BondBasis)


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return load_reference("v143/cf/lineartsr")


@pytest.fixture(autouse=True)
def _eval_date() -> None:  # pyright: ignore[reportUnusedFunction]
    ObservableSettings().evaluation_date = _TODAY


# ---------------------------------------------------------------------------
# Market — mirrors the probe's makeMarket(): three DISTINCT flat curves so a
# mis-wired curve slot cannot hide.
# ---------------------------------------------------------------------------


def _flat(rate: float) -> FlatForward:
    return FlatForward.from_rate(_TODAY, rate, Actual365Fixed())


class _Market:
    def __init__(self) -> None:
        self.forwarding = _flat(0.030)
        self.discounting = _flat(0.025)
        self.coupon_discount = _flat(0.028)
        self.ibor = Euribor.six_months(self.forwarding)
        self.swap_index = SwapIndex(
            "EuriborSwapIsdaFixA",
            Period(10, TimeUnit.Years),
            self.ibor.fixing_days(),
            EURCurrency(),
            TARGET(),
            Period(1, TimeUnit.Years),
            BusinessDayConvention.Unadjusted,
            _FIXED_DC,
            self.ibor,
            self.discounting,
        )


@pytest.fixture
def market() -> _Market:
    return _Market()


def _const_vol(
    volatility: float,
    volatility_type: VolatilityType = VolatilityType.ShiftedLognormal,
    shift: float = 0.0,
) -> SwaptionConstantVolatility:
    return SwaptionConstantVolatility(
        reference_date=_TODAY,
        calendar=TARGET(),
        business_day_convention=BusinessDayConvention.Following,
        volatility=volatility,
        day_counter=Actual365Fixed(),
        volatility_type=volatility_type,
        shift=shift,
    )


# ---------------------------------------------------------------------------
# Coupons — mirrors the probe's makeCouponSpecs().
# ---------------------------------------------------------------------------


class _CouponSpec:
    def __init__(
        self,
        key: str,
        years_out: int,
        nominal: float,
        gearing: float,
        spread: float,
        in_arrears: bool,
        cap: float | None,
        floor: float | None,
        test_cap: float,
        test_floor: float,
    ) -> None:
        cal = TARGET()
        self.key = key
        self.start = cal.advance(_TODAY, years_out, TimeUnit.Years)
        self.end = cal.advance(self.start, 1, TimeUnit.Years)
        self.payment = self.end
        self.nominal = nominal
        self.gearing = gearing
        self.spread = spread
        self.in_arrears = in_arrears
        self.cap = cap
        self.floor = floor
        self.test_cap = test_cap
        self.test_floor = test_floor


_SPECS: list[_CouponSpec] = [
    _CouponSpec("advance", 5, 1.0, 1.0, 0.0, False, None, None, 0.045, 0.025),
    _CouponSpec("arrears", 7, 1.0e6, 1.5, 0.0025, True, None, None, 0.05, 0.02),
    _CouponSpec("capped", 3, 1.0e6, 1.0, 0.0, False, 0.045, None, 0.045, 0.025),
    _CouponSpec("floored", 12, 1.0e6, 1.0, 0.0, False, None, 0.030, 0.05, 0.030),
]
_SPEC_BY_KEY = {s.key: s for s in _SPECS}


def _build_coupon(spec: _CouponSpec, swap_index: SwapIndex) -> FloatingRateCoupon:
    if spec.cap is None and spec.floor is None:
        return CmsCoupon(
            spec.payment,
            spec.nominal,
            spec.start,
            spec.end,
            swap_index.fixing_days(),
            swap_index,
            spec.gearing,
            spec.spread,
            spec.start,
            spec.end,
            _FIXED_DC,
            spec.in_arrears,
        )
    return CappedFlooredCmsCoupon(
        spec.payment,
        spec.nominal,
        spec.start,
        spec.end,
        swap_index.fixing_days(),
        swap_index,
        spec.gearing,
        spec.spread,
        spec.cap,
        spec.floor,
        spec.start,
        spec.end,
        _FIXED_DC,
        spec.in_arrears,
    )


def _pricing_coupon(coupon: FloatingRateCoupon) -> CmsCoupon:
    """The CmsCoupon the pricer initialises on.

    ``CappedFlooredCmsCoupon`` is NOT a ``CmsCoupon`` — it wraps one, exactly
    as in C++, and ``CappedFlooredCoupon.rate()`` goes through the underlying.
    """
    base = coupon.underlying() if isinstance(coupon, CappedFlooredCoupon) else coupon
    assert isinstance(base, CmsCoupon)
    return base


# ---------------------------------------------------------------------------
# Configuration sweep — mirrors the probe's makeCases(), same keys.
# ---------------------------------------------------------------------------


class _Case:
    def __init__(
        self,
        key: str,
        settings: Settings,
        mean_reversion: float,
        volatility: SwaptionVolatilityStructure,
        with_coupon_discount: bool = True,
        integrator: Integrator | None = None,
    ) -> None:
        self.key = key
        self.settings = settings
        self.mean_reversion = mean_reversion
        self.volatility = volatility
        self.with_coupon_discount = with_coupon_discount
        self.integrator = integrator


def _cases() -> list[_Case]:
    mr = 0.01
    ln = _const_vol(0.20)
    normal = _const_vol(0.0060, VolatilityType.Normal)
    shifted = _const_vol(0.18, VolatilityType.ShiftedLognormal, 0.02)
    return [
        _Case("default", Settings(), mr, ln),
        _Case("rate_bound", Settings().with_rate_bound(0.01, 0.08), mr, ln),
        _Case("rate_bound_clamped", Settings().with_rate_bound(0.028, 0.045), mr, ln),
        _Case("rate_bound_narrow", Settings().with_rate_bound(0.005, 0.045), mr, ln),
        _Case("vega_ratio_1arg", Settings().with_vega_ratio(0.05), mr, ln),
        _Case("vega_ratio_3arg", Settings().with_vega_ratio(0.05, 0.005, 0.045), mr, ln),
        _Case("price_threshold_1arg", Settings().with_price_threshold(1.0e-6), mr, ln),
        _Case(
            "price_threshold_3arg",
            Settings().with_price_threshold(1.0e-6, 0.005, 0.045),
            mr,
            ln,
        ),
        _Case(
            "price_threshold_quirk",
            Settings().with_vega_ratio(1.0e-3).with_price_threshold(1.0e-6),
            mr,
            ln,
        ),
        _Case("bs_std_devs_1arg", Settings().with_bs_std_devs(4.0), mr, ln),
        _Case("bs_std_devs_3arg", Settings().with_bs_std_devs(4.0, 0.02, 0.05), mr, ln),
        _Case("mean_reversion_high", Settings(), 0.05, ln),
        _Case("mean_reversion_tiny", Settings(), 5.0e-5, ln),
        _Case("no_coupon_discount_curve", Settings(), mr, ln, with_coupon_discount=False),
        _Case("segment_integrator", Settings(), mr, ln, integrator=SegmentIntegral(50)),
        _Case("normal_vol_default_bounds", Settings(), mr, normal),
        _Case(
            "normal_vol_explicit_bounds",
            Settings().with_rate_bound(-0.01, 0.10),
            mr,
            normal,
        ),
        _Case("shifted_lognormal", Settings(), mr, shifted),
    ]


_CASE_BY_KEY = {c.key: c for c in _cases()}
_CASE_KEYS = [c.key for c in _cases()]
_SPEC_KEYS = [s.key for s in _SPECS]


def _price(case: _Case, spec: _CouponSpec, market: _Market) -> dict[str, float]:
    """Run one (configuration, coupon) pair and collect everything pinned."""
    coupon = _build_coupon(spec, market.swap_index)
    base = _pricing_coupon(coupon)
    pricer = LinearTsrPricer(
        case.volatility,
        SimpleQuote(case.mean_reversion),
        market.coupon_discount if case.with_coupon_discount else None,
        case.settings,
        case.integrator,
    )
    coupon.set_pricer(pricer)
    pricer.initialize(base)

    swap = market.swap_index.underlying_swap(base.fixing_date())
    swap_rate = swap.fair_rate()
    discount_payment = market.discounting.discount(base.date())
    coupon_payment = market.coupon_discount.discount(base.date()) if case.with_coupon_discount else 1.0
    swaplet_rate = pricer.swaplet_rate()

    return {
        "fixing_serial": base.fixing_date().serial_number(),
        "payment_serial": base.date().serial_number(),
        "accrual_start_serial": base.accrual_start_date().serial_number(),
        "accrual_end_serial": base.accrual_end_date().serial_number(),
        "accrual_period": base.accrual_period(),
        "nominal": base.nominal(),
        "gearing": base.gearing(),
        "spread": base.spread(),
        "swap_rate": swap_rate,
        "annuity": 1.0e4 * abs(swap.fixed_leg_bps()),
        "atm_vol": case.volatility.volatility(base.fixing_date(), market.swap_index.tenor(), swap_rate),
        "discount_payment": discount_payment,
        "coupon_discount_ratio": coupon_payment / discount_payment,
        "mean_reversion": pricer.mean_reversion(),
        "swaplet_rate": swaplet_rate,
        "swaplet_price": pricer.swaplet_price(),
        "convexity_adjustment": (swaplet_rate - base.spread()) / base.gearing() - swap_rate,
        "caplet_rate": pricer.caplet_rate(spec.test_cap),
        "caplet_price": pricer.caplet_price(spec.test_cap),
        "floorlet_rate": pricer.floorlet_rate(spec.test_floor),
        "floorlet_price": pricer.floorlet_price(spec.test_floor),
        "test_cap": spec.test_cap,
        "test_floor": spec.test_floor,
        "rate": coupon.rate(),
        "amount": coupon.amount(),
    }


# Structural + market fields: short chains of arithmetic over identical inputs.
_TIGHT_FIELDS = (
    "accrual_period",
    "nominal",
    "gearing",
    "spread",
    "swap_rate",
    "annuity",
    "atm_vol",
    "discount_payment",
    "coupon_discount_ratio",
    "mean_reversion",
    "test_cap",
    "test_floor",
)
# Model output: numerical quadrature (GaussKronrodNonAdaptive / SegmentIntegral)
# plus, for the VegaRatio strategy, a Brent root solve. LOOSE (1e-8) is the
# tier the project reserves for exactly that; the observed worst deviation
# across the whole sweep is ~8e-14, so the tier is not being leaned on.
_LOOSE_FIELDS = (
    "swaplet_rate",
    "swaplet_price",
    "convexity_adjustment",
    "caplet_rate",
    "caplet_price",
    "floorlet_rate",
    "floorlet_price",
    "rate",
    "amount",
)
_SERIAL_FIELDS = (
    "fixing_serial",
    "payment_serial",
    "accrual_start_serial",
    "accrual_end_serial",
)


@pytest.mark.parametrize("case_key", _CASE_KEYS)
@pytest.mark.parametrize("spec_key", _SPEC_KEYS)
def test_case(case_key: str, spec_key: str, market: _Market, cpp: dict[str, Any]) -> None:
    """Every pinned value of every coupon under every configuration."""
    got = _price(_CASE_BY_KEY[case_key], _SPEC_BY_KEY[spec_key], market)
    ref = cpp["cases"][case_key][spec_key]

    for field in _SERIAL_FIELDS:
        assert got[field] == ref[field], f"{case_key}/{spec_key}/{field}"
    for field in _TIGHT_FIELDS:
        tight(got[field], ref[field], reason=f"{case_key}/{spec_key}/{field}")
    for field in _LOOSE_FIELDS:
        loose(got[field], ref[field], reason=f"{case_key}/{spec_key}/{field}")


def test_setup_matches_probe(market: _Market, cpp: dict[str, Any]) -> None:
    """The market the test builds is the market the probe built."""
    setup = cpp["setup"]
    assert _TODAY.serial_number() == setup["evaluation_date_serial"]
    assert market.swap_index.family_name() == setup["swap_index_family"]
    assert market.swap_index.fixing_days() == setup["swap_index_fixing_days"]
    assert market.swap_index.exogenous_discount() == setup["swap_index_exogenous_discount"]
    exact(DEFAULT_LOWER_BOUND, setup["default_lower_bound"])
    exact(DEFAULT_UPPER_BOUND, setup["default_upper_bound"])


# ---------------------------------------------------------------------------
# The defect this subsystem is prone to: an optional argument that is accepted
# and then dropped. Each optional input must MOVE the answer.
# ---------------------------------------------------------------------------


def _signature(case_key: str, market: _Market) -> list[float]:
    out: list[float] = []
    for spec in _SPECS:
        priced = _price(_CASE_BY_KEY[case_key], spec, market)
        out.extend(priced[f] for f in _LOOSE_FIELDS)
    return out


def _max_rel_diff(a: list[float], b: list[float]) -> float:
    return max(abs(x - y) / max(abs(x), abs(y), 1e-30) for x, y in zip(a, b, strict=True))


# Configurations whose non-default argument must be visible in the answer.
_MUST_DIFFER = [
    "rate_bound",
    "rate_bound_clamped",
    "rate_bound_narrow",
    "vega_ratio_1arg",
    "vega_ratio_3arg",
    "bs_std_devs_1arg",
    "bs_std_devs_3arg",
    "mean_reversion_high",
    "mean_reversion_tiny",
    "no_coupon_discount_curve",
    "segment_integrator",
    "normal_vol_default_bounds",
    "shifted_lognormal",
]


@pytest.mark.parametrize("case_key", _MUST_DIFFER)
def test_every_configuration_changes_the_answer(case_key: str, market: _Market) -> None:
    """A non-default optional argument must have an observable consequence.

    This is the guard against the ``payment_lag``-class defect: an argument
    accepted, stored, and never passed on. Asserting only against the reference
    JSON would not catch it if the reference were ever regenerated from a
    broken port, so the *relative* effect is asserted independently here.
    """
    base = _signature("default", market)
    other = _signature(case_key, market)
    assert _max_rel_diff(base, other) > 1e-6, (
        f"{case_key} produced the same numbers as the default configuration — "
        "its argument is not reaching the model"
    )


def test_one_arg_and_three_arg_overloads_differ(market: _Market) -> None:
    """The 3-arg overloads' explicit bounds must reach ``adjusted_*_bound``.

    Only ``lower_rate_bound`` / ``upper_rate_bound`` distinguish the two
    overloads, so this is the direct test that they are not dropped.
    """
    for one, three in (
        ("vega_ratio_1arg", "vega_ratio_3arg"),
        ("bs_std_devs_1arg", "bs_std_devs_3arg"),
    ):
        assert _max_rel_diff(_signature(one, market), _signature(three, market)) > 1e-6, (
            f"{three} matched {one}: the explicit bounds were dropped"
        )


def test_normal_vol_default_bounds_flag_is_honoured(market: _Market) -> None:
    """``default_bounds`` drives the normal-vol ``min(lower, -upper)`` pull.

    Under normal volatility the lower bound is pulled to ``min(lower, -upper)``
    ONLY when the bounds were not set explicitly. The two cases therefore have
    to differ even though both are Normal with the same vol.
    """
    a = _signature("normal_vol_default_bounds", market)
    b = _signature("normal_vol_explicit_bounds", market)
    assert _max_rel_diff(a, b) > 1e-6


def test_rate_bound_clamped_zeroes_the_optionlets(market: _Market) -> None:
    """``optionlet_price``'s two early-outs.

    With bounds (0.028, 0.045) the probed cap 0.045 sits at/above the upper
    bound and the probed floor 0.025 below the lower one, so both optionlets
    are exactly zero — not merely small.
    """
    priced = _price(_CASE_BY_KEY["rate_bound_clamped"], _SPEC_BY_KEY["advance"], market)
    exact(priced["caplet_rate"], 0.0)
    exact(priced["caplet_price"], 0.0)
    exact(priced["floorlet_rate"], 0.0)
    exact(priced["floorlet_price"], 0.0)


# ---------------------------------------------------------------------------
# Upstream defects, reproduced on purpose (see the module docstring of
# linear_tsr_pricer.py and the probe header).
# ---------------------------------------------------------------------------


def test_price_threshold_strategy_is_inert_upstream(market: _Market) -> None:
    """``PriceThreshold`` degenerates to ``RateBound`` in C++ v1.43.

    Two upstream defects compound: ``optionletPrice`` passes ``vegaRatio_``
    (not ``priceThreshold_``) to ``strikeFromPrice``, and ``strikeFromPrice``
    hands Brent a guess equal to one of its own bracket ends, so the solve
    always throws and the blanket ``catch (...)`` restores the plain rate
    bound. The port must reproduce that, so these identities must hold — a
    "helpfully fixed" port breaks them.
    """
    default = _signature("default", market)
    narrow = _signature("rate_bound_narrow", market)
    three_arg = _signature("price_threshold_3arg", market)
    assert _max_rel_diff(default, _signature("price_threshold_1arg", market)) == 0.0
    assert _max_rel_diff(default, _signature("price_threshold_quirk", market)) == 0.0
    assert _max_rel_diff(narrow, three_arg) == 0.0
    # The 3-arg overload's BOUNDS are not part of the defect — they do reach
    # the model, so that overload must still move the answer.
    assert _max_rel_diff(default, three_arg) > 1e-6


def _pricer_smile_section(case: _Case, spec: _CouponSpec, market: _Market) -> SmileSection:
    """Rebuild the smile section the pricer builds internally.

    ``initialize`` takes ``swaption_volatility().smile_section(fixing, tenor)``
    and — because a constant-vol surface's flat section has no ATM level —
    re-anchors it with :class:`AtmSmileSection` at the swap rate. Reconstructed
    from public API here rather than reaching into the pricer's private state.
    """
    coupon = _build_coupon(spec, market.swap_index)
    base = _pricing_coupon(coupon)
    section = case.volatility.smile_section(base.fixing_date(), market.swap_index.tenor())
    assert math.isnan(section.atm_level())
    swap_rate = market.swap_index.underlying_swap(base.fixing_date()).fair_rate()
    return AtmSmileSection(base=section, atm=swap_rate)


def test_price_helper_solve_raises_on_bracket_end_guess(market: _Market) -> None:
    """The mechanism behind the inertness, isolated.

    ``PriceHelper`` itself is a perfectly good objective — it is the guess that
    C++ passes that makes the solve unusable. Verified directly so that the
    identity assertions above cannot be satisfied by a port that simply never
    builds the helper.
    """
    section = _pricer_smile_section(_CASE_BY_KEY["default"], _SPEC_BY_KEY["advance"], market)
    swap_rate = section.atm_level()
    helper = PriceHelper(section, OptionType.Call, 1.0e-6)
    # The objective is well defined and brackets a root...
    assert helper(swap_rate) > 0.0
    assert helper(DEFAULT_UPPER_BOUND) < 0.0
    # ...but C++ passes swap_rate as BOTH the guess and the lower bracket end.
    with pytest.raises(LibraryException):
        Brent().solve(helper, 1.0e-5, swap_rate, swap_rate, DEFAULT_UPPER_BOUND)


def test_vega_ratio_helper_is_used(market: _Market) -> None:
    """``VegaRatioHelper`` roots where the smile vega equals the target.

    Unlike ``PriceHelper`` the VegaRatio path passes a midpoint guess, so its
    solve genuinely runs; check the helper's own contract directly.
    """
    section = _pricer_smile_section(_CASE_BY_KEY["default"], _SPEC_BY_KEY["advance"], market)
    atm = section.atm_level()
    target = section.vega(atm) * 0.05
    helper = VegaRatioHelper(section, target)
    exact(helper(atm), section.vega(atm) - target)
    # vega decays away from the money, so the objective changes sign.
    assert helper(atm) > 0.0
    assert helper(DEFAULT_UPPER_BOUND) < 0.0


# ---------------------------------------------------------------------------
# Settings bookkeeping (the state the setters are supposed to install)
# ---------------------------------------------------------------------------


def test_settings_defaults() -> None:
    s = Settings()
    assert s.strategy == Strategy.RateBound
    exact(s.vega_ratio, 0.01)
    exact(s.price_threshold, 1.0e-8)
    exact(s.std_devs, 3.0)
    exact(s.lower_rate_bound, DEFAULT_LOWER_BOUND)
    exact(s.upper_rate_bound, DEFAULT_UPPER_BOUND)
    assert s.default_bounds is True


def test_settings_setters_install_the_documented_state() -> None:
    """Each ``with_*`` sets the strategy, its parameter and the bounds flag."""
    s = Settings().with_rate_bound(0.02, 0.5)
    assert s.strategy == Strategy.RateBound
    exact(s.lower_rate_bound, 0.02)
    exact(s.upper_rate_bound, 0.5)
    assert s.default_bounds is False

    s = Settings().with_vega_ratio(0.07)
    assert s.strategy == Strategy.VegaRatio
    exact(s.vega_ratio, 0.07)
    exact(s.lower_rate_bound, DEFAULT_LOWER_BOUND)
    exact(s.upper_rate_bound, DEFAULT_UPPER_BOUND)
    assert s.default_bounds is True

    s = Settings().with_vega_ratio(0.07, 0.01, 0.4)
    assert s.default_bounds is False
    exact(s.lower_rate_bound, 0.01)
    exact(s.upper_rate_bound, 0.4)

    s = Settings().with_price_threshold(2.0e-7)
    assert s.strategy == Strategy.PriceThreshold
    exact(s.price_threshold, 2.0e-7)
    assert s.default_bounds is True

    s = Settings().with_price_threshold(2.0e-7, 0.01, 0.4)
    assert s.default_bounds is False

    s = Settings().with_bs_std_devs(5.0)
    assert s.strategy == Strategy.BSStdDevs
    exact(s.std_devs, 5.0)
    assert s.default_bounds is True

    s = Settings().with_bs_std_devs(5.0, 0.01, 0.4)
    assert s.default_bounds is False
    exact(s.std_devs, 5.0)


def test_settings_setters_chain() -> None:
    """The C++ setters return ``*this``; the Python ones return ``self``."""
    s = Settings()
    assert s.with_rate_bound(0.01, 0.5) is s
    assert s.with_vega_ratio(0.02) is s
    assert s.with_price_threshold(1e-7) is s
    assert s.with_bs_std_devs(2.0) is s


def test_settings_partial_bounds_rejected() -> None:
    """C++ has no one-and-a-half-argument overload; neither do we."""
    with pytest.raises(LibraryException):
        Settings().with_vega_ratio(0.05, 0.01)


# ---------------------------------------------------------------------------
# Mean reversion — the other optional input
# ---------------------------------------------------------------------------


def test_set_mean_reversion_relinks_and_reprices(market: _Market, cpp: dict[str, Any]) -> None:
    """``set_mean_reversion`` must both re-link the quote AND move the price."""
    ref = cpp["set_mean_reversion"]
    coupon = _build_coupon(_SPEC_BY_KEY["advance"], market.swap_index)
    base = _pricing_coupon(coupon)
    pricer = LinearTsrPricer(_const_vol(0.20), SimpleQuote(0.01), market.coupon_discount)
    coupon.set_pricer(pricer)

    pricer.initialize(base)
    tight(pricer.mean_reversion(), ref["mean_reversion_before"])
    loose(pricer.swaplet_rate(), ref["swaplet_rate_before"])

    pricer.set_mean_reversion(SimpleQuote(0.08))
    pricer.initialize(base)
    tight(pricer.mean_reversion(), ref["mean_reversion_after"])
    loose(pricer.swaplet_rate(), ref["swaplet_rate_after"])

    assert ref["swaplet_rate_before"] != ref["swaplet_rate_after"]


def test_empty_mean_reversion_raises(market: _Market, cpp: dict[str, Any]) -> None:
    """C++ dereferences an empty ``Handle<Quote>``; ``None`` raises here too."""
    ref = cpp["empty_mean_reversion"]
    assert ref["initialize_throws"] is True
    assert ref["mean_reversion_throws"] is True

    coupon = _build_coupon(_SPEC_BY_KEY["advance"], market.swap_index)
    pricer = LinearTsrPricer(_const_vol(0.20), None, market.coupon_discount)
    with pytest.raises(LibraryException):
        pricer.initialize(_pricing_coupon(coupon))
    with pytest.raises(LibraryException):
        pricer.mean_reversion()


def test_non_cms_coupon_rejected(market: _Market, cpp: dict[str, Any]) -> None:
    """``initialize`` requires a ``CmsCoupon`` ("CMS coupon needed")."""
    assert cpp["non_cms_coupon"]["initialize_throws"] is True
    cal = TARGET()
    start = cal.advance(_TODAY, 5, TimeUnit.Years)
    end = cal.advance(start, 6, TimeUnit.Months)
    ibor_coupon = IborCoupon(end, 1.0, start, end, market.ibor.fixing_days(), market.ibor)
    pricer = LinearTsrPricer(_const_vol(0.20), SimpleQuote(0.01))
    with pytest.raises(LibraryException):
        pricer.initialize(ibor_coupon)


# ---------------------------------------------------------------------------
# Past fixing — initialize() skips the model entirely
# ---------------------------------------------------------------------------


def test_past_fixing(market: _Market, cpp: dict[str, Any]) -> None:
    """A determined fixing bypasses the replication in all three price members."""
    ref = cpp["past_fixing"]
    cal = TARGET()
    start = cal.advance(_TODAY, -3, TimeUnit.Months)
    end = cal.advance(start, 1, TimeUnit.Years)
    coupon = CmsCoupon(
        end,
        1.0e6,
        start,
        end,
        market.swap_index.fixing_days(),
        market.swap_index,
        1.2,
        0.001,
        start,
        end,
        _FIXED_DC,
    )
    fixing_date = coupon.fixing_date()
    market.swap_index.add_fixing(fixing_date, ref["recorded_fixing"], True)
    try:
        pricer = LinearTsrPricer(_const_vol(0.20), SimpleQuote(0.01), market.coupon_discount)
        coupon.set_pricer(pricer)
        pricer.initialize(coupon)

        assert fixing_date.serial_number() == ref["fixing_serial"]
        assert coupon.date().serial_number() == ref["payment_serial"]
        assert coupon.accrual_start_date().serial_number() == ref["accrual_start_serial"]
        assert coupon.accrual_end_date().serial_number() == ref["accrual_end_serial"]
        tight(coupon.accrual_period(), ref["accrual_period"])
        tight(coupon.nominal(), ref["nominal"])
        tight(coupon.gearing(), ref["gearing"])
        tight(coupon.spread(), ref["spread"])
        tight(market.discounting.discount(coupon.date()), ref["discount_payment"])
        # Determined fixing: no quadrature, no solver — a short chain of
        # arithmetic over the recorded fixing, so TIGHT applies here.
        tight(pricer.swaplet_rate(), ref["swaplet_rate"])
        tight(pricer.swaplet_price(), ref["swaplet_price"])
        tight(pricer.caplet_rate(ref["test_cap"]), ref["caplet_rate"])
        tight(pricer.caplet_price(ref["test_cap"]), ref["caplet_price"])
        tight(pricer.floorlet_rate(ref["test_floor"]), ref["floorlet_rate"])
        tight(pricer.floorlet_price(ref["test_floor"]), ref["floorlet_price"])
        tight(coupon.rate(), ref["rate"])
        tight(coupon.amount(), ref["amount"])
        # The recorded fixing genuinely bites: both optionlets are in the money.
        assert ref["caplet_rate"] > 0.0
        assert ref["floorlet_rate"] > 0.0
    finally:
        market.swap_index.clear_fixings()


# ---------------------------------------------------------------------------
# GaussKronrodNonAdaptive — the pricer's default integrator, ported alongside
# ---------------------------------------------------------------------------


def _integrands() -> dict[str, tuple[Callable[[float], float], float, float]]:
    return {
        "poly": (lambda x: x * x * x - 2.0 * x + 1.0, 0.0, 1.0),
        "gauss": (lambda x: math.exp(-x * x), -3.0, 3.0),
        "runge": (lambda x: 1.0 / (1.0 + 25.0 * x * x), -1.0, 1.0),
        "sqrt": (math.sqrt, 0.0, 1.0),
        "kink": (lambda x: abs(x - 0.3), 0.0, 1.0),
    }


@pytest.mark.parametrize("key", ["poly", "gauss", "runge", "sqrt", "kink"])
def test_gauss_kronrod_non_adaptive(key: str, cpp: dict[str, Any]) -> None:
    """Value, evaluation count and error estimate of the 10/21/43/87 cascade.

    The evaluation COUNT is what pins ``_rescale_error``: it says which rule
    the convergence test stopped at. A port with correct weights but a broken
    error rescaling matches the values and fails here.
    """
    ref = cpp["gauss_kronrod_non_adaptive"][key]
    f, a, b = _integrands()[key]
    gk = GaussKronrodNonAdaptive(1.0e-10, 5000, 1.0e-10)
    # LOOSE: quadrature. The `poly`/`gauss` cases are effectively exact; the
    # `runge`/`sqrt`/`kink` cases are the rule's own truncation error, which is
    # deterministic and identical in both languages.
    loose(gk(f, a, b), ref["value"])
    assert gk.number_of_evaluations() == ref["evaluations"]
    loose(gk.absolute_error(), ref["absolute_error"])
    loose(gk(f, b, a), ref["reversed"])


def test_gauss_kronrod_non_adaptive_accessors(cpp: dict[str, Any]) -> None:
    """Degenerate interval, relative-accuracy accessor and its setter's effect."""
    ref = cpp["gauss_kronrod_non_adaptive"]
    gk = GaussKronrodNonAdaptive(1.0e-10, 5000, 1.0e-10)
    f, _a, _b = _integrands()["gauss"]
    exact(gk(f, 1.0, 1.0), ref["degenerate_interval"])
    exact(gk.relative_accuracy(), ref["relative_accuracy"])

    gk.set_relative_accuracy(1.0e-3)
    exact(gk.relative_accuracy(), ref["relative_accuracy_after_set"])
    kink, ka, kb = _integrands()["kink"]
    # A slacker relative accuracy stops the cascade earlier — a different
    # answer from FEWER evaluations, which is the setter's observable effect.
    loose(gk(kink, ka, kb), ref["kink_loose_rel"])
    assert gk.number_of_evaluations() == ref["kink_loose_rel_evaluations"]
    assert ref["kink_loose_rel_evaluations"] < cpp["gauss_kronrod_non_adaptive"]["kink"]["evaluations"]


def test_default_integrator_is_the_non_adaptive_kronrod(market: _Market) -> None:
    """C++ installs ``GaussKronrodNonAdaptive(1e-10, 5000, 1e-10)`` when none given."""
    pricer = LinearTsrPricer(_const_vol(0.20), SimpleQuote(0.01))
    integrator = pricer.integrator()
    assert isinstance(integrator, GaussKronrodNonAdaptive)
    exact(integrator.absolute_accuracy(), 1.0e-10)
    assert integrator.max_evaluations() == 5000
    exact(integrator.relative_accuracy(), 1.0e-10)

    supplied = SegmentIntegral(50)
    with_custom = LinearTsrPricer(
        _const_vol(0.20), SimpleQuote(0.01), market.coupon_discount, Settings(), supplied
    )
    assert with_custom.integrator() is supplied


def test_coupon_discount_curve_is_stored(market: _Market) -> None:
    """The third constructor argument is retained, not silently dropped."""
    with_curve = LinearTsrPricer(_const_vol(0.20), SimpleQuote(0.01), market.coupon_discount)
    assert with_curve.coupon_discount_curve() is market.coupon_discount
    without = LinearTsrPricer(_const_vol(0.20), SimpleQuote(0.01))
    assert without.coupon_discount_curve() is None


def test_settings_argument_is_stored() -> None:
    """The fourth constructor argument is retained, not silently dropped."""
    s = Settings().with_bs_std_devs(4.0, 0.02, 0.05)
    pricer = LinearTsrPricer(_const_vol(0.20), SimpleQuote(0.01), None, s)
    assert pricer.settings() is s
    assert LinearTsrPricer(_const_vol(0.20), SimpleQuote(0.01)).settings().strategy == (Strategy.RateBound)


def test_with_rate_bound_no_args_still_clears_default_bounds() -> None:
    """``with_rate_bound()`` with no arguments is not a no-op.

    C++'s ``withRateBound`` defaults its two parameters to the very bounds a
    default-constructed ``Settings`` already carries, but it ALWAYS sets
    ``defaultBounds_ = false``. That flag is what suppresses the normal-vol
    ``min(lower, -upper)`` pull, so the call has an observable consequence even
    when both bounds keep their default value.
    """
    s = Settings().with_rate_bound()
    exact(s.lower_rate_bound, DEFAULT_LOWER_BOUND)
    exact(s.upper_rate_bound, DEFAULT_UPPER_BOUND)
    assert s.default_bounds is False
    assert Settings().default_bounds is True


def test_smile_section_vega_honours_the_discount_argument(market: _Market) -> None:
    """``SmileSection.vega``'s optional ``discount`` must not be dropped.

    ``blackFormulaVolDerivative`` multiplies by ``discount`` linearly, so a
    non-default discount scales the vega exactly — an argument that were
    accepted and ignored would leave the two values equal.
    """
    section = _pricer_smile_section(_CASE_BY_KEY["default"], _SPEC_BY_KEY["advance"], market)
    strike = section.atm_level() * 1.1
    undiscounted = section.vega(strike)
    assert undiscounted > 0.0
    tight(section.vega(strike, 0.5), 0.5 * undiscounted)
