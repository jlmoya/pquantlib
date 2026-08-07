"""Tests for the basis-models cluster (W8-A cluster c + v1.43 smile sections).

# C++ parity:
#   ql/experimental/basismodels/swaptioncfs.hpp
#   ql/experimental/basismodels/tenorswaptionvts.hpp
#   ql/experimental/basismodels/tenoroptionletvts.hpp

Two reference files are consumed:

- ``migration-harness/references/cluster/w8a.json`` — the original
  TenorSwaptionVTS.volatility / SwaptionCashFlows pins.
- ``migration-harness/references/v143/experimental/basismodels.json`` — the
  v1.43 pins for ``TenorOptionletSmileSection`` and
  ``TenorSwaptionSmileSection``: the full SmileSection surface (volatility over
  a strike grid, variance, atmLevel, minStrike, maxStrike, exerciseTime, shift,
  volatilityType) over both a flat-in-strike and a smiley base, plus the
  TwoParameterCorrelation values and the error paths.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from pquantlib.cashflows.coupon_pricer import BlackIborCouponPricer, set_coupon_pricer
from pquantlib.currencies.europe import EURCurrency
from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.daycounters.thirty_360 import Convention as T360Convention
from pquantlib.daycounters.thirty_360 import Thirty360
from pquantlib.exceptions import LibraryException
from pquantlib.exercise import EuropeanExercise
from pquantlib.experimental.basismodels.swaption_cfs import (
    SwaptionCashFlows,
    swaption_cashflows,
)
from pquantlib.experimental.basismodels.tenor_optionlet_vts import (
    CorrelationStructure,
    TenorOptionletSmileSection,
    TenorOptionletVTS,
    TwoParameterCorrelation,
)
from pquantlib.experimental.basismodels.tenor_swaption_vts import (
    TenorSwaptionSmileSection,
    TenorSwaptionVTS,
)
from pquantlib.indexes.ibor.euribor import Euribor
from pquantlib.indexes.ibor_index import IborIndex
from pquantlib.instruments.make_vanilla_swap import make_vanilla_swap
from pquantlib.instruments.swaption import Swaption
from pquantlib.math.interpolations.linear import LinearInterpolation
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.pricingengines.swap.discounting_swap_engine import DiscountingSwapEngine
from pquantlib.termstructures.volatility.optionlet.constant_optionlet_vol import (
    ConstantOptionletVolatility,
)
from pquantlib.termstructures.volatility.optionlet.optionlet_volatility_structure import (
    OptionletVolatilityStructure,
)
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
from pquantlib.time.calendars.null_calendar import NullCalendar
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.date import Date
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit


@pytest.fixture
def reference_data() -> dict[str, Any]:
    return load_reference("cluster/w8a")


@pytest.fixture(scope="module")
def bm_ref() -> dict[str, Any]:
    """The v1.43 basismodels smile-section reference."""
    return load_reference("v143/experimental/basismodels")


def _d(day: int, month: Month, year: int) -> Date:
    return Date.from_ymd(day, month, year)


_TODAY = _d(15, Month.January, 2024)


@pytest.fixture(autouse=True)
def _fixed_evaluation_date() -> Any:  # pyright: ignore[reportUnusedFunction]
    """Pin the global evaluation date to the probe's ``today``.

    The probe runs with ``Settings::instance().evaluationDate() = 2024-01-15``;
    index fixings are only forecastable for dates at or after it, so the tests
    must use the same date.
    """
    settings = ObservableSettings()
    previous = settings.evaluation_date
    settings.evaluation_date = _TODAY
    try:
        yield
    finally:
        settings.evaluation_date = previous


def _curve(rate: float = 0.03) -> FlatForward:
    return FlatForward.from_rate(_TODAY, rate, Actual365Fixed())


# ---------------------------------------------------------------------------
# TenorSwaptionVTS — cross-validated
# ---------------------------------------------------------------------------


def _tenor_swaption_vts(curve: FlatForward) -> TenorSwaptionVTS:
    """Reproduce the probe's TenorSwaptionVTS (base 6M -> target 3M)."""
    cal = TARGET()
    base6m = Euribor.six_months(curve)
    targ3m = Euribor.three_months(curve)
    base_vol = SwaptionConstantVolatility(
        reference_date=_TODAY,
        calendar=cal,
        business_day_convention=BusinessDayConvention.ModifiedFollowing,
        volatility=0.0090,
        day_counter=Actual365Fixed(),
        volatility_type=VolatilityType.Normal,
    )
    return TenorSwaptionVTS(
        base_vol, curve, base6m, targ3m,
        Period(1, TimeUnit.Years), Period(1, TimeUnit.Years),
        Thirty360(T360Convention.BondBasis), Thirty360(T360Convention.BondBasis),
    )


def test_tenor_swaption_vts_volatility(reference_data: dict[str, Any]) -> None:
    """Rescaled normal vol at (option, swap, strike) vs C++.

    LOOSE: affine-TSR rescaling (3 vanilla-swap fair rates + cash-flow
    decomposition).
    """
    vts = _tenor_swaption_vts(_curve())
    loose(vts.volatility(5.0, 10.0, 0.03), reference_data["tsvts_vol_5x10_atm"])
    loose(vts.volatility(2.0, 5.0, 0.03), reference_data["tsvts_vol_2x5_atm"])
    loose(vts.volatility(5.0, 10.0, 0.04), reference_data["tsvts_vol_5x10_otm"])


def test_tenor_swaption_vts_normal_type() -> None:
    vts = _tenor_swaption_vts(_curve())
    assert vts.volatility_type() == VolatilityType.Normal


def test_tenor_swaption_vts_smile_section() -> None:
    """The smile section round-trips through the rescaled volatility."""
    vts = _tenor_swaption_vts(_curve())
    section = vts.smile_section(5.0, 10.0)
    loose(section.volatility(0.03), vts.volatility(5.0, 10.0, 0.03))
    # atm_level is the final-tenor swap rate (positive on a 3% curve).
    assert section.atm_level() > 0.0


# ---------------------------------------------------------------------------
# SwaptionCashFlows — cross-validated
# ---------------------------------------------------------------------------


def _swaption_cfs(curve: FlatForward) -> SwaptionCashFlows:
    """Reproduce the probe's SwaptionCashFlows (5Y swap)."""
    cal = TARGET()
    euribor6m = Euribor.six_months(curve)
    exercise = _d(15, Month.January, 2026)
    swap = make_vanilla_swap(
        swap_tenor=Period(5, TimeUnit.Years),
        ibor_index=euribor6m,
        fixed_rate=0.03,
        effective_date=cal.advance(exercise, 2, TimeUnit.Days),
        fixed_leg_tenor=Period(1, TimeUnit.Years),
        fixed_leg_day_count=Thirty360(T360Convention.BondBasis),
        discount_curve=curve,
    )
    # the float leg needs a pricer for coupon.rate() in the decomposition.
    set_coupon_pricer(swap.floating_leg(), BlackIborCouponPricer())
    swap.set_pricing_engine(DiscountingSwapEngine(curve))
    swaption = Swaption(swap, EuropeanExercise(exercise))
    return SwaptionCashFlows(swaption, curve)


def test_swaption_cfs_weight_sums(reference_data: dict[str, Any]) -> None:
    """Annuity / float / fixed weight sums vs C++.

    TIGHT: deterministic cash-flow decomposition on a shared flat curve.
    """
    cfs = _swaption_cfs(_curve())
    tight(sum(cfs.annuity_weights()), reference_data["scfs_sum_annuity"])
    tight(sum(cfs.float_weights()), reference_data["scfs_sum_float"])
    tight(sum(cfs.fixed_weights()), reference_data["scfs_sum_fixed"])


def test_swaption_cfs_counts(reference_data: dict[str, Any]) -> None:
    cfs = _swaption_cfs(_curve())
    assert len(cfs.exercise_times()) == int(reference_data["scfs_num_exercise"])
    assert len(cfs.float_times()) == int(reference_data["scfs_num_float"])
    tight(cfs.float_times()[0], reference_data["scfs_first_float_time"])


def test_swaption_cashflows_free_function() -> None:
    """The free function matches the class."""
    curve = _curve()
    cal = TARGET()
    euribor6m = Euribor.six_months(curve)
    exercise = _d(15, Month.January, 2026)
    swap = make_vanilla_swap(
        swap_tenor=Period(5, TimeUnit.Years),
        ibor_index=euribor6m,
        fixed_rate=0.03,
        effective_date=cal.advance(exercise, 2, TimeUnit.Days),
        fixed_leg_tenor=Period(1, TimeUnit.Years),
        fixed_leg_day_count=Thirty360(T360Convention.BondBasis),
        discount_curve=curve,
    )
    set_coupon_pricer(swap.floating_leg(), BlackIborCouponPricer())
    swaption = Swaption(swap, EuropeanExercise(exercise))
    cfs = swaption_cashflows(swaption, curve)
    assert cfs.swaption() is swaption
    assert len(cfs.fixed_times()) == 5


# ---------------------------------------------------------------------------
# TenorOptionletVTS — structural
# ---------------------------------------------------------------------------


def test_tenor_optionlet_vts_volatility_positive() -> None:
    """The tenor-rescaled optionlet vol is finite and positive.

    Uses a NullCalendar base/target so the internally-generated sub-schedule
    never lands a fixing on a weekend (matches the probe setup).
    """
    curve = _curve()
    ncal = NullCalendar()
    base3m = IborIndex(
        "Base3M", Period(3, TimeUnit.Months), 2, EURCurrency(), ncal,
        BusinessDayConvention.ModifiedFollowing, False, Actual360(), curve,
    )
    targ6m = IborIndex(
        "Targ6M", Period(6, TimeUnit.Months), 2, EURCurrency(), ncal,
        BusinessDayConvention.ModifiedFollowing, False, Actual360(), curve,
    )
    base_vol = ConstantOptionletVolatility(
        business_day_convention=BusinessDayConvention.ModifiedFollowing,
        volatility=0.0070,
        calendar=ncal,
        day_counter=Actual365Fixed(),
        reference_date=_TODAY,
        volatility_type=VolatilityType.Normal,
    )
    times = np.array([0.0, 30.0])
    rho_inf = LinearInterpolation(times, np.array([0.3, 0.3]))
    beta = LinearInterpolation(times, np.array([0.1, 0.1]))
    corr = TwoParameterCorrelation(rho_inf, beta)

    vts = TenorOptionletVTS(base_vol, base3m, targ6m, corr)
    vol = vts.volatility(5.0, 0.03, True)
    assert vol > 0.0
    assert np.isfinite(vol)
    assert vts.volatility_type() == VolatilityType.Normal


def test_tenor_optionlet_vts_requires_freq_multiple() -> None:
    """baseFreq must be a multiple of targFreq (6M base, 3M target fails)."""
    curve = _curve()
    ncal = NullCalendar()
    base6m = IborIndex(
        "Base6M", Period(6, TimeUnit.Months), 2, EURCurrency(), ncal,
        BusinessDayConvention.ModifiedFollowing, False, Actual360(), curve,
    )
    targ3m = IborIndex(
        "Targ3M", Period(3, TimeUnit.Months), 2, EURCurrency(), ncal,
        BusinessDayConvention.ModifiedFollowing, False, Actual360(), curve,
    )
    base_vol = ConstantOptionletVolatility(
        business_day_convention=BusinessDayConvention.ModifiedFollowing,
        volatility=0.0070, calendar=ncal, day_counter=Actual365Fixed(),
        reference_date=_TODAY, volatility_type=VolatilityType.Normal,
    )
    times = np.array([0.0, 30.0])
    corr = TwoParameterCorrelation(
        LinearInterpolation(times, np.array([0.3, 0.3])),
        LinearInterpolation(times, np.array([0.1, 0.1])),
    )
    with pytest.raises(LibraryException, match="multiple of target"):
        TenorOptionletVTS(base_vol, base6m, targ3m, corr)


# ===========================================================================
# v1.43 smile sections — TenorOptionletSmileSection / TenorSwaptionSmileSection
# ===========================================================================
#
# The probe (migration-harness/cpp/probes/v143_experimental_basismodels) layers
# the two tenor VTSs on analytic quadratic base surfaces
#
#     optionlet:  vol(t, K)       = (a0 + aT*t)           + b*(K-k0) + c*(K-k0)^2
#     swaption:   vol(t, len, K)  = (a0 + aT*t + aL*len)  + b*(K-k0) + c*(K-k0)^2
#
# whose ``smileSectionImpl`` and ``volatilityImpl`` return the same closed form.
# The classes below are the exact Python mirrors of those fixtures. They are
# test scaffolding, not ports — the thing under test is the tenor rescaling.
#
# A flat base (b = c = 0) makes the strike transform invisible, so both a flat
# and a smiley parameterisation are pinned.


_QUAD_K0 = 0.03
_QUAD_MIN_STRIKE = -0.02
_QUAD_MAX_STRIKE = 0.12
_K_GRID: list[float] = [-0.01, 0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.08]
# C++ QL_MAX_REAL; PQuantLib spells the same "no bound" as +-inf.
_QL_MAX_REAL = 1.7976931348623157e308


def _quad_vol(a: float, b: float, c: float, strike: float) -> float:
    x = strike - _QUAD_K0
    return a + b * x + c * x * x


class _QuadSmileSection(SmileSection):
    """Mirror of the probe's ``QuadSmileSection``."""

    def __init__(self, t: float, a: float, b: float, c: float, day_counter: DayCounter) -> None:
        super().__init__(
            exercise_time=t,
            day_counter=day_counter,
            volatility_type=VolatilityType.Normal,
            shift=0.0,
        )
        self._a: float = a
        self._b: float = b
        self._c: float = c

    def min_strike(self) -> float:
        return _QUAD_MIN_STRIKE

    def max_strike(self) -> float:
        return _QUAD_MAX_STRIKE

    def atm_level(self) -> float:
        return math.nan  # C++ Null<Real>(); never consulted by these sections

    def _volatility_impl(self, strike: float) -> float:
        return _quad_vol(self._a, self._b, self._c, strike)


class _QuadOptionletVol(OptionletVolatilityStructure):
    """Mirror of the probe's ``QuadOptionletVol``."""

    def __init__(self, a0: float, a_t: float, b: float, c: float) -> None:
        super().__init__(
            business_day_convention=BusinessDayConvention.ModifiedFollowing,
            reference_date=_TODAY,
            calendar=NullCalendar(),
            day_counter=Actual365Fixed(),
        )
        self._a0: float = a0
        self._aT: float = a_t
        self._b: float = b
        self._c: float = c

    def max_date(self) -> Date:
        return Date.max_date()

    def min_strike(self) -> float:
        return _QUAD_MIN_STRIKE

    def max_strike(self) -> float:
        return _QUAD_MAX_STRIKE

    def volatility_type(self) -> VolatilityType:
        return VolatilityType.Normal

    def _volatility_impl(self, t: float, strike: float) -> float:
        return _quad_vol(self._a0 + self._aT * t, self._b, self._c, strike)


class _QuadSwaptionVol(SwaptionVolatilityStructure):
    """Mirror of the probe's ``QuadSwaptionVol``."""

    def __init__(self, a0: float, a_t: float, a_l: float, b: float, c: float) -> None:
        super().__init__(
            business_day_convention=BusinessDayConvention.ModifiedFollowing,
            reference_date=_TODAY,
            calendar=TARGET(),
            day_counter=Actual365Fixed(),
        )
        self._a0: float = a0
        self._aT: float = a_t
        self._aL: float = a_l
        self._b: float = b
        self._c: float = c
        self._max_swap_tenor: Period = Period(100, TimeUnit.Years)

    def max_date(self) -> Date:
        return Date.max_date()

    def min_strike(self) -> float:
        return _QUAD_MIN_STRIKE

    def max_strike(self) -> float:
        return _QUAD_MAX_STRIKE

    def max_swap_tenor(self) -> Period:
        return self._max_swap_tenor

    def volatility_type(self) -> VolatilityType:
        return VolatilityType.Normal

    def _a(self, t: float, length: float) -> float:
        return self._a0 + self._aT * t + self._aL * length

    def _volatility_impl(self, option_time: float, swap_length: float, strike: float) -> float:
        return _quad_vol(self._a(option_time, swap_length), self._b, self._c, strike)

    def smile_section(
        self,
        option_expiry: Period | Date | float,
        swap_tenor: Period | float,
        extrapolate: bool = False,
    ) -> SmileSection:
        del extrapolate
        assert isinstance(option_expiry, float)
        assert isinstance(swap_tenor, float)
        return _QuadSmileSection(
            option_expiry,
            self._a(option_expiry, swap_tenor),
            self._b,
            self._c,
            self.day_counter(),
        )


def _flat_optionlet_vol() -> _QuadOptionletVol:
    return _QuadOptionletVol(0.0070, 0.0, 0.0, 0.0)


def _smiley_optionlet_vol() -> _QuadOptionletVol:
    return _QuadOptionletVol(0.0060, 0.0002, -0.010, 0.30)


def _smiley_swaption_vol() -> _QuadSwaptionVol:
    return _QuadSwaptionVol(0.0080, 0.0002, 0.0001, -0.010, 0.30)


def _times(*values: float) -> npt.NDArray[np.float64]:
    return np.array(values, dtype=np.float64)


def _correlation(
    t0: float, t1: float, rho0: float, rho1: float, beta0: float, beta1: float
) -> TwoParameterCorrelation:
    xs = _times(t0, t1)
    return TwoParameterCorrelation(
        LinearInterpolation(xs, _times(rho0, rho1)),
        LinearInterpolation(xs, _times(beta0, beta1)),
    )


def _ibor(name: str, tenor: Period, curve: FlatForward) -> IborIndex:
    return IborIndex(
        name, tenor, 2, EURCurrency(), NullCalendar(),
        BusinessDayConvention.ModifiedFollowing, False, Actual360(), curve,
    )


def _check_strike_bound(actual: float, expected: float) -> None:
    """Compare a strike bound across the ``QL_MAX_REAL`` <-> ``inf`` convention.

    C++ spells "unbounded" as ``+-QL_MAX_REAL`` (a finite poison value);
    PQuantLib spells it ``+-math.inf`` (documented in FlatSmileSection). Adding
    a finite rate offset to either leaves it unchanged, so the two encode the
    same fact. Every finite bound is compared numerically.
    """
    if expected == _QL_MAX_REAL:
        assert actual == math.inf
    elif expected == -_QL_MAX_REAL:
        assert actual == -math.inf
    else:
        tight(actual, expected)


def _check_section(ref: dict[str, Any], prefix: str, section: SmileSection) -> None:
    """Pin every SmileSection observable of ``section`` against the probe.

    TIGHT throughout: the transformation is a deterministic composition of
    schedule generation, index fixings and (for the swaption side) three
    discounted-cash-flow swap valuations on a shared flat curve — no iteration,
    no root-finding, no cancellation.
    """
    exact(section.exercise_time(), ref[f"{prefix}_exercise_time"])
    exact(section.shift(), ref[f"{prefix}_shift"])
    assert (section.volatility_type() == VolatilityType.Normal) == bool(
        ref[f"{prefix}_vol_type_is_normal"]
    )
    tight(section.atm_level(), ref[f"{prefix}_atm_level"])
    _check_strike_bound(section.min_strike(), ref[f"{prefix}_min_strike"])
    _check_strike_bound(section.max_strike(), ref[f"{prefix}_max_strike"])

    expected_vols = ref[f"{prefix}_vols"]
    assert len(expected_vols) == len(_K_GRID)
    for strike, expected in zip(_K_GRID, expected_vols, strict=True):
        tight(section.volatility(strike), expected)

    tight(section.variance(0.03), ref[f"{prefix}_variance_atm"])
    tight(section.volatility(section.min_strike()), ref[f"{prefix}_vol_at_min_strike"])
    tight(section.volatility(section.max_strike()), ref[f"{prefix}_vol_at_max_strike"])


# ---------------------------------------------------------------------------
# TenorOptionletSmileSection
# ---------------------------------------------------------------------------


def _optionlet_vts(
    base_vol: OptionletVolatilityStructure,
    correlation: CorrelationStructure,
    *,
    targ_tenor: Period | None = None,
) -> TenorOptionletVTS:
    if targ_tenor is None:
        targ_tenor = Period(6, TimeUnit.Months)
    curve = _curve()
    return TenorOptionletVTS(
        base_vol,
        _ibor("Base3M", Period(3, TimeUnit.Months), curve),
        _ibor("Targ", targ_tenor, curve),
        correlation,
    )


def test_eval_date_matches_probe(bm_ref: dict[str, Any]) -> None:
    exact(float(_TODAY.serial_number()), float(bm_ref["meta_eval_date_serial"]))


@pytest.mark.parametrize("idx", [0, 1, 2])
def test_tenor_optionlet_smile_section_flat_base(bm_ref: dict[str, Any], idx: int) -> None:
    """Flat-in-strike Normal base: pins the v_i / correlation aggregation."""
    vts = _optionlet_vts(_flat_optionlet_vol(), _correlation(0.0, 30.0, 0.3, 0.3, 0.1, 0.1))
    prefix = f"optflat_t{idx}"
    option_time = float(bm_ref[f"{prefix}_option_time"])
    section = vts.smile_section(option_time, True)
    assert isinstance(section, TenorOptionletSmileSection)
    _check_section(bm_ref, prefix, section)
    # the VTS must route volatility()/blackVariance() through the section
    tight(vts.volatility(option_time, 0.03, True), bm_ref[f"{prefix}_vts_vol_atm"])
    tight(
        vts.black_variance(option_time, 0.03, True),
        bm_ref[f"{prefix}_vts_black_variance_atm"],
    )


def test_tenor_optionlet_vts_surface_metadata(bm_ref: dict[str, Any]) -> None:
    vts = _optionlet_vts(_flat_optionlet_vol(), _correlation(0.0, 30.0, 0.3, 0.3, 0.1, 0.1))
    assert (vts.volatility_type() == VolatilityType.Normal) == bool(
        bm_ref["optflat_vts_vol_type_is_normal"]
    )
    _check_strike_bound(vts.min_strike(), bm_ref["optflat_vts_min_strike"])
    _check_strike_bound(vts.max_strike(), bm_ref["optflat_vts_max_strike"])
    exact(
        float(vts.max_date().serial_number()),
        float(bm_ref["optflat_vts_max_date_serial"]),
    )


@pytest.mark.parametrize("idx", [0, 1, 2])
def test_tenor_optionlet_smile_section_smiley_base(bm_ref: dict[str, Any], idx: int) -> None:
    """Smiley Normal base: pins the per-FRA strike transform as well."""
    vts = _optionlet_vts(_smiley_optionlet_vol(), _correlation(0.0, 30.0, 0.3, 0.3, 0.1, 0.1))
    prefix = f"optsm_t{idx}"
    option_time = float(bm_ref[f"{prefix}_option_time"])
    _check_section(bm_ref, prefix, vts.smile_section(option_time, True))
    tight(vts.volatility(option_time, 0.03, True), bm_ref[f"{prefix}_vts_vol_atm"])


@pytest.mark.parametrize("idx", [0, 1, 2])
def test_tenor_optionlet_smile_section_term_correlation(
    bm_ref: dict[str, Any], idx: int
) -> None:
    """rhoInf and beta both vary with FRA start time."""
    vts = _optionlet_vts(_smiley_optionlet_vol(), _correlation(0.0, 30.0, 0.2, 0.8, 0.05, 0.50))
    option_time = float(bm_ref[f"optsm_t{idx}_option_time"])
    _check_section(bm_ref, f"optsmc_t{idx}", vts.smile_section(option_time, True))


def test_tenor_optionlet_smile_section_four_fras(bm_ref: dict[str, Any]) -> None:
    """A 12M target over a 3M base: four FRAs, six correlation cross terms."""
    vts = _optionlet_vts(
        _smiley_optionlet_vol(),
        _correlation(0.0, 30.0, 0.2, 0.8, 0.05, 0.50),
        targ_tenor=Period(12, TimeUnit.Months),
    )
    section = vts.smile_section(5.0, True)
    assert isinstance(section, TenorOptionletSmileSection)
    assert len(section.v()) == 4
    assert len(section.base_fixing_dates()) == 4
    _check_section(bm_ref, "optsm4_t0", section)


def test_tenor_optionlet_smile_section_target_calendar(bm_ref: dict[str, Any]) -> None:
    """Euribor3M/6M on TARGET: the base schedule rolls off business days.

    The option time is 4.0, not 5.0: TenorOptionletSmileSection turns optionTime
    into an exercise date by raw day arithmetic with no calendar adjustment
    (tenoroptionletvts.cpp:57-60), and 2024-01-15 + 1825d is a Saturday, which
    ``targIndex_->fixing`` rejects. C++ v1.43 raises there and so does the port
    (``optsmcal_weekend_exercise_throws``).
    """
    curve = _curve()
    vts = TenorOptionletVTS(
        _smiley_optionlet_vol(),
        Euribor.three_months(curve),
        Euribor.six_months(curve),
        _correlation(0.0, 30.0, 0.2, 0.8, 0.05, 0.50),
    )
    _check_section(bm_ref, "optsmcal_t0", vts.smile_section(4.0, True))

    assert bm_ref["optsmcal_weekend_exercise_throws"] is True
    with pytest.raises(LibraryException):
        vts.smile_section(5.0, True)


def test_tenor_optionlet_vts_frequency_precondition(bm_ref: dict[str, Any]) -> None:
    assert bm_ref["opt_ctor_throws_on_bad_frequency"] is True
    curve = _curve()
    with pytest.raises(LibraryException, match="multiple of target"):
        TenorOptionletVTS(
            _smiley_optionlet_vol(),
            _ibor("Base6M", Period(6, TimeUnit.Months), curve),
            _ibor("Targ3M", Period(3, TimeUnit.Months), curve),
            _correlation(0.0, 30.0, 0.3, 0.3, 0.1, 0.1),
        )


# ---------------------------------------------------------------------------
# TwoParameterCorrelation
# ---------------------------------------------------------------------------


_CORR_POINTS: list[tuple[float, float]] = [
    (0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (2.5, 7.5),
    (5.0, 5.0), (5.0, 5.25), (7.5, 2.5), (10.0, 0.0),
]


def test_two_parameter_correlation_values(bm_ref: dict[str, Any]) -> None:
    corr = _correlation(0.0, 10.0, 0.2, 0.8, 0.05, 0.50)
    for (s1, s2), expected in zip(_CORR_POINTS, bm_ref["corr_values"], strict=True):
        tight(corr(s1, s2), expected)


def test_two_parameter_correlation_does_not_extrapolate(bm_ref: dict[str, Any]) -> None:
    """``(*rhoInf_)(start1)`` uses the default ``allowExtrapolation=false``.

    Only ``start1`` is looked up in the parameter grid; ``start2`` enters solely
    through ``|start2 - start1|`` and may be arbitrarily far outside it.
    """
    corr = _correlation(0.0, 10.0, 0.2, 0.8, 0.05, 0.50)
    assert bm_ref["corr_throws_below_grid"] is True
    with pytest.raises(LibraryException):
        corr(-1.0, 0.0)
    assert bm_ref["corr_throws_above_grid"] is True
    with pytest.raises(LibraryException):
        corr(10.5, 0.0)
    assert bm_ref["corr_throws_on_second_arg"] is False
    tight(corr(5.0, 1.0e6), bm_ref["corr_far_second_arg_value"])


def test_tenor_optionlet_smile_section_off_grid_correlation(bm_ref: dict[str, Any]) -> None:
    """A correlation grid that does not cover the FRA start times must raise."""
    assert bm_ref["corr_section_throws_off_grid"] is True
    vts = _optionlet_vts(_smiley_optionlet_vol(), _correlation(0.0, 1.0, 0.2, 0.8, 0.05, 0.50))
    with pytest.raises(LibraryException):
        vts.smile_section(5.0, True).volatility(0.03)


# ---------------------------------------------------------------------------
# TenorSwaptionSmileSection
# ---------------------------------------------------------------------------


def _swaption_vts(
    base_vol: SwaptionVolatilityStructure,
    *,
    reverse: bool = False,
    targ_fixed_freq: Period | None = None,
    targ_fixed_dc: DayCounter | None = None,
) -> TenorSwaptionVTS:
    curve = _curve()
    base6m = Euribor.six_months(curve)
    targ3m = Euribor.three_months(curve)
    base_index, targ_index = (targ3m, base6m) if reverse else (base6m, targ3m)
    return TenorSwaptionVTS(
        base_vol, curve, base_index, targ_index,
        Period(1, TimeUnit.Years),
        targ_fixed_freq if targ_fixed_freq is not None else Period(1, TimeUnit.Years),
        Thirty360(T360Convention.BondBasis),
        targ_fixed_dc if targ_fixed_dc is not None else Thirty360(T360Convention.BondBasis),
    )


def _flat_swaption_vol() -> SwaptionConstantVolatility:
    return SwaptionConstantVolatility(
        reference_date=_TODAY,
        calendar=TARGET(),
        business_day_convention=BusinessDayConvention.ModifiedFollowing,
        volatility=0.0090,
        day_counter=Actual365Fixed(),
        volatility_type=VolatilityType.Normal,
    )


@pytest.mark.parametrize("idx", [0, 1])
def test_tenor_swaption_smile_section_flat_base(bm_ref: dict[str, Any], idx: int) -> None:
    """Flat-in-strike Normal base: pins annuityScaling*(1+lambda) and swapRateFinl.

    ``ConstantSwaptionVolatility::smileSectionImpl`` DOES forward
    ``volatilityType_``/``shift_`` (swaptionconstantvol.cpp:82-96), so the
    ``volatility(K, Normal, 0.0)`` read short-circuits and no type conversion
    is involved. (Its optionlet twin does not — see
    ``test_constant_optionlet_base_is_a_known_divergence``.)
    """
    vts = _swaption_vts(_flat_swaption_vol())
    prefix = f"swpflat_c{idx}"
    option_time = float(bm_ref[f"{prefix}_option_time"])
    swap_length = float(bm_ref[f"{prefix}_swap_length"])
    section = vts.smile_section(option_time, swap_length, True)
    assert isinstance(section, TenorSwaptionSmileSection)
    _check_section(bm_ref, prefix, section)
    tight(
        vts.volatility(option_time, swap_length, 0.03, True),
        bm_ref[f"{prefix}_vts_vol_atm"],
    )
    tight(
        vts.black_variance(option_time, swap_length, 0.03, True),
        bm_ref[f"{prefix}_vts_black_variance_atm"],
    )


def test_tenor_swaption_vts_surface_metadata(bm_ref: dict[str, Any]) -> None:
    vts = _swaption_vts(_flat_swaption_vol())
    assert (vts.volatility_type() == VolatilityType.Normal) == bool(
        bm_ref["swpflat_vts_vol_type_is_normal"]
    )
    _check_strike_bound(vts.min_strike(), bm_ref["swpflat_vts_min_strike"])
    _check_strike_bound(vts.max_strike(), bm_ref["swpflat_vts_max_strike"])
    exact(
        float(vts.max_date().serial_number()),
        float(bm_ref["swpflat_vts_max_date_serial"]),
    )
    assert vts.max_swap_tenor().length == bm_ref["swpflat_vts_max_swap_tenor_years"]


@pytest.mark.parametrize("idx", [0, 1, 2])
def test_tenor_swaption_smile_section_smiley_base(bm_ref: dict[str, Any], idx: int) -> None:
    """Smiley Normal base: pins the affine strike transform too."""
    vts = _swaption_vts(_smiley_swaption_vol())
    prefix = f"swpsm_c{idx}"
    option_time = float(bm_ref[f"{prefix}_option_time"])
    swap_length = float(bm_ref[f"{prefix}_swap_length"])
    _check_section(bm_ref, prefix, vts.smile_section(option_time, swap_length, True))
    tight(
        vts.volatility(option_time, swap_length, 0.03, True),
        bm_ref[f"{prefix}_vts_vol_atm"],
    )


def test_tenor_swaption_smile_section_target_fixed_leg_conventions(
    bm_ref: dict[str, Any],
) -> None:
    """targFixedFreq / targFixedDC feed finlSwap only: annuityScaling + atmLevel."""
    vts = _swaption_vts(
        _smiley_swaption_vol(),
        targ_fixed_freq=Period(6, TimeUnit.Months),
        targ_fixed_dc=Actual360(),
    )
    _check_section(bm_ref, "swpsm2_c0", vts.smile_section(5.0, 10.0, True))


def test_tenor_swaption_smile_section_reverse_basis(bm_ref: dict[str, Any]) -> None:
    """Base 3M / target 6M — the other basis direction."""
    vts = _swaption_vts(_smiley_swaption_vol(), reverse=True)
    _check_section(bm_ref, "swpsm3_c0", vts.smile_section(5.0, 10.0, True))


@pytest.mark.parametrize(("idx", "swap_length"), [(0, 7.5), (1, 7.0), (2, 7.9999)])
def test_tenor_swaption_smile_section_truncates_swap_length_to_years(
    bm_ref: dict[str, Any], idx: int, swap_length: float
) -> None:
    """``((BigInteger)swapLength * 12.0) * Months`` truncates to whole YEARS.

    The cast binds tighter than the multiplication (tenorswaptionvts.cpp:48-49),
    so 7.5 and 7.9999 both build an 84-month swap, not 90 / 96. Only the base
    smile section itself sees the raw swap length. The three cases therefore
    share atmLevel/minStrike/maxStrike exactly and differ only through the base
    surface's length dependence — which is what makes this discriminating: a
    port computing ``int(7.5 * 12)`` gets a different swap and a different
    atmLevel.
    """
    vts = _swaption_vts(_smiley_swaption_vol())
    _check_section(bm_ref, f"swpsmlen_c{idx}", vts.smile_section(5.0, swap_length, True))
    # all three share the 7y swap, hence the identical ATM level
    exact(bm_ref[f"swpsmlen_c{idx}_atm_level"], bm_ref["swpsmlen_c1_atm_level"])


def test_tenor_swaption_smile_section_affine_parameters(bm_ref: dict[str, Any]) -> None:
    """The published (lambda, annuityScaling, swap rates) reproduce the section.

    Cross-checks the C++-pinned observables against each other:
    ``atmLevel == swapRateFinl``, ``minStrike - baseMin == swapRateTarg -
    swapRateBase``, and ``vol(K) == A * volBase((K - C) / A)``.
    """
    vts = _swaption_vts(_smiley_swaption_vol())
    section = vts.smile_section(5.0, 10.0, True)
    assert isinstance(section, TenorSwaptionSmileSection)

    tight(section.atm_level(), bm_ref["swpsm_c0_atm_level"])
    tight(section.swap_rate_finl(), bm_ref["swpsm_c0_atm_level"])
    tight(
        section.swap_rate_targ() - section.swap_rate_base(),
        bm_ref["swpsm_c0_min_strike"] - _QUAD_MIN_STRIKE,
    )

    a = section.annuity_scaling() * (1.0 + section.lambda_())
    c = section.swap_rate_targ() - (1.0 + section.lambda_()) * section.swap_rate_base()
    for strike, expected in zip(_K_GRID, bm_ref["swpsm_c0_vols"], strict=True):
        tight(a * section.base_smile_section().volatility((strike - c) / a), expected)


# ---------------------------------------------------------------------------
# Known divergence: ConstantOptionletVolatility as a TenorOptionletVTS base
# ---------------------------------------------------------------------------


def test_constant_optionlet_base_is_a_known_divergence(bm_ref: dict[str, Any]) -> None:
    """C++ v1.43 CANNOT layer TenorOptionletVTS on ConstantOptionletVolatility.

    ``ConstantOptionletVolatility::smileSectionImpl`` (constantoptionletvol.cpp:
    74-84) builds its ``FlatSmileSection`` without forwarding
    ``volatilityType()`` / ``displacement()``, so the section always reports
    ShiftedLognormal with a ``Null<Real>`` atmLevel even when the surface was
    constructed with ``Normal``. ``TenorOptionletSmileSection::volatilityImpl``
    then asks it for ``volatility(K, Normal, 0.0)``, the types disagree, and
    ``SmileSection::volatility`` (smilesection.cpp:119-143) aborts on the null
    atmLevel. The probe records both facts.

    PQuantLib returns a number instead, because its TenorOptionletSmileSection
    reads the base vol through ``baseVTS.volatility(fixingDate, K)`` — there is
    no ``OptionletVolatilityStructure.smile_section`` to mis-type. Closing this
    gap needs ``smile_section`` on OptionletVolatilityStructure plus the
    type-converting ``SmileSection.volatility(strike, type, shift)`` overload,
    both outside this module. Pinned here so the divergence is explicit and so
    this test fails loudly the day that work lands.
    """
    # C++: the surface knows its type, its smile section does not.
    assert bm_ref["cov_normal_surface_is_normal"] == 1
    assert bm_ref["cov_normal_section_is_normal"] == 0
    assert bm_ref["cov_sln_surface_is_normal"] == 0
    assert bm_ref["cov_sln_section_is_normal"] == 0
    # C++: either way, asking the tenor section for a vol throws.
    assert bm_ref["cov_normal_section_vol_throws"] is True
    assert bm_ref["cov_sln_section_vol_throws"] is True

    # PQuantLib: finite and positive (see the docstring).
    curve = _curve()
    base_vol = ConstantOptionletVolatility(
        business_day_convention=BusinessDayConvention.ModifiedFollowing,
        volatility=0.0070,
        calendar=NullCalendar(),
        day_counter=Actual365Fixed(),
        reference_date=_TODAY,
        volatility_type=VolatilityType.Normal,
    )
    vts = TenorOptionletVTS(
        base_vol,
        _ibor("Base3M", Period(3, TimeUnit.Months), curve),
        _ibor("Targ6M", Period(6, TimeUnit.Months), curve),
        _correlation(0.0, 30.0, 0.3, 0.3, 0.1, 0.1),
    )
    vol = vts.smile_section(5.0, True).volatility(0.03)
    assert math.isfinite(vol)
    assert vol > 0.0
