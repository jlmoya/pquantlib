"""Cross-validate the Black overnight-coupon pricers against C++ QuantLib v1.43.

Probe:     ``migration-harness/cpp/probes/v143_cf_blackovernight/probe.cpp``
Reference: ``v143/cf/blackovernight``

Two scenarios, both on SOFR:

``fwd_*``
    evaluation date 15-Jan-2024, coupon 1-Apr-2024..1-Jul-2024 projected off a
    flat 3% curve, so every fixing is in the future and the Black / Bachelier
    branch runs.
``past_*``
    evaluation date 1-Mar-2024, coupon 1-Feb-2024..1-Mar-2024 with a complete
    fixing history, so the optionlet collapses to its intrinsic value and no
    volatility surface is needed at all.

The price alone is not enough to prove the port: the standard deviation that
feeds it is a four-step chain (fixing start/end times off the vol surface, the
Lyashenko/Mercurio damped time, sigma, ``sigma*sqrt(T)``), and two errors in it
can cancel. Every step is therefore compared separately, recovered from the
public API only — ``effective_caplet_volatility() == stdDev / sqrt(t_eff)`` makes
the standard deviation observable without reaching into the pricer.

Tolerance: TIGHT throughout. Every quantity is a short chain of arithmetic,
``log``/``exp``/``erf`` over identical inputs; nothing iterates.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from typing import Any

import pytest

from pquantlib.cashflows.black_overnight_indexed_coupon_pricer import (
    BlackAveragingOvernightIndexedCouponPricer,
    BlackCompoundingOvernightIndexedCouponPricer,
)
from pquantlib.cashflows.capped_floored_coupon import CappedFlooredOvernightIndexedCoupon
from pquantlib.cashflows.overnight_indexed_coupon import OvernightIndexedCoupon
from pquantlib.cashflows.overnight_indexed_coupon_pricer import (
    ArithmeticAveragedOvernightIndexedCouponPricer,
    CompoundingOvernightIndexedCouponPricer,
    OvernightIndexedCouponPricer,
)
from pquantlib.cashflows.rate_averaging import RateAveraging
from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exceptions import LibraryException
from pquantlib.indexes.ibor.sofr import Sofr
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.termstructures.volatility.optionlet.constant_optionlet_vol import (
    ConstantOptionletVolatility,
)
from pquantlib.termstructures.volatility.optionlet.optionlet_volatility_structure import (
    OptionletVolatilityStructure,
)
from pquantlib.termstructures.volatility.volatility_type import VolatilityType
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing import reference_reader, tolerance
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.null_calendar import NullCalendar
from pquantlib.time.date import Date
from pquantlib.time.month import Month

# --- probe constants (kept byte-for-byte in step with probe.cpp) ------------

_FWD_TODAY = Date.from_ymd(15, Month.January, 2024)
_FWD_START = Date.from_ymd(1, Month.April, 2024)
_FWD_END = Date.from_ymd(1, Month.July, 2024)
_NOMINAL = 1_000_000.0
_GEARING = 2.0
_SPREAD = 0.001
_CAP = 0.075
_FLOOR = 0.045
_FLAT_RATE = 0.03
_LN_VOL = 0.20
_DISPLACEMENT = 0.01
_NORMAL_VOL = 0.0080

_PAST_TODAY = Date.from_ymd(1, Month.March, 2024)
_PAST_START = Date.from_ymd(1, Month.February, 2024)
_PAST_END = Date.from_ymd(1, Month.March, 2024)


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("v143/cf/blackovernight")


def _const_vol(vol: float, vol_type: VolatilityType, shift: float) -> ConstantOptionletVolatility:
    return ConstantOptionletVolatility(
        business_day_convention=BusinessDayConvention.Following,
        volatility=vol,
        calendar=NullCalendar(),
        day_counter=Actual365Fixed(),
        reference_date=_FWD_TODAY,
        volatility_type=vol_type,
        displacement=shift,
    )


@pytest.fixture
def fwd_index() -> Iterator[Sofr]:
    """SOFR on a flat 3% curve, evaluated before the coupon starts."""
    ObservableSettings().evaluation_date = _FWD_TODAY
    idx = Sofr()
    idx.clear_fixings()
    curve = FlatForward.from_rate(_FWD_TODAY, _FLAT_RATE, Actual365Fixed())
    yield Sofr(curve)
    idx.clear_fixings()
    ObservableSettings().evaluation_date = None


@pytest.fixture
def past_index() -> Iterator[Sofr]:
    """SOFR with the full 0.05 + 0.0001*i fixing ramp, evaluated after the coupon."""
    ObservableSettings().evaluation_date = _PAST_TODAY
    idx = Sofr()
    idx.clear_fixings()
    template = OvernightIndexedCoupon(_PAST_END, _NOMINAL, _PAST_START, _PAST_END, idx)
    for i, d in enumerate(template.fixing_dates()):
        idx.add_fixing(d, 0.05 + 0.0001 * i, force_overwrite=True)
    yield idx
    idx.clear_fixings()
    ObservableSettings().evaluation_date = None


@pytest.fixture
def ln_vol() -> ConstantOptionletVolatility:
    return _const_vol(_LN_VOL, VolatilityType.ShiftedLognormal, _DISPLACEMENT)


@pytest.fixture
def normal_vol() -> ConstantOptionletVolatility:
    return _const_vol(_NORMAL_VOL, VolatilityType.Normal, 0.0)


def _fwd_coupon(index: Sofr, averaging: RateAveraging) -> OvernightIndexedCoupon:
    return OvernightIndexedCoupon(
        _FWD_END,
        _NOMINAL,
        _FWD_START,
        _FWD_END,
        index,
        _GEARING,
        _SPREAD,
        averaging_method=averaging,
    )


def _past_coupon(index: Sofr, averaging: RateAveraging) -> OvernightIndexedCoupon:
    return OvernightIndexedCoupon(
        _PAST_END,
        _NOMINAL,
        _PAST_START,
        _PAST_END,
        index,
        _GEARING,
        _SPREAD,
        averaging_method=averaging,
    )


def _eff_cap() -> float:
    return (_CAP - _SPREAD) / _GEARING


def _eff_floor() -> float:
    return (_FLOOR - _SPREAD) / _GEARING


# --- coupon geometry -------------------------------------------------------


def _check_coupon_shape(coupon: OvernightIndexedCoupon, ref: dict[str, Any]) -> None:
    tolerance.tight(coupon.nominal(), ref["nominal"])
    assert coupon.accrual_start_date().serial_number() == ref["accrual_start_serial"]
    assert coupon.accrual_end_date().serial_number() == ref["accrual_end_serial"]
    assert coupon.date().serial_number() == ref["payment_serial"]
    tolerance.tight(coupon.accrual_period(), ref["accrual_period"])
    tolerance.tight(coupon.gearing(), ref["gearing"])
    tolerance.tight(coupon.spread(), ref["spread"])
    assert len(coupon.fixing_dates()) == ref["n_fixings"]
    assert coupon.fixing_dates()[0].serial_number() == ref["first_fixing_serial"]
    assert coupon.fixing_dates()[-1].serial_number() == ref["last_fixing_serial"]
    assert coupon.fixing_date().serial_number() == ref["fixing_date_serial"]
    assert coupon.value_dates()[0].serial_number() == ref["first_value_date_serial"]
    assert coupon.value_dates()[-1].serial_number() == ref["last_value_date_serial"]
    tolerance.tight(coupon.dt()[0], ref["first_dt"])
    tolerance.tight(coupon.dt()[-1], ref["last_dt"])


def test_fwd_coupon_shape(fwd_index: Sofr, cpp: dict[str, Any]) -> None:
    _check_coupon_shape(_fwd_coupon(fwd_index, RateAveraging.Compound), cpp["coupon"])


def test_past_coupon_shape(past_index: Sofr, cpp: dict[str, Any]) -> None:
    _check_coupon_shape(_past_coupon(past_index, RateAveraging.Compound), cpp["past_coupon"])


def test_effective_cap_and_floor(cpp: dict[str, Any]) -> None:
    """``(level - spread) / gearing`` — the strike the optionlets are struck at."""
    tolerance.tight(_eff_cap(), cpp["eff_cap"])
    tolerance.tight(_eff_floor(), cpp["eff_floor"])


# --- the standard-deviation chain ------------------------------------------


def _check_std_dev_chain(
    coupon: OvernightIndexedCoupon,
    vol: OptionletVolatilityStructure,
    effective_caplet_volatility: float,
    ref: dict[str, Any],
) -> None:
    """Pin every input of the Black standard deviation, not just the price.

    ``effective_caplet_volatility() == stdDev / sqrt(t_eff)`` makes the standard
    deviation observable through the public API, and ``(stdDev / sigma)**2``
    recovers the damped time, so a compensating pair of errors inside the
    Lyashenko/Mercurio correction cannot pass.
    """
    fixing_dates = coupon.fixing_dates()
    fixing_start_time = vol.time_from_reference(fixing_dates[0])
    fixing_end_time = vol.time_from_reference(fixing_dates[-1])
    tolerance.tight(fixing_start_time, ref["fixing_start_time"])
    tolerance.tight(fixing_end_time, ref["fixing_end_time"])
    tolerance.tight(fixing_end_time, ref["effective_time"])

    sigma = vol.volatility(max(fixing_dates[0], vol.reference_date() + 1), ref["eff_strike"])
    tolerance.tight(sigma, ref["sigma"])

    std_dev = effective_caplet_volatility * math.sqrt(ref["effective_time"])
    tolerance.tight(std_dev, ref["std_dev"])
    tolerance.tight((std_dev / sigma) ** 2, ref["damped_time"])


# --- forward-looking pricer blocks -----------------------------------------


def _check_pricer_block(
    pricer: OvernightIndexedCouponPricer,
    coupon: OvernightIndexedCoupon,
    vol: OptionletVolatilityStructure,
    ref: dict[str, Any],
) -> None:
    coupon.set_pricer(pricer)
    pricer.initialize(coupon)
    eff_cap, eff_floor = _eff_cap(), _eff_floor()

    tolerance.tight(pricer.swaplet_rate(), ref["swaplet_rate"])

    caplet = pricer.caplet_rate(eff_cap, False)
    tolerance.tight(caplet, ref["caplet_rate_global"])
    eff_caplet_vol = pricer.effective_caplet_volatility()
    assert eff_caplet_vol is not None
    tolerance.tight(eff_caplet_vol, ref["effective_caplet_volatility"])

    floorlet = pricer.floorlet_rate(eff_floor, False)
    tolerance.tight(floorlet, ref["floorlet_rate_global"])
    eff_floorlet_vol = pricer.effective_floorlet_volatility()
    assert eff_floorlet_vol is not None
    tolerance.tight(eff_floorlet_vol, ref["effective_floorlet_volatility"])

    _check_std_dev_chain(coupon, vol, eff_caplet_vol, ref["std_dev_chain"])

    # The one-argument overload must be bit-identical to the two-argument one
    # with False (C++ cpp:347-353 forwards literally), and match the reference.
    tolerance.exact(pricer.caplet_rate(eff_cap), pricer.caplet_rate(eff_cap, False))
    tolerance.exact(pricer.floorlet_rate(eff_floor), pricer.floorlet_rate(eff_floor, False))
    tolerance.tight(pricer.caplet_rate(eff_cap), ref["caplet_rate_one_arg"])
    tolerance.tight(pricer.floorlet_rate(eff_floor), ref["floorlet_rate_one_arg"])

    if "caplet_rate_local" in ref:
        tolerance.tight(pricer.caplet_rate(eff_cap, True), ref["caplet_rate_local"])
        local_cap_vol = pricer.effective_caplet_volatility()
        assert local_cap_vol is not None
        tolerance.tight(local_cap_vol, ref["effective_caplet_volatility_local"])
        tolerance.tight(pricer.floorlet_rate(eff_floor, True), ref["floorlet_rate_local"])
        local_floor_vol = pricer.effective_floorlet_volatility()
        assert local_floor_vol is not None
        tolerance.tight(local_floor_vol, ref["effective_floorlet_volatility_local"])
    else:
        assert ref["caplet_rate_local_raises"]["raises"] is True
        assert ref["floorlet_rate_local_raises"]["raises"] is True
        with pytest.raises(LibraryException, match="effective volatility input"):
            pricer.caplet_rate(eff_cap, True)
        with pytest.raises(LibraryException, match="effective volatility input"):
            pricer.floorlet_rate(eff_floor, True)

    for key, call in (
        ("swaplet_price_raises", pricer.swaplet_price),
        ("caplet_price_raises", lambda: pricer.caplet_price(eff_cap)),
        ("floorlet_price_raises", lambda: pricer.floorlet_price(eff_floor)),
    ):
        assert ref[key]["raises"] is True
        with pytest.raises(LibraryException, match="not provided"):
            call()


def test_fwd_compounding_lognormal(
    fwd_index: Sofr, ln_vol: ConstantOptionletVolatility, cpp: dict[str, Any]
) -> None:
    _check_pricer_block(
        BlackCompoundingOvernightIndexedCouponPricer(ln_vol),
        _fwd_coupon(fwd_index, RateAveraging.Compound),
        ln_vol,
        cpp["fwd_compounding_lognormal"],
    )


def test_fwd_compounding_normal(
    fwd_index: Sofr, normal_vol: ConstantOptionletVolatility, cpp: dict[str, Any]
) -> None:
    """A Normal vol selects the Bachelier formula instead of Black-76."""
    _check_pricer_block(
        BlackCompoundingOvernightIndexedCouponPricer(normal_vol),
        _fwd_coupon(fwd_index, RateAveraging.Compound),
        normal_vol,
        cpp["fwd_compounding_normal"],
    )


def test_fwd_compounding_effective_vol(
    fwd_index: Sofr, ln_vol: ConstantOptionletVolatility, cpp: dict[str, Any]
) -> None:
    """``effective_volatility_input=True`` drops the damping and forbids daily caps."""
    _check_pricer_block(
        BlackCompoundingOvernightIndexedCouponPricer(ln_vol, effective_volatility_input=True),
        _fwd_coupon(fwd_index, RateAveraging.Compound),
        ln_vol,
        cpp["fwd_compounding_effective_vol"],
    )


def test_fwd_averaging_lognormal(
    fwd_index: Sofr, ln_vol: ConstantOptionletVolatility, cpp: dict[str, Any]
) -> None:
    _check_pricer_block(
        BlackAveragingOvernightIndexedCouponPricer(ln_vol),
        _fwd_coupon(fwd_index, RateAveraging.Simple),
        ln_vol,
        cpp["fwd_averaging_lognormal"],
    )


def test_fwd_averaging_normal(
    fwd_index: Sofr, normal_vol: ConstantOptionletVolatility, cpp: dict[str, Any]
) -> None:
    _check_pricer_block(
        BlackAveragingOvernightIndexedCouponPricer(normal_vol),
        _fwd_coupon(fwd_index, RateAveraging.Simple),
        normal_vol,
        cpp["fwd_averaging_normal"],
    )


def test_fwd_averaging_effective_vol(
    fwd_index: Sofr, ln_vol: ConstantOptionletVolatility, cpp: dict[str, Any]
) -> None:
    _check_pricer_block(
        BlackAveragingOvernightIndexedCouponPricer(ln_vol, effective_volatility_input=True),
        _fwd_coupon(fwd_index, RateAveraging.Simple),
        ln_vol,
        cpp["fwd_averaging_effective_vol"],
    )


def test_averaging_forward_rate_is_ungeared_swaplet(
    fwd_index: Sofr, ln_vol: ConstantOptionletVolatility, cpp: dict[str, Any]
) -> None:
    """The averaging pricer writes its option on ``(swaplet - spread) / gearing``."""
    coupon = _fwd_coupon(fwd_index, RateAveraging.Simple)
    pricer = BlackAveragingOvernightIndexedCouponPricer(ln_vol)
    coupon.set_pricer(pricer)
    pricer.initialize(coupon)
    expected = (cpp["fwd_averaging_lognormal"]["swaplet_rate"] - _SPREAD) / _GEARING
    tolerance.tight(pricer.forward_rate(), expected)


# --- guard rails -----------------------------------------------------------


def test_averaging_pricer_rejects_compounded_coupon(
    fwd_index: Sofr, ln_vol: ConstantOptionletVolatility, cpp: dict[str, Any]
) -> None:
    assert cpp["averaging_pricer_rejects_compounded"]["raises"] is True
    coupon = _fwd_coupon(fwd_index, RateAveraging.Compound)
    pricer = BlackAveragingOvernightIndexedCouponPricer(ln_vol)
    with pytest.raises(LibraryException, match="Averaging method required to be simple"):
        pricer.initialize(coupon)


def test_missing_volatility_raises(fwd_index: Sofr, cpp: dict[str, Any]) -> None:
    """The default (no) vol surface is only usable on an already-fixed coupon."""
    assert cpp["vol_missing_raises"]["raises"] is True
    coupon = _fwd_coupon(fwd_index, RateAveraging.Compound)
    pricer = BlackCompoundingOvernightIndexedCouponPricer()
    coupon.set_pricer(pricer)
    pricer.initialize(coupon)
    with pytest.raises(LibraryException, match="missing optionlet volatility"):
        pricer.caplet_rate(_eff_cap())


def test_set_caplet_volatility_relinks(
    fwd_index: Sofr, ln_vol: ConstantOptionletVolatility, cpp: dict[str, Any]
) -> None:
    """``set_caplet_volatility`` on a bare pricer reproduces the constructed one."""
    coupon = _fwd_coupon(fwd_index, RateAveraging.Compound)
    pricer = BlackCompoundingOvernightIndexedCouponPricer()
    assert pricer.caplet_volatility() is None
    pricer.set_caplet_volatility(ln_vol)
    assert pricer.caplet_volatility() is ln_vol
    coupon.set_pricer(pricer)
    pricer.initialize(coupon)
    tolerance.tight(
        pricer.caplet_rate(_eff_cap()),
        cpp["fwd_compounding_lognormal"]["caplet_rate_global"],
    )


def test_set_effective_volatility_input_switches_formula(
    fwd_index: Sofr, ln_vol: ConstantOptionletVolatility, cpp: dict[str, Any]
) -> None:
    """The setter must move the price from the damped stdDev to the plain one."""
    coupon = _fwd_coupon(fwd_index, RateAveraging.Compound)
    pricer = BlackCompoundingOvernightIndexedCouponPricer(ln_vol)
    assert pricer.effective_volatility_input() is False
    coupon.set_pricer(pricer)
    pricer.initialize(coupon)
    tolerance.tight(
        pricer.caplet_rate(_eff_cap()),
        cpp["fwd_compounding_lognormal"]["caplet_rate_global"],
    )
    pricer.set_effective_volatility_input(True)
    assert pricer.effective_volatility_input() is True
    tolerance.tight(
        pricer.caplet_rate(_eff_cap()),
        cpp["fwd_compounding_effective_vol"]["caplet_rate_global"],
    )


def test_effective_volatilities_start_unset(fwd_index: Sofr, ln_vol: ConstantOptionletVolatility) -> None:
    """``initialize`` resets both slots to ``None`` (C++ ``Null<Real>``)."""
    coupon = _fwd_coupon(fwd_index, RateAveraging.Compound)
    pricer = BlackCompoundingOvernightIndexedCouponPricer(ln_vol)
    coupon.set_pricer(pricer)
    pricer.initialize(coupon)
    assert pricer.effective_caplet_volatility() is None
    assert pricer.effective_floorlet_volatility() is None
    pricer.caplet_rate(_eff_cap())
    assert pricer.effective_caplet_volatility() is not None
    # Pricing the caplet must not fill in the floorlet slot.
    assert pricer.effective_floorlet_volatility() is None


def test_pricer_accepts_capped_floored_coupon(
    fwd_index: Sofr, ln_vol: ConstantOptionletVolatility, cpp: dict[str, Any]
) -> None:
    """``initialize`` unwraps a CappedFlooredOvernightIndexedCoupon (cpp:46-50)."""
    underlying = _fwd_coupon(fwd_index, RateAveraging.Compound)
    cf = CappedFlooredOvernightIndexedCoupon(underlying, _CAP, _FLOOR)
    pricer = BlackCompoundingOvernightIndexedCouponPricer(ln_vol)
    pricer.initialize(cf)
    tolerance.tight(pricer.swaplet_rate(), cpp["fwd_compounding_lognormal"]["swaplet_rate"])


def test_base_pricers_carry_the_vol_arguments(ln_vol: ConstantOptionletVolatility) -> None:
    """The C++ base pricers store the vol handle and the effective-vol flag.

    They never price an optionlet themselves, so the only observable
    consequence of the two constructor arguments is that the accessors return
    them — but a dropped argument still has to fail here.
    """
    compounding = CompoundingOvernightIndexedCouponPricer(ln_vol, True)
    assert compounding.caplet_volatility() is ln_vol
    assert compounding.effective_volatility_input() is True
    averaging = ArithmeticAveragedOvernightIndexedCouponPricer(0.05, 0.01, True, ln_vol, True)
    assert averaging.caplet_volatility() is ln_vol
    assert averaging.effective_volatility_input() is True
    # Defaults, for contrast.
    assert CompoundingOvernightIndexedCouponPricer().caplet_volatility() is None
    assert CompoundingOvernightIndexedCouponPricer().effective_volatility_input() is False


def test_coupon_carve_out_inspectors(fwd_index: Sofr) -> None:
    """Lockout / daily-spread are fixed at the C++ defaults, not settable.

    ``_optionlet_rate_local`` branches on both, so the values have to be
    readable; they are deliberately *not* constructor arguments, so there is
    nothing that could be accepted and dropped.
    """
    coupon = _fwd_coupon(fwd_index, RateAveraging.Compound)
    assert coupon.lockout_days() == 0
    assert coupon.compound_spread_daily() is False


def test_base_compounding_pricer_refuses_daily_cap_floor(fwd_index: Sofr) -> None:
    """Only the Black pricers implement the two-argument form."""
    coupon = _fwd_coupon(fwd_index, RateAveraging.Compound)
    pricer = CompoundingOvernightIndexedCouponPricer()
    pricer.initialize(coupon)
    with pytest.raises(LibraryException, match="not implemented"):
        pricer.caplet_rate(0.03, True)
    with pytest.raises(LibraryException, match="not implemented"):
        pricer.floorlet_rate(0.03, True)


# --- capped / floored / collared coupons -----------------------------------


def _check_capped_floored(
    coupon: CappedFlooredOvernightIndexedCoupon,
    pricer: OvernightIndexedCouponPricer,
    ref: dict[str, Any],
) -> None:
    coupon.set_pricer(pricer)
    assert coupon.is_capped() is ref["is_capped"]
    assert coupon.is_floored() is ref["is_floored"]
    assert coupon.naked_option() is ref["naked_option"]
    if ref["is_capped"]:
        cap = coupon.cap()
        eff_cap = coupon.effective_cap()
        assert cap is not None
        assert eff_cap is not None
        tolerance.tight(cap, ref["cap"])
        tolerance.tight(eff_cap, ref["effective_cap"])
    if ref["is_floored"]:
        floor = coupon.floor()
        eff_floor = coupon.effective_floor()
        assert floor is not None
        assert eff_floor is not None
        tolerance.tight(floor, ref["floor"])
        tolerance.tight(eff_floor, ref["effective_floor"])
    underlying = coupon.underlying()
    tolerance.tight(underlying.rate(), ref["underlying_rate"])
    tolerance.tight(coupon.rate(), ref["rate"])
    tolerance.tight(coupon.amount(), ref["amount"])
    for got, key in (
        (pricer.effective_caplet_volatility(), "effective_caplet_volatility"),
        (pricer.effective_floorlet_volatility(), "effective_floorlet_volatility"),
    ):
        if ref[key] is None:
            assert got is None
        else:
            assert got is not None
            tolerance.tight(got, ref[key])


@pytest.mark.parametrize(
    ("key", "cap", "floor", "naked"),
    [
        ("fwd_capped_lognormal", _CAP, None, False),
        ("fwd_floored_lognormal", None, _FLOOR, False),
        ("fwd_collared_lognormal", _CAP, _FLOOR, False),
        ("fwd_capped_naked_lognormal", _CAP, None, True),
    ],
)
def test_fwd_capped_floored_lognormal(
    fwd_index: Sofr,
    ln_vol: ConstantOptionletVolatility,
    cpp: dict[str, Any],
    key: str,
    cap: float | None,
    floor: float | None,
    naked: bool,
) -> None:
    underlying = _fwd_coupon(fwd_index, RateAveraging.Compound)
    _check_capped_floored(
        CappedFlooredOvernightIndexedCoupon(underlying, cap, floor, naked),
        BlackCompoundingOvernightIndexedCouponPricer(ln_vol),
        cpp[key],
    )


@pytest.mark.parametrize(
    ("key", "cap", "floor"),
    [
        ("fwd_capped_normal", _CAP, None),
        ("fwd_floored_normal", None, _FLOOR),
        ("fwd_collared_normal", _CAP, _FLOOR),
    ],
)
def test_fwd_capped_floored_normal(
    fwd_index: Sofr,
    normal_vol: ConstantOptionletVolatility,
    cpp: dict[str, Any],
    key: str,
    cap: float | None,
    floor: float | None,
) -> None:
    underlying = _fwd_coupon(fwd_index, RateAveraging.Compound)
    _check_capped_floored(
        CappedFlooredOvernightIndexedCoupon(underlying, cap, floor),
        BlackCompoundingOvernightIndexedCouponPricer(normal_vol),
        cpp[key],
    )


def test_fwd_collared_averaging(
    fwd_index: Sofr, ln_vol: ConstantOptionletVolatility, cpp: dict[str, Any]
) -> None:
    underlying = _fwd_coupon(fwd_index, RateAveraging.Simple)
    _check_capped_floored(
        CappedFlooredOvernightIndexedCoupon(underlying, _CAP, _FLOOR),
        BlackAveragingOvernightIndexedCouponPricer(ln_vol),
        cpp["fwd_collared_averaging_lognormal"],
    )


# --- already-fixed (intrinsic) scenario -------------------------------------


def test_past_compounding_intrinsic(past_index: Sofr, cpp: dict[str, Any]) -> None:
    """A fully-fixed coupon needs no vol surface: the optionlet is intrinsic."""
    ref = cpp["past_compounding"]
    coupon = _past_coupon(past_index, RateAveraging.Compound)
    pricer = BlackCompoundingOvernightIndexedCouponPricer()
    coupon.set_pricer(pricer)
    pricer.initialize(coupon)
    tolerance.tight(pricer.swaplet_rate(), ref["swaplet_rate"])
    tolerance.tight(pricer.effective_index_fixing(), ref["effective_index_fixing"])
    tolerance.tight(pricer.effective_spread(), ref["effective_spread"])
    tolerance.tight(pricer.caplet_rate(0.0400), ref["caplet_rate_itm"])
    tolerance.tight(pricer.caplet_rate(0.0600), ref["caplet_rate_otm"])
    tolerance.tight(pricer.floorlet_rate(0.0600), ref["floorlet_rate_itm"])
    tolerance.tight(pricer.floorlet_rate(0.0400), ref["floorlet_rate_otm"])
    # Daily cap/floor with no future fixings left never touches a vol surface.
    tolerance.tight(pricer.caplet_rate(0.0510, True), ref["caplet_rate_local"])
    tolerance.tight(pricer.floorlet_rate(0.0510, True), ref["floorlet_rate_local"])
    assert ref["effective_caplet_volatility_is_null"] is True
    assert pricer.effective_caplet_volatility() is None


def test_past_intrinsic_is_geared(past_index: Sofr, cpp: dict[str, Any]) -> None:
    """The intrinsic branch multiplies by the coupon gearing (cpp:151)."""
    ref = cpp["past_compounding"]
    expected = _GEARING * max(ref["effective_index_fixing"] - 0.0400, 0.0)
    tolerance.tight(ref["caplet_rate_itm"], expected)


def test_past_averaging_intrinsic(past_index: Sofr, cpp: dict[str, Any]) -> None:
    ref = cpp["past_averaging"]
    coupon = _past_coupon(past_index, RateAveraging.Simple)
    pricer = BlackAveragingOvernightIndexedCouponPricer()
    coupon.set_pricer(pricer)
    pricer.initialize(coupon)
    tolerance.tight(pricer.swaplet_rate(), ref["swaplet_rate"])
    tolerance.tight(pricer.forward_rate(), ref["forward_rate"])
    tolerance.tight(pricer.caplet_rate(0.0400), ref["caplet_rate_itm"])
    tolerance.tight(pricer.caplet_rate(0.0600), ref["caplet_rate_otm"])
    tolerance.tight(pricer.floorlet_rate(0.0600), ref["floorlet_rate_itm"])
    tolerance.tight(pricer.floorlet_rate(0.0400), ref["floorlet_rate_otm"])
    tolerance.tight(pricer.caplet_rate(0.0510, True), ref["caplet_rate_local"])
    tolerance.tight(pricer.floorlet_rate(0.0510, True), ref["floorlet_rate_local"])


@pytest.mark.parametrize(
    ("key", "cap", "floor"),
    [
        ("past_capped", 0.102, None),
        ("past_floored", None, 0.105),
        ("past_collared", 0.102, 0.098),
    ],
)
def test_past_capped_floored(
    past_index: Sofr,
    cpp: dict[str, Any],
    key: str,
    cap: float | None,
    floor: float | None,
) -> None:
    """With the fixing known, a binding cap/floor pins the rate on the level."""
    underlying = _past_coupon(past_index, RateAveraging.Compound)
    _check_capped_floored(
        CappedFlooredOvernightIndexedCoupon(underlying, cap, floor),
        BlackCompoundingOvernightIndexedCouponPricer(),
        cpp[key],
    )
