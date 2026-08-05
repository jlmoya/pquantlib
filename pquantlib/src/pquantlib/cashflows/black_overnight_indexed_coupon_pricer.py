"""Black pricers for capped / floored overnight-indexed coupons.

# C++ parity: ql/cashflows/blackovernightindexedcouponpricer.{hpp,cpp} (v1.43).

Two pricers, one per averaging convention of the underlying
:class:`~pquantlib.cashflows.overnight_indexed_coupon.OvernightIndexedCoupon`:

- :class:`BlackCompoundingOvernightIndexedCouponPricer` — daily-compounded
  coupon (``RateAveraging.Compound``);
- :class:`BlackAveragingOvernightIndexedCouponPricer` — arithmetically
  averaged coupon (``RateAveraging.Simple``); its ``initialize`` rejects a
  compounded coupon outright (cpp:383-384).

Both add optionality on top of their base pricer's swaplet rate, and both
offer *two* mutually exclusive optionlet formulas, selected by the
``daily_cap_floor`` flag that
:class:`~pquantlib.cashflows.capped_floored_coupon.CappedFlooredOvernightIndexedCoupon`
passes through:

``daily_cap_floor=False`` — :func:`_optionlet_rate_global`
    A single option on the whole-period rate. If the coupon is already fully
    fixed (``fixing_date <= evaluation date``) the answer is the intrinsic
    value ``gearing * max(fwd - K, 0)``; otherwise it is a Black-76 price
    (``ShiftedLognormal`` vol) or a Bachelier price (``Normal`` vol). The
    standard deviation is *not* ``vol * sqrt(t)``: unless the vol input is
    declared effective, the average vol between fixing start and fixing end is
    damped by the Lyashenko/Mercurio backward-looking-rate correction
    (cpp:165-176, "Looking forward to backward looking rates", section 6.3)::

        T = max(Ts, 0)
        T += (Te - T)^3 / (Te - Ts)^2 / 3        # unless Te == T
        stdDev = sigma * sqrt(T)

``daily_cap_floor=True`` — :func:`_optionlet_rate_local`
    The cap / floor applies to each *daily* rate. C++ approximates this by
    capping the already-fixed days exactly, pricing ONE cap / floor in the
    middle of the remaining period, folding it into an average daily rate,
    and rebuilding the coupon rate from there; the returned number is the
    DIFFERENCE against the same computation without the cap / floor, i.e. the
    option component alone (cpp:198-343 for compounding, cpp:441-586 for
    averaging). It is explicitly not available with an effective-vol input.

Both formulas record the effective (Black-equivalent) volatility they used, so
that ``pricer.effective_caplet_volatility()`` / ``...floorlet...()`` can be
read afterwards; the intrinsic branch records nothing, leaving the value at
``None`` (C++ ``Null<Real>``).

C++ comments the whole file as "highly experimental and ad-hoc" pending a
market best practice; the port reproduces it as written rather than
improving it.

Python divergences from C++:

- ``swaplet_price`` / ``caplet_price`` / ``floorlet_price`` raise, exactly as
  the C++ ``QL_FAIL("... not provided")`` (cpp:365-373, 608-618).
- The lockout (rate-cutoff) and compound-spread-daily branches of
  :func:`_optionlet_rate_local` are written out but are dead code in this
  port: ``OvernightIndexedCoupon.lockout_days()`` is fixed at 0 and
  ``compound_spread_daily()`` at ``False`` (both are deferred carve-outs of
  the coupon, not of these pricers). They are kept so that the day a lockout
  coupon lands, the pricer is already right.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from pquantlib import qassert
from pquantlib.cashflows.overnight_indexed_coupon_pricer import (
    ArithmeticAveragedOvernightIndexedCouponPricer,
    CompoundingOvernightIndexedCouponPricer,
)
from pquantlib.cashflows.rate_averaging import RateAveraging
from pquantlib.exceptions import LibraryException
from pquantlib.indexes.overnight_index import OvernightIndex
from pquantlib.math.closeness import close_enough
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.payoffs import OptionType
from pquantlib.pricingengines.black_formula import (
    bachelier_black_formula,
    black_formula,
)
from pquantlib.termstructures.volatility.volatility_type import VolatilityType

if TYPE_CHECKING:
    from pquantlib.cashflows.floating_rate_coupon import FloatingRateCoupon
    from pquantlib.cashflows.overnight_indexed_coupon import OvernightIndexedCoupon
    from pquantlib.termstructures.volatility.optionlet.optionlet_volatility_structure import (
        OptionletVolatilityStructure,
    )
    from pquantlib.time.date import Date


# -----------------------------------------------------------------------
# Shared optionlet machinery
#
# C++ spells both pricers' optionletRateGlobal / optionletRateLocal out
# twice, identically apart from which forward they write the option on and
# whether the daily rates compound or accumulate. Here the shared body lives
# in module-level helpers that return ``(rate, effective_volatility)`` — the
# caller stores the effective volatility in the caplet or floorlet slot, and
# leaves the slot untouched when the helper reports ``None`` (which is what
# C++ does by simply not assigning in the intrinsic branch).
# -----------------------------------------------------------------------


def _black_or_bachelier(
    vol: OptionletVolatilityStructure,
    option_type: OptionType,
    eff_strike: float,
    forward: float,
    std_dev: float,
) -> float:
    """Black-76 for a shifted-lognormal vol, Bachelier for a normal one.

    # C++ parity: cpp:182-183 / 299-300 / 435-436 / 542-543.
    """
    if vol.volatility_type() == VolatilityType.ShiftedLognormal:
        return black_formula(option_type, eff_strike, forward, std_dev, 1.0, vol.displacement())
    return bachelier_black_formula(option_type, eff_strike, forward, std_dev, 1.0)


def _global_std_dev(
    vol: OptionletVolatilityStructure,
    fixing_dates: list[Date],
    eff_strike: float,
    *,
    effective_volatility_input: bool,
) -> tuple[float, float]:
    """Return ``(std_dev, effective_time)`` for the whole-period optionlet.

    # C++ parity: cpp:159-177 (compounding) == cpp:412-430 (averaging).
    """
    effective_time = vol.time_from_reference(fixing_dates[-1])
    if effective_volatility_input:
        # The input vol is already the effective one: plain Black model.
        std_dev = vol.volatility(fixing_dates[-1], eff_strike) * math.sqrt(effective_time)
        return std_dev, effective_time
    # Lyashenko / Mercurio damping of the average vol over the fixing window.
    fixing_start_time = vol.time_from_reference(fixing_dates[0])
    fixing_end_time = vol.time_from_reference(fixing_dates[-1])
    sigma = vol.volatility(max(fixing_dates[0], vol.reference_date() + 1), eff_strike)
    damped = max(fixing_start_time, 0.0)
    if not close_enough(fixing_end_time, damped):
        damped += (fixing_end_time - damped) ** 3.0 / (fixing_end_time - fixing_start_time) ** 2.0 / 3.0
    return sigma * math.sqrt(damped), effective_time


def _optionlet_rate_global(
    coupon: OvernightIndexedCoupon,
    vol: OptionletVolatilityStructure | None,
    option_type: OptionType,
    eff_strike: float,
    forward: float,
    gearing: float,
    *,
    effective_volatility_input: bool,
    label: str,
) -> tuple[float, float | None]:
    """Whole-period optionlet: intrinsic if fixed, else Black / Bachelier.

    # C++ parity: cpp:139-186 (compounding) / cpp:392-439 (averaging). The two
    # differ only in ``forward``: ``effectiveIndexFixing_`` for the compounding
    # pricer, ``forwardRate_`` for the averaging one.
    """
    today = ObservableSettings().evaluation_date_or_today()
    if coupon.fixing_date() <= today:
        # The amount is determined; nothing is left to model.
        if option_type == OptionType.Call:
            a, b = forward, eff_strike
        else:
            a, b = eff_strike, forward
        return gearing * max(a - b, 0.0), None

    qassert.require(vol is not None, f"{label}: missing optionlet volatility")
    assert vol is not None
    fixing_dates = coupon.fixing_dates()
    qassert.require(len(fixing_dates) > 0, f"{label}: empty fixing dates")

    std_dev, effective_time = _global_std_dev(
        vol,
        fixing_dates,
        eff_strike,
        effective_volatility_input=effective_volatility_input,
    )
    fixing = _black_or_bachelier(vol, option_type, eff_strike, forward, std_dev)
    return gearing * fixing, std_dev / math.sqrt(effective_time)


def _capped_floored_rate(rate: float, option_type: OptionType, strike: float) -> float:
    """``min(r, k)`` for a cap, ``max(r, k)`` for a floor.

    # C++ parity: the anonymous-namespace helper at cpp:188-196.
    """
    if option_type == OptionType.Call:
        return min(rate, strike)
    return max(rate, strike)


def _local_past_part(
    coupon: OvernightIndexedCoupon,
    index: OvernightIndex,
    option_type: OptionType,
    abs_strike: float,
    n_cutoff: int,
    *,
    compounding: bool,
) -> tuple[float, float, int]:
    """The already-fixed stretch of the daily cap / floor formula.

    # C++ parity: cpp:233-268 (compounding) == cpp:476-511 (averaging). Each
    # historical daily fixing is capped / floored exactly; today's fixing is a
    # border case and is only used if it has actually been published.

    Returns ``(accumulated, accumulated_raw, first_unfixed_index)`` where the
    two accumulators start from the identity of the relevant operation — 1.0
    for the compounding product, 0.0 for the averaging sum.
    """
    spread = coupon.spread()
    compound_spread_daily = coupon.compound_spread_daily()
    fixing_dates = coupon.fixing_dates()
    dt = coupon.dt()
    n = len(dt)
    today = ObservableSettings().evaluation_date_or_today()

    accumulated = 1.0 if compounding else 0.0
    accumulated_raw = 1.0 if compounding else 0.0

    def accumulate(total: float, rate: float, span: float) -> float:
        return total * (1.0 + rate * span) if compounding else total + rate * span

    def absorb(i: int, fixing: float) -> tuple[float, float]:
        capped = _capped_floored_rate(fixing, option_type, abs_strike)
        return accumulate(accumulated, capped, dt[i]), accumulate(accumulated_raw, fixing, dt[i])

    i = 0
    while i < n and fixing_dates[min(i, n_cutoff)] < today:
        d = fixing_dates[min(i, n_cutoff)]
        qassert.require(index.has_historical_fixing(d), f"Missing {index.name()} fixing for {d}")
        fixing = index.past_fixing(d) + (spread if compound_spread_daily else 0.0)
        accumulated, accumulated_raw = absorb(i, fixing)
        i += 1

    # Today is a border case: use the fixing only if it has been published.
    if i < n and fixing_dates[min(i, n_cutoff)] == today and index.has_historical_fixing(today):
        fixing = index.past_fixing(today) + (spread if compound_spread_daily else 0.0)
        accumulated, accumulated_raw = absorb(i, fixing)
        i += 1

    return accumulated, accumulated_raw, i


def _local_forward_part(
    coupon: OvernightIndexedCoupon,
    index: OvernightIndex,
    vol: OptionletVolatilityStructure | None,
    option_type: OptionType,
    eff_strike: float,
    first_unfixed: int,
    n_cutoff: int,
    *,
    label: str,
) -> tuple[float, float, int, float, float]:
    """The not-yet-fixed stretch of the daily cap / floor formula.

    # C++ parity: cpp:270-322 (compounding) == cpp:513-565 (averaging), up to
    # the point where the two diverge into compounding vs accumulating.

    Returns ``(average_rate, average_rate_raw, n_days, daily_tau,
    effective_volatility)``: the average daily rate with and without the
    cap / floor folded in, the number of calendar days the average applies
    to, the average daily accrual fraction, and the Black-equivalent
    volatility used.
    """
    qassert.require(vol is not None, f"{label}: missing optionlet volatility")
    assert vol is not None
    curve = index.forecast_term_structure()
    qassert.require(
        curve is not None,
        f"null term structure set to this instance of {index.name()}",
    )
    assert curve is not None

    dates = coupon.value_dates()
    n = len(coupon.dt())
    day_counter = coupon.day_counter()
    i = first_unfixed

    start_discount = curve.discount(dates[i])
    end_discount = curve.discount(dates[max(n_cutoff, i)])
    if n_cutoff < n:
        # Freeze the one-day forward discount factor over the cutoff period.
        discount_cutoff_date = curve.discount(dates[n_cutoff] + 1) / curve.discount(dates[n_cutoff])
        end_discount *= discount_cutoff_date ** (dates[n] - dates[n_cutoff])

    # Continuously-compounded average daily rate over the future stretch.
    tau_fwd = day_counter.year_fraction(dates[i], dates[-1])
    average_rate = -math.log(end_discount / start_discount) / tau_fwd

    # One cap / floor priced at the mid-point of the remaining period.
    mid_point = (vol.time_from_reference(dates[i]) + vol.time_from_reference(dates[n_cutoff])) / 2.0
    std_dev = vol.volatility(mid_point, eff_strike) * math.sqrt(mid_point)
    cf_value = _black_or_bachelier(vol, option_type, eff_strike, average_rate, std_dev)
    effective_time = vol.time_from_reference(coupon.fixing_dates()[-1])
    effective_volatility = std_dev / math.sqrt(effective_time)

    if coupon.compound_spread_daily():
        average_rate += coupon.spread()
    average_rate_raw = average_rate
    average_rate += -cf_value if option_type == OptionType.Call else cf_value

    # "Ester / Daily Spread Curve Setup in ORE" formula (4): treat the average
    # rate as the effective daily rate over the whole future period.
    n_days = dates[-1] - dates[i]
    daily_tau = tau_fwd / n_days
    return average_rate, average_rate_raw, n_days, daily_tau, effective_volatility


def _optionlet_rate_local(
    coupon: OvernightIndexedCoupon,
    vol: OptionletVolatilityStructure | None,
    option_type: OptionType,
    eff_strike: float,
    *,
    compounding: bool,
    effective_volatility_input: bool,
    label: str,
) -> tuple[float, float | None]:
    """Daily cap / floor: the option component of a per-day capped rate.

    # C++ parity: cpp:198-343 (compounding) / cpp:441-586 (averaging). The two
    # bodies are identical apart from compounding ``1 + r*dt`` factors versus
    # accumulating ``r*dt`` terms, and the ``tau`` fallback (``lockoutDays``
    # for compounding, ``fixingDays`` for averaging — a C++ asymmetry that is
    # reproduced as written).

    Returns the *difference* between the capped/floored rate and the raw rate,
    signed so that ``rate = swaplet + floorlet - caplet`` reconstructs the
    coupon (cpp:339-342).
    """
    qassert.require(
        not effective_volatility_input,
        f"{label}::optionletRateLocal() does not support effective volatility input.",
    )

    spread = coupon.spread()
    gearing = coupon.gearing()
    compound_spread_daily = coupon.compound_spread_daily()
    # The strike the DAILY rate is capped at: with a daily-compounded spread
    # the spread is part of the daily rate, so it must be added back in.
    abs_strike = eff_strike + spread if compound_spread_daily else eff_strike

    # C++ ``dynamic_pointer_cast<OvernightIndex>`` (cpp:221 / 464): the daily
    # formula needs the index's own fixing history and forwarding curve, which
    # the structural OvernightIndexProtocol does not carry.
    index = coupon.overnight_index()
    qassert.require(
        isinstance(index, OvernightIndex),
        f"{label}: expected an OvernightIndex on the coupon",
    )
    assert isinstance(index, OvernightIndex)
    dates = coupon.value_dates()
    day_counter = coupon.day_counter()

    n = len(coupon.dt())
    lockout_days = coupon.lockout_days()
    qassert.require(
        lockout_days < n,
        f"rate cutoff ({lockout_days}) must be less than number of fixings in period ({n})",
    )
    n_cutoff = n - lockout_days

    accumulated, accumulated_raw, i = _local_past_part(
        coupon, index, option_type, abs_strike, n_cutoff, compounding=compounding
    )

    effective_volatility: float | None = None
    if i < n:
        average_rate, average_rate_raw, n_days, daily_tau, effective_volatility = _local_forward_part(
            coupon, index, vol, option_type, eff_strike, i, n_cutoff, label=label
        )
        if compounding:
            accumulated *= (1.0 + daily_tau * average_rate) ** n_days
            accumulated_raw *= (1.0 + daily_tau * average_rate_raw) ** n_days
        else:
            accumulated += daily_tau * average_rate * n_days
            accumulated_raw += daily_tau * average_rate_raw * n_days

    fallback = lockout_days if compounding else coupon.fixing_days()
    tau = coupon.accrual_period() if fallback == 0 else day_counter.year_fraction(dates[0], dates[-1])
    if compounding:
        rate = (accumulated - 1.0) / tau
        raw_rate = (accumulated_raw - 1.0) / tau
    else:
        rate = accumulated / tau
        raw_rate = accumulated_raw / tau

    rate *= gearing
    raw_rate *= gearing
    if not compound_spread_daily:
        rate += spread
        raw_rate += spread

    sign = -1.0 if option_type == OptionType.Call else 1.0
    return sign * (rate - raw_rate), effective_volatility


# -----------------------------------------------------------------------
# BlackCompoundingOvernightIndexedCouponPricer
# -----------------------------------------------------------------------


class BlackCompoundingOvernightIndexedCouponPricer(CompoundingOvernightIndexedCouponPricer):
    """Black pricer for a capped / floored **compounded** overnight coupon.

    # C++ parity: ``BlackCompoundingOvernightIndexedCouponPricer``
    # (blackovernightindexedcouponpricer.hpp:39-63 + .cpp:126-373).
    """

    def __init__(
        self,
        caplet_volatility: OptionletVolatilityStructure | None = None,
        effective_volatility_input: bool = False,
    ) -> None:
        super().__init__(caplet_volatility, effective_volatility_input)
        self._gearing: float = 1.0

    def initialize(self, coupon: FloatingRateCoupon) -> None:
        """Bind the coupon and cache the compounded swaplet rate.

        # C++ parity: cpp:131-137 — the effective caplet / floorlet vols are
        # reset to ``Null<Real>`` (``None`` here) on every initialize.
        """
        super().initialize(coupon)
        c = self._coupon_required()
        self._gearing = c.gearing()
        swaplet_rate, effective_spread, effective_index_fixing = self._compute(
            c.accrual_end_date(), full=True
        )
        self._swaplet_rate = swaplet_rate
        self._effective_spread = effective_spread
        self._effective_index_fixing = effective_index_fixing
        self._effective_caplet_volatility = None
        self._effective_floorlet_volatility = None

    # --- FloatingRateCoupon interface -----------------------------------

    def swaplet_rate(self) -> float:
        """# C++ parity: cpp:345 — the value cached by ``initialize``."""
        return self._swaplet_rate

    def caplet_rate(self, effective_cap: float, daily_cap_floor: bool = False) -> float:
        """# C++ parity: cpp:347-349 (one-arg) + cpp:355-358 (two-arg)."""
        return self._optionlet_rate(OptionType.Call, effective_cap, daily_cap_floor)

    def floorlet_rate(self, effective_floor: float, daily_cap_floor: bool = False) -> float:
        """# C++ parity: cpp:351-353 (one-arg) + cpp:360-363 (two-arg)."""
        return self._optionlet_rate(OptionType.Put, effective_floor, daily_cap_floor)

    def swaplet_price(self) -> float:
        """# C++ parity: cpp:365-367 — ``QL_FAIL``."""
        msg = "BlackCompoundingOvernightIndexedCouponPricer::swapletPrice() not provided"
        raise LibraryException(msg)

    def caplet_price(self, effective_cap: float) -> float:
        """# C++ parity: cpp:368-370 — ``QL_FAIL``."""
        del effective_cap
        msg = "BlackCompoundingOvernightIndexedCouponPricer::capletPrice() not provided"
        raise LibraryException(msg)

    def floorlet_price(self, effective_floor: float) -> float:
        """# C++ parity: cpp:371-373 — ``QL_FAIL``."""
        del effective_floor
        msg = "BlackCompoundingOvernightIndexedCouponPricer::floorletPrice() not provided"
        raise LibraryException(msg)

    # --- internals -------------------------------------------------------

    def _optionlet_rate(self, option_type: OptionType, eff_strike: float, daily_cap_floor: bool) -> float:
        coupon = self._coupon_required()
        label = "BlackCompoundingOvernightIndexedCouponPricer"
        if daily_cap_floor:
            rate, effective_vol = _optionlet_rate_local(
                coupon,
                self._caplet_vol,
                option_type,
                eff_strike,
                compounding=True,
                effective_volatility_input=self._effective_volatility_input,
                label=label,
            )
        else:
            rate, effective_vol = _optionlet_rate_global(
                coupon,
                self._caplet_vol,
                option_type,
                eff_strike,
                self._effective_index_fixing,
                self._gearing,
                effective_volatility_input=self._effective_volatility_input,
                label=label,
            )
        if effective_vol is not None:
            if option_type == OptionType.Call:
                self._effective_caplet_volatility = effective_vol
            else:
                self._effective_floorlet_volatility = effective_vol
        return rate


# -----------------------------------------------------------------------
# BlackAveragingOvernightIndexedCouponPricer
# -----------------------------------------------------------------------


class BlackAveragingOvernightIndexedCouponPricer(ArithmeticAveragedOvernightIndexedCouponPricer):
    """Black pricer for a capped / floored **arithmetically averaged** ON coupon.

    # C++ parity: ``BlackAveragingOvernightIndexedCouponPricer``
    # (blackovernightindexedcouponpricer.hpp:69-93 + .cpp:375-618).

    The C++ constructor forwards ``(0.03, 0.0, false, v, effectiveVolatilityInput)``
    to :class:`~pquantlib.cashflows.overnight_indexed_coupon_pricer.ArithmeticAveragedOvernightIndexedCouponPricer`
    (cpp:375-378), i.e. mean reversion 0.03 with **zero** convexity volatility
    and no Takada approximation — so no convexity adjustment is applied.
    """

    def __init__(
        self,
        caplet_volatility: OptionletVolatilityStructure | None = None,
        effective_volatility_input: bool = False,
    ) -> None:
        super().__init__(0.03, 0.0, False, caplet_volatility, effective_volatility_input)
        self._gearing: float = 1.0
        self._swaplet_rate: float = 0.0
        self._forward_rate: float = 0.0

    def initialize(self, coupon: FloatingRateCoupon) -> None:
        """Bind the coupon, reject a compounded one, cache the average rate.

        # C++ parity: cpp:380-390.
        """
        super().initialize(coupon)
        c = self._coupon_required()
        qassert.require(
            c.averaging_method() != RateAveraging.Compound,
            "Averaging method required to be simple for BlackAveragingOvernightIndexedCouponPricer",
        )
        self._gearing = c.gearing()
        self._swaplet_rate = super().swaplet_rate()
        self._forward_rate = (self._swaplet_rate - c.spread()) / c.gearing()
        self._effective_caplet_volatility = None
        self._effective_floorlet_volatility = None

    # --- FloatingRateCoupon interface -----------------------------------

    def swaplet_rate(self) -> float:
        """# C++ parity: cpp:588 — the value cached by ``initialize``."""
        return self._swaplet_rate

    def forward_rate(self) -> float:
        """The un-geared, un-spread average rate the optionlet is written on.

        # C++ parity: the private ``forwardRate_`` member (hpp:92), computed at
        # cpp:388. Exposed here because it is the forward of the Black call.
        """
        return self._forward_rate

    def caplet_rate(self, effective_cap: float, daily_cap_floor: bool = False) -> float:
        """# C++ parity: cpp:590-592 (one-arg) + cpp:598-601 (two-arg)."""
        return self._optionlet_rate(OptionType.Call, effective_cap, daily_cap_floor)

    def floorlet_rate(self, effective_floor: float, daily_cap_floor: bool = False) -> float:
        """# C++ parity: cpp:594-596 (one-arg) + cpp:603-606 (two-arg)."""
        return self._optionlet_rate(OptionType.Put, effective_floor, daily_cap_floor)

    def swaplet_price(self) -> float:
        """# C++ parity: cpp:608-610 — ``QL_FAIL``."""
        msg = "BlackAveragingOvernightIndexedCouponPricer::swapletPrice() not provided"
        raise LibraryException(msg)

    def caplet_price(self, effective_cap: float) -> float:
        """# C++ parity: cpp:612-614 — ``QL_FAIL``."""
        del effective_cap
        msg = "BlackAveragingOvernightIndexedCouponPricer::capletPrice() not provided"
        raise LibraryException(msg)

    def floorlet_price(self, effective_floor: float) -> float:
        """# C++ parity: cpp:616-618 — ``QL_FAIL``."""
        del effective_floor
        msg = "BlackAveragingOvernightIndexedCouponPricer::floorletPrice() not provided"
        raise LibraryException(msg)

    # --- internals -------------------------------------------------------

    def _optionlet_rate(self, option_type: OptionType, eff_strike: float, daily_cap_floor: bool) -> float:
        coupon = self._coupon_required()
        label = "BlackAveragingOvernightIndexedCouponPricer"
        if daily_cap_floor:
            rate, effective_vol = _optionlet_rate_local(
                coupon,
                self._caplet_vol,
                option_type,
                eff_strike,
                compounding=False,
                effective_volatility_input=self._effective_volatility_input,
                label=label,
            )
        else:
            rate, effective_vol = _optionlet_rate_global(
                coupon,
                self._caplet_vol,
                option_type,
                eff_strike,
                self._forward_rate,
                self._gearing,
                effective_volatility_input=self._effective_volatility_input,
                label=label,
            )
        if effective_vol is not None:
            if option_type == OptionType.Call:
                self._effective_caplet_volatility = effective_vol
            else:
                self._effective_floorlet_volatility = effective_vol
        return rate


__all__ = [
    "BlackAveragingOvernightIndexedCouponPricer",
    "BlackCompoundingOvernightIndexedCouponPricer",
]
