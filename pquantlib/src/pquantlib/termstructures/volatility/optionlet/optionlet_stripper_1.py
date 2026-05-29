"""OptionletStripper1 — strip caplet vols from cap term vols.

# C++ parity: ql/termstructures/volatility/optionlet/optionletstripper1.{hpp,cpp}
# + optionletstripper.{hpp,cpp} (v1.42.1).

The class consumes a ``CapFloorTermVolSurface`` + ``IborIndex`` and
back-solves caplet-by-caplet implied vols that reproduce the cap NPVs
at the input vols. C++ uses ``MakeCapFloor`` factories + Black/
Bachelier engines + ``blackFormulaImpliedStdDev`` (Newton iteration).

PQuantLib divergences:

- ``MakeCapFloor`` factory not ported; the stripper builds the
  floating leg via ``ibor_leg`` and wraps it as a ``Cap`` directly.
- We merge the abstract ``OptionletStripper`` parent into this
  concrete class — the abstract layer only exists in C++ to support
  ``OptionletStripper2``, which is deferred (Phase 9 carve-out).
- ``dontThrow`` / ``optionletFrequency`` flags are ported but the
  custom-frequency branch only exercises the default frequency
  (mirrors the C++ code path when ``optionletFrequency_`` is unset).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from pquantlib import qassert
from pquantlib.cashflows.coupon_pricer import IborCouponPricer, set_coupon_pricer
from pquantlib.cashflows.floating_rate_coupon import FloatingRateCoupon
from pquantlib.cashflows.ibor_leg import ibor_leg
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.instruments.cap_floor import Cap, CapFloorType, Floor
from pquantlib.payoffs import OptionType
from pquantlib.pricingengines.black_formula import (
    bachelier_black_formula_implied_vol,
    black_formula_implied_std_dev,
)
from pquantlib.pricingengines.capfloor.black_capfloor_engine import (
    BachelierCapFloorEngine,
    BlackCapFloorEngine,
)
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.volatility.capfloor.cap_floor_term_vol_surface import (
    CapFloorTermVolSurface,
)
from pquantlib.termstructures.volatility.optionlet.stripped_optionlet_base import (
    StrippedOptionletBase,
)
from pquantlib.termstructures.volatility.volatility_type import VolatilityType
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendar import Calendar
from pquantlib.time.date import Date
from pquantlib.time.date_generation import DateGeneration
from pquantlib.time.period import Period
from pquantlib.time.schedule import Schedule
from pquantlib.time.time_unit import TimeUnit

if TYPE_CHECKING:
    from pquantlib.indexes.ibor_index import IborIndex
    from pquantlib.termstructures.protocols import YieldTermStructureProtocol


# C++ initial guess for the implied-std-dev Newton solve.
_FIRST_GUESS_STD_DEV: float = 0.14


def _build_cap(
    *,
    length: Period,
    index: IborIndex,
    strike: float,
    reference_date: Date,
) -> Cap:
    """Build a vanilla cap on ``index`` with ``length`` maturity at ``strike``.

    Mirrors C++ ``MakeCapFloor(CapFloor::Cap, length, index, strike, 0*Days)``
    with default settings: forward-start = ``today + 0 days``, paymentTenor
    = index.tenor, ModifiedFollowing payment adjustment, fixingDays =
    index.fixingDays, Backward date generation, Actual360 day count
    (inherited from the index).
    """
    cal = index.fixing_calendar()
    bdc = index.business_day_convention()
    # Forward-start date: the C++ MakeCapFloor delegates to
    # MakeVanillaSwap which sets the effective date to
    # ``cal.advance(today, fixingDays, Days)`` and the termination to
    # ``effective + tenor`` (the un-adjusted nominal end). The schedule
    # generation then handles BDC. Using the BDC-adjusted end here
    # would let stub periods appear (e.g. 24M caps where the BDC bump
    # creates a 2-day stub) — pass the un-adjusted nominal end and let
    # Schedule do the adjustment.
    start = cal.advance(
        reference_date,
        index.fixing_days(),
        TimeUnit.Days,
        BusinessDayConvention.Following,
    )
    end = start + length  # no BDC; let Schedule.from_rule adjust
    schedule = Schedule.from_rule(
        start,
        end,
        index.tenor(),
        cal,
        bdc,
        bdc,
        DateGeneration.Backward,
        index.end_of_month(),
    )
    leg = ibor_leg(
        schedule,
        index,
        nominals=[1.0],
        payment_adjustment=bdc,
        payment_calendar=cal,
        fixing_days=index.fixing_days(),
    )
    # Attach a trivial IborCouponPricer so ``adjusted_fixing`` (called
    # by ``CapFloor.setup_arguments``) resolves to ``index_fixing``.
    # The Black/Bachelier engines don't read this pricer's caplet
    # prices — they redo the per-optionlet Black evaluation
    # themselves — so the trivial choice is correct.
    set_coupon_pricer(leg, IborCouponPricer())
    return Cap(leg, [strike])


class OptionletStripper1(StrippedOptionletBase):
    """Strip caplet vols from cap term vols (caplet-by-caplet Newton solve)."""

    def __init__(
        self,
        term_vol_surface: CapFloorTermVolSurface,
        ibor_index: IborIndex,
        *,
        switch_strike: float | None = None,
        accuracy: float = 1.0e-6,
        max_iter: int = 100,
        discount_curve: YieldTermStructureProtocol | None = None,
        volatility_type: VolatilityType = VolatilityType.ShiftedLognormal,
        displacement: float = 0.0,
        dont_throw: bool = False,
        optionlet_frequency: Period | None = None,
    ) -> None:
        # # C++ parity: OptionletStripper1::OptionletStripper1 (which
        # delegates to ``OptionletStripper`` for the index-frequency
        # tenor walk).
        self._term_vol_surface: CapFloorTermVolSurface = term_vol_surface
        self._ibor_index: IborIndex = ibor_index
        self._discount_curve: YieldTermStructureProtocol | None = discount_curve
        self._volatility_type: VolatilityType = volatility_type
        self._displacement: float = displacement
        self._accuracy: float = accuracy
        self._max_iter: int = max_iter
        self._dont_throw: bool = dont_throw
        self._floating_switch_strike: bool = switch_strike is None
        self._switch_strike: float = (
            0.0 if switch_strike is None else float(switch_strike)
        )

        # # C++ parity: OptionletStripper ctor — walk by index tenor
        # from indexTenor up to max cap-floor tenor; capFloorLengths is
        # one-tenor longer per step so each successive cap adds exactly
        # one new optionlet.
        index_tenor = (
            optionlet_frequency if optionlet_frequency is not None else ibor_index.tenor()
        )
        max_cap_floor_tenor = term_vol_surface.option_tenors()[-1]
        self._optionlet_tenors: list[Period] = [index_tenor]
        # capFloorLengths_[0] = 2 * indexTenor (the first cap has 1
        # caplet starting at indexTenor and maturing at 2*indexTenor).
        self._cap_lengths: list[Period] = [self._optionlet_tenors[-1] + index_tenor]
        qassert.require(
            self._period_le(self._cap_lengths[-1], max_cap_floor_tenor),
            f"too short ({max_cap_floor_tenor}) capfloor term vol surface",
        )
        next_cap_floor_length = self._cap_lengths[-1] + index_tenor
        while self._period_le(next_cap_floor_length, max_cap_floor_tenor):
            self._optionlet_tenors.append(self._cap_lengths[-1])
            self._cap_lengths.append(next_cap_floor_length)
            next_cap_floor_length = next_cap_floor_length + index_tenor
        self._n_option_tenors: int = len(self._optionlet_tenors)
        self._strikes: list[float] = list(term_vol_surface.strikes())
        self._n_strikes: int = len(self._strikes)

        # Sized once now, populated by _perform_calculations on first
        # call.
        self._optionlet_dates: list[Date] = [Date()] * self._n_option_tenors
        self._optionlet_times: list[float] = [0.0] * self._n_option_tenors
        self._optionlet_payment_dates: list[Date] = [Date()] * self._n_option_tenors
        self._optionlet_accrual_periods: list[float] = [0.0] * self._n_option_tenors
        self._atm_optionlet_rate: list[float] = [0.0] * self._n_option_tenors
        # Per-(tenor, strike) caches.
        self._optionlet_volatilities: list[list[float]] = [
            [_FIRST_GUESS_STD_DEV] * self._n_strikes for _ in range(self._n_option_tenors)
        ]
        self._optionlet_std_devs: list[list[float]] = [
            [_FIRST_GUESS_STD_DEV] * self._n_strikes for _ in range(self._n_option_tenors)
        ]
        self._cap_floor_prices: list[list[float]] = [
            [0.0] * self._n_strikes for _ in range(self._n_option_tenors)
        ]
        self._optionlet_prices: list[list[float]] = [
            [0.0] * self._n_strikes for _ in range(self._n_option_tenors)
        ]
        self._cap_floor_vols: list[list[float]] = [
            [0.0] * self._n_strikes for _ in range(self._n_option_tenors)
        ]

        self._calculated: bool = False

    # --- internal -------------------------------------------------------

    @staticmethod
    def _period_le(a: Period, b: Period) -> bool:
        """C++-style ``Period::operator<=`` modulo normalized equality.

        PQuantLib's ``Period`` is a ``@dataclass(frozen=True)`` so its
        equality compares (length, units) field-by-field, meaning
        ``60M != 5Y`` even though they're the same period. The C++
        ``Period`` uses a min/max day-bound comparison that treats
        them as equal. We normalize both operands here before the
        ``<`` / ``==`` test, mirroring the C++ semantics for the
        common multiple-of-12 case.
        """
        return a.normalized() < b.normalized() or a.normalized() == b.normalized()

    def _ensure_calculated(self) -> None:
        if not self._calculated:
            self._perform_calculations()
            self._calculated = True

    def _discount_handle(self) -> YieldTermStructureProtocol:
        if self._discount_curve is not None:
            return self._discount_curve
        ts = self._ibor_index.forecast_term_structure()
        qassert.require(
            ts is not None,
            "no discount curve and IBOR index has no forecasting curve",
        )
        assert ts is not None
        return ts

    def _perform_calculations(self) -> None:  # noqa: PLR0915 (faithful port of C++ loop)
        # # C++ parity: OptionletStripper1::performCalculations
        # (optionletstripper1.cpp:61-178).
        ref_date = self._term_vol_surface.reference_date()
        dc: DayCounter = self._term_vol_surface.day_counter()

        # First pass: build a dummy cap per tenor to extract its last
        # coupon's fixing/payment dates + ATM rate. C++ uses a
        # BlackCapFloorEngine with dummy vol=0.20.
        dummy_engine = BlackCapFloorEngine(
            self._ibor_index.forecast_term_structure() or self._discount_handle(),
            0.20,
            dc,
        )
        for i in range(self._n_option_tenors):
            temp = _build_cap(
                length=self._cap_lengths[i],
                index=self._ibor_index,
                strike=0.04,
                reference_date=ref_date,
            )
            temp.set_pricing_engine(dummy_engine)
            last = temp.last_floating_rate_coupon()
            assert last is not None
            assert isinstance(last, FloatingRateCoupon)
            self._optionlet_dates[i] = last.fixing_date()
            self._optionlet_payment_dates[i] = last.date()
            self._optionlet_accrual_periods[i] = last.accrual_period()
            self._optionlet_times[i] = dc.year_fraction(
                ref_date, self._optionlet_dates[i]
            )
            # # C++ parity: lFRC->indexFixing(). We use
            # ``adjusted_fixing()`` to mirror what the engine stores
            # in ``args.forwards[i]`` (par-coupon-adjusted). Using
            # ``index_fixing()`` directly here would introduce a tiny
            # par-coupon mismatch and break the implied-vol round-
            # trip at the 1e-5 scale.
            self._atm_optionlet_rate[i] = last.adjusted_fixing()

        if self._floating_switch_strike:
            total = 0.0
            for i in range(self._n_option_tenors):
                total += self._atm_optionlet_rate[i]
            self._switch_strike = total / self._n_option_tenors

        discount_curve = self._discount_handle()
        strikes = self._strikes
        vol_quote = SimpleQuote(0.20)
        if self._volatility_type == VolatilityType.ShiftedLognormal:
            engine = BlackCapFloorEngine(
                discount_curve,
                vol_quote,
                dc,
                self._displacement,
            )
        elif self._volatility_type == VolatilityType.Normal:
            engine = BachelierCapFloorEngine(  # type: ignore[assignment]
                discount_curve,
                vol_quote,
                dc,
            )
        else:
            qassert.fail(f"unknown volatility type: {self._volatility_type}")

        for j in range(self._n_strikes):
            # # C++ parity: use out-of-the-money options — Cap above
            # ``switch_strike``, Floor below.
            cap_floor_type = (
                CapFloorType.Floor if strikes[j] < self._switch_strike else CapFloorType.Cap
            )
            option_type = (
                OptionType.Put if strikes[j] < self._switch_strike else OptionType.Call
            )
            previous_price = 0.0

            for i in range(self._n_option_tenors):
                self._cap_floor_vols[i][j] = self._term_vol_surface.volatility(
                    self._cap_lengths[i], strikes[j], True
                )
                vol_quote.set_value(self._cap_floor_vols[i][j])
                # Build cap/floor at this (length, strike).
                length = self._cap_lengths[i]
                if cap_floor_type == CapFloorType.Cap:
                    capfloor = _build_cap(
                        length=length,
                        index=self._ibor_index,
                        strike=strikes[j],
                        reference_date=ref_date,
                    )
                else:
                    # Floor — re-use the cap's floating leg with the
                    # Floor wrapper (same schedule/index, different
                    # payoff shape).
                    leg_helper = _build_cap(
                        length=length,
                        index=self._ibor_index,
                        strike=strikes[j],
                        reference_date=ref_date,
                    ).floating_leg()
                    capfloor = Floor(leg_helper, [strikes[j]])
                capfloor.set_pricing_engine(engine)
                self._cap_floor_prices[i][j] = capfloor.npv()
                self._optionlet_prices[i][j] = (
                    self._cap_floor_prices[i][j] - previous_price
                )
                previous_price = self._cap_floor_prices[i][j]
                df = discount_curve.discount(self._optionlet_payment_dates[i])
                optionlet_annuity = self._optionlet_accrual_periods[i] * df
                try:
                    if self._volatility_type == VolatilityType.ShiftedLognormal:
                        self._optionlet_std_devs[i][j] = black_formula_implied_std_dev(
                            option_type,
                            strikes[j],
                            self._atm_optionlet_rate[i],
                            self._optionlet_prices[i][j],
                            optionlet_annuity,
                            self._displacement,
                            self._optionlet_std_devs[i][j],
                            self._accuracy,
                            self._max_iter,
                        )
                    else:
                        # Normal — bachelier implied vol returns sigma
                        # (not sigma * sqrt(t)).
                        sigma = bachelier_black_formula_implied_vol(
                            option_type,
                            strikes[j],
                            self._atm_optionlet_rate[i],
                            self._optionlet_times[i],
                            self._optionlet_prices[i][j],
                            optionlet_annuity,
                        )
                        self._optionlet_std_devs[i][j] = (
                            math.sqrt(self._optionlet_times[i]) * sigma
                        )
                except Exception:
                    if self._dont_throw:
                        self._optionlet_std_devs[i][j] = 0.0
                    else:
                        raise
                self._optionlet_volatilities[i][j] = self._optionlet_std_devs[i][j] / (
                    math.sqrt(self._optionlet_times[i])
                    if self._optionlet_times[i] > 0
                    else 1.0
                )

    # --- StrippedOptionletBase interface --------------------------------

    def optionlet_strikes(self, i: int) -> list[float]:
        self._ensure_calculated()
        _ = i
        return list(self._strikes)

    def optionlet_volatilities(self, i: int) -> list[float]:
        self._ensure_calculated()
        return list(self._optionlet_volatilities[i])

    def optionlet_fixing_dates(self) -> list[Date]:
        self._ensure_calculated()
        return list(self._optionlet_dates)

    def optionlet_fixing_times(self) -> list[float]:
        self._ensure_calculated()
        return list(self._optionlet_times)

    def optionlet_maturities(self) -> int:
        return self._n_option_tenors

    def atm_optionlet_rates(self) -> list[float]:
        self._ensure_calculated()
        return list(self._atm_optionlet_rate)

    def day_counter(self) -> DayCounter:
        return self._term_vol_surface.day_counter()

    def calendar(self) -> Calendar:
        return self._term_vol_surface.calendar()

    def settlement_days(self) -> int:
        # The C++ termVolSurface_ may be moving-mode; expose 0 if not
        # provided (PQuantLib parity: term-vol surface holds the
        # settlement_days only in moving mode).
        try:
            return self._term_vol_surface.settlement_days()
        except Exception:
            return 0

    def business_day_convention(self) -> BusinessDayConvention:
        return self._term_vol_surface.business_day_convention()

    def volatility_type(self) -> VolatilityType:
        return self._volatility_type

    def displacement(self) -> float:
        return self._displacement

    def switch_strike(self) -> float:
        if self._floating_switch_strike:
            self._ensure_calculated()
        return self._switch_strike
