"""OptionletStripper1 — strip caplet vols from cap term vols.

# C++ parity: ql/termstructures/volatility/optionlet/optionletstripper1.{hpp,cpp}
# (v1.43).

The class consumes a ``CapFloorTermVolSurface`` + ``IborIndex`` and
back-solves caplet-by-caplet implied vols that reproduce the cap NPVs
at the input vols. C++ uses ``MakeCapFloor`` factories + Black/
Bachelier engines + ``blackFormulaImpliedStdDev`` (Newton iteration).

All shared state — the tenor walk, the date/time/strike/vol grid and the
whole ``StrippedOptionletBase`` read interface — lives in
:class:`~pquantlib.termstructures.volatility.optionlet.optionlet_stripper.OptionletStripper`,
exactly as in C++. This module contributes only the constructor's extra
matrices and ``_perform_calculations``.

PQuantLib divergences (pre-existing; see the module-level notes in
``optionlet_stripper.py`` and the ``align()`` list in the wave report):

- ``_build_cap`` hand-rolls the leg instead of delegating to the ported
  :class:`~pquantlib.instruments.make_cap_floor.MakeCapFloor`, and it does
  NOT drop the first caplet the way ``MakeCapFloor`` does for a
  ``0*Days`` forward start (makecapfloor.cpp:51-53).
- ``bachelier_black_formula_implied_vol`` returns 0 where C++'s
  ``bachelierBlackFormulaImpliedVol`` recovers the input vol for the
  shortest Normal optionlet at the far strike — see the probe key
  ``euribor3m_normal.optionlet_volatilities[0][2]``.
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
from pquantlib.patterns.observable_settings import ObservableSettings
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
from pquantlib.termstructures.volatility.optionlet.optionlet_stripper import (
    OptionletStripper,
)
from pquantlib.termstructures.volatility.volatility_type import VolatilityType
from pquantlib.time.date import Date
from pquantlib.time.date_generation import DateGeneration
from pquantlib.time.period import Period
from pquantlib.time.schedule import Schedule

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
    evaluation_date: Date,
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
    # # C++ parity: MakeCapFloor delegates to MakeVanillaSwap, whose start
    # date is (makevanillaswap.cpp:67-77):
    #
    #     Date refDate = Settings::instance().evaluationDate();
    #     refDate  = iborIndex_->fixingCalendar().adjust(refDate);
    #     spotDate = iborIndex_->valueDate(refDate);
    #
    # Two details matter and both used to be wrong here:
    #  * "today" is the EVALUATION date, not the term-vol surface's
    #    reference date. They coincide only for a fixed-reference surface
    #    pinned to the evaluation date.
    #  * the evaluation date is rolled onto the index's fixing calendar
    #    FIRST. Without that, an evaluation date on a holiday advances two
    #    business days from the holiday itself and the whole schedule lands
    #    two business days early. Pinned by the probe's
    #    ``euribor6m_holiday_evaldate`` scenario (eval date = 1 May 2024,
    #    a TARGET holiday).
    #
    # The termination date is ``start + tenor`` un-adjusted: passing a
    # BDC-adjusted end would let stub periods appear (e.g. 24M caps where
    # the BDC bump creates a 2-day stub). Schedule does the adjustment.
    start = index.value_date(cal.adjust(evaluation_date))
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


class OptionletStripper1(OptionletStripper):
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
        # # C++ parity: OptionletStripper1::OptionletStripper1
        # (optionletstripper1.cpp:37-59). The tenor walk, the vector sizing
        # and the whole read interface belong to the base.
        super().__init__(
            term_vol_surface,
            ibor_index,
            discount_curve,
            volatility_type,
            displacement,
            optionlet_frequency,
        )
        self._accuracy: float = accuracy
        self._max_iter: int = max_iter
        self._dont_throw: bool = dont_throw
        # # C++ parity: ``floatingSwitchStrike_(switchStrike == Null<Rate>())``
        # (optionletstripper1.cpp:49).
        self._floating_switch_strike: bool = switch_strike is None
        self._switch_strike: float = (
            0.0 if switch_strike is None else float(switch_strike)
        )

        # # C++ parity: optionletstripper1.cpp:52-58 — the matrices that are
        # OptionletStripper1's alone. ``optionletStDevs_`` starts at the
        # Newton first guess; the price/vol matrices start at zero.
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

    # --- internal -------------------------------------------------------

    def _perform_calculations(self) -> None:  # noqa: PLR0915 (faithful port of C++ loop)
        # # C++ parity: OptionletStripper1::performCalculations
        # (optionletstripper1.cpp:61-178).
        # # C++ parity: optionletstripper1.cpp:64-65 — the fixing TIMES are
        # measured from the term-vol surface's reference date, while the cap
        # SCHEDULES start from Settings::evaluationDate() (via MakeCapFloor →
        # MakeVanillaSwap). The two are distinct inputs; do not conflate them.
        ref_date = self._term_vol_surface.reference_date()
        eval_date = ObservableSettings().evaluation_date
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
                evaluation_date=eval_date,
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
        # # C++ parity: ``const std::vector<Rate>& strikes =
        # termVolSurface_->strikes();`` (optionletstripper1.cpp:99).
        strikes = list(self._term_vol_surface.strikes())
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
                        evaluation_date=eval_date,
                    )
                else:
                    # Floor — re-use the cap's floating leg with the
                    # Floor wrapper (same schedule/index, different
                    # payoff shape).
                    leg_helper = _build_cap(
                        length=length,
                        index=self._ibor_index,
                        strike=strikes[j],
                        evaluation_date=eval_date,
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

    # --- OptionletStripper1's own inspectors ----------------------------
    #
    # The whole StrippedOptionletBase read interface is inherited from
    # OptionletStripper — # C++ parity: optionletstripper.cpp:87-175.

    def switch_strike(self) -> float:
        # # C++ parity: OptionletStripper1::switchStrike
        # (optionletstripper1.cpp:199-203) — calculate() only when the strike
        # floats, because a pinned strike is already final.
        if self._floating_switch_strike:
            self._ensure_calculated()
        return self._switch_strike

    def cap_floor_prices(self) -> list[list[float]]:
        """# C++ parity: ``OptionletStripper1::capFloorPrices`` (…1.cpp:184-187)."""
        self._ensure_calculated()
        return [list(row) for row in self._cap_floor_prices]

    def cap_floor_volatilities(self) -> list[list[float]]:
        """# C++ parity: ``OptionletStripper1::capFloorVolatilities`` (…1.cpp:189-192)."""
        self._ensure_calculated()
        return [list(row) for row in self._cap_floor_vols]

    def optionlet_prices(self) -> list[list[float]]:
        """# C++ parity: ``OptionletStripper1::optionletPrices`` (…1.cpp:194-197)."""
        self._ensure_calculated()
        return [list(row) for row in self._optionlet_prices]
