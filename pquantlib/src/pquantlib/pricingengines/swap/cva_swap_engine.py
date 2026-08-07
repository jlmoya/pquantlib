"""CounterpartyAdjSwapEngine — bilateral CVA/DVA-adjusted vanilla-swap engine.

# C++ parity: ql/pricingengines/swap/cvaswapengine.{hpp,cpp} (v1.43),
#             ``class CounterpartyAdjSwapEngine : public VanillaSwap::engine``.

Sorensen-Bollier (1994) exposure model: the counterparty's exposure over each
remaining fixed-coupon period is approximated by a European swaption on the
*residual* swap struck at the base swap's fair rate, and the investor's
exposure by the mirror-image receiver swaption::

    NPV = risklessNPV
        - (1 - R_ctpty) * sum_i Swaption_i    * Q_ctpty(t_{i-1}, t_i)
        + (1 - R_invst) * sum_i RevSwaption_i * Q_invst(t_{i-1}, t_i)

Collateral and wrong-way risk are out of scope (rates and default are assumed
independent), as in C++.

Details that are easy to get wrong and are cross-validated:

* **The empty investor curve is not a zero curve.** When no investor default
  term structure is supplied the C++ constructors substitute
  ``FlatHazardRate(0, NullCalendar(), 1e-12, ctptyDTS->dayCounter())`` — the
  DVA term is tiny but NOT exactly zero, and it uses the *counterparty's* day
  counter.
* The default investor recovery is ``0.999``, so ``(1 - R_invst)`` is 1e-3.
* The swaptionlet strip starts at the first fixed pay date **on or after** the
  counterparty curve's reference date, and the first exposure window starts at
  that reference date, not at the swap's effective date.
* Each swaplet is a fresh ``MakeVanillaSwap`` whose tenor is expressed in
  **days** (``fixedPayDates.back() - swapletStart``) and whose effective and
  termination dates are then overridden — the tenor only steers the schedule
  generation, and every swaplet matures on the swap's last fixed pay date.
* ``results.fair_rate`` is **not** the fair rate of the adjusted swap; it is
  the approximation written at cvaswapengine.cpp:211-214.
* The engine only sets ``value`` and ``fair_rate``; the per-leg results are
  left unset.

Three constructors in C++ (an arbitrary swaption engine, a plain
``Volatility``, and a ``Handle<Quote>`` volatility) collapse into one Python
constructor whose ``swaption_engine_or_vol`` argument selects between them:
a ``PricingEngine`` is used as-is, while a ``float`` or a ``Quote`` is wrapped
in a ``BlackSwaptionEngine`` on the discount curve exactly as C++ does.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from pquantlib import qassert
from pquantlib.cashflows.fixed_rate_coupon import FixedRateCoupon
from pquantlib.cashflows.floating_rate_coupon import FloatingRateCoupon
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.exercise import EuropeanExercise
from pquantlib.instruments.fixed_vs_floating_swap import (
    FixedVsFloatingSwapArguments,
    FixedVsFloatingSwapResults,
)
from pquantlib.instruments.make_vanilla_swap import MakeVanillaSwap
from pquantlib.instruments.swap import SwapType
from pquantlib.instruments.swaption import Swaption
from pquantlib.pricingengines.generic_engine import GenericEngine
from pquantlib.pricingengines.pricing_engine import PricingEngine
from pquantlib.pricingengines.swap.discounting_swap_engine import DiscountingSwapEngine
from pquantlib.pricingengines.swaption.black_swaption_engine import BlackSwaptionEngine
from pquantlib.quotes.quote import Quote
from pquantlib.termstructures.credit.flat_hazard_rate import FlatHazardRate
from pquantlib.time.calendars.null_calendar import NullCalendar
from pquantlib.time.date import Date
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit

if TYPE_CHECKING:
    from pquantlib.indexes.ibor_index import IborIndex
    from pquantlib.termstructures.credit.default_probability_term_structure import (
        DefaultProbabilityTermStructure,
    )
    from pquantlib.termstructures.protocols import YieldTermStructureProtocol

#: C++ default ``invstRecoveryRate = 0.999`` (cvaswapengine.hpp:72/90/109).
_DEFAULT_INVESTOR_RECOVERY: float = 0.999
#: C++ ``FlatHazardRate(0, NullCalendar(), 1.e-12, ...)`` — the stand-in used
#: when the investor default curve handle is empty (cvaswapengine.cpp:46-48).
_DEFAULT_FREE_HAZARD_RATE: float = 1.0e-12


class CounterpartyAdjSwapEngine(
    GenericEngine[FixedVsFloatingSwapArguments, FixedVsFloatingSwapResults]
):
    """Bilateral (CVA and DVA) default-adjusted vanilla-swap engine.

    # C++ parity: ``class CounterpartyAdjSwapEngine`` (cvaswapengine.hpp:49-121).
    """

    def __init__(
        self,
        discount_curve: YieldTermStructureProtocol,
        swaption_engine_or_vol: PricingEngine | Quote | float,
        ctpty_dts: DefaultProbabilityTermStructure,
        ctpty_recovery_rate: float,
        invst_dts: DefaultProbabilityTermStructure | None = None,
        invst_recovery_rate: float = _DEFAULT_INVESTOR_RECOVERY,
    ) -> None:
        super().__init__(FixedVsFloatingSwapArguments(), FixedVsFloatingSwapResults())
        self._discount_curve: YieldTermStructureProtocol = discount_curve
        self._base_swap_engine: DiscountingSwapEngine = DiscountingSwapEngine(discount_curve)

        engine: PricingEngine
        if isinstance(swaption_engine_or_vol, PricingEngine):
            # C++ ctor #1 (cvaswapengine.cpp:33-55).
            engine = swaption_engine_or_vol
        else:
            # C++ ctors #2/#3 (cvaswapengine.cpp:57-106) — a BlackSwaptionEngine
            # on the SAME discount curve, from a Volatility or a Quote.
            engine = BlackSwaptionEngine(discount_curve, swaption_engine_or_vol)
        self._swaptionlet_engine: PricingEngine = engine

        self._default_ts: DefaultProbabilityTermStructure = ctpty_dts
        self._ctpty_recovery_rate: float = float(ctpty_recovery_rate)
        if invst_dts is None:
            # NOT a zero curve: 1e-12, on the counterparty's day counter.
            day_counter = ctpty_dts.day_counter()
            assert isinstance(day_counter, DayCounter)
            invst_dts = FlatHazardRate.with_settlement_days(
                0, NullCalendar(), _quote(_DEFAULT_FREE_HAZARD_RATE), day_counter
            )
        self._invst_dts: DefaultProbabilityTermStructure = invst_dts
        self._invst_recovery_rate: float = float(invst_recovery_rate)

        # C++ parity: registerWith(discountCurve / ctptyDTS / invstDTS_ /
        # swaptionEngine|blackVol).
        register = getattr(discount_curve, "register_with", None)
        if register is not None:
            register(self)
        ctpty_dts.register_with(self)
        self._invst_dts.register_with(self)
        if isinstance(swaption_engine_or_vol, PricingEngine | Quote):
            swaption_engine_or_vol.register_with(self)

    # --- engine ------------------------------------------------------------

    def calculate(self) -> None:
        # C++ parity: cvaswapengine.cpp:108-216.
        args = self._arguments
        results = self._results
        results.reset()

        qassert.require(args.nominal is not None, "non-constant nominals are not supported yet")
        assert args.nominal is not None

        price_date = self._default_ts.reference_date()

        cum_opt_val = 0.0
        cum_put_val = 0.0

        # ``FixedVsFloatingSwapArguments`` types its date lists as ``list[object]``;
        # they are Dates.
        fixed_pay_dates = cast("list[Date]", list(args.fixed_pay_dates))
        # C++ ``while (*nextFD < priceDate) ++nextFD;`` — walks off the end if
        # every fixed payment is in the past, exactly as here.
        next_index = 0
        while next_index < len(fixed_pay_dates) and fixed_pay_dates[next_index] < price_date:
            next_index += 1
        swaplet_start = price_date

        # Riskless NPV via a DiscountingSwapEngine sharing the discount curve:
        # C++ copies the legs and payer flags into the base engine's arguments
        # and calls calculate() directly rather than re-pricing an instrument.
        base_args = self._base_swap_engine.get_arguments()
        base_args.legs = list(args.legs)
        base_args.payer = list(args.payer)
        self._base_swap_engine.calculate()
        base_results = self._base_swap_engine.get_results()

        first_fixed = args.legs[0][0]
        qassert.require(
            isinstance(first_fixed, FixedRateCoupon),
            "dynamic cast of fixed leg coupon failed.",
        )
        assert isinstance(first_fixed, FixedRateCoupon)
        base_swap_rate = first_fixed.rate()

        base_swap_fair_rate = (
            -base_swap_rate * base_results.leg_npv[1] / base_results.leg_npv[0]
        )
        base_swap_npv = base_results.value
        assert base_swap_npv is not None

        reversed_type = (
            SwapType.Receiver if args.swap_type == SwapType.Payer else SwapType.Payer
        )

        first_float = args.legs[1][0]
        qassert.require(
            isinstance(first_float, FloatingRateCoupon),
            "dynamic cast of floating leg coupon failed.",
        )
        assert isinstance(first_float, FloatingRateCoupon)
        swap_index: IborIndex = first_float.index()  # type: ignore[assignment]

        last_fixed_date = fixed_pay_dates[-1]
        while next_index < len(fixed_pay_dates):
            next_fd = fixed_pay_dates[next_index]
            # C++ ``Period baseSwapsTenor(back().serialNumber() -
            # swapletStart.serialNumber(), Days)``.
            base_swaps_tenor = Period(
                last_fixed_date.serial_number() - swaplet_start.serial_number(),
                TimeUnit.Days,
            )
            swaplet = (
                MakeVanillaSwap(base_swaps_tenor, swap_index, base_swap_fair_rate)
                .with_type(args.swap_type)
                .with_nominal(args.nominal)
                .with_effective_date(swaplet_start)
                .with_termination_date(last_fixed_date)
                .build()
            )
            rev_swaplet = (
                MakeVanillaSwap(base_swaps_tenor, swap_index, base_swap_fair_rate)
                .with_type(reversed_type)
                .with_nominal(args.nominal)
                .with_effective_date(swaplet_start)
                .with_termination_date(last_fixed_date)
                .build()
            )

            swaptionlet = Swaption(swaplet, EuropeanExercise(swaplet_start))
            put_swaplet = Swaption(rev_swaplet, EuropeanExercise(swaplet_start))
            swaptionlet.set_pricing_engine(self._swaptionlet_engine)
            put_swaplet.set_pricing_engine(self._swaptionlet_engine)

            cum_opt_val += swaptionlet.npv() * self._default_ts.default_probability(
                swaplet_start, next_fd
            )
            cum_put_val += put_swaplet.npv() * self._invst_dts.default_probability(
                swaplet_start, next_fd
            )

            swaplet_start = next_fd
            next_index += 1

        results.value = (
            base_swap_npv
            - (1.0 - self._ctpty_recovery_rate) * cum_opt_val
            + (1.0 - self._invst_recovery_rate) * cum_put_val
        )
        # C++ cvaswapengine.cpp:211-214 — an approximation, not the true fair
        # rate of the adjusted swap.
        results.fair_rate = (
            -base_swap_rate
            * (
                base_results.leg_npv[1]
                - (1.0 - self._ctpty_recovery_rate) * cum_opt_val
                + (1.0 - self._invst_recovery_rate) * cum_put_val
            )
            / base_results.leg_npv[0]
        )


def _quote(value: float) -> Quote:
    from pquantlib.quotes.simple_quote import SimpleQuote  # noqa: PLC0415

    return SimpleQuote(value)


__all__ = ["CounterpartyAdjSwapEngine"]
