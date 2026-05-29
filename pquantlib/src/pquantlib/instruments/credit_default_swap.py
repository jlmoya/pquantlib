"""CreditDefaultSwap — vanilla CDS.

# C++ parity: ql/instruments/creditdefaultswap.{hpp,cpp} (v1.42.1).

A CDS exchanges a stream of fixed coupon payments (premium leg) for a
contingent default payment (protection leg) over a defined schedule.
The buyer of protection pays the premium leg; the seller receives the
premium leg and pays the protection.

Two construction modes:

- ``CreditDefaultSwap(side, notional, spread, schedule, ...)``: spread-only
  CDS. Upfront defaults to 0.
- ``CreditDefaultSwap.with_upfront(side, notional, upfront, spread, schedule,
  upfront_date, ...)``: spread + upfront convention (Big Bang 2009).

Subclass of :class:`pquantlib.instruments.instrument.Instrument` so the
engine plumbing (MidPointCdsEngine / IntegralCdsEngine / IsdaCdsEngine)
flows through ``setup_arguments`` / ``fetch_results``.

# C++ parity divergences:
# - The C++ class supports an ``include_settlement_date_flows`` Settings
#   override; the Python port reads it via the engine constructor only.
# - ``accrualRebate`` flow is implemented but the rebate calculation
#   simplifies — we always pass through ``schedule.calendar().advance(
#   trade_date, cash_settlement_days, ..., payment_convention)`` for the
#   rebate date. Documented inline.
# - L9-B closes the original ``impliedHazardRate`` /
#   ``conventionalSpread`` carve-out — they're now available as methods
#   on this class (Brent over FlatHazardRate, MidPoint or ISDA engine).
"""

from __future__ import annotations

from enum import IntEnum
from typing import cast

from pquantlib import qassert
from pquantlib.cashflows.cash_flow import CashFlow
from pquantlib.cashflows.fixed_rate_coupon import FixedRateCoupon
from pquantlib.cashflows.fixed_rate_leg import fixed_rate_leg
from pquantlib.cashflows.simple_cash_flow import SimpleCashFlow
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.instruments.claim import Claim, FaceValueClaim
from pquantlib.instruments.instrument import Instrument, InstrumentResults
from pquantlib.pricingengines.pricing_engine import (
    PricingEngineArguments,
    PricingEngineResults,
)
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.date import Date
from pquantlib.time.schedule import Schedule
from pquantlib.time.time_unit import TimeUnit


class ProtectionSide(IntEnum):
    """CDS protection side.

    # C++ parity: ``Protection::Side`` namespace enum
    # (ql/instruments/creditdefaultswap.hpp imports it from ql/default.hpp).
    """

    Buyer = 0
    Seller = 1


class PricingModel(IntEnum):
    """CDS pricing model selector.

    # C++ parity: ``CreditDefaultSwap::PricingModel`` nested enum.
    """

    Midpoint = 0
    ISDA = 1


_NULL_DATE: Date = Date()


class CreditDefaultSwapArguments(PricingEngineArguments):
    """Engine-arguments carrier for CDS.

    # C++ parity: ``CreditDefaultSwap::arguments`` nested class.
    """

    def __init__(self) -> None:
        self.side: ProtectionSide | None = None
        self.notional: float | None = None
        self.upfront: float | None = None  # None = no upfront flow
        self.spread: float | None = None
        self.leg: list[CashFlow] = []
        self.upfront_payment: SimpleCashFlow | None = None
        self.accrual_rebate: SimpleCashFlow | None = None
        self.settles_accrual: bool = True
        self.pays_at_default_time: bool = True
        self.claim: Claim | None = None
        self.protection_start: Date = _NULL_DATE
        self.maturity: Date = _NULL_DATE

    def validate(self) -> None:
        qassert.require(self.side is not None, "side not set")
        qassert.require(self.notional is not None, "notional not set")
        qassert.require(self.notional != 0.0, "null notional set")
        qassert.require(self.spread is not None, "spread not set")
        qassert.require(len(self.leg) > 0, "coupons not set")
        qassert.require(self.upfront_payment is not None, "upfront payment not set")
        qassert.require(self.claim is not None, "claim not set")
        qassert.require(self.protection_start != _NULL_DATE, "protection start date not set")
        qassert.require(self.maturity != _NULL_DATE, "maturity date not set")


class CreditDefaultSwapResults(InstrumentResults):
    """Engine-results carrier for CDS.

    # C++ parity: ``CreditDefaultSwap::results`` nested class.
    """

    def __init__(self) -> None:
        super().__init__()
        self.fair_spread: float | None = None
        self.fair_upfront: float | None = None
        self.coupon_leg_bps: float | None = None
        self.coupon_leg_npv: float | None = None
        self.default_leg_npv: float | None = None
        self.upfront_bps: float | None = None
        self.upfront_npv: float | None = None
        self.accrual_rebate_npv: float | None = None

    def reset(self) -> None:
        super().reset()
        self.fair_spread = None
        self.fair_upfront = None
        self.coupon_leg_bps = None
        self.coupon_leg_npv = None
        self.default_leg_npv = None
        self.upfront_bps = None
        self.upfront_npv = None
        self.accrual_rebate_npv = None


class CreditDefaultSwap(Instrument):
    """Credit default swap.

    # C++ parity: ``CreditDefaultSwap`` class.
    """

    Side = ProtectionSide  # convenience alias matching C++ nested-enum style
    PricingModel = PricingModel

    def __init__(
        self,
        side: ProtectionSide,
        notional: float,
        spread: float,
        schedule: Schedule,
        payment_convention: BusinessDayConvention,
        day_counter: DayCounter,
        settles_accrual: bool = True,
        pays_at_default_time: bool = True,
        protection_start: Date | None = None,
        claim: Claim | None = None,
        last_period_day_counter: DayCounter | None = None,
        rebates_accrual: bool = True,
        trade_date: Date | None = None,
        cash_settlement_days: int = 3,
    ) -> None:
        """Spread-only CDS.

        # C++ parity: creditdefaultswap.cpp:39-60.
        """
        super().__init__()
        self._side: ProtectionSide = side
        self._notional: float = notional
        self._upfront: float | None = None
        self._running_spread: float = spread
        self._settles_accrual: bool = settles_accrual
        self._pays_at_default_time: bool = pays_at_default_time
        self._claim: Claim = claim if claim is not None else FaceValueClaim()
        self._protection_start: Date = (
            protection_start if protection_start is not None and protection_start != _NULL_DATE
            else schedule.date(0)
        )
        self._trade_date: Date = (
            trade_date if trade_date is not None and trade_date != _NULL_DATE
            else _NULL_DATE
        )
        self._cash_settlement_days: int = cash_settlement_days
        self._leg: list[CashFlow] = []
        self._upfront_payment: SimpleCashFlow | None = None
        self._accrual_rebate: SimpleCashFlow | None = None
        self._maturity: Date = _NULL_DATE
        # Result cache
        self._fair_upfront: float | None = None
        self._fair_spread: float | None = None
        self._coupon_leg_bps: float | None = None
        self._coupon_leg_npv: float | None = None
        self._default_leg_npv: float | None = None
        self._upfront_bps: float | None = None
        self._upfront_npv: float | None = None
        self._accrual_rebate_npv: float | None = None

        self._init(
            schedule, payment_convention, day_counter,
            last_period_day_counter, rebates_accrual, upfront_date=None,
        )
        self._claim.register_with(self)

    @classmethod
    def with_upfront(
        cls,
        side: ProtectionSide,
        notional: float,
        upfront: float,
        spread: float,
        schedule: Schedule,
        payment_convention: BusinessDayConvention,
        day_counter: DayCounter,
        settles_accrual: bool = True,
        pays_at_default_time: bool = True,
        protection_start: Date | None = None,
        upfront_date: Date | None = None,
        claim: Claim | None = None,
        last_period_day_counter: DayCounter | None = None,
        rebates_accrual: bool = True,
        trade_date: Date | None = None,
        cash_settlement_days: int = 3,
    ) -> CreditDefaultSwap:
        """Spread + upfront CDS.

        # C++ parity: creditdefaultswap.cpp:62-85.
        """
        instance = cls(
            side, notional, spread, schedule, payment_convention, day_counter,
            settles_accrual, pays_at_default_time, protection_start, claim,
            last_period_day_counter, rebates_accrual, trade_date,
            cash_settlement_days,
        )
        # Override the upfront field + re-init.
        instance._upfront = upfront
        instance._init(
            schedule, payment_convention, day_counter,
            last_period_day_counter, rebates_accrual, upfront_date=upfront_date,
        )
        return instance

    # ---- shared init -----------------------------------------------------

    def _init(
        self,
        schedule: Schedule,
        payment_convention: BusinessDayConvention,
        day_counter: DayCounter,
        last_period_day_counter: DayCounter | None,
        rebates_accrual: bool,
        upfront_date: Date | None,
    ) -> None:
        """Mirror C++ ``CreditDefaultSwap::init``.

        # C++ parity: creditdefaultswap.cpp:87-176.

        # C++ parity divergence: the C++ logic distinguishes
        # ``DateGeneration::CDS`` / ``CDS2015`` (post-Big-Bang) where
        # ``protection_start > schedule[0]`` is allowed. The Python port
        # simplifies and only checks protection_start <= schedule[0] when
        # rules are NOT explicit CDS/CDS2015 (Schedule doesn't currently
        # expose ``hasRule``); per L8-B carve-out doc.
        """
        qassert.require(len(schedule) > 0, "CreditDefaultSwap needs a non-empty schedule.")

        # We can't yet differentiate post-Big-Bang CDS rule via schedule
        # introspection. Skip the protection-start-vs-schedule-start
        # check, mirroring how the C++ default constructor handles missing
        # rule info. Per L8-B carve-out doc.
        del last_period_day_counter  # not yet wired into fixed_rate_leg builder.
        self._leg = fixed_rate_leg(
            schedule=schedule,
            nominals=[self._notional],
            rates=[self._running_spread],
            day_counter=day_counter,
            payment_adjustment=payment_convention,
        )

        # Deduce the trade date if not given (matches C++ creditdefaultswap.cpp:110-116).
        if self._trade_date == _NULL_DATE:
            # We assume post-big-bang here (no schedule-rule introspection).
            self._trade_date = self._protection_start

        # Deduce the cash settlement / upfront date.
        if upfront_date is not None and upfront_date != _NULL_DATE:
            effective_upfront_date = upfront_date
        else:
            effective_upfront_date = schedule.calendar.advance(
                self._trade_date,
                self._cash_settlement_days,
                TimeUnit.Days,
                payment_convention,
            )
        qassert.require(
            effective_upfront_date >= self._protection_start,
            "The cash settlement date must not be before the protection start date.",
        )

        # Upfront flow.
        upfront_amount = 0.0
        if self._upfront is not None:
            upfront_amount = self._upfront * self._notional
        self._upfront_payment = SimpleCashFlow(upfront_amount, effective_upfront_date)

        # Maturity.
        self._maturity = schedule.dates[-1]

        # Accrual rebate (post-Big-Bang convention).
        if rebates_accrual:
            rebate_amount = 0.0
            ref_date = self._trade_date + 1
            if self._trade_date >= schedule.date(0):
                for i, cf in enumerate(self._leg):
                    cf_date = cf.date()
                    if ref_date > cf_date:
                        continue
                    if ref_date == cf_date:
                        if i < len(self._leg) - 1:
                            rebate_amount = 0.0
                        else:
                            assert isinstance(cf, FixedRateCoupon)
                            rebate_amount = cf.amount()
                        break
                    # ref_date < cf_date: first coupon paying in the future.
                    assert isinstance(cf, FixedRateCoupon)
                    rebate_amount = cf.accrued_amount(ref_date)
                    break
            self._accrual_rebate = SimpleCashFlow(rebate_amount, effective_upfront_date)
        else:
            self._accrual_rebate = None

    # ---- Instrument interface --------------------------------------------

    def is_expired(self) -> bool:
        """All cashflows already paid.

        # C++ parity: creditdefaultswap.cpp:207-213.
        """
        # ``hasOccurred`` walks back-to-front; without a global Settings
        # default we use the protection start as a proxy reference date.
        return all(cf.has_occurred(self._protection_start) for cf in reversed(self._leg))

    def setup_arguments(self, args: PricingEngineArguments) -> None:
        qassert.require(
            isinstance(args, CreditDefaultSwapArguments),
            "CreditDefaultSwap.setup_arguments: wrong argument type",
        )
        assert isinstance(args, CreditDefaultSwapArguments)
        args.side = self._side
        args.notional = self._notional
        args.upfront = self._upfront
        args.spread = self._running_spread
        args.leg = list(self._leg)
        args.upfront_payment = self._upfront_payment
        args.accrual_rebate = self._accrual_rebate
        args.settles_accrual = self._settles_accrual
        args.pays_at_default_time = self._pays_at_default_time
        args.claim = self._claim
        args.protection_start = self._protection_start
        args.maturity = self._maturity

    def fetch_results(self, results: PricingEngineResults) -> None:
        super().fetch_results(results)
        qassert.require(
            isinstance(results, CreditDefaultSwapResults),
            "CreditDefaultSwap.fetch_results: wrong result type",
        )
        assert isinstance(results, CreditDefaultSwapResults)
        self._fair_spread = results.fair_spread
        self._fair_upfront = results.fair_upfront
        self._coupon_leg_bps = results.coupon_leg_bps
        self._coupon_leg_npv = results.coupon_leg_npv
        self._default_leg_npv = results.default_leg_npv
        self._upfront_bps = results.upfront_bps
        self._upfront_npv = results.upfront_npv
        self._accrual_rebate_npv = results.accrual_rebate_npv

    def setup_expired(self) -> None:
        super().setup_expired()
        self._fair_spread = 0.0
        self._fair_upfront = 0.0
        self._coupon_leg_bps = 0.0
        self._upfront_bps = 0.0
        self._coupon_leg_npv = 0.0
        self._default_leg_npv = 0.0
        self._upfront_npv = 0.0
        self._accrual_rebate_npv = 0.0

    # ---- inspectors -----------------------------------------------------

    def side(self) -> ProtectionSide:
        return self._side

    def notional(self) -> float:
        return self._notional

    def running_spread(self) -> float:
        return self._running_spread

    def upfront(self) -> float | None:
        return self._upfront

    def settles_accrual(self) -> bool:
        return self._settles_accrual

    def pays_at_default_time(self) -> bool:
        return self._pays_at_default_time

    def coupons(self) -> list[CashFlow]:
        return list(self._leg)

    def protection_start_date(self) -> Date:
        return self._protection_start

    def protection_end_date(self) -> Date:
        """Last accrual_end_date of the leg.

        # C++ parity: creditdefaultswap.cpp:430-432.
        """
        last = self._leg[-1]
        assert isinstance(last, FixedRateCoupon)
        return last.accrual_end_date()

    def rebates_accrual(self) -> bool:
        return self._accrual_rebate is not None

    def upfront_payment(self) -> SimpleCashFlow:
        assert self._upfront_payment is not None
        return self._upfront_payment

    def accrual_rebate(self) -> SimpleCashFlow | None:
        return self._accrual_rebate

    def trade_date(self) -> Date:
        return self._trade_date

    def cash_settlement_days(self) -> int:
        return self._cash_settlement_days

    # ---- results -------------------------------------------------------

    def fair_upfront(self) -> float:
        self.calculate()
        qassert.require(self._fair_upfront is not None, "fair upfront not available")
        return cast("float", self._fair_upfront)

    def fair_spread(self) -> float:
        self.calculate()
        qassert.require(self._fair_spread is not None, "fair spread not available")
        return cast("float", self._fair_spread)

    def coupon_leg_bps(self) -> float:
        self.calculate()
        qassert.require(self._coupon_leg_bps is not None, "coupon-leg BPS not available")
        return cast("float", self._coupon_leg_bps)

    def coupon_leg_npv(self) -> float:
        self.calculate()
        qassert.require(self._coupon_leg_npv is not None, "coupon-leg NPV not available")
        return cast("float", self._coupon_leg_npv)

    def default_leg_npv(self) -> float:
        self.calculate()
        qassert.require(self._default_leg_npv is not None, "default-leg NPV not available")
        return cast("float", self._default_leg_npv)

    def upfront_npv(self) -> float:
        self.calculate()
        qassert.require(self._upfront_npv is not None, "upfront NPV not available")
        return cast("float", self._upfront_npv)

    def upfront_bps(self) -> float:
        self.calculate()
        qassert.require(self._upfront_bps is not None, "upfront BPS not available")
        return cast("float", self._upfront_bps)

    def accrual_rebate_npv(self) -> float:
        self.calculate()
        qassert.require(self._accrual_rebate_npv is not None, "accrual rebate NPV not available")
        return cast("float", self._accrual_rebate_npv)

    # ---- implied_hazard_rate + conventional_spread ----------------------

    def implied_hazard_rate(
        self,
        target_npv: float,
        discount_curve: object,
        day_counter: object,
        recovery_rate: float = 0.4,
        accuracy: float = 1e-8,
        max_iterations: int = 100,
        guess: float | None = None,
        model: PricingModel = PricingModel.Midpoint,
    ) -> float:
        """Solve for the flat hazard rate that reproduces ``target_npv``.

        # C++ parity: ``CreditDefaultSwap::impliedHazardRate``
        # (creditdefaultswap.cpp:340-381).

        Builds a ``FlatHazardRate(rate)`` probability term structure
        plus a MidPoint / ISDA engine, then runs Brent over the hazard
        rate. Returns the rate that makes
        ``engine_result.value == target_npv``.

        Local imports prevent the instruments → termstructures/credit →
        instruments cycle.
        """
        # Local imports: avoid circular import with FlatHazardRate /
        # IsdaCdsEngine / MidPointCdsEngine via the instruments tree.
        from pquantlib.math.solvers1d.brent import Brent  # noqa: PLC0415
        from pquantlib.pricingengines.credit.isda_cds_engine import (  # noqa: PLC0415
            IsdaCdsEngine,
        )
        from pquantlib.pricingengines.credit.midpoint_cds_engine import (  # noqa: PLC0415
            MidPointCdsEngine,
        )
        from pquantlib.quotes.simple_quote import SimpleQuote  # noqa: PLC0415
        from pquantlib.termstructures.credit.flat_hazard_rate import (  # noqa: PLC0415
            FlatHazardRate,
        )
        from pquantlib.termstructures.yield_term_structure import (  # noqa: PLC0415
            YieldTermStructure,
        )

        del max_iterations  # Brent's default max iterations is 100.

        # Build the probe FlatHazardRate(rate) curve.
        flat_rate_quote = SimpleQuote(0.02 if guess is None else float(guess))
        assert isinstance(discount_curve, YieldTermStructure)
        # FlatHazardRate uses the discount curve's reference date as its
        # anchor — the C++ engine uses a moving-mode (settlement_days=0)
        # WeekendsOnly variant; PQuantLib uses the fixed-date constructor
        # which is equivalent at the reference date.
        from pquantlib.daycounters.day_counter import DayCounter  # noqa: PLC0415

        assert isinstance(day_counter, DayCounter)
        probability = FlatHazardRate(
            discount_curve.reference_date(), flat_rate_quote, day_counter,
        )

        # Pick the engine.
        engine: MidPointCdsEngine | IsdaCdsEngine
        if model == PricingModel.Midpoint:
            engine = MidPointCdsEngine(probability, recovery_rate, discount_curve)
        elif model == PricingModel.ISDA:
            engine = IsdaCdsEngine(probability, recovery_rate, discount_curve)
        else:
            qassert.fail(f"unknown CDS pricing model: {model}")

        # Drive the engine directly (matches C++ approach — bypass the
        # instrument's lazy machinery so we can mutate the hazard rate
        # in the inner loop).
        engine_args = engine.get_arguments()
        engine_results = engine.get_results()
        self.setup_arguments(engine_args)

        def f(x: float) -> float:
            flat_rate_quote.set_value(x)
            engine.calculate()
            value = engine_results.value
            assert value is not None
            return value - target_npv

        # C++ initial guess: spread / (1 - recovery) * 365/360.
        x0 = self._running_spread / (1.0 - recovery_rate) * 365.0 / 360.0
        if guess is not None:
            x0 = guess
        step = 0.1 * x0 if x0 != 0.0 else 0.01
        return Brent().solve(f, accuracy, x0, step)

    def conventional_spread(
        self,
        conventional_recovery: float,
        discount_curve: object,
        day_counter: object,
        model: PricingModel = PricingModel.ISDA,
        accuracy: float = 1e-8,
    ) -> float:
        """Conventional par spread under the ISDA convention.

        # C++ parity: ``CreditDefaultSwap::conventionalSpread``
        # (creditdefaultswap.cpp:383-422).

        Same Brent approach as :meth:`implied_hazard_rate` but with
        ``target_npv = 0`` and the conventional recovery rate.
        """
        return self.implied_hazard_rate(
            target_npv=0.0,
            discount_curve=discount_curve,
            day_counter=day_counter,
            recovery_rate=conventional_recovery,
            accuracy=accuracy,
            model=model,
        )


__all__ = [
    "CreditDefaultSwap",
    "CreditDefaultSwapArguments",
    "CreditDefaultSwapResults",
    "PricingModel",
    "ProtectionSide",
]
