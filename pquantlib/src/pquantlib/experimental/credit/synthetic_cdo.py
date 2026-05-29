"""SyntheticCDO — mezzanine CDO tranche instrument.

# C++ parity: ql/experimental/credit/syntheticcdo.{hpp,cpp} (v1.42.1).

A synthetic CDO exchanges a stream of premium payments on the
outstanding tranche notional for a contingent protection payment when
basket-wide losses pierce the attachment / detachment bounds.

For a buyer of protection (Side.Buyer), the instrument value is::

    V = V_protection - V_premium

For a seller, the signs flip (the seller earns premium, owes
protection). The engine (IntegralCDOEngine / MidPointCDOEngine in
:mod:`pquantlib.pricingengines.credit`) populates the result struct;
the instrument exposes inspector methods that route through the
engine via ``calculate()``.

# C++ parity divergence: the C++ ctor takes an ``ext::optional<Real>``
# for ``notional`` (None = use basket.tranche_notional). The Python port
# uses a plain ``float | None`` with the same semantics.
"""

from __future__ import annotations

from enum import IntEnum
from typing import TYPE_CHECKING, cast

from pquantlib import qassert
from pquantlib.cashflows.cash_flow import CashFlow
from pquantlib.cashflows.coupon import Coupon
from pquantlib.cashflows.fixed_rate_leg import fixed_rate_leg
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.experimental.credit.basket import Basket
from pquantlib.instruments.instrument import Instrument, InstrumentResults
from pquantlib.pricingengines.pricing_engine import (
    PricingEngineArguments,
    PricingEngineResults,
)
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.date import Date

if TYPE_CHECKING:
    from pquantlib.time.schedule import Schedule


class ProtectionSide(IntEnum):
    """CDO protection side.

    # C++ parity: ``Protection::Side`` namespace enum (ql/default.hpp).
    # Mirrors the same-named enum in
    # ``pquantlib.instruments.credit_default_swap`` (intentional —
    # both CDS and CDO share the convention).
    """

    Buyer = 0
    Seller = 1


class SyntheticCDOArguments(PricingEngineArguments):
    """Engine-arguments carrier for SyntheticCDO.

    # C++ parity: ``SyntheticCDO::arguments`` nested class.
    """

    def __init__(self) -> None:
        self.basket: Basket | None = None
        self.side: ProtectionSide | None = None
        self.normalized_leg: list[CashFlow] = []
        self.upfront_rate: float | None = None
        self.running_rate: float | None = None
        self.leverage_factor: float = 1.0
        self.day_counter: DayCounter | None = None
        self.payment_convention: BusinessDayConvention = (
            BusinessDayConvention.Following
        )

    def validate(self) -> None:
        # C++ parity: SyntheticCDO::arguments::validate.
        qassert.require(self.side is not None, "side not set")
        qassert.require(
            self.basket is not None and len(self.basket.names()) > 0,
            "no basket given",
        )
        qassert.require(self.running_rate is not None, "no premium rate given")
        qassert.require(self.upfront_rate is not None, "no upfront rate given")
        qassert.require(self.day_counter is not None, "no day counter given")


class SyntheticCDOResults(InstrumentResults):
    """Engine-results carrier.

    # C++ parity: ``SyntheticCDO::results`` nested class.
    """

    def __init__(self) -> None:
        super().__init__()
        self.premium_value: float | None = None
        self.protection_value: float | None = None
        self.upfront_premium_value: float | None = None
        self.remaining_notional: float | None = None
        self.x_min: float | None = None
        self.x_max: float | None = None
        self.error: int = 0
        self.expected_tranche_loss: list[float] = []

    def reset(self) -> None:
        super().reset()
        self.premium_value = None
        self.protection_value = None
        self.upfront_premium_value = None
        self.remaining_notional = None
        self.x_min = None
        self.x_max = None
        self.error = 0
        self.expected_tranche_loss = []


class SyntheticCDO(Instrument):
    """Mezzanine synthetic CDO tranche.

    Parameters
    ----------
    basket
        Basket carrying the issuer pool + attach/detach structure +
        loss model.
    side
        ``ProtectionSide.Buyer`` or ``ProtectionSide.Seller``.
    schedule
        Premium-payment schedule.
    upfront_rate
        Upfront fee as a fraction of basket tranche notional.
    running_rate
        Running premium rate (per year, as a fraction).
    day_counter
        Day-count convention for the running premium.
    payment_convention
        Business-day-convention applied to coupon payment dates.
    notional
        Optional override of the tranche notional (default: use
        ``basket.tranche_notional()``). If a different value is given,
        ``leverage_factor = notional / basket.tranche_notional()``.

    # C++ parity divergence: the C++ ctor moves ``schedule`` into a
    # ``FixedRateLeg`` builder. The Python port calls the
    # :func:`pquantlib.cashflows.fixed_rate_leg.fixed_rate_leg` factory
    # with the equivalent ``with*`` kwargs.
    """

    def __init__(
        self,
        basket: Basket,
        side: ProtectionSide,
        schedule: Schedule,
        upfront_rate: float,
        running_rate: float,
        day_counter: DayCounter,
        payment_convention: BusinessDayConvention = (
            BusinessDayConvention.Following
        ),
        notional: float | None = None,
    ) -> None:
        super().__init__()
        qassert.require(len(basket.names()) > 0, "basket is empty")
        qassert.require(
            basket.ref_date() <= schedule.start_date,
            "Basket did not exist before contract start.",
        )

        self._basket: Basket = basket
        self._side: ProtectionSide = side
        self._upfront_rate: float = upfront_rate
        self._running_rate: float = running_rate
        self._day_counter: DayCounter = day_counter
        self._payment_convention: BusinessDayConvention = payment_convention

        # Leverage factor (C++ optional<Real> → Python float | None).
        if notional is None:
            self._leverage_factor: float = 1.0
        else:
            self._leverage_factor = notional / basket.tranche_notional()

        # Build the normalized fixed-rate leg.
        tranche_amt = basket.tranche_notional() * self._leverage_factor
        self._normalized_leg: list[CashFlow] = fixed_rate_leg(
            schedule=schedule,
            nominals=[tranche_amt],
            rates=[running_rate],
            day_counter=day_counter,
            payment_adjustment=payment_convention,
        )

        # Result cache fields.
        self._premium_value: float | None = None
        self._protection_value: float | None = None
        self._upfront_premium_value: float | None = None
        self._remaining_notional: float | None = None
        self._error: int = 0
        self._expected_tranche_loss: list[float] = []

        # Register with the basket + each issuer's curve (mirror C++
        # constructor's registerWith loop).
        basket.register_with(self)
        for i in range(len(basket.names())):
            curve = basket.pool().get(basket.names()[i]).default_probability(
                basket.pool().default_keys()[i],
            )
            curve.register_with(self)

    # ---- inspectors --------------------------------------------------------

    def basket(self) -> Basket:
        return self._basket

    def side(self) -> ProtectionSide:
        return self._side

    def leverage_factor(self) -> float:
        return self._leverage_factor

    def maturity(self) -> Date:
        """Last protection date (last coupon's accrual end)."""
        last = self._normalized_leg[-1]
        # The last leg item is a Coupon (FixedRateCoupon); we cast for
        # accrual_end_date access.
        coupon = cast("Coupon", last)
        return coupon.accrual_end_date()

    def is_expired(self) -> bool:
        # C++ parity: hasOccurred on the last payment date.
        # Use the last coupon's payment date (Coupon inherits from CashFlow).
        last = self._normalized_leg[-1]
        return last.has_occurred()

    def premium_value(self) -> float:
        self.calculate()
        assert self._premium_value is not None
        return self._premium_value

    def protection_value(self) -> float:
        self.calculate()
        assert self._protection_value is not None
        return self._protection_value

    def upfront_premium_value(self) -> float:
        self.calculate()
        assert self._upfront_premium_value is not None
        return self._upfront_premium_value

    def remaining_notional(self) -> float:
        self.calculate()
        assert self._remaining_notional is not None
        return self._remaining_notional

    def expected_tranche_loss(self) -> list[float]:
        self.calculate()
        return list(self._expected_tranche_loss)

    def error(self) -> int:
        self.calculate()
        return self._error

    def fair_premium(self) -> float:
        """Running premium that zeroes NPV given the upfront."""
        self.calculate()
        assert self._premium_value is not None
        assert self._protection_value is not None
        assert self._upfront_premium_value is not None
        qassert.require(
            self._premium_value != 0.0,
            "Attempted divide by zero while calculating syntheticCDO premium.",
        )
        return (
            self._running_rate
            * (self._protection_value - self._upfront_premium_value)
            / self._premium_value
        )

    def fair_upfront_premium(self) -> float:
        """Upfront fee that zeroes NPV given the running rate."""
        self.calculate()
        assert self._premium_value is not None
        assert self._protection_value is not None
        assert self._remaining_notional is not None
        return (
            (self._protection_value - self._premium_value)
            / self._remaining_notional
        )

    # ---- Instrument plumbing ----------------------------------------------

    def setup_arguments(self, args: PricingEngineArguments) -> None:
        # C++ parity: SyntheticCDO::setupArguments.
        a = cast("SyntheticCDOArguments", args)
        a.basket = self._basket
        a.side = self._side
        a.normalized_leg = list(self._normalized_leg)
        a.upfront_rate = self._upfront_rate
        a.running_rate = self._running_rate
        a.day_counter = self._day_counter
        a.payment_convention = self._payment_convention
        a.leverage_factor = self._leverage_factor

    def fetch_results(self, results: PricingEngineResults) -> None:
        # C++ parity: SyntheticCDO::fetchResults.
        Instrument.fetch_results(self, results)
        r = cast("SyntheticCDOResults", results)
        self._premium_value = r.premium_value
        self._protection_value = r.protection_value
        self._upfront_premium_value = r.upfront_premium_value
        self._remaining_notional = r.remaining_notional
        self._error = r.error
        self._expected_tranche_loss = list(r.expected_tranche_loss)

    def setup_expired(self) -> None:
        # C++ parity: SyntheticCDO::setupExpired.
        self._premium_value = 0.0
        self._protection_value = 0.0
        self._upfront_premium_value = 0.0
        self._remaining_notional = 1.0
        self._expected_tranche_loss = []
        self._npv = 0.0


__all__ = [
    "ProtectionSide",
    "SyntheticCDO",
    "SyntheticCDOArguments",
    "SyntheticCDOResults",
]
