"""NthToDefault — n-th-to-default basket CDS instrument.

# C++ parity: ql/experimental/credit/nthtodefault.{hpp,cpp} (v1.42.1).

An NTD swap exchanges protection against the n-th default in a
basket of underlying credits for premium payments over the
protected notional amount.

The pricing follows the Hull-White 2004 framework (or any equivalent
default-loss model wired via ``BasketProtocol.prob_at_least_n_events``).
The instrument is a single product instance bound to a basket plus a
premium schedule; pricing is delegated to a registered engine (see
``IntegralNTDEngine`` for the canonical implementation).
"""

from __future__ import annotations

from typing import cast

from pquantlib import qassert
from pquantlib.cashflows.cash_flow import CashFlow
from pquantlib.cashflows.fixed_rate_leg import fixed_rate_leg
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.experimental.credit.basket_protocol import BasketProtocol
from pquantlib.instruments.credit_default_swap import ProtectionSide
from pquantlib.instruments.instrument import Instrument, InstrumentResults
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.pricingengines.pricing_engine import (
    PricingEngineArguments,
    PricingEngineResults,
)
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.date import Date
from pquantlib.time.schedule import Schedule


class NthToDefaultArguments(PricingEngineArguments):
    """Engine-arguments carrier for NthToDefault.

    # C++ parity: ``NthToDefault::arguments`` nested class.
    """

    def __init__(self) -> None:
        self.basket: BasketProtocol | None = None
        self.side: ProtectionSide | None = None
        self.premium_leg: list[CashFlow] = []
        self.ntd_order: int | None = None
        self.settle_premium_accrual: bool = True
        self.notional: float | None = None
        self.premium_rate: float | None = None
        self.upfront_rate: float | None = None

    def validate(self) -> None:
        qassert.require(
            self.basket is not None and len(self.basket.names()) > 0,
            "no basket given",
        )
        qassert.require(self.side is not None, "side not set")
        qassert.require(self.premium_rate is not None, "no premium rate given")
        qassert.require(self.upfront_rate is not None, "no upfront rate given")
        qassert.require(self.notional is not None, "no notional given")
        qassert.require(self.ntd_order is not None, "no NTD order given")


class NthToDefaultResults(InstrumentResults):
    """Engine-results carrier for NthToDefault.

    # C++ parity: ``NthToDefault::results`` nested class.
    """

    def __init__(self) -> None:
        super().__init__()
        self.premium_value: float | None = None
        self.protection_value: float | None = None
        self.upfront_premium_value: float | None = None
        self.fair_premium: float | None = None
        # Note: ``error_estimate`` is inherited from InstrumentResults.

    def reset(self) -> None:
        super().reset()
        self.premium_value = None
        self.protection_value = None
        self.upfront_premium_value = None
        self.fair_premium = None


class NthToDefault(Instrument):
    """N-th to default basket CDS.

    # C++ parity: ``NthToDefault`` class.

    Construction binds the instrument to a basket plus a premium schedule.
    The basket's reference date must lie at or before the schedule's
    start date (mirrors C++ check). The fixed-rate premium leg is built
    eagerly at construction.
    """

    def __init__(
        self,
        basket: BasketProtocol,
        n: int,
        side: ProtectionSide,
        premium_schedule: Schedule,
        upfront_rate: float,
        premium_rate: float,
        day_counter: DayCounter,
        nominal: float,
        settle_premium_accrual: bool,
    ) -> None:
        """Build the NTD instrument.

        # C++ parity: nthtodefault.cpp:31-62.
        """
        super().__init__()
        qassert.require(
            n <= basket.size(),
            "NTD order provided is larger than the basket size.",
        )
        qassert.require(
            basket.ref_date() <= premium_schedule.start_date,
            "Basket did not exist before contract start.",
        )

        self._basket: BasketProtocol = basket
        self._n: int = n
        self._side: ProtectionSide = side
        self._nominal: float = nominal
        self._premium_schedule: Schedule = premium_schedule
        self._premium_rate: float = premium_rate
        self._upfront_rate: float = upfront_rate
        self._day_counter: DayCounter = day_counter
        self._settle_premium_accrual: bool = settle_premium_accrual

        self._premium_leg: list[CashFlow] = fixed_rate_leg(
            schedule=premium_schedule,
            nominals=[nominal],
            rates=[premium_rate],
            day_counter=day_counter,
            payment_adjustment=BusinessDayConvention.Unadjusted,
        )

        # Result cache (mirrors C++ mutable fields).
        self._premium_value: float | None = None
        self._protection_value: float | None = None
        self._upfront_premium_value: float | None = None
        self._fair_premium: float | None = None
        self._error_estimate_local: float | None = None

    # ---- Instrument interface --------------------------------------

    def is_expired(self) -> bool:
        """The last coupon's date is in the past.

        # C++ parity: nthtodefault.cpp:68-70 uses
        # ``detail::simple_event(premiumLeg_.back()->date()).hasOccurred()``
        # which checks against ``Settings::evaluationDate()``. The
        # Python port reads the global evaluation date from
        # ``ObservableSettings`` (falling back to today when not set —
        # matches the C++ default).
        """
        today = ObservableSettings().evaluation_date_or_today()
        last_date = self._premium_leg[-1].date()
        return last_date <= today

    def setup_arguments(self, args: PricingEngineArguments) -> None:
        qassert.require(
            isinstance(args, NthToDefaultArguments),
            "NthToDefault.setup_arguments: wrong argument type",
        )
        assert isinstance(args, NthToDefaultArguments)
        args.basket = self._basket
        args.side = self._side
        args.premium_leg = list(self._premium_leg)
        args.ntd_order = self._n
        args.settle_premium_accrual = self._settle_premium_accrual
        args.notional = self._nominal
        args.premium_rate = self._premium_rate
        args.upfront_rate = self._upfront_rate

    def fetch_results(self, results: PricingEngineResults) -> None:
        super().fetch_results(results)
        qassert.require(
            isinstance(results, NthToDefaultResults),
            "NthToDefault.fetch_results: wrong result type",
        )
        assert isinstance(results, NthToDefaultResults)
        self._premium_value = results.premium_value
        self._protection_value = results.protection_value
        self._upfront_premium_value = results.upfront_premium_value
        self._fair_premium = results.fair_premium
        self._error_estimate_local = results.error_estimate

    def setup_expired(self) -> None:
        super().setup_expired()
        self._premium_value = 0.0
        self._protection_value = 0.0
        self._upfront_premium_value = 0.0
        self._fair_premium = 0.0
        self._error_estimate_local = 0.0

    # ---- inspectors -----------------------------------------------

    def premium(self) -> float:
        return self._premium_rate

    def nominal(self) -> float:
        return self._nominal

    def day_counter(self) -> DayCounter:
        return self._day_counter

    def side(self) -> ProtectionSide:
        return self._side

    def rank(self) -> int:
        return self._n

    def basket_size(self) -> int:
        return self._basket.size()

    def basket(self) -> BasketProtocol:
        return self._basket

    def maturity(self) -> Date:
        """Last coupon-end date of the premium schedule.

        # C++ parity: ``NthToDefault::maturity`` returns
        # ``premiumSchedule_.endDate()``.
        """
        return self._premium_schedule.end_date

    def premium_leg(self) -> list[CashFlow]:
        return list(self._premium_leg)

    def settle_premium_accrual(self) -> bool:
        return self._settle_premium_accrual

    def upfront_rate(self) -> float:
        return self._upfront_rate

    # ---- results --------------------------------------------------

    def fair_premium(self) -> float:
        self.calculate()
        qassert.require(self._fair_premium is not None, "fair premium not available")
        return cast("float", self._fair_premium)

    def premium_leg_npv(self) -> float:
        """Premium leg NPV + upfront-premium contribution.

        # C++ parity: nthtodefault.cpp:79-86 — sum of ``premiumValue_``
        # and ``upfrontPremiumValue_``.
        """
        self.calculate()
        qassert.require(self._premium_value is not None, "premium leg not available")
        qassert.require(
            self._upfront_premium_value is not None, "upfront value not available"
        )
        return cast("float", self._premium_value) + cast(
            "float", self._upfront_premium_value
        )

    def protection_leg_npv(self) -> float:
        self.calculate()
        qassert.require(
            self._protection_value is not None, "protection leg not available"
        )
        return cast("float", self._protection_value)

    def error_estimate(self) -> float:
        self.calculate()
        qassert.require(
            self._error_estimate_local is not None, "error estimate not available"
        )
        return cast("float", self._error_estimate_local)


__all__ = [
    "NthToDefault",
    "NthToDefaultArguments",
    "NthToDefaultResults",
]
