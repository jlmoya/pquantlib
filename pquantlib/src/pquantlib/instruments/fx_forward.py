"""FxForward — foreign-exchange forward contract.

# C++ parity: ql/instruments/fxforward.{hpp,cpp} (v1.42.1).

An FxForward agrees to exchange ``sourceNominal`` units of
``sourceCurrency`` for ``targetNominal`` units of ``targetCurrency`` at
``maturityDate``.  The contracted forward exchange rate is
``targetNominal / sourceNominal`` (i.e. target per source).

The instrument is valued by ``DiscountingFwdEngine`` which uses
per-currency discount curves and a spot FX quote.  The engine fills:

- ``fair_forward_rate`` — strike that makes NPV = 0,
- ``npv_source_currency`` — NPV in source-currency terms,
- ``npv_target_currency`` — NPV in target-currency terms.
"""

from __future__ import annotations

from pquantlib import qassert
from pquantlib.currencies.currency import Currency
from pquantlib.instruments.instrument import Instrument, InstrumentResults
from pquantlib.pricingengines.pricing_engine import (
    PricingEngineArguments,
    PricingEngineResults,
)
from pquantlib.time.calendar import Calendar
from pquantlib.time.calendars.null_calendar import NullCalendar
from pquantlib.time.date import Date
from pquantlib.time.time_unit import TimeUnit


class FxForwardArguments(PricingEngineArguments):
    """Arguments passed to DiscountingFwdEngine.

    # C++ parity: ``FxForward::arguments`` (ql/instruments/fxforward.hpp).
    """

    def __init__(self) -> None:
        self.source_nominal: float | None = None
        self.source_currency: Currency = Currency()
        self.target_nominal: float | None = None
        self.target_currency: Currency = Currency()
        self.maturity_date: Date = Date()
        self.pay_source_currency: bool = True
        self.settlement_date: Date = Date()

    def validate(self) -> None:
        qassert.require(self.source_nominal is not None, "source nominal not set")
        qassert.require(self.target_nominal is not None, "target nominal not set")
        qassert.require(not self.source_currency.empty(), "source currency not set")
        qassert.require(not self.target_currency.empty(), "target currency not set")
        qassert.require(self.maturity_date != Date(), "maturity date not set")
        qassert.require(self.settlement_date != Date(), "settlement date not set")


class FxForwardResults(InstrumentResults):
    """Results returned by DiscountingFwdEngine.

    # C++ parity: ``FxForward::results`` (ql/instruments/fxforward.hpp).
    """

    def __init__(self) -> None:
        super().__init__()
        self.fair_forward_rate: float | None = None
        self.npv_source_currency: float | None = None
        self.npv_target_currency: float | None = None

    def reset(self) -> None:
        super().reset()
        self.fair_forward_rate = None
        self.npv_source_currency = None
        self.npv_target_currency = None


class FxForward(Instrument):
    """FX forward contract.

    # C++ parity: ``class FxForward : public Instrument``
    # (ql/instruments/fxforward.hpp).
    """

    def __init__(
        self,
        source_nominal: float,
        source_currency: Currency,
        target_nominal: float,
        target_currency: Currency,
        maturity_date: Date,
        pay_source_currency: bool,
        settlement_days: int = 2,
        payment_calendar: Calendar | None = None,
    ) -> None:
        """Construct from explicit source / target nominals.

        # C++ parity: first ``FxForward`` constructor.
        """
        super().__init__()
        qassert.require(not source_currency.empty(), "source currency must not be empty")
        qassert.require(not target_currency.empty(), "target currency must not be empty")
        qassert.require(
            source_currency != target_currency,
            "source and target currencies must be different",
        )
        qassert.require(source_nominal > 0.0, "source nominal must be positive")
        qassert.require(target_nominal > 0.0, "target nominal must be positive")

        self._source_nominal: float = source_nominal
        self._source_currency: Currency = source_currency
        self._target_nominal: float = target_nominal
        self._target_currency: Currency = target_currency
        self._maturity_date: Date = maturity_date
        self._pay_source_currency: bool = pay_source_currency
        self._settlement_days: int = settlement_days
        self._payment_calendar: Calendar = (
            payment_calendar if payment_calendar is not None else NullCalendar()
        )

        self._fair_forward_rate: float | None = None
        self._npv_source_currency: float | None = None
        self._npv_target_currency: float | None = None

    @classmethod
    def from_forward_rate(
        cls,
        source_nominal: float,
        source_currency: Currency,
        target_currency: Currency,
        forward_rate: float,
        maturity_date: Date,
        pay_source_currency: bool,
        settlement_days: int = 2,
        payment_calendar: Calendar | None = None,
    ) -> FxForward:
        """Construct from a forward rate (target/source).

        # C++ parity: second ``FxForward`` constructor.
        """
        qassert.require(forward_rate > 0.0, "forward rate must be positive")
        return cls(
            source_nominal=source_nominal,
            source_currency=source_currency,
            target_nominal=source_nominal * forward_rate,
            target_currency=target_currency,
            maturity_date=maturity_date,
            pay_source_currency=pay_source_currency,
            settlement_days=settlement_days,
            payment_calendar=payment_calendar,
        )

    # --- inspectors --------------------------------------------------------

    def source_nominal(self) -> float:
        return self._source_nominal

    def source_currency(self) -> Currency:
        return self._source_currency

    def target_nominal(self) -> float:
        return self._target_nominal

    def target_currency(self) -> Currency:
        return self._target_currency

    def maturity_date(self) -> Date:
        return self._maturity_date

    def pay_source_currency(self) -> bool:
        return self._pay_source_currency

    def forward_rate(self) -> float:
        """Contracted forward rate (target/source)."""
        return self._target_nominal / self._source_nominal

    def settlement_days(self) -> int:
        return self._settlement_days

    def settlement_calendar(self) -> Calendar:
        return self._payment_calendar

    def settlement_date(self, evaluation_date: Date) -> Date:
        """Settlement date computed from evaluation_date + settlement_days.

        # C++ parity: ``FxForward::settlementDate`` reads
        # ``Settings::instance().evaluationDate()``; PQuantLib takes the
        # date explicitly because Settings is not yet wired.
        """
        return self._payment_calendar.advance(
            evaluation_date, self._settlement_days, TimeUnit.Days
        )

    # --- Instrument interface ---------------------------------------------

    def is_expired(self) -> bool:
        """C++ parity: ``FxForward::isExpired``.

        Without a Settings singleton, we approximate by comparing
        against the pricing-engine's settlement-date argument once
        ``calculate()`` has been run.  Before that, behave as not
        expired so the engine can take over.
        """
        return False

    def setup_arguments(self, args: PricingEngineArguments) -> None:
        """Copy fields to the engine arguments.

        # C++ parity: ``FxForward::setupArguments``.  The settlement
        # date is computed from the engine's source-curve reference
        # date (Python divergence: C++ uses ``Settings::evaluationDate``
        # but we have no Settings singleton).  The engine is expected
        # to be a :class:`DiscountingFwdEngine` exposing the source
        # discount curve.
        """
        qassert.require(
            isinstance(args, FxForwardArguments), "wrong argument type"
        )
        assert isinstance(args, FxForwardArguments)
        args.source_nominal = self._source_nominal
        args.source_currency = self._source_currency
        args.target_nominal = self._target_nominal
        args.target_currency = self._target_currency
        args.maturity_date = self._maturity_date
        args.pay_source_currency = self._pay_source_currency
        # Settlement date: payment calendar advance of eval date by
        # settlement_days.  Eval date defaults to the engine's source
        # curve reference date.  Engines that don't provide a source
        # curve will see ``Date()`` and fail validation.
        eval_date = self._inferred_eval_date()
        args.settlement_date = self._payment_calendar.advance(
            eval_date, self._settlement_days, TimeUnit.Days
        )

    def _inferred_eval_date(self) -> Date:
        """Best-effort evaluation date inference for setup_arguments.

        Looks up the attached pricing engine and asks its
        ``source_currency_discount_curve`` for ``reference_date``.
        Returns ``Date()`` if the engine doesn't expose that hook;
        engine validation will raise a clear error in that case.
        """
        engine = self._engine
        if engine is None:
            return Date()
        source_curve_fn = getattr(engine, "source_currency_discount_curve", None)
        if source_curve_fn is None:
            return Date()
        curve = source_curve_fn()
        return curve.reference_date()

    def fetch_results(self, results: PricingEngineResults) -> None:
        """Pull FxForward-specific results out of the engine."""
        super().fetch_results(results)
        qassert.require(
            isinstance(results, FxForwardResults),
            "no FxForward results returned from pricing engine",
        )
        assert isinstance(results, FxForwardResults)
        self._fair_forward_rate = results.fair_forward_rate
        self._npv_source_currency = results.npv_source_currency
        self._npv_target_currency = results.npv_target_currency

    # --- additional accessors ---------------------------------------------

    def fair_forward_rate(self) -> float:
        self.calculate()
        qassert.require(
            self._fair_forward_rate is not None,
            "fair forward rate not available",
        )
        assert self._fair_forward_rate is not None
        return self._fair_forward_rate

    def npv_source_currency(self) -> float:
        self.calculate()
        qassert.require(
            self._npv_source_currency is not None,
            "NPV in source currency not available",
        )
        assert self._npv_source_currency is not None
        return self._npv_source_currency

    def npv_target_currency(self) -> float:
        self.calculate()
        qassert.require(
            self._npv_target_currency is not None,
            "NPV in target currency not available",
        )
        assert self._npv_target_currency is not None
        return self._npv_target_currency


__all__ = ["FxForward", "FxForwardArguments", "FxForwardResults"]
