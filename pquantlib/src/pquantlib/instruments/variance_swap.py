"""Variance swap.

# C++ parity: ql/instruments/varianceswap.{hpp,cpp} (v1.43).

A forward contract on realized variance: at maturity the long pays

    notional * (realized_variance - variance_strike)

discounted to today. The instrument itself carries no market data — the
whole price comes from the engine, which supplies the fair variance
(``variance()``) alongside the NPV. See
:mod:`pquantlib.pricingengines.forward.replicating_variance_swap_engine`
for the Demeterfi-Derman-Kamal-Zou static-replication engine.

.. warning::
   As in C++, this class does not manage *seasoned* variance swaps: the
   accrued realized variance between ``start_date`` and today is not
   modelled, and ``start_date`` is carried for the record only — no
   engine in QuantLib reads it.
"""

from __future__ import annotations

from pquantlib import qassert
from pquantlib.instruments.instrument import Instrument, InstrumentResults
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.position import PositionType
from pquantlib.pricingengines.pricing_engine import (
    PricingEngineArguments,
    PricingEngineResults,
)
from pquantlib.time.date import Date

_NULL_DATE: Date = Date()


class VarianceSwapArguments(PricingEngineArguments):
    """Engine arguments for a variance swap.

    # C++ parity: ``VarianceSwap::arguments``. ``strike`` and
    # ``notional`` start at ``Null<Real>()``; the Python port uses
    # ``None`` for the same "not supplied" state.
    """

    def __init__(self) -> None:
        self.position: PositionType = PositionType.Long
        self.strike: float | None = None
        self.notional: float | None = None
        self.start_date: Date = _NULL_DATE
        self.maturity_date: Date = _NULL_DATE

    def validate(self) -> None:
        """# C++ parity: ``VarianceSwap::arguments::validate`` (varianceswap.cpp:65-72)."""
        qassert.require(self.strike is not None, "no strike given")
        assert self.strike is not None
        qassert.require(self.strike > 0.0, "negative or null strike given")
        qassert.require(self.notional is not None, "no notional given")
        assert self.notional is not None
        qassert.require(self.notional > 0.0, "negative or null notional given")
        qassert.require(self.start_date != _NULL_DATE, "null start date given")
        qassert.require(self.maturity_date != _NULL_DATE, "null maturity date given")


class VarianceSwapResults(InstrumentResults):
    """Results carrier: NPV plus the engine's fair variance.

    # C++ parity: ``VarianceSwap::results``.
    """

    def __init__(self) -> None:
        super().__init__()
        self.variance: float | None = None

    def reset(self) -> None:
        super().reset()
        self.variance = None


class VarianceSwap(Instrument):
    """Forward contract on realized variance.

    # C++ parity: ``VarianceSwap(position, strike, notional, startDate,
    # maturityDate)``.

    ``strike`` is a *variance* strike (e.g. ``0.04`` for 20% vol), not a
    volatility strike, and ``notional`` is a variance notional; both are
    required to be strictly positive by ``VarianceSwapArguments.validate``.
    """

    def __init__(
        self,
        position: PositionType,
        strike: float,
        notional: float,
        start_date: Date,
        maturity_date: Date,
    ) -> None:
        super().__init__()
        self._position: PositionType = position
        self._strike: float = strike
        self._notional: float = notional
        self._start_date: Date = start_date
        self._maturity_date: Date = maturity_date
        self._variance: float | None = None

    # --- inspectors --------------------------------------------------------

    def strike(self) -> float:
        return self._strike

    def position(self) -> PositionType:
        return self._position

    def start_date(self) -> Date:
        return self._start_date

    def maturity_date(self) -> Date:
        return self._maturity_date

    def notional(self) -> float:
        return self._notional

    # --- results -----------------------------------------------------------

    def variance(self) -> float:
        """Fair variance implied by the engine.

        # C++ parity: ``VarianceSwap::variance`` (varianceswap.cpp:35-39).
        """
        self.calculate()
        qassert.require(self._variance is not None, "result not available")
        assert self._variance is not None
        return self._variance

    # --- Instrument interface ---------------------------------------------

    def is_expired(self) -> bool:
        """Whether maturity has passed at the global evaluation date.

        # C++ parity: ``VarianceSwap::isExpired`` is
        # ``detail::simple_event(maturityDate_).hasOccurred()``
        # (varianceswap.cpp:80-82), i.e. ``Event::hasOccurred`` with the
        # reference date defaulted to ``Settings::evaluationDate()`` and
        # the inclusion flag defaulted to
        # ``Settings::includeReferenceDateEvents()`` (event.cpp:28-39):
        # the flag decides the boundary only, ``maturity < ref`` when set
        # and ``maturity <= ref`` when not.
        #
        # Consequence worth knowing: a NULL maturity date has serial 0 and
        # is therefore <= any evaluation date, so a swap built with one
        # reports itself expired and prices to 0 instead of tripping the
        # ``no maturity date given`` check in ``validate``.
        """
        settings = ObservableSettings()
        ref_date = settings.evaluation_date_or_today()
        if settings.include_reference_date_events:
            return self._maturity_date < ref_date
        return self._maturity_date <= ref_date

    def setup_arguments(self, args: PricingEngineArguments) -> None:
        """# C++ parity: ``VarianceSwap::setupArguments`` (varianceswap.cpp:46-55)."""
        qassert.require(
            isinstance(args, VarianceSwapArguments),
            "wrong argument type (expected VarianceSwapArguments)",
        )
        assert isinstance(args, VarianceSwapArguments)
        args.position = self._position
        args.strike = self._strike
        args.notional = self._notional
        args.start_date = self._start_date
        args.maturity_date = self._maturity_date

    def fetch_results(self, results: PricingEngineResults) -> None:
        """# C++ parity: ``VarianceSwap::fetchResults`` (varianceswap.cpp:57-61)."""
        super().fetch_results(results)
        qassert.require(
            isinstance(results, VarianceSwapResults),
            "no variance-carrying results returned from pricing engine "
            "(expected VarianceSwapResults subclass)",
        )
        assert isinstance(results, VarianceSwapResults)
        self._variance = results.variance

    def setup_expired(self) -> None:
        """# C++ parity: ``VarianceSwap::setupExpired`` (varianceswap.cpp:41-44)."""
        super().setup_expired()
        self._variance = None


__all__ = ["VarianceSwap", "VarianceSwapArguments", "VarianceSwapResults"]
