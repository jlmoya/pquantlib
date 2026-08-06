"""FuturesConvAdjustmentQuote — futures/forward convexity adjustment of an index.

# C++ parity: ql/quotes/futuresconvadjustmentquote.{hpp,cpp} (v1.43)

``value()`` is ``HullWhite::convexityBias(futuresPrice, t, T, sigma, a)`` where
``t`` and ``T`` are the index-day-count year fractions from the *evaluation
date* to the futures date and to the index maturity respectively. The result is
cached in ``rate_`` and dropped by ``update()``; the quote registers with the
evaluation date, so rolling the date invalidates the cache.

Reference for the bias itself: G. Kirikos, D. Novak, "Convexity Conundrums",
Risk Magazine, March 1997.

# C++ parity divergence (Handle vs object): C++ takes ``Handle<Quote>`` for the
futures price, the volatility and the mean reversion. This port threads the
``Quote`` directly (see ``cashflows/cms_coupon_pricer.py``); ``None`` models
C++'s *empty* handle, which ``isValid()`` explicitly tests for.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pquantlib import qassert
from pquantlib.models.shortrate.onefactor.hull_white import HullWhite
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.quotes.quote import Quote
from pquantlib.time import imm

if TYPE_CHECKING:
    from pquantlib.daycounters.day_counter import DayCounter
    from pquantlib.indexes.ibor_index import IborIndex
    from pquantlib.time.date import Date


class FuturesConvAdjustmentQuote(Quote):
    """Quote for the futures-convexity adjustment of an index.

    # C++ parity: ``class FuturesConvAdjustmentQuote : public Quote, public
    # Observer``.
    """

    __slots__ = (
        "_day_counter",
        "_futures_date",
        "_futures_quote",
        "_index_maturity_date",
        "_mean_reversion",
        "_rate",
        "_volatility",
    )

    def __init__(
        self,
        index: IborIndex,
        futures_date: Date,
        futures_quote: Quote | None,
        volatility: Quote | None,
        mean_reversion: Quote | None,
    ) -> None:
        super().__init__()
        self._day_counter: DayCounter = index.day_counter()
        self._futures_date: Date = futures_date
        self._index_maturity_date: Date = index.maturity_date(futures_date)
        self._futures_quote: Quote | None = futures_quote
        self._volatility: Quote | None = volatility
        self._mean_reversion: Quote | None = mean_reversion
        self._rate: float | None = None
        for element in (futures_quote, volatility, mean_reversion):
            if element is not None:
                element.register_with(self)
        ObservableSettings().register_with(self)

    @classmethod
    def from_imm_code(
        cls,
        index: IborIndex,
        imm_code: str,
        futures_quote: Quote | None,
        volatility: Quote | None,
        mean_reversion: Quote | None,
    ) -> FuturesConvAdjustmentQuote:
        """Build from an IMM code instead of an explicit futures date.

        # C++ parity: the second ``FuturesConvAdjustmentQuote`` constructor,
        # which delegates to the first with ``IMM::date(immCode)``. Python has
        # no constructor overloading, so the delegating overload becomes a
        # named factory.
        #
        # The reference date is passed EXPLICITLY. C++ ``IMM::date`` defaults
        # its reference date to ``Settings::instance().evaluationDate()``
        # (ql/time/imm.cpp:130-132), whereas ``pquantlib.time.imm.date``
        # defaults to ``Date.todays_date()`` — a pre-existing divergence in
        # that module. Passing the evaluation date here keeps this class
        # faithful regardless.
        """
        reference_date = ObservableSettings().evaluation_date_or_today()
        return cls(
            index,
            imm.date(imm_code, reference_date),
            futures_quote,
            volatility,
            mean_reversion,
        )

    # --- Quote interface --------------------------------------------------

    def value(self) -> float:
        if self._rate is None:
            qassert.require(
                self._futures_quote is not None
                and self._volatility is not None
                and self._mean_reversion is not None,
                "empty Handle cannot be dereferenced",
            )
            assert self._futures_quote is not None
            assert self._volatility is not None
            assert self._mean_reversion is not None
            settlement_date = ObservableSettings().evaluation_date_or_today()
            start_time = self._day_counter.year_fraction(settlement_date, self._futures_date)
            index_maturity = self._day_counter.year_fraction(
                settlement_date, self._index_maturity_date
            )
            self._rate = HullWhite.convexity_bias(
                self._futures_quote.value(),
                start_time,
                index_maturity,
                self._volatility.value(),
                self._mean_reversion.value(),
            )
        return self._rate

    def is_valid(self) -> bool:
        return (
            self._futures_quote is not None
            and self._volatility is not None
            and self._mean_reversion is not None
            and self._futures_quote.is_valid()
            and self._volatility.is_valid()
            and self._mean_reversion.is_valid()
        )

    # --- Observer interface -----------------------------------------------

    def update(self) -> None:
        self._rate = None
        self.notify_observers()

    # --- inspectors -------------------------------------------------------

    def futures_value(self) -> float:
        assert self._futures_quote is not None
        return self._futures_quote.value()

    def volatility(self) -> float:
        assert self._volatility is not None
        return self._volatility.value()

    def mean_reversion(self) -> float:
        assert self._mean_reversion is not None
        return self._mean_reversion.value()

    def imm_date(self) -> Date:
        return self._futures_date
