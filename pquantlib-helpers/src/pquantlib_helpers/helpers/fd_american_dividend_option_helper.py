"""FDAmericanDividendOptionHelper — American FD dividend-option helper.

# Retired-API compat layer — NOT a port of C++ QuantLib v1.42.1.

Java parity:
``org.jquantlib.helpers.FDAmericanDividendOptionHelper`` (jquantlib-helpers).

Concrete subclass of
:class:`~pquantlib_helpers.helpers.fd_dividend_option_helper.FDDividendOptionHelper`
that wires an :class:`~pquantlib.exercise.AmericanExercise` and the
:class:`~pquantlib_helpers.pricingengines.vanilla.finitedifferences.fd_dividend_american_engine.FDDividendAmericanEngine`.

# Faithful-port note: the Java FDDividendAmericanHelper wraps
# FDDividendAmericanEngine, which in turn is
# ``FDEngineAdapter<FDAmericanCondition<FDDividendEngine>, ...>``.
# The FDAmericanCondition path does NOT do multi-period dividend stepping;
# dividends are carried in the option arguments but the American rollback
# ignores them (known JQuantLib ``@bug results are not overly reliable``).
# We reproduce this verbatim for full Java same-algorithm cross-validation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.exercise import AmericanExercise
from pquantlib.payoffs import OptionType
from pquantlib.time.calendar import Calendar
from pquantlib.time.date import Date
from pquantlib_helpers.helpers.fd_dividend_option_helper import (
    FDDividendOptionHelper,
)
from pquantlib_helpers.pricingengines.vanilla.finitedifferences.fd_dividend_american_engine import (
    FDDividendAmericanEngine,
)

if TYPE_CHECKING:
    from pquantlib.processes.black_scholes_merton_process import (
        BlackScholesMertonProcess,
    )
    from pquantlib_helpers.pricingengines.vanilla.finitedifferences.fd_engine_adapter import (
        FDEngineAdapter,
    )


class FDAmericanDividendOptionHelper(FDDividendOptionHelper):
    """American dividend-option helper using the FD American-condition engine.

    Java parity: ``org.jquantlib.helpers.FDAmericanDividendOptionHelper``.

    Parameters
    ----------
    option_type:
        Call or Put (:class:`~pquantlib.payoffs.OptionType`).
    underlying:
        Spot price of the underlying asset.
    strike:
        Option strike price.
    r:
        Risk-free rate (continuously compounded).
    q:
        Dividend yield (continuously compounded).
    vol:
        Implied volatility.
    settlement_date:
        Earliest exercise date; also anchors the flat curves.
    expiration_date:
        Latest exercise (expiry) date.
    dividend_dates:
        Dates when discrete dividends are paid.
    dividend_amounts:
        Corresponding dividend cash amounts.
    cal:
        Calendar. Default: :class:`~pquantlib.time.calendars.null_calendar.NullCalendar`.
    dc:
        Day counter. Default: :class:`~pquantlib.daycounters.actual_360.Actual360`.
    """

    def __init__(
        self,
        option_type: OptionType,
        underlying: float,
        strike: float,
        r: float,
        q: float,
        vol: float,
        settlement_date: Date,
        expiration_date: Date,
        dividend_dates: list[Date],
        dividend_amounts: list[float],
        cal: Calendar | None = None,
        dc: DayCounter | None = None,
    ) -> None:
        # Java parity: ``new AmericanExercise(settlementDate, expirationDate)``.
        super().__init__(
            option_type,
            underlying,
            strike,
            r,
            q,
            vol,
            settlement_date,
            AmericanExercise(settlement_date, expiration_date),
            dividend_dates,
            dividend_amounts,
            cal,
            dc,
        )

    def _build_engine(
        self, process: BlackScholesMertonProcess, time_steps: int
    ) -> FDEngineAdapter:
        """Wire an FDDividendAmericanEngine (American-condition single rollback)."""
        return FDDividendAmericanEngine(process, time_steps)


__all__ = ["FDAmericanDividendOptionHelper"]
