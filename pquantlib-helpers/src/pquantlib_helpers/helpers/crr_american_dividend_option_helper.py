"""CRRAmericanDividendOptionHelper — American CRR binomial dividend-option helper.

# Retired-API compat layer — NOT a port of C++ QuantLib v1.42.1.

Java parity:
``org.jquantlib.helpers.CRRAmericanDividendOptionHelper`` (jquantlib-helpers).

Concrete subclass of
:class:`~pquantlib_helpers.helpers.crr_dividend_option_helper.CRRDividendOptionHelper`
that wires an :class:`~pquantlib.exercise.AmericanExercise`.
"""

from __future__ import annotations

from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.exercise import AmericanExercise
from pquantlib.payoffs import OptionType
from pquantlib.time.calendar import Calendar
from pquantlib.time.date import Date
from pquantlib_helpers.helpers.crr_dividend_option_helper import (
    CRRDividendOptionHelper,
)


class CRRAmericanDividendOptionHelper(CRRDividendOptionHelper):
    """American dividend-option helper using Cox-Ross-Rubinstein binomial trees.

    Java parity: ``org.jquantlib.helpers.CRRAmericanDividendOptionHelper``.

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
        Earliest exercise date; also used to anchor the flat curves.
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


__all__ = ["CRRAmericanDividendOptionHelper"]
