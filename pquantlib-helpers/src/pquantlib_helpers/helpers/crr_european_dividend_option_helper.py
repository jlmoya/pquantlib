"""CRREuropeanDividendOptionHelper — European CRR binomial dividend-option helper.

# Retired-API compat layer — NOT a port of C++ QuantLib v1.42.1.

Java parity:
``org.jquantlib.helpers.CRREuropeanDividendOptionHelper`` (jquantlib-helpers).

Concrete subclass of
:class:`~pquantlib_helpers.helpers.crr_dividend_option_helper.CRRDividendOptionHelper`
that wires a :class:`~pquantlib.exercise.EuropeanExercise`.
"""

from __future__ import annotations

from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.exercise import EuropeanExercise
from pquantlib.payoffs import OptionType
from pquantlib.time.calendar import Calendar
from pquantlib.time.date import Date
from pquantlib_helpers.helpers.crr_dividend_option_helper import (
    CRRDividendOptionHelper,
)


class CRREuropeanDividendOptionHelper(CRRDividendOptionHelper):
    """European dividend-option helper using Cox-Ross-Rubinstein binomial trees.

    Java parity: ``org.jquantlib.helpers.CRREuropeanDividendOptionHelper``.

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
        Pricing reference / settlement date; anchors the flat curves.
    expiration_date:
        Option expiry date; wrapped in a :class:`~pquantlib.exercise.EuropeanExercise`.
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
        super().__init__(
            option_type,
            underlying,
            strike,
            r,
            q,
            vol,
            settlement_date,
            EuropeanExercise(expiration_date),
            dividend_dates,
            dividend_amounts,
            cal,
            dc,
        )


__all__ = ["CRREuropeanDividendOptionHelper"]
