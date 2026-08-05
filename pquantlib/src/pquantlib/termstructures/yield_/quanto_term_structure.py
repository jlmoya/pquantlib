"""QuantoTermStructure — dividend curve adjusted for the quanto effect.

# C++ parity: ql/termstructures/yield/quantotermstructure.hpp (v1.43)

The quanto-adjusted "dividend" zero yield seen from the evaluation date::

    q(t) + r(t) - r_f(t) + rho * sigma_S(t, K) * sigma_X(t, X_atm)

where q is the underlying's dividend curve, r the domestic risk-free curve,
r_f the foreign risk-free curve, sigma_S the underlying's Black vol at the
option strike, sigma_X the exchange rate's Black vol at its ATM level, and
rho their correlation.

C++ notes in a comment that all term structures are ASSUMED to share a day
counter without requiring it; this port keeps that behaviour rather than
tightening it, because tightening would reject inputs C++ accepts.
"""

from __future__ import annotations

from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.termstructures.volatility.equity_fx.black_vol_term_structure import (
    BlackVolTermStructure,
)
from pquantlib.termstructures.yield_.zero_yield_structure import ZeroYieldStructure
from pquantlib.termstructures.yield_term_structure import YieldTermStructure
from pquantlib.time.calendar import Calendar
from pquantlib.time.compounding import Compounding
from pquantlib.time.date import Date
from pquantlib.time.frequency import Frequency


class QuantoTermStructure(ZeroYieldStructure):
    """Quanto-adjusted dividend term structure."""

    def __init__(
        self,
        underlying_dividend_ts: YieldTermStructure,
        risk_free_ts: YieldTermStructure,
        foreign_risk_free_ts: YieldTermStructure,
        underlying_black_vol_ts: BlackVolTermStructure,
        strike: float,
        exch_rate_black_vol_ts: BlackVolTermStructure,
        exch_rate_atm_level: float,
        underlying_exch_rate_correlation: float,
    ) -> None:
        ZeroYieldStructure.__init__(
            self, day_counter=underlying_dividend_ts.day_counter()
        )
        self._underlying_dividend_ts: YieldTermStructure = underlying_dividend_ts
        self._risk_free_ts: YieldTermStructure = risk_free_ts
        self._foreign_risk_free_ts: YieldTermStructure = foreign_risk_free_ts
        self._underlying_black_vol_ts: BlackVolTermStructure = underlying_black_vol_ts
        self._exch_rate_black_vol_ts: BlackVolTermStructure = exch_rate_black_vol_ts
        self._strike: float = strike
        self._exch_rate_atm_level: float = exch_rate_atm_level
        self._underlying_exch_rate_correlation: float = underlying_exch_rate_correlation
        underlying_dividend_ts.register_with(self)
        risk_free_ts.register_with(self)
        foreign_risk_free_ts.register_with(self)
        underlying_black_vol_ts.register_with(self)
        exch_rate_black_vol_ts.register_with(self)

    # ---- TermStructure overrides -------------------------------------------

    def day_counter(self) -> DayCounter:
        return self._underlying_dividend_ts.day_counter()

    def calendar(self) -> Calendar:
        return self._underlying_dividend_ts.calendar()

    def settlement_days(self) -> int:
        return self._underlying_dividend_ts.settlement_days()

    def reference_date(self) -> Date:
        return self._underlying_dividend_ts.reference_date()

    def max_date(self) -> Date:
        return min(
            self._underlying_dividend_ts.max_date(),
            self._risk_free_ts.max_date(),
            self._foreign_risk_free_ts.max_date(),
            self._underlying_black_vol_ts.max_date(),
            self._exch_rate_black_vol_ts.max_date(),
        )

    # ---- ZeroYieldStructure implementation ---------------------------------

    def _zero_yield_impl(self, t: float) -> float:
        # C++ parity: ``QuantoTermStructure::zeroYieldImpl``.
        return (
            self._underlying_dividend_ts.zero_rate(
                t, Compounding.Continuous, Frequency.NoFrequency, extrapolate=True
            ).rate()
            + self._risk_free_ts.zero_rate(
                t, Compounding.Continuous, Frequency.NoFrequency, extrapolate=True
            ).rate()
            - self._foreign_risk_free_ts.zero_rate(
                t, Compounding.Continuous, Frequency.NoFrequency, extrapolate=True
            ).rate()
            + self._underlying_exch_rate_correlation
            * self._underlying_black_vol_ts.black_vol_at_time(t, self._strike, True)
            * self._exch_rate_black_vol_ts.black_vol_at_time(
                t, self._exch_rate_atm_level, True
            )
        )
