"""QuantoTermStructure — quanto-adjusted dividend curve.

# C++ parity: ql/termstructures/yield/quantotermstructure.hpp (v1.43).

Ported as a dependency of ``QuantoEngine``: pricing a quanto option is
pricing the *same* option under a dividend curve shifted by the domestic /
foreign rate differential plus the quanto convexity term

    q(t) + r(t) - r_f(t) + rho * sigma_S(t, K) * sigma_X(t, X_atm)

The structure stays linked to the five inputs, so any change in them
propagates.
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
    """Quanto-adjusted dividend term structure.

    # C++ parity: ``class QuantoTermStructure``
    # (quantotermstructure.hpp:42-70 + the inline definitions at :75-135).
    """

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
        # # C++ parity: base ctor is ``ZeroYieldStructure(underlyingDividendTS->dayCounter())``.
        super().__init__(day_counter=underlying_dividend_ts.day_counter())
        self._underlying_dividend_ts: YieldTermStructure = underlying_dividend_ts
        self._risk_free_ts: YieldTermStructure = risk_free_ts
        self._foreign_risk_free_ts: YieldTermStructure = foreign_risk_free_ts
        self._underlying_black_vol_ts: BlackVolTermStructure = underlying_black_vol_ts
        self._exch_rate_black_vol_ts: BlackVolTermStructure = exch_rate_black_vol_ts
        self._underlying_exch_rate_correlation: float = underlying_exch_rate_correlation
        self._strike: float = strike
        self._exch_rate_atm_level: float = exch_rate_atm_level

        underlying_dividend_ts.register_with(self)
        risk_free_ts.register_with(self)
        foreign_risk_free_ts.register_with(self)
        underlying_black_vol_ts.register_with(self)
        exch_rate_black_vol_ts.register_with(self)

    # --- YieldTermStructure interface -------------------------------------

    def day_counter(self) -> DayCounter:
        """# C++ parity: ``dayCounter()`` (quantotermstructure.hpp:99-101)."""
        return self._underlying_dividend_ts.day_counter()

    def calendar(self) -> Calendar:
        """# C++ parity: ``calendar()`` (quantotermstructure.hpp:103-105)."""
        return self._underlying_dividend_ts.calendar()

    def settlement_days(self) -> int:
        """# C++ parity: ``settlementDays()`` (quantotermstructure.hpp:107-109)."""
        return self._underlying_dividend_ts.settlement_days()

    def reference_date(self) -> Date:
        """# C++ parity: ``referenceDate()`` (quantotermstructure.hpp:111-113)."""
        return self._underlying_dividend_ts.reference_date()

    def max_date(self) -> Date:
        """# C++ parity: ``maxDate()`` (quantotermstructure.hpp:115-123)."""
        max_date = min(
            self._underlying_dividend_ts.max_date(), self._risk_free_ts.max_date()
        )
        max_date = min(max_date, self._foreign_risk_free_ts.max_date())
        max_date = min(max_date, self._underlying_black_vol_ts.max_date())
        return min(max_date, self._exch_rate_black_vol_ts.max_date())

    # --- ZeroYieldStructure hook -------------------------------------------

    def _zero_yield_impl(self, t: float) -> float:
        """# C++ parity: ``zeroYieldImpl`` (quantotermstructure.hpp:125-133).

        # C++ parity note: the C++ comment flags that all five structures are
        # *assumed* to share a day counter and that the assumption is not
        # QL_REQUIREd. Reproduced as-is.
        """
        return (
            self._underlying_dividend_ts.zero_rate(
                t, Compounding.Continuous, Frequency.NoFrequency, True
            ).rate()
            + self._risk_free_ts.zero_rate(
                t, Compounding.Continuous, Frequency.NoFrequency, True
            ).rate()
            - self._foreign_risk_free_ts.zero_rate(
                t, Compounding.Continuous, Frequency.NoFrequency, True
            ).rate()
            + self._underlying_exch_rate_correlation
            * self._underlying_black_vol_ts.black_vol_at_time(t, self._strike, True)
            * self._exch_rate_black_vol_ts.black_vol_at_time(
                t, self._exch_rate_atm_level, True
            )
        )


__all__ = ["QuantoTermStructure"]
