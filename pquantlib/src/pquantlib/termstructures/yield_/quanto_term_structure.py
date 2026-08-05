"""QuantoTermStructure — dividend curve carrying the quanto adjustment.

# C++ parity: ql/termstructures/yield/quantotermstructure.hpp (v1.43).

A quanto option is written on a foreign asset but settles in domestic
currency at a fixed exchange rate. Under the domestic risk-neutral
measure the asset's drift picks up two corrections: the interest-rate
differential and a convexity term proportional to the correlation
between the asset and the exchange rate. QuantLib implements both by
handing the pricing engine an ordinary Black-Scholes process whose
*dividend* curve is this term structure:

    q_quanto(t) = q(t) + r_domestic(t) - r_foreign(t)
                  + rho * sigma_S(t, K) * sigma_X(t, X_atm)

(``zeroYieldImpl``, quantotermstructure.hpp:123-131). Everything else in
the process is untouched, which is why an engine that never reads the
foreign curve, the FX volatility or the correlation still produces a
plausible-looking price — the whole quanto effect lives here.

The structure stays linked to the five curves it is built from: any
change in them is reflected here.

Note (inherited from C++): all five curves are assumed to share a day
count convention; C++ flags this as something that *should* be
``QL_REQUIRE``d but is not.
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
    """Quanto-adjusted dividend curve.

    # C++ parity: ``class QuantoTermStructure : public ZeroYieldStructure``.

    Args:
        underlying_dividend_ts: dividend curve of the underlying asset.
        risk_free_ts: domestic risk-free curve.
        foreign_risk_free_ts: risk-free curve of the asset's own currency.
        underlying_black_vol_ts: Black vol surface of the underlying.
        strike: strike at which ``underlying_black_vol_ts`` is sampled.
        exch_rate_black_vol_ts: Black vol surface of the exchange rate.
        exch_rate_atm_level: level at which the FX vol is sampled.
        underlying_exch_rate_correlation: correlation between the
            underlying and the exchange rate.
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
        # C++ delegates the day counter to the underlying dividend curve and
        # overrides ``referenceDate()`` (delegated construction mode).
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

    # --- YieldTermStructure interface (all delegated to the dividend curve) ---

    def day_counter(self) -> DayCounter:
        return self._underlying_dividend_ts.day_counter()

    def calendar(self) -> Calendar:
        return self._underlying_dividend_ts.calendar()

    def settlement_days(self) -> int:
        return self._underlying_dividend_ts.settlement_days()

    def reference_date(self) -> Date:
        return self._underlying_dividend_ts.reference_date()

    def max_date(self) -> Date:
        # C++ parity: quantotermstructure.hpp:113-121 — the earliest of the
        # five underlying max dates.
        return min(
            self._underlying_dividend_ts.max_date(),
            self._risk_free_ts.max_date(),
            self._foreign_risk_free_ts.max_date(),
            self._underlying_black_vol_ts.max_date(),
            self._exch_rate_black_vol_ts.max_date(),
        )

    # --- ZeroYieldStructure implementation ------------------------------------

    def _zero_yield_impl(self, t: float) -> float:
        """Zero yield seen from the evaluation date.

        # C++ parity: ``QuantoTermStructure::zeroYieldImpl``
        # (quantotermstructure.hpp:123-131). All five curves are queried
        # with extrapolation enabled, as in C++.
        """
        return (
            self._underlying_dividend_ts.zero_rate(
                t, Compounding.Continuous, Frequency.NoFrequency, True
            ).rate()
            + self._risk_free_ts.zero_rate(t, Compounding.Continuous, Frequency.NoFrequency, True).rate()
            - self._foreign_risk_free_ts.zero_rate(
                t, Compounding.Continuous, Frequency.NoFrequency, True
            ).rate()
            + self._underlying_exch_rate_correlation
            * self._underlying_black_vol_ts.black_vol_at_time(t, self._strike, True)
            * self._exch_rate_black_vol_ts.black_vol_at_time(t, self._exch_rate_atm_level, True)
        )


__all__ = ["QuantoTermStructure"]
