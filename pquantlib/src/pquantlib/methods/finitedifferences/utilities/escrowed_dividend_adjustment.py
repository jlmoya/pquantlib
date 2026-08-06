"""EscrowedDividendAdjustment — escrowed-dividend spot shift.

# C++ parity: ql/methods/finitedifferences/utilities/escroweddividendadjustment.{hpp,cpp}
# @ v1.43 (6b57206e0).

For the escrowed-dividend model the diffusing variable is the spot net of
the present value of all dividends still to be paid before maturity::

    divAdj(t) = - sum_{t <= t_i <= maturity} d_i * P_r(t_i)/P_r(t) * P_q(t)/P_q(t_i)

The sign is negative, so ``spot = s_t - divAdj(t)`` *adds* the escrowed
value back (see ``FdmEscrowedLogInnerValueCalculator``).

C++ takes ``DividendSchedule`` = ``std::vector<shared_ptr<Dividend>>``;
Python takes a ``Sequence[Dividend]`` from ``pquantlib.cashflows.dividend``.
``Handle<YieldTermStructure>`` collapses to the term structure itself —
PQuantLib does not port the Handle indirection.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import final

from pquantlib.cashflows.dividend import Dividend
from pquantlib.termstructures.yield_term_structure import YieldTermStructure
from pquantlib.time.date import Date


@final
class EscrowedDividendAdjustment:
    """Present value of the not-yet-paid dividends, as a spot shift.

    # C++ parity: ``class EscrowedDividendAdjustment``.
    """

    __slots__ = ("_dividend_schedule", "_maturity", "_q_ts", "_r_ts", "_to_time")

    def __init__(
        self,
        dividend_schedule: Sequence[Dividend],
        r_ts: YieldTermStructure,
        q_ts: YieldTermStructure,
        to_time: Callable[[Date], float],
        maturity: float,
    ) -> None:
        self._dividend_schedule: tuple[Dividend, ...] = tuple(dividend_schedule)
        self._r_ts: YieldTermStructure = r_ts
        self._q_ts: YieldTermStructure = q_ts
        self._to_time: Callable[[Date], float] = to_time
        self._maturity: float = maturity

    def dividend_adjustment(self, t: float) -> float:
        """Escrowed-dividend adjustment at time ``t`` (negative or zero).

        # C++ parity: ``Real dividendAdjustment(Time t) const``.
        """
        div_adj = 0.0
        for dividend in self._dividend_schedule:
            div_time = self._to_time(dividend.date())

            if div_time >= t and div_time <= self._maturity:
                div_adj -= (
                    dividend.amount()
                    * self._r_ts.discount(div_time)
                    / self._r_ts.discount(t)
                    * self._q_ts.discount(t)
                    / self._q_ts.discount(div_time)
                )

        return div_adj

    def risk_free_rate(self) -> YieldTermStructure:
        """# C++ parity: ``riskFreeRate()``."""
        return self._r_ts

    def dividend_yield(self) -> YieldTermStructure:
        """# C++ parity: ``dividendYield()``."""
        return self._q_ts


__all__ = ["EscrowedDividendAdjustment"]
