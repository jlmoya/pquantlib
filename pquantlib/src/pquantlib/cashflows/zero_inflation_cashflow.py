"""ZeroInflationCashFlow — single cashflow on a zero-inflation index ratio.

# C++ parity: ql/cashflows/zeroinflationcashflow.{hpp,cpp} (v1.42.1).

The C++ class derives from ``IndexedCashFlow`` and overrides
``baseFixing()`` / ``indexFixing()`` to apply the CPI lagged-fixing
math (``CPI::laggedFixing``). The base date is computed as
``start_date - observation_lag``; the fixing date is computed as
``end_date - observation_lag`` — these are the dates passed up to the
``IndexedCashFlow`` constructor.

Python divergences from C++:

- We take the start/end/observation_lag explicitly and forward to the
  base ``IndexedCashFlow`` constructor with the lag-adjusted dates.
- The C++ ``Visitor`` accept() is omitted.
"""

from __future__ import annotations

from pquantlib.cashflows.indexed_cashflow import IndexedCashFlow
from pquantlib.indexes.inflation.cpi import InterpolationType, lagged_fixing
from pquantlib.indexes.inflation.inflation_index import ZeroInflationIndex
from pquantlib.time.date import Date
from pquantlib.time.period import Period


class ZeroInflationCashFlow(IndexedCashFlow):
    """Cash flow whose amount is ``notional * I(end-lag) / I(start-lag)``.

    # C++ parity: ``ZeroInflationCashFlow`` in zeroinflationcashflow.hpp.
    """

    def __init__(
        self,
        notional: float,
        index: ZeroInflationIndex,
        observation_interpolation: InterpolationType,
        start_date: Date,
        end_date: Date,
        observation_lag: Period,
        payment_date: Date,
        growth_only: bool = False,
    ) -> None:
        # C++ parity: ql/cashflows/zeroinflationcashflow.cpp:34-39 — base
        # date / fixing date passed to IndexedCashFlow are the lag-adjusted
        # start / end dates.
        super().__init__(
            notional=notional,
            index=index,
            base_date=start_date - observation_lag,
            fixing_date=end_date - observation_lag,
            payment_date=payment_date,
            growth_only=growth_only,
        )
        self._zero_inflation_index: ZeroInflationIndex = index
        self._interpolation: InterpolationType = observation_interpolation
        self._start_date: Date = start_date
        self._end_date: Date = end_date
        self._observation_lag: Period = observation_lag

    # ---- inspectors --------------------------------------------------

    def zero_inflation_index(self) -> ZeroInflationIndex:
        return self._zero_inflation_index

    def observation_interpolation(self) -> InterpolationType:
        return self._interpolation

    # ---- IndexedCashFlow overrides -----------------------------------

    def base_fixing(self) -> float:
        """C++ parity: ql/cashflows/zeroinflationcashflow.cpp:41-43."""
        return lagged_fixing(
            self._zero_inflation_index,
            self._start_date,
            self._observation_lag,
            self._interpolation,
        )

    def index_fixing(self) -> float:
        """C++ parity: ql/cashflows/zeroinflationcashflow.cpp:45-47."""
        return lagged_fixing(
            self._zero_inflation_index,
            self._end_date,
            self._observation_lag,
            self._interpolation,
        )
