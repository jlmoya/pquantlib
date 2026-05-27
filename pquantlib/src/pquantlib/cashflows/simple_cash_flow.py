"""SimpleCashFlow — fixed amount on a known date.

# C++ parity: ql/cashflows/simplecashflow.hpp (v1.42.1).
"""

from __future__ import annotations

from pquantlib.cashflows.cash_flow import CashFlow
from pquantlib.time.date import Date


class SimpleCashFlow(CashFlow):
    """A predetermined cash flow paying ``amount`` on ``date``."""

    def __init__(self, amount: float, date: Date) -> None:
        super().__init__()
        self._amount: float = amount
        self._date: Date = date

    def amount(self) -> float:
        return self._amount

    def date(self) -> Date:
        return self._date


class Redemption(SimpleCashFlow):
    """Bond redemption — specializes SimpleCashFlow for visitor dispatch.

    C++ parity: ql/cashflows/simplecashflow.hpp:61-70. In Python visitors
    aren't used at the CashFlow level (deferred carve-out), but the
    subclass distinction is preserved for callers that ``isinstance``-check.
    """


class AmortizingPayment(SimpleCashFlow):
    """Amortizing payment — specializes SimpleCashFlow.

    C++ parity: ql/cashflows/simplecashflow.hpp:76-85.
    """
