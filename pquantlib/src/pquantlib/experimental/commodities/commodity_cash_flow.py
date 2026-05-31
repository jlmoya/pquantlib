"""CommodityCashFlow — a commodity-swap payment cash flow.

# C++ parity: ql/experimental/commodities/commoditycashflow.hpp +
#             commoditycashflow.cpp (v1.42.1).

A :class:`~pquantlib.cashflows.cash_flow.CashFlow` carrying the four Money
amounts an energy-swap period produces — discounted / undiscounted amount
(in the base currency) and discounted / undiscounted *payment* amount (in
the payment currency) — plus the discount factors and a ``finalized`` flag.
``amount()`` returns the discounted (base-currency) value.

The C++ ``CommodityCashFlows`` typedef (``map<Date, shared_ptr<CommodityCashFlow>>``)
maps to ``dict[Date, CommodityCashFlow]`` on the Python side.
"""

from __future__ import annotations

from pquantlib.cashflows.cash_flow import CashFlow
from pquantlib.currencies.currency import Currency
from pquantlib.currencies.money import Money
from pquantlib.time.date import Date


class CommodityCashFlow(CashFlow):
    """A commodity payment cash flow (discounted/undiscounted Money amounts)."""

    def __init__(
        self,
        date: Date,
        discounted_amount: Money,
        undiscounted_amount: Money,
        discounted_payment_amount: Money,
        undiscounted_payment_amount: Money,
        discount_factor: float,
        payment_discount_factor: float,
        finalized: bool,
    ) -> None:
        super().__init__()
        self._date = date
        self._discounted_amount = discounted_amount
        self._undiscounted_amount = undiscounted_amount
        self._discounted_payment_amount = discounted_payment_amount
        self._undiscounted_payment_amount = undiscounted_payment_amount
        self._discount_factor = discount_factor
        self._payment_discount_factor = payment_discount_factor
        self._finalized = finalized

    # ---- Event / CashFlow interface ----

    def date(self) -> Date:
        return self._date

    def amount(self) -> float:
        return self._discounted_amount.value

    # ---- inspectors ----

    @property
    def currency(self) -> Currency:
        return self._discounted_amount.currency

    @property
    def discounted_amount(self) -> Money:
        return self._discounted_amount

    @property
    def undiscounted_amount(self) -> Money:
        return self._undiscounted_amount

    @property
    def discounted_payment_amount(self) -> Money:
        return self._discounted_payment_amount

    @property
    def undiscounted_payment_amount(self) -> Money:
        return self._undiscounted_payment_amount

    @property
    def discount_factor(self) -> float:
        return self._discount_factor

    @property
    def payment_discount_factor(self) -> float:
        return self._payment_discount_factor

    @property
    def finalized(self) -> bool:
        return self._finalized


# C++ parity: ``typedef std::map<Date, ext::shared_ptr<CommodityCashFlow>> CommodityCashFlows;``
CommodityCashFlows = dict[Date, CommodityCashFlow]
