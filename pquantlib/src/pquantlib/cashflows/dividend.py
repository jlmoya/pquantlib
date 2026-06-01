"""Dividend — predetermined / fractional stock dividend cash flows.

# C++ parity: ql/cashflows/dividend.{hpp,cpp} (v1.42.1, 099987f0).

A :class:`Dividend` is a :class:`~pquantlib.cashflows.cash_flow.CashFlow`
that pays a predetermined amount at a given date. It adds one method beyond
the plain ``CashFlow`` interface — ``amount(underlying)`` — so a dividend can
be expressed either as a fixed cash amount or as a fraction of the underlying
spot at the dividend date.

Two concretes:

- :class:`FixedDividend` — pays a fixed cash amount regardless of the
  underlying (both ``amount()`` and ``amount(underlying)`` return the same
  value).
- :class:`FractionalDividend` — pays ``rate * underlying``. ``amount()``
  (no underlying) returns ``rate * nominal`` and requires a nominal to have
  been supplied.

Plus the :func:`dividend_vector` helper building a sequence of
:class:`FixedDividend` from parallel date/amount lists.

Python divergences from C++:

- ``accept(AcyclicVisitor&)`` (visitability) is omitted — the Visitor
  protocol is a deferred carve-out across all cashflow classes in this port.
- C++ ``Null<Real>()`` sentinel for "no nominal" is replaced by Python
  ``None`` on the ``nominal`` parameter.
- C++ names the helper ``DividendVector`` (PascalCase free function).
  Python idiom is ``dividend_vector`` (snake_case); the C++ name is noted
  in the docstring.
"""

from __future__ import annotations

from abc import abstractmethod

from pquantlib import qassert
from pquantlib.cashflows.cash_flow import CashFlow
from pquantlib.time.date import Date


class Dividend(CashFlow):
    """Abstract predetermined cash flow paid at ``date``.

    # C++ parity: ``Dividend`` in dividend.hpp:36-55. Adds the
    # ``amount(underlying)`` method on top of the ``CashFlow`` interface;
    # ``amount()`` (no args) stays abstract.
    """

    def __init__(self, date: Date) -> None:
        super().__init__()
        self._date: Date = date

    def date(self) -> Date:
        """C++ parity: ``Dividend::date`` (inline hpp:42) — the dividend date."""
        return self._date

    @abstractmethod
    def amount(self) -> float:
        """Cash amount with no underlying. Concrete subclass defines this."""

    @abstractmethod
    def amount_with_underlying(self, underlying: float) -> float:
        """Cash amount given the underlying spot at the dividend date.

        # C++ parity: ``virtual Real amount(Real underlying) const = 0;``
        # (dividend.hpp:48). Python cannot overload ``amount`` by arity, so
        # the two-argument form is exposed under a distinct method name.
        """


class FixedDividend(Dividend):
    """Dividend paying a fixed cash amount independent of the underlying.

    # C++ parity: ``FixedDividend`` in dividend.hpp:59-70.
    """

    def __init__(self, amount: float, date: Date) -> None:
        super().__init__(date)
        self._amount: float = amount

    def amount(self) -> float:
        """The fixed amount. C++ parity: hpp:65."""
        return self._amount

    def amount_with_underlying(self, underlying: float) -> float:
        """The fixed amount (underlying ignored). C++ parity: hpp:66."""
        del underlying
        return self._amount


class FractionalDividend(Dividend):
    """Dividend paying ``rate * underlying`` (or ``rate * nominal``).

    # C++ parity: ``FractionalDividend`` in dividend.hpp:74-97.

    If a ``nominal`` is supplied, ``amount()`` (no underlying) returns
    ``rate * nominal``; otherwise ``amount()`` raises (matching the C++
    ``QL_REQUIRE(nominal_ != Null<Real>(), "no nominal given")``).
    """

    def __init__(self, rate: float, date: Date, nominal: float | None = None) -> None:
        # C++ has two ctors: (rate, date) with nominal = Null, and
        # (rate, nominal, date). We collapse them into one with an optional
        # ``nominal`` defaulting to None (== C++ Null<Real>()).
        super().__init__(date)
        self._rate: float = rate
        self._nominal: float | None = nominal

    def amount(self) -> float:
        """``rate * nominal``; requires a nominal. C++ parity: hpp:83-86."""
        qassert.require(self._nominal is not None, "no nominal given")
        assert self._nominal is not None
        return self._rate * self._nominal

    def amount_with_underlying(self, underlying: float) -> float:
        """``rate * underlying``. C++ parity: hpp:87."""
        return self._rate * underlying

    def rate(self) -> float:
        """C++ parity: ``FractionalDividend::rate`` (hpp:91)."""
        return self._rate

    def nominal(self) -> float | None:
        """C++ parity: ``FractionalDividend::nominal`` (hpp:92).

        Returns ``None`` when no nominal was supplied (C++ returns
        ``Null<Real>()``).
        """
        return self._nominal


def dividend_vector(dividend_dates: list[Date], dividends: list[float]) -> list[Dividend]:
    """Build a sequence of :class:`FixedDividend` from parallel lists.

    # C++ parity: ``DividendVector`` free function (dividend.cpp:34-51).
    """
    qassert.require(
        len(dividend_dates) == len(dividends),
        "size mismatch between dividend dates and amounts",
    )
    return [
        FixedDividend(amount, date)
        for date, amount in zip(dividend_dates, dividends, strict=True)
    ]
