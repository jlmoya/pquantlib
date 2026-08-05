"""Exchange rate between two currencies.

# C++ parity: ql/exchangerate.hpp + ql/exchangerate.cpp (v1.43).

A rate is quoted with the convention that one unit of ``source`` is worth
``rate`` units of ``target``. A *Direct* rate carries that number; a *Derived*
rate carries the pair of rates it was chained from and exchanges by walking
them, so precision follows the same route C++ takes.
"""

from __future__ import annotations

from enum import IntEnum

from pquantlib import qassert
from pquantlib.currencies.currency import Currency
from pquantlib.currencies.money import Money


class ExchangeRateType(IntEnum):
    """How a rate came to be (parity with the C++ nested ``ExchangeRate::Type``)."""

    Direct = 0  # given directly by the user
    Derived = 1  # derived from rates between other currencies


class ExchangeRate:
    """A rate between two currencies, optionally derived from a chain."""

    # Nested-enum alias for the C++ idiom ``ExchangeRate.Type.Direct``.
    Type = ExchangeRateType

    def __init__(
        self,
        source: Currency | None = None,
        target: Currency | None = None,
        rate: float | None = None,
    ) -> None:
        """Build a Direct rate, or (with no arguments) an unusable placeholder.

        # C++ parity: the default ctor leaves the rate at ``Null<Decimal>()``.
        # Python has no poison value for Real, so the placeholder holds
        # ``None`` and :meth:`exchange` refuses to use it.
        """
        self._source: Currency = source if source is not None else Currency()
        self._target: Currency = target if target is not None else Currency()
        self._rate: float | None = rate
        self._type: ExchangeRateType = ExchangeRateType.Direct
        self._rate_chain: tuple[ExchangeRate, ExchangeRate] | None = None

    # ---- inspectors ----

    @property
    def source(self) -> Currency:
        return self._source

    @property
    def target(self) -> Currency:
        return self._target

    @property
    def type(self) -> ExchangeRateType:
        return self._type

    @property
    def rate(self) -> float | None:
        """The rate, or ``None`` for a placeholder built by the empty ctor."""
        return self._rate

    # ---- utility methods ----

    def exchange(self, amount: Money) -> Money:
        """Convert ``amount``, in either direction, through this rate."""
        if self._type == ExchangeRateType.Direct:
            qassert.require(self._rate is not None, "no exchange rate given")
            assert self._rate is not None
            if amount.currency == self._source:
                return Money(amount.value * self._rate, self._target)
            if amount.currency == self._target:
                return Money(amount.value / self._rate, self._source)
            qassert.fail("exchange rate not applicable")
        # Derived: delegate to whichever leg of the chain knows the currency,
        # then hand the result to the other leg.
        assert self._rate_chain is not None
        first, second = self._rate_chain
        if amount.currency in (first.source, first.target):
            return second.exchange(first.exchange(amount))
        if amount.currency in (second.source, second.target):
            return first.exchange(second.exchange(amount))
        qassert.fail("exchange rate not applicable")

    @staticmethod
    def chain(r1: ExchangeRate, r2: ExchangeRate) -> ExchangeRate:
        """Compose two rates that share a currency, in any of the four orientations."""
        qassert.require(r1.rate is not None and r2.rate is not None, "no exchange rate given")
        rate1 = r1.rate
        rate2 = r2.rate
        assert rate1 is not None
        assert rate2 is not None

        result = ExchangeRate()
        result._type = ExchangeRateType.Derived
        result._rate_chain = (r1, r2)
        if r1.source == r2.source:
            result._source = r1.target
            result._target = r2.target
            result._rate = rate2 / rate1
        elif r1.source == r2.target:
            result._source = r1.target
            result._target = r2.source
            result._rate = 1.0 / (rate1 * rate2)
        elif r1.target == r2.source:
            result._source = r1.source
            result._target = r2.target
            result._rate = rate1 * rate2
        elif r1.target == r2.target:
            result._source = r1.source
            result._target = r2.source
            result._rate = rate1 / rate2
        else:
            qassert.fail("exchange rates not chainable")
        return result

    def __repr__(self) -> str:
        return f"ExchangeRate({self._source.code} -> {self._target.code} @ {self._rate})"
