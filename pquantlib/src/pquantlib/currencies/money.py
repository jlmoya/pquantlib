"""Money — a cash amount in a given currency (L1-B foundation).

# C++ parity: ql/money.hpp + ql/money.cpp (v1.42.1).

Same-currency arithmetic (+, -, *, /) and comparisons are exact. Combining
amounts in *different* currencies is governed by ``Money.Settings`` (a
singleton with ``conversion_type`` / ``base_currency``), mirroring the C++
``Money::Settings`` singleton.

# Deferral: the cross-currency branches in C++ delegate to
# ``ExchangeRateManager``, which is not yet ported. Until it lands, any
# cross-currency combination under ``BASE_CURRENCY_CONVERSION`` /
# ``AUTOMATED_CONVERSION`` raises ``LibraryException`` (documented in
# docs/carve-outs.md). The default ``NO_CONVERSION`` setting already raises
# on a currency mismatch, exactly as C++ does, so the common same-currency
# path — used by ``CommodityUnitCost`` / ``CommodityPricingHelper`` — is
# fully faithful.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import IntEnum

from pquantlib import qassert
from pquantlib.currencies.currency import Currency
from pquantlib.math.closeness import close as _close_real
from pquantlib.math.closeness import close_enough as _close_enough_real
from pquantlib.patterns.singleton import Singleton

_DEFAULT_N = 42


class ConversionType(IntEnum):
    """How to combine money in different currencies (parity with C++)."""

    NO_CONVERSION = 0
    BASE_CURRENCY_CONVERSION = 1
    AUTOMATED_CONVERSION = 2


class MoneySettings(Singleton):
    """Per-session settings for :class:`Money` (parity with ``Money::Settings``)."""

    def __init__(self) -> None:
        super().__init__()
        self._conversion_type: ConversionType = ConversionType.NO_CONVERSION
        self._base_currency: Currency = Currency()

    @classmethod
    def instance(cls) -> MoneySettings:
        """Return the singleton instance (parity with C++ ``::instance()``)."""
        return cls()

    @property
    def conversion_type(self) -> ConversionType:
        return self._conversion_type

    @conversion_type.setter
    def conversion_type(self, value: ConversionType) -> None:
        self._conversion_type = value

    @property
    def base_currency(self) -> Currency:
        return self._base_currency

    @base_currency.setter
    def base_currency(self, value: Currency) -> None:
        self._base_currency = value


def _convert_to(m: Money, target: Currency) -> Money:
    if m.currency != target:
        # C++ delegates to ExchangeRateManager here (not yet ported).
        qassert.fail(
            "Money cross-currency conversion requires ExchangeRateManager, "
            "which is not yet ported (see docs/carve-outs.md)"
        )
    return m


def _convert_to_base(m: Money) -> Money:
    base = MoneySettings.instance().base_currency
    qassert.require(not base.empty(), "no base currency set")
    return _convert_to(m, base)


def _apply(m1: Money, m2: Money, f: Callable[[float, float], float]) -> float:
    conv = MoneySettings.instance().conversion_type
    if m1.currency == m2.currency:
        return f(m1.value, m2.value)
    if conv == ConversionType.BASE_CURRENCY_CONVERSION:
        return f(_convert_to_base(m1).value, _convert_to_base(m2).value)
    if conv == ConversionType.AUTOMATED_CONVERSION:
        return f(m1.value, _convert_to(m2, m1.currency).value)
    qassert.fail("currency mismatch and no conversion specified")


def _apply_bool(m1: Money, m2: Money, f: Callable[[float, float], bool]) -> bool:
    conv = MoneySettings.instance().conversion_type
    if m1.currency == m2.currency:
        return f(m1.value, m2.value)
    if conv == ConversionType.BASE_CURRENCY_CONVERSION:
        return f(_convert_to_base(m1).value, _convert_to_base(m2).value)
    if conv == ConversionType.AUTOMATED_CONVERSION:
        return f(m1.value, _convert_to(m2, m1.currency).value)
    qassert.fail("currency mismatch and no conversion specified")


class Money:
    """A cash amount in a currency.

    Both ``Money(currency, value)`` and ``Money(value, currency)`` C++ ctor
    orders are supported by inspecting argument types.
    """

    # Nested enum alias for the C++ idiom ``Money.ConversionType.NO_CONVERSION``.
    ConversionType = ConversionType
    Settings = MoneySettings

    def __init__(
        self,
        first: Currency | float | None = None,
        second: Currency | float | None = None,
    ) -> None:
        if first is None and second is None:
            self._value: float = 0.0
            self._currency: Currency = Currency()
            return
        # Accept (currency, value) or (value, currency).
        if isinstance(first, Currency):
            assert not isinstance(second, Currency)
            self._currency = first
            self._value = float(second) if second is not None else 0.0
        else:
            assert isinstance(second, Currency)
            self._value = float(first) if first is not None else 0.0
            self._currency = second

    # ---- inspectors ----

    @property
    def currency(self) -> Currency:
        return self._currency

    @property
    def value(self) -> float:
        return self._value

    def rounded(self) -> Money:
        return Money(self._currency.rounding(self._value), self._currency)

    # ---- unary + in-place arithmetic ----

    def __pos__(self) -> Money:
        return Money(self._value, self._currency)

    def __neg__(self) -> Money:
        return Money(-self._value, self._currency)

    def __iadd__(self, other: Money) -> Money:
        conv = MoneySettings.instance().conversion_type
        if self._currency == other._currency:
            self._value += other._value
        elif conv == ConversionType.BASE_CURRENCY_CONVERSION:
            base_self = _convert_to_base(self)
            base_other = _convert_to_base(other)
            self._value = base_self._value + base_other._value
            self._currency = base_self._currency
        elif conv == ConversionType.AUTOMATED_CONVERSION:
            tmp = _convert_to(other, self._currency)
            self._value += tmp._value
        else:
            qassert.fail("currency mismatch and no conversion specified")
        return self

    def __isub__(self, other: Money) -> Money:
        return self.__iadd__(-other)

    def __imul__(self, x: float) -> Money:
        self._value *= x
        return self

    def __itruediv__(self, x: float) -> Money:
        self._value /= x
        return self

    # ---- binary arithmetic ----

    def __add__(self, other: Money) -> Money:
        tmp = Money(self._value, self._currency)
        tmp += other
        return tmp

    def __sub__(self, other: Money) -> Money:
        tmp = Money(self._value, self._currency)
        tmp -= other
        return tmp

    def __mul__(self, x: float) -> Money:
        return Money(self._value * x, self._currency)

    def __rmul__(self, x: float) -> Money:
        return self.__mul__(x)

    def __truediv__(self, other: Money | float) -> Money | float:
        """``money / scalar`` -> Money; ``money / money`` -> Real ratio."""
        if isinstance(other, Money):
            return _apply(self, other, lambda x, y: x / y)
        return Money(self._value / other, self._currency)

    # ---- comparisons (parity with the C++ apply<bool> free functions) ----

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Money):
            return NotImplemented
        return _apply_bool(self, other, lambda x, y: x == y)

    def __ne__(self, other: object) -> bool:
        result = self.__eq__(other)
        if result is NotImplemented:
            return result
        return not result

    def __lt__(self, other: Money) -> bool:
        return _apply_bool(self, other, lambda x, y: x < y)

    def __le__(self, other: Money) -> bool:
        return _apply_bool(self, other, lambda x, y: x <= y)

    def __gt__(self, other: Money) -> bool:
        return _apply_bool(other, self, lambda x, y: x < y)

    def __ge__(self, other: Money) -> bool:
        return _apply_bool(other, self, lambda x, y: x <= y)

    def __hash__(self) -> int:
        return hash((self._currency, self._value))

    def __str__(self) -> str:
        return f"{self.rounded().value} {self._currency.code}"

    def __repr__(self) -> str:
        return f"Money({self.__str__()!r})"


def close(m1: Money, m2: Money, n: int = _DEFAULT_N) -> bool:
    """Floating-point closeness of two money amounts (parity with ``close``)."""
    return _apply_bool(m1, m2, lambda x, y: _close_real(x, y, n))


def close_enough(m1: Money, m2: Money, n: int = _DEFAULT_N) -> bool:
    """Looser closeness (parity with ``close_enough``)."""
    return _apply_bool(m1, m2, lambda x, y: _close_enough_real(x, y, n))
