"""Option payoffs.

# C++ parity: ql/payoff.hpp + ql/option.hpp + ql/instruments/payoffs.{hpp,cpp}
#             (v1.42.1).

Three abstract levels:

* ``Payoff`` — describes how a contract pays at exercise. Subclasses
  override ``__call__(price)``, ``name()``, and ``description()``.
* ``TypePayoff`` — adds a Call/Put discriminant via ``OptionType``.
* ``StrikedTypePayoff`` — adds a fixed strike.

Concrete payoffs:

* ``PlainVanillaPayoff(option_type, strike)`` — ``max(opt*(S-K), 0)``.
* ``CashOrNothingPayoff(option_type, strike, cash_payoff)`` — pays
  ``cash_payoff`` if ITM, else 0.
* ``AssetOrNothingPayoff(option_type, strike)`` — pays ``price`` if
  ITM, else 0.
* ``GapPayoff(option_type, strike, second_strike)`` — long
  PlainVanilla at strike, short CashOrNothing at strike of
  ``(second_strike - strike)``.
* ``SuperFundPayoff(strike, second_strike)`` — ``price/strike`` if
  ``strike <= price < second_strike`` else 0.
* ``SuperSharePayoff(strike, second_strike, cash_payoff)`` —
  ``cash_payoff`` if ``strike <= price < second_strike`` else 0.

C++ ``NullPayoff``, ``PercentageStrikePayoff``, ``FloatingTypePayoff``
are deferred (rarely used in vanilla path, expanded in later phases).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import IntEnum
from typing import final

from pquantlib import qassert


class OptionType(IntEnum):
    """Call/Put discriminant.

    # C++ parity: ``Option::Type`` (ql/option.hpp). Integer values
    # match C++ exactly: Put=-1, Call=+1.
    """

    Put = -1
    Call = 1

    def __str__(self) -> str:
        return "Call" if self == OptionType.Call else "Put"


class Payoff(ABC):
    """Abstract payoff at exercise.

    # C++ parity: ql/payoff.hpp ``class Payoff``.
    """

    @abstractmethod
    def name(self) -> str:
        """Short identifier (e.g. "Vanilla", "CashOrNothing")."""

    @abstractmethod
    def description(self) -> str:
        """Human-readable description (includes type + strike + extra)."""

    @abstractmethod
    def __call__(self, price: float) -> float:
        """Payoff at given underlying price."""


class TypePayoff(Payoff, ABC):
    """Payoff that depends on Call/Put discriminant.

    # C++ parity: ql/instruments/payoffs.hpp ``class TypePayoff``.
    """

    def __init__(self, option_type: OptionType) -> None:
        self._option_type: OptionType = option_type

    def option_type(self) -> OptionType:
        return self._option_type

    def description(self) -> str:
        return f"{self.name()} {self._option_type}"


class StrikedTypePayoff(TypePayoff, ABC):
    """Payoff with a fixed strike.

    # C++ parity: ql/instruments/payoffs.hpp ``class StrikedTypePayoff``.
    """

    def __init__(self, option_type: OptionType, strike: float) -> None:
        super().__init__(option_type)
        self._strike: float = strike

    def strike(self) -> float:
        return self._strike

    def description(self) -> str:
        return f"{super().description()}, {self._strike} strike"


@final
class PlainVanillaPayoff(StrikedTypePayoff):
    """Plain vanilla Call/Put payoff: ``max(opt*(S-K), 0)``.

    # C++ parity: ql/instruments/payoffs.{hpp,cpp} ``PlainVanillaPayoff``.
    """

    def name(self) -> str:
        return "Vanilla"

    def __call__(self, price: float) -> float:
        if self._option_type == OptionType.Call:
            return max(price - self._strike, 0.0)
        return max(self._strike - price, 0.0)


@final
class CashOrNothingPayoff(StrikedTypePayoff):
    """Binary payoff: pays ``cash_payoff`` if ITM, else 0.

    # C++ parity: ql/instruments/payoffs.{hpp,cpp} ``CashOrNothingPayoff``.
    """

    def __init__(self, option_type: OptionType, strike: float, cash_payoff: float) -> None:
        super().__init__(option_type, strike)
        self._cash_payoff: float = cash_payoff

    def cash_payoff(self) -> float:
        return self._cash_payoff

    def name(self) -> str:
        return "CashOrNothing"

    def description(self) -> str:
        return f"{super().description()}, {self._cash_payoff} cash payoff"

    def __call__(self, price: float) -> float:
        if self._option_type == OptionType.Call:
            return self._cash_payoff if (price - self._strike) > 0.0 else 0.0
        return self._cash_payoff if (self._strike - price) > 0.0 else 0.0


@final
class AssetOrNothingPayoff(StrikedTypePayoff):
    """Binary payoff: pays ``price`` if ITM, else 0.

    # C++ parity: ql/instruments/payoffs.{hpp,cpp} ``AssetOrNothingPayoff``.
    """

    def name(self) -> str:
        return "AssetOrNothing"

    def __call__(self, price: float) -> float:
        if self._option_type == OptionType.Call:
            return price if (price - self._strike) > 0.0 else 0.0
        return price if (self._strike - price) > 0.0 else 0.0


@final
class GapPayoff(StrikedTypePayoff):
    """Gap payoff: like PlainVanilla but pays ``S - second_strike`` (Call)
    or ``second_strike - S`` (Put) when ``strike`` crosses the trigger.

    # C++ parity: ql/instruments/payoffs.{hpp,cpp} ``GapPayoff``.
    # The C++ payoff returns ``price - secondStrike`` (Call) when the
    # trigger ``price >= strike`` fires; this CAN be negative.
    """

    def __init__(self, option_type: OptionType, strike: float, second_strike: float) -> None:
        super().__init__(option_type, strike)
        self._second_strike: float = second_strike

    def second_strike(self) -> float:
        return self._second_strike

    def name(self) -> str:
        return "Gap"

    def description(self) -> str:
        return f"{super().description()}, {self._second_strike} strike payoff"

    def __call__(self, price: float) -> float:
        if self._option_type == OptionType.Call:
            return (price - self._second_strike) if (price - self._strike) >= 0.0 else 0.0
        return (self._second_strike - price) if (self._strike - price) >= 0.0 else 0.0


@final
class SuperFundPayoff(StrikedTypePayoff):
    """Binary supershare/superfund payoff: ``price/strike`` if
    ``strike <= price < second_strike``, else 0.

    # C++ parity: ql/instruments/payoffs.{hpp,cpp} ``SuperFundPayoff``.
    # Note: the C++ constructor is fixed to Option::Call.
    """

    def __init__(self, strike: float, second_strike: float) -> None:
        qassert.require(strike > 0.0, f"strike ({strike}) must be positive")
        qassert.require(
            second_strike > strike,
            f"second strike ({second_strike}) must be higher than first strike ({strike})",
        )
        super().__init__(OptionType.Call, strike)
        self._second_strike: float = second_strike

    def second_strike(self) -> float:
        return self._second_strike

    def name(self) -> str:
        return "SuperFund"

    def __call__(self, price: float) -> float:
        if self._strike <= price < self._second_strike:
            return price / self._strike
        return 0.0


@final
class SuperSharePayoff(StrikedTypePayoff):
    """Binary supershare payoff: ``cash_payoff`` if
    ``strike <= price < second_strike``, else 0.

    # C++ parity: ql/instruments/payoffs.{hpp,cpp} ``SuperSharePayoff``.
    # Constructor fixed to Option::Call (mirrors C++).
    """

    def __init__(self, strike: float, second_strike: float, cash_payoff: float) -> None:
        qassert.require(
            second_strike > strike,
            f"second strike ({second_strike}) must be higher than first strike ({strike})",
        )
        super().__init__(OptionType.Call, strike)
        self._second_strike: float = second_strike
        self._cash_payoff: float = cash_payoff

    def second_strike(self) -> float:
        return self._second_strike

    def cash_payoff(self) -> float:
        return self._cash_payoff

    def name(self) -> str:
        return "SuperShare"

    def description(self) -> str:
        return (
            f"{super().description()}, {self._second_strike} second strike, "
            f"{self._cash_payoff} amount"
        )

    def __call__(self, price: float) -> float:
        if self._strike <= price < self._second_strike:
            return self._cash_payoff
        return 0.0


__all__ = [
    "AssetOrNothingPayoff",
    "CashOrNothingPayoff",
    "GapPayoff",
    "OptionType",
    "Payoff",
    "PlainVanillaPayoff",
    "StrikedTypePayoff",
    "SuperFundPayoff",
    "SuperSharePayoff",
    "TypePayoff",
]
