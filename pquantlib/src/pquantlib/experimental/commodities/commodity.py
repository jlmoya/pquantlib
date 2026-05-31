"""Commodity — abstract base class for commodity instruments.

# C++ parity: ql/experimental/commodities/commodity.hpp + commodity.cpp
#             (v1.42.1).

A lazy-object NPV holder (subclasses :class:`Instrument`) that additionally
tracks per-trade *secondary costs* and a list of *pricing errors* accumulated
during pricing. Concrete commodity instruments (EnergyCommodity / EnergySwap /
EnergyFuture, ...) land in W7-C and subclass this.

The C++ ``SecondaryCosts`` (``map<string, any>``) and ``SecondaryCostAmounts``
(``map<string, Money>``) typedefs map to plain dicts here.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any

from pquantlib.currencies.money import Money
from pquantlib.instruments.instrument import Instrument

# C++ parity: typedef std::map<std::string, ext::any> SecondaryCosts;
SecondaryCosts = dict[str, Any]
# C++ parity: typedef std::map<std::string, Money> SecondaryCostAmounts;
SecondaryCostAmounts = dict[str, Money]


class PricingErrorLevel(IntEnum):
    """Severity of a pricing error (parity with ``PricingError::Level``)."""

    INFO = 0
    WARNING = 1
    ERROR = 2
    FATAL = 3


@dataclass
class PricingError:
    """A single pricing diagnostic (parity with C++ ``PricingError``)."""

    error_level: PricingErrorLevel
    error: str
    detail: str = ""
    trade_id: str = ""

    # Nested enum alias for the C++ idiom ``PricingError.Level.INFO``.
    Level = PricingErrorLevel

    def __str__(self) -> str:
        prefix = {
            PricingErrorLevel.INFO: "info: ",
            PricingErrorLevel.WARNING: "warning: ",
            PricingErrorLevel.ERROR: "*** error: ",
            PricingErrorLevel.FATAL: "*** fatal: ",
        }[self.error_level]
        out = prefix + self.error
        if self.detail:
            out += f": {self.detail}"
        return out


PricingErrors = list[PricingError]


class Commodity(Instrument):
    """Abstract commodity instrument base (lazy-object NPV + diagnostics).

    Concrete subclasses (W7-C) must implement :meth:`is_expired` and the
    engine plumbing inherited from :class:`Instrument`.
    """

    def __init__(self, secondary_costs: SecondaryCosts | None = None) -> None:
        super().__init__()
        self._secondary_costs: SecondaryCosts = (
            secondary_costs if secondary_costs is not None else {}
        )
        # ``mutable`` in C++: populated during pricing.
        self._pricing_errors: PricingErrors = []
        self._secondary_cost_amounts: SecondaryCostAmounts = {}

    @property
    def secondary_costs(self) -> SecondaryCosts:
        return self._secondary_costs

    @property
    def secondary_cost_amounts(self) -> SecondaryCostAmounts:
        return self._secondary_cost_amounts

    @property
    def pricing_errors(self) -> PricingErrors:
        return self._pricing_errors

    def add_pricing_error(
        self,
        error_level: PricingErrorLevel,
        error: str,
        detail: str = "",
    ) -> None:
        """Append a pricing diagnostic (parity with ``addPricingError``)."""
        self._pricing_errors.append(PricingError(error_level, error, detail))
