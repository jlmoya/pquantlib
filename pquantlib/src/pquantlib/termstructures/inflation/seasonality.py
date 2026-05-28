"""Seasonality — abstract base for inflation-rate seasonality corrections.

# C++ parity: ql/termstructures/inflation/seasonality.hpp (v1.42.1) —
   ``Seasonality`` interface (top of file).

This module ships only the ``Seasonality`` abstract in Stage 3 because
the ``InflationTermStructure`` constructor takes a ``Seasonality |
None`` parameter and we don't want a pyright forward-ref dangling.
``MultiplicativePriceSeasonality`` (the only concrete in the L7-A
scope) lands separately in Stage 4 — it carries the multiplicative
math + factor-cycle bookkeeping that the abstract by itself doesn't.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from pquantlib.time.date import Date

if TYPE_CHECKING:
    from pquantlib.termstructures.inflation.inflation_term_structure import (
        InflationTermStructure,
    )


class Seasonality(ABC):
    """Abstract base for seasonality corrections.

    # C++ parity: ``Seasonality`` in seasonality.hpp.
    """

    @abstractmethod
    def correct_zero_rate(
        self, d: Date, r: float, ts: InflationTermStructure
    ) -> float:
        """Apply seasonality to a zero-coupon inflation rate."""

    @abstractmethod
    def correct_yoy_rate(
        self, d: Date, r: float, ts: InflationTermStructure
    ) -> float:
        """Apply seasonality to a year-on-year inflation rate."""

    def is_consistent(self, ts: InflationTermStructure) -> bool:
        """Whether the seasonality is consistent with ``ts``.

        # C++ parity: ``Seasonality::isConsistent`` defaults to True.
        """
        del ts
        return True
