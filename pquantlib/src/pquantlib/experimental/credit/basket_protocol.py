"""Basket protocol — structural-typing surface for credit baskets.

# C++ parity: ql/experimental/credit/basket.hpp (v1.42.1) — partial.

The C++ ``Basket`` class is heavyweight (LazyObject + DefaultLossModel
hooks + tranche math + cached eval-date snapshots). Phase 11 W3-C will
land the full port; W3-D needs only the slice that ``IntegralNTDEngine``
and ``NthToDefault`` actually call.

This Protocol defines that slice so:

* ``IntegralNTDEngine`` can be implemented and unit-tested against a
  lightweight in-memory stub.

* The full Basket (when it lands in W3-C) satisfies the Protocol
  structurally — no inheritance required, just method signatures.

* Tests that need only the NTD pricing surface don't pull the entire
  loss-model layer into scope.

# C++ parity divergence: Python ``Protocol`` (PEP 544) replaces
# concrete-type coupling. The C++ basket exposes ~30 methods; we
# surface only the seven used by the engines in this cluster.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pquantlib.instruments.claim import Claim
from pquantlib.time.date import Date


@runtime_checkable
class BasketProtocol(Protocol):
    """Minimal basket surface used by ``IntegralNTDEngine``.

    Implementations must provide deterministic, cacheable answers to
    the methods below. The C++ ``Basket`` overrides ``LazyObject`` to
    cache; Python implementations can cache or compute on demand.
    """

    def size(self) -> int:
        """Number of names in the basket at inception.

        # C++ parity: ``Basket::size``.
        """
        ...

    def names(self) -> list[str]:
        """Names in the basket at inception.

        # C++ parity: ``Basket::names``.
        """
        ...

    def ref_date(self) -> Date:
        """Basket inception date.

        # C++ parity: ``Basket::refDate``.
        """
        ...

    def claim(self) -> Claim:
        """Default-claim model (recovery hook).

        # C++ parity: ``Basket::claim``.
        """
        ...

    def remaining_size(self) -> int:
        """Number of names still alive at the evaluation date.

        # C++ parity: ``Basket::remainingSize``.
        """
        ...

    def remaining_notional(self) -> float:
        """Total notional of names still alive at the evaluation date.

        # C++ parity: ``Basket::remainingNotional()``.
        """
        ...

    def recovery_rate(self, d: Date, i: int) -> float:
        """Recovery rate for name ``i`` at date ``d``.

        # C++ parity: ``Basket::recoveryRate(Date, Size)``.
        """
        ...

    def prob_at_least_n_events(self, n: int, d: Date) -> float:
        """Probability that at least ``n`` defaults have occurred by ``d``.

        # C++ parity: ``Basket::probAtLeastNEvents``. Encapsulates the
        # default-loss-model call; for NTD pricing this is the only
        # basket-loss statistic actually consumed.
        """
        ...


__all__ = ["BasketProtocol"]
