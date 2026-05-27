"""Cross-cluster runtime-checkable Protocols.

# C++ parity: none — this is Python's solution to the cross-cluster
# import-cycle problem.

L3-B (bonds), L3-C (swaps), L3-D (equity options + processes), and
L3-E (forwards) build classes that need to reference each other's
types in signatures. Direct imports create cycles (e.g.
``DiscountingBondEngine`` takes a ``Bond``, but ``Bond.npv()`` calls
the engine — and the bond lives in ``instruments.bond`` while the
engine lives in ``pricingengines.bond.discounting_bond_engine``).

The Protocols defined here let each cluster type ITS OWN inputs as
the relevant Protocol without importing concrete classes. Structural
matching at call time gives full duck-typing semantics; at type-
check time pyright + mypy resolve via :pep:`544` structural
subtyping.

Protocols:

* ``InstrumentProtocol`` — anything that quacks like ``Instrument``.
* ``PricingEngineProtocol`` — anything that quacks like
  ``PricingEngine`` (Args/Results plumbing).
* ``StochasticProcessProtocol`` — anything that quacks like
  ``StochasticProcess`` (drift / diffusion / evolve / initial_values).
  L3-D's processes will conform.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pquantlib.time.date import Date


@runtime_checkable
class InstrumentProtocol(Protocol):
    """Anything that quacks like ``pquantlib.instruments.Instrument``."""

    def npv(self) -> float: ...
    def is_expired(self) -> bool: ...
    def valuation_date(self) -> Date: ...


@runtime_checkable
class PricingEngineProtocol(Protocol):
    """Anything that quacks like ``pquantlib.pricingengines.PricingEngine``."""

    def calculate(self) -> None: ...
    def reset(self) -> None: ...
    def get_arguments(self) -> Any: ...
    def get_results(self) -> Any: ...

    # Observer plumbing (from ``Observable``).
    def register_with(self, observer: Any) -> None: ...
    def unregister_with(self, observer: Any) -> None: ...
    def notify_observers(self) -> None: ...


@runtime_checkable
class StochasticProcessProtocol(Protocol):
    """Anything that quacks like a stochastic process (1-D or multi-D).

    The methods below match the abstract C++ interface
    ``StochasticProcess`` in ``ql/stochasticprocess.hpp`` (drift /
    diffusion / evolve / initial_values). L3-D will land concrete
    process classes that conform structurally.
    """

    def initial_values(self) -> Any: ...
    def drift(self, t: float, x: Any) -> Any: ...
    def diffusion(self, t: float, x: Any) -> Any: ...
    def evolve(self, t0: float, x0: Any, dt: float, dw: Any) -> Any: ...


__all__ = [
    "InstrumentProtocol",
    "PricingEngineProtocol",
    "StochasticProcessProtocol",
]
