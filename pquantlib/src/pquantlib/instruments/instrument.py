"""Instrument abstract base + InstrumentResults carrier.

# C++ parity: ql/instrument.hpp + ql/instrument.cpp (v1.42.1).

C++ design:

* ``Instrument : public LazyObject`` — caches NPV until ``update()``
  invalidates.
* ``setPricingEngine(engine)`` registers the engine as an observer
  (so the instrument invalidates on engine update) AND registers the
  instrument WITH the engine (so the engine fires its own observers
  when the instrument changes).
* On ``NPV()`` the instrument calls ``calculate()``, which lazy-calls
  ``performCalculations()`` — which in turn:
  - calls ``engine_->reset()``,
  - copies instrument arguments via ``setupArguments(args)``,
  - validates,
  - calls ``engine_->calculate()``,
  - copies engine results back via ``fetchResults(results)``.

Python port keeps the same lifecycle; ``set_pricing_engine``,
``npv``, ``valuation_date``, ``additional_results``, ``setup_arguments``,
``fetch_results``, and ``is_expired`` are 1:1.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pquantlib import qassert
from pquantlib.patterns.lazy_object import LazyObject
from pquantlib.pricingengines.pricing_engine import (
    PricingEngine,
    PricingEngineArguments,
    PricingEngineResults,
)
from pquantlib.time.date import Date


class InstrumentResults(PricingEngineResults):
    """Generic instrument results: NPV + error estimate + valuation date.

    # C++ parity: ``Instrument::results`` nested class. Subclasses
    # extend with product-specific fields.
    """

    def __init__(self) -> None:
        self.value: float | None = None
        self.error_estimate: float | None = None
        self.valuation_date: Date = Date()
        self.additional_results: dict[str, Any] = {}

    def reset(self) -> None:
        self.value = None
        self.error_estimate = None
        self.valuation_date = Date()
        self.additional_results = {}


class Instrument(LazyObject, ABC):
    """Abstract financial instrument.

    # C++ parity: ``class Instrument : public LazyObject``.

    Subclasses MUST override:
    - ``is_expired() -> bool``: market expiration check; the cache short-
      circuits when expired.

    Subclasses CAN override:
    - ``setup_arguments(args)`` to copy product fields into the engine's
      ``arguments`` carrier (mandatory if a non-trivial engine is used).
    - ``fetch_results(results)`` to copy engine results back; the default
      copies the standard NPV / error / valuation date / additional results
      and works for any engine whose results inherit ``InstrumentResults``.
    - ``perform_calculations`` if no engine is used (rare).
    """

    def __init__(self) -> None:
        super().__init__()
        self._engine: PricingEngine | None = None
        # Result fields — populated by ``fetch_results`` (or by
        # ``setup_expired`` when expired).
        self._npv: float | None = None
        self._error_estimate: float | None = None
        self._valuation_date: Date = Date()
        self._additional_results: dict[str, Any] = {}

    # --- engine plumbing ---------------------------------------------------

    def set_pricing_engine(self, engine: PricingEngine) -> None:
        """Attach an engine; both directions of observer registration.

        # C++ parity: ``Instrument::setPricingEngine`` registers the
        # engine as an observable of the instrument (so engine updates
        # invalidate the instrument's cache), and registers the
        # instrument as observer of the engine (so engine fires
        # ``notifyObservers()`` propagates the cascade).
        """
        if self._engine is not None:
            self._engine.unregister_with(self)
        self._engine = engine
        engine.register_with(self)
        self.update()

    def pricing_engine(self) -> PricingEngine | None:
        return self._engine

    # --- inspectors --------------------------------------------------------

    def npv(self) -> float:
        """Net present value. Triggers calculation if cache is dirty."""
        self.calculate()
        qassert.require(self._npv is not None, "NPV not provided")
        assert self._npv is not None
        return self._npv

    def error_estimate(self) -> float:
        self.calculate()
        qassert.require(self._error_estimate is not None, "error estimate not provided")
        assert self._error_estimate is not None
        return self._error_estimate

    def valuation_date(self) -> Date:
        self.calculate()
        qassert.require(
            self._valuation_date != Date(), "valuation date not provided"
        )
        return self._valuation_date

    def additional_results(self) -> dict[str, Any]:
        self.calculate()
        return self._additional_results

    def result(self, tag: str) -> Any:
        """Lookup an additional result by string tag.

        # C++ parity: ``Instrument::result<T>(tag)`` template; the
        # Python port returns ``Any`` (typing is left to the caller).
        """
        self.calculate()
        qassert.require(tag in self._additional_results, f"{tag} not provided")
        return self._additional_results[tag]

    # --- subclass hooks ---------------------------------------------------

    @abstractmethod
    def is_expired(self) -> bool:
        """Return True if the instrument is past its expiration date."""

    def setup_arguments(self, args: PricingEngineArguments) -> None:
        """Subclass: copy product fields into the engine's argument carrier.

        Default implementation is a no-op (matches C++ default that
        only sets the base arguments shared across all instruments).
        """

    def fetch_results(self, results: PricingEngineResults) -> None:
        """Subclass: copy engine results into the instrument's cached fields.

        Default implementation accepts any results that subclass
        ``InstrumentResults`` and copies the standard NPV / error /
        valuation date / additional results fields.
        """
        qassert.require(
            isinstance(results, InstrumentResults),
            "no results returned from pricing engine "
            "(expected InstrumentResults subclass)",
        )
        assert isinstance(results, InstrumentResults)
        self._npv = results.value
        self._error_estimate = results.error_estimate
        self._valuation_date = results.valuation_date
        self._additional_results = dict(results.additional_results)

    # --- lazy-object lifecycle --------------------------------------------

    def calculate(self) -> None:
        """Override LazyObject.calculate to short-circuit when expired.

        # C++ parity: ``Instrument::calculate`` checks ``isExpired()``
        # and, if true, calls ``setupExpired()`` and marks the cache as
        # calculated (skipping the engine).
        """
        if not self._calculated and self.is_expired():
            self.setup_expired()
            self._calculated = True
            return
        super().calculate()

    def setup_expired(self) -> None:
        """Subclass-overridable expired-state setter.

        # C++ parity: ``Instrument::setupExpired`` — zero out NPV /
        # error, clear additional results, leave valuation_date null.
        """
        self._npv = 0.0
        self._error_estimate = 0.0
        self._valuation_date = Date()
        self._additional_results = {}

    def _perform_calculations(self) -> None:
        """Engine-backed default calculation.

        # C++ parity: ``Instrument::performCalculations``.
        """
        qassert.require(self._engine is not None, "null pricing engine")
        assert self._engine is not None
        self._engine.reset()
        args = self._engine.get_arguments()
        self.setup_arguments(args)
        args.validate()
        self._engine.calculate()
        self.fetch_results(self._engine.get_results())


__all__ = ["Instrument", "InstrumentResults"]
