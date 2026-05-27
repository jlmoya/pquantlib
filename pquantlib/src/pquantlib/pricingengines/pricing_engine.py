"""Pricing engine abstract base.

# C++ parity: ql/pricingengine.hpp (v1.42.1).

The C++ interface is:

    class PricingEngine : public Observable {
      public:
        class arguments;
        class results;
        virtual arguments* getArguments() const = 0;
        virtual const results* getResults() const = 0;
        virtual void reset() = 0;
        virtual void calculate() const = 0;
    };

with nested ``arguments`` and ``results`` abstract types.

The Python port uses three module-level abstract classes:

* ``PricingEngineArguments`` — base for engine input arguments. C++
  nested-class equivalent. Subclasses declare typed argument fields
  and override ``validate()``.
* ``PricingEngineResults`` — base for engine output results. C++
  nested-class equivalent. Subclasses declare result fields and
  override ``reset()``.
* ``PricingEngine`` — observable engine that produces a result from
  the supplied arguments. Subclasses (specifically ``GenericEngine``)
  hold concrete ``arguments`` + ``results`` instances.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from pquantlib.patterns.observer import Observable


class PricingEngineArguments(ABC):
    """Abstract engine-arguments carrier.

    # C++ parity: ``PricingEngine::arguments`` nested class. Subclasses
    # carry input fields and validate them; the engine reads them in
    # ``calculate()``.
    """

    @abstractmethod
    def validate(self) -> None:
        """Raise unless the argument set is valid. Called by Instrument."""


class PricingEngineResults(ABC):
    """Abstract engine-results carrier.

    # C++ parity: ``PricingEngine::results`` nested class. Subclasses
    # carry output fields and reset them between calculations.
    """

    @abstractmethod
    def reset(self) -> None:
        """Clear all fields to their null/default values."""


class PricingEngine(Observable, ABC):
    """Abstract pricing engine.

    # C++ parity: ``class PricingEngine : public Observable``.

    Subclasses provide concrete ``arguments`` + ``results`` instances
    via :meth:`get_arguments` + :meth:`get_results`, and implement
    :meth:`calculate` to fill results from arguments.
    """

    def __init__(self) -> None:
        Observable.__init__(self)

    @abstractmethod
    def get_arguments(self) -> PricingEngineArguments:
        """Return the (mutable) arguments instance owned by this engine."""

    @abstractmethod
    def get_results(self) -> PricingEngineResults:
        """Return the (mutable) results instance owned by this engine."""

    @abstractmethod
    def reset(self) -> None:
        """Reset the results to null/default values."""

    @abstractmethod
    def calculate(self) -> None:
        """Compute the results given the current arguments."""


__all__ = [
    "PricingEngine",
    "PricingEngineArguments",
    "PricingEngineResults",
]
