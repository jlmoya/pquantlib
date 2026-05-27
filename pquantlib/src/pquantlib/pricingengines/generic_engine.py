"""Generic engine — typed pricing engine via PEP 695 generics.

# C++ parity: ql/pricingengine.hpp ``template<class ArgumentsType,
# class ResultsType> class GenericEngine`` (v1.42.1).

The C++ template binds the engine to its own ``arguments`` /
``results`` types (e.g. ``GenericEngine<VanillaOption::arguments,
VanillaOption::results>``); concrete engines (analytic European,
binomial, MC, FD, etc.) subclass ``GenericEngine<ArgsT, ResultsT>``
and implement ``calculate()``.

The Python port uses PEP 695 generic syntax
``class GenericEngine[ArgsT, ResultsT]``. The base supplies
``get_arguments`` / ``get_results`` / ``reset`` (the latter via
``results_.reset()``); subclasses only implement ``calculate()``.

The base also implements ``Observer.update`` (forwards to
``notify_observers``) so an engine can be wired into a TS / quote /
process observer chain, mirroring C++ ``Observer`` mixin in
``GenericEngine``.
"""

from __future__ import annotations

from abc import abstractmethod

from pquantlib.pricingengines.pricing_engine import (
    PricingEngine,
    PricingEngineArguments,
    PricingEngineResults,
)


class GenericEngine[ArgsT: PricingEngineArguments, ResultsT: PricingEngineResults](PricingEngine):
    """Type-parameterized pricing engine base.

    Subclasses supply concrete ``ArgsT`` and ``ResultsT`` types and
    implement :meth:`calculate`. The base owns the ``arguments`` and
    ``results`` instances.
    """

    def __init__(self, arguments: ArgsT, results: ResultsT) -> None:
        super().__init__()
        self._arguments: ArgsT = arguments
        self._results: ResultsT = results

    def get_arguments(self) -> ArgsT:
        return self._arguments

    def get_results(self) -> ResultsT:
        return self._results

    def reset(self) -> None:
        self._results.reset()

    @abstractmethod
    def calculate(self) -> None:
        """Subclass: compute ``self._results`` from ``self._arguments``."""

    def update(self) -> None:
        """Observer.update — propagate to own observers.

        # C++ parity: ``GenericEngine`` mixes in ``Observer`` and its
        # ``update()`` calls ``notifyObservers()``.
        """
        self.notify_observers()


__all__ = ["GenericEngine"]
