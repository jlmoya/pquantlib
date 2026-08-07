"""Basket option (multi-asset).

# C++ parity: ql/instruments/basketoption.{hpp,cpp} +
# ql/instruments/multiassetoption.{hpp,cpp} (v1.42.1).

C++ design:

* ``BasketPayoff`` — payoff hierarchy that delegates ``operator()(price)``
  to an embedded "base payoff" (a one-asset Payoff like PlainVanilla).
  Adds ``accumulate(Array)`` that reduces a vector of asset prices to a
  scalar; ``operator()(Array)`` then applies the base payoff to the
  reduced scalar.
* ``MinBasketPayoff`` — accumulate = min over assets.
* ``MaxBasketPayoff`` — accumulate = max over assets.
* ``AverageBasketPayoff`` — accumulate = weighted average.
* ``SpreadBasketPayoff`` — 2-asset spread, accumulate = ``a[0] - a[1]``.
* ``BasketOption`` — wraps a ``BasketPayoff`` + ``Exercise``.

The Python port simplifies:

* ``MultiAssetOption`` is not a separate class — the Python
  ``BasketOption`` inherits directly from ``Option``. Its results
  carrier does mirror ``MultiAssetOption::results`` (``Instrument::results``
  + ``Greeks``), because ``Fd2dBlackScholesVanillaEngine`` fills
  ``delta``, ``gamma`` and ``theta`` there. Analytic basket engines fill
  ``value`` only, so the Greek accessors raise the C++ "delta not
  provided" style error unless an engine supplied them.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence
from typing import final

import numpy as np

from pquantlib import qassert
from pquantlib.exercise import Exercise
from pquantlib.instruments.instrument import Instrument
from pquantlib.option import Greeks, Option
from pquantlib.payoffs import Payoff
from pquantlib.pricingengines.pricing_engine import (
    PricingEngineArguments,
    PricingEngineResults,
)


class BasketPayoff(Payoff):
    """Abstract basket payoff — delegates ``__call__(price)`` to a base
    Payoff and adds ``accumulate(prices)`` to reduce a price vector to
    a scalar.

    # C++ parity: ``BasketPayoff(ext::shared_ptr<Payoff>)``. Subclasses
    # override ``accumulate(Array)``; ``operator()(Array)`` then routes
    # through the base payoff.
    """

    def __init__(self, base_payoff: Payoff) -> None:
        self._base_payoff: Payoff = base_payoff

    def name(self) -> str:
        return self._base_payoff.name()

    def description(self) -> str:
        return self._base_payoff.description()

    def __call__(self, price: float) -> float:
        """Apply the base payoff to a scalar.

        # C++ parity: ``BasketPayoff::operator()(Real)`` forwards to the
        # base payoff.
        """
        return self._base_payoff(price)

    def base_payoff(self) -> Payoff:
        """Access the embedded one-asset payoff."""
        return self._base_payoff

    def evaluate(self, prices: Sequence[float]) -> float:
        """Reduce ``prices`` to a scalar via ``accumulate``, then apply
        the base payoff.

        # C++ parity: ``BasketPayoff::operator()(const Array&)``.
        """
        return self._base_payoff(self.accumulate(prices))

    @abstractmethod
    def accumulate(self, prices: Sequence[float]) -> float:
        """Reduce a sequence of asset prices to a single scalar.

        # C++ parity: ``BasketPayoff::accumulate(const Array&) = 0``.
        """


@final
class MinBasketPayoff(BasketPayoff):
    """Min-basket payoff: accumulate = min over assets.

    # C++ parity: ``MinBasketPayoff(const ext::shared_ptr<Payoff>&)``.
    """

    def accumulate(self, prices: Sequence[float]) -> float:
        qassert.require(len(prices) > 0, "empty price array")
        return min(prices)


@final
class MaxBasketPayoff(BasketPayoff):
    """Max-basket payoff: accumulate = max over assets.

    # C++ parity: ``MaxBasketPayoff(const ext::shared_ptr<Payoff>&)``.
    """

    def accumulate(self, prices: Sequence[float]) -> float:
        qassert.require(len(prices) > 0, "empty price array")
        return max(prices)


@final
class AverageBasketPayoff(BasketPayoff):
    """Average-basket payoff: accumulate = weighted average.

    # C++ parity: ``AverageBasketPayoff(payoff, weights)`` and
    # ``AverageBasketPayoff(payoff, n)`` (uniform 1/n weights).
    """

    def __init__(
        self,
        base_payoff: Payoff,
        weights: Sequence[float] | None = None,
        *,
        n: int | None = None,
    ) -> None:
        super().__init__(base_payoff)
        if weights is not None:
            self._weights: np.ndarray = np.asarray(weights, dtype=np.float64)
        else:
            qassert.require(
                n is not None and n > 0,
                "weights or positive ``n`` (uniform) required",
            )
            assert n is not None
            self._weights = np.full(n, 1.0 / float(n), dtype=np.float64)

    def accumulate(self, prices: Sequence[float]) -> float:
        arr = np.asarray(prices, dtype=np.float64)
        qassert.require(
            arr.shape == self._weights.shape,
            f"prices/weights length mismatch: {arr.shape} vs {self._weights.shape}",
        )
        return float(np.dot(self._weights, arr))

    def weights(self) -> np.ndarray:
        return self._weights


@final
class SpreadBasketPayoff(BasketPayoff):
    """2-asset spread payoff: accumulate = ``a[0] - a[1]``.

    # C++ parity: ``SpreadBasketPayoff(const ext::shared_ptr<Payoff>&)``.
    """

    def accumulate(self, prices: Sequence[float]) -> float:
        qassert.require(
            len(prices) == 2, "spread payoff is only defined for two underlyings"
        )
        return prices[0] - prices[1]


class BasketOptionResults(Greeks):
    """Results carrier for a basket option.

    # C++ parity: ``class MultiAssetOption::results : public
    # Instrument::results, public Greeks`` (multiassetoption.hpp:62-69).
    # In this port ``Greeks`` already derives from ``InstrumentResults``,
    # so the C++ virtual-inheritance diamond collapses to a single base
    # and ``reset()`` needs no override.
    """


class BasketOption(Option):
    """Basket option on N assets via a ``BasketPayoff`` + ``Exercise``.

    # C++ parity: ``BasketOption(ext::shared_ptr<BasketPayoff>,
    # ext::shared_ptr<Exercise>)``. Inherits from C++'s
    # ``MultiAssetOption``; the Python port routes through ``Option``
    # directly and folds ``MultiAssetOption``'s Greek accessors in here.
    """

    def __init__(self, payoff: BasketPayoff, exercise: Exercise) -> None:
        super().__init__(payoff, exercise)
        self._delta: float | None = None
        self._gamma: float | None = None
        self._theta: float | None = None

    # --- greeks ----------------------------------------------------------

    def delta(self) -> float:
        """# C++ parity: ``MultiAssetOption::delta``."""
        self.calculate()
        qassert.require(self._delta is not None, "delta not provided")
        assert self._delta is not None
        return self._delta

    def gamma(self) -> float:
        """# C++ parity: ``MultiAssetOption::gamma``."""
        self.calculate()
        qassert.require(self._gamma is not None, "gamma not provided")
        assert self._gamma is not None
        return self._gamma

    def theta(self) -> float:
        """# C++ parity: ``MultiAssetOption::theta``."""
        self.calculate()
        qassert.require(self._theta is not None, "theta not provided")
        assert self._theta is not None
        return self._theta

    def is_expired(self) -> bool:
        """Return ``False`` — defers to engine.

        # C++ parity: ``MultiAssetOption::isExpired`` uses
        # ``Settings::evaluationDate``. Not wired into pquantlib yet
        # (Phase 1 carve-out).
        """
        return False

    def setup_arguments(self, args: PricingEngineArguments) -> None:
        """Copy payoff + exercise into engine arguments — same as Option base."""
        Option.setup_arguments(self, args)

    def fetch_results(self, results: PricingEngineResults) -> None:
        """Pull value and Greeks out of the engine results.

        # C++ parity: ``MultiAssetOption::fetchResults``.
        """
        Instrument.fetch_results(self, results)
        if isinstance(results, Greeks):
            self._delta = results.delta
            self._gamma = results.gamma
            self._theta = results.theta

    def setup_expired(self) -> None:
        """# C++ parity: ``MultiAssetOption::setupExpired``."""
        Instrument.setup_expired(self)
        self._delta = 0.0
        self._gamma = 0.0
        self._theta = 0.0


__all__ = [
    "AverageBasketPayoff",
    "BasketOption",
    "BasketOptionResults",
    "BasketPayoff",
    "MaxBasketPayoff",
    "MinBasketPayoff",
    "SpreadBasketPayoff",
]
