"""BlackScholesCalculator — Black-Scholes 1973 calculator (BSM-shifted Black 1976).

# C++ parity: ql/pricingengines/blackscholescalculator.{hpp,cpp}
# (v1.42.1).

Convenience wrapper around :class:`BlackCalculator` that takes
``(payoff, spot, growth, std_dev, discount)`` where
``growth = exp(-q * T)`` is the dividend discount factor. Internally
it computes the forward ``spot * growth / discount`` and routes through
``BlackCalculator``; the Greeks delta/gamma/theta etc. then bind the
spot/growth via spotted overloads.
"""

from __future__ import annotations

from pquantlib import qassert
from pquantlib.payoffs import StrikedTypePayoff
from pquantlib.pricingengines.black_calculator import BlackCalculator


class BlackScholesCalculator(BlackCalculator):
    """BSM calculator: takes spot + growth instead of a precomputed forward.

    # C++ parity: ``BlackScholesCalculator``.
    """

    def __init__(
        self,
        payoff: StrikedTypePayoff,
        spot: float,
        growth: float,
        std_dev: float,
        discount: float = 1.0,
    ) -> None:
        qassert.require(spot > 0.0, f"spot ({spot}) must be positive")
        qassert.require(growth > 0.0, f"growth ({growth}) must be positive")
        forward = spot * growth / discount
        super().__init__(payoff, forward, std_dev, discount)
        self._spot: float = spot
        self._growth: float = growth

    def delta(self) -> float:  # type: ignore[override]
        """BSM spot delta — uses the spot stored on the calculator.

        # C++ parity: ``BlackScholesCalculator::delta()`` (no-arg
        # overload binds the spot from the BSM ctor).
        """
        return super().delta(self._spot)

    def gamma(self) -> float:  # type: ignore[override]
        """BSM spot gamma."""
        return super().gamma(self._spot)

    def theta(self, maturity: float) -> float:  # type: ignore[override]
        """BSM theta."""
        return super().theta(self._spot, maturity)


__all__ = ["BlackScholesCalculator"]
