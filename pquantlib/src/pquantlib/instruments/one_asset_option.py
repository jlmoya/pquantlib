"""OneAssetOption — abstract option on a single underlying asset.

# C++ parity: ql/instruments/oneassetoption.{hpp,cpp} (v1.42.1).

Adds Greek accessors (delta / gamma / vega / theta / rho /
dividend_rho) that delegate to the engine results (``OneAssetOption.results``,
which inherits from ``Greeks`` + ``MoreGreeks``).

The Python results carrier is ``OneAssetOptionResults``, which
multi-inherits Greeks + MoreGreeks so each Greek accessor is a
single ``getattr`` away.
"""

from __future__ import annotations

from abc import ABC

from pquantlib import qassert
from pquantlib.exercise import Exercise
from pquantlib.option import Greeks, MoreGreeks, Option
from pquantlib.payoffs import Payoff
from pquantlib.pricingengines.pricing_engine import PricingEngineResults


class OneAssetOptionResults(Greeks, MoreGreeks):
    """Combined Greeks + MoreGreeks for single-asset options.

    # C++ parity: ``OneAssetOption::results`` is the diamond inheritance
    # of ``Greeks`` + ``MoreGreeks``; the Python MRO collapses both
    # ``reset()`` calls correctly via ``super().reset()`` chaining.
    """

    def __init__(self) -> None:
        # Both bases share InstrumentResults; the diamond inherits
        # correctly because each class calls super().__init__().
        # Manually walk the MRO to make sure all bases are initialized.
        Greeks.__init__(self)
        # MoreGreeks.__init__ would re-initialize the base
        # InstrumentResults fields — skip that by directly initializing
        # the MoreGreeks-only fields.
        self.itm_cash_probability: float | None = None
        self.delta_forward: float | None = None
        self.elasticity: float | None = None
        self.theta_per_day: float | None = None
        self.strike_sensitivity: float | None = None

    def reset(self) -> None:
        Greeks.reset(self)
        # Reset MoreGreeks-only fields.
        self.itm_cash_probability = None
        self.delta_forward = None
        self.elasticity = None
        self.theta_per_day = None
        self.strike_sensitivity = None


class OneAssetOption(Option, ABC):
    """Abstract option on a single asset.

    # C++ parity: ``class OneAssetOption : public Option``.

    Exposes Greek accessors that proxy to the engine results.
    """

    def __init__(self, payoff: Payoff, exercise: Exercise) -> None:
        super().__init__(payoff, exercise)
        self._delta: float | None = None
        self._gamma: float | None = None
        self._theta: float | None = None
        self._vega: float | None = None
        self._rho: float | None = None
        self._dividend_rho: float | None = None
        self._itm_cash_probability: float | None = None
        self._delta_forward: float | None = None
        self._elasticity: float | None = None
        self._theta_per_day: float | None = None
        self._strike_sensitivity: float | None = None

    # --- Greek accessors --------------------------------------------------

    def delta(self) -> float:
        self.calculate()
        qassert.require(self._delta is not None, "delta not provided")
        assert self._delta is not None
        return self._delta

    def gamma(self) -> float:
        self.calculate()
        qassert.require(self._gamma is not None, "gamma not provided")
        assert self._gamma is not None
        return self._gamma

    def theta(self) -> float:
        self.calculate()
        qassert.require(self._theta is not None, "theta not provided")
        assert self._theta is not None
        return self._theta

    def vega(self) -> float:
        self.calculate()
        qassert.require(self._vega is not None, "vega not provided")
        assert self._vega is not None
        return self._vega

    def rho(self) -> float:
        self.calculate()
        qassert.require(self._rho is not None, "rho not provided")
        assert self._rho is not None
        return self._rho

    def dividend_rho(self) -> float:
        self.calculate()
        qassert.require(self._dividend_rho is not None, "dividend rho not provided")
        assert self._dividend_rho is not None
        return self._dividend_rho

    def itm_cash_probability(self) -> float:
        self.calculate()
        qassert.require(
            self._itm_cash_probability is not None, "ITM cash probability not provided"
        )
        assert self._itm_cash_probability is not None
        return self._itm_cash_probability

    # The four MoreGreeks accessors below were fetched into private fields but
    # never exposed. C++ v1.43 declares all eleven greeks on OneAssetOption
    # (oneassetoption.hpp:45-55); several engines — AnalyticEuropeanEngine,
    # BaroneAdesiWhaleyApproximationEngine, JuQuadraticApproximationEngine,
    # BjerksundStenslandApproximationEngine — fill them, and without the
    # accessor there was no way to read the result back off the instrument.

    def delta_forward(self) -> float:
        """# C++ parity: ``OneAssetOption::deltaForward`` (oneassetoption.cpp:43)."""
        self.calculate()
        qassert.require(self._delta_forward is not None, "forward delta not provided")
        assert self._delta_forward is not None
        return self._delta_forward

    def elasticity(self) -> float:
        """# C++ parity: ``OneAssetOption::elasticity`` (oneassetoption.cpp:50)."""
        self.calculate()
        qassert.require(self._elasticity is not None, "elasticity not provided")
        assert self._elasticity is not None
        return self._elasticity

    def theta_per_day(self) -> float:
        """# C++ parity: ``OneAssetOption::thetaPerDay`` (oneassetoption.cpp:68)."""
        self.calculate()
        qassert.require(self._theta_per_day is not None, "theta per-day not provided")
        assert self._theta_per_day is not None
        return self._theta_per_day

    def strike_sensitivity(self) -> float:
        """# C++ parity: ``OneAssetOption::strikeSensitivity`` (oneassetoption.cpp:92)."""
        self.calculate()
        qassert.require(self._strike_sensitivity is not None, "strike sensitivity not provided")
        assert self._strike_sensitivity is not None
        return self._strike_sensitivity

    # --- result fetch -----------------------------------------------------

    def fetch_results(self, results: PricingEngineResults) -> None:
        """Pull Greeks + MoreGreeks out of the engine results.

        # C++ parity: ``OneAssetOption::fetchResults``.
        """
        super().fetch_results(results)
        qassert.require(
            isinstance(results, OneAssetOptionResults),
            "no Greeks-carrying results returned from pricing engine "
            "(expected OneAssetOptionResults subclass)",
        )
        assert isinstance(results, OneAssetOptionResults)
        self._delta = results.delta
        self._gamma = results.gamma
        self._theta = results.theta
        self._vega = results.vega
        self._rho = results.rho
        self._dividend_rho = results.dividend_rho
        self._itm_cash_probability = results.itm_cash_probability
        self._delta_forward = results.delta_forward
        self._elasticity = results.elasticity
        self._theta_per_day = results.theta_per_day
        self._strike_sensitivity = results.strike_sensitivity

    # --- expired override -------------------------------------------------

    def setup_expired(self) -> None:
        """Clear NPV + all Greeks on expiry.

        # C++ parity: ``OneAssetOption::setupExpired``.
        """
        super().setup_expired()
        self._delta = None
        self._gamma = None
        self._theta = None
        self._vega = None
        self._rho = None
        self._dividend_rho = None
        self._itm_cash_probability = None
        self._delta_forward = None
        self._elasticity = None
        self._theta_per_day = None
        self._strike_sensitivity = None


__all__ = ["OneAssetOption", "OneAssetOptionResults"]
