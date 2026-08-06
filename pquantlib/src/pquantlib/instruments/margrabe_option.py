"""MargrabeOption — option to exchange one asset for another.

# C++ parity: ql/instruments/margrabeoption.{hpp,cpp} (v1.43).

The holder has the right to exchange ``q2`` stocks of the second asset
for ``q1`` stocks of the first at expiration, i.e. the terminal payoff
is ``max(q1*S1 - q2*S2, 0)``.  Because that payoff is not a function of
a single terminal price, the instrument carries a
:class:`~pquantlib.payoffs.NullPayoff` and the quantities travel to the
engine through :class:`MargrabeOptionArguments` — exactly as in C++.

Priced by
:class:`~pquantlib.pricingengines.exotic.analytic_european_margrabe_engine.AnalyticEuropeanMargrabeEngine`
(Margrabe 1978 closed form) or
:class:`~pquantlib.pricingengines.exotic.analytic_american_margrabe_engine.AnalyticAmericanMargrabeEngine`.
"""

from __future__ import annotations

from pquantlib import qassert
from pquantlib.exercise import Exercise
from pquantlib.instruments.multi_asset_option import (
    MultiAssetOption,
    MultiAssetOptionResults,
)
from pquantlib.option import OptionArguments
from pquantlib.payoffs import NullPayoff
from pquantlib.pricingengines.pricing_engine import (
    PricingEngineArguments,
    PricingEngineResults,
)


class MargrabeOptionArguments(OptionArguments):
    """Engine arguments for :class:`MargrabeOption`.

    # C++ parity: ``MargrabeOption::arguments``.
    """

    def __init__(self) -> None:
        super().__init__()
        self.q1: int | None = None
        self.q2: int | None = None

    def validate(self) -> None:
        """# C++ parity: ``MargrabeOption::arguments::validate``."""
        super().validate()
        qassert.require(self.q1 is not None, "unspecified quantity for asset 1")
        qassert.require(self.q2 is not None, "unspecified quantity for asset 2")
        assert self.q1 is not None
        assert self.q2 is not None
        qassert.require(self.q1 > 0, "quantity of asset 1 must be positive")
        qassert.require(self.q2 > 0, "quantity of asset 2 must be positive")


class MargrabeOptionResults(MultiAssetOptionResults):
    """Results carrier for :class:`MargrabeOption`.

    # C++ parity: ``MargrabeOption::results : MultiAssetOption::results``
    # plus the four per-asset Greeks.
    """

    def __init__(self) -> None:
        super().__init__()
        self.delta1: float | None = None
        self.delta2: float | None = None
        self.gamma1: float | None = None
        self.gamma2: float | None = None

    def reset(self) -> None:
        """# C++ parity: ``MargrabeOption::results::reset``."""
        super().reset()
        self.delta1 = None
        self.delta2 = None
        self.gamma1 = None
        self.gamma2 = None


class MargrabeOption(MultiAssetOption):
    """Margrabe exchange option on two assets.

    # C++ parity: ``MargrabeOption(Integer Q1, Integer Q2,
    #               const ext::shared_ptr<Exercise>&)``.

    Args:
        q1: Quantity of the first (received) asset. Must be positive.
        q2: Quantity of the second (delivered) asset. Must be positive.
        exercise: ``EuropeanExercise`` for the European engine,
            ``AmericanExercise`` for the American one.
    """

    def __init__(self, q1: int, q2: int, exercise: Exercise) -> None:
        # C++ constructs the base with a NullPayoff: the exchange payoff
        # is not expressible as a function of a single terminal price.
        super().__init__(NullPayoff(), exercise)
        self._q1: int = q1
        self._q2: int = q2
        self._delta1: float | None = None
        self._delta2: float | None = None
        self._gamma1: float | None = None
        self._gamma2: float | None = None

    # --- Greek accessors --------------------------------------------------

    def delta1(self) -> float:
        """# C++ parity: ``MargrabeOption::delta1``."""
        self.calculate()
        qassert.require(self._delta1 is not None, "delta1 not provided")
        assert self._delta1 is not None
        return self._delta1

    def delta2(self) -> float:
        """# C++ parity: ``MargrabeOption::delta2``."""
        self.calculate()
        qassert.require(self._delta2 is not None, "delta2 not provided")
        assert self._delta2 is not None
        return self._delta2

    def gamma1(self) -> float:
        """# C++ parity: ``MargrabeOption::gamma1``."""
        self.calculate()
        qassert.require(self._gamma1 is not None, "gamma1 not provided")
        assert self._gamma1 is not None
        return self._gamma1

    def gamma2(self) -> float:
        """# C++ parity: ``MargrabeOption::gamma2``."""
        self.calculate()
        qassert.require(self._gamma2 is not None, "gamma2 not provided")
        assert self._gamma2 is not None
        return self._gamma2

    # --- engine plumbing --------------------------------------------------

    def setup_arguments(self, args: PricingEngineArguments) -> None:
        """Inject the two quantities into the engine arguments.

        # C++ parity: ``MargrabeOption::setupArguments``.
        """
        super().setup_arguments(args)
        qassert.require(isinstance(args, MargrabeOptionArguments), "wrong argument type")
        assert isinstance(args, MargrabeOptionArguments)
        args.q1 = self._q1
        args.q2 = self._q2

    def fetch_results(self, results: PricingEngineResults) -> None:
        """Pull the four per-asset Greeks out of the engine results.

        # C++ parity: ``MargrabeOption::fetchResults``.
        """
        super().fetch_results(results)
        qassert.require(isinstance(results, MargrabeOptionResults), "wrong result type")
        assert isinstance(results, MargrabeOptionResults)
        self._delta1 = results.delta1
        self._delta2 = results.delta2
        self._gamma1 = results.gamma1
        self._gamma2 = results.gamma2


__all__ = [
    "MargrabeOption",
    "MargrabeOptionArguments",
    "MargrabeOptionResults",
]
