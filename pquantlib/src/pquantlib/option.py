"""Option abstract base + Option.Arguments / Greeks / MoreGreeks results.

# C++ parity: ql/option.hpp (v1.42.1) — ``class Option : public
# Instrument``.

C++ design:

* ``Option : public Instrument`` carries a ``Payoff`` and an
  ``Exercise``.
* ``Option::arguments`` (nested) is the engine argument bundle
  shared across all option engines — it stores the same ``payoff``
  and ``exercise`` pair.
* ``Greeks`` / ``MoreGreeks`` mixins extend ``PricingEngine::results``
  with the standard Greek fields (delta / gamma / vega / theta /
  rho / dividend_rho / etc.).

The Python port mirrors all three, plus the ``setup_arguments``
default that copies payoff + exercise.
"""

from __future__ import annotations

from abc import ABC

from pquantlib import qassert
from pquantlib.exercise import Exercise
from pquantlib.instruments.instrument import Instrument, InstrumentResults
from pquantlib.payoffs import Payoff
from pquantlib.pricingengines.pricing_engine import PricingEngineArguments


class OptionArguments(PricingEngineArguments):
    """Engine argument bundle for any option.

    # C++ parity: ``Option::arguments`` nested class. Carries the
    # same ``payoff`` and ``exercise`` pair as the instrument.
    """

    def __init__(self) -> None:
        self.payoff: Payoff | None = None
        self.exercise: Exercise | None = None

    def validate(self) -> None:
        qassert.require(self.payoff is not None, "no payoff given")
        qassert.require(self.exercise is not None, "no exercise given")


class Greeks(InstrumentResults):
    """Standard Greek result fields.

    # C++ parity: ``class Greeks : public virtual PricingEngine::results``.

    Subclasses (concrete engines' result types) populate the fields
    they support; consumers check for ``None`` to detect "not
    computed".
    """

    def __init__(self) -> None:
        super().__init__()
        self.delta: float | None = None
        self.gamma: float | None = None
        self.theta: float | None = None
        self.vega: float | None = None
        self.rho: float | None = None
        self.dividend_rho: float | None = None

    def reset(self) -> None:
        super().reset()
        self.delta = None
        self.gamma = None
        self.theta = None
        self.vega = None
        self.rho = None
        self.dividend_rho = None


class MoreGreeks(InstrumentResults):
    """Extended Greek result fields.

    # C++ parity: ``class MoreGreeks : public virtual
    # PricingEngine::results``.
    """

    def __init__(self) -> None:
        super().__init__()
        self.itm_cash_probability: float | None = None
        self.delta_forward: float | None = None
        self.elasticity: float | None = None
        self.theta_per_day: float | None = None
        self.strike_sensitivity: float | None = None

    def reset(self) -> None:
        super().reset()
        self.itm_cash_probability = None
        self.delta_forward = None
        self.elasticity = None
        self.theta_per_day = None
        self.strike_sensitivity = None


class Option(Instrument, ABC):
    """Abstract option = Instrument + Payoff + Exercise.

    # C++ parity: ``class Option : public Instrument``.

    Concrete subclasses (VanillaOption / EuropeanOption / ...) inherit
    via ``OneAssetOption`` in ``instruments.one_asset_option``.
    """

    def __init__(self, payoff: Payoff, exercise: Exercise) -> None:
        super().__init__()
        self._payoff: Payoff = payoff
        self._exercise: Exercise = exercise

    def payoff(self) -> Payoff:
        return self._payoff

    def exercise(self) -> Exercise:
        return self._exercise

    def setup_arguments(self, args: PricingEngineArguments) -> None:
        """Copy payoff + exercise into the engine arguments.

        # C++ parity: ``Option::setupArguments``.
        """
        qassert.require(
            isinstance(args, OptionArguments),
            "wrong argument type (expected OptionArguments subclass)",
        )
        assert isinstance(args, OptionArguments)
        args.payoff = self._payoff
        args.exercise = self._exercise


__all__ = [
    "Greeks",
    "MoreGreeks",
    "Option",
    "OptionArguments",
]
