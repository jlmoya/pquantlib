"""VanillaVPPOption — vanilla virtual-power-plant option.

# C++ parity: ql/experimental/finitedifferences/vanillavppoption.{hpp,cpp}
# (v1.42.1).

A virtual power plant (VPP) is a financial instrument that pays the
spark-spread profit between a power index and a fuel (gas) index,
subject to operational constraints (min up-time, min down-time,
start-up fuel & fixed cost). Optionally an upper bound on the number
of starts or on the total running hours can be enforced.

The C++ ``VanillaVPPOption`` is a :class:`MultiAssetOption` whose
embedded payoff is an :class:`AverageBasketPayoff` with weights
``[1.0, -heat_rate]`` over an embedded ``IdenticalPayoff`` (returns
its argument unchanged). The Python port mirrors that shape exactly.

``arguments.validate()`` enforces the C++ invariant
``nStarts == Null<Size> || nRunningHours == Null<Size>`` — only one
of the two limits can be set at a time. Both limits omitted (the
vanilla case) is allowed.
"""

from __future__ import annotations

from pquantlib import qassert
from pquantlib.experimental.finitedifferences.swing_exercise import SwingExercise
from pquantlib.instruments.basket_option import AverageBasketPayoff
from pquantlib.instruments.multi_asset_option import MultiAssetOption
from pquantlib.option import OptionArguments
from pquantlib.payoffs import Payoff
from pquantlib.pricingengines.pricing_engine import PricingEngineArguments


class _IdenticalPayoff(Payoff):
    """Identity payoff: ``f(p) = p``.

    # C++ parity: ql/experimental/finitedifferences/vanillavppoption.cpp
    # anonymous-namespace ``IdenticalPayoff``.
    """

    def name(self) -> str:
        return "IdenticalPayoff"

    def description(self) -> str:
        return self.name()

    def __call__(self, price: float) -> float:
        return price


class VanillaVPPOptionArguments(OptionArguments):
    """Engine arguments for :class:`VanillaVPPOption`.

    # C++ parity: ``VanillaVPPOption::arguments``.

    All fields default to ``None`` until populated by
    :meth:`VanillaVPPOption.setup_arguments`. ``n_starts`` /
    ``n_running_hours`` are ``None`` to represent the C++
    ``Null<Size>()`` sentinel; at most one may be set.
    """

    def __init__(self) -> None:
        super().__init__()
        self.heat_rate: float | None = None
        self.p_min: float | None = None
        self.p_max: float | None = None
        self.t_min_up: int | None = None
        self.t_min_down: int | None = None
        self.start_up_fuel: float | None = None
        self.start_up_fix_cost: float | None = None
        # C++ ``Null<Size>`` sentinel → ``None`` here.
        self.n_starts: int | None = None
        self.n_running_hours: int | None = None

    def validate(self) -> None:
        # C++ parity: ``VanillaVPPOption::arguments::validate()``.
        qassert.require(self.exercise is not None, "no exercise given")
        qassert.require(
            self.n_starts is None or self.n_running_hours is None,
            "either a start limit or fuel limit is supported",
        )


class VanillaVPPOption(MultiAssetOption):
    """Vanilla virtual-power-plant option.

    # C++ parity: ``class VanillaVPPOption : public MultiAssetOption``.

    The instrument carries the operational parameters of the plant and
    delegates to a pricing engine (typically
    :class:`DynProgVPPIntrinsicValueEngine` for intrinsic value, or
    a FD engine for full pricing under a stochastic price model).

    Args:
        heat_rate: MMBtu fuel consumed per MWh of power produced (the
            spark-spread weight).
        p_min: Minimum power output when running (MW).
        p_max: Maximum power output when running (MW).
        t_min_up: Minimum consecutive hours the plant must run after a
            start. C++ ``Size``.
        t_min_down: Minimum consecutive hours the plant must remain off
            after a stop. C++ ``Size``.
        start_up_fuel: Fuel consumed at each start-up (MMBtu).
        start_up_fix_cost: Cash cost charged at each start-up.
        exercise: Swing exercise specifying the hourly (or sub-daily)
            exercise instants.
        n_starts: Optional upper bound on the number of starts over the
            option's life. Mutually exclusive with ``n_running_hours``.
        n_running_hours: Optional upper bound on the total fuel quota
            measured in running hours. Mutually exclusive with
            ``n_starts``.
    """

    def __init__(
        self,
        heat_rate: float,
        p_min: float,
        p_max: float,
        t_min_up: int,
        t_min_down: int,
        start_up_fuel: float,
        start_up_fix_cost: float,
        exercise: SwingExercise,
        n_starts: int | None = None,
        n_running_hours: int | None = None,
    ) -> None:
        # C++ parity: ``MultiAssetOption(ext::shared_ptr<Payoff>(), exercise)``
        # followed by overwriting ``payoff_`` with the AverageBasketPayoff
        # with weights ``[1, -heat_rate]``. Python builds the payoff
        # upfront and passes it through.
        weights = [1.0, -heat_rate]
        payoff = AverageBasketPayoff(_IdenticalPayoff(), weights=weights)
        super().__init__(payoff, exercise)
        self._heat_rate: float = float(heat_rate)
        self._p_min: float = float(p_min)
        self._p_max: float = float(p_max)
        self._t_min_up: int = int(t_min_up)
        self._t_min_down: int = int(t_min_down)
        self._start_up_fuel: float = float(start_up_fuel)
        self._start_up_fix_cost: float = float(start_up_fix_cost)
        self._n_starts: int | None = n_starts
        self._n_running_hours: int | None = n_running_hours

    # -- accessors -------------------------------------------------------

    def heat_rate(self) -> float:
        return self._heat_rate

    def p_min(self) -> float:
        return self._p_min

    def p_max(self) -> float:
        return self._p_max

    def t_min_up(self) -> int:
        return self._t_min_up

    def t_min_down(self) -> int:
        return self._t_min_down

    def start_up_fuel(self) -> float:
        return self._start_up_fuel

    def start_up_fix_cost(self) -> float:
        return self._start_up_fix_cost

    def n_starts(self) -> int | None:
        return self._n_starts

    def n_running_hours(self) -> int | None:
        return self._n_running_hours

    # -- engine plumbing --------------------------------------------------

    def is_expired(self) -> bool:
        """Always returns ``False`` — deferred to engine.

        # C++ parity: ``VanillaVPPOption::isExpired`` consults
        # ``Settings::evaluationDate``. The Python port preserves the
        # MultiAssetOption convention of deferring to the engine.
        """
        return False

    def setup_arguments(self, args: PricingEngineArguments) -> None:
        """Copy the VPP parameters into the engine arguments.

        # C++ parity: ``VanillaVPPOption::setupArguments``.
        """
        super().setup_arguments(args)
        qassert.require(
            isinstance(args, VanillaVPPOptionArguments),
            "wrong argument type (expected VanillaVPPOptionArguments)",
        )
        assert isinstance(args, VanillaVPPOptionArguments)
        args.heat_rate = self._heat_rate
        args.p_min = self._p_min
        args.p_max = self._p_max
        args.t_min_up = self._t_min_up
        args.t_min_down = self._t_min_down
        args.start_up_fuel = self._start_up_fuel
        args.start_up_fix_cost = self._start_up_fix_cost
        args.n_starts = self._n_starts
        args.n_running_hours = self._n_running_hours


__all__ = ["VanillaVPPOption", "VanillaVPPOptionArguments"]
