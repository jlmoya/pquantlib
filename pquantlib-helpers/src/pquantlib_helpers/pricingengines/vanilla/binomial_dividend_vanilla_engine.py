"""BinomialDividendVanillaEngine — binomial pricing for dividend vanilla options.

# Retired-API compat layer — NOT a port of C++ QuantLib v1.42.1.

Java parity:
``org.jquantlib.pricingengines.vanilla.BinomialDividendVanillaEngine<T>``
(jquantlib-helpers).

Prices a
:class:`~pquantlib_helpers.instruments.dividend_vanilla_option.DividendVanillaOption`
on a binomial tree whose underlying nodes are escrow-adjusted for discrete
dividends by
:class:`~pquantlib_helpers.methods.lattices.black_scholes_dividend_lattice.BlackScholesDividendLattice`.
The engine reconstructs constant (flat) curves from the process's zero rates at
maturity — exactly as the Java engine does — builds the tree + dividend
lattice, rolls a discretized vanilla option back to t=0, and extracts
NPV/delta/gamma/theta via the Odegaard three-point pattern.

The Java engine is generic over the tree type (``T extends Tree``); the Python
port takes a :class:`TreeBuilder` enum (matching pquantlib core's
``BinomialVanillaEngine``) and constructs the corresponding concrete tree.
Both CRR (used by the JQuantLib helpers) and Tian are supported.

# The rollback uses a self-contained :class:`_DiscretizedVanillaOption`
# (mirroring the retired JQuantLib ``DiscretizedVanillaOption``: it applies
# ``max(values, payoff(grid))`` directly off the lattice's escrow-adjusted
# grid).  pquantlib core's ``DiscretizedOption`` instead wraps a separate
# underlying asset (the v1.42.1 shape), which does not fit the
# escrow-adjusted-grid model here — hence the local class.
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np

from pquantlib import qassert
from pquantlib.cashflows.dividend import Dividend
from pquantlib.exercise import Exercise
from pquantlib.methods.lattices.binomial_tree import (
    BinomialTree,
    CoxRossRubinstein,
    Tian,
)
from pquantlib.methods.lattices.discretized_asset import DiscretizedAsset
from pquantlib.payoffs import PlainVanillaPayoff
from pquantlib.pricingengines.generic_engine import GenericEngine
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)
from pquantlib.termstructures.volatility.equity_fx.black_constant_vol import (
    BlackConstantVol,
)
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.time.compounding import Compounding
from pquantlib.time.frequency import Frequency
from pquantlib.time.time_grid import TimeGrid
from pquantlib_helpers.instruments.dividend_vanilla_option import (
    DividendVanillaOptionArguments,
    DividendVanillaOptionResults,
)
from pquantlib_helpers.methods.lattices.black_scholes_dividend_lattice import (
    BlackScholesDividendLattice,
)


class DividendTreeBuilder(IntEnum):
    """Binomial-tree builder discriminator for the dividend engine.

    Java parity: the ``T extends Tree`` template parameter of
    ``BinomialDividendVanillaEngine<T>``. The JQuantLib helpers use
    ``CoxRossRubinstein``; ``Tian`` is offered for parity with the C++/Java
    tree family.
    """

    CoxRossRubinstein = 0
    Tian = 1


class _DiscretizedVanillaOption(DiscretizedAsset):
    """Self-contained discretized vanilla option (escrow-adjusted grid).

    Java parity: ``org.jquantlib.pricingengines.vanilla.DiscretizedVanillaOption``
    (the retired self-contained variant — applies ``max(values, payoff(grid))``
    directly off the lattice grid, rather than wrapping a separate underlying
    asset as the v1.42.1 ``DiscretizedOption`` does).
    """

    def __init__(
        self,
        payoff: PlainVanillaPayoff,
        exercise: Exercise,
        process: GeneralizedBlackScholesProcess,
        grid: TimeGrid,
    ) -> None:
        super().__init__()
        self._payoff: PlainVanillaPayoff = payoff
        self._exercise: Exercise = exercise
        # Java parity: stoppingTimes snapped to the supplied grid.
        self._stopping_times: list[float] = []
        for i in range(len(exercise.dates())):
            t = process.time(exercise.date(i))
            if not grid.empty():
                t = grid.closest_time(t)
            self._stopping_times.append(t)

    def reset(self, size: int) -> None:
        # Java parity: ``values_ = new Array(size); adjustValues();``.
        self._values = np.zeros(size, dtype=np.float64)
        self.adjust_values()

    def mandatory_times(self) -> list[float]:
        # Java parity: ``return stoppingTimes;``.
        return list(self._stopping_times)

    def _apply_specific_condition(self) -> None:
        # Java parity: ``values[j] = max(values[j], payoff(grid[j]))`` where
        # ``grid = method().grid(time())`` reads the escrow-adjusted lattice.
        grid = self._require_method().grid(self._time)
        self._values = np.maximum(
            self._values,
            np.array([self._payoff(float(g)) for g in grid], dtype=np.float64),
        )

    def _post_adjust_values_impl(self) -> None:
        # Java parity: ``DiscretizedVanillaOption.postAdjustValuesImpl``.
        now = self._time
        et = self._exercise.type()
        if et == Exercise.Type.American:
            if self._stopping_times[0] <= now <= self._stopping_times[1]:
                self._apply_specific_condition()
        elif et == Exercise.Type.European:
            if self.is_on_time(self._stopping_times[0]):
                self._apply_specific_condition()
        elif et == Exercise.Type.Bermudan:
            for t in self._stopping_times:
                if self.is_on_time(t):
                    self._apply_specific_condition()
        else:  # pragma: no cover - defensive
            qassert.require(False, f"invalid option type: {et!r}")


class BinomialDividendVanillaEngine(
    GenericEngine[DividendVanillaOptionArguments, DividendVanillaOptionResults]
):
    """Binomial engine for dividend vanilla options.

    Java parity: ``BinomialDividendVanillaEngine<T>``.

    Greeks (NPV, delta, gamma, theta) are extracted via the Odegaard
    three-point pattern at the last three tree slices; theta uses the BSM
    PDE identity. vega / rho are NOT computed here (the JQuantLib helpers
    finite-difference them by re-pricing — that belongs in W-S3).
    """

    def __init__(
        self,
        process: GeneralizedBlackScholesProcess,
        time_steps: int,
        tree_builder: DividendTreeBuilder = DividendTreeBuilder.CoxRossRubinstein,
    ) -> None:
        super().__init__(
            DividendVanillaOptionArguments(), DividendVanillaOptionResults()
        )
        qassert.require(
            time_steps >= 2,
            "at least 2 time steps required: calculate() reads grid.at(2), "
            "lattice.underlying(2, 2), and grid.at(1) for the Odegaard "
            "three-point greek extraction (matches BinomialVanillaEngine guard)",
        )
        self._process: GeneralizedBlackScholesProcess = process
        self._time_steps: int = time_steps
        self._tree_builder: DividendTreeBuilder = tree_builder
        process.register_with(self)

    def _make_tree(
        self,
        process: GeneralizedBlackScholesProcess,
        maturity: float,
        strike: float,
    ) -> BinomialTree:
        # Java parity: ``getTreeInstance(bs, maturity, timeSteps, strike)``.
        if self._tree_builder == DividendTreeBuilder.CoxRossRubinstein:
            return CoxRossRubinstein(process, maturity, self._time_steps, strike)
        return Tian(process, maturity, self._time_steps, strike)

    def calculate(self) -> None:  # noqa: PLR0915 - mirrors the Java method body
        """Run the dividend binomial tree.

        Java parity: ``BinomialDividendVanillaEngine.calculate``.
        """
        args = self._arguments
        results = self._results

        qassert.require(args.payoff is not None, "no payoff given")
        qassert.require(args.exercise is not None, "no exercise given")
        assert args.payoff is not None
        assert args.exercise is not None
        qassert.require(
            isinstance(args.payoff, PlainVanillaPayoff), "non-plain payoff given"
        )
        assert isinstance(args.payoff, PlainVanillaPayoff)
        payoff: PlainVanillaPayoff = args.payoff
        exercise: Exercise = args.exercise
        cash_flow: list[Dividend] = args.cash_flow

        process = self._process
        rfdc = process.risk_free_rate().day_counter()
        divdc = process.dividend_yield().day_counter()
        volcal = process.black_volatility().calendar()
        voldc = process.black_volatility().day_counter()

        s0 = process.state_variable().value()
        qassert.require(s0 > 0.0, "negative or null underlying given")
        maturity_date = exercise.last_date()
        v = process.black_volatility().black_vol(maturity_date, s0, extrapolate=True)

        r_rate = process.risk_free_rate().zero_rate(
            maturity_date,
            compounding=Compounding.Continuous,
            frequency=Frequency.NoFrequency,
            result_day_counter=rfdc,
        ).rate()
        q_rate = process.dividend_yield().zero_rate(
            maturity_date,
            compounding=Compounding.Continuous,
            frequency=Frequency.NoFrequency,
            result_day_counter=divdc,
        ).rate()
        reference_date = process.risk_free_rate().reference_date()

        # Java parity: rebuild constant-coefficient flat curves.
        flat_risk_free = FlatForward.from_rate(reference_date, r_rate, rfdc)
        flat_dividends = FlatForward.from_rate(reference_date, q_rate, divdc)
        flat_vol = BlackConstantVol(
            reference_date=reference_date,
            calendar=volcal,
            day_counter=voldc,
            volatility=v,
        )

        maturity = rfdc.year_fraction(reference_date, maturity_date)

        bs = GeneralizedBlackScholesProcess(
            x0=process.state_variable(),
            dividend_ts=flat_dividends,
            risk_free_ts=flat_risk_free,
            black_vol_ts=flat_vol,
        )
        grid = TimeGrid.regular(end=maturity, steps=self._time_steps)
        tree = self._make_tree(bs, maturity, payoff.strike())

        lattice = BlackScholesDividendLattice(
            tree,
            r_rate,
            maturity,
            self._time_steps,
            rfdc,
            grid,
            reference_date,
            cash_flow,
        )
        option = _DiscretizedVanillaOption(payoff, exercise, process, grid)

        option.initialize(lattice, maturity)

        # Odegaard three-point partial derivatives.
        # Java parity: rollback to grid.at(2), read high node + s2.
        option.rollback(grid.at(2))
        va2 = option.values
        qassert.require(len(va2) == 3, "expect 3 nodes in grid at second step")
        p2h = float(va2[2])  # high-price option value
        s2 = lattice.underlying(2, 2)  # high price underlying

        # Rollback to grid.at(1), read upper node value.
        option.rollback(grid.at(1))
        va = option.values
        qassert.require(len(va) == 2, "expect 2 nodes in grid at first step")
        p1 = float(va[1])

        # Rollback to t=0.
        option.rollback(0.0)
        p0 = option.present_value()
        s1 = lattice.underlying(1, 1)

        delta0 = (p1 - p0) / (s1 - s0)
        delta1 = (p2h - p1) / (s2 - s1)

        results.value = p0
        results.delta = delta0
        results.gamma = 2.0 * (delta1 - delta0) / (s2 - s0)
        results.theta = _black_scholes_theta(
            value=p0,
            delta=delta0,
            gamma=results.gamma,
            s0=s0,
            r=r_rate,
            q=q_rate,
            sigma=v,
        )


# duplicated from BinomialVanillaEngine._black_scholes_theta (binomial_engine.py)
def _black_scholes_theta(
    *,
    value: float,
    delta: float,
    gamma: float,
    s0: float,
    r: float,
    q: float,
    sigma: float,
) -> float:
    """Theta from value/delta/gamma via the BSM PDE.

    Java parity: ``Greeks.blackScholesTheta`` —
    ``theta = r*V - (r-q)*S*delta - 0.5*sigma^2*S^2*gamma``.
    """
    return r * value - (r - q) * s0 * delta - 0.5 * sigma * sigma * s0 * s0 * gamma


__all__ = [
    "BinomialDividendVanillaEngine",
    "DividendTreeBuilder",
]
