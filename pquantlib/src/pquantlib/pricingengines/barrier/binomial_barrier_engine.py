"""BinomialBarrierEngine — binomial-tree pricing for single-barrier options.

# C++ parity: ql/pricingengines/barrier/binomialbarrierengine.hpp (v1.43) —
# ``template <class T, class D> class BinomialBarrierEngine``.

C++ has **two** template parameters and gives neither a default:

* ``T`` — the tree builder (``JarrowRudd``, ``CoxRossRubinstein``,
  ``AdditiveEQPBinomialTree``, ``Trigeorgis``, ``Tian``, ``LeisenReimer``,
  ``Joshi4``);
* ``D`` — the discretized asset, i.e. ``DiscretizedBarrierOption`` or
  ``DiscretizedDermanKaniBarrierOption``.

The Python port keeps both as required keyword arguments carrying the classes
themselves rather than an enum, so ``BinomialBarrierEngine<CoxRossRubinstein,
DiscretizedBarrierOption>(process, 400)`` transcribes directly to
``BinomialBarrierEngine(process, 400, tree=CoxRossRubinstein,
discretization=DiscretizedBarrierOption)``.  They are required, matching C++,
where omitting a template argument is a compile error.  (The
``TreeBuilder`` enum on
:class:`~pquantlib.pricingengines.vanilla.binomial_engine.BinomialVanillaEngine`
covers only four of the seven builders, so it cannot serve here.)

Two details a port routinely gets wrong
---------------------------------------

**Theta.** ``results_.theta = (p2m - p0) / grid[2]`` — a finite difference
between the *middle* node of the third-last slice and the t=0 value, at the
same underlying price.  This is **not** the Black-Scholes-PDE theta that
``BinomialVanillaEngine`` computes from value/delta/gamma.

**Boyle-Lau step correction.**  Guarded by
``std::is_base_of_v<CoxRossRubinstein, T>``, so it fires for
``CoxRossRubinstein`` and for nothing else — in particular *not* for
``Trigeorgis``, which is a sibling under ``EqualJumpsBinomialTree``, not a
subclass.  When it fires, the requested ``timeSteps`` is promoted to the first
local minimum of the barrier-node misalignment (Boyle & Lau, *Journal of
Derivatives* 1/1994), then clamped at ``maxTimeSteps``.  ``maxTimeSteps``
defaults to ``max(1000, 5 * timeSteps)``; passing ``maxTimeSteps ==
timeSteps`` disables the correction entirely.  The promoted count drives the
``TimeGrid``, the tree *and* the lattice — not just the tree.

There are no ``additionalResults`` and the tree is built once, not twice.
"""

from __future__ import annotations

import math

from pquantlib import qassert
from pquantlib.instruments.barrier_option import BarrierOptionArguments, BarrierType
from pquantlib.instruments.one_asset_option import OneAssetOptionResults
from pquantlib.math.closeness import close
from pquantlib.methods.lattices.binomial_tree import (
    AdditiveEQPBinomialTree,
    CoxRossRubinstein,
    JarrowRudd,
    Joshi4,
    LeisenReimer,
    Tian,
    Trigeorgis,
)
from pquantlib.methods.lattices.bsm_lattice import BlackScholesLattice
from pquantlib.payoffs import StrikedTypePayoff
from pquantlib.pricingengines.barrier.discretized_barrier_option import (
    DiscretizedBarrierOption,
    DiscretizedDermanKaniBarrierOption,
)
from pquantlib.pricingengines.generic_engine import GenericEngine
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.volatility.equity_fx.black_constant_vol import (
    BlackConstantVol,
)
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.time.compounding import Compounding
from pquantlib.time.frequency import Frequency
from pquantlib.time.time_grid import TimeGrid

#: The seven tree builders C++ can instantiate ``BinomialBarrierEngine<T, D>``
#: with — every concrete subclass of ``BinomialTree`` in
#: ql/methods/lattices/binomialtree.hpp.
#:
#: # C++ parity: the ``T`` template parameter (binomialbarrierengine.hpp:49).
type BarrierTreeBuilder = (
    type[JarrowRudd]
    | type[CoxRossRubinstein]
    | type[AdditiveEQPBinomialTree]
    | type[Trigeorgis]
    | type[Tian]
    | type[LeisenReimer]
    | type[Joshi4]
)

#: The two discretized assets C++ instantiates the engine with.
#:
#: # C++ parity: the ``D`` template parameter (binomialbarrierengine.hpp:49).
type BarrierDiscretization = (
    type[DiscretizedBarrierOption] | type[DiscretizedDermanKaniBarrierOption]
)


class BinomialBarrierEngine(
    GenericEngine[BarrierOptionArguments, OneAssetOptionResults]
):
    """Binomial-tree pricing engine for single-barrier options.

    # C++ parity: ``BinomialBarrierEngine<T, D>``
    # (binomialbarrierengine.hpp:49-82).

    Supports European / American / Bermudan exercise (the exercise rule lives
    in the discretized asset, not here).  Fills NPV, delta, gamma and theta;
    vega / rho / dividend_rho are not computed, matching C++.
    """

    def __init__(
        self,
        process: GeneralizedBlackScholesProcess,
        time_steps: int,
        max_time_steps: int = 0,
        *,
        tree: BarrierTreeBuilder,
        discretization: BarrierDiscretization,
    ) -> None:
        """# C++ parity: ``BinomialBarrierEngine::BinomialBarrierEngine``
        # (binomialbarrierengine.hpp:61-75).

        ``max_time_steps`` limits the Boyle-Lau promotion.  Zero (the default)
        means "use the heuristic ``max(1000, 5 * time_steps)``"; equal to
        ``time_steps`` disables the promotion.
        """
        super().__init__(BarrierOptionArguments(), OneAssetOptionResults())
        qassert.require(
            time_steps > 0, f"timeSteps must be positive, {time_steps} not allowed"
        )
        qassert.require(
            max_time_steps == 0 or max_time_steps >= time_steps,
            "maxTimeSteps must be zero or greater than or equal to timeSteps, "
            f"{max_time_steps} not allowed",
        )
        self._process: GeneralizedBlackScholesProcess = process
        self._time_steps: int = time_steps
        self._max_time_steps: int = (
            max(1000, time_steps * 5) if max_time_steps == 0 else max_time_steps
        )
        self._tree: BarrierTreeBuilder = tree
        self._discretization: BarrierDiscretization = discretization
        process.register_with(self)

    # --- helpers ----------------------------------------------------------

    def _triggered(self, underlying: float) -> bool:
        """# C++ parity: ``BarrierOption::engine::triggered``
        # (ql/instruments/barrieroption.cpp:126-137). Strict ``<`` / ``>``:
        # a spot sitting exactly on the barrier is not triggered."""
        barrier = self._arguments.barrier
        assert barrier is not None
        if self._arguments.barrier_type in (BarrierType.DownIn, BarrierType.DownOut):
            return underlying < barrier
        return underlying > barrier

    def _boyle_lau_steps(self, s0: float, vol: float, maturity: float) -> int:
        """Promote ``time_steps`` to the first Boyle-Lau local minimum.

        # C++ parity: binomialbarrierengine.hpp:131-156.

        Only ever called when the tree is (or derives from)
        ``CoxRossRubinstein``; the caller owns that guard, exactly as the
        ``if constexpr``-style ``std::is_base_of_v`` test does in C++.
        """
        barrier = self._arguments.barrier
        assert barrier is not None
        optimum_steps = self._time_steps
        ratio = s0 / barrier if s0 > barrier else barrier / s0
        divisor = math.pow(math.log(ratio), 2)
        if not close(divisor, 0.0):
            for i in range(1, self._time_steps):
                # C++ ``Size(...)`` is a truncation, not a rounding.
                optimum = int((i * i * vol * vol * maturity) / divisor)
                if self._time_steps < optimum:
                    optimum_steps = optimum
                    break  # found first minimum with iterations>=timesteps
        # C++ ``if (optimum_steps > maxTimeSteps_) optimum_steps = maxTimeSteps_;``
        return min(optimum_steps, self._max_time_steps)  # too high, limit

    # --- engine entry point -----------------------------------------------

    def calculate(self) -> None:  # noqa: PLR0915 - one-to-one with the C++ body
        """Build one tree, roll the discretized asset back, read the Greeks off.

        # C++ parity: ``BinomialBarrierEngine<T,D>::calculate``
        # (binomialbarrierengine.hpp:87-214).
        """
        args = self._arguments
        results = self._results

        payoff = args.payoff
        qassert.require(
            isinstance(payoff, StrikedTypePayoff), "non-striked payoff given"
        )
        assert isinstance(payoff, StrikedTypePayoff)
        qassert.require(payoff.strike() > 0.0, "strike must be positive")

        process = self._process
        s0 = process.state_variable().value()
        qassert.require(s0 > 0.0, "negative or null underlying given")
        qassert.require(not self._triggered(s0), "barrier touched")

        qassert.require(args.exercise is not None, "no exercise given")
        assert args.exercise is not None

        rfdc = process.risk_free_rate().day_counter()
        divdc = process.dividend_yield().day_counter()
        voldc = process.black_volatility().day_counter()
        volcal = process.black_volatility().calendar()

        maturity_date = args.exercise.last_date()
        v = process.black_volatility().black_vol(maturity_date, s0)
        r = process.risk_free_rate().zero_rate(
            maturity_date,
            Compounding.Continuous,
            Frequency.NoFrequency,
            result_day_counter=rfdc,
        ).rate()
        q = process.dividend_yield().zero_rate(
            maturity_date,
            Compounding.Continuous,
            Frequency.NoFrequency,
            result_day_counter=divdc,
        ).rate()
        reference_date = process.risk_free_rate().reference_date()

        maturity = rfdc.year_fraction(reference_date, maturity_date)

        # Binomial trees with constant coefficients.
        bs = GeneralizedBlackScholesProcess(
            x0=SimpleQuote(s0),
            dividend_ts=FlatForward.from_rate(
                reference_date=reference_date, forward_rate=q, day_counter=divdc
            ),
            risk_free_ts=FlatForward.from_rate(
                reference_date=reference_date, forward_rate=r, day_counter=rfdc
            ),
            black_vol_ts=BlackConstantVol(
                reference_date=reference_date,
                calendar=volcal,
                day_counter=voldc,
                volatility=v,
            ),
        )

        # Correct timesteps to ensure a (local) minimum, using the Boyle-Lau
        # approach. Works only for CoxRossRubinstein lattices, so it is
        # disabled when T is not CoxRossRubinstein or derived from it.
        barrier = args.barrier
        assert barrier is not None
        optimum_steps = self._time_steps
        if (
            issubclass(self._tree, CoxRossRubinstein)
            and self._max_time_steps > self._time_steps
            and s0 > 0
            and barrier > 0
        ):
            optimum_steps = self._boyle_lau_steps(s0, v, maturity)

        grid = TimeGrid.regular(end=maturity, steps=optimum_steps)
        tree = self._tree(bs, maturity, optimum_steps, payoff.strike())
        lattice = BlackScholesLattice(tree, r, maturity, optimum_steps)

        option = self._discretization(args, process, grid)
        option.initialize(lattice, maturity)

        # Partial derivatives from various points in the tree
        # (Hull, "Options, Futures and other derivatives", 6th ed., pp 397/398).
        option.rollback(grid[2])
        va2 = option.values
        qassert.require(len(va2) == 3, "Expect 3 nodes in grid at second step")
        p2u, p2m, p2d = float(va2[2]), float(va2[1]), float(va2[0])
        s2u = lattice.underlying(2, 2)
        s2m = lattice.underlying(2, 1)
        s2d = lattice.underlying(2, 0)
        delta2u = (p2u - p2m) / (s2u - s2m)
        delta2d = (p2m - p2d) / (s2m - s2d)
        gamma = (delta2u - delta2d) / ((s2u - s2d) / 2)

        option.rollback(grid[1])
        va = option.values
        qassert.require(len(va) == 2, "Expect 2 nodes in grid at first step")
        p1u, p1d = float(va[1]), float(va[0])
        s1u = lattice.underlying(1, 1)
        s1d = lattice.underlying(1, 0)
        delta = (p1u - p1d) / (s1u - s1d)

        option.rollback(0.0)
        p0 = option.present_value()

        results.value = p0
        results.delta = delta
        results.gamma = gamma
        # C++ parity: theta by numerical derivative between the mid value at
        # the third-last step and t=0 — the underlying price is the same, only
        # time varies. NOT the BSM-PDE theta.
        results.theta = (p2m - p0) / grid[2]


__all__ = [
    "BarrierDiscretization",
    "BarrierTreeBuilder",
    "BinomialBarrierEngine",
]
