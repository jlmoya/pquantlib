"""BinomialVanillaEngine — binomial-tree pricing for vanilla options.

# C++ parity: ql/pricingengines/vanilla/binomialengine.hpp +
# ql/methods/lattices/binomialtree.{hpp,cpp} (v1.42.1).

C++ uses a template parameterised by a tree-builder class:

    template <class T> class BinomialVanillaEngine : public
    VanillaOption::engine { ... };

with T one of ``CoxRossRubinstein`` / ``JarrowRudd`` / ``Tian`` /
``LeisenReimer`` (each a subclass of ``BinomialTree<T>``). The
template uses a generic ``BlackScholesLattice<T>`` to roll back from
maturity to t=0, and ``DiscretizedVanillaOption`` to handle exercise
discounting.

The Python port keeps the C++ structure and replaces only the template
parameter with a ``TreeBuilder`` enum: it builds the concrete
:class:`~pquantlib.methods.lattices.binomial_tree.BinomialTree`, wraps it in
a :class:`~pquantlib.methods.lattices.bsm_lattice.BlackScholesLattice`, and
rolls a :class:`~pquantlib.pricingengines.vanilla.discretized_vanilla_option.DiscretizedVanillaOption`
back through it — the same objects, in the same order, as
``BinomialVanillaEngine<T>::calculate``.

The algorithm:

1. Flatten the term structures into constants: the continuously-compounded
   zero rates at maturity (r, q) and the Black vol at (expiry, spot). This
   is the C++ ``FlatForward(referenceDate, r, rfdc)`` reconstruction.
2. Build ``TimeGrid(maturity, timeSteps)``, the tree, and the lattice.
3. Wrap ``arguments_`` in a ``DiscretizedVanillaOption`` and ``initialize``
   it at the maturity. The exercise condition — American range test,
   European ``isOnTime``, Bermudan per-date ``isOnTime`` — lives in that
   class, not here.
4. Roll back to ``grid[2]`` (3 nodes) for gamma and to ``grid[1]``
   (2 nodes) for delta, following Hull pp. 397-398, then to 0 for the PV.
5. Derive theta from value/delta/gamma via the BSM PDE.
"""

from __future__ import annotations

from enum import IntEnum

from pquantlib import qassert
from pquantlib.instruments.one_asset_option import OneAssetOptionResults
from pquantlib.methods.lattices.binomial_tree import (
    BinomialTree,
    CoxRossRubinstein,
    JarrowRudd,
    LeisenReimer,
    Tian,
)
from pquantlib.methods.lattices.bsm_lattice import BlackScholesLattice
from pquantlib.option import OptionArguments
from pquantlib.payoffs import PlainVanillaPayoff
from pquantlib.pricingengines.generic_engine import GenericEngine
from pquantlib.pricingengines.vanilla.discretized_vanilla_option import (
    DiscretizedVanillaOption,
)
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.volatility.equity_fx.black_constant_vol import BlackConstantVol
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.time.compounding import Compounding
from pquantlib.time.frequency import Frequency
from pquantlib.time.time_grid import TimeGrid


class TreeBuilder(IntEnum):
    """Binomial-tree builder discriminator.

    # C++ parity: replaces the C++ ``template <class T>`` parameter on
    # ``BinomialVanillaEngine``. Each tree builder picks a different
    # (up, down, pu, pd) parameterisation.
    """

    CoxRossRubinstein = 0
    """Cox-Ross-Rubinstein: equal-jumps (up = exp(sigma*sqrt(dt)),
    down = 1/up), risk-neutral pu / pd."""
    JarrowRudd = 1
    """Jarrow-Rudd: equal-probabilities (pu = pd = 0.5), drift-adjusted
    up / down."""
    Tian = 2
    """Tian: third-moment-matching, multiplicative up / down."""
    LeisenReimer = 3
    """Leisen-Reimer: uses Peizer-Pratt method-2 inversion of the
    cumulative binomial to converge much faster than CRR for European
    options; recommended default."""


class BinomialVanillaEngine(GenericEngine[OptionArguments, OneAssetOptionResults]):
    """Binomial-tree vanilla option engine.

    Supports European / American / Bermudan exercise. The tree-builder
    enum selects the up/down/probability parameterisation.

    # C++ parity: ``template <class T> class BinomialVanillaEngine``.

    Greeks: NPV, delta, gamma, theta (via Hull's three-step lookahead
    pattern). vega / rho / dividend_rho are NOT computed by this engine
    (matches C++ behaviour — they require finite-difference re-pricing
    and are left to the caller).
    """

    def __init__(
        self,
        process: GeneralizedBlackScholesProcess,
        time_steps: int,
        tree_builder: TreeBuilder = TreeBuilder.CoxRossRubinstein,
    ) -> None:
        super().__init__(OptionArguments(), OneAssetOptionResults())
        qassert.require(
            time_steps >= 2,
            f"at least 2 time steps required, {time_steps} provided",
        )
        self._process: GeneralizedBlackScholesProcess = process
        self._time_steps: int = time_steps
        self._tree_builder: TreeBuilder = tree_builder
        process.register_with(self)

    def calculate(self) -> None:  # noqa: PLR0915 - one-to-one with the C++ body
        """Run the binomial tree.

        # C++ parity: ``BinomialVanillaEngine<T>::calculate``
        # (binomialengine.hpp:75-172). The ``TreeBuilder`` enum replaces the
        # template specialisation; everything else is one-to-one.
        """
        args = self._arguments
        results = self._results

        qassert.require(args.exercise is not None, "no exercise given")
        qassert.require(args.payoff is not None, "no payoff given")
        assert args.exercise is not None
        assert args.payoff is not None

        process = self._process
        rfdc = process.risk_free_rate().day_counter()
        divdc = process.dividend_yield().day_counter()
        voldc = process.black_volatility().day_counter()
        volcal = process.black_volatility().calendar()

        s0 = process.state_variable().value()
        qassert.require(s0 > 0.0, "negative or null underlying given")
        maturity_date = args.exercise.last_date()
        v = process.black_volatility().black_vol(maturity_date, s0, extrapolate=True)
        r = process.risk_free_rate().zero_rate(
            maturity_date,
            compounding=Compounding.Continuous,
            frequency=Frequency.NoFrequency,
            result_day_counter=rfdc,
        ).rate()
        q = process.dividend_yield().zero_rate(
            maturity_date,
            compounding=Compounding.Continuous,
            frequency=Frequency.NoFrequency,
            result_day_counter=divdc,
        ).rate()
        reference_date = process.risk_free_rate().reference_date()

        qassert.require(
            isinstance(args.payoff, PlainVanillaPayoff),
            "non-plain payoff given",
        )
        assert isinstance(args.payoff, PlainVanillaPayoff)
        payoff: PlainVanillaPayoff = args.payoff

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

        grid = TimeGrid.regular(end=maturity, steps=self._time_steps)
        tree = self._build_tree(bs, maturity, self._time_steps, payoff.strike())
        lattice = BlackScholesLattice(tree, r, maturity, self._time_steps)

        option = DiscretizedVanillaOption(args, process, grid)
        option.initialize(lattice, maturity)

        # Partial derivatives from various points in the tree (Hull, "Options,
        # Futures and other derivatives", 6th ed., pp 397/398).
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
        results.theta = self._black_scholes_theta(
            value=p0, delta=delta, gamma=gamma, s0=s0, r=r, q=q, sigma=v
        )

    # --- tree-builder dispatch ------------------------------------------

    def _build_tree(
        self,
        process: GeneralizedBlackScholesProcess,
        end: float,
        steps: int,
        strike: float,
    ) -> BinomialTree:
        """Instantiate the concrete tree for the selected builder.

        # C++ parity: the ``T`` template argument of
        # ``BinomialVanillaEngine<T>``; ``new T(bs, maturity, timeSteps_,
        # payoff->strike())``.
        """
        if self._tree_builder == TreeBuilder.CoxRossRubinstein:
            return CoxRossRubinstein(process, end, steps, strike)
        if self._tree_builder == TreeBuilder.JarrowRudd:
            return JarrowRudd(process, end, steps, strike)
        if self._tree_builder == TreeBuilder.Tian:
            return Tian(process, end, steps, strike)
        if self._tree_builder == TreeBuilder.LeisenReimer:
            return LeisenReimer(process, end, steps, strike)
        msg = f"unknown tree builder: {self._tree_builder}"
        raise RuntimeError(msg)

    @staticmethod
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
        """Theta from value/delta/gamma using BSM PDE.

        # C++ parity: ``blackScholesTheta`` in
        # ``ql/pricingengines/greeks.hpp`` —
        # theta = r*V - (r-q)*S*delta - 0.5*sigma^2*S^2*gamma.
        """
        return r * value - (r - q) * s0 * delta - 0.5 * sigma * sigma * s0 * s0 * gamma


__all__ = ["BinomialVanillaEngine", "TreeBuilder"]
