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

The Python port collapses all this into a single class parameterised
by a ``TreeBuilder`` enum (CRR / JarrowRudd / Tian / LeisenReimer).
Backward induction is implemented directly on numpy arrays — much
simpler than reproducing the full ``Tree`` / ``Lattice`` /
``DiscretizedAsset`` machinery (those are deferred carve-outs; the
binomial path is the only use-case in L3-D).

The algorithm:

1. Build a constant-coefficient GBSM analogue using the spot, the
   continuously-compounded zero rates at maturity (r, q), and the
   constant Black vol evaluated at the expiry/strike — i.e. flatten
   the term structures into single numbers. Matches the C++ pattern
   of constructing a ``FlatForward(referenceDate, r, dc)`` etc.
2. Compute the tree-specific (up, down, pu, pd) coefficients.
3. Initialise the option values at maturity: payoff(S_T) for each
   terminal node.
4. Walk back in time. At each node, the continuation value is
   ``discount * (pu * V_up + pd * V_down)``. For American/Bermudan
   exercise, take max(continuation, exercise) at any exercise date.
5. Extract Greeks from the three-step lookahead nodes (Hull's
   p. 397-398 pattern).
"""

from __future__ import annotations

import math
from enum import IntEnum

import numpy as np
import numpy.typing as npt

from pquantlib import qassert
from pquantlib.exercise import Exercise
from pquantlib.instruments.one_asset_option import OneAssetOptionResults
from pquantlib.math.distributions.binomial_distribution import (
    peizer_pratt_method2_inversion as _peizer_pratt_method2_inversion,
)
from pquantlib.option import OptionArguments
from pquantlib.payoffs import PlainVanillaPayoff
from pquantlib.pricingengines.generic_engine import GenericEngine
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)
from pquantlib.time.compounding import Compounding
from pquantlib.time.frequency import Frequency


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

    def calculate(self) -> None:  # noqa: PLR0915
        """Run the binomial tree.

        # C++ parity: ``BinomialVanillaEngine<T>::calculate``. The
        # branching by ``TreeBuilder`` enum replaces the template
        # specialisation; ``_build_tree_coefficients`` returns
        # ``(up, down, pu, pd, oddSteps)`` where ``oddSteps`` is the
        # actual step count (LeisenReimer / Joshi4 force odd).
        """
        args = self._arguments
        results = self._results

        qassert.require(args.exercise is not None, "no exercise given")
        qassert.require(args.payoff is not None, "no payoff given")
        assert args.exercise is not None
        assert args.payoff is not None

        qassert.require(
            isinstance(args.payoff, PlainVanillaPayoff),
            "non-plain payoff given",
        )
        assert isinstance(args.payoff, PlainVanillaPayoff)
        payoff: PlainVanillaPayoff = args.payoff
        exercise: Exercise = args.exercise

        process = self._process
        rfdc = process.risk_free_rate().day_counter()
        s0 = process.state_variable().value()
        qassert.require(s0 > 0.0, "negative or null underlying given")
        maturity_date = exercise.last_date()
        ref_date = process.risk_free_rate().reference_date()
        maturity = rfdc.year_fraction(ref_date, maturity_date)

        # Constant-coefficient flatten: pull the zero rates / vol at
        # the maturity, matching the C++ ``FlatForward`` /
        # ``BlackConstantVol`` reconstruction.
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
            result_day_counter=process.dividend_yield().day_counter(),
        ).rate()
        v = process.black_volatility().black_vol(maturity_date, s0, extrapolate=True)

        # Tree-builder dispatch.
        n_steps, up, down, pu, pd = self._build_tree_coefficients(
            n_steps=self._time_steps,
            x0=s0,
            r=r,
            q=q,
            sigma=v,
            maturity=maturity,
            strike=payoff.strike(),
        )

        dt = maturity / n_steps
        discount = math.exp(-r * dt)

        # Build the terminal underlying prices: S_T at index j is
        # x0 * up^j * down^(n-j).  All trees use a forward-centered
        # exponential parameterisation so this works uniformly.
        prices: npt.NDArray[np.float64] = self._terminal_prices(
            s0=s0, n=n_steps, up=up, down=down
        )

        # Terminal option values.
        values: npt.NDArray[np.float64] = np.array(
            [payoff(p) for p in prices], dtype=np.float64
        )

        # Determine exercise dates by year fraction (for
        # American/Bermudan support). For European exercise the only
        # exercise date is at maturity (no interior check needed).
        exercise_type = exercise.type()

        # Walk back. For multi-Greek extraction we capture values at
        # step 2 (3 nodes) and step 1 (2 nodes) on the way down.
        v_step2: npt.NDArray[np.float64] | None = None
        s_step2: npt.NDArray[np.float64] | None = None
        v_step1: npt.NDArray[np.float64] | None = None
        s_step1: npt.NDArray[np.float64] | None = None

        # Pre-compute exercise dates for Bermudan if needed.
        bermudan_times: list[float] | None = None
        if exercise_type == Exercise.Type.Bermudan:
            bermudan_times = [
                rfdc.year_fraction(ref_date, d) for d in exercise.dates()
            ]

        for step in range(n_steps - 1, -1, -1):
            # Continuation values.
            new_values = discount * (pu * values[1 : step + 2] + pd * values[0 : step + 1])
            new_prices = prices[0 : step + 1] / down  # walk prices back one step

            # American exercise: max(continuation, immediate exercise).
            if exercise_type == Exercise.Type.American:
                immediate = np.array([payoff(p) for p in new_prices], dtype=np.float64)
                new_values = np.maximum(new_values, immediate)
            elif exercise_type == Exercise.Type.Bermudan:
                # Exercise on listed dates only.
                t_at_step = step * dt
                assert bermudan_times is not None
                # Match the nearest exercise time. With deterministic
                # times this is robust enough; tighten tolerance if a
                # test ever fails.
                exercisable = any(
                    abs(bt - t_at_step) < dt * 0.5 for bt in bermudan_times
                )
                if exercisable:
                    immediate = np.array([payoff(p) for p in new_prices], dtype=np.float64)
                    new_values = np.maximum(new_values, immediate)

            values = new_values
            prices = new_prices

            # Cache the values + underlyings at step 2 and step 1 for
            # Hull-style Greek extraction.
            if step == 2:
                v_step2 = values.copy()
                s_step2 = prices.copy()
            elif step == 1:
                v_step1 = values.copy()
                s_step1 = prices.copy()

        results.value = float(values[0])

        # Greeks via Hull pattern. Available only if we observed step 1
        # and step 2 (n_steps >= 2 guarantees this).
        if v_step2 is not None and s_step2 is not None and len(v_step2) >= 3:
            p2u = float(v_step2[2])
            p2m = float(v_step2[1])
            p2d = float(v_step2[0])
            s2u = float(s_step2[2])
            s2m = float(s_step2[1])
            s2d = float(s_step2[0])
            delta2u = (p2u - p2m) / (s2u - s2m)
            delta2d = (p2m - p2d) / (s2m - s2d)
            results.gamma = (delta2u - delta2d) / ((s2u - s2d) / 2.0)

        if v_step1 is not None and s_step1 is not None and len(v_step1) >= 2:
            p1u = float(v_step1[1])
            p1d = float(v_step1[0])
            s1u = float(s_step1[1])
            s1d = float(s_step1[0])
            results.delta = (p1u - p1d) / (s1u - s1d)

        # Theta via Hull's BS theta from value + delta + gamma. Match
        # the C++ ``blackScholesTheta`` helper exactly.
        if results.delta is not None and results.gamma is not None:
            results.theta = self._black_scholes_theta(
                value=results.value,
                delta=results.delta,
                gamma=results.gamma,
                s0=s0,
                r=r,
                q=q,
                sigma=v,
            )

    # --- tree-builder dispatch ------------------------------------------

    def _build_tree_coefficients(
        self,
        *,
        n_steps: int,
        x0: float,
        r: float,
        q: float,
        sigma: float,
        maturity: float,
        strike: float,
    ) -> tuple[int, float, float, float, float]:
        """Return (n_steps_eff, up, down, pu, pd) for the chosen builder.

        # C++ parity: the four constructors in ``binomialtree.cpp`` —
        # each one sets ``up_``, ``down_``, ``pu_``, ``pd_`` (and may
        # adjust ``oddSteps`` for LeisenReimer / Joshi4).
        """
        dt = maturity / n_steps
        drift_per_step = (r - q - 0.5 * sigma * sigma) * dt
        variance_per_step = sigma * sigma * dt
        sqrt_variance_per_step = math.sqrt(variance_per_step)

        if self._tree_builder == TreeBuilder.CoxRossRubinstein:
            # Equal-jumps: dx = sigma * sqrt(dt); pu = 0.5 + 0.5*drift/dx.
            dx = sqrt_variance_per_step
            pu = 0.5 + 0.5 * drift_per_step / dx
            pd = 1.0 - pu
            qassert.require(0.0 <= pu <= 1.0, "negative probability")
            up = math.exp(dx)
            down = math.exp(-dx)
            return n_steps, up, down, pu, pd

        if self._tree_builder == TreeBuilder.JarrowRudd:
            # Equal-probabilities (pu = pd = 0.5); up = exp(sigma * sqrt(dt))
            # in a drift-adjusted forward-centered tree. C++ sets up_ =
            # process.stdDeviation(0, x0, dt) which is sigma*sqrt(dt);
            # the tree underlying is x0 * exp(i*driftPerStep + j*up_) so
            # the multiplicative-up factor is exp(drift + sigma*sqrt(dt))
            # — i.e. up * exp(drift), down * exp(drift) when expressed
            # purely multiplicatively for our terminal-price formula.
            log_up = drift_per_step + sqrt_variance_per_step
            log_down = drift_per_step - sqrt_variance_per_step
            up = math.exp(log_up)
            down = math.exp(log_down)
            return n_steps, up, down, 0.5, 0.5

        if self._tree_builder == TreeBuilder.Tian:
            # C++ Tian:
            # q_ = exp(variance_per_step), r_ = exp(drift_per_step)*sqrt(q_),
            # up_  = 0.5*r*q*(q+1 + sqrt(q^2+2q-3)),
            # down_= 0.5*r*q*(q+1 - sqrt(q^2+2q-3)).
            qq = math.exp(variance_per_step)
            rr = math.exp(drift_per_step) * math.sqrt(qq)
            disc = math.sqrt(qq * qq + 2 * qq - 3)
            up = 0.5 * rr * qq * (qq + 1.0 + disc)
            down = 0.5 * rr * qq * (qq + 1.0 - disc)
            pu = (rr - down) / (up - down)
            pd = 1.0 - pu
            qassert.require(0.0 <= pu <= 1.0, "negative probability")
            return n_steps, up, down, pu, pd

        if self._tree_builder == TreeBuilder.LeisenReimer:
            # C++ forces oddSteps. dt_per_oddstep = end / oddSteps.
            odd = n_steps if n_steps % 2 != 0 else n_steps + 1
            variance = sigma * sigma * maturity  # variance over full T
            drift_full = (r - q - 0.5 * sigma * sigma) * maturity
            drift_per_odd_step = drift_full / odd
            ermqdt = math.exp(drift_per_odd_step + 0.5 * variance / odd)
            d2 = (math.log(x0 / strike) + drift_per_odd_step * odd) / math.sqrt(variance)
            pu = _peizer_pratt_method2_inversion(d2, odd)
            pd = 1.0 - pu
            pdash = _peizer_pratt_method2_inversion(d2 + math.sqrt(variance), odd)
            up = ermqdt * pdash / pu
            down = (ermqdt - pu * up) / (1.0 - pu)
            return odd, up, down, pu, pd

        qassert.require(False, f"unknown tree builder: {self._tree_builder}")
        msg = f"unreachable tree builder: {self._tree_builder}"
        raise RuntimeError(msg)

    @staticmethod
    def _terminal_prices(
        *,
        s0: float,
        n: int,
        up: float,
        down: float,
    ) -> npt.NDArray[np.float64]:
        """Terminal underlying-price vector S_T at each leaf node.

        Index 0 = all-down, index n = all-up. Matches the C++
        ``BinomialTree<T>::underlying(n, j)`` convention.
        """
        # log-form to avoid catastrophic round-off for very large n.
        ln_up = math.log(up)
        ln_down = math.log(down)
        ln_s0 = math.log(s0)
        return np.exp(
            np.array(
                [ln_s0 + j * ln_up + (n - j) * ln_down for j in range(n + 1)],
                dtype=np.float64,
            )
        )

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
