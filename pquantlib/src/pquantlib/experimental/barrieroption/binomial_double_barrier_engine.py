"""BinomialDoubleBarrierEngine — binomial-tree pricing for double barriers.

# C++ parity: ql/experimental/barrieroption/binomialdoublebarrierengine.hpp
# + ql/experimental/barrieroption/discretizeddoublebarrieroption.{hpp,cpp}
# @ v1.42.1.

C++ uses a template parameterised by:
  * the tree-builder T (CRR / JarrowRudd / Tian / LeisenReimer)
  * the discretization D (DiscretizedDoubleBarrierOption or
    DiscretizedDermanKaniDoubleBarrierOption)

The Python port collapses tree-builder dispatch into the existing
``TreeBuilder`` IntEnum from ``BinomialVanillaEngine`` and inlines the
discretization barrier-check directly into the rollback loop, mirroring
the same algorithmic shortcut already used by ``BinomialVanillaEngine``
(direct numpy-array walk instead of the C++
``DiscretizedAsset``/``Lattice`` machinery).

For KI variants the rollback needs to track a *vanilla* parallel array
that prices the same option without the barrier, so we can apply the
``payoff_at_knock_in`` formula at touched nodes — this matches the C++
``DiscretizedDoubleBarrierOption::checkBarrier`` ``vanilla()`` access.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

from pquantlib import qassert
from pquantlib.exceptions import LibraryException
from pquantlib.exercise import Exercise
from pquantlib.instruments.double_barrier_option import (
    DoubleBarrierOptionArguments,
    DoubleBarrierType,
)
from pquantlib.instruments.one_asset_option import OneAssetOptionResults
from pquantlib.math.distributions.binomial_distribution import (
    peizer_pratt_method2_inversion as _peizer_pratt_method2_inversion,
)
from pquantlib.payoffs import StrikedTypePayoff
from pquantlib.pricingengines.generic_engine import GenericEngine
from pquantlib.pricingengines.vanilla.binomial_engine import TreeBuilder
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)
from pquantlib.time.compounding import Compounding
from pquantlib.time.frequency import Frequency


class BinomialDoubleBarrierEngine(
    GenericEngine[DoubleBarrierOptionArguments, OneAssetOptionResults]
):
    """Binomial-tree double-barrier engine.

    # C++ parity: ``template <class T, class D=DiscretizedDoubleBarrierOption>
    # class BinomialDoubleBarrierEngine``. The Python port supports the
    # standard (non-Derman-Kani) variant only; Derman-Kani is deferred
    # because it requires the full DiscretizedAsset/Lattice machinery to
    # do the interpolated barrier correction (a single-step gradient
    # adjustment near the barrier).
    """

    def __init__(
        self,
        process: GeneralizedBlackScholesProcess,
        time_steps: int,
        tree_builder: TreeBuilder = TreeBuilder.CoxRossRubinstein,
    ) -> None:
        super().__init__(
            DoubleBarrierOptionArguments(), OneAssetOptionResults()
        )
        qassert.require(
            time_steps > 0,
            f"timeSteps must be positive, {time_steps} not allowed",
        )
        self._process: GeneralizedBlackScholesProcess = process
        self._time_steps: int = time_steps
        self._tree_builder: TreeBuilder = tree_builder
        process.register_with(self)

    def calculate(self) -> None:  # noqa: PLR0915
        """# C++ parity: ``BinomialDoubleBarrierEngine<T, D>::calculate``.

        Direct numpy walk; sets value, delta, gamma, theta from the
        three-step Hull lookahead (same pattern as BinomialVanillaEngine
        but with per-step double-barrier KO/KI/KIKO/KOKI checks).
        """
        args = self._arguments
        results = self._results

        qassert.require(args.exercise is not None, "no exercise given")
        qassert.require(args.payoff is not None, "no payoff given")
        qassert.require(args.barrier_type is not None, "no barrier type given")
        qassert.require(args.barrier_lo is not None, "no low barrier given")
        qassert.require(args.barrier_hi is not None, "no high barrier given")
        qassert.require(args.rebate is not None, "no rebate given")
        assert args.exercise is not None
        assert args.payoff is not None
        assert args.barrier_type is not None
        assert args.barrier_lo is not None
        assert args.barrier_hi is not None
        assert args.rebate is not None

        payoff = args.payoff
        qassert.require(
            isinstance(payoff, StrikedTypePayoff), "non-striked payoff given"
        )
        assert isinstance(payoff, StrikedTypePayoff)
        exercise: Exercise = args.exercise
        barrier_type: DoubleBarrierType = args.barrier_type
        barrier_lo: float = args.barrier_lo
        barrier_hi: float = args.barrier_hi
        rebate: float = args.rebate

        process = self._process
        rfdc = process.risk_free_rate().day_counter()
        s0 = process.state_variable().value()
        qassert.require(s0 > 0.0, "negative or null underlying given")
        maturity_date = exercise.last_date()
        ref_date = process.risk_free_rate().reference_date()
        maturity = rfdc.year_fraction(ref_date, maturity_date)

        # Constant-coefficient flatten matching the C++ engine
        # (.hpp:90-98): FlatForward(r), FlatForward(q), BlackConstantVol(v).
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
        v = process.black_volatility().black_vol(
            maturity_date, payoff.strike(), extrapolate=True
        )

        # Tree-builder dispatch — same logic as BinomialVanillaEngine.
        n_steps, up, down, pu, pd = _build_tree_coefficients(
            tree_builder=self._tree_builder,
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

        # Terminal underlyings and payoffs.
        prices: npt.NDArray[np.float64] = _terminal_prices(
            s0=s0, n=n_steps, up=up, down=down
        )
        values: npt.NDArray[np.float64] = np.array(
            [payoff(p) for p in prices], dtype=np.float64
        )
        # Apply the *initial* barrier adjustment at maturity (this is
        # the equivalent of DiscretizedDoubleBarrierOption.adjustValues()
        # being called from reset() at the leaves).
        _apply_barrier_at_leaves(
            values,
            prices,
            barrier_type=barrier_type,
            barrier_lo=barrier_lo,
            barrier_hi=barrier_hi,
            rebate=rebate,
        )

        # For KI / KIKO / KOKI we also need a vanilla parallel array
        # holding the un-barriered option value at each step. C++ does
        # this via the contained ``DiscretizedVanillaOption`` member.
        needs_vanilla = barrier_type in (
            DoubleBarrierType.KnockIn,
            DoubleBarrierType.KIKO,
            DoubleBarrierType.KOKI,
        )
        vanilla = (
            np.array([payoff(p) for p in prices], dtype=np.float64)
            if needs_vanilla
            else None
        )

        # Walk back. We capture step 2 / step 1 values+underlyings for
        # Hull-style Greek extraction (same as BinomialVanillaEngine).
        exercise_type = exercise.type()

        v_step2: npt.NDArray[np.float64] | None = None
        s_step2: npt.NDArray[np.float64] | None = None
        v_step1: npt.NDArray[np.float64] | None = None
        s_step1: npt.NDArray[np.float64] | None = None

        for step in range(n_steps - 1, -1, -1):
            # Continuation values.
            new_values = discount * (
                pu * values[1 : step + 2] + pd * values[0 : step + 1]
            )
            new_prices = prices[0 : step + 1] / down  # walk underlyings back

            # The C++ engine also rolls back the *vanilla* array before
            # the barrier check for KI / KIKO / KOKI; do the same.
            new_vanilla: npt.NDArray[np.float64] | None = None
            if vanilla is not None:
                new_vanilla = discount * (
                    pu * vanilla[1 : step + 2] + pd * vanilla[0 : step + 1]
                )

            # American exercise check (max with immediate exercise).
            # For European we only apply the payoff at the leaves (already
            # done) — interior steps stay continuation-only. C++ checks
            # ``isOnTime(stoppingTimes_[0])`` at the European leaf which
            # is precisely the maturity step.
            is_stopping_time = False
            if exercise_type == Exercise.Type.American:
                is_stopping_time = True
            elif exercise_type == Exercise.Type.Bermudan:
                t_at_step = step * dt
                exercise_times = [
                    rfdc.year_fraction(ref_date, d) for d in exercise.dates()
                ]
                is_stopping_time = any(
                    abs(et - t_at_step) < dt * 0.5 for et in exercise_times
                )

            # Apply the double-barrier check at every step (the barrier
            # is monitored continuously, modelled as monitored at every
            # tree step — discretization error decreases as n grows).
            # ``end_time`` is the C++ ``isOnTime(stoppingTimes_.back())``
            # flag — true only at the maturity step. We applied the
            # maturity-step rule already in ``_apply_barrier_at_leaves``
            # outside this loop, so ``end_time`` is always False here.
            _apply_barrier_step(
                new_values,
                new_prices,
                vanilla_values=new_vanilla,
                barrier_type=barrier_type,
                barrier_lo=barrier_lo,
                barrier_hi=barrier_hi,
                rebate=rebate,
                is_stopping_time=is_stopping_time,
                end_time=False,
                payoff=payoff,
            )

            values = new_values
            prices = new_prices
            if new_vanilla is not None:
                vanilla = new_vanilla

            # Cache the values + prices at step 2 and step 1 for Greek
            # extraction (matches C++ rollback to grid[2] and grid[1]).
            if step == 2:
                v_step2 = values.copy()
                s_step2 = prices.copy()
            elif step == 1:
                v_step1 = values.copy()
                s_step1 = prices.copy()

        results.value = float(values[0])

        # Greeks via Hull pattern. See BinomialVanillaEngine for the
        # derivation. The C++ engine sets value/delta/gamma/theta only.
        if (
            v_step2 is not None
            and s_step2 is not None
            and len(v_step2) >= 3
        ):
            p2u = float(v_step2[2])
            p2m = float(v_step2[1])
            p2d = float(v_step2[0])
            s2u = float(s_step2[2])
            s2m = float(s_step2[1])
            s2d = float(s_step2[0])
            delta2u = (p2u - p2m) / (s2u - s2m)
            delta2d = (p2m - p2d) / (s2m - s2d)
            results.gamma = (delta2u - delta2d) / ((s2u - s2d) / 2.0)
            # C++ theta: (p2m - p0) / grid[2] where grid[2] = 2*dt.
            results.theta = (p2m - results.value) / (2.0 * dt)

        if (
            v_step1 is not None
            and s_step1 is not None
            and len(v_step1) >= 2
        ):
            p1u = float(v_step1[1])
            p1d = float(v_step1[0])
            s1u = float(s_step1[1])
            s1d = float(s_step1[0])
            results.delta = (p1u - p1d) / (s1u - s1d)


# --------------------------------------------------------------------------
# Barrier-check helpers — port of
# DiscretizedDoubleBarrierOption::checkBarrier (discretizeddoublebarrieroption.cpp).
# --------------------------------------------------------------------------


def _apply_barrier_at_leaves(
    values: npt.NDArray[np.float64],
    prices: npt.NDArray[np.float64],
    *,
    barrier_type: DoubleBarrierType,
    barrier_lo: float,
    barrier_hi: float,
    rebate: float,
) -> None:
    """Apply KO/KI/KIKO/KOKI at terminal step.

    # C++ parity: the ``endTime`` branches of ``checkBarrier``
    # (discretizeddoublebarrieroption.cpp:57-152) applied at the
    # maturity step.
    """
    for j in range(len(values)):
        sj = prices[j]
        if barrier_type == DoubleBarrierType.KnockOut:
            if sj <= barrier_lo or sj >= barrier_hi:
                values[j] = rebate
        elif barrier_type == DoubleBarrierType.KnockIn:
            if barrier_lo < sj < barrier_hi:
                values[j] = rebate  # never knocked in -> pay rebate
        elif barrier_type == DoubleBarrierType.KIKO:
            if sj >= barrier_hi:
                values[j] = rebate  # knocked out hi
            elif sj > barrier_lo:
                values[j] = rebate  # not knocked in
        elif sj <= barrier_lo or sj < barrier_hi:
            values[j] = rebate


def _apply_barrier_step(
    values: npt.NDArray[np.float64],
    prices: npt.NDArray[np.float64],
    *,
    vanilla_values: npt.NDArray[np.float64] | None,
    barrier_type: DoubleBarrierType,
    barrier_lo: float,
    barrier_hi: float,
    rebate: float,
    is_stopping_time: bool,
    end_time: bool,
    payoff: StrikedTypePayoff,
) -> None:
    """Barrier check at an interior or final rollback step.

    # C++ parity: ``DiscretizedDoubleBarrierOption::checkBarrier``
    # (discretizeddoublebarrieroption.cpp:57-152).
    """
    for j in range(len(values)):
        sj = prices[j]
        if barrier_type == DoubleBarrierType.KnockIn:
            assert vanilla_values is not None
            if sj <= barrier_lo or sj >= barrier_hi:
                if is_stopping_time:
                    values[j] = max(vanilla_values[j], payoff(sj))
                else:
                    values[j] = vanilla_values[j]
            elif end_time:
                values[j] = rebate
        elif barrier_type == DoubleBarrierType.KnockOut:
            if sj <= barrier_lo or sj >= barrier_hi:
                values[j] = rebate
            elif is_stopping_time:
                values[j] = max(values[j], payoff(sj))
        elif barrier_type == DoubleBarrierType.KIKO:
            # lo barrier is KI, hi is KO.
            assert vanilla_values is not None
            if sj <= barrier_lo:
                if is_stopping_time:
                    values[j] = max(vanilla_values[j], payoff(sj))
                else:
                    values[j] = vanilla_values[j]
            elif sj >= barrier_hi or end_time:
                values[j] = rebate
        elif barrier_type == DoubleBarrierType.KOKI:
            # lo barrier is KO, hi is KI.
            assert vanilla_values is not None
            if sj <= barrier_lo:
                values[j] = rebate
            elif sj >= barrier_hi:
                if is_stopping_time:
                    values[j] = max(vanilla_values[j], payoff(sj))
                else:
                    values[j] = vanilla_values[j]
            elif end_time:
                values[j] = rebate
        else:
            raise LibraryException(f"invalid barrier type: {barrier_type}")


# --------------------------------------------------------------------------
# Tree coefficient + terminal price helpers (duplicated from
# BinomialVanillaEngine intentionally to keep the engine self-contained;
# refactor into a shared helper module if other engines need them).
# --------------------------------------------------------------------------


def _build_tree_coefficients(
    *,
    tree_builder: TreeBuilder,
    n_steps: int,
    x0: float,
    r: float,
    q: float,
    sigma: float,
    maturity: float,
    strike: float,
) -> tuple[int, float, float, float, float]:
    """# C++ parity: ``binomialtree.cpp`` four ctors.

    See ``BinomialVanillaEngine._build_tree_coefficients`` for the full
    derivation per tree builder. Inlined here to avoid coupling the
    double-barrier engine to the private helper.
    """
    _ppmi = _peizer_pratt_method2_inversion
    dt = maturity / n_steps
    drift_per_step = (r - q - 0.5 * sigma * sigma) * dt
    variance_per_step = sigma * sigma * dt
    sqrt_variance_per_step = math.sqrt(variance_per_step)

    if tree_builder == TreeBuilder.CoxRossRubinstein:
        dx = sqrt_variance_per_step
        pu = 0.5 + 0.5 * drift_per_step / dx
        pd = 1.0 - pu
        qassert.require(0.0 <= pu <= 1.0, "negative probability")
        up = math.exp(dx)
        down = math.exp(-dx)
        return n_steps, up, down, pu, pd

    if tree_builder == TreeBuilder.JarrowRudd:
        log_up = drift_per_step + sqrt_variance_per_step
        log_down = drift_per_step - sqrt_variance_per_step
        return n_steps, math.exp(log_up), math.exp(log_down), 0.5, 0.5

    if tree_builder == TreeBuilder.Tian:
        qq = math.exp(variance_per_step)
        rr = math.exp(drift_per_step) * math.sqrt(qq)
        disc = math.sqrt(qq * qq + 2 * qq - 3)
        up = 0.5 * rr * qq * (qq + 1.0 + disc)
        down = 0.5 * rr * qq * (qq + 1.0 - disc)
        pu = (rr - down) / (up - down)
        pd = 1.0 - pu
        qassert.require(0.0 <= pu <= 1.0, "negative probability")
        return n_steps, up, down, pu, pd

    if tree_builder == TreeBuilder.LeisenReimer:
        odd = n_steps if n_steps % 2 != 0 else n_steps + 1
        variance = sigma * sigma * maturity
        drift_full = (r - q - 0.5 * sigma * sigma) * maturity
        drift_per_odd_step = drift_full / odd
        ermqdt = math.exp(drift_per_odd_step + 0.5 * variance / odd)
        d2 = (
            math.log(x0 / strike) + drift_per_odd_step * odd
        ) / math.sqrt(variance)
        pu = _ppmi(d2, odd)
        pd = 1.0 - pu
        pdash = _ppmi(d2 + math.sqrt(variance), odd)
        up = ermqdt * pdash / pu
        down = (ermqdt - pu * up) / (1.0 - pu)
        return odd, up, down, pu, pd

    raise LibraryException(f"unknown tree builder: {tree_builder}")


def _terminal_prices(
    *, s0: float, n: int, up: float, down: float
) -> npt.NDArray[np.float64]:
    """Terminal underlying price vector.

    # C++ parity: ``BinomialTree<T>::underlying(n, j)``.
    """
    ln_up = math.log(up)
    ln_down = math.log(down)
    ln_s0 = math.log(s0)
    return np.exp(
        np.array(
            [ln_s0 + j * ln_up + (n - j) * ln_down for j in range(n + 1)],
            dtype=np.float64,
        )
    )


__all__ = ["BinomialDoubleBarrierEngine"]
