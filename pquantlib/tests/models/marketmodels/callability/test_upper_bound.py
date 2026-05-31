"""W11-C UpperBoundEngine — Andersen-Broadie dual upper bound (MC).

Reproduces the upper-bound leg of the C++ ``marketmodel.cpp``
``testCallableSwapLS``: after calibrating the Longstaff-Schwartz exercise
strategy, an Andersen-Broadie ``UpperBoundEngine`` (with one inner evolver per
exercise step) estimates the additive duality gap ``delta`` on top of the LS
lower bound. The dual upper bound is ``lower_bound + delta``.

``delta`` is the expected running maximum of the dual martingale minus the
already-collected cash flows; it is non-negative up to Monte-Carlo noise, so the
upper bound is ``>=`` the LS lower-bound price (tier LOOSE — MC).

C++ parity:
  test-suite/marketmodel.cpp testCallableSwapLS (upper-bound block) +
    UpperBoundEngine::singlePathValue @ v1.42.1 (099987f0).
"""

from __future__ import annotations

import pytest

from pquantlib.models.marketmodels.browniangenerators.mt_brownian_generator import (
    MTBrownianGeneratorFactory,
)
from pquantlib.models.marketmodels.callability import (
    LongstaffSchwartzExerciseStrategy,
    NothingExerciseValue,
    SwapBasisSystem,
    SwapRateTrigger,
    UpperBoundEngine,
    collect_node_data,
    generic_longstaff_schwartz_regression,
)
from pquantlib.models.marketmodels.correlations.exp_correlations import (
    ExponentialForwardCorrelation,
)
from pquantlib.models.marketmodels.evolvers.lognormal_fwd_rate_pc import (
    LogNormalFwdRatePc,
)
from pquantlib.models.marketmodels.models.flat_vol import FlatVol
from pquantlib.models.marketmodels.products import (
    CallSpecifiedMultiProduct,
    ExerciseAdapter,
    MultiStepSwap,
)
from pquantlib.models.marketmodels.utilities import is_in_subset

_N = 5
_SEED = 42
_FIXED_RATE = 0.04


class _Stats:
    def __init__(self) -> None:
        self._values: list[float] = []
        self._weights: list[float] = []

    def add(self, value: float, weight: float = 1.0) -> None:
        self._values.append(value)
        self._weights.append(weight)

    def mean(self) -> float:
        total_w = sum(self._weights)
        return sum(v * w for v, w in zip(self._values, self._weights, strict=True)) / (
            total_w
        )


def _world() -> tuple[list[float], list[float], list[float], list[float]]:
    rate_times = [0.5 * (i + 1) for i in range(_N + 1)]
    accruals = [rate_times[i + 1] - rate_times[i] for i in range(_N)]
    payment_times = rate_times[1:]
    forwards = [0.05] * _N
    return rate_times, accruals, payment_times, forwards


@pytest.mark.slow
def test_upper_bound_dominates_lower_bound() -> None:
    rate_times, accruals, payment_times, forwards = _world()
    receiver_swap = MultiStepSwap(
        rate_times, accruals, accruals, payment_times, _FIXED_RATE, payer=False
    )
    exercise_times = rate_times[:-1]
    naif = SwapRateTrigger(rate_times, [_FIXED_RATE] * len(exercise_times), exercise_times)
    null_rebate = NothingExerciseValue(rate_times)
    control = NothingExerciseValue(rate_times)
    basis = SwapBasisSystem(rate_times, exercise_times)
    dummy = CallSpecifiedMultiProduct(receiver_swap, naif, ExerciseAdapter(null_rebate))
    evolution = dummy.evolution()
    numeraires = [len(rate_times) - 1] * evolution.number_of_steps()

    def make_evolver(seed: int, initial_step: int = 0) -> LogNormalFwdRatePc:
        vols = [0.20] * _N
        disp = [0.0] * _N
        corr = ExponentialForwardCorrelation(rate_times, 0.5, 0.2)
        mm = FlatVol(vols, corr, evolution, _N, forwards, disp)
        fac = MTBrownianGeneratorFactory(seed)
        return LogNormalFwdRatePc(mm, fac, numeraires, initial_step)

    # calibrate the LS strategy.
    train = make_evolver(_SEED)
    collected = collect_node_data(
        train, receiver_swap, basis, null_rebate, control, 1023
    )
    coeffs, lower_bound = generic_longstaff_schwartz_regression(collected)
    strat = LongstaffSchwartzExerciseStrategy(
        basis, coeffs, evolution, numeraires, null_rebate, control
    )
    assert lower_bound >= -1e-10

    # one inner evolver per exercise step (started mid-path).
    is_ex = is_in_subset(evolution.evolution_times(), strat.exercise_times())
    inner = [make_evolver(_SEED + s, initial_step=s) for s in range(len(is_ex)) if is_ex[s]]
    assert len(inner) == sum(is_ex)

    ub_evolver = make_evolver(_SEED + 142)
    engine = UpperBoundEngine(
        ub_evolver, inner, receiver_swap, null_rebate, receiver_swap, null_rebate, strat, 1.0
    )
    stats = _Stats()
    engine.multiple_path_values(stats, outer_paths=16, inner_paths=24)
    delta = stats.mean()

    # the additive duality gap is non-negative up to MC noise; hence the dual
    # upper bound (lower_bound + delta) dominates the LS lower bound.
    assert delta >= -2.0e-4, f"duality gap markedly negative: {delta}"
    upper_bound = lower_bound + delta
    assert upper_bound >= lower_bound - 2.0e-4, (
        f"upper bound {upper_bound} below lower bound {lower_bound}"
    )
