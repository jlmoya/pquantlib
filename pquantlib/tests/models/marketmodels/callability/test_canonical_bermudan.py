"""THE CANONICAL BERMUDAN-SWAPTION callability test (W11-C).

Reproduces the C++ ``marketmodel.cpp`` ``testCallableSwapLS``: a callable
receiver swap and the equivalent Bermudan swaption are built from a
Longstaff-Schwartz exercise strategy (calibrated via ``collect_node_data`` +
``generic_longstaff_schwartz_regression`` over MC paths using a
``SwapBasisSystem``), then all four products (payer swap, receiver swap,
Bermudan swaption, callable receiver) are evolved together through an
``AccountingEngine`` over a ``LogNormalFwdRatePc`` evolver.

The C++ ``checkCallableSwap`` assertions are pure structural identities, so the
test is self-validating (no external reference needed):

  * payer swap + receiver swap == 0  (exact negatives; tier TIGHT)
  * bermudan swaption value >= 0
  * callable receiver >= plain receiver
  * receiver + bermudan == callable     (tier TIGHT)

This single test is the payoff validation for the ENTIRE W11-C callability
stack: ``SwapBasisSystem``, ``NothingExerciseValue``, ``collect_node_data``,
``generic_longstaff_schwartz_regression``, ``LongstaffSchwartzExerciseStrategy``
and ``CallSpecifiedMultiProduct`` (W11-A) all exercised end-to-end.

C++ parity:
  test-suite/marketmodel.cpp testCallableSwapLS + checkCallableSwap + simulate
  @ v1.42.1 (099987f0).
"""

from __future__ import annotations

import pytest

from pquantlib.math.statistics.incremental_statistics import IncrementalStatistics
from pquantlib.models.marketmodels.accounting_engine import AccountingEngine
from pquantlib.models.marketmodels.browniangenerators.mt_brownian_generator import (
    MTBrownianGeneratorFactory,
)
from pquantlib.models.marketmodels.callability import (
    LongstaffSchwartzExerciseStrategy,
    NothingExerciseValue,
    SwapBasisSystem,
    SwapRateTrigger,
    collect_node_data,
    generic_longstaff_schwartz_regression,
)
from pquantlib.models.marketmodels.correlations.exp_correlations import (
    ExponentialForwardCorrelation,
)
from pquantlib.models.marketmodels.evolution_description import EvolutionDescription
from pquantlib.models.marketmodels.evolvers.lognormal_fwd_rate_pc import (
    LogNormalFwdRatePc,
)
from pquantlib.models.marketmodels.models.flat_vol import FlatVol
from pquantlib.models.marketmodels.products import (
    CallSpecifiedMultiProduct,
    ExerciseAdapter,
    MultiProductComposite,
    MultiStepNothing,
    MultiStepSwap,
)
from pquantlib.testing.tolerance import tight

_SEED = 42
_TRAINING_PATHS = 4095  # 2^12 - 1 LS-calibration paths
_PRICING_PATHS = 8191  # 2^13 - 1 pricing paths
_LONG_TERM_CORR = 0.5
_BETA = 0.2
_FIXED_RATE = 0.04


class _SeqStats:
    """Per-dimension IncrementalStatistics (SequenceStatisticsInc shim)."""

    def __init__(self, dimension: int) -> None:
        self._stats = [IncrementalStatistics() for _ in range(dimension)]

    def add(self, values: list[float], weight: float = 1.0) -> None:
        for i, v in enumerate(values):
            self._stats[i].add(v, weight)

    def mean(self) -> list[float]:
        return [s.mean() for s in self._stats]

    def error_estimate(self) -> list[float]:
        return [s.error_estimate() for s in self._stats]


def _make_world() -> tuple[list[float], list[float], list[float], list[float]]:
    """Semiannual rate-time grid to 3y; flat 0.05 forwards; flat 0.20 vols."""
    n = 5  # 5 forward rates → rateTimes = {0.5, 1.0, ..., 3.0}
    rate_times = [0.5 * (i + 1) for i in range(n + 1)]
    accruals = [rate_times[i + 1] - rate_times[i] for i in range(n)]
    payment_times = rate_times[1:]
    forwards = [0.05] * n
    return rate_times, accruals, payment_times, forwards


def _build_evolver(
    rate_times: list[float],
    forwards: list[float],
    evolution: EvolutionDescription,
    seed: int,
) -> LogNormalFwdRatePc:
    n = len(forwards)
    volatilities = [0.20] * n
    displacements = [0.0] * n
    correlations = ExponentialForwardCorrelation(rate_times, _LONG_TERM_CORR, _BETA)
    market_model = FlatVol(
        volatilities, correlations, evolution, n, forwards, displacements
    )
    numeraires = [len(rate_times) - 1] * evolution.number_of_steps()
    factory = MTBrownianGeneratorFactory(seed)
    return LogNormalFwdRatePc(market_model, factory, numeraires, 0)


@pytest.mark.slow
def test_canonical_callable_swap_ls() -> None:
    rate_times, accruals, payment_times, forwards = _make_world()

    # 0. payer swap; 1. receiver swap (the exact negative).
    payer_swap = MultiStepSwap(
        rate_times, accruals, accruals, payment_times, _FIXED_RATE, payer=True
    )
    receiver_swap = MultiStepSwap(
        rate_times, accruals, accruals, payment_times, _FIXED_RATE, payer=False
    )

    # exercise schedule = all rate times but the last.
    exercise_times = rate_times[:-1]

    # naive strategy (only used to build the dummy product's evolution).
    swap_triggers = [_FIXED_RATE] * len(exercise_times)
    naif_strategy = SwapRateTrigger(rate_times, swap_triggers, exercise_times)

    null_rebate = NothingExerciseValue(rate_times)
    control = NothingExerciseValue(rate_times)
    basis_system = SwapBasisSystem(rate_times, exercise_times)

    dummy_product = CallSpecifiedMultiProduct(
        receiver_swap, naif_strategy, ExerciseAdapter(null_rebate)
    )
    evolution = dummy_product.evolution()
    numeraires = [len(rate_times) - 1] * evolution.number_of_steps()

    # --- calibrate the Longstaff-Schwartz exercise strategy ------------------
    train_evolver = _build_evolver(rate_times, forwards, evolution, _SEED)
    collected = collect_node_data(
        train_evolver,
        receiver_swap,
        basis_system,
        null_rebate,
        control,
        _TRAINING_PATHS,
    )
    basis_coefficients, lower_bound_estimate = (
        generic_longstaff_schwartz_regression(collected)
    )
    exercise_strategy = LongstaffSchwartzExerciseStrategy(
        basis_system,
        basis_coefficients,
        evolution,
        numeraires,
        null_rebate,
        control,
    )

    # 2. bermudan swaption to enter the payer swap.
    bermudan_product = CallSpecifiedMultiProduct(
        MultiStepNothing(evolution), exercise_strategy, payer_swap
    )
    # 3. callable receiver swap.
    callable_product = CallSpecifiedMultiProduct(
        receiver_swap, exercise_strategy, ExerciseAdapter(null_rebate)
    )

    # --- lower bound: evolve all 4 products together -------------------------
    all_products = MultiProductComposite()
    all_products.add(payer_swap)
    all_products.add(receiver_swap)
    all_products.add(bermudan_product)
    all_products.add(callable_product)
    all_products.finalize()

    price_evolver = _build_evolver(rate_times, forwards, evolution, _SEED + 1)
    # flat 0.05 forwards under the terminal measure → numeraire-0 value 1.0.
    initial_numeraire_value = 1.0
    engine = AccountingEngine(
        price_evolver, all_products, initial_numeraire_value
    )
    stats = _SeqStats(all_products.number_of_products())
    engine.multiple_path_values(stats, _PRICING_PATHS)

    means = stats.mean()
    payer_npv, receiver_npv, bermudan_npv, callable_npv = means

    # --- structural identities (C++ checkCallableSwap) -----------------------
    # 1) payer + receiver == 0 (exact negatives, identical paths).
    tight(
        payer_npv + receiver_npv,
        0.0,
        reason="payer/receiver swaps are exact negatives on identical paths",
    )
    # 2) bermudan >= 0.
    assert bermudan_npv >= -1e-12, f"negative bermudan value: {bermudan_npv}"
    # 3) callable receiver >= plain receiver.
    assert callable_npv >= receiver_npv - 1e-12, (
        f"callable ({callable_npv}) < receiver ({receiver_npv})"
    )
    # 4) receiver + bermudan == callable.
    tight(
        receiver_npv + bermudan_npv,
        callable_npv,
        reason="receiver + bermudan must equal the callable receiver (identity)",
    )

    # the in-sample LS lower-bound estimate must be a sensible non-negative
    # number (it is the biased Bermudan value used to calibrate the strategy).
    assert lower_bound_estimate >= -1e-10
