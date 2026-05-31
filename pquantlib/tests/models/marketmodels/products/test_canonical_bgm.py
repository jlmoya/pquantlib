"""THE CANONICAL BGM END-TO-END TEST.

Reproduces the C++ ``marketmodel.cpp`` ``testMultiStepForwardsAndOptionlets``:
an ``AccountingEngine`` driving a ``LogNormalFwdRatePc`` evolver (over a
``FlatVol`` market model with ``ExponentialForwardCorrelation``, seeded
``MTBrownianGeneratorFactory``) over a ``MultiProductComposite`` of
``MultiStepForwards`` + ``MultiStepOptionlets`` reproduces the closed-form Black
FRA + caplet prices.

This single test is the payoff-validation for the ENTIRE W9 + W10 + W11
marketmodels stack: it exercises the product framework (composite + the two
canonical concrete products), the market model, the evolver, the Brownian
generator, the curve state, the discounter, and the accounting engine together.

The exact setup (rate times, accruals, forwards, discounts, per-rate flat
volatilities) and the analytic Black references are emitted by the C++ probe
``migration-harness/references/cluster/w11a.json`` (so they are bit-faithful to
the C++ ``setup()``). Reference values are closed-form (tier EXACT); the MC
means are checked against the C++ 2.5-standard-error band (LOOSE / MC) — the
deviation ``(mc_mean - expected) / standard_error`` must lie in ``[-2.5, 2.5]``,
exactly as the C++ test's ``errorThreshold = 2.50`` check.

C++ parity:
  test-suite/marketmodel.cpp testMultiStepForwardsAndOptionlets +
    checkForwardsAndOptionlets + simulate
  @ v1.42.1 (099987f0).
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.math.statistics.incremental_statistics import IncrementalStatistics
from pquantlib.models.marketmodels.accounting_engine import AccountingEngine
from pquantlib.models.marketmodels.browniangenerators.mt_brownian_generator import (
    MTBrownianGeneratorFactory,
)
from pquantlib.models.marketmodels.correlations.exp_correlations import (
    ExponentialForwardCorrelation,
)
from pquantlib.models.marketmodels.evolution_description import is_in_terminal_measure
from pquantlib.models.marketmodels.evolvers.lognormal_fwd_rate_pc import (
    LogNormalFwdRatePc,
)
from pquantlib.models.marketmodels.models.flat_vol import FlatVol
from pquantlib.models.marketmodels.products import (
    MultiProductComposite,
    MultiStepForwards,
    MultiStepOptionlets,
)
from pquantlib.payoffs import OptionType, Payoff, PlainVanillaPayoff
from pquantlib.testing.reference_reader import load as load_reference

# Monte-Carlo path count. The test is self-normalizing (it checks deviations in
# units of the MC standard error), so any count is valid; fewer paths just
# widen the bars. 16383 (2^14 - 1) keeps the band tight while running in ~a few
# seconds in pure Python (C++ uses 32767).
_PATHS = 16383
_SEED = 42
_ERROR_THRESHOLD = 2.50
# Long-term correlation / beta from the C++ setup (Cap/Floor Correlation block).
_LONG_TERM_CORR = 0.5
_BETA = 0.2


class _SeqStats:
    """Minimal SequenceStatisticsInc shim: per-dimension IncrementalStatistics."""

    def __init__(self, dimension: int) -> None:
        self._stats = [IncrementalStatistics() for _ in range(dimension)]

    def add(self, values: list[float], weight: float = 1.0) -> None:
        for i, v in enumerate(values):
            self._stats[i].add(v, weight)

    def mean(self) -> list[float]:
        return [s.mean() for s in self._stats]

    def error_estimate(self) -> list[float]:
        return [s.error_estimate() for s in self._stats]


@pytest.fixture
def ref() -> dict[str, Any]:
    return load_reference("cluster/w11a")


@pytest.mark.slow
def test_canonical_forwards_and_optionlets(ref: dict[str, Any]) -> None:
    rate_times: list[float] = list(ref["can_rate_times"])
    payment_times: list[float] = list(ref["can_payment_times"])
    accruals: list[float] = list(ref["can_accruals"])
    forwards: list[float] = list(ref["can_forwards"])
    volatilities: list[float] = list(ref["can_volatilities"])
    displacement = float(ref["can_displacement"])
    n = int(ref["can_n"])
    forward_strikes: list[float] = list(ref["can_forward_strikes"])
    expected_forwards: list[float] = list(ref["can_expected_forwards"])
    expected_caplets: list[float] = list(ref["can_expected_caplets"])
    discounts: list[float] = list(ref["can_discounts"])

    # --- build the composite product: forwards (0..N-1) then optionlets ------
    optionlet_payoffs: list[Payoff] = [
        PlainVanillaPayoff(OptionType.Call, forwards[i]) for i in range(n)
    ]
    forwards_product = MultiStepForwards(
        rate_times, accruals, payment_times, forward_strikes
    )
    optionlets_product = MultiStepOptionlets(
        rate_times, accruals, payment_times, optionlet_payoffs
    )
    product = MultiProductComposite()
    product.add(forwards_product)
    product.add(optionlets_product)
    product.finalize()

    evolution = product.evolution()
    assert product.number_of_products() == 2 * n

    # --- market model: full-factor FlatVol, exp-correlation ------------------
    number_of_factors = n
    displacements = [displacement] * n
    correlations = ExponentialForwardCorrelation(
        rate_times, _LONG_TERM_CORR, _BETA
    )
    market_model = FlatVol(
        volatilities,
        correlations,
        evolution,
        number_of_factors,
        forwards,
        displacements,
    )

    # --- evolver under the terminal measure ----------------------------------
    numeraires = [len(rate_times) - 1] * evolution.number_of_steps()
    assert is_in_terminal_measure(evolution, numeraires)
    factory = MTBrownianGeneratorFactory(_SEED)
    evolver = LogNormalFwdRatePc(market_model, factory, numeraires, 0)

    # --- run the accounting engine -------------------------------------------
    initial_numeraire = evolver.numeraires()[0]
    initial_numeraire_value = discounts[initial_numeraire]
    engine = AccountingEngine(evolver, product, initial_numeraire_value)
    stats = _SeqStats(product.number_of_products())
    engine.multiple_path_values(stats, _PATHS)

    results = stats.mean()
    errors = stats.error_estimate()

    # --- check: each MC mean within 2.5 standard errors of the Black value ---
    max_dev = 0.0
    for i in range(n):
        fwd_dev = (results[i] - expected_forwards[i]) / errors[i]
        cap_dev = (results[i + n] - expected_caplets[i]) / errors[i + n]
        max_dev = max(max_dev, abs(fwd_dev), abs(cap_dev))
        assert abs(fwd_dev) < _ERROR_THRESHOLD, (
            f"forward {i}: mc={results[i]:.6g} expected={expected_forwards[i]:.6g} "
            f"dev={fwd_dev:.2f} sigma"
        )
        assert abs(cap_dev) < _ERROR_THRESHOLD, (
            f"caplet {i}: mc={results[i + n]:.6g} "
            f"expected={expected_caplets[i]:.6g} dev={cap_dev:.2f} sigma"
        )
    # Sanity: the simulation should reproduce all 2N prices well within the band.
    assert max_dev < _ERROR_THRESHOLD
