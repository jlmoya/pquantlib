"""Cross-validation for the v1.43 exotic-basket Monte-Carlo engine factories.

Probe source: ``migration-harness/cpp/probes/v143_experimental_mcengines/probe.cpp``
Reference:    ``migration-harness/references/v143/experimental/mcengines.json``

Classes covered here:

* ``MakeMCEverestEngine``  — ql/experimental/exoticoptions/mceverestengine.hpp
* ``MakeMCHimalayaEngine`` — ql/experimental/exoticoptions/mchimalayaengine.hpp
* ``MakeMCPagodaEngine``   — ql/experimental/exoticoptions/mcpagodaengine.hpp

(The three engines and their path pricers are exercised through the builders,
which is the only way C++ hands one out — ``operator shared_ptr<PricingEngine>``.)

.. rubric:: Why these are *pinned* Monte-Carlo values, not statistical bands

``MersenneTwisterUniformRng`` is pure integer state and ``SobolRsg`` is a
deterministic direction-number construction; both go through
``InverseCumulativeNormal``. A fixed seed therefore pins an exact NPV. Every
basket below uses the **identity** correlation, because that is the one
correlation whose ``pseudoSqrt(·, Spectral)`` is basis-independent (it is the
identity in every basis) — so the sampled paths do not depend on which
eigen-solver ``StochasticProcessArray`` uses. See the probe's block-Z comment.

.. rubric:: Tolerance tiers

* ``exact`` — the ``allowsErrorEstimate`` flags and the notional/guarantee/
  roof/fraction/strike echoes of the constructor arguments.
* ``tight`` (1e-14 abs / 1e-12 rel) — every NPV, yield and error estimate.
  Largest deviation observed across this module: 3.6e-15 relative
  (``C2_npv``); every value sits below 4e-15, i.e. three orders of magnitude
  inside the tier.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exceptions import LibraryException
from pquantlib.exercise import EuropeanExercise
from pquantlib.experimental.exoticoptions.everest_option import EverestOption
from pquantlib.experimental.exoticoptions.himalaya_option import HimalayaOption
from pquantlib.experimental.exoticoptions.pagoda_option import PagodaOption
from pquantlib.math.randomnumbers.rng_traits import LowDiscrepancy, PseudoRandom
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.pricingengines.exoticoptions.mc_everest_engine import MakeMCEverestEngine
from pquantlib.pricingengines.exoticoptions.mc_himalaya_engine import (
    MakeMCHimalayaEngine,
)
from pquantlib.pricingengines.exoticoptions.mc_pagoda_engine import MakeMCPagodaEngine
from pquantlib.processes.black_scholes_merton_process import BlackScholesMertonProcess
from pquantlib.processes.stochastic_process_array import StochasticProcessArray
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.volatility.equity_fx.black_constant_vol import (
    BlackConstantVol,
)
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing.reference_reader import load as load_reference
from pquantlib.testing.tolerance import exact, tight
from pquantlib.time.calendars.null_calendar import NullCalendar
from pquantlib.time.date import Date
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit

# --- the probe's market (probe.cpp "Shared market", lines 190-233) -----------

_TODAY = Date.from_ymd(15, Month.January, 2024)
_RISK_FREE = 0.05
_DIVIDEND = 0.02
_SPOTS = (100.0, 95.0, 105.0)
_VOLS = (0.20, 0.25, 0.30)

_NOTIONAL = 1.0e6
_GUARANTEE = 0.03
_ROOF = 15.0
_FRACTION = 0.62
_STRIKE = 100.0


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return load_reference("v143/experimental/mcengines")


def _flat(rate: float) -> FlatForward:
    return FlatForward.from_rate(
        reference_date=_TODAY, forward_rate=rate, day_counter=Actual365Fixed()
    )


def _bsm(i: int) -> BlackScholesMertonProcess:
    return BlackScholesMertonProcess(
        x0=SimpleQuote(_SPOTS[i]),
        dividend_ts=_flat(_DIVIDEND),
        risk_free_ts=_flat(_RISK_FREE),
        black_vol_ts=BlackConstantVol(
            reference_date=_TODAY,
            calendar=NullCalendar(),
            day_counter=Actual365Fixed(),
            volatility=_VOLS[i],
        ),
    )


@pytest.fixture
def basket3() -> StochasticProcessArray:
    """Three uncorrelated assets — probe.cpp ``independentBasket(3)``."""
    ObservableSettings().evaluation_date = _TODAY
    return StochasticProcessArray([_bsm(i) for i in range(3)], np.eye(3, dtype=np.float64))


def _three_fixings() -> list[Date]:
    """probe.cpp ``threeFixings`` — +4M, +8M, +12M from the evaluation date."""
    return [
        _TODAY + Period(4, TimeUnit.Months),
        _TODAY + Period(8, TimeUnit.Months),
        _TODAY + Period(12, TimeUnit.Months),
    ]


def _one_year_exercise() -> EuropeanExercise:
    return EuropeanExercise(_TODAY + Period(1, TimeUnit.Years))


# --- block A: MakeMCEverestEngine --------------------------------------------


def test_everest_contract_terms_match_probe(cpp: dict[str, Any]) -> None:
    exact(_NOTIONAL, float(cpp["A_notional"]))
    exact(_GUARANTEE, cpp["A_guarantee"])


def test_make_mc_everest_engine_pseudo_random(
    cpp: dict[str, Any], basket3: StochasticProcessArray
) -> None:
    """# C++ parity: ``MakeMCEverestEngine<PseudoRandom>`` (probe block A1).

    ``yield`` is filled by ``MCEverestEngine::calculate`` itself
    (mceverestengine.hpp:67-69): ``value / (notional * endDiscount) - 1``.
    """
    option = EverestOption(_NOTIONAL, _GUARANTEE, _one_year_exercise())
    option.set_pricing_engine(
        MakeMCEverestEngine(basket3).with_steps(4).with_samples(255).with_seed(42).build()
    )
    tight(option.npv(), cpp["A1_npv"])
    tight(option.yield_(), cpp["A1_yield"])
    tight(option.error_estimate(), cpp["A1_error_estimate"])


def test_make_mc_everest_engine_steps_per_year_antithetic(
    cpp: dict[str, Any], basket3: StochasticProcessArray
) -> None:
    """# C++ parity: probe block A2 — ``withStepsPerYear(12)`` + antithetic.

    ``timeStepsPerYear_ * residualTime`` is truncated, not rounded:
    ``12 * 366/365 = 12.033`` becomes 12 steps (mceverestengine.hpp:178-180).
    """
    option = EverestOption(_NOTIONAL, _GUARANTEE, _one_year_exercise())
    option.set_pricing_engine(
        MakeMCEverestEngine(basket3)
        .with_steps_per_year(12)
        .with_samples(255)
        .with_antithetic_variate()
        .with_seed(7)
        .build()
    )
    tight(option.npv(), cpp["A2_npv"])
    tight(option.yield_(), cpp["A2_yield"])
    tight(option.error_estimate(), cpp["A2_error_estimate"])


def test_make_mc_everest_engine_low_discrepancy(
    cpp: dict[str, Any], basket3: StochasticProcessArray
) -> None:
    """# C++ parity: probe block A3 — Sobol; no error estimate is published."""
    option = EverestOption(_NOTIONAL, _GUARANTEE, _one_year_exercise())
    option.set_pricing_engine(
        MakeMCEverestEngine(basket3, LowDiscrepancy)
        .with_steps(4)
        .with_samples(256)
        .with_seed(11)
        .build()
    )
    tight(option.npv(), cpp["A3_npv"])
    tight(option.yield_(), cpp["A3_yield"])
    with pytest.raises(LibraryException, match=cpp["A3_error_estimate_raises"]):
        option.error_estimate()


def test_make_mc_everest_engine_absolute_tolerance(
    cpp: dict[str, Any], basket3: StochasticProcessArray
) -> None:
    """Tolerance-driven termination.

    # C++ parity: ``McSimulation::value`` — ``minSamples`` defaults to 1023,
    # then batches grow by ``sampleNumber * (error/tolerance)^2 * 0.8``
    # (mcsimulation.hpp). The stopping point, and therefore the NPV, is pinned
    # by the seed (probe block A4).
    """
    option = EverestOption(_NOTIONAL, _GUARANTEE, _one_year_exercise())
    option.set_pricing_engine(
        MakeMCEverestEngine(basket3)
        .with_steps(2)
        .with_absolute_tolerance(4000.0)
        .with_max_samples(20000)
        .with_seed(3)
        .build()
    )
    tight(option.npv(), cpp["A4_npv"])
    tight(option.error_estimate(), cpp["A4_error_estimate"])


def test_make_mc_everest_engine_guards(
    cpp: dict[str, Any], basket3: StochasticProcessArray
) -> None:
    """The builder's named-parameter guards (probe block G)."""
    with pytest.raises(LibraryException, match=cpp["G_everest_samples_after_tolerance"]):
        MakeMCEverestEngine(basket3).with_absolute_tolerance(1.0).with_samples(10)
    with pytest.raises(LibraryException, match=cpp["G_everest_tolerance_after_samples"]):
        MakeMCEverestEngine(basket3).with_samples(10).with_absolute_tolerance(1.0)
    with pytest.raises(LibraryException, match=cpp["G_everest_tolerance_lowdiscrepancy"]):
        MakeMCEverestEngine(basket3, LowDiscrepancy).with_absolute_tolerance(1.0)
    with pytest.raises(LibraryException, match=cpp["G_everest_no_steps"]):
        MakeMCEverestEngine(basket3).with_samples(10).build()
    with pytest.raises(LibraryException, match=cpp["G_everest_steps_overspecified"]):
        (
            MakeMCEverestEngine(basket3)
            .with_steps(2)
            .with_steps_per_year(2)
            .with_samples(10)
            .build()
        )


def test_make_mc_everest_engine_single_sample_raises(
    cpp: dict[str, Any], basket3: StochasticProcessArray
) -> None:
    """A single pseudo-random sample raises out of the error estimate.

    # C++ parity: ``MCEverestEngine::calculate`` writes
    # ``results_.errorEstimate`` unconditionally once
    # ``RNG::allowsErrorEstimate``, and ``GeneralStatistics::variance``
    # refuses on ``N == 1`` (probe block H). Note the raise happens *before*
    # the yield is computed, so ``NPV()`` itself throws.
    """
    option = EverestOption(_NOTIONAL, _GUARANTEE, _one_year_exercise())
    option.set_pricing_engine(
        MakeMCEverestEngine(basket3).with_steps(2).with_samples(1).with_seed(42).build()
    )
    with pytest.raises(LibraryException, match=cpp["H_everest_single_sample"]):
        option.npv()


def test_make_mc_everest_engine_single_sample_low_discrepancy(
    cpp: dict[str, Any], basket3: StochasticProcessArray
) -> None:
    """LowDiscrepancy never enters that branch, so one sample prices fine.

    # C++ parity: probe block H.
    """
    option = EverestOption(_NOTIONAL, _GUARANTEE, _one_year_exercise())
    option.set_pricing_engine(
        MakeMCEverestEngine(basket3, LowDiscrepancy)
        .with_steps(2)
        .with_samples(1)
        .with_seed(11)
        .build()
    )
    tight(option.npv(), cpp["H_everest_lowdiscrepancy_single_sample_npv"])
    with pytest.raises(
        LibraryException,
        match=cpp["H_everest_lowdiscrepancy_single_sample_error_estimate_raises"],
    ):
        option.error_estimate()


# --- block B: MakeMCPagodaEngine ---------------------------------------------


def test_pagoda_contract_terms_match_probe(cpp: dict[str, Any]) -> None:
    exact(_ROOF, cpp["B_roof"])
    exact(_FRACTION, cpp["B_fraction"])


def test_make_mc_pagoda_engine_pseudo_random(
    cpp: dict[str, Any], basket3: StochasticProcessArray
) -> None:
    """# C++ parity: ``MakeMCPagodaEngine<PseudoRandom>`` (probe block B1).

    The Pagoda grid is exactly the fixing dates — this builder takes no step
    count at all (mcpagodaengine.hpp:144-157).
    """
    option = PagodaOption(_three_fixings(), _ROOF, _FRACTION)
    option.set_pricing_engine(
        MakeMCPagodaEngine(basket3).with_samples(255).with_seed(42).build()
    )
    tight(option.npv(), cpp["B1_npv"])
    tight(option.error_estimate(), cpp["B1_error_estimate"])


def test_make_mc_pagoda_engine_antithetic(
    cpp: dict[str, Any], basket3: StochasticProcessArray
) -> None:
    """# C++ parity: probe block B2."""
    option = PagodaOption(_three_fixings(), _ROOF, _FRACTION)
    option.set_pricing_engine(
        MakeMCPagodaEngine(basket3)
        .with_samples(255)
        .with_antithetic_variate()
        .with_seed(42)
        .build()
    )
    tight(option.npv(), cpp["B2_npv"])
    tight(option.error_estimate(), cpp["B2_error_estimate"])


def test_make_mc_pagoda_engine_low_discrepancy(
    cpp: dict[str, Any], basket3: StochasticProcessArray
) -> None:
    """# C++ parity: probe block B3."""
    option = PagodaOption(_three_fixings(), _ROOF, _FRACTION)
    option.set_pricing_engine(
        MakeMCPagodaEngine(basket3, LowDiscrepancy).with_samples(256).with_seed(11).build()
    )
    tight(option.npv(), cpp["B3_npv"])
    with pytest.raises(LibraryException, match=cpp["B3_error_estimate_raises"]):
        option.error_estimate()


def test_make_mc_pagoda_engine_guards(
    cpp: dict[str, Any], basket3: StochasticProcessArray
) -> None:
    """# C++ parity: probe block G."""
    with pytest.raises(LibraryException, match=cpp["G_pagoda_samples_after_tolerance"]):
        MakeMCPagodaEngine(basket3).with_absolute_tolerance(1.0).with_samples(10)


def test_make_mc_pagoda_engine_single_sample_raises(
    cpp: dict[str, Any], basket3: StochasticProcessArray
) -> None:
    """# C++ parity: probe block H."""
    option = PagodaOption(_three_fixings(), _ROOF, _FRACTION)
    option.set_pricing_engine(
        MakeMCPagodaEngine(basket3).with_samples(1).with_seed(42).build()
    )
    with pytest.raises(LibraryException, match=cpp["H_pagoda_single_sample"]):
        option.npv()


# --- block C: MakeMCHimalayaEngine -------------------------------------------


def test_himalaya_contract_terms_match_probe(cpp: dict[str, Any]) -> None:
    exact(_STRIKE, cpp["C_strike"])


def test_make_mc_himalaya_engine_pseudo_random(
    cpp: dict[str, Any], basket3: StochasticProcessArray
) -> None:
    """# C++ parity: ``MakeMCHimalayaEngine<PseudoRandom>`` (probe block C1)."""
    option = HimalayaOption(_three_fixings(), _STRIKE)
    option.set_pricing_engine(
        MakeMCHimalayaEngine(basket3).with_samples(255).with_seed(42).build()
    )
    tight(option.npv(), cpp["C1_npv"])
    tight(option.error_estimate(), cpp["C1_error_estimate"])


def test_make_mc_himalaya_engine_antithetic(
    cpp: dict[str, Any], basket3: StochasticProcessArray
) -> None:
    """# C++ parity: probe block C2."""
    option = HimalayaOption(_three_fixings(), _STRIKE)
    option.set_pricing_engine(
        MakeMCHimalayaEngine(basket3)
        .with_samples(255)
        .with_antithetic_variate()
        .with_seed(42)
        .build()
    )
    tight(option.npv(), cpp["C2_npv"])
    tight(option.error_estimate(), cpp["C2_error_estimate"])


def test_make_mc_himalaya_engine_low_discrepancy(
    cpp: dict[str, Any], basket3: StochasticProcessArray
) -> None:
    """# C++ parity: probe block C3."""
    option = HimalayaOption(_three_fixings(), _STRIKE)
    option.set_pricing_engine(
        MakeMCHimalayaEngine(basket3, LowDiscrepancy).with_samples(256).with_seed(11).build()
    )
    tight(option.npv(), cpp["C3_npv"])
    with pytest.raises(LibraryException, match=cpp["C3_error_estimate_raises"]):
        option.error_estimate()


def test_make_mc_himalaya_engine_guards(
    cpp: dict[str, Any], basket3: StochasticProcessArray
) -> None:
    """# C++ parity: probe block G.

    Unlike the Everest builder there is no step count to give — the Himalaya
    grid is the fixing dates — so converting with nothing but a sample count
    must succeed.
    """
    with pytest.raises(LibraryException, match=cpp["G_himalaya_tolerance_lowdiscrepancy"]):
        MakeMCHimalayaEngine(basket3, LowDiscrepancy).with_absolute_tolerance(1.0)
    converted = MakeMCHimalayaEngine(basket3).with_samples(10).build()
    assert (converted is not None) is cpp["G_himalaya_converts_without_steps"]


def test_make_mc_himalaya_engine_single_sample_raises(
    cpp: dict[str, Any], basket3: StochasticProcessArray
) -> None:
    """# C++ parity: probe block H."""
    option = HimalayaOption(_three_fixings(), _STRIKE)
    option.set_pricing_engine(
        MakeMCHimalayaEngine(basket3).with_samples(1).with_seed(42).build()
    )
    with pytest.raises(LibraryException, match=cpp["H_himalaya_single_sample"]):
        option.npv()


def test_rng_traits_allow_error_estimate_flags(cpp: dict[str, Any]) -> None:
    """# C++ parity: ``PseudoRandom::allowsErrorEstimate`` /
    ``LowDiscrepancy::allowsErrorEstimate`` (rngtraits.hpp)."""
    assert PseudoRandom.allows_error_estimate == cpp["G_pseudorandom_allows_error_estimate"]
    assert LowDiscrepancy.allows_error_estimate == cpp["G_lowdiscrepancy_allows_error_estimate"]


def test_builder_call_operator_matches_build(basket3: StochasticProcessArray) -> None:
    """``__call__`` mirrors the C++ conversion operator; it must equal ``build``.

    # C++ parity: ``operator ext::shared_ptr<PricingEngine>() const`` on each
    # of the three factories.
    """
    option_a = HimalayaOption(_three_fixings(), _STRIKE)
    option_b = HimalayaOption(_three_fixings(), _STRIKE)
    option_a.set_pricing_engine(
        MakeMCHimalayaEngine(basket3).with_samples(255).with_seed(42).build()
    )
    option_b.set_pricing_engine(
        MakeMCHimalayaEngine(basket3).with_samples(255).with_seed(42)()
    )
    exact(option_b.npv(), option_a.npv())
