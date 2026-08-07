"""Cross-validation for the v1.43 ``ql/experimental/mcbasket`` path engines.

Probe source: ``migration-harness/cpp/probes/v143_experimental_mcengines/probe.cpp``
Reference:    ``migration-harness/references/v143/experimental/mcengines.json``

Classes covered here:

* ``LongstaffSchwartzMultiPathPricer`` (+ its ``PathInfo``) —
  ql/experimental/mcbasket/longstaffschwartzmultipathpricer.{hpp,cpp}
* ``MCLongstaffSchwartzPathEngine`` —
  ql/experimental/mcbasket/mclongstaffschwartzpathengine.hpp
* ``MCAmericanPathEngine`` / ``MakeMCAmericanPathEngine`` —
  ql/experimental/mcbasket/mcamericanpathengine.hpp
* ``MakeMCPathBasketEngine`` — ql/experimental/mcbasket/mcpathbasketengine.hpp

.. rubric:: Why these are *pinned* Monte-Carlo values, not statistical bands

The generators are reproducible: ``MersenneTwisterUniformRng`` is pure integer
state and ``SobolRsg`` is a deterministic direction-number construction, both
mapped through ``InverseCumulativeNormal``. A fixed seed therefore pins an
exact NPV, and every ``*_npv`` / ``*_error_estimate`` assertion below is the
number C++ v1.43 printed for that seed and sample count.

The one thing that would break pathwise reproducibility is
``StochasticProcessArray``, which premultiplies each Gaussian increment by
``pseudoSqrt(correlation, Spectral)``. That pseudo-root is basis-dependent —
any ``M`` with ``M Mᵀ == corr`` is a valid answer and different eigen-solvers
pick different ``M``. Every basket in this module is therefore built with the
**identity** correlation, whose pseudo-root is the identity in any basis, so
the sampled paths do not depend on the eigen-solver. (Block ``Z`` of the probe
pins the correlated pseudo-root separately; it belongs to
``StochasticProcessArray``, not to these engines.)

.. rubric:: Tolerance tiers

* ``exact`` — integer/structural facts and payoff arithmetic that is exact in
  binary (the hand-built block-E paths are whole numbers and the coupon is
  1.5): path lengths, state sizes, ``payments``, ``exercises``, ``states``,
  the zero returned during the calibration phase, the calibration-sample
  default, and the ``allowsErrorEstimate`` flags.
* ``tight`` (1e-14 abs / 1e-12 rel) — everything produced by floating-point
  machinery: discount factors, regression coefficients, lower bounds, path
  prices and every Monte-Carlo NPV / error estimate. Largest deviation
  observed across this module: 8.1e-14 relative, on the block-E regression
  coefficients, which come out of an SVD (C++ ``SVD`` vs. this port's
  :class:`~pquantlib.math.matrixutilities.svd.SVD`); the MC NPVs themselves
  agree to 3.5e-15 relative or better.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exceptions import LibraryException
from pquantlib.experimental.mcbasket.adapted_path_payoff import (
    AdaptedPathPayoff,
    ValuationData,
)
from pquantlib.experimental.mcbasket.longstaff_schwartz_multi_path_pricer import (
    LongstaffSchwartzMultiPathPricer,
    PathInfo,
)
from pquantlib.experimental.mcbasket.mc_american_path_engine import (
    MakeMCAmericanPathEngine,
)
from pquantlib.experimental.mcbasket.mc_longstaff_schwartz_path_engine import (
    MCLongstaffSchwartzPathEngine,
)
from pquantlib.experimental.mcbasket.mc_path_basket_engine import (
    MakeMCPathBasketEngine,
)
from pquantlib.experimental.mcbasket.path_multi_asset_option import (
    PathMultiAssetOption,
)
from pquantlib.math.randomnumbers.rng_traits import LowDiscrepancy, PseudoRandom
from pquantlib.methods.montecarlo.lsm_basis_system import PolynomialType
from pquantlib.methods.montecarlo.multi_path import MultiPath
from pquantlib.methods.montecarlo.path import Path
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.processes.black_scholes_merton_process import BlackScholesMertonProcess
from pquantlib.processes.stochastic_process_array import StochasticProcessArray
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.volatility.equity_fx.black_constant_vol import (
    BlackConstantVol,
)
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.termstructures.yield_.implied_term_structure import ImpliedTermStructure
from pquantlib.testing.reference_reader import load as load_reference
from pquantlib.testing.tolerance import exact, tight
from pquantlib.time.calendars.null_calendar import NullCalendar
from pquantlib.time.date import Date
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.time_grid import TimeGrid
from pquantlib.time.time_unit import TimeUnit

if TYPE_CHECKING:
    import numpy.typing as npt

    from pquantlib.termstructures.yield_term_structure import YieldTermStructure

# --- the probe's market (probe.cpp "Shared market", lines 190-233) -----------

_TODAY = Date.from_ymd(15, Month.January, 2024)
_RISK_FREE = 0.05
_DIVIDEND = 0.02
_SPOTS = (100.0, 95.0, 105.0)
_VOLS = (0.20, 0.25, 0.30)

# probe.cpp block D/E/F constants.
_COUPON = 1.5
_CALL_STRIKE = 100.0
_PUT_STRIKE = 105.0


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


def _independent_basket(n: int) -> StochasticProcessArray:
    """``n`` uncorrelated assets — probe.cpp ``independentBasket``.

    The pseudo-root of the identity is the identity in every basis, so the
    sampled paths are eigen-solver independent (see the module docstring).
    """
    return StochasticProcessArray([_bsm(i) for i in range(n)], np.eye(n, dtype=np.float64))


def _three_fixings() -> list[Date]:
    """probe.cpp ``threeFixings`` — +4M, +8M, +12M from the evaluation date."""
    return [
        _TODAY + Period(4, TimeUnit.Months),
        _TODAY + Period(8, TimeUnit.Months),
        _TODAY + Period(12, TimeUnit.Months),
    ]


class BasketPathPayoff(AdaptedPathPayoff):
    """The one concrete ``PathPayoff`` the probe uses (probe.cpp:251-285).

    * ``payments[i]``  = coupon at ``i == 0``, plus ``max(avg(T) - kCall, 0)``
      at the last fixing;
    * ``exercises[i]`` = ``max(kPut - avg(i), 0)`` for ``i >= 1``;
    * ``states[i]``    = ``{avg(i)}`` for ``i >= 1`` — fixing 0 is left
      non-exercisable on purpose, which is what drives the "states empty =>
      cannot exercise" branch of both ``calibrate`` and ``__call__``.
    """

    def __init__(self, coupon: float, k_call: float, k_put: float) -> None:
        self._coupon = coupon
        self._k_call = k_call
        self._k_put = k_put

    def name(self) -> str:
        return "basket-path-payoff"

    def description(self) -> str:
        return "coupon at first fixing, basket call at last, American basket put in between"

    def basis_system_dimension(self) -> int:
        return 1

    def _evaluate(self, data: ValuationData) -> None:
        n_times = data.number_of_times()
        n_assets = data.number_of_assets()
        for i in range(n_times):
            total = 0.0
            for j in range(n_assets):
                total += data.get_asset_value(i, j)
            avg = total / n_assets

            payment = 0.0
            if i == 0:
                payment += self._coupon
            if i == n_times - 1:
                payment += max(avg - self._k_call, 0.0)
            data.set_payoff_value(i, payment)

            if i > 0:
                state = np.array([avg], dtype=np.float64)
                data.set_exercise_data(i, max(self._k_put - avg, 0.0), state)


# --- block E: the LSM pricer on hand-built paths ------------------------------
#
# One asset, four grid nodes (t = 0, 1/3, 2/3, 1), three fixings at grid
# positions {1, 2, 3}. probe.cpp:566-586.

_E_GRID = (0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0)
_E_TIME_POSITIONS = (1, 2, 3)

_E_CALIB = (
    (100.0, 108.0, 112.0, 120.0),
    (100.0, 96.0, 92.0, 88.0),
    (100.0, 102.0, 99.0, 104.0),
    (100.0, 94.0, 97.0, 93.0),
    (100.0, 110.0, 101.0, 95.0),
    (100.0, 90.0, 85.0, 101.0),
    (100.0, 105.0, 115.0, 109.0),
    (100.0, 98.0, 103.0, 91.0),
)

_E_EVAL = (
    (100.0, 101.0, 96.0, 99.0),
    (100.0, 92.0, 106.0, 113.0),
    (100.0, 115.0, 118.0, 84.0),
    (100.0, 99.0, 99.0, 99.0),
)


def _hand_built_path(values: tuple[float, ...], grid: TimeGrid) -> MultiPath:
    """probe.cpp ``handBuiltPath`` — a single-asset MultiPath on ``grid``."""
    return MultiPath([Path(grid, np.array(values, dtype=np.float64))])


class _LsmFixture:
    """Everything block E needs, rebuilt per test so state never leaks."""

    __slots__ = ("discounts", "forward_ts", "grid", "payoff")

    def __init__(self) -> None:
        ObservableSettings().evaluation_date = _TODAY
        self.grid: TimeGrid = TimeGrid.with_mandatory(_E_GRID)
        risk_free = _flat(_RISK_FREE)
        self.discounts: npt.NDArray[np.float64] = np.array(
            [risk_free.discount(_E_GRID[p]) for p in _E_TIME_POSITIONS], dtype=np.float64
        )
        self.forward_ts: list[YieldTermStructure] = [
            ImpliedTermStructure(risk_free, d) for d in _three_fixings()
        ]
        self.payoff: BasketPathPayoff = BasketPathPayoff(_COUPON, _CALL_STRIKE, _PUT_STRIKE)

    def pricer(self) -> LongstaffSchwartzMultiPathPricer:
        # C++ parity: the arguments MCAmericanPathEngine::lsmPathPricer feeds
        # in (mcamericanpathengine.hpp:151-160) — order 2, Monomial.
        return LongstaffSchwartzMultiPathPricer(
            self.payoff,
            _E_TIME_POSITIONS,
            self.forward_ts,
            self.discounts,
            2,
            PolynomialType.Monomial,
        )

    def calibrated_pricer(self) -> LongstaffSchwartzMultiPathPricer:
        pricer = self.pricer()
        for values in _E_CALIB:
            pricer(_hand_built_path(values, self.grid))
        pricer.calibrate()
        return pricer


@pytest.fixture
def lsm() -> _LsmFixture:
    return _LsmFixture()


def test_lsm_setup_matches_cpp(cpp: dict[str, Any], lsm: _LsmFixture) -> None:
    """The grid, discounts and basis size the pricer is constructed with."""
    for actual, expected in zip(lsm.grid.times, cpp["E_times"], strict=True):
        tight(actual, expected)
    for actual, expected in zip(lsm.discounts, cpp["E_discounts"], strict=True):
        tight(float(actual), expected)

    pricer = lsm.pricer()
    # LsmBasisSystem::multiPathBasisSystem(1, 2, Monomial) => {1, x, x^2}.
    assert len(pricer.basis_system()) == cpp["E_basis_size"]
    assert pricer.calibration_phase() is cpp["E_calibration_phase_initial"]


def test_lsm_rejects_unsupported_polynomial_type(lsm: _LsmFixture) -> None:
    """# C++ parity: ``QL_REQUIRE(... , "insufficient polynomial type")``
    (longstaffschwartzmultipathpricer.cpp:50-55)."""
    with pytest.raises(LibraryException, match="insufficient polynomial type"):
        LongstaffSchwartzMultiPathPricer(
            lsm.payoff,
            _E_TIME_POSITIONS,
            lsm.forward_ts,
            lsm.discounts,
            2,
            PolynomialType.Legendre,
        )


@pytest.mark.parametrize("path_index", range(len(_E_CALIB)))
def test_lsm_transform_path_path_info(
    cpp: dict[str, Any], lsm: _LsmFixture, path_index: int
) -> None:
    """``PathInfo`` straight out of ``transformPath``.

    # C++ parity: ``LongstaffSchwartzMultiPathPricer::transformPath``
    # (longstaffschwartzmultipathpricer.cpp:61-80) and ``PathInfo``
    # (longstaffschwartzmultipathpricer.cpp:27-35).
    """
    pricer = lsm.pricer()
    info: PathInfo = pricer.transform_path(_hand_built_path(_E_CALIB[path_index], lsm.grid))
    tag = f"E_pathinfo_{path_index}"

    assert info.path_length() == cpp[f"{tag}_path_length"]
    for actual, expected in zip(info.payments, cpp[f"{tag}_payments"], strict=True):
        exact(float(actual), float(expected))
    for actual, expected in zip(info.exercises, cpp[f"{tag}_exercises"], strict=True):
        exact(float(actual), float(expected))
    # An empty state is how the payoff says "exercise impossible here"; fixing
    # 0 must therefore carry size 0 while 1 and 2 carry the basis dimension.
    assert [s.size for s in info.states] == cpp[f"{tag}_state_sizes"]
    flat = [float(v) for state in info.states for v in state]
    for actual, expected in zip(flat, cpp[f"{tag}_state_values"], strict=True):
        exact(actual, float(expected))


def test_lsm_calibration_phase_returns_zero(cpp: dict[str, Any], lsm: _LsmFixture) -> None:
    """During calibration ``operator()`` records the path and returns 0.0.

    # C++ parity: longstaffschwartzmultipathpricer.cpp:86-92.
    """
    pricer = lsm.pricer()
    returns = [pricer(_hand_built_path(values, lsm.grid)) for values in _E_CALIB]
    for actual, expected in zip(returns, cpp["E_calibration_phase_returns"], strict=True):
        exact(actual, float(expected))


def test_lsm_calibrate_regression(cpp: dict[str, Any], lsm: _LsmFixture) -> None:
    """The fitted exercise policy: ``coeff_``, its size code, ``lowerBounds_``.

    # C++ parity: ``LongstaffSchwartzMultiPathPricer::calibrate``
    # (longstaffschwartzmultipathpricer.cpp:155-308).

    ``len(coeff_[i])`` is the encoded decision: 0 => never exercise,
    ``len(v_)`` => use the fitted continuation value, ``len(v_) + 1`` =>
    always exercise. At fixing 0 the payoff makes exercise impossible, so no
    path joins the regression and C++ falls through to "never" (size 0).
    """
    pricer = lsm.calibrated_pricer()
    assert pricer.calibration_phase() is cpp["E_calibration_phase_after"]

    coefficients = pricer.coefficients()
    assert len(coefficients) == len(_E_TIME_POSITIONS) - 1
    for i, coeff in enumerate(coefficients):
        assert coeff.size == cpp[f"E_coeff_{i}_size"]
        expected_coeff = cpp[f"E_coeff_{i}"]
        for actual, expected in zip(coeff, expected_coeff, strict=True):
            # SVD-driven: C++ ``SVD`` vs. this port's ``SVD``. Observed
            # agreement 8.1e-14 relative — inside the TIGHT 1e-12 rel band.
            tight(float(actual), expected)

    for actual, expected in zip(pricer.lower_bounds(), cpp["E_lower_bounds"], strict=True):
        tight(float(actual), expected)


def test_lsm_pricing_phase(cpp: dict[str, Any], lsm: _LsmFixture) -> None:
    """``operator()`` after calibration, on fresh paths.

    # C++ parity: longstaffschwartzmultipathpricer.cpp:82-153.
    """
    pricer = lsm.calibrated_pricer()
    prices = [pricer(_hand_built_path(values, lsm.grid)) for values in _E_EVAL]
    for actual, expected in zip(prices, cpp["E_eval_prices"], strict=True):
        tight(actual, expected)


def test_lsm_reprices_calibration_paths(cpp: dict[str, Any], lsm: _LsmFixture) -> None:
    """The calibration paths re-priced after ``calibrate`` — the in-sample set
    ``MCLongstaffSchwartzPathEngine`` actually prices (see block F note)."""
    pricer = lsm.calibrated_pricer()
    prices = [pricer(_hand_built_path(values, lsm.grid)) for values in _E_CALIB]
    for actual, expected in zip(prices, cpp["E_calib_prices_after"], strict=True):
        tight(actual, expected)


# --- block D: MCPathBasketEngine / MakeMCPathBasketEngine --------------------


@pytest.fixture
def basket_option() -> PathMultiAssetOption:
    ObservableSettings().evaluation_date = _TODAY
    return PathMultiAssetOption(
        BasketPathPayoff(_COUPON, _CALL_STRIKE, _PUT_STRIKE), _three_fixings()
    )


def test_path_payoff_metadata(cpp: dict[str, Any]) -> None:
    """The probe's payoff, as the probe describes it (probe.cpp:520-526)."""
    payoff = BasketPathPayoff(_COUPON, _CALL_STRIKE, _PUT_STRIKE)
    assert payoff.basis_system_dimension() == cpp["D_basis_system_dimension"]
    assert payoff.name() == cpp["D_payoff_name"]
    assert payoff.description() == cpp["D_payoff_description"]
    exact(_COUPON, cpp["D_coupon"])
    exact(_CALL_STRIKE, cpp["D_call_strike"])
    exact(_PUT_STRIKE, cpp["D_put_strike"])


def test_make_mc_path_basket_engine_pseudo_random(
    cpp: dict[str, Any], basket_option: PathMultiAssetOption
) -> None:
    """# C++ parity: ``MakeMCPathBasketEngine<PseudoRandom>`` (probe block D1)."""
    basket_option.set_pricing_engine(
        MakeMCPathBasketEngine(_independent_basket(3))
        .with_steps(3)
        .with_samples(255)
        .with_seed(42)
        .build()
    )
    tight(basket_option.npv(), cpp["D1_npv"])
    tight(basket_option.error_estimate(), cpp["D1_error_estimate"])


def test_make_mc_path_basket_engine_steps_per_year_antithetic(
    cpp: dict[str, Any], basket_option: PathMultiAssetOption
) -> None:
    """# C++ parity: probe block D2 — ``withStepsPerYear`` + antithetic."""
    basket_option.set_pricing_engine(
        MakeMCPathBasketEngine(_independent_basket(3))
        .with_steps_per_year(6)
        .with_samples(255)
        .with_antithetic_variate()
        .with_seed(42)
        .build()
    )
    tight(basket_option.npv(), cpp["D2_npv"])
    tight(basket_option.error_estimate(), cpp["D2_error_estimate"])


def test_make_mc_path_basket_engine_low_discrepancy(
    cpp: dict[str, Any], basket_option: PathMultiAssetOption
) -> None:
    """# C++ parity: probe block D3 — Sobol, and no error estimate published."""
    basket_option.set_pricing_engine(
        MakeMCPathBasketEngine(_independent_basket(3), LowDiscrepancy)
        .with_steps(3)
        .with_samples(256)
        .with_seed(11)
        .build()
    )
    tight(basket_option.npv(), cpp["D3_npv"])
    with pytest.raises(LibraryException, match=cpp["D3_error_estimate_raises"]):
        basket_option.error_estimate()


def test_make_mc_path_basket_engine_guards(cpp: dict[str, Any]) -> None:
    """The builder's named-parameter guards (probe block G).

    Unlike Everest and American, this builder does **not** guard the step
    count in its conversion operator: the engine constructor is what raises.
    """
    processes = _independent_basket(3)
    with pytest.raises(LibraryException, match=cpp["G_pathbasket_samples_after_tolerance"]):
        MakeMCPathBasketEngine(processes).with_absolute_tolerance(1.0).with_samples(10)
    with pytest.raises(LibraryException, match=cpp["G_pathbasket_no_steps"]):
        MakeMCPathBasketEngine(processes).with_samples(10).build()


def test_make_mc_path_basket_engine_single_sample_raises(cpp: dict[str, Any]) -> None:
    """A single pseudo-random sample raises out of the error estimate.

    # C++ parity: ``MCPathBasketEngine::calculate`` writes
    # ``results_.errorEstimate`` unconditionally once
    # ``RNG::allowsErrorEstimate``; ``GeneralStatistics::variance`` refuses on
    # ``N == 1`` (probe block H).
    """
    ObservableSettings().evaluation_date = _TODAY
    option = PathMultiAssetOption(
        BasketPathPayoff(_COUPON, _CALL_STRIKE, _PUT_STRIKE), _three_fixings()
    )
    option.set_pricing_engine(
        MakeMCPathBasketEngine(_independent_basket(3))
        .with_steps(3)
        .with_samples(1)
        .with_seed(42)
        .build()
    )
    with pytest.raises(LibraryException, match=cpp["H_pathbasket_single_sample"]):
        option.npv()


# --- block F: MCAmericanPathEngine / MakeMCAmericanPathEngine ----------------


def test_make_mc_american_path_engine_pseudo_random(
    cpp: dict[str, Any], basket_option: PathMultiAssetOption
) -> None:
    """# C++ parity: ``MakeMCAmericanPathEngine<PseudoRandom>`` (probe block F1)."""
    basket_option.set_pricing_engine(
        MakeMCAmericanPathEngine(_independent_basket(3))
        .with_steps(3)
        .with_samples(255)
        .with_calibration_samples(128)
        .with_seed(42)
        .build()
    )
    tight(basket_option.npv(), cpp["F1_npv"])
    tight(basket_option.error_estimate(), cpp["F1_error_estimate"])


def test_make_mc_american_path_engine_steps_per_year_antithetic(
    cpp: dict[str, Any], basket_option: PathMultiAssetOption
) -> None:
    """# C++ parity: probe block F2."""
    basket_option.set_pricing_engine(
        MakeMCAmericanPathEngine(_independent_basket(3))
        .with_steps_per_year(6)
        .with_samples(255)
        .with_calibration_samples(128)
        .with_antithetic_variate()
        .with_seed(42)
        .build()
    )
    tight(basket_option.npv(), cpp["F2_npv"])
    tight(basket_option.error_estimate(), cpp["F2_error_estimate"])


def test_make_mc_american_path_engine_low_discrepancy(
    cpp: dict[str, Any], basket_option: PathMultiAssetOption
) -> None:
    """# C++ parity: probe block F3 — Sobol, no error estimate published."""
    basket_option.set_pricing_engine(
        MakeMCAmericanPathEngine(_independent_basket(3), LowDiscrepancy)
        .with_steps(3)
        .with_samples(256)
        .with_calibration_samples(128)
        .with_seed(11)
        .build()
    )
    tight(basket_option.npv(), cpp["F3_npv"])
    with pytest.raises(LibraryException, match=cpp["F3_error_estimate_raises"]):
        basket_option.error_estimate()


def test_mc_longstaff_schwartz_path_engine_default_calibration_samples(
    cpp: dict[str, Any], basket_option: PathMultiAssetOption
) -> None:
    """Omitting ``withCalibrationSamples`` means 2048, not "unset".

    # C++ parity: ``nCalibrationSamples_((nCalibrationSamples == Null<Size>())
    # ? 2048 : nCalibrationSamples)`` (mclongstaffschwartzpathengine.hpp:104).

    F4 equals F1 to the last bit even though F1 calibrated on 128 paths and F4
    on 2048: ``McSimulation::calculate`` discards the calibration model and
    rebuilds the generator from the same seed, so the *pricing* stream is
    identical and here the two fitted policies make the same decisions.
    """
    engine = MakeMCAmericanPathEngine(_independent_basket(3))
    engine_built = engine.with_steps(3).with_samples(255).with_seed(42).build()
    n_calibration = engine_built._n_calibration_samples  # pyright: ignore[reportPrivateUsage]
    assert n_calibration == cpp["F_default_calibration_samples"]
    basket_option.set_pricing_engine(engine_built)
    tight(basket_option.npv(), cpp["F4_npv_default_calibration_samples"])


def test_mc_longstaff_schwartz_path_engine_path_pricer_unknown() -> None:
    """``pathPricer()`` before ``calculate()`` raises.

    # C++ parity: ``QL_REQUIRE(pathPricer_, "path pricer unknown")``
    # (mclongstaffschwartzpathengine.hpp:128).
    """
    ObservableSettings().evaluation_date = _TODAY
    engine = MakeMCAmericanPathEngine(_independent_basket(3)).with_steps(3).build()
    assert isinstance(engine, MCLongstaffSchwartzPathEngine)
    with pytest.raises(LibraryException, match="path pricer unknown"):
        engine.path_pricer()


def test_make_mc_american_path_engine_guards(cpp: dict[str, Any]) -> None:
    """The builder's step-count guards (probe block G)."""
    processes = _independent_basket(3)
    with pytest.raises(LibraryException, match=cpp["G_american_no_steps"]):
        MakeMCAmericanPathEngine(processes).with_samples(10).build()
    with pytest.raises(LibraryException, match=cpp["G_american_steps_overspecified"]):
        (
            MakeMCAmericanPathEngine(processes)
            .with_steps(2)
            .with_steps_per_year(2)
            .with_samples(10)
            .build()
        )


def test_make_mc_american_path_engine_single_sample_raises(cpp: dict[str, Any]) -> None:
    """# C++ parity: probe block H — one pseudo-random sample, no variance."""
    ObservableSettings().evaluation_date = _TODAY
    option = PathMultiAssetOption(
        BasketPathPayoff(_COUPON, _CALL_STRIKE, _PUT_STRIKE), _three_fixings()
    )
    option.set_pricing_engine(
        MakeMCAmericanPathEngine(_independent_basket(3))
        .with_steps(3)
        .with_samples(1)
        .with_calibration_samples(128)
        .with_seed(42)
        .build()
    )
    with pytest.raises(LibraryException, match=cpp["H_american_single_sample"]):
        option.npv()


def test_rng_traits_allow_error_estimate_flags(cpp: dict[str, Any]) -> None:
    """The compile-time flag the engines branch on.

    # C++ parity: ``PseudoRandom::allowsErrorEstimate == 1``,
    # ``LowDiscrepancy::allowsErrorEstimate == 0`` (rngtraits.hpp).
    """
    assert PseudoRandom.allows_error_estimate == cpp["G_pseudorandom_allows_error_estimate"]
    assert LowDiscrepancy.allows_error_estimate == cpp["G_lowdiscrepancy_allows_error_estimate"]
