"""Cross-validation of ``LiborForwardModel`` + ``LfmSwaptionEngine`` vs C++ v1.43.

Expected values come from
``migration-harness/references/v143/legacy/lmm_swaption.json``
(``migration-harness/cpp/probes/v143_legacy_lmmswaption/probe.cpp``).

EVALUATION DATE: ``probe.cpp`` ``makeIndex`` sets
``Settings::instance().evaluationDate() = index->fixingCalendar().adjust(
Date(4, September, 2005))``. The autouse fixture pins the same date and
restores the previous one.

The market setup mirrors the C++ test-suite ``testSwaptionPricing``: Euribor6M
over a two-pillar ZeroCurve 0.04 -> 0.08 by 2011, ten forward rates, an
exponential correlation model at rho = 0.5 and a linear-exponential volatility
model at (0.291, 1.483, 0.116, 0.00001).
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Final

import numpy as np
import pytest

from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.exceptions import LibraryException
from pquantlib.exercise import EuropeanExercise
from pquantlib.indexes.ibor.euribor import Euribor6M
from pquantlib.instruments.swap import SwapType
from pquantlib.instruments.swaption import (
    SettlementMethod,
    SettlementType,
    Swaption,
)
from pquantlib.instruments.vanilla_swap import VanillaSwap
from pquantlib.legacy.libormarketmodels.lfm_covar_proxy import LfmCovarianceProxy
from pquantlib.legacy.libormarketmodels.lfm_process import LiborForwardModelProcess
from pquantlib.legacy.libormarketmodels.lfm_swaption_engine import LfmSwaptionEngine
from pquantlib.legacy.libormarketmodels.libor_forward_model import LiborForwardModel
from pquantlib.legacy.libormarketmodels.lm_exp_corr_model import (
    LmExponentialCorrelationModel,
)
from pquantlib.legacy.libormarketmodels.lm_lin_exp_vol_model import (
    LmLinearExponentialVolatilityModel,
)
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.payoffs import OptionType
from pquantlib.pricingengines.swap.discounting_swap_engine import DiscountingSwapEngine
from pquantlib.termstructures.protocols import YieldTermStructureProtocol
from pquantlib.termstructures.yield_.interpolated_zero_curve import (
    InterpolatedZeroCurve,
)
from pquantlib.testing import reference_reader
from pquantlib.testing.tolerance import tight
from pquantlib.time.date import Date
from pquantlib.time.date_generation import DateGeneration
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.schedule import Schedule
from pquantlib.time.time_unit import TimeUnit

ANCHOR: Final[Date] = Date.from_ymd(4, Month.September, 2005)
SIZE: Final[int] = 10
VOL_A: Final[float] = 0.291
VOL_B: Final[float] = 1.483
VOL_C: Final[float] = 0.116
VOL_D: Final[float] = 0.00001
CORR_RHO: Final[float] = 0.5
# probe.cpp: struck away from fair so the Black term is not degenerate.
STRIKE_OFFSET: Final[float] = 0.0035


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("v143/legacy/lmm_swaption")


@pytest.fixture(autouse=True)
def _pinned_evaluation_date() -> Iterator[None]:  # pyright: ignore[reportUnusedFunction]
    """probe.cpp ``makeIndex``: evaluationDate = TARGET().adjust(4-Sep-2005)."""
    settings = ObservableSettings()
    previous = settings.evaluation_date
    settings.evaluation_date = Euribor6M(None).fixing_calendar().adjust(ANCHOR)
    try:
        yield
    finally:
        settings.evaluation_date = previous


def _assert_close(actual: Any, expected: Any, label: str) -> None:
    act = np.asarray(actual, dtype=np.float64)
    exp = np.asarray(expected, dtype=np.float64)
    assert act.shape == exp.shape, f"{label}: shape {act.shape} != {exp.shape}"
    for idx, (a, e) in enumerate(zip(act.ravel(), exp.ravel(), strict=True)):
        tight(float(a), float(e), reason=f"{label}[{idx}]")


class _Setup:
    """The whole probe scenario, built once per test."""

    def __init__(self) -> None:
        day_counter = Actual360()
        today = Euribor6M(None).fixing_calendar().adjust(ANCHOR)
        ObservableSettings().evaluation_date = today
        spot = Euribor6M(None).fixing_calendar().advance(
            today, Euribor6M(None).fixing_days(), TimeUnit.Days
        )
        self.curve: YieldTermStructureProtocol = InterpolatedZeroCurve(
            [spot, Date.from_ymd(4, Month.September, 2011)], [0.04, 0.08], day_counter
        )
        self.index = Euribor6M(self.curve)
        self.process = LiborForwardModelProcess(SIZE, self.index)
        self.corr_model = LmExponentialCorrelationModel(SIZE, CORR_RHO)
        self.vola_model = LmLinearExponentialVolatilityModel(
            self.process.fixing_times(), VOL_A, VOL_B, VOL_C, VOL_D
        )
        self.process.set_covar_param(
            LfmCovarianceProxy(self.vola_model, self.corr_model)
        )
        self.model = LiborForwardModel(self.process, self.vola_model, self.corr_model)

    def schedule(self, i: int, j: int) -> Schedule:
        settlement = self.curve.reference_date()
        fwd_start = settlement + Period(6 * i, TimeUnit.Months)
        fwd_maturity = fwd_start + Period(6 * j, TimeUnit.Months)
        convention = self.index.business_day_convention()
        return Schedule.from_rule(
            fwd_start,
            fwd_maturity,
            self.index.tenor(),
            self.index.fixing_calendar(),
            convention,
            convention,
            DateGeneration.Forward,
            False,
        )

    def swap(self, i: int, j: int, rate: float, swap_type: SwapType, spread: float = 0.0) -> VanillaSwap:
        schedule = self.schedule(i, j)
        swap = VanillaSwap(
            swap_type,
            1.0,
            schedule,
            rate,
            self.curve.day_counter(),
            schedule,
            self.index,
            spread,
            self.index.day_counter(),
        )
        swap.set_pricing_engine(DiscountingSwapEngine(self.curve))
        return swap


@pytest.fixture
def setup() -> _Setup:
    return _Setup()


# --- setup echo ---------------------------------------------------------------


def test_setup_matches_the_probe(cpp: dict[str, Any], setup: _Setup) -> None:
    """If the market setup drifts, every other expectation below is meaningless."""
    ref = cpp["setup"]
    assert setup.process.size() == ref["size"] == SIZE
    _assert_close(setup.process.initial_values(), ref["initial_values"], "setup.initial")
    _assert_close(setup.process.fixing_times(), ref["fixing_times"], "setup.fixing_times")
    assert [d.serial_number() for d in setup.process.fixing_dates()] == ref[
        "fixing_date_serials"
    ]
    _assert_close(
        setup.process.accrual_start_times(), ref["accrual_start_times"], "setup.ast"
    )
    _assert_close(setup.process.accrual_end_times(), ref["accrual_end_times"], "setup.aet")


def test_model_arguments_concatenate_both_sub_models(
    cpp: dict[str, Any], setup: _Setup
) -> None:
    """4 volatility parameters then 1 correlation parameter, in that order."""
    ref = cpp["setup"]
    assert len(setup.model.params()) == ref["n_model_params"] == 5
    _assert_close(setup.model.params(), ref["model_params"], "setup.model_params")
    assert list(setup.model.params()) == [VOL_A, VOL_B, VOL_C, VOL_D, CORR_RHO]


# --- S_0 ----------------------------------------------------------------------


def test_s_0_matches_cpp_over_the_whole_triangle(
    cpp: dict[str, Any], setup: _Setup
) -> None:
    ref = cpp["s_0"]
    got = [
        setup.model.s_0(alpha, beta)
        for alpha, beta in zip(ref["alpha"], ref["beta"], strict=True)
    ]
    _assert_close(got, ref["value"], "s_0")


def test_s_0_reproduces_the_fair_forward_swap_rate(
    cpp: dict[str, Any], setup: _Setup
) -> None:
    """C++ test-suite testSwaptionPricing: ``S_0(i-1, i+j-1)`` must equal
    ``VanillaSwap::fairRate`` of the matching forward swap.

    Tolerance: the C++ test uses 1e-12 with at-par coupons; both the model and
    the swap read the same curve, so the two routes differ only by the order of
    the arithmetic.
    """
    ref = cpp["forward_swap_fair_rates"]
    fair_from_swaps: list[float] = []
    fair_from_model: list[float] = []
    for raw_i, raw_j in zip(ref["i"], ref["j"], strict=True):
        i, j = int(raw_i), int(raw_j)
        fair_from_swaps.append(setup.swap(i, j, 0.0404, SwapType.Receiver).fair_rate())
        fair_from_model.append(setup.model.s_0(i - 1, i + j - 1))
    _assert_close(fair_from_swaps, ref["fair_rate"], "fair_rate")
    for a, e in zip(fair_from_model, ref["fair_rate"], strict=True):
        tight(a, e, reason="s_0 vs fairRate")


def test_s_0_rejects_a_non_increasing_index_pair(setup: _Setup) -> None:
    with pytest.raises(LibraryException, match="alpha needs to be smaller than beta"):
        setup.model.s_0(3, 3)


# --- AffineModel surface ------------------------------------------------------


def test_discount_and_discount_bond_delegate_to_the_curve(
    cpp: dict[str, Any], setup: _Setup
) -> None:
    """Both are "meaningless within this context" in C++ but still implemented:
    ``discount_bond`` ignores ``now`` and ``factors`` entirely.
    """
    ref = cpp["affine"]
    _assert_close([setup.model.discount(t) for t in ref["times"]], ref["discount"], "discount")
    factors = np.full(2, 0.7)
    _assert_close(
        [setup.model.discount_bond(0.3, t, factors) for t in ref["times"]],
        ref["discount_bond"],
        "discount_bond",
    )
    # ... and the factors really are ignored
    assert setup.model.discount_bond(0.3, 1.25, factors) == setup.model.discount_bond(
        2.0, 1.25, np.full(4, -9.0)
    )


def test_discount_bond_option_matches_cpp_for_both_option_types(
    cpp: dict[str, Any], setup: _Setup
) -> None:
    """The exact caplet price. Call and Put at four strikes and four maturities
    along the accrual grid; a single strike would not separate the cap-rate
    conversion from the type flip.
    """
    ref = cpp["discount_bond_option"]
    ast = setup.process.accrual_start_times()
    aet = setup.process.accrual_end_times()
    indices = [int(i) for i in ref["index"]]
    strikes = [float(k) for k in ref["strike"]]
    calls = [
        setup.model.discount_bond_option(OptionType.Call, k, ast[i], aet[i])
        for i, k in zip(indices, strikes, strict=True)
    ]
    puts = [
        setup.model.discount_bond_option(OptionType.Put, k, ast[i], aet[i])
        for i, k in zip(indices, strikes, strict=True)
    ]
    _assert_close(calls, ref["call"], "discount_bond_option.call")
    _assert_close(puts, ref["put"], "discount_bond_option.put")


def test_discount_bond_option_rejects_a_maturity_off_the_process_grid(
    setup: _Setup,
) -> None:
    ast = setup.process.accrual_start_times()
    aet = setup.process.accrual_end_times()
    with pytest.raises(LibraryException, match="does not fit to the process"):
        setup.model.discount_bond_option(OptionType.Call, 0.98, ast[-1] + 1.0, aet[-1])
    with pytest.raises(LibraryException, match="irregular fixings"):
        setup.model.discount_bond_option(
            OptionType.Call, 0.98, 0.5 * (ast[2] + ast[3]), aet[3]
        )


# --- Rebonato swaption volatility matrix --------------------------------------


def test_swaption_volatility_matrix_matches_cpp(
    cpp: dict[str, Any], setup: _Setup
) -> None:
    ref = cpp["swaption_vol_matrix"]
    matrix = setup.model.get_swaption_volatility_matrix()

    assert matrix.reference_date().serial_number() == ref["reference_date_serial"]
    assert [d.serial_number() for d in matrix.option_dates()] == ref["option_date_serials"]
    _assert_close(matrix.option_times(), ref["option_times"], "vol.option_times")
    _assert_close(matrix.swap_lengths(), ref["swap_lengths"], "vol.swap_lengths")
    assert [p.length for p in matrix.swap_tenors()] == ref["swap_tenor_lengths_months"]
    _assert_close(matrix.volatilities(), ref["volatilities"], "vol.volatilities")


def test_swaption_volatility_read_back_matches_cpp_on_and_off_pillar(
    cpp: dict[str, Any], setup: _Setup
) -> None:
    """This — not the raw matrix — is what LfmSwaptionEngine consults."""
    ref = cpp["swaption_vol_matrix"]
    matrix = setup.model.get_swaption_volatility_matrix()
    got = [
        matrix.volatility(t, length, 0.04, True)
        for t, length in zip(
            ref["query_option_times"], ref["query_swap_lengths"], strict=True
        )
    ]
    _assert_close(got, ref["query_volatilities"], "vol.query")


def test_swaption_volatility_matrix_is_cached(cpp: dict[str, Any], setup: _Setup) -> None:
    assert cpp["swaption_vol_matrix_is_cached"] == 1
    first = setup.model.get_swaption_volatility_matrix()
    assert setup.model.get_swaption_volatility_matrix() is first


# --- LfmSwaptionEngine --------------------------------------------------------


def test_lfm_swaption_engine_matches_cpp(cpp: dict[str, Any], setup: _Setup) -> None:
    """Payer, receiver and at-the-money over the co-terminal grid.

    Payer/receiver differ only by the Black option type, and the ATM leg pins
    the ``fixedRate == fairRate`` case where the intrinsic term vanishes.
    """
    ref = cpp["lfm_swaption_engine"]
    engine = LfmSwaptionEngine(setup.model, setup.curve)

    payer: list[float] = []
    receiver: list[float] = []
    atm: list[float] = []
    for raw_i, raw_j, raw_fair in zip(ref["i"], ref["j"], ref["fair_rate"], strict=True):
        i, j, fair = int(raw_i), int(raw_j), float(raw_fair)
        exercise = EuropeanExercise(setup.process.fixing_dates()[i])
        struck = fair + STRIKE_OFFSET
        for bucket, swap_type, rate in (
            (payer, SwapType.Payer, struck),
            (receiver, SwapType.Receiver, struck),
            (atm, SwapType.Payer, fair),
        ):
            swaption = Swaption(setup.swap(i, j, rate, swap_type), exercise)
            swaption.set_pricing_engine(engine)
            bucket.append(swaption.npv())

    _assert_close(payer, ref["payer_npv"], "engine.payer")
    _assert_close(receiver, ref["receiver_npv"], "engine.receiver")
    _assert_close(atm, ref["atm_payer_npv"], "engine.atm")


def test_lfm_swaption_engine_applies_the_spread_correction(
    cpp: dict[str, Any], setup: _Setup
) -> None:
    """``spread != 0`` shifts BOTH the struck and the fair rate by
    ``spread * |floatingLegBPS / fixedLegBPS|`` — otherwise dead code.
    """
    ref = cpp["lfm_swaption_with_spread"]
    engine = LfmSwaptionEngine(setup.model, setup.curve)
    swap = setup.swap(2, 3, 0.0425, SwapType.Payer, spread=0.0017)
    swaption = Swaption(swap, EuropeanExercise(setup.process.fixing_dates()[2]))
    swaption.set_pricing_engine(engine)

    tight(swap.spread(), ref["spread"], reason="spread")
    tight(swap.fixed_rate(), ref["fixed_rate"], reason="fixed_rate")
    tight(swap.fair_rate(), ref["fair_rate"], reason="fair_rate")
    tight(swap.fixed_leg_bps(), ref["fixed_leg_bps"], reason="fixed_leg_bps")
    tight(swap.floating_leg_bps(), ref["floating_leg_bps"], reason="floating_leg_bps")
    tight(swaption.npv(), ref["npv"], reason="spread swaption npv")


def test_lfm_swaption_engine_rejects_a_par_yield_cash_settled_swaption(
    cpp: dict[str, Any], setup: _Setup
) -> None:
    assert cpp["lfm_swaption_par_yield_rejected"] is True
    engine = LfmSwaptionEngine(setup.model, setup.curve)
    swaption = Swaption(
        setup.swap(2, 2, 0.0425, SwapType.Payer),
        EuropeanExercise(setup.process.fixing_dates()[2]),
        SettlementType.Cash,
        SettlementMethod.ParYieldCurve,
    )
    swaption.set_pricing_engine(engine)
    with pytest.raises(LibraryException, match="ParYieldCurve"):
        swaption.npv()


def test_lfm_swaption_engine_inspectors(setup: _Setup) -> None:
    engine = LfmSwaptionEngine(setup.model, setup.curve)
    assert engine.model() is setup.model
    assert engine.discount_curve() is setup.curve


# --- setParams ----------------------------------------------------------------


def test_set_params_splits_the_array_between_the_two_sub_models(
    cpp: dict[str, Any], setup: _Setup
) -> None:
    """The split point is ``len(volatility_model.params())``. Getting it wrong
    is silent — both halves are still "some numbers" — so both sub-models are
    read back.
    """
    ref = cpp["after_set_params"]
    setup.model.set_params(np.asarray([0.35, 1.10, 0.145, 0.0175, 0.28]))

    _assert_close(setup.model.params(), ref["params"], "after_set.params")
    _assert_close(
        [p(0.0) for p in setup.vola_model.params()],
        ref["vol_model_params"],
        "after_set.vol_params",
    )
    _assert_close(
        [p(0.0) for p in setup.corr_model.params()],
        ref["corr_model_params"],
        "after_set.corr_params",
    )


def test_set_params_leaves_s_0_unchanged(cpp: dict[str, Any], setup: _Setup) -> None:
    """S_0 depends only on the initial forward curve, never on the vol/corr
    parameters — worth pinning because it separates the two.
    """
    ref = cpp["after_set_params"]
    before = [
        setup.model.s_0(a, b) for a in range(SIZE - 1) for b in range(a + 1, SIZE)
    ]
    setup.model.set_params(np.asarray([0.35, 1.10, 0.145, 0.0175, 0.28]))
    after = [setup.model.s_0(a, b) for a in range(SIZE - 1) for b in range(a + 1, SIZE)]
    assert before == after
    _assert_close(after, ref["s_0"], "after_set.s_0")


def test_set_params_invalidates_the_cached_swaption_volatility_matrix(
    cpp: dict[str, Any], setup: _Setup
) -> None:
    ref = cpp["after_set_params"]
    assert ref["cache_was_invalidated"] == 1

    first = setup.model.get_swaption_volatility_matrix()
    setup.model.set_params(np.asarray([0.35, 1.10, 0.145, 0.0175, 0.28]))
    second = setup.model.get_swaption_volatility_matrix()

    assert second is not first
    _assert_close(
        second.volatilities(), ref["swaption_volatilities"], "after_set.swaption_vols"
    )
    assert not np.allclose(second.volatilities(), first.volatilities())
