"""Cross-validate the v1.43 ``ql/experimental/barrieroption`` wave.

Probe source: migration-harness/cpp/probes/v143_experimental_barrier/probe.cpp
Reference:    migration-harness/references/v143/experimental/barrier.json

Covers the eight ``experimental/barrieroption`` classes ported in this wave —
``DiscretizedDoubleBarrierOption``,
``DiscretizedDermanKaniDoubleBarrierOption``, ``DoubleBarrierPathPricer``,
``MCDoubleBarrierEngine``, ``MakeMCDoubleBarrierEngine``,
``QuantoDoubleBarrierOption``, ``SuoWangDoubleBarrierEngine`` and
``PerturbativeBarrierOptionEngine`` — plus the four supporting classes they
dragged in: ``DiscretizedVanillaOption``, ``QuantoOptionResults``,
``QuantoEngine``, ``QuantoTermStructure``, and the ``RngTraits`` policy alias.

.. rubric:: How the discretized assets are pinned

Pinning only the converged NPV would hide a wrong barrier adjustment behind
binomial convergence, so the probe drives each asset one lattice slice at a
time and emits the whole post-``adjustValues`` value vector, flattened. The
tests replay exactly that walk, so a wrong ``check_barrier`` branch or a wrong
Derman-Kani interpolation shows up in the first slice rather than in the
fourth decimal of a price.

.. rubric:: Why Monte Carlo is pinned exactly and not as a statistical band

``MCDoubleBarrierEngine`` is deterministic once the seed is fixed:
``PseudoRandom::make_sequence_generator(dimension, seed)`` is a MT19937 →
inverse-cumulative-normal stack that PQuantLib reproduces draw for draw. Every
MC assertion therefore pins the C++ NPV and error estimate at TIGHT; a broken
path pricer or a shifted RNG stream fails loudly instead of hiding inside a
confidence interval.

.. rubric:: The two assertions that are not TIGHT

``suowang_ki_put_wide`` is a knock-in whose value is built as
``european - barrierOut`` from two numbers near 4.70 whose difference is
1.92e-3 — roughly three and a half digits of cancellation.
:func:`_SUOWANG_CANCELLATION_ABS` derives the resulting bound; see its
docstring for the arithmetic. Everything else in this module is TIGHT.

.. rubric:: Two C++ defects reproduced verbatim

``PNTGND`` squares nothing where Genz's Fortran squares the numerator
(``std::pow(BA - R*BB, 0.5)`` for ``(BA-R*BB)**2``), which makes ``FT`` NaN
whenever ``BA - R*BB < 0``. The NaN propagates through the Kronrod rule into
``tvtl``'s final ``max(ZRO, min(TVT, ONE))``, and because both C++'s
``std::max``/``std::min`` and Python's ``max``/``min`` return their first
argument on an unordered comparison, the trivariate probability comes back as
exactly ``0`` in both languages. ``test_tvtl_adaptive_branch_is_nan_clamped``
pins that, and ``test_pntgnd_defect_produces_nan`` pins the NaN itself.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.exceptions import LibraryException
from pquantlib.exercise import (
    AmericanExercise,
    BermudanExercise,
    EuropeanExercise,
    Exercise,
)
from pquantlib.experimental.barrieroption import (
    perturbative_barrier_option_engine as _pert,
)
from pquantlib.experimental.barrieroption.discretized_double_barrier_option import (
    DiscretizedDermanKaniDoubleBarrierOption,
    DiscretizedDoubleBarrierOption,
)
from pquantlib.experimental.barrieroption.mc_double_barrier_engine import (
    DoubleBarrierPathPricer,
    MakeMCDoubleBarrierEngine,
    MCDoubleBarrierEngine,
)
from pquantlib.experimental.barrieroption.perturbative_barrier_option_engine import (
    PerturbativeBarrierOptionEngine,
)
from pquantlib.experimental.barrieroption.quanto_double_barrier_option import (
    QuantoDoubleBarrierOption,
)
from pquantlib.experimental.barrieroption.suo_wang_double_barrier_engine import (
    SuoWangDoubleBarrierEngine,
)
from pquantlib.instruments.barrier_option import BarrierOption, BarrierType
from pquantlib.instruments.double_barrier_option import (
    DoubleBarrierOption,
    DoubleBarrierOptionArguments,
    DoubleBarrierType,
)
from pquantlib.instruments.one_asset_option import OneAssetOptionResults
from pquantlib.instruments.quanto_option_results import QuantoOptionResults
from pquantlib.instruments.vanilla_option import VanillaOption
from pquantlib.math.randomnumbers.rng_traits import LowDiscrepancy, PseudoRandom
from pquantlib.methods.lattices.binomial_tree import CoxRossRubinstein
from pquantlib.methods.lattices.bsm_lattice import BlackScholesLattice
from pquantlib.methods.montecarlo.path import Path
from pquantlib.option import OptionArguments
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.payoffs import OptionType, PlainVanillaPayoff
from pquantlib.pricingengines.barrier.analytic_double_barrier_engine import (
    AnalyticDoubleBarrierEngine,
)
from pquantlib.pricingengines.generic_engine import GenericEngine
from pquantlib.pricingengines.mc_rng_traits import RngTraits
from pquantlib.pricingengines.quanto.quanto_engine import QuantoEngine
from pquantlib.pricingengines.vanilla.analytic_european_engine import (
    AnalyticEuropeanEngine,
)
from pquantlib.pricingengines.vanilla.discretized_vanilla_option import (
    DiscretizedVanillaOption,
)
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.volatility.equity_fx.black_constant_vol import (
    BlackConstantVol,
)
from pquantlib.termstructures.volatility.equity_fx.black_variance_curve import (
    BlackVarianceCurve,
)
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.termstructures.yield_.quanto_term_structure import QuantoTermStructure
from pquantlib.testing import reference_reader
from pquantlib.testing.tolerance import custom, tight
from pquantlib.time.calendars.null_calendar import NullCalendar
from pquantlib.time.compounding import Compounding
from pquantlib.time.date import Date
from pquantlib.time.frequency import Frequency
from pquantlib.time.month import Month
from pquantlib.time.time_grid import TimeGrid

# The probe's fixed evaluation date (probe.cpp: ``const Date TODAY``).
_TODAY: Date = Date.from_ymd(15, Month.May, 2023)

_ACT365: DayCounter = Actual365Fixed()
_ACT360: DayCounter = Actual360()


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("v143/experimental/barrier")


@pytest.fixture(autouse=True)
def pinned_evaluation_date() -> Iterator[None]:
    """Pin the global evaluation date to the probe's ``TODAY`` and restore it."""
    settings = ObservableSettings()
    previous = settings.evaluation_date
    settings.evaluation_date = _TODAY
    try:
        yield
    finally:
        settings.evaluation_date = previous


# ---------------------------------------------------------------------------
# Reaching the perturbative engine's module-private numerics
# ---------------------------------------------------------------------------
#
# ``PHID``, ``ND2``, ``tvtl`` and friends live in an anonymous namespace inside
# perturbativebarrieroptionengine.cpp (:50-90), so they are not linkable; the
# probe reaches them by ``#include``-ing the translation unit itself. The port
# spells that namespace as module-level underscore names, and the tests reach
# them through the module dict — the same deliberate breach, made explicit —
# binding each to its signature so the call sites stay type-checked.

_Fn2 = Callable[[float, float], float]
_Fn3 = Callable[[float, float, float], float]
_Fn5 = Callable[[float, float, float, float, float], float]
_Fn6 = Callable[[float, float, float, float, float, float], float]
_Fn7 = Callable[[float, float, float, float, float, float, float], float]

_NUMERICS: Mapping[str, Any] = vars(_pert)

_sign: _Fn2 = _NUMERICS["_sign"]
_phid: Callable[[float], float] = _NUMERICS["_phid"]
_nd2: _Fn3 = _NUMERICS["_nd2"]
_studnt: Callable[[int, float], float] = _NUMERICS["_studnt"]
_bvtl: Callable[[int, float, float, float], float] = _NUMERICS["_bvtl"]
_ff: _Fn5 = _NUMERICS["_ff"]
_v: _Fn5 = _NUMERICS["_v"]
_llold: _Fn6 = _NUMERICS["_llold"]
_tvtl: Callable[[int, list[float], list[float], float], float] = _NUMERICS["_tvtl"]
_derivn3: Callable[[list[float], list[float], int], float] = _NUMERICS["_derivn3"]
_pntgnd: Callable[
    [int, float, float, float, float, float, float, float], float
] = _NUMERICS["_pntgnd"]
_dvv: _Fn6 = _NUMERICS["_dvv"]
_dff: _Fn6 = _NUMERICS["_dff"]
_dll: _Fn7 = _NUMERICS["_dll"]
_ddvv: _Fn6 = _NUMERICS["_ddvv"]
_ddff: _Fn6 = _NUMERICS["_ddff"]
_ddll: _Fn7 = _NUMERICS["_ddll"]
_barrier_upd: Callable[..., float] = _NUMERICS["_barrier_upd"]


# ---------------------------------------------------------------------------
# market-data helpers — one-to-one with probe.cpp's ``make_bsm``
# ---------------------------------------------------------------------------


def _flat_vol(reference_date: Date, vol: float, dc: DayCounter) -> BlackConstantVol:
    return BlackConstantVol(
        reference_date=reference_date,
        calendar=NullCalendar(),
        day_counter=dc,
        volatility=vol,
    )


def _make_bsm(
    spot: float, q: float, r: float, vol: float, dc: DayCounter
) -> GeneralizedBlackScholesProcess:
    """# C++ parity: ``make_bsm`` (probe.cpp) — a ``BlackScholesMertonProcess``."""
    return GeneralizedBlackScholesProcess(
        x0=SimpleQuote(spot),
        dividend_ts=FlatForward.from_rate(_TODAY, q, dc),
        risk_free_ts=FlatForward.from_rate(_TODAY, r, dc),
        black_vol_ts=_flat_vol(_TODAY, vol, dc),
    )


def _zero(ts: Any, d: Date, dc: DayCounter) -> float:
    return float(
        ts.zero_rate(
            d,
            compounding=Compounding.Continuous,
            frequency=Frequency.NoFrequency,
            result_day_counter=dc,
        ).rate()
    )


def _binomial_lattice(
    process: GeneralizedBlackScholesProcess,
    ex_date: Date,
    steps: int,
    strike: float,
) -> tuple[BlackScholesLattice, TimeGrid, float]:
    """Rebuild the flat-coefficient CRR lattice the binomial engine uses.

    # C++ parity: ``BinomialDoubleBarrierEngine::calculate``
    # (binomialdoublebarrierengine.hpp:71-118), replayed verbatim by
    # ``run_discretized`` in probe.cpp.
    """
    spot = process.state_variable().value()
    vol = process.black_volatility().black_vol(ex_date, spot)
    r = _zero(process.risk_free_rate(), ex_date, _ACT365)
    q = _zero(process.dividend_yield(), ex_date, _ACT365)
    ref_date = process.risk_free_rate().reference_date()

    flat = GeneralizedBlackScholesProcess(
        x0=process.state_variable(),
        dividend_ts=FlatForward.from_rate(ref_date, q, _ACT365),
        risk_free_ts=FlatForward.from_rate(ref_date, r, _ACT365),
        black_vol_ts=_flat_vol(ref_date, vol, _ACT365),
    )
    maturity = _ACT365.year_fraction(ref_date, ex_date)
    grid = TimeGrid.regular(maturity, steps)
    tree = CoxRossRubinstein(flat, maturity, steps, strike)
    return BlackScholesLattice(tree, r, maturity, steps), grid, maturity


def _double_barrier_arguments(
    barrier_type: DoubleBarrierType,
    lo: float,
    hi: float,
    rebate: float,
    payoff: PlainVanillaPayoff,
    ex_date: Date,
) -> DoubleBarrierOptionArguments:
    option = DoubleBarrierOption(
        barrier_type, lo, hi, rebate, payoff, EuropeanExercise(ex_date)
    )
    args = DoubleBarrierOptionArguments()
    option.setup_arguments(args)
    args.validate()
    return args


def _assert_vector(
    actual: Sequence[float] | npt.NDArray[np.float64],
    expected: Sequence[float],
    *,
    label: str,
) -> None:
    actual_list = [float(x) for x in actual]
    assert len(actual_list) == len(expected), (
        f"{label}: length {len(actual_list)} != C++ {len(expected)}"
    )
    for i, (got, want) in enumerate(zip(actual_list, expected, strict=True)):
        tight(got, float(want), reason=f"{label}[{i}]")


# ---------------------------------------------------------------------------
# DiscretizedDoubleBarrierOption / DiscretizedDermanKaniDoubleBarrierOption
# ---------------------------------------------------------------------------

# (tag, barrier type, lo, hi, rebate, option type, spot, strike, q, r, vol,
#  days, steps) — probe.cpp ``dcases``.
_DISCRETIZED_PLAIN: list[tuple[str, DoubleBarrierType, float, float, float, OptionType]] = [
    ("ko_call", DoubleBarrierType.KnockOut, 90.0, 110.0, 0.0, OptionType.Call),
    ("ki_call", DoubleBarrierType.KnockIn, 90.0, 110.0, 0.0, OptionType.Call),
    ("kiko_put", DoubleBarrierType.KIKO, 90.0, 110.0, 2.0, OptionType.Put),
    ("koki_put", DoubleBarrierType.KOKI, 90.0, 110.0, 2.0, OptionType.Put),
]
_DISCRETIZED_DK: list[tuple[str, DoubleBarrierType, float, float, float, OptionType]] = [
    ("dk_ko_call", DoubleBarrierType.KnockOut, 90.0, 110.0, 0.0, OptionType.Call),
    ("dk_ki_call", DoubleBarrierType.KnockIn, 90.0, 110.0, 0.0, OptionType.Call),
    ("dk_ko_rebate", DoubleBarrierType.KnockOut, 90.0, 110.0, 4.0, OptionType.Put),
]

_DISCRETIZED_SPOT = 100.0
_DISCRETIZED_STRIKE = 100.0
_DISCRETIZED_Q = 0.02
_DISCRETIZED_R = 0.05
_DISCRETIZED_VOL = 0.25
_DISCRETIZED_DAYS = 180
_DISCRETIZED_STEPS = 12


class TestDiscretizedDoubleBarrierOption:
    """# C++ parity: ``DiscretizedDoubleBarrierOption``
    # (discretizeddoublebarrieroption.hpp:39-65 + .cpp:25-152)."""

    @pytest.mark.parametrize(
        ("tag", "barrier_type", "lo", "hi", "rebate", "option_type"),
        _DISCRETIZED_PLAIN,
        ids=[c[0] for c in _DISCRETIZED_PLAIN],
    )
    def test_lattice_walk_matches_cpp(
        self,
        cpp: dict[str, Any],
        tag: str,
        barrier_type: DoubleBarrierType,
        lo: float,
        hi: float,
        rebate: float,
        option_type: OptionType,
    ) -> None:
        """Every slice of the rollback, plus the contained vanilla's slices."""
        prefix = f"discretized_{tag}"
        process = _make_bsm(
            _DISCRETIZED_SPOT, _DISCRETIZED_Q, _DISCRETIZED_R, _DISCRETIZED_VOL, _ACT365
        )
        ex_date = _TODAY + _DISCRETIZED_DAYS
        payoff = PlainVanillaPayoff(option_type, _DISCRETIZED_STRIKE)
        args = _double_barrier_arguments(
            barrier_type, lo, hi, rebate, payoff, ex_date
        )
        lattice, grid, maturity = _binomial_lattice(
            process, ex_date, _DISCRETIZED_STEPS, payoff.strike()
        )

        tight(maturity, cpp[prefix + "_maturity"], reason="maturity")
        _assert_vector(list(grid), cpp[prefix + "_grid"], label="grid")
        underlyings = [
            lattice.underlying(i, j)
            for i in range(_DISCRETIZED_STEPS + 1)
            for j in range(i + 1)
        ]
        _assert_vector(
            underlyings, cpp[prefix + "_underlyings_flat"], label="underlyings"
        )

        asset = DiscretizedDoubleBarrierOption(args, process, grid)
        asset.initialize(lattice, maturity)
        values = list(asset.values)
        vanilla = list(asset.vanilla())
        for i in range(_DISCRETIZED_STEPS, 0, -1):
            asset.rollback(grid[i - 1])
            values += list(asset.values)
            vanilla += list(asset.vanilla())

        _assert_vector(values, cpp[prefix + "_values_flat"], label="values")
        # For a pure knock-out ``postAdjustValuesImpl`` never rolls the
        # contained vanilla back (.cpp:50-51), so its slice stays at the
        # maturity width for the whole walk — the reference vector is
        # correspondingly longer. Pinning it catches a port that rolls it
        # unconditionally.
        _assert_vector(vanilla, cpp[prefix + "_vanilla_flat"], label="vanilla")
        tight(
            asset.present_value(),
            cpp[prefix + "_present_value"],
            reason="present value",
        )
        _assert_vector(
            asset.mandatory_times(),
            cpp[prefix + "_mandatory_times"],
            label="mandatory times",
        )

    def test_converged_present_value(self, cpp: dict[str, Any]) -> None:
        """300 steps — the configuration the C++ test suite prices at."""
        process = _make_bsm(
            _DISCRETIZED_SPOT, _DISCRETIZED_Q, _DISCRETIZED_R, _DISCRETIZED_VOL, _ACT365
        )
        ex_date = _TODAY + _DISCRETIZED_DAYS
        payoff = PlainVanillaPayoff(OptionType.Call, _DISCRETIZED_STRIKE)
        args = _double_barrier_arguments(
            DoubleBarrierType.KnockOut, 90.0, 110.0, 0.0, payoff, ex_date
        )
        lattice, grid, maturity = _binomial_lattice(
            process, ex_date, 300, payoff.strike()
        )
        asset = DiscretizedDoubleBarrierOption(args, process, grid)
        asset.initialize(lattice, maturity)
        asset.rollback(0.0)
        tight(asset.present_value(), cpp["discretized_conv_std_present_value"])

    def test_empty_exercise_dates_rejected(self) -> None:
        """# C++ parity: ``QL_REQUIRE(!args.exercise->dates().empty(), ...)``
        # (discretizeddoublebarrieroption.cpp:30).

        The check sits in the ctor *body*, so C++ has already built the
        contained ``DiscretizedVanillaOption`` (a member initialiser) by the
        time it fires; an exercise that exists but carries no dates is the only
        input that reaches it, and the port keeps that ordering.
        """
        process = _make_bsm(100.0, 0.02, 0.05, 0.25, _ACT365)
        args = DoubleBarrierOptionArguments()
        args.payoff = PlainVanillaPayoff(OptionType.Call, 100.0)
        args.exercise = Exercise(Exercise.Type.European)
        assert args.exercise.dates() == []
        args.barrier_type = DoubleBarrierType.KnockOut
        args.barrier_lo = 90.0
        args.barrier_hi = 110.0
        args.rebate = 0.0
        with pytest.raises(LibraryException, match="specify at least one stopping date"):
            DiscretizedDoubleBarrierOption(args, process, TimeGrid.regular(1.0, 4))


class TestCheckBarrier:
    """``checkBarrier`` driven on a synthetic grid, independent of any lattice.

    # C++ parity: ``DiscretizedDoubleBarrierOption::checkBarrier``
    # (discretizeddoublebarrieroption.cpp:57-152); probe block
    # ``run_check_barrier``.
    """

    @pytest.mark.parametrize(
        ("barrier_type", "name"),
        [
            (DoubleBarrierType.KnockIn, "KnockIn"),
            (DoubleBarrierType.KnockOut, "KnockOut"),
            (DoubleBarrierType.KIKO, "KIKO"),
            (DoubleBarrierType.KOKI, "KOKI"),
        ],
        ids=["KnockIn", "KnockOut", "KIKO", "KOKI"],
    )
    def test_branch_table(
        self, cpp: dict[str, Any], barrier_type: DoubleBarrierType, name: str
    ) -> None:
        process = _make_bsm(100.0, 0.02, 0.05, 0.25, _ACT365)
        ex_date = _TODAY + 180
        payoff = PlainVanillaPayoff(OptionType.Call, 100.0)
        maturity = _ACT365.year_fraction(_TODAY, ex_date)
        steps = 8
        grid = TimeGrid.regular(maturity, steps)
        args = _double_barrier_arguments(
            barrier_type, 90.0, 110.0, 3.0, payoff, ex_date
        )

        asset = DiscretizedDoubleBarrierOption(args, process, grid)
        tree = CoxRossRubinstein(process, maturity, steps, payoff.strike())
        asset.initialize(BlackScholesLattice(tree, 0.05, maturity, steps), maturity)

        prefix = f"check_barrier_{name}"
        levels = np.asarray(cpp["check_barrier_grid_levels"], dtype=np.float64)
        optvalues = np.full(levels.size, 1.5, dtype=np.float64)
        asset.check_barrier(optvalues, levels)
        _assert_vector(optvalues, cpp[prefix + "_at_maturity"], label="at maturity")
        _assert_vector(
            asset.vanilla(),
            cpp[prefix + "_vanilla_at_maturity"],
            label="vanilla at maturity",
        )

        # One step back: `endTime` and `stoppingTime` are both false. The
        # probe uses an eight-level grid here because the contained vanilla
        # has only eight nodes at step 7 and the knocked-in branches index it
        # with the same j as `optvalues`.
        asset.rollback(grid[steps - 1])
        levels8 = np.asarray(
            cpp["check_barrier_grid_levels_one_step_back"], dtype=np.float64
        )
        optvalues2 = np.full(levels8.size, 1.5, dtype=np.float64)
        asset.check_barrier(optvalues2, levels8)
        _assert_vector(
            optvalues2, cpp[prefix + "_one_step_back"], label="one step back"
        )
        _assert_vector(
            asset.vanilla(),
            cpp[prefix + "_vanilla_one_step_back"],
            label="vanilla one step back",
        )
        tight(asset.time, cpp[prefix + "_time_one_step_back"], reason="time")


class TestDiscretizedDermanKaniDoubleBarrierOption:
    """# C++ parity: ``DiscretizedDermanKaniDoubleBarrierOption``
    # (discretizeddoublebarrieroption.hpp:76-92 + .cpp:156-230)."""

    @pytest.mark.parametrize(
        ("tag", "barrier_type", "lo", "hi", "rebate", "option_type"),
        _DISCRETIZED_DK,
        ids=[c[0] for c in _DISCRETIZED_DK],
    )
    def test_lattice_walk_matches_cpp(
        self,
        cpp: dict[str, Any],
        tag: str,
        barrier_type: DoubleBarrierType,
        lo: float,
        hi: float,
        rebate: float,
        option_type: OptionType,
    ) -> None:
        prefix = f"discretized_{tag}"
        process = _make_bsm(
            _DISCRETIZED_SPOT, _DISCRETIZED_Q, _DISCRETIZED_R, _DISCRETIZED_VOL, _ACT365
        )
        ex_date = _TODAY + _DISCRETIZED_DAYS
        payoff = PlainVanillaPayoff(option_type, _DISCRETIZED_STRIKE)
        args = _double_barrier_arguments(
            barrier_type, lo, hi, rebate, payoff, ex_date
        )
        lattice, grid, maturity = _binomial_lattice(
            process, ex_date, _DISCRETIZED_STEPS, payoff.strike()
        )

        asset = DiscretizedDermanKaniDoubleBarrierOption(args, process, grid)
        asset.initialize(lattice, maturity)
        values = list(asset.values)
        for i in range(_DISCRETIZED_STEPS, 0, -1):
            asset.rollback(grid[i - 1])
            values += list(asset.values)

        _assert_vector(values, cpp[prefix + "_values_flat"], label="values")
        tight(
            asset.present_value(),
            cpp[prefix + "_present_value"],
            reason="present value",
        )
        _assert_vector(
            asset.mandatory_times(),
            cpp[prefix + "_mandatory_times"],
            label="mandatory times",
        )

    def test_converged_present_value(self, cpp: dict[str, Any]) -> None:
        process = _make_bsm(
            _DISCRETIZED_SPOT, _DISCRETIZED_Q, _DISCRETIZED_R, _DISCRETIZED_VOL, _ACT365
        )
        ex_date = _TODAY + _DISCRETIZED_DAYS
        payoff = PlainVanillaPayoff(OptionType.Call, _DISCRETIZED_STRIKE)
        args = _double_barrier_arguments(
            DoubleBarrierType.KnockOut, 90.0, 110.0, 0.0, payoff, ex_date
        )
        lattice, grid, maturity = _binomial_lattice(
            process, ex_date, 300, payoff.strike()
        )
        asset = DiscretizedDermanKaniDoubleBarrierOption(args, process, grid)
        asset.initialize(lattice, maturity)
        asset.rollback(0.0)
        tight(asset.present_value(), cpp["discretized_conv_dk_present_value"])

    @pytest.mark.parametrize(
        "barrier_type", [DoubleBarrierType.KIKO, DoubleBarrierType.KOKI]
    )
    def test_kiko_koki_unsupported(self, barrier_type: DoubleBarrierType) -> None:
        """# C++ parity: ``QL_FAIL("unsupported barrier type")``
        # (discretizeddoublebarrieroption.cpp:227) — ``adjustBarrier`` only
        # implements the two pure barrier types."""
        process = _make_bsm(100.0, 0.02, 0.05, 0.25, _ACT365)
        ex_date = _TODAY + 180
        payoff = PlainVanillaPayoff(OptionType.Call, 100.0)
        args = _double_barrier_arguments(
            barrier_type, 90.0, 110.0, 2.0, payoff, ex_date
        )
        lattice, grid, maturity = _binomial_lattice(process, ex_date, 8, 100.0)
        asset = DiscretizedDermanKaniDoubleBarrierOption(args, process, grid)
        with pytest.raises(LibraryException, match="unsupported barrier type"):
            asset.initialize(lattice, maturity)


# ---------------------------------------------------------------------------
# DiscretizedVanillaOption
# ---------------------------------------------------------------------------


class TestDiscretizedVanillaOption:
    """# C++ parity: ``DiscretizedVanillaOption``
    # (discretizedvanillaoption.hpp:34-51 + .cpp:26-77).

    The barrier asset only ever holds one under a European exercise, so the
    American and Bermudan arms of ``postAdjustValuesImpl`` are driven directly
    here (probe block ``run_discretized_vanilla``).
    """

    _STEPS = 8

    def _setup(
        self, cpp: dict[str, Any]
    ) -> tuple[GeneralizedBlackScholesProcess, Date, PlainVanillaPayoff, TimeGrid, float]:
        process = _make_bsm(100.0, 0.02, 0.05, 0.25, _ACT365)
        ex_date = _TODAY + 180
        maturity = _ACT365.year_fraction(_TODAY, ex_date)
        payoff = PlainVanillaPayoff(OptionType.Put, 100.0)
        grid = TimeGrid.regular(maturity, self._STEPS)
        tight(maturity, cpp["dvo_maturity"], reason="maturity")
        _assert_vector(list(grid), cpp["dvo_grid"], label="grid")
        return process, ex_date, payoff, grid, maturity

    def _exercise(self, cpp: dict[str, Any], tag: str, ex_date: Date) -> Any:
        if tag == "european":
            return EuropeanExercise(ex_date)
        if tag == "american":
            return AmericanExercise(_TODAY, ex_date)
        offsets = [
            int(s) - int(cpp["today_serial"]) for s in cpp["dvo_bermudan_date_serials"]
        ]
        return BermudanExercise([_TODAY + n for n in offsets])

    @pytest.mark.parametrize("tag", ["european", "american", "bermudan"])
    def test_lattice_walk_matches_cpp(self, cpp: dict[str, Any], tag: str) -> None:
        process, ex_date, payoff, grid, maturity = self._setup(cpp)
        option = VanillaOption(payoff, self._exercise(cpp, tag, ex_date))
        args = OptionArguments()
        option.setup_arguments(args)
        args.validate()

        tree = CoxRossRubinstein(process, maturity, self._STEPS, payoff.strike())
        lattice = BlackScholesLattice(tree, 0.05, maturity, self._STEPS)
        underlyings = [
            lattice.underlying(i, j)
            for i in range(self._STEPS + 1)
            for j in range(i + 1)
        ]
        _assert_vector(
            underlyings, cpp["dvo_underlyings_flat"], label="underlyings"
        )

        asset = DiscretizedVanillaOption(args, process, grid)
        asset.initialize(lattice, maturity)
        values = list(asset.values)
        for i in range(self._STEPS, 0, -1):
            asset.rollback(grid[i - 1])
            values += list(asset.values)

        _assert_vector(values, cpp[f"dvo_{tag}_values_flat"], label="values")
        tight(asset.present_value(), cpp[f"dvo_{tag}_present_value"], reason="pv")
        _assert_vector(
            asset.mandatory_times(),
            cpp[f"dvo_{tag}_mandatory_times"],
            label="mandatory times",
        )

    def test_missing_exercise_rejected(self) -> None:
        """Port-added guard with no C++ counterpart.

        ``DiscretizedVanillaOption``'s ctor goes straight to
        ``args.exercise->dates()`` (discretizedvanillaoption.cpp:31), so a null
        exercise is undefined behaviour in C++ rather than a diagnosable error.
        The port raises instead; there is nothing to cross-validate, only a
        crash to avoid.
        """
        process = _make_bsm(100.0, 0.02, 0.05, 0.25, _ACT365)
        args = OptionArguments()
        args.payoff = PlainVanillaPayoff(OptionType.Put, 100.0)
        args.exercise = None
        with pytest.raises(LibraryException, match="no exercise given"):
            DiscretizedVanillaOption(args, process, TimeGrid.regular(1.0, 4))

    def test_empty_exercise_dates_give_no_stopping_times(self) -> None:
        """# C++ parity: ``stoppingTimes_.resize(args.exercise->dates().size())``
        # with an empty date list is a no-op (discretizedvanillaoption.cpp:31)."""
        process = _make_bsm(100.0, 0.02, 0.05, 0.25, _ACT365)
        args = OptionArguments()
        args.payoff = PlainVanillaPayoff(OptionType.Put, 100.0)
        args.exercise = Exercise(Exercise.Type.European)
        asset = DiscretizedVanillaOption(args, process, TimeGrid.regular(1.0, 4))
        assert asset.mandatory_times() == []


# ---------------------------------------------------------------------------
# DoubleBarrierPathPricer
# ---------------------------------------------------------------------------

_PP_LO = 90.0
_PP_HI = 110.0
_PP_REBATE = 2.5
_PP_STRIKE = 100.0
_PP_TAGS = [
    "inside",
    "cross_hi",
    "cross_lo",
    "touch_first",
    "touch_last",
    "touch_zero",
]


class TestDoubleBarrierPathPricer:
    """# C++ parity: ``DoubleBarrierPathPricer``
    # (mcdoublebarrierengine.hpp:112-130 + .cpp:25-96)."""

    @staticmethod
    def _grid_and_discounts(cpp: dict[str, Any]) -> tuple[TimeGrid, list[float]]:
        grid = TimeGrid.regular(1.0, 5)
        _assert_vector(list(grid), cpp["pp_grid"], label="grid")
        discounts = [math.exp(-0.04 * grid[i]) for i in range(len(grid))]
        _assert_vector(discounts, cpp["pp_discounts"], label="discounts")
        return grid, discounts

    @pytest.mark.parametrize("tag", _PP_TAGS)
    @pytest.mark.parametrize(
        ("barrier_type", "barrier_name"),
        [
            (DoubleBarrierType.KnockOut, "KnockOut"),
            (DoubleBarrierType.KnockIn, "KnockIn"),
        ],
    )
    @pytest.mark.parametrize(
        ("option_type", "option_name"),
        [(OptionType.Call, "Call"), (OptionType.Put, "Put")],
    )
    def test_single_path(
        self,
        cpp: dict[str, Any],
        tag: str,
        barrier_type: DoubleBarrierType,
        barrier_name: str,
        option_type: OptionType,
        option_name: str,
    ) -> None:
        grid, discounts = self._grid_and_discounts(cpp)
        values = np.asarray(cpp["pp_path_" + tag], dtype=np.float64)
        pricer = DoubleBarrierPathPricer(
            barrier_type,
            _PP_LO,
            _PP_HI,
            _PP_REBATE,
            option_type,
            _PP_STRIKE,
            discounts,
        )
        tight(
            pricer(Path(grid, values)),
            cpp[f"pp_{tag}_{barrier_name}_{option_name}"],
            reason=f"{tag}/{barrier_name}/{option_name}",
        )

    def test_index_zero_is_never_scanned(self, cpp: dict[str, Any]) -> None:
        """# C++ parity note (defect, reproduced verbatim): the knock scan runs
        # ``for (i = 0; i < n-1; i++) { new_asset_price = path[i + 1]; ... }``
        # (mcdoublebarrierengine.cpp:56-58, 69-71), so ``path[0]`` is never
        # compared with either barrier. The ``touch_zero`` path starts at 90.0,
        # exactly on the low barrier, and still prices as if it never knocked.
        """
        grid, discounts = self._grid_and_discounts(cpp)
        values = np.asarray(cpp["pp_path_touch_zero"], dtype=np.float64)
        assert values[0] <= _PP_LO
        ko = DoubleBarrierPathPricer(
            DoubleBarrierType.KnockOut,
            _PP_LO,
            _PP_HI,
            _PP_REBATE,
            OptionType.Call,
            _PP_STRIKE,
            discounts,
        )
        # Knocked out at t=0 would have paid the rebate; C++ pays the payoff.
        priced = ko(Path(grid, values))
        tight(priced, cpp["pp_touch_zero_KnockOut_Call"])
        assert abs(priced - _PP_REBATE * discounts[0]) > 1.0

    @pytest.mark.parametrize(
        ("barrier_type", "name"),
        [(DoubleBarrierType.KIKO, "KIKO"), (DoubleBarrierType.KOKI, "KOKI")],
    )
    def test_kiko_koki_unknown_barrier_type(
        self, cpp: dict[str, Any], barrier_type: DoubleBarrierType, name: str
    ) -> None:
        """# C++ parity: ``QL_FAIL("unknown barrier type")``
        # (mcdoublebarrierengine.cpp:81)."""
        assert cpp[f"pp_{name}_throws"] is True
        grid, discounts = self._grid_and_discounts(cpp)
        pricer = DoubleBarrierPathPricer(
            barrier_type,
            _PP_LO,
            _PP_HI,
            _PP_REBATE,
            OptionType.Call,
            _PP_STRIKE,
            discounts,
        )
        values = np.asarray(cpp["pp_path_inside"], dtype=np.float64)
        with pytest.raises(LibraryException, match="unknown barrier type"):
            pricer(Path(grid, values))

    @pytest.mark.parametrize(
        ("key", "strike", "low", "high", "message"),
        [
            (
                "pp_ctor_negative_strike_throws",
                -1.0,
                _PP_LO,
                _PP_HI,
                "strike less than zero not allowed",
            ),
            (
                "pp_ctor_zero_low_barrier_throws",
                _PP_STRIKE,
                0.0,
                _PP_HI,
                "low barrier less/equal zero not allowed",
            ),
            (
                "pp_ctor_zero_high_barrier_throws",
                _PP_STRIKE,
                _PP_LO,
                0.0,
                "high barrier less/equal zero not allowed",
            ),
        ],
    )
    def test_constructor_requirements(
        self,
        cpp: dict[str, Any],
        key: str,
        strike: float,
        low: float,
        high: float,
        message: str,
    ) -> None:
        """# C++ parity: the three ctor QL_REQUIREs (.cpp:34-39)."""
        assert cpp[key] is True
        _, discounts = self._grid_and_discounts(cpp)
        with pytest.raises(LibraryException, match=message):
            DoubleBarrierPathPricer(
                DoubleBarrierType.KnockOut,
                low,
                high,
                _PP_REBATE,
                OptionType.Call,
                strike,
                discounts,
            )

    def test_two_point_path(self, cpp: dict[str, Any]) -> None:
        """A length-2 path is the shortest one ``QL_REQUIRE(n>1)`` admits."""
        grid = TimeGrid.regular(1.0, 1)
        pricer = DoubleBarrierPathPricer(
            DoubleBarrierType.KnockOut,
            _PP_LO,
            _PP_HI,
            _PP_REBATE,
            OptionType.Call,
            _PP_STRIKE,
            [1.0, 1.0],
        )
        path = Path(grid, np.array([100.0, 100.0], dtype=np.float64))
        tight(pricer(path), cpp["pp_two_point_path"])


# ---------------------------------------------------------------------------
# MCDoubleBarrierEngine / MakeMCDoubleBarrierEngine
# ---------------------------------------------------------------------------

_MC_SPOT = 100.0
_MC_Q = 0.02
_MC_R = 0.05
_MC_VOL = 0.20

# (tag, barrier type, option type, lo, hi, rebate, strike, steps, samples,
#  seed, antithetic) — probe.cpp ``McCase cases[4]``.
_MC_CASES: list[
    tuple[str, DoubleBarrierType, OptionType, float, float, float, float, int, int, int, bool]
] = [
    ("ko_call", DoubleBarrierType.KnockOut, OptionType.Call, 90.0, 110.0, 0.0, 100.0, 32, 4095, 42, False),
    ("ki_call", DoubleBarrierType.KnockIn, OptionType.Call, 90.0, 110.0, 0.0, 100.0, 32, 4095, 42, False),
    ("ko_put_rebate", DoubleBarrierType.KnockOut, OptionType.Put, 85.0, 115.0, 3.0, 100.0, 16, 2047, 7, False),
    ("ki_put_anti", DoubleBarrierType.KnockIn, OptionType.Put, 85.0, 115.0, 1.5, 100.0, 16, 2048, 7, True),
]


def _mc_process() -> GeneralizedBlackScholesProcess:
    return _make_bsm(_MC_SPOT, _MC_Q, _MC_R, _MC_VOL, _ACT365)


def _mc_option(
    barrier_type: DoubleBarrierType,
    lo: float,
    hi: float,
    rebate: float,
    option_type: OptionType,
    strike: float,
    ex_date: Date,
) -> DoubleBarrierOption:
    return DoubleBarrierOption(
        barrier_type,
        lo,
        hi,
        rebate,
        PlainVanillaPayoff(option_type, strike),
        EuropeanExercise(ex_date),
    )


class TestMCDoubleBarrierEngine:
    """# C++ parity: ``MCDoubleBarrierEngine<PseudoRandom>``
    # (mcdoublebarrierengine.hpp:35-86, 134-201)."""

    def test_residual_time(self, cpp: dict[str, Any]) -> None:
        tight(_mc_process().time(_TODAY + 180), cpp["mc_residual_time"])

    @pytest.mark.parametrize(
        (
            "tag", "barrier_type", "option_type", "lo", "hi", "rebate", "strike",
            "steps", "samples", "seed", "antithetic",
        ),
        _MC_CASES,
        ids=[c[0] for c in _MC_CASES],
    )
    def test_fixed_seed_reproduces_cpp_stream(
        self,
        cpp: dict[str, Any],
        tag: str,
        barrier_type: DoubleBarrierType,
        option_type: OptionType,
        lo: float,
        hi: float,
        rebate: float,
        strike: float,
        steps: int,
        samples: int,
        seed: int,
        antithetic: bool,
    ) -> None:
        option = _mc_option(
            barrier_type, lo, hi, rebate, option_type, strike, _TODAY + 180
        )
        option.set_pricing_engine(
            MCDoubleBarrierEngine(
                _mc_process(),
                time_steps=steps,
                time_steps_per_year=None,
                brownian_bridge=False,
                antithetic_variate=antithetic,
                required_samples=samples,
                required_tolerance=None,
                max_samples=None,
                seed=seed,
            )
        )
        tight(option.npv(), cpp[f"mc_{tag}_npv"], reason="npv")
        tight(
            option.error_estimate(),
            cpp[f"mc_{tag}_error_estimate"],
            reason="error estimate",
        )

    def test_steps_per_year(self, cpp: dict[str, Any]) -> None:
        """# C++ parity: ``TimeGrid(residual, max<Size>(spy*residual, 1))``
        # (mcdoublebarrierengine.hpp:170-172) — note the truncating cast."""
        option = _mc_option(
            DoubleBarrierType.KnockOut, 90.0, 110.0, 0.0, OptionType.Call, 100.0,
            _TODAY + 180,
        )
        option.set_pricing_engine(
            MCDoubleBarrierEngine(
                _mc_process(),
                time_steps=None,
                time_steps_per_year=52,
                required_samples=1023,
                seed=3,
            )
        )
        tight(option.npv(), cpp["mc_steps_per_year_npv"], reason="npv")
        tight(
            option.error_estimate(),
            cpp["mc_steps_per_year_error_estimate"],
            reason="error estimate",
        )

    def test_tolerance_driven_sampling(self, cpp: dict[str, Any]) -> None:
        """Exercises ``McSimulation::value``'s sample-growth loop."""
        option = _mc_option(
            DoubleBarrierType.KnockOut, 90.0, 110.0, 0.0, OptionType.Call, 100.0,
            _TODAY + 180,
        )
        option.set_pricing_engine(
            MCDoubleBarrierEngine(
                _mc_process(),
                time_steps=16,
                required_tolerance=0.02,
                seed=11,
            )
        )
        tight(option.npv(), cpp["mc_tolerance_npv"], reason="npv")
        tight(
            option.error_estimate(),
            cpp["mc_tolerance_error_estimate"],
            reason="error estimate",
        )

    def test_brownian_bridge(self, cpp: dict[str, Any]) -> None:
        option = _mc_option(
            DoubleBarrierType.KnockOut, 90.0, 110.0, 0.0, OptionType.Call, 100.0,
            _TODAY + 180,
        )
        option.set_pricing_engine(
            MCDoubleBarrierEngine(
                _mc_process(),
                time_steps=8,
                brownian_bridge=True,
                required_samples=1023,
                seed=5,
            )
        )
        tight(option.npv(), cpp["mc_brownian_bridge_npv"], reason="npv")
        tight(
            option.error_estimate(),
            cpp["mc_brownian_bridge_error_estimate"],
            reason="error estimate",
        )

    def test_no_time_steps(self, cpp: dict[str, Any]) -> None:
        assert cpp["mc_engine_no_time_steps_throws"] is True
        with pytest.raises(LibraryException, match="no time steps provided"):
            MCDoubleBarrierEngine(_mc_process(), required_samples=1023, seed=1)

    def test_both_time_steps(self, cpp: dict[str, Any]) -> None:
        assert cpp["mc_engine_both_time_steps_throws"] is True
        with pytest.raises(
            LibraryException, match="both time steps and time steps per year"
        ):
            MCDoubleBarrierEngine(
                _mc_process(),
                time_steps=10,
                time_steps_per_year=12,
                required_samples=1023,
                seed=1,
            )

    def test_zero_time_steps(self, cpp: dict[str, Any]) -> None:
        assert cpp["mc_engine_zero_time_steps_throws"] is True
        with pytest.raises(LibraryException, match="timeSteps must be positive"):
            MCDoubleBarrierEngine(
                _mc_process(), time_steps=0, required_samples=1023, seed=1
            )

    def test_barrier_already_touched(self, cpp: dict[str, Any]) -> None:
        """# C++ parity: ``QL_REQUIRE(!triggered(spot), "barrier touched")``
        # (mcdoublebarrierengine.hpp:56)."""
        assert cpp["mc_engine_barrier_touched_throws"] is True
        option = _mc_option(
            DoubleBarrierType.KnockOut, 100.0, 110.0, 0.0, OptionType.Call, 100.0,
            _TODAY + 180,
        )
        option.set_pricing_engine(
            MCDoubleBarrierEngine(
                _mc_process(), time_steps=8, required_samples=1023, seed=1
            )
        )
        with pytest.raises(LibraryException, match="barrier touched"):
            option.npv()


class TestMakeMCDoubleBarrierEngine:
    """# C++ parity: ``MakeMCDoubleBarrierEngine<PseudoRandom>``
    # (mcdoublebarrierengine.hpp:89-110, 203-290)."""

    def test_matches_direct_construction(self, cpp: dict[str, Any]) -> None:
        option = _mc_option(
            DoubleBarrierType.KnockOut, 90.0, 110.0, 0.0, OptionType.Call, 100.0,
            _TODAY + 180,
        )
        option.set_pricing_engine(
            MakeMCDoubleBarrierEngine(_mc_process())
            .with_steps(32)
            .with_samples(4095)
            .with_seed(42)
            .build()
        )
        tight(option.npv(), cpp["make_mc_ko_call_npv"], reason="npv")
        tight(
            option.error_estimate(),
            cpp["make_mc_ko_call_error_estimate"],
            reason="error estimate",
        )
        # Same engine, same stream, whichever way it was built.
        tight(option.npv(), cpp["mc_ko_call_npv"], reason="fluent == direct")

    def test_every_setter(self, cpp: dict[str, Any]) -> None:
        option = _mc_option(
            DoubleBarrierType.KnockOut, 90.0, 110.0, 0.0, OptionType.Call, 100.0,
            _TODAY + 180,
        )
        option.set_pricing_engine(
            MakeMCDoubleBarrierEngine(_mc_process())
            .with_steps_per_year(52)
            .with_brownian_bridge(True)
            .with_antithetic_variate(True)
            .with_absolute_tolerance(0.05)
            .with_max_samples(1 << 20)
            .with_seed(99)
            .build()
        )
        tight(option.npv(), cpp["make_mc_fluent_all_npv"], reason="npv")
        tight(
            option.error_estimate(),
            cpp["make_mc_fluent_all_error_estimate"],
            reason="error estimate",
        )

    def test_no_steps(self, cpp: dict[str, Any]) -> None:
        assert cpp["make_mc_no_steps_throws"] is True
        with pytest.raises(LibraryException, match="number of steps not given"):
            MakeMCDoubleBarrierEngine(_mc_process()).with_samples(1023).build()

    def test_steps_overspecified(self, cpp: dict[str, Any]) -> None:
        assert cpp["make_mc_steps_overspecified_throws"] is True
        with pytest.raises(LibraryException, match="number of steps overspecified"):
            (
                MakeMCDoubleBarrierEngine(_mc_process())
                .with_steps(10)
                .with_steps_per_year(12)
                .with_samples(1023)
                .build()
            )

    def test_samples_after_tolerance(self, cpp: dict[str, Any]) -> None:
        assert cpp["make_mc_samples_after_tolerance_throws"] is True
        with pytest.raises(LibraryException, match="tolerance already set"):
            (
                MakeMCDoubleBarrierEngine(_mc_process())
                .with_absolute_tolerance(0.01)
                .with_samples(1023)
            )

    def test_tolerance_after_samples(self, cpp: dict[str, Any]) -> None:
        assert cpp["make_mc_tolerance_after_samples_throws"] is True
        with pytest.raises(LibraryException, match="number of samples already set"):
            (
                MakeMCDoubleBarrierEngine(_mc_process())
                .with_samples(1023)
                .with_absolute_tolerance(0.01)
            )


class TestRngTraits:
    """# C++ parity: ``SingleVariate<RNG>::allowsErrorEstimate``
    # (mctraits.hpp:44) re-exporting ``rngtraits.hpp``'s policy flag.

    ``RngTraits`` is the Python spelling of the ``RNG`` template parameter of
    ``MCDoubleBarrierEngine`` / ``MakeMCDoubleBarrierEngine``. The only member
    with observable behaviour is ``allowsErrorEstimate``, which is what gates
    ``results_.errorEstimate`` in ``calculate()``.
    """

    def test_policy_flags_match_cpp(self, cpp: dict[str, Any]) -> None:
        assert PseudoRandom.allows_error_estimate == cpp[
            "rng_pseudo_allows_error_estimate"
        ]
        assert LowDiscrepancy.allows_error_estimate == cpp[
            "rng_lowdiscrepancy_allows_error_estimate"
        ]
        # SingleVariate<RNG> just forwards the flag.
        assert (
            cpp["rng_singlevariate_pseudo_allows_error_estimate"]
            == cpp["rng_pseudo_allows_error_estimate"]
        )
        assert (
            cpp["rng_singlevariate_lowdiscrepancy_allows_error_estimate"]
            == cpp["rng_lowdiscrepancy_allows_error_estimate"]
        )

    def test_alias_admits_both_policies(self) -> None:
        # The alias is a union of the two policy *classes*; C++ never
        # instantiates the traits struct either.
        admitted: list[RngTraits] = [PseudoRandom, LowDiscrepancy]
        for policy in admitted:
            assert hasattr(policy, "allows_error_estimate")
            assert callable(policy.make_sequence_generator)


# ---------------------------------------------------------------------------
# QuantoTermStructure
# ---------------------------------------------------------------------------

_QUANTO_SPOT = 100.0
_QUANTO_STRIKE = 102.0
_QUANTO_Q = 0.01
_QUANTO_R = 0.1
_QUANTO_VOL = 0.15
_QUANTO_FXR = 0.05
_QUANTO_FXVOL = 0.2
_QUANTO_CORR = 0.3


def _quanto_curves() -> tuple[FlatForward, FlatForward, BlackConstantVol, FlatForward, BlackConstantVol]:
    return (
        FlatForward.from_rate(_TODAY, _QUANTO_Q, _ACT360),
        FlatForward.from_rate(_TODAY, _QUANTO_R, _ACT360),
        _flat_vol(_TODAY, _QUANTO_VOL, _ACT360),
        FlatForward.from_rate(_TODAY, _QUANTO_FXR, _ACT360),
        _flat_vol(_TODAY, _QUANTO_FXVOL, _ACT360),
    )


class TestQuantoTermStructure:
    """# C++ parity: ``QuantoTermStructure`` (quantotermstructure.hpp:42-133)."""

    @staticmethod
    def _build() -> QuantoTermStructure:
        q_ts, r_ts, vol_ts, fxr_ts, fxvol_ts = _quanto_curves()
        return QuantoTermStructure(
            q_ts, r_ts, fxr_ts, vol_ts, _QUANTO_STRIKE, fxvol_ts, 1.0, _QUANTO_CORR
        )

    def test_zero_yield_and_discounts(self, cpp: dict[str, Any]) -> None:
        ts = self._build()
        times = cpp["quanto_ts_times"]
        zeros = [
            ts.zero_rate(
                float(t), Compounding.Continuous, Frequency.NoFrequency, True
            ).rate()
            for t in times
        ]
        discounts = [ts.discount(float(t), True) for t in times]
        _assert_vector(zeros, cpp["quanto_ts_zero_rates"], label="zero rates")
        _assert_vector(discounts, cpp["quanto_ts_discounts"], label="discounts")

    def test_forwarding_accessors(self, cpp: dict[str, Any]) -> None:
        ts = self._build()
        assert ts.reference_date().serial_number() == cpp[
            "quanto_ts_reference_date_serial"
        ]
        assert ts.max_date().serial_number() == cpp["quanto_ts_max_date_serial"]
        assert ts.day_counter().name() == cpp["quanto_ts_day_counter"]

    def test_settlement_days_and_calendar_forward_the_failure(
        self, cpp: dict[str, Any]
    ) -> None:
        """Both accessors delegate to the *underlying dividend* curve, which is
        a reference-date ``FlatForward`` carrying neither. C++ throws; so does
        the port."""
        assert cpp["quanto_ts_settlement_days_throws"] is True
        assert cpp["quanto_ts_calendar_name_throws"] is True
        ts = self._build()
        with pytest.raises(LibraryException, match="settlement days not provided"):
            ts.settlement_days()
        with pytest.raises(LibraryException, match="calendar not provided"):
            ts.calendar().name()


# ---------------------------------------------------------------------------
# QuantoEngine / QuantoOptionResults / QuantoDoubleBarrierOption
# ---------------------------------------------------------------------------


class _StubDoubleBarrierEngine(
    GenericEngine[DoubleBarrierOptionArguments, OneAssetOptionResults]
):
    """Deterministic stand-in for the wrapped engine.

    Mirrors ``StubDoubleBarrierEngine`` in probe.cpp exactly: every result is a
    fixed multiple of a discount factor read off the *quanto-adjusted* process,
    so ``QuantoTermStructure::zeroYieldImpl`` is pinned through it and the
    greek arithmetic runs on non-``Null`` inputs (which the real analytic
    engine never supplies).
    """

    def __init__(self, process: GeneralizedBlackScholesProcess) -> None:
        super().__init__(DoubleBarrierOptionArguments(), OneAssetOptionResults())
        self._process = process

    def calculate(self) -> None:
        exercise = self._arguments.exercise
        assert exercise is not None
        t = self._process.time(exercise.last_date())
        qdisc = self._process.dividend_yield().discount(t)
        rdisc = self._process.risk_free_rate().discount(t)
        self._results.value = 10.0 * qdisc
        self._results.delta = 0.5 * qdisc
        self._results.gamma = 0.05 * rdisc
        self._results.theta = -1.25 * qdisc
        self._results.rho = 7.0 * rdisc
        self._results.dividend_rho = -3.0 * qdisc
        self._results.vega = 21.0 * qdisc


class TestQuantoEngine:
    """# C++ parity: ``QuantoEngine<Instr, Engine>`` (quantoengine.hpp:85-169)."""

    @staticmethod
    def _stub_option() -> QuantoDoubleBarrierOption:
        q_ts, r_ts, vol_ts, fxr_ts, fxvol_ts = _quanto_curves()
        process = GeneralizedBlackScholesProcess(
            x0=SimpleQuote(_QUANTO_SPOT),
            dividend_ts=q_ts,
            risk_free_ts=r_ts,
            black_vol_ts=vol_ts,
        )
        option = QuantoDoubleBarrierOption(
            DoubleBarrierType.KnockOut,
            90.0,
            110.0,
            1.5,
            PlainVanillaPayoff(OptionType.Call, _QUANTO_STRIKE),
            EuropeanExercise(_TODAY + 90),
        )
        option.set_pricing_engine(
            QuantoEngine(
                process,
                fxr_ts,
                fxvol_ts,
                SimpleQuote(_QUANTO_CORR),
                DoubleBarrierOptionArguments(),
                _StubDoubleBarrierEngine,
            )
        )
        return option

    @pytest.mark.parametrize(
        "greek",
        [
            "npv",
            "delta",
            "gamma",
            "theta",
            "rho",
            "dividend_rho",
            "vega",
            "qvega",
            "qrho",
            "qlambda",
        ],
    )
    def test_greek_arithmetic(self, cpp: dict[str, Any], greek: str) -> None:
        option = self._stub_option()
        getter: Callable[[], float] = getattr(option, greek)
        tight(getter(), cpp[f"quanto_stub_{greek}"], reason=greek)

    def test_wrong_engine_type(self, cpp: dict[str, Any]) -> None:
        """# C++ parity: ``QL_REQUIRE(originalArguments, "wrong engine type")``
        # (quantoengine.hpp:121). ``AnalyticEuropeanEngine`` carries
        # ``VanillaOption::arguments``, a *base* of the double-barrier bundle,
        # so the C++ ``dynamic_cast`` returns null."""
        assert cpp["quanto_wrong_engine_type_throws"] is True
        q_ts, r_ts, vol_ts, fxr_ts, fxvol_ts = _quanto_curves()
        process = GeneralizedBlackScholesProcess(
            x0=SimpleQuote(_QUANTO_SPOT),
            dividend_ts=q_ts,
            risk_free_ts=r_ts,
            black_vol_ts=vol_ts,
        )
        option = QuantoDoubleBarrierOption(
            DoubleBarrierType.KnockOut,
            90.0,
            110.0,
            1.5,
            PlainVanillaPayoff(OptionType.Call, _QUANTO_STRIKE),
            EuropeanExercise(_TODAY + 90),
        )
        option.set_pricing_engine(
            QuantoEngine(
                process,
                fxr_ts,
                fxvol_ts,
                SimpleQuote(_QUANTO_CORR),
                DoubleBarrierOptionArguments(),
                AnalyticEuropeanEngine,
            )
        )
        with pytest.raises(LibraryException, match="wrong engine type"):
            option.npv()


class TestQuantoOptionResults:
    """# C++ parity: ``QuantoOptionResults<ResultsType>``
    # (quantovanillaoption.hpp:33-45)."""

    def test_reset_nulls_the_three_sensitivities(self) -> None:
        results = QuantoOptionResults()
        assert results.qvega is None
        assert results.qrho is None
        assert results.qlambda is None
        results.qvega = 1.0
        results.qrho = 2.0
        results.qlambda = 3.0
        results.value = 4.0
        results.reset()
        assert results.qvega is None
        assert results.qrho is None
        assert results.qlambda is None
        # The base reset still runs (C++ calls ``ResultsType::reset()`` first).
        assert results.value is None

    def test_is_a_one_asset_option_results(self) -> None:
        assert isinstance(QuantoOptionResults(), OneAssetOptionResults)


# (barrier type, lo, hi, rebate, option type, spot, strike, q, r, t, vol,
#  fx rate, fx vol, corr) — probe.cpp ``QuantoCase cases[5]``.
_QUANTO_ANALYTIC: list[
    tuple[DoubleBarrierType, float, float, float, OptionType, float, float, float, float, float, float, float, float, float]
] = [
    (DoubleBarrierType.KnockOut, 50.0, 150.0, 0.0, OptionType.Call, 100.0, 100.0, 0.0, 0.1, 0.25, 0.15, 0.05, 0.2, 0.3),
    (DoubleBarrierType.KnockOut, 90.0, 110.0, 0.0, OptionType.Call, 100.0, 100.0, 0.0, 0.1, 0.50, 0.15, 0.05, 0.2, 0.3),
    (DoubleBarrierType.KnockOut, 90.0, 110.0, 0.0, OptionType.Put, 100.0, 100.0, 0.0, 0.1, 0.25, 0.15, 0.05, 0.2, 0.3),
    (DoubleBarrierType.KnockIn, 80.0, 120.0, 0.0, OptionType.Call, 100.0, 102.0, 0.0, 0.1, 0.25, 0.25, 0.05, 0.2, 0.3),
    (DoubleBarrierType.KnockIn, 80.0, 120.0, 0.0, OptionType.Call, 100.0, 102.0, 0.0, 0.1, 0.50, 0.15, 0.05, 0.2, 0.3),
]


class TestQuantoDoubleBarrierOption:
    """# C++ parity: ``QuantoDoubleBarrierOption``
    # (quantodoublebarrieroption.hpp:34-58 + .cpp:24-67)."""

    @pytest.mark.parametrize("index", range(len(_QUANTO_ANALYTIC)))
    def test_priced_through_the_analytic_engine(
        self, cpp: dict[str, Any], index: int
    ) -> None:
        (
            barrier_type, lo, hi, rebate, option_type, spot, strike,
            q, r, t, vol, fxr, fxvol, corr,
        ) = _QUANTO_ANALYTIC[index]
        process = GeneralizedBlackScholesProcess(
            x0=SimpleQuote(spot),
            dividend_ts=FlatForward.from_rate(_TODAY, q, _ACT360),
            risk_free_ts=FlatForward.from_rate(_TODAY, r, _ACT360),
            black_vol_ts=_flat_vol(_TODAY, vol, _ACT360),
        )
        # ``timeToDays(t)`` with the test-suite's 360-day year.
        ex_date = _TODAY + int(t * 360 + 0.5)
        assert ex_date.serial_number() == cpp[f"quanto_{index}_ex_date_serial"]

        option = QuantoDoubleBarrierOption(
            barrier_type,
            lo,
            hi,
            rebate,
            PlainVanillaPayoff(option_type, strike),
            EuropeanExercise(ex_date),
        )
        option.set_pricing_engine(
            QuantoEngine(
                process,
                FlatForward.from_rate(_TODAY, fxr, _ACT360),
                _flat_vol(_TODAY, fxvol, _ACT360),
                SimpleQuote(corr),
                DoubleBarrierOptionArguments(),
                AnalyticDoubleBarrierEngine,
            )
        )
        tight(option.npv(), cpp[f"quanto_{index}_npv"], reason="npv")

        # AnalyticDoubleBarrierEngine publishes only ``value``, so every greek
        # QuantoEngine copies over stays Null and the accessors throw. Real
        # v1.43 behaviour, pinned as such.
        for greek in ("delta", "qvega", "qrho", "qlambda"):
            assert cpp[f"quanto_{index}_{greek}_throws"] is True
            with pytest.raises(LibraryException):
                getattr(option, greek)()


# ---------------------------------------------------------------------------
# SuoWangDoubleBarrierEngine
# ---------------------------------------------------------------------------

# (tag, barrier type, lo, hi, rebate, option type, spot, strike, q, r, vol,
#  days, series) — probe.cpp ``SuoWangCase cases[8]``.
_SUOWANG_CASES: list[
    tuple[str, DoubleBarrierType, float, float, float, OptionType, float, float, float, float, float, int, int]
] = [
    ("ko_call_atm", DoubleBarrierType.KnockOut, 90.0, 110.0, 0.0, OptionType.Call, 100.0, 100.0, 0.0, 0.10, 0.15, 90, 5),
    ("ko_put_atm", DoubleBarrierType.KnockOut, 90.0, 110.0, 0.0, OptionType.Put, 100.0, 100.0, 0.0, 0.10, 0.15, 90, 5),
    ("ki_call_atm", DoubleBarrierType.KnockIn, 90.0, 110.0, 0.0, OptionType.Call, 100.0, 100.0, 0.0, 0.10, 0.15, 90, 5),
    ("ki_put_wide", DoubleBarrierType.KnockIn, 50.0, 150.0, 0.0, OptionType.Put, 100.0, 100.0, 0.0, 0.10, 0.25, 180, 5),
    ("ko_call_otm", DoubleBarrierType.KnockOut, 80.0, 120.0, 0.0, OptionType.Call, 100.0, 110.0, 0.04, 0.10, 0.20, 180, 5),
    ("ko_put_itm", DoubleBarrierType.KnockOut, 80.0, 120.0, 0.0, OptionType.Put, 100.0, 110.0, 0.04, 0.10, 0.20, 180, 5),
    ("ko_rebate", DoubleBarrierType.KnockOut, 90.0, 110.0, 3.0, OptionType.Call, 100.0, 100.0, 0.0, 0.10, 0.15, 90, 5),
    ("ko_series_12", DoubleBarrierType.KnockOut, 90.0, 110.0, 0.0, OptionType.Call, 100.0, 100.0, 0.0, 0.10, 0.15, 90, 12),
]

_SUOWANG_CANCELLATION_ABS: float = 1e-13
"""Absolute bound for the ``ki_put_wide`` knock-in value; see the derivation.

``barrierIn = european - barrierOut`` (suowangdoublebarrierengine.cpp:125,128).
For that case the two operands are 4.705177510574539 and 4.703257197155276, so
the subtraction cancels down to 1.9203e-3 — a loss of about 3.4 decimal digits.

``barrierOut`` itself is a ten-term image series, each term a product of four
``CumulativeNormalDistribution`` evaluations with ``std::pow`` factors; C++ and
Python agree on it to 3.9e-14 absolute (8.3e-15 relative on 4.70, i.e. ~37
double eps, which is what a chain of that depth accumulates). Subtraction is
exact once the operands are equal in magnitude, so the *absolute* error of
``barrierOut`` passes straight through to the difference:

    |barrierIn_py - barrierIn_cpp| <~ 2 * 3.9e-14 ~ 8e-14

1e-13 is that bound rounded up. Expressed as a relative tolerance it would be
4e-11, which is why the TIGHT tier (1e-12 relative) cannot hold here. Every
other SuoWang number in the reference is asserted at TIGHT, including
``barrierOut`` and ``vanilla`` for this same case.
"""


class TestSuoWangDoubleBarrierEngine:
    """# C++ parity: ``SuoWangDoubleBarrierEngine``
    # (suowangdoublebarrierengine.hpp:42-62 + .cpp:28-168)."""

    @pytest.mark.parametrize(
        (
            "tag", "barrier_type", "lo", "hi", "rebate", "option_type", "spot",
            "strike", "q", "r", "vol", "days", "series",
        ),
        _SUOWANG_CASES,
        ids=[c[0] for c in _SUOWANG_CASES],
    )
    def test_value_and_additional_results(
        self,
        cpp: dict[str, Any],
        tag: str,
        barrier_type: DoubleBarrierType,
        lo: float,
        hi: float,
        rebate: float,
        option_type: OptionType,
        spot: float,
        strike: float,
        q: float,
        r: float,
        vol: float,
        days: int,
        series: int,
    ) -> None:
        process = _make_bsm(spot, q, r, vol, _ACT360)
        ex_date = _TODAY + days
        option = DoubleBarrierOption(
            barrier_type,
            lo,
            hi,
            rebate,
            PlainVanillaPayoff(option_type, strike),
            EuropeanExercise(ex_date),
        )
        option.set_pricing_engine(SuoWangDoubleBarrierEngine(process, series))

        prefix = f"suowang_{tag}"
        assert ex_date.serial_number() == cpp[prefix + "_ex_date_serial"]
        assert series == cpp[prefix + "_series"]

        # The knock-in leg is `european - barrierOut`; for `ki_put_wide` that
        # subtraction cancels, so it (and the NPV, which *is* that leg for a
        # KnockIn) carries a derived absolute bound instead of TIGHT.
        cancels = tag == "ki_put_wide"
        results = option.additional_results()

        def check(actual: float, key: str, *, cancelling: bool) -> None:
            expected = float(cpp[prefix + key])
            if cancelling:
                custom(
                    actual,
                    expected,
                    abs_tol=_SUOWANG_CANCELLATION_ABS,
                    rel_tol=0.0,
                    reason="european - barrierOut cancellation; see "
                    "_SUOWANG_CANCELLATION_ABS",
                )
            else:
                tight(actual, expected, reason=key)

        check(option.npv(), "_npv", cancelling=cancels)
        check(float(results["vanilla"]), "_vanilla", cancelling=False)
        check(float(results["barrierOut"]), "_barrier_out", cancelling=False)
        check(float(results["barrierIn"]), "_barrier_in", cancelling=cancels)
        check(float(results["rebateIn"]), "_rebate_in", cancelling=False)

    def test_rebate_in_is_published_but_never_added_to_the_value(
        self, cpp: dict[str, Any]
    ) -> None:
        """# C++ parity note (defect, reproduced verbatim): the comment
        # ``//rebate paid at maturity`` sits above
        # ``if(barrierType == DoubleBarrier::KnockOut) results_.value =
        # barrierOut;`` (suowangdoublebarrierengine.cpp:121-123) — ``rebateIn``
        # is computed, published as an additional result, and then dropped.
        # The ``ko_rebate`` case has rebate 3.0 and a rebateIn of -2091.66, yet
        # its NPV is bit-identical to the rebate-free ``ko_call_atm`` case.
        """
        assert cpp["suowang_ko_rebate_rebate_in"] != 0.0
        tight(
            float(cpp["suowang_ko_rebate_npv"]),
            float(cpp["suowang_ko_call_atm_npv"]),
            reason="rebate does not move the value",
        )

    def test_series_is_a_half_open_range(self, cpp: dict[str, Any]) -> None:
        """``for (int n = -series_; n < series_; n++)`` — asymmetric by one
        term, so widening the series changes nothing here only because the
        extra image terms underflow."""
        tight(
            float(cpp["suowang_ko_series_12_npv"]),
            float(cpp["suowang_ko_call_atm_npv"]),
            reason="series 12 vs 5 both converged",
        )

    @pytest.mark.parametrize(
        ("barrier_type", "name"),
        [(DoubleBarrierType.KIKO, "KIKO"), (DoubleBarrierType.KOKI, "KOKI")],
    )
    def test_kiko_koki_unsupported(
        self, cpp: dict[str, Any], barrier_type: DoubleBarrierType, name: str
    ) -> None:
        assert cpp[f"suowang_{name}_throws"] is True
        process = _make_bsm(100.0, 0.0, 0.10, 0.15, _ACT360)
        option = DoubleBarrierOption(
            barrier_type,
            90.0,
            110.0,
            0.0,
            PlainVanillaPayoff(OptionType.Call, 100.0),
            EuropeanExercise(_TODAY + 90),
        )
        option.set_pricing_engine(SuoWangDoubleBarrierEngine(process, 5))
        with pytest.raises(
            LibraryException, match="only KnockIn and KnockOut options supported"
        ):
            option.npv()


# ---------------------------------------------------------------------------
# PerturbativeBarrierOptionEngine
# ---------------------------------------------------------------------------

# (tag, spot, strike, barrier, q, r, vol@+90d, vol@+180d, days, order,
#  zeroGamma) — probe.cpp ``PerturbCase cases[12]``.
_PERT_CASES: list[tuple[str, float, float, float, float, float, float, float, int, int, bool]] = [
    ("ref_o0", 100.0, 101.0, 101.0, 0.02, 0.03, 0.105, 0.11, 180, 0, False),
    ("ref_o1", 100.0, 101.0, 101.0, 0.02, 0.03, 0.105, 0.11, 180, 1, False),
    ("ref_o0_zg", 100.0, 101.0, 101.0, 0.02, 0.03, 0.105, 0.11, 180, 0, True),
    ("ref_o1_zg", 100.0, 101.0, 101.0, 0.02, 0.03, 0.105, 0.11, 180, 1, True),
    ("k_below_o0", 100.0, 95.0, 110.0, 0.02, 0.03, 0.105, 0.11, 180, 0, False),
    ("k_below_o1", 100.0, 95.0, 110.0, 0.02, 0.03, 0.105, 0.11, 180, 1, False),
    ("k_below_o1_zg", 100.0, 95.0, 110.0, 0.02, 0.03, 0.105, 0.11, 180, 1, True),
    ("k_above_o1", 100.0, 120.0, 105.0, 0.02, 0.03, 0.105, 0.11, 180, 1, False),
    ("near_barrier_o0", 100.0, 105.0, 100.5, 0.02, 0.03, 0.105, 0.11, 180, 0, False),
    ("near_barrier_o1", 100.0, 105.0, 100.5, 0.02, 0.03, 0.105, 0.11, 180, 1, False),
    ("near_barrier_o1_zg", 100.0, 105.0, 100.5, 0.02, 0.03, 0.105, 0.11, 180, 1, True),
    ("short_o1", 100.0, 101.0, 101.0, 0.02, 0.03, 0.105, 0.11, 90, 1, False),
]


def _pert_process(
    spot: float, q: float, r: float, v0: float, v1: float
) -> GeneralizedBlackScholesProcess:
    return GeneralizedBlackScholesProcess(
        x0=SimpleQuote(spot),
        dividend_ts=FlatForward.from_rate(_TODAY, q, _ACT360),
        risk_free_ts=FlatForward.from_rate(_TODAY, r, _ACT360),
        black_vol_ts=BlackVarianceCurve(
            reference_date=_TODAY,
            dates=[_TODAY + 90, _TODAY + 180],
            black_vol_curve=[v0, v1],
            day_counter=_ACT360,
        ),
    )


class TestPerturbativeBarrierOptionEngine:
    """# C++ parity: ``PerturbativeBarrierOptionEngine``
    # (perturbativebarrieroptionengine.hpp:40-51 + .cpp:1516-1547)."""

    @pytest.mark.parametrize(
        (
            "tag", "spot", "strike", "barrier", "q", "r", "v0", "v1", "days",
            "order", "zero_gamma",
        ),
        _PERT_CASES,
        ids=[c[0] for c in _PERT_CASES],
    )
    def test_price(
        self,
        cpp: dict[str, Any],
        tag: str,
        spot: float,
        strike: float,
        barrier: float,
        q: float,
        r: float,
        v0: float,
        v1: float,
        days: int,
        order: int,
        zero_gamma: bool,
    ) -> None:
        process = _pert_process(spot, q, r, v0, v1)
        ex_date = _TODAY + days
        option = BarrierOption(
            BarrierType.UpOut,
            barrier,
            0.0,
            PlainVanillaPayoff(OptionType.Put, strike),
            EuropeanExercise(ex_date),
        )
        option.set_pricing_engine(
            PerturbativeBarrierOptionEngine(process, order, zero_gamma)
        )
        prefix = f"pert_{tag}"
        assert ex_date.serial_number() == cpp[prefix + "_ex_date_serial"]
        assert order == cpp[prefix + "_order"]
        assert zero_gamma is cpp[prefix + "_zero_gamma"]
        tight(process.time(ex_date), cpp[prefix + "_tau_max"], reason="tau max")
        tight(option.npv(), cpp[prefix + "_npv"], reason="npv")

    @pytest.mark.parametrize(
        ("key", "barrier_type", "rebate", "option_type", "order", "message"),
        [
            (
                "pert_down_out_throws", BarrierType.DownOut, 0.0, OptionType.Put, 1,
                "only manages up-and-out options",
            ),
            (
                "pert_rebate_throws", BarrierType.UpOut, 1.0, OptionType.Put, 1,
                "does not manage non-null rebates",
            ),
            (
                "pert_call_throws", BarrierType.UpOut, 0.0, OptionType.Call, 1,
                "only manages put options",
            ),
            (
                "pert_order_3_throws", BarrierType.UpOut, 0.0, OptionType.Put, 3,
                "order must be <= 2",
            ),
        ],
    )
    def test_requirements(
        self,
        cpp: dict[str, Any],
        key: str,
        barrier_type: BarrierType,
        rebate: float,
        option_type: OptionType,
        order: int,
        message: str,
    ) -> None:
        """# C++ parity: the four QL_REQUIREs of ``calculate`` (.cpp:1518-1536)."""
        assert cpp[key] is True
        process = _make_bsm(100.0, 0.02, 0.03, 0.11, _ACT360)
        option = BarrierOption(
            barrier_type,
            101.0,
            rebate,
            PlainVanillaPayoff(option_type, 101.0),
            EuropeanExercise(_TODAY + 180),
        )
        option.set_pricing_engine(
            PerturbativeBarrierOptionEngine(process, order, False)
        )
        with pytest.raises(LibraryException, match=message):
            option.npv()


class TestPerturbativeInternals:
    """The transliterated Fortran numerics, pinned one function at a time.

    Order-2 pricing is "Too slow, skip" even in the C++ test suite
    (test-suite/barrieroption.cpp ``testPerturbative``), so the helpers the
    second-order term composes can only be cross-validated individually. The
    probe reaches them by ``#include``-ing the engine's translation unit.
    """

    def test_phid(self, cpp: dict[str, Any]) -> None:
        """The engine ships its own Hart/Miller normal CDF rather than using
        ``CumulativeNormalDistribution``; the published values are PHID's."""
        got = [_phid(float(z)) for z in cpp["int_phid_z"]]
        _assert_vector(got, cpp["int_phid"], label="PHID")

    def test_nd2(self, cpp: dict[str, Any]) -> None:
        got = [
            _nd2(float(a), float(b), float(r))
            for a, b, r in zip(
                cpp["int_nd2_a"], cpp["int_nd2_b"], cpp["int_nd2_rho"], strict=True
            )
        ]
        _assert_vector(got, cpp["int_nd2"], label="ND2")

    def test_studnt(self, cpp: dict[str, Any]) -> None:
        got = [_studnt(int(nu), 0.7) for nu in cpp["int_studnt_nu"]]
        _assert_vector(got, cpp["int_studnt"], label="STUDNT")

    def test_bvtl(self, cpp: dict[str, Any]) -> None:
        got = [_bvtl(int(nu), 0.4, -0.8, 0.35) for nu in cpp["int_studnt_nu"]]
        _assert_vector(got, cpp["int_bvtl"], label="BVTL")
        # The two |r| == 1 shortcuts. Note the first is a signed zero in C++
        # (`-0` in the raw JSON), which json.load flattens to 0.
        tight(_bvtl(0, 0.4, -0.8, -1.0), float(cpp["int_bvtl_r_plus_1"]))
        tight(_bvtl(0, 0.4, -0.8, 1.0), float(cpp["int_bvtl_r_minus_1"]))

    def test_first_order_helpers(self, cpp: dict[str, Any]) -> None:
        tight(_ff(0.02, 0.05, 0.11, -0.4, 0.2), cpp["int_ff_a"], reason="ff a")
        tight(_ff(0.005, 0.05, -0.11, 1.4, 0.2), cpp["int_ff_b"], reason="ff b")
        tight(_v(0.02, 0.05, 0.11, -0.06, 0.2), cpp["int_v_a"], reason="v a")
        tight(_v(0.005, 0.05, -0.11, 0.06, 0.2), cpp["int_v_b"], reason="v b")
        tight(
            _llold(0.02, 0.05, 0.11, -0.8, 0.06, 0.2),
            cpp["int_llold_a"],
            reason="llold a",
        )
        tight(
            _llold(0.005, 0.05, -0.11, -1.2, -0.06, 0.2),
            cpp["int_llold_b"],
            reason="llold b",
        )

    @pytest.mark.parametrize(
        ("tag", "limit", "sigmarho"),
        [
            ("generic", [0.0, 0.3, -0.6, 0.9], [0.0, 0.5, 0.35, 0.7]),
            ("negative", [0.0, -0.4, 0.8, -0.2], [0.0, -0.6, -0.3, 0.45]),
            ("zero_limits", [0.0, 0.0, 0.0, 0.0], [0.0, 0.2, 0.4, 0.6]),
            ("r23_one", [0.0, 0.5, -0.5, 0.25], [0.0, 0.1, 0.1, 1.0]),
            ("r23_minus_one", [0.0, 0.5, 0.6, 0.25], [0.0, 0.1, 0.1, -1.0]),
            ("r12_r13_zero", [0.0, 0.5, -0.5, 0.25], [0.0, 0.0, 0.0, 0.6]),
        ],
    )
    def test_tvtl(
        self, cpp: dict[str, Any], tag: str, limit: list[float], sigmarho: list[float]
    ) -> None:
        tight(
            _tvtl(0, list(limit), list(sigmarho), 1e-12),
            cpp["int_tvtl_" + tag],
            reason=f"tvtl {tag}",
        )

    def test_pntgnd_defect_produces_nan(self) -> None:
        """# C++ parity note (DEFECT, reproduced verbatim):

            FT = std::pow(( BA - R*BB ),0.5)/RR + BB*BB;

        (perturbativebarrieroptionengine.cpp:1263). Genz's Fortran has
        ``FT = ( (BA-R*BB)**2/RR + BB**2 )`` — the exponent should be 2, not
        0.5, and the square should be inside the division. With ``BA - R*BB``
        negative, ``std::pow`` of a negative base to 0.5 is NaN, and the NaN
        reaches the result.
        """
        # BA - R*BB = 0.3 - 0.4 * (-0.6) ... choose arguments that go negative:
        ba, bb, bc, ra, rb, r, rr = -0.6, 0.3, 0.9, 0.35, 0.7, 0.5, 0.84
        assert ba - r * bb < 0.0
        assert math.isnan(_pntgnd(0, ba, bb, bc, ra, rb, r, rr))

    def test_tvtl_adaptive_branch_is_nan_clamped(self, cpp: dict[str, Any]) -> None:
        """The NaN from ``PNTGND`` reaches ``tvtl``'s clamp, which returns 0.

        ``max(ZRO, min(TVT, ONE))`` with ``TVT`` NaN: C++'s ``std::min(a,b)``
        is ``b < a ? b : a`` and ``std::max(a,b)`` is ``a < b ? b : a``; both
        comparisons against NaN are false, so ``min`` returns NaN and ``max``
        returns ``ZRO``. Python's ``min``/``max`` have the same first-argument
        bias, so both languages return exactly 0.0 — which is what the
        reference records, and it is *not* the true trivariate probability
        (the non-degenerate seed for this configuration is ~0.167).
        """
        assert float(cpp["int_tvtl_generic"]) == 0.0
        assert float(cpp["int_tvtl_negative"]) == 0.0
        assert _tvtl(0, [0.0, 0.3, -0.6, 0.9], [0.0, 0.5, 0.35, 0.7], 1e-12) == 0.0
        # ... while the branches that never reach ADONET are non-degenerate.
        assert float(cpp["int_tvtl_zero_limits"]) > 0.0
        assert float(cpp["int_tvtl_r12_r13_zero"]) > 0.0

    def test_derivn3(self, cpp: dict[str, Any]) -> None:
        limit = [0.0, 0.3, -0.6, 0.9]
        sigmarho = [0.0, 0.5, 0.35, 0.7]
        for idx in (1, 2, 3):
            tight(
                _derivn3(list(limit), list(sigmarho), idx),
                cpp[f"int_derivn3_{idx}"],
                reason=f"derivn3 idx={idx}",
            )

    def test_second_order_helpers(self, cpp: dict[str, Any]) -> None:
        s, p, tt, gm = 0.004, 0.012, 0.05, 0.2
        x, xstar = 0.11, -0.06
        cases: list[tuple[str, float]] = [
            ("dvv", _dvv(s, p, tt, x, xstar, gm)),
            ("dvv_neg", _dvv(s, p, tt, -x, xstar, gm)),
            ("dff", _dff(s, p, tt, x, -1.0 + gm, gm)),
            ("dff_neg", _dff(s, p, tt, -x, 1.0 + gm, gm)),
            ("dll", _dll(s, p, tt, x, -1.0 + gm, -xstar, gm)),
            ("dll_neg", _dll(s, p, tt, -x, -1.0 - gm, xstar, gm)),
            ("ddvv", _ddvv(s, p, tt, x, xstar, gm)),
            ("ddvv_neg", _ddvv(s, p, tt, -x, xstar, gm)),
            ("ddff", _ddff(s, p, tt, x, -1.0 + gm, gm)),
            ("ddff_neg", _ddff(s, p, tt, -x, 1.0 + gm, gm)),
            ("ddll", _ddll(s, p, tt, x, -1.0 + gm, -xstar, gm)),
            ("ddll_neg", _ddll(s, p, tt, -x, 1.0 + gm, -xstar, gm)),
        ]
        for tag, got in cases:
            tight(got, cpp["int_" + tag], reason=tag)

    def test_ddff_uses_tt_minus_p_where_dff_uses_tt_minus_s(self) -> None:
        """# C++ parity note (DEFECT, reproduced verbatim):

            dff  : aa=exp((b*b-(1.0-gm)*(1.0-gm))*(tt-s)/4.0);   // :512
            ddff : aa=exp((b*b-(1.0-gm)*(1.0-gm))*(tt-p)/4.0);   // :606

        ``ddff`` is documented as d/da of ``dff``, and that derivative cannot
        touch the exponential prefactor (it has no ``a`` in it), so the two
        should carry the same time argument. They do not.

        This test locates the defect rather than re-pinning a value: the
        prefactor is the only place ``gm`` enters either function, so the ratio
        of two evaluations that differ *only* in ``gm`` isolates it exactly.
        ``test_second_order_helpers`` is what pins the values against C++.
        """
        s, p, tt = 0.004, 0.012, 0.05
        a, b = 0.11, 1.2
        gm1, gm2 = 0.2, 0.5

        def prefactor(gm: float, time: float) -> float:
            return math.exp((b * b - (1.0 - gm) * (1.0 - gm)) * time / 4.0)

        expected_p = prefactor(gm1, tt - p) / prefactor(gm2, tt - p)
        expected_s = prefactor(gm1, tt - s) / prefactor(gm2, tt - s)
        assert abs(expected_p - expected_s) > 1e-6 * abs(expected_s)

        ddff_ratio = _ddff(s, p, tt, a, b, gm1) / _ddff(
            s, p, tt, a, b, gm2
        )
        dff_ratio = _dff(s, p, tt, a, b, gm1) / _dff(s, p, tt, a, b, gm2)

        # ddff scales with (tt - p) ...
        tight(ddff_ratio, expected_p, reason="ddff prefactor uses (tt - p)")
        # ... while its own antiderivative dff scales with (tt - s).
        tight(dff_ratio, expected_s, reason="dff prefactor uses (tt - s)")

    @pytest.mark.parametrize(
        ("tag", "kprice", "stock", "hbarr", "iord", "igm"),
        [
            ("o0", 101.0, 100.0, 101.0, 0, 1),
            ("o0_gm0", 101.0, 100.0, 101.0, 0, 0),
            ("o1", 101.0, 100.0, 101.0, 1, 1),
            ("o1_gm0", 101.0, 100.0, 101.0, 1, 0),
            ("o1_kbelow", 95.0, 100.0, 110.0, 1, 1),
        ],
    )
    def test_barrier_upd(
        self,
        cpp: dict[str, Any],
        tag: str,
        kprice: float,
        stock: float,
        hbarr: float,
        iord: int,
        igm: int,
    ) -> None:
        """``BarrierUPD`` driven with analytic closures, so the quadrature is
        pinned independently of any term structure (flat r, q, sigma)."""
        r, q, sigma = 0.03, 0.02, 0.11

        def integr(t1: float, t2: float) -> float:
            return r * (t2 - t1)

        def integalpha(t1: float, t2: float) -> float:
            return (r - q) * (t2 - t1)

        def integs(t1: float, t2: float) -> float:
            return sigma * sigma * (t2 - t1)

        def alpha(_t: float) -> float:
            return r - q

        def sigmaq(_t: float) -> float:
            return sigma * sigma

        tight(
            _barrier_upd(
                kprice, stock, hbarr, 0.0, 0.5, iord, igm,
                integr, integalpha, integs, alpha, sigmaq,
            ),
            cpp["int_barrierupd_" + tag],
            reason=f"BarrierUPD {tag}",
        )

    def test_first_order_term_vanishes_for_flat_curves(self, cpp: dict[str, Any]) -> None:
        """With flat r, q and sigma and ``igm == 1``, ``gm`` is chosen so that
        ``alpha(t) - gm*0.5*sigmaq(t)`` is identically zero, so ``P_1`` is zero
        and order 1 collapses onto order 0. C++ records the same coincidence.
        """
        tight(
            float(cpp["int_barrierupd_o1"]),
            float(cpp["int_barrierupd_o0"]),
            reason="P_1 == 0 for flat curves",
        )
        # ...but not when gamma is forced to zero.
        assert float(cpp["int_barrierupd_o1_gm0"]) != float(
            cpp["int_barrierupd_o0_gm0"]
        )

    def test_sign_helper_treats_zero_as_negative(self) -> None:
        """# C++ parity: ``SIGN`` (.cpp:36-42) tests ``b > 0.0``, so ``b == 0``
        takes the negative branch — unlike Fortran's SIGN intrinsic."""
        assert _sign(1.0, 1.0) == 1.0
        assert _sign(1.0, -1.0) == -1.0
        assert _sign(1.0, 0.0) == -1.0
