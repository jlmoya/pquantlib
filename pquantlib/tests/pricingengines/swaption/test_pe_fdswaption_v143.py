"""Cross-validate the v1.43 finite-difference swaption engines against C++.

Reference: ``migration-harness/references/v143/pe/fdheston`` (probe at
``migration-harness/cpp/probes/v143_pe_fdheston/probe.cpp``) — the same probe
that backs ``tests/pricingengines/vanilla/test_pe_fdheston_v143.py``, because
both engine families are built out of the same FDM machinery.

Covers :class:`FdHullWhiteSwaptionEngine` (1-D, ``FdmHullWhiteSolver``) and
:class:`FdG2SwaptionEngine` (2-D, ``FdmG2Solver``), each on a real
:class:`VanillaSwap` with a Euribor6M leg forecast off a *different* curve from
the discounting one.

Why the assertions are exact
----------------------------
Both engines are deterministic backward rollbacks — no RNG, no clock, no
iterate-to-tolerance step — over a mesh whose locations are closed-form
Ornstein-Uhlenbeck quantiles, with the inner value coming from
``FdmAffineModelSwapInnerValue`` repricing the swap analytically at each node.
So the tier is **TIGHT** (1e-14 abs / 1e-12 rel), not a band; measured worst
case over the ten pricing cases is 3.4e-15 relative. These engines fill only
``results_.value``, so there is no theta cancellation to account for here.

Each case carries its whole market description in ``inputs``, so the tests
rebuild the setup rather than restating constants.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.daycounters.thirty_360 import Convention, Thirty360
from pquantlib.exceptions import LibraryException
from pquantlib.exercise import BermudanExercise, EuropeanExercise, Exercise
from pquantlib.indexes.ibor.euribor import Euribor
from pquantlib.instruments.swap import SwapType
from pquantlib.instruments.swaption import Swaption
from pquantlib.instruments.vanilla_swap import VanillaSwap
from pquantlib.methods.finitedifferences.schemes.fdm_scheme_desc import FdmSchemeDesc
from pquantlib.models.shortrate.onefactor.hull_white import HullWhite
from pquantlib.models.shortrate.twofactor.g2 import G2
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.pricingengines.swaption.fd_g2_swaption_engine import FdG2SwaptionEngine
from pquantlib.pricingengines.swaption.fd_hull_white_swaption_engine import (
    FdHullWhiteSwaptionEngine,
)
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.termstructures.yield_term_structure import YieldTermStructure
from pquantlib.testing import reference_reader
from pquantlib.testing.tolerance import tight
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendar import Calendar
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.date import Date
from pquantlib.time.date_generation import DateGeneration
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.schedule import Schedule
from pquantlib.time.time_unit import TimeUnit

# probe.cpp — ``const Date kToday(15, May, 2025);`` and
# ``Settings::instance().evaluationDate() = kToday;`` in ``main()``.
TODAY = Date.from_ymd(15, Month.May, 2025)

_SCHEMES = {
    "hundsdorfer": FdmSchemeDesc.hundsdorfer,
    "douglas": FdmSchemeDesc.douglas,
    "craigsneyd": FdmSchemeDesc.craig_sneyd,
}


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("v143/pe/fdheston")


@pytest.fixture(autouse=True)
def _eval_date() -> Iterator[None]:  # pyright: ignore[reportUnusedFunction]
    settings = ObservableSettings()
    saved = settings.evaluation_date
    # probe.cpp:main() — Settings::instance().evaluationDate() = Date(15, May, 2025)
    settings.evaluation_date = TODAY
    yield
    settings.evaluation_date = saved


# --- market reconstruction (mirrors probe.cpp ``describeRates``) -----------


def _calendar() -> Calendar:
    return TARGET()


def _flat_curve(rate: float) -> FlatForward:
    # probe.cpp ``ratesCurve`` — FlatForward(kToday, r, Actual365Fixed()).
    return FlatForward.from_rate(TODAY, rate, Actual365Fixed())


def _swaption_expiry() -> Date:
    # probe.cpp ``swaptionExpiry()`` — TARGET().advance(kToday, 2, Years).
    return _calendar().advance(TODAY, 2, TimeUnit.Years)


def _swap_start() -> Date:
    # probe.cpp ``swapStart()`` — TARGET().advance(swaptionExpiry(), 2, Days).
    return _calendar().advance(_swaption_expiry(), 2, TimeUnit.Days)


def _swap_end() -> Date:
    # probe.cpp ``swapEnd()`` — TARGET().advance(swapStart(), 3, Years).
    return _calendar().advance(_swap_start(), 3, TimeUnit.Years)


def _bermudan_dates() -> list[Date]:
    # probe.cpp ``bermudanDates()`` — 1y, 18m, then the 2y expiry.
    cal = _calendar()
    return [
        cal.advance(TODAY, 1, TimeUnit.Years),
        cal.advance(TODAY, 18, TimeUnit.Months),
        _swaption_expiry(),
    ]


def _schedule(tenor: Period) -> Schedule:
    # probe.cpp ``swapSchedule`` — ModifiedFollowing both ends, Backward, not EOM.
    return Schedule.from_rule(
        _swap_start(),
        _swap_end(),
        tenor,
        _calendar(),
        BusinessDayConvention.ModifiedFollowing,
        BusinessDayConvention.ModifiedFollowing,
        DateGeneration.Backward,
        False,
    )


def _make_swap(
    fwd_curve: YieldTermStructure, swap_type: SwapType, fixed_rate: float
) -> VanillaSwap:
    # probe.cpp ``makeSwap`` — notional 1.0, annual Thirty360(BondBasis) fixed
    # leg, semi-annual Actual360 Euribor6M float leg, zero spread.
    return VanillaSwap(
        swap_type,
        1.0,
        _schedule(Period(1, TimeUnit.Years)),
        fixed_rate,
        Thirty360(Convention.BondBasis),
        _schedule(Period(6, TimeUnit.Months)),
        Euribor(Period(6, TimeUnit.Months), fwd_curve),
        0.0,
        Actual360(),
    )


def _swap_type(inp: dict[str, Any]) -> SwapType:
    return SwapType.Payer if inp["swap_type"] == "Payer" else SwapType.Receiver


def _exercise(inp: dict[str, Any]) -> Exercise:
    if str(inp["exercise"]) == "bermudan":
        return BermudanExercise(_bermudan_dates())
    return EuropeanExercise(_swaption_expiry())


def _hull_white_swaption(inp: dict[str, Any]) -> Swaption:
    fwd = _flat_curve(float(inp["fwd_rate"]))
    disc = _flat_curve(float(inp["disc_rate"]))
    model = HullWhite(disc, float(inp["hw_a"]), float(inp["hw_sigma"]))
    swaption = Swaption(
        _make_swap(fwd, _swap_type(inp), float(inp["fixed_rate"])), _exercise(inp)
    )
    swaption.set_pricing_engine(
        FdHullWhiteSwaptionEngine(
            model,
            int(inp["t_grid"]),
            int(inp["x_grid"]),
            int(inp["damping_steps"]),
            float(inp["inv_eps"]),
            _SCHEMES[str(inp["scheme"])](),
        )
    )
    return swaption


def _g2_swaption(inp: dict[str, Any]) -> Swaption:
    fwd = _flat_curve(float(inp["fwd_rate"]))
    disc = _flat_curve(float(inp["disc_rate"]))
    model = G2(
        disc,
        float(inp["g2_a"]),
        float(inp["g2_sigma"]),
        float(inp["g2_b"]),
        float(inp["g2_eta"]),
        float(inp["g2_rho"]),
    )
    swaption = Swaption(
        _make_swap(fwd, _swap_type(inp), float(inp["fixed_rate"])), _exercise(inp)
    )
    swaption.set_pricing_engine(
        FdG2SwaptionEngine(
            model,
            int(inp["t_grid"]),
            int(inp["x_grid"]),
            int(inp["y_grid"]),
            int(inp["damping_steps"]),
            float(inp["inv_eps"]),
            _SCHEMES[str(inp["scheme"])](),
        )
    )
    return swaption


# --- FdHullWhiteSwaptionEngine --------------------------------------------

_HW_CASES = [
    "hw_swaption_payer",
    "hw_swaption_receiver",
    "hw_swaption_payer_otm",
    "hw_swaption_payer_bermudan",
    "hw_swaption_payer_damped",
]


@pytest.mark.parametrize("case", _HW_CASES)
def test_fd_hull_white_swaption_engine(cpp: dict[str, Any], case: str) -> None:
    inp, exp = cpp[case]["inputs"], cpp[case]["expected"]
    tight(_hull_white_swaption(inp).npv(), float(exp["npv"]))


def _hull_white_npv_with_scheme(
    inp: dict[str, Any], scheme: FdmSchemeDesc | None
) -> float:
    disc = _flat_curve(float(inp["disc_rate"]))
    model = HullWhite(disc, float(inp["hw_a"]), float(inp["hw_sigma"]))
    swaption = Swaption(
        _make_swap(
            _flat_curve(float(inp["fwd_rate"])), SwapType.Payer, float(inp["fixed_rate"])
        ),
        EuropeanExercise(_swaption_expiry()),
    )
    swaption.set_pricing_engine(
        FdHullWhiteSwaptionEngine(
            model,
            int(inp["t_grid"]),
            int(inp["x_grid"]),
            int(inp["damping_steps"]),
            float(inp["inv_eps"]),
            scheme,
        )
    )
    return swaption.npv()


def test_hull_white_default_scheme_is_douglas(cpp: dict[str, Any]) -> None:
    """# C++ parity: ``schemeDesc = FdmSchemeDesc::Douglas()`` in the header.

    Every Heston engine defaults to Hundsdorfer; this one does not. Checked
    behaviourally: the default-constructed engine must reproduce the pinned
    Douglas value and must *not* reproduce the Hundsdorfer one.
    """
    inp = cpp["hw_swaption_payer"]["inputs"]
    assert inp["scheme"] == "douglas"
    default_npv = _hull_white_npv_with_scheme(inp, None)
    tight(default_npv, float(cpp["hw_swaption_payer"]["expected"]["npv"]))
    hundsdorfer_npv = _hull_white_npv_with_scheme(inp, FdmSchemeDesc.hundsdorfer())
    assert default_npv != hundsdorfer_npv


def test_hull_white_payer_receiver_and_moneyness(cpp: dict[str, Any]) -> None:
    """Payer, receiver and an out-of-the-money payer are three distinct numbers.

    ``arguments_.swap`` reaches ``FdmAffineModelSwapInnerValue``, so flipping
    the swap direction or moving the fixed rate has to move the answer; an
    engine that ignored the swap would return the same value for all three.
    """
    payer = float(cpp["hw_swaption_payer"]["expected"]["npv"])
    receiver = float(cpp["hw_swaption_receiver"]["expected"]["npv"])
    otm = float(cpp["hw_swaption_payer_otm"]["expected"]["npv"])
    assert payer > receiver > otm > 0.0
    assert payer - receiver > 1e-3
    assert payer / otm > 5.0


def test_hull_white_bermudan_dominates_european(cpp: dict[str, Any]) -> None:
    """The Bermudan schedule reaches ``FdmBermudanStepCondition``.

    Its two extra dates (1y, 18m) are deep out of the money, so the premium is
    only 8.5e-9 — but it is strictly positive and 6 orders of magnitude above
    the 1e-14 absolute tolerance, so a dropped step condition is still caught.
    """
    european = float(cpp["hw_swaption_payer"]["expected"]["npv"])
    bermudan = float(cpp["hw_swaption_payer_bermudan"]["expected"]["npv"])
    assert bermudan > european
    assert bermudan - european > 1e-9


def test_hull_white_throws_on_day_counter_mismatch(cpp: dict[str, Any]) -> None:
    exp = cpp["hw_swaption_throw_daycounter"]["expected"]
    assert exp["throws"] is True
    inp = cpp["hw_swaption_payer"]["inputs"]
    fwd = FlatForward.from_rate(TODAY, float(inp["fwd_rate"]), Actual360())
    disc = _flat_curve(float(inp["disc_rate"]))
    swaption = Swaption(
        _make_swap(fwd, SwapType.Payer, float(inp["fixed_rate"])),
        EuropeanExercise(_swaption_expiry()),
    )
    swaption.set_pricing_engine(
        FdHullWhiteSwaptionEngine(
            HullWhite(disc, float(inp["hw_a"]), float(inp["hw_sigma"])), 20, 21
        )
    )
    with pytest.raises(LibraryException, match=exp["what"]):
        swaption.npv()


def test_hull_white_throws_on_reference_date_mismatch(cpp: dict[str, Any]) -> None:
    exp = cpp["hw_swaption_throw_refdate"]["expected"]
    assert exp["throws"] is True
    inp = cpp["hw_swaption_payer"]["inputs"]
    fwd = FlatForward.from_rate(TODAY + 1, float(inp["fwd_rate"]), Actual365Fixed())
    disc = _flat_curve(float(inp["disc_rate"]))
    swaption = Swaption(
        _make_swap(fwd, SwapType.Payer, float(inp["fixed_rate"])),
        EuropeanExercise(_swaption_expiry()),
    )
    swaption.set_pricing_engine(
        FdHullWhiteSwaptionEngine(
            HullWhite(disc, float(inp["hw_a"]), float(inp["hw_sigma"])), 20, 21
        )
    )
    with pytest.raises(LibraryException, match=exp["what"]):
        swaption.npv()


# --- FdG2SwaptionEngine ---------------------------------------------------

_G2_CASES = [
    "g2_swaption_payer",
    "g2_swaption_receiver",
    "g2_swaption_payer_otm",
    "g2_swaption_payer_bermudan",
    "g2_swaption_payer_damped",
]


@pytest.mark.parametrize("case", _G2_CASES)
def test_fd_g2_swaption_engine(cpp: dict[str, Any], case: str) -> None:
    inp, exp = cpp[case]["inputs"], cpp[case]["expected"]
    tight(_g2_swaption(inp).npv(), float(exp["npv"]))


def _g2_npv_with_scheme(inp: dict[str, Any], scheme: FdmSchemeDesc | None) -> float:
    model = G2(
        _flat_curve(float(inp["disc_rate"])),
        float(inp["g2_a"]),
        float(inp["g2_sigma"]),
        float(inp["g2_b"]),
        float(inp["g2_eta"]),
        float(inp["g2_rho"]),
    )
    swaption = Swaption(
        _make_swap(
            _flat_curve(float(inp["fwd_rate"])), SwapType.Payer, float(inp["fixed_rate"])
        ),
        EuropeanExercise(_swaption_expiry()),
    )
    swaption.set_pricing_engine(
        FdG2SwaptionEngine(
            model,
            int(inp["t_grid"]),
            int(inp["x_grid"]),
            int(inp["y_grid"]),
            int(inp["damping_steps"]),
            float(inp["inv_eps"]),
            scheme,
        )
    )
    return swaption.npv()


def test_g2_default_scheme_is_hundsdorfer(cpp: dict[str, Any]) -> None:
    """# C++ parity: ``schemeDesc = FdmSchemeDesc::Hundsdorfer()`` in the header.

    Checked behaviourally, as for the Hull-White engine.
    """
    inp = cpp["g2_swaption_payer"]["inputs"]
    assert inp["scheme"] == "hundsdorfer"
    default_npv = _g2_npv_with_scheme(inp, None)
    tight(default_npv, float(cpp["g2_swaption_payer"]["expected"]["npv"]))
    douglas_npv = _g2_npv_with_scheme(inp, FdmSchemeDesc.douglas())
    assert default_npv != douglas_npv


def test_g2_payer_receiver_and_moneyness(cpp: dict[str, Any]) -> None:
    payer = float(cpp["g2_swaption_payer"]["expected"]["npv"])
    receiver = float(cpp["g2_swaption_receiver"]["expected"]["npv"])
    otm = float(cpp["g2_swaption_payer_otm"]["expected"]["npv"])
    assert payer > receiver > otm > 0.0
    assert payer - receiver > 1e-3
    assert payer / otm > 10.0


def test_g2_bermudan_dominates_european(cpp: dict[str, Any]) -> None:
    """The 2-D Bermudan premium is 1.1% of the European value."""
    european = float(cpp["g2_swaption_payer"]["expected"]["npv"])
    bermudan = float(cpp["g2_swaption_payer_bermudan"]["expected"]["npv"])
    assert bermudan > european
    assert (bermudan - european) / european > 0.01


def test_g2_throws_on_day_counter_mismatch(cpp: dict[str, Any]) -> None:
    exp = cpp["g2_swaption_throw_daycounter"]["expected"]
    assert exp["throws"] is True
    inp = cpp["g2_swaption_payer"]["inputs"]
    fwd = FlatForward.from_rate(TODAY, float(inp["fwd_rate"]), Actual360())
    disc = _flat_curve(float(inp["disc_rate"]))
    model = G2(
        disc,
        float(inp["g2_a"]),
        float(inp["g2_sigma"]),
        float(inp["g2_b"]),
        float(inp["g2_eta"]),
        float(inp["g2_rho"]),
    )
    swaption = Swaption(
        _make_swap(fwd, SwapType.Payer, float(inp["fixed_rate"])),
        EuropeanExercise(_swaption_expiry()),
    )
    swaption.set_pricing_engine(FdG2SwaptionEngine(model, 15, 11, 11))
    with pytest.raises(LibraryException, match=exp["what"]):
        swaption.npv()


# --- market-setup guard ---------------------------------------------------


def test_market_setup_matches_probe(cpp: dict[str, Any]) -> None:
    """Every constant this file rebuilds is the one the probe recorded."""
    inp = cpp["hw_swaption_payer"]["inputs"]
    assert (int(inp["eval_year"]), int(inp["eval_month"]), int(inp["eval_day"])) == (
        TODAY.year(),
        int(TODAY.month()),
        TODAY.day_of_month(),
    )
    assert inp["day_counter"] == "Actual365Fixed"
    assert inp["calendar"] == "TARGET"
    assert inp["index"] == "Euribor6M"
    assert inp["fixed_day_counter"] == "Thirty360BondBasis"
    assert inp["float_day_counter"] == "Actual360"
    assert float(inp["notional"]) == 1.0
    assert int(inp["expiry_years"]) == 2
    assert int(inp["swap_tenor_years"]) == 3
    # The schedules the probe describes, rebuilt.
    assert _swaption_expiry() == _calendar().advance(TODAY, 2, TimeUnit.Years)
    assert _swap_end() == _calendar().advance(_swap_start(), 3, TimeUnit.Years)
    assert len(_bermudan_dates()) == 3
