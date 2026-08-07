"""Cross-validate CdsOption + BlackCdsOptionEngine against C++ QuantLib v1.43.

Probe source: migration-harness/cpp/probes/v143_experimental_cdsntd/probe.cpp
              (section A, ``sectionCdsOption``)
Reference:    migration-harness/references/v143/experimental/cdsntd.json

C++ classes:
    ``CdsOption``            ql/experimental/credit/cdsoption.hpp:43
    ``BlackCdsOptionEngine`` ql/experimental/credit/blackcdsoptionengine.hpp:36

The pre-existing ``test_black_cds_option_engine.py`` pins three numbers from
an earlier cluster probe. This file replaces that coverage with six market
configurations, both protection sides, the knock-out and non-knock-out payer,
and — new at this wave — ``implied_volatility``, which the port did not have
until the preceding align commit.

Evaluation date
---------------
The probe sets ``Settings::instance().evaluationDate() = Date(15, January,
2024)`` (probe.cpp ``sectionCdsOption``, ``A_TODAY``). The fixture pins the
same date and restores the previous value in teardown.

Tolerance
---------
LOOSE. The chain is schedule generation -> MidPointCdsEngine (a survival-
probability-weighted sum over coupon midpoints) -> ``fairSpread`` ->
``blackFormula``. Every step is a faithful port but the accumulation order
inside the CDS engine is not bit-reproducible across the two languages, and
the option value is a difference of large discounted sums.
``implied_volatility`` additionally inherits Brent's own accuracy, which the
probe requests at 1e-10 for the ``from_lo`` / ``from_hi`` cases.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exceptions import LibraryException
from pquantlib.exercise import EuropeanExercise
from pquantlib.instruments.cds_option import CdsOption
from pquantlib.instruments.credit_default_swap import (
    CreditDefaultSwap,
    ProtectionSide,
)
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.pricingengines.credit.black_cds_option_engine import (
    BlackCdsOptionEngine,
)
from pquantlib.pricingengines.credit.midpoint_cds_engine import MidPointCdsEngine
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.credit.flat_hazard_rate import FlatHazardRate
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing import reference_reader, tolerance
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.compounding import Compounding
from pquantlib.time.date import Date
from pquantlib.time.date_generation import DateGeneration
from pquantlib.time.frequency import Frequency
from pquantlib.time.period import Period
from pquantlib.time.schedule import MakeSchedule
from pquantlib.time.time_unit import TimeUnit

_CASES = [
    "cdso_payer_ko",
    "cdso_payer_nko",
    "cdso_recv_ko",
    "cdso_payer_otm",
    "cdso_payer_itm",
    "cdso_payer_lowvol",
]


@pytest.fixture(scope="module")
def cpp_ref() -> dict[str, Any]:
    return reference_reader.load("v143/experimental/cdsntd")


@pytest.fixture(autouse=True)
def _pinned_evaluation_date(cpp_ref: dict[str, Any]) -> Iterator[None]:  # pyright: ignore[reportUnusedFunction]
    """Probe: ``Settings::instance().evaluationDate() = A_TODAY``.

    probe.cpp ``sectionCdsOption``, first statement.
    """
    settings = ObservableSettings()
    previous = settings.evaluation_date
    settings.evaluation_date = Date(int(cpp_ref["a_evaluationDate_serial"]))
    yield
    settings.evaluation_date = previous


class _Built:
    __slots__ = ("dts", "option", "underlying", "vol_quote", "yts")

    def __init__(
        self,
        option: CdsOption,
        underlying: CreditDefaultSwap,
        yts: FlatForward,
        dts: FlatHazardRate,
        vol_quote: SimpleQuote,
    ) -> None:
        self.option = option
        self.underlying = underlying
        self.yts = yts
        self.dts = dts
        self.vol_quote = vol_quote


def _build(cpp_ref: dict[str, Any], tag: str) -> _Built:
    """Rebuild the probe's fixture for ``tag`` from the pinned inputs."""
    today = Date(int(cpp_ref["a_evaluationDate_serial"]))
    hazard = cpp_ref[f"{tag}_hazard"]
    recovery = cpp_ref[f"{tag}_recovery"]
    flat_rate = cpp_ref[f"{tag}_flatRate"]

    yts = FlatForward(
        today,
        SimpleQuote(flat_rate),
        Actual365Fixed(),
        Compounding.Continuous,
        Frequency.Annual,
    )
    dts = FlatHazardRate(today, SimpleQuote(hazard), Actual365Fixed())

    start = Date(int(cpp_ref[f"{tag}_cdsStart_serial"]))
    end = Date(int(cpp_ref[f"{tag}_cdsEnd_serial"]))
    sched = (
        MakeSchedule()
        .from_date(start)
        .to(end)
        .with_tenor(Period(3, TimeUnit.Months))
        .with_calendar(TARGET())
        .with_termination_date_convention(BusinessDayConvention.Unadjusted)
        .with_rule(DateGeneration.CDS)
        .build()
    )
    side = ProtectionSide(int(cpp_ref[f"{tag}_side"]))
    cds = CreditDefaultSwap(
        side=side,
        notional=cpp_ref[f"{tag}_notional"],
        spread=cpp_ref[f"{tag}_spread"],
        schedule=sched,
        payment_convention=BusinessDayConvention.Following,
        day_counter=Actual360(),
        settles_accrual=True,
        pays_at_default_time=True,
        protection_start=start,
    )
    cds.set_pricing_engine(MidPointCdsEngine(dts, recovery, yts))

    vol_quote = SimpleQuote(cpp_ref[f"{tag}_vol"])
    option = CdsOption(
        underlying=cds,
        exercise=EuropeanExercise(Date(int(cpp_ref[f"{tag}_exercise_serial"]))),
        knocks_out=bool(cpp_ref[f"{tag}_knocksOut"]),
    )
    option.set_pricing_engine(
        BlackCdsOptionEngine(dts, recovery, yts, vol_quote)
    )
    return _Built(option, cds, yts, dts, vol_quote)


@pytest.mark.parametrize("tag", _CASES)
def test_underlying_matches_cpp(cpp_ref: dict[str, Any], tag: str) -> None:
    """The three CDS quantities the option engine reads.

    ``blackcdsoptionengine.cpp:50-58`` reads ``fairSpread()``,
    ``runningSpread()`` and ``couponLegNPV()``. If the underlying disagrees,
    the option value cannot agree for the right reason, so this runs first.
    """
    b = _build(cpp_ref, tag)
    tolerance.loose(b.underlying.fair_spread(), cpp_ref[f"{tag}_underlying_fairSpread"])
    tolerance.loose(
        b.underlying.coupon_leg_npv(), cpp_ref[f"{tag}_underlying_couponLegNPV"]
    )
    tolerance.loose(b.underlying.npv(), cpp_ref[f"{tag}_underlying_NPV"])


@pytest.mark.parametrize("tag", _CASES)
def test_option_npv_and_risky_annuity_match_cpp(
    cpp_ref: dict[str, Any], tag: str
) -> None:
    """``BlackCdsOptionEngine::calculate`` (blackcdsoptionengine.cpp:42-80).

    ``cdso_payer_nko`` is the only case that reaches the front-end-protection
    branch, which fires on ``side == Buyer and not knocksOut``.
    """
    b = _build(cpp_ref, tag)
    tolerance.loose(b.option.npv(), cpp_ref[f"{tag}_NPV"])
    tolerance.loose(b.option.risky_annuity(), cpp_ref[f"{tag}_riskyAnnuity"])
    tolerance.loose(b.option.atm_rate(), cpp_ref[f"{tag}_atmRate"])
    assert b.option.is_expired() == bool(cpp_ref[f"{tag}_isExpired"])


def test_front_end_protection_is_the_only_difference(cpp_ref: dict[str, Any]) -> None:
    """Knock-out and non-knock-out share every input but the flag.

    So their C++ NPVs differ by exactly the front-end-protection term
    ``notional * (1 - R) * P_default(exercise) * D(exercise)``
    (blackcdsoptionengine.cpp:72-79). Pinning the DIFFERENCE catches a port
    that gets the term right but adds it on the wrong branch.
    """
    ko = _build(cpp_ref, "cdso_payer_ko")
    nko = _build(cpp_ref, "cdso_payer_nko")
    exercise_date = Date(int(cpp_ref["cdso_payer_nko_exercise_serial"]))
    expected_term = (
        cpp_ref["cdso_payer_nko_notional"]
        * (1.0 - cpp_ref["cdso_payer_nko_recovery"])
        * nko.dts.default_probability(exercise_date)
        * nko.yts.discount(exercise_date)
    )
    tolerance.loose(nko.option.npv() - ko.option.npv(), expected_term)
    tolerance.loose(
        cpp_ref["cdso_payer_nko_NPV"] - cpp_ref["cdso_payer_ko_NPV"], expected_term
    )


@pytest.mark.parametrize("tag", _CASES)
def test_implied_volatility_round_trips(cpp_ref: dict[str, Any], tag: str) -> None:
    """``CdsOption::impliedVolatility`` (cdsoption.cpp:120-139).

    The probe's targets come from RE-PRICING the option at half and twice the
    input vol, not from scaling its NPV: an arbitrary scaling can fall below
    intrinsic value, where Brent legitimately reports "root not bracketed"
    (it does for ``cdso_payer_itm``). Re-priced targets always have a root and
    they test the stronger property — that the solve inverts the engine.
    """
    b = _build(cpp_ref, tag)
    npv = b.option.npv()
    tolerance.loose(
        b.option.implied_volatility(npv, b.yts, b.dts, cpp_ref[f"{tag}_recovery"]),
        cpp_ref[f"{tag}_impliedVol_roundtrip"],
    )
    tolerance.loose(
        b.option.implied_volatility(
            npv, b.yts, b.dts, cpp_ref[f"{tag}_recovery"], 1e-10, 500
        ),
        cpp_ref[f"{tag}_impliedVol_tight"],
    )


@pytest.mark.parametrize("tag", _CASES)
def test_implied_volatility_inverts_the_engine(
    cpp_ref: dict[str, Any], tag: str
) -> None:
    """Imply back the vols the probe used to generate off-market targets."""
    b = _build(cpp_ref, tag)
    recovery = cpp_ref[f"{tag}_recovery"]

    b.vol_quote.set_value(cpp_ref[f"{tag}_volLo"])
    npv_lo = b.option.npv()
    tolerance.loose(npv_lo, cpp_ref[f"{tag}_NPV_at_volLo"])
    b.vol_quote.set_value(cpp_ref[f"{tag}_volHi"])
    npv_hi = b.option.npv()
    tolerance.loose(npv_hi, cpp_ref[f"{tag}_NPV_at_volHi"])
    b.vol_quote.set_value(cpp_ref[f"{tag}_vol"])

    tolerance.loose(
        b.option.implied_volatility(npv_lo, b.yts, b.dts, recovery, 1e-10, 500),
        cpp_ref[f"{tag}_impliedVol_from_lo"],
    )
    tolerance.loose(
        b.option.implied_volatility(npv_hi, b.yts, b.dts, recovery, 1e-10, 500),
        cpp_ref[f"{tag}_impliedVol_from_hi"],
    )


def test_receiver_must_knock_out(cpp_ref: dict[str, Any]) -> None:
    """C++ ctor: ``QL_REQUIRE(swap->side() == Buyer || knocksOut_, ...)``.

    cdsoption.cpp:74-75. The probe pins the message text.
    """
    b = _build(cpp_ref, "cdso_recv_ko")
    with pytest.raises(LibraryException) as exc:
        CdsOption(
            underlying=b.underlying,
            exercise=EuropeanExercise(
                Date(int(cpp_ref["cdso_recv_ko_exercise_serial"]))
            ),
            knocks_out=False,
        )
    assert cpp_ref["cdso_reject_receiver_nko"] in str(exc.value)
