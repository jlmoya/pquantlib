"""Cross-validate AssetSwapHelper against C++ QuantLib v1.43.

Probe source: migration-harness/cpp/probes/v143_experimental_creditts/probe.cpp
              (block E)
Reference:    migration-harness/references/v143/experimental/creditts.json

# C++ parity: ql/experimental/credit/riskyassetswap.{hpp,cpp} — ``class
# AssetSwapHelper : public DefaultProbabilityHelper`` (riskyassetswap.hpp:89).

The probe builds four helpers (2Y/3Y/5Y/7Y) on a flat 3% continuous curve,
bootstraps a ``PiecewiseDefaultCurve<HazardRate, BackwardFlat>`` off them (the
Python default interpolator for ``HazardRateTrait`` is already BackwardFlat), and
emits the helper dates, the bootstrapped pillars, and the round-tripped
implied quotes.
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.daycounters.thirty_360 import Convention as Thirty360Convention
from pquantlib.daycounters.thirty_360 import Thirty360
from pquantlib.experimental.credit.risky_asset_swap import AssetSwapHelper
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.bootstrap_helper import BootstrapHelper
from pquantlib.termstructures.credit.default_probability_term_structure import (
    DefaultProbabilityTermStructure,
)
from pquantlib.termstructures.credit.piecewise_default_curve import PiecewiseDefaultCurve
from pquantlib.termstructures.credit.probability_traits import HazardRateTrait
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing import reference_reader, tolerance
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.compounding import Compounding
from pquantlib.time.date import Date
from pquantlib.time.frequency import Frequency
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit

_EVAL_DATE = Date.from_ymd(15, Month.January, 2024)
_RECOVERY = 0.4
_TENORS = [
    Period(2, TimeUnit.Years),
    Period(3, TimeUnit.Years),
    Period(5, TimeUnit.Years),
    Period(7, TimeUnit.Years),
]
_SPREADS = [0.0060, 0.0080, 0.0110, 0.0135]


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("v143/experimental/creditts")


@pytest.fixture(autouse=True)
def _set_eval_date() -> None:  # pyright: ignore[reportUnusedFunction]
    ObservableSettings().evaluation_date = _EVAL_DATE


def _helpers() -> list[BootstrapHelper[DefaultProbabilityTermStructure]]:
    """The probe's four helpers, argument for argument (probe.cpp:521-530)."""
    cal = TARGET()
    yts = FlatForward(
        _EVAL_DATE,
        SimpleQuote(0.03),
        Actual365Fixed(),
        Compounding.Continuous,
        Frequency.Annual,
    )
    return [
        AssetSwapHelper(
            SimpleQuote(spread),
            tenor,
            2,
            cal,
            Period(1, TimeUnit.Years),
            BusinessDayConvention.Following,
            Thirty360(Thirty360Convention.BondBasis),
            Period(3, TimeUnit.Months),
            BusinessDayConvention.Following,
            Actual360(),
            _RECOVERY,
            yts,
            Period(1, TimeUnit.Years),
        )
        for tenor, spread in zip(_TENORS, _SPREADS, strict=True)
    ]


def test_asset_swap_helper_dates(cpp: dict[str, Any]) -> None:
    """``initializeDates`` (riskyassetswap.cpp:232-259) — before any curve."""
    helpers = _helpers()
    for h, expected in zip(helpers, cpp["asw_helper_earliest_dates"], strict=True):
        assert h.earliest_date().serial_number() == int(expected)
    for h, expected in zip(helpers, cpp["asw_helper_latest_dates"], strict=True):
        assert h.latest_date().serial_number() == int(expected)
    for h, expected in zip(helpers, cpp["asw_helper_maturity_dates"], strict=True):
        assert h.maturity_date().serial_number() == int(expected)
    for h, expected in zip(helpers, cpp["asw_helper_pillar_dates"], strict=True):
        assert h.pillar_date().serial_number() == int(expected)


def test_asset_swap_helper_quotes(cpp: dict[str, Any]) -> None:
    for h, expected in zip(_helpers(), cpp["asw_helper_quotes"], strict=True):
        tolerance.exact(h.quote().value(), float(expected))


def test_asset_swap_helper_requires_a_curve() -> None:
    """``impliedQuote`` QL_REQUIREs the handle (riskyassetswap.cpp:208-209)."""
    from pquantlib.exceptions import LibraryException  # noqa: PLC0415

    with pytest.raises(LibraryException, match="default term structure not set"):
        _helpers()[0].implied_quote()


def _bootstrapped() -> PiecewiseDefaultCurve:
    curve = PiecewiseDefaultCurve(
        HazardRateTrait,
        _EVAL_DATE,
        _helpers(),
        Actual365Fixed(),
    )
    curve.enable_extrapolation()
    return curve


def test_asset_swap_helper_bootstraps_a_hazard_curve(cpp: dict[str, Any]) -> None:
    """The pillars C++'s ``PiecewiseDefaultCurve<HazardRate, BackwardFlat>``
    lands on when fed these four helpers (probe.cpp:544-551).

    Both sides run the same Brent solve to the same 1e-12 quote accuracy and
    land within 2.7e-15 relative of each other, so this is TIGHT.
    """
    curve = _bootstrapped()
    for d, expected in zip(curve.dates(), cpp["asw_curve_dates"], strict=True):
        assert d.serial_number() == int(expected)
    for t, expected in zip(curve.times(), cpp["asw_curve_times"], strict=True):
        tolerance.tight(float(t), float(expected))
    for v, expected in zip(curve.data(), cpp["asw_curve_data"], strict=True):
        tolerance.tight(float(v), float(expected))


def test_asset_swap_helper_implied_quotes_round_trip(cpp: dict[str, Any]) -> None:
    """After the bootstrap each helper reprices its own quote.

    C++'s own round-trip residuals are up to 5.3e-14 absolute
    (``asw_helper_implied_quotes`` vs ``asw_helper_quotes``), i.e. the
    bootstrap's own accuracy, so this compares against what C++ produced
    rather than against the input spreads.
    """
    curve = _bootstrapped()
    curve.survival_probability(_EVAL_DATE + Period(1, TimeUnit.Years), True)
    for h, expected in zip(
        curve.instruments(), cpp["asw_helper_implied_quotes"], strict=True
    ):
        tolerance.tight(h.implied_quote(), float(expected))


def test_asset_swap_curve_sampled(cpp: dict[str, Any]) -> None:
    curve = _bootstrapped()
    dates = [Date(int(s)) for s in cpp["asw_sample_dates"]]
    for d, expected in zip(dates, cpp["asw_sample_survival"], strict=True):
        tolerance.tight(curve.survival_probability(d, True), float(expected))
    for d, expected in zip(dates, cpp["asw_sample_hazard_rate"], strict=True):
        tolerance.tight(curve.hazard_rate(d, True), float(expected))
