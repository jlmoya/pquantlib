"""Cross-validate the base-correlation cluster against C++ QuantLib v1.43.

Covers ``BaseCorrelationTermStructure`` (basecorrelationstructure.hpp/.cpp) and
``BaseCorrelationLossModel`` (basecorrelationlossmodel.hpp).

Probe source: migration-harness/cpp/probes/v143_experimental_creditloss/probe.cpp
Reference:    migration-harness/references/v143/experimental/creditloss.json

The surface is deliberately pinned with an ASYMMETRIC quote matrix. A symmetric
one would hide the row/column transposition C++ performs between
``updateMatrix`` and ``setupInterpolation`` — a defect this port reproduces
verbatim, so the test asserts the transposed reading on purpose.
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exceptions import LibraryException
from pquantlib.experimental.credit.base_correlation_loss_model import (
    BaseCorrelationLossModel,
    GaussianLHPFlatBCLM,
)
from pquantlib.experimental.credit.base_correlation_structure import (
    BaseCorrelationTermStructure,
)
from pquantlib.math.array import Array
from pquantlib.math.interpolations.bicubic_spline import BicubicSpline
from pquantlib.math.matrix import Matrix
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.testing import tolerance
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.date import Date
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit
from tests.experimental.credit import _creditloss_fixture as fx

#: probe.cpp:471 / :920 — tenors shared by both blocks.
TENORS = [
    Period(1, TimeUnit.Years),
    Period(3, TimeUnit.Years),
    Period(5, TimeUnit.Years),
]
#: probe.cpp:472 (block C) and :921 (block J) — deliberately different.
BCTS_LOSS_LEVELS = [0.03, 0.07, 0.15]
BCLM_LOSS_LEVELS = [0.03, 0.06, 0.15]
#: probe.cpp:476 — asymmetric on purpose; see the module docstring.
BCTS_QUOTES = [[0.10, 0.20, 0.30], [0.40, 0.50, 0.60], [0.70, 0.80, 0.90]]
#: probe.cpp:922.
BCLM_QUOTES = [[0.20, 0.30, 0.45], [0.25, 0.35, 0.50], [0.30, 0.40, 0.55]]


@pytest.fixture(scope="module")
def cpp_ref() -> dict[str, Any]:
    return fx.load_reference()


def _bicubic_factory(xs: Array, ys: Array, z: Matrix) -> BicubicSpline:
    """The C++ ``BaseCorrelationTermStructure<BicubicSpline>`` specialisation.

    # C++ parity: basecorrelationstructure.cpp:40-47.
    """
    return BicubicSpline(xs, ys, z)


def _make_surface(
    loss_levels: list[float],
    raw: list[list[float]],
    *,
    bicubic: bool = False,
) -> tuple[BaseCorrelationTermStructure, list[list[SimpleQuote]]]:
    fx.build_fixture()  # pins the global evaluation date
    cal = TARGET()
    quotes = [[SimpleQuote(v) for v in row] for row in raw]
    factory = _bicubic_factory if bicubic else None
    surface = BaseCorrelationTermStructure(
        0,
        cal,
        BusinessDayConvention.Following,
        TENORS,
        loss_levels,
        quotes,
        Actual365Fixed(),
        factory,
    )
    return surface, quotes


# -----------------------------------------------------------------------------
# BaseCorrelationTermStructure
# -----------------------------------------------------------------------------


def test_surface_shape_and_dates_match_cpp(cpp_ref: dict[str, Any]) -> None:
    surface, _ = _make_surface(BCTS_LOSS_LEVELS, BCTS_QUOTES)
    assert surface.correlation_size() == cpp_ref["bcts_correlation_size"]
    assert surface.max_date().serial_number() == cpp_ref["bcts_max_date_serial"]
    assert (
        surface.reference_date().serial_number()
        == cpp_ref["bcts_reference_date_serial"]
    )
    for t, expected in zip(
        surface.tranche_times(), cpp_ref["bcts_tranche_times"], strict=True
    ):
        tolerance.tight(t, expected)


def test_surface_on_node_reproduces_cpp_transposition(cpp_ref: dict[str, Any]) -> None:
    """The DEFECT test: on-node lookups come back transposed, as in C++.

    Quotes are documented as ``correls[iYear][iLoss]`` but read back as
    ``correls[iLoss][iYear]``; walking (tenor-major, loss-minor) over the
    matrix ``{{.1,.2,.3},{.4,.5,.6},{.7,.8,.9}}`` therefore yields
    ``[.1,.4,.7, .2,.5,.8, .3,.6,.9]`` and not ``[.1,.2,.3, ...]``.
    """
    surface, _ = _make_surface(BCTS_LOSS_LEVELS, BCTS_QUOTES)
    values = [
        surface.correlation_at_time(t, level)
        for t in surface.tranche_times()
        for level in BCTS_LOSS_LEVELS
    ]
    for actual, expected in zip(values, cpp_ref["bcts_on_node"], strict=True):
        tolerance.tight(actual, expected)
    # Guard the guard: the *documented* layout would give this instead.
    documented = [v for row in BCTS_QUOTES for v in row]
    assert values != documented


def test_surface_off_node_and_extrapolation_match_cpp(cpp_ref: dict[str, Any]) -> None:
    surface, _ = _make_surface(BCTS_LOSS_LEVELS, BCTS_QUOTES)
    values = [
        surface.correlation_at_time(t, level)
        for t in cpp_ref["bcts_off_t"]
        for level in cpp_ref["bcts_off_l"]
    ]
    for actual, expected in zip(values, cpp_ref["bcts_off_node"], strict=True):
        tolerance.tight(actual, expected)


def test_surface_date_and_time_overloads_agree(cpp_ref: dict[str, Any]) -> None:
    surface, _ = _make_surface(BCTS_LOSS_LEVELS, BCTS_QUOTES)
    cal = TARGET()
    d3 = cal.advance_period(
        surface.reference_date(),
        Period(2, TimeUnit.Years),
        BusinessDayConvention.Following,
    )
    by_date = surface.correlation(d3, 0.07)
    by_time = surface.correlation_at_time(surface.time_from_reference(d3), 0.07)
    tolerance.tight(by_date, cpp_ref["bcts_by_date_2y_l007"])
    tolerance.tight(by_time, cpp_ref["bcts_by_time_2y_l007"])
    tolerance.exact(by_date, by_time)


def test_surface_quote_bump_propagates(cpp_ref: dict[str, Any]) -> None:
    surface, quotes = _make_surface(BCTS_LOSS_LEVELS, BCTS_QUOTES)
    quotes[0][2].set_value(0.99)
    values = [
        surface.correlation_at_time(t, level)
        for t in surface.tranche_times()
        for level in BCTS_LOSS_LEVELS
    ]
    for actual, expected in zip(
        values, cpp_ref["bcts_on_node_after_bump_00_02_to_099"], strict=True
    ):
        tolerance.tight(actual, expected)


def test_surface_bicubic_matches_cpp(cpp_ref: dict[str, Any]) -> None:
    surface, _ = _make_surface(BCTS_LOSS_LEVELS, BCTS_QUOTES, bicubic=True)
    on_node = [
        surface.correlation_at_time(t, level)
        for t in surface.tranche_times()
        for level in BCTS_LOSS_LEVELS
    ]
    off_node = [
        surface.correlation_at_time(t, level)
        for t in cpp_ref["bcts_off_t"]
        for level in cpp_ref["bcts_off_l"]
    ]
    for actual, expected in zip(on_node, cpp_ref["bcts_bicubic_on_node"], strict=True):
        tolerance.tight(actual, expected)
    for actual, expected in zip(
        off_node, cpp_ref["bcts_bicubic_off_node"], strict=True
    ):
        tolerance.tight(actual, expected)


def test_surface_rejects_non_square_matrix(cpp_ref: dict[str, Any]) -> None:
    """The other DEFECT: ``checkInputs`` compares transposed dimensions.

    A 2-tenor by 3-loss surface is perfectly well formed, yet C++ throws.
    """
    assert cpp_ref["bcts_defect_non_square_throws"] is True
    fx.build_fixture()
    cal = TARGET()
    with pytest.raises(LibraryException, match="mismatch between number of"):
        BaseCorrelationTermStructure(
            0,
            cal,
            BusinessDayConvention.Following,
            TENORS[:2],
            BCTS_LOSS_LEVELS,
            [[SimpleQuote(0.3) for _ in range(3)] for _ in range(2)],
            Actual365Fixed(),
        )


def test_surface_constructor_does_not_call_check_losses() -> None:
    """# C++ parity: the ctor (hpp:77-88) never calls ``checkLosses`` (:142)."""
    fx.build_fixture()
    cal = TARGET()
    unsorted_levels = [0.15, 0.07, 0.03]
    surface = BaseCorrelationTermStructure(
        0,
        cal,
        BusinessDayConvention.Following,
        TENORS,
        unsorted_levels,
        [[SimpleQuote(v) for v in row] for row in BCTS_QUOTES],
        Actual365Fixed(),
    )
    # Construction succeeded; the public checker still rejects the input.
    with pytest.raises(LibraryException, match="non-increasing loss level"):
        surface.check_losses()


# -----------------------------------------------------------------------------
# BaseCorrelationLossModel
# -----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def bclm_setup() -> tuple[
    fx.CreditLossFixture, BaseCorrelationTermStructure, BaseCorrelationLossModel
]:
    f = fx.build_fixture()
    cal = TARGET()
    surface = BaseCorrelationTermStructure(
        0,
        cal,
        BusinessDayConvention.Following,
        TENORS,
        BCLM_LOSS_LEVELS,
        [[SimpleQuote(v) for v in row] for row in BCLM_QUOTES],
        Actual365Fixed(),
    )
    model = BaseCorrelationLossModel(surface, fx.RECOVERIES)
    f.basket.set_loss_model(model)
    return f, surface, model


def _bclm_dates(f: fx.CreditLossFixture) -> list[Date]:
    """probe.cpp:956-960 — 12/24/36/60 months off today."""
    return [
        f.calendar.advance_period(
            f.today, Period(m, TimeUnit.Months), BusinessDayConvention.Following
        )
        for m in (12, 24, 36, 60)
    ]


def test_bclm_typedef_is_the_class() -> None:
    """# C++ parity: basecorrelationlossmodel.hpp:286-290."""
    assert GaussianLHPFlatBCLM is BaseCorrelationLossModel


def test_bclm_setup_matches_cpp(
    bclm_setup: tuple[
        fx.CreditLossFixture, BaseCorrelationTermStructure, BaseCorrelationLossModel
    ],
    cpp_ref: dict[str, Any],
) -> None:
    f, surface, model = bclm_setup
    tolerance.tight(model.attach_ratio(), cpp_ref["bclm_attach_ratio"])
    tolerance.tight(model.detach_ratio(), cpp_ref["bclm_detach_ratio"])
    for t, expected in zip(
        surface.tranche_times(), cpp_ref["bclm_tranche_times"], strict=True
    ):
        tolerance.tight(t, expected)
    assert [d.serial_number() for d in _bclm_dates(f)] == cpp_ref["bclm_dates_serial"]


def test_bclm_interpolated_correlations_match_cpp(
    bclm_setup: tuple[
        fx.CreditLossFixture, BaseCorrelationTermStructure, BaseCorrelationLossModel
    ],
    cpp_ref: dict[str, Any],
) -> None:
    """The two correlations the model pulls off the surface at attach/detach."""
    f, surface, model = bclm_setup
    dates = _bclm_dates(f)
    for d, expected in zip(dates, cpp_ref["bclm_correl_at_attach"], strict=True):
        tolerance.tight(surface.correlation(d, model.attach_ratio()), expected)
    for d, expected in zip(dates, cpp_ref["bclm_correl_at_detach"], strict=True):
        tolerance.tight(surface.correlation(d, model.detach_ratio()), expected)


def test_bclm_expected_tranche_loss_matches_cpp(
    bclm_setup: tuple[
        fx.CreditLossFixture, BaseCorrelationTermStructure, BaseCorrelationLossModel
    ],
    cpp_ref: dict[str, Any],
) -> None:
    f, _surface, model = bclm_setup
    for d, expected in zip(
        _bclm_dates(f), cpp_ref["bclm_expected_tranche_loss"], strict=True
    ):
        tolerance.tight(model.expected_tranche_loss(d), expected)


def test_bclm_quote_row_major_matches_cpp(cpp_ref: dict[str, Any]) -> None:
    """Sanity: the fixture feeds C++'s quote matrix, not a transposed copy."""
    flat = [v for row in BCLM_QUOTES for v in row]
    for actual, expected in zip(
        flat, cpp_ref["bclm_quotes_row_major"], strict=True
    ):
        tolerance.exact(actual, expected)
    for actual, expected in zip(
        BCLM_LOSS_LEVELS, cpp_ref["bclm_loss_levels"], strict=True
    ):
        tolerance.exact(actual, expected)
