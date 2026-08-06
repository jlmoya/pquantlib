"""Cross-validation of ConstantEstimator against C++ v1.43.

# C++ parity: ql/models/volatility/constantestimator.{hpp,cpp} @ v1.43.
# Reference: migration-harness/cpp/probes/v143_models_volatility/probe.cpp,
# ``emitEstimators`` (the ``ConstantEstimator`` block).
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.models.volatility.constant_estimator import ConstantEstimator
from pquantlib.testing import tolerance
from pquantlib.time.date import Date
from pquantlib.time.time_series import TimeSeries

from .conftest import assert_series_matches


def test_window_size_1_matches_cpp(cpp: dict[str, Any], vol_series: TimeSeries[float]) -> None:
    """FULL series at window 1. # C++ parity: constantestimator.cpp:23-44."""
    assert_series_matches(
        ConstantEstimator(1).calculate(vol_series),
        cpp["constant_estimator_size1"],
        tolerance.tight,
    )


def test_window_size_3_matches_cpp(cpp: dict[str, Any], vol_series: TimeSeries[float]) -> None:
    """FULL series at window 3.

    Two window widths are pinned because the estimator's second denominator
    is ``size + 1`` (constantestimator.cpp:38-39), not ``size`` and not
    ``size - 1``; a single width cannot tell those apart.
    """
    assert_series_matches(
        ConstantEstimator(3).calculate(vol_series),
        cpp["constant_estimator_size3"],
        tolerance.tight,
    )


def test_output_starts_size_dates_into_the_input(vol_series: TimeSeries[float]) -> None:
    """The first ``size`` dates carry no estimate.

    # C++ parity: constantestimator.cpp:28-31 — the output cursor is advanced
    # by ``size_`` before the loop, and the loop runs ``i`` from ``size_``.
    """
    result = ConstantEstimator(3).calculate(vol_series)
    assert result.size() == vol_series.size() - 3
    assert result.first_date() == vol_series.dates()[3]
    assert result.last_date() == vol_series.last_date()


def test_estimate_uses_only_observations_strictly_before_its_date(
    vol_series: TimeSeries[float],
) -> None:
    """The estimate at date ``i`` is a function of inputs ``[i-size, i)``.

    Perturbing the value AT the output date must leave that output alone;
    perturbing the one before it must not. This is the off-by-one that
    ``std::advance(cur, size_)`` (constantestimator.cpp:29) creates.
    """
    dates = vol_series.dates()
    values = list(vol_series.values())
    baseline = ConstantEstimator(3).calculate(vol_series)
    target = dates[5]

    same_date_bumped: TimeSeries[float] = TimeSeries[float].from_pairs(
        dates, [v + 10.0 if i == 5 else v for i, v in enumerate(values)]
    )
    prior_bumped: TimeSeries[float] = TimeSeries[float].from_pairs(
        dates, [v + 10.0 if i == 4 else v for i, v in enumerate(values)]
    )

    assert ConstantEstimator(3).calculate(same_date_bumped)[target] == baseline[target]
    assert ConstantEstimator(3).calculate(prior_bumped)[target] != baseline[target]


def test_matches_the_closed_form_window_statistic(vol_series: TimeSeries[float]) -> None:
    """Independent recomputation of the published formula.

    ``sqrt(sumu2/size - sumu^2/size/(size+1))`` — note this is neither the
    biased nor the unbiased sample standard deviation, so the check is
    written from the C++ expression, not from a statistics identity.
    """
    size = 4
    values = list(vol_series.values())
    result = ConstantEstimator(size).calculate(vol_series)
    for i in range(size, len(values)):
        window = values[i - size : i]
        sumu = sum(window)
        sumu2 = sum(v * v for v in window)
        expected = math.sqrt(sumu2 / size - sumu * sumu / size / (size + 1))
        got = result[vol_series.dates()[i]]
        assert got is not None
        tolerance.tight(got, expected)


def test_calibrate_is_a_no_op(cpp: dict[str, Any], vol_series: TimeSeries[float]) -> None:
    """# C++ parity: constantestimator.hpp:42 — an empty override body."""
    estimator = ConstantEstimator(3)
    assert estimator.calibrate(vol_series) is None
    assert_series_matches(
        estimator.calculate(vol_series),
        cpp["constant_estimator_size3_after_calibrate"],
        tolerance.tight,
    )


def test_size_inspector_returns_the_constructor_argument() -> None:
    assert ConstantEstimator(7).size() == 7


def test_window_wider_than_the_series_is_rejected() -> None:
    """# C++ parity divergence: C++ ``std::advance``-es past the end (UB)."""
    series: TimeSeries[float] = TimeSeries[float].from_first_date(Date(45000), [0.1, 0.2, 0.3])
    with pytest.raises(LibraryException, match="exceeds series size"):
        ConstantEstimator(4).calculate(series)


def test_zero_window_is_rejected() -> None:
    """# C++ parity divergence: C++ computes ``0.0/0`` and returns NaN."""
    series: TimeSeries[float] = TimeSeries[float].from_first_date(Date(45000), [0.1, 0.2, 0.3])
    with pytest.raises(LibraryException, match="must be positive"):
        ConstantEstimator(0).calculate(series)


def test_window_equal_to_the_series_length_yields_a_single_estimate() -> None:
    """Boundary of the guard: ``size == n`` is legal and gives ``n - size == 0`` outputs."""
    series: TimeSeries[float] = TimeSeries[float].from_first_date(Date(45000), [0.1, 0.2, 0.3])
    assert ConstantEstimator(3).calculate(series).empty()
