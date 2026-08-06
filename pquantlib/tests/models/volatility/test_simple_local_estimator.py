"""Cross-validation of SimpleLocalEstimator against C++ v1.43.

# C++ parity: ql/models/volatility/simplelocalestimator.hpp @ v1.43.
# Reference: migration-harness/cpp/probes/v143_models_volatility/probe.cpp,
# ``emitEstimators`` (the ``SimpleLocalEstimator`` block).
"""

from __future__ import annotations

import math
from typing import Any

from pquantlib.models.volatility.simple_local_estimator import SimpleLocalEstimator
from pquantlib.prices import IntervalPrice, IntervalPriceType
from pquantlib.testing import tolerance
from pquantlib.time.date import Date
from pquantlib.time.time_series import TimeSeries

from .conftest import assert_series_matches


def test_matches_cpp(cpp: dict[str, Any], closes: TimeSeries[float], year_fraction: float) -> None:
    """FULL series. # C++ parity: simplelocalestimator.hpp:42-54."""
    assert_series_matches(
        SimpleLocalEstimator(year_fraction).calculate(closes),
        cpp["simple_local_estimator"],
        tolerance.tight,
    )


def test_matches_cpp_at_a_second_year_fraction(
    cpp: dict[str, Any], closes: TimeSeries[float]
) -> None:
    """A second ``y`` pins that the scaling is ``1/sqrt(y)``, not ``1/y``.

    ``y = 4`` and ``y = 1/252`` differ by more than a factor of 1000, so a
    port that divided by ``y`` instead of ``sqrt(y)`` cannot reproduce both.
    """
    assert_series_matches(
        SimpleLocalEstimator(4.0).calculate(closes),
        cpp["simple_local_estimator_y4"],
        tolerance.tight,
    )


def test_year_fraction_inspector() -> None:
    assert SimpleLocalEstimator(0.25).year_fraction() == 0.25


def test_output_drops_the_first_date(closes: TimeSeries[float]) -> None:
    """# C++ parity: simplelocalestimator.hpp:45-46 — ``start = begin(); ++start``."""
    result = SimpleLocalEstimator(1.0).calculate(closes)
    assert result.size() == closes.size() - 1
    assert result.first_date() == closes.dates()[1]
    assert result.last_date() == closes.last_date()


def test_estimate_is_the_absolute_log_return(closes: TimeSeries[float]) -> None:
    """Independent recomputation, including the ABSOLUTE value.

    A down move and an up move of the same magnitude must give the same
    estimate — ``std::fabs`` (simplelocalestimator.hpp:50) is what makes the
    output a volatility rather than a signed return.
    """
    y = 1.0 / 252.0
    result = SimpleLocalEstimator(y).calculate(closes)
    items = closes.items()
    for i in range(1, len(items)):
        date, cur = items[i]
        prev = items[i - 1][1]
        got = result[date]
        assert got is not None
        assert got >= 0.0
        tolerance.tight(got, abs(math.log(cur / prev)) / math.sqrt(y))


def test_symmetric_moves_give_the_same_estimate() -> None:
    """A doubling and a halving must estimate identically.

    The ratios 2.0 and 0.5 are both exact binary doubles whose logs are exact
    negatives, so ``fabs`` (simplelocalestimator.hpp:50) makes the two results
    bit-identical; a port that dropped it would return a negative number for
    the halving.
    """
    up: TimeSeries[float] = TimeSeries[float].from_first_date(Date(45000), [100.0, 200.0])
    down: TimeSeries[float] = TimeSeries[float].from_first_date(Date(45000), [200.0, 100.0])
    estimator = SimpleLocalEstimator(1.0 / 252.0)
    up_value = estimator.calculate(up)[Date(45001)]
    down_value = estimator.calculate(down)[Date(45001)]
    assert up_value is not None
    assert down_value is not None
    assert up_value > 0.0
    tolerance.exact(up_value, down_value)


def test_empty_and_single_point_series_give_no_output() -> None:
    """C++ increments past ``begin()`` unconditionally; Python just yields nothing."""
    estimator = SimpleLocalEstimator(1.0)
    assert estimator.calculate(TimeSeries[float]()).empty()
    assert estimator.calculate(TimeSeries[float].from_first_date(Date(45000), [1.0])).empty()


def test_accepts_the_close_component_extracted_from_an_ohlc_series(
    cpp: dict[str, Any], ohlc: TimeSeries[IntervalPrice], closes: TimeSeries[float]
) -> None:
    """``IntervalPrice.extract_component`` feeds this estimator directly.

    That is how the probe builds its input (probe.cpp, ``emitEstimators``),
    so the two routes must agree exactly.
    """
    extracted = IntervalPrice.extract_component(ohlc, IntervalPriceType.Close)
    assert list(extracted.values()) == list(closes.values())
    assert_series_matches(
        SimpleLocalEstimator(float(cpp["year_fraction"])).calculate(closes),
        cpp["simple_local_estimator"],
        tolerance.tight,
    )
