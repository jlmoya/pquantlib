"""Cross-validation of the Garman-Klass family against C++ v1.43.

# C++ parity: ql/models/volatility/garmanklass.hpp @ v1.43.
# Reference: migration-harness/cpp/probes/v143_models_volatility/probe.cpp,
# ``emitEstimators`` (the Garman-Klass block).

Every bar of the pinned OHLC table has four DISTINCT fields and no two bars
share a shape, because the estimators read different subsets of open / close /
high / low: a table where, say, ``high == close`` would let a formula that
dropped one field still reproduce it.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.models.volatility.garman_klass import (
    GarmanKlassAbstract,
    GarmanKlassOpenClose,
    GarmanKlassSigma1,
    GarmanKlassSigma3,
    GarmanKlassSigma4,
    GarmanKlassSigma5,
    GarmanKlassSigma6,
    GarmanKlassSimpleSigma,
    ParkinsonSigma,
)
from pquantlib.prices import IntervalPrice
from pquantlib.testing import tolerance
from pquantlib.time.date import Date
from pquantlib.time.time_series import TimeSeries

from .conftest import assert_series_matches

# The three concrete ``GarmanKlassOpenClose`` specialisations. Spelled as a
# union of the concrete types rather than ``type[GarmanKlassOpenClose]`` so
# their TWO-argument constructors are what a parametrised test sees; the base
# takes three (garmanklass.hpp:83 vs :110/:134/:173).
_OpenCloseClass = type[GarmanKlassSigma1] | type[GarmanKlassSigma3] | type[GarmanKlassSigma6]


def _degenerate_sigma4_point(c: float) -> float:
    """``GarmanKlassSigma4`` point with ``u == d == 0`` (garmanklass.hpp:152)."""
    return -0.383 * c * c


def _degenerate_sigma5_point(c: float) -> float:
    """``GarmanKlassSigma5`` point with ``u == d == 0`` (garmanklass.hpp:166)."""
    return -(2.0 * math.log(2.0) - 1.0) * c * c


# --- the per-bar estimators (GarmanKlassAbstract::calculate) ------------------


@pytest.mark.parametrize(
    ("cls", "key"),
    [
        (GarmanKlassSimpleSigma, "garman_klass_simple_sigma"),
        (ParkinsonSigma, "parkinson_sigma"),
        (GarmanKlassSigma4, "garman_klass_sigma4"),
        (GarmanKlassSigma5, "garman_klass_sigma5"),
    ],
)
def test_per_bar_estimators_match_cpp(
    cls: type[GarmanKlassAbstract],
    key: str,
    cpp: dict[str, Any],
    ohlc: TimeSeries[IntervalPrice],
    year_fraction: float,
) -> None:
    """FULL series. # C++ parity: garmanklass.hpp:49-59, :68-71, :123-127, :148-153, :162-167."""
    assert_series_matches(cls(year_fraction).calculate(ohlc), cpp[key], tolerance.tight)


@pytest.mark.parametrize(
    ("cls", "key"),
    [
        (GarmanKlassSimpleSigma, "degenerate_garman_klass_simple_sigma"),
        (ParkinsonSigma, "degenerate_parkinson_sigma"),
        (GarmanKlassSigma4, "degenerate_garman_klass_sigma4"),
        (GarmanKlassSigma5, "degenerate_garman_klass_sigma5"),
    ],
)
def test_per_bar_estimators_match_cpp_on_degenerate_bars(
    cls: type[GarmanKlassAbstract],
    key: str,
    cpp: dict[str, Any],
    ohlc_degenerate: TimeSeries[IntervalPrice],
    year_fraction: float,
) -> None:
    """``high == low == open != close`` — the only way to reach the ``fabs``.

    For an OHLC-consistent bar ``(u-d)^2 >= c^2``, which keeps Sigma4 and
    Sigma5 non-negative, so ``std::fabs`` (garmanklass.hpp:55) is dead code
    there. These bars make both formulas negative.
    """
    assert_series_matches(cls(year_fraction).calculate(ohlc_degenerate), cpp[key], tolerance.tight)


def test_sigma4_and_sigma5_are_negative_on_degenerate_bars(
    ohlc_degenerate: TimeSeries[IntervalPrice], year_fraction: float
) -> None:
    """Proves the previous test actually exercises the ``fabs`` branch.

    The point formulas are recomputed here from garmanklass.hpp rather than
    read off the estimator, so the claim "these bars make the proxy negative"
    is established independently of the code under test. Each published
    estimate must then equal ``sqrt(|point| / y)`` — with the absolute value,
    which is the only thing keeping the sqrt out of the negative reals.
    """
    for bar in ohlc_degenerate.values():
        open_ = bar.open()
        close = bar.close()
        assert open_ is not None
        assert close is not None
        c = math.log(close / open_)
        # high == low == open, so u == d == 0 and both formulas reduce to a
        # negative multiple of c^2.
        sigma4_point = -0.383 * c * c
        sigma5_point = -(2.0 * math.log(2.0) - 1.0) * c * c
        assert sigma4_point < 0.0
        assert sigma5_point < 0.0

    degenerate_points: list[tuple[GarmanKlassAbstract, Callable[[float], float]]] = [
        (GarmanKlassSigma4(year_fraction), _degenerate_sigma4_point),
        (GarmanKlassSigma5(year_fraction), _degenerate_sigma5_point),
    ]
    for estimator, point_of in degenerate_points:
        for bar, value in zip(
            ohlc_degenerate.values(), estimator.calculate(ohlc_degenerate).values(), strict=True
        ):
            open_ = bar.open()
            close = bar.close()
            assert open_ is not None
            assert close is not None
            assert value > 0.0
            tolerance.tight(
                value, math.sqrt(abs(point_of(math.log(close / open_))) / year_fraction)
            )


def test_per_bar_estimators_keep_every_date(
    ohlc: TimeSeries[IntervalPrice], year_fraction: float
) -> None:
    """# C++ parity: garmanklass.hpp:52-53 — the loop starts at ``begin()``."""
    result = GarmanKlassSigma4(year_fraction).calculate(ohlc)
    assert [d.serial for d in result.dates()] == [d.serial for d in ohlc.dates()]


# --- the open-close estimators (GarmanKlassOpenClose::calculate) --------------


@pytest.mark.parametrize(
    ("cls", "key"),
    [
        (GarmanKlassSigma1, "garman_klass_sigma1"),
        (GarmanKlassSigma3, "garman_klass_sigma3"),
        (GarmanKlassSigma6, "garman_klass_sigma6"),
    ],
)
def test_open_close_estimators_match_cpp(
    cls: _OpenCloseClass,
    key: str,
    cpp: dict[str, Any],
    ohlc: TimeSeries[IntervalPrice],
    year_fraction: float,
    market_open_fraction: float,
) -> None:
    """FULL series. # C++ parity: garmanklass.hpp:86-103 plus each subclass's weight."""
    estimator = cls(year_fraction, market_open_fraction)
    assert_series_matches(estimator.calculate(ohlc), cpp[key], tolerance.tight)


def test_open_close_estimators_drop_the_first_date(
    ohlc: TimeSeries[IntervalPrice], year_fraction: float, market_open_fraction: float
) -> None:
    """# C++ parity: garmanklass.hpp:89-90 — ``start = begin(); ++start``.

    This is the structural difference from ``GarmanKlassAbstract::calculate``,
    which keeps every date.
    """
    result = GarmanKlassSigma1(year_fraction, market_open_fraction).calculate(ohlc)
    assert result.size() == ohlc.size() - 1
    assert result.first_date() == ohlc.dates()[1]


def test_open_close_estimators_read_the_previous_bars_close(
    ohlc: TimeSeries[IntervalPrice], year_fraction: float, market_open_fraction: float
) -> None:
    """Perturbing bar ``i-1``'s close must move the estimate at bar ``i``.

    That overnight gap ``o1 - c0`` (garmanklass.hpp:93-94) is the whole
    reason these classes exist; the per-bar estimators never look back.
    """
    dates = list(ohlc.dates())
    bars = list(ohlc.values())
    target = dates[3]
    baseline = GarmanKlassSigma1(year_fraction, market_open_fraction).calculate(ohlc)[target]

    bumped: TimeSeries[IntervalPrice] = TimeSeries()
    for i, (date, bar) in enumerate(ohlc.items()):
        close = bar.close()
        assert close is not None
        bumped[date] = IntervalPrice(bar.open(), close + 0.5 if i == 2 else close, bar.high(), bar.low())
    moved = GarmanKlassSigma1(year_fraction, market_open_fraction).calculate(bumped)[target]

    assert baseline is not None
    assert moved is not None
    assert baseline != moved
    # Bar 3 itself is untouched, so a per-bar estimator would NOT move.
    unchanged = GarmanKlassSigma4(year_fraction).calculate(ohlc)[target]
    assert unchanged == GarmanKlassSigma4(year_fraction).calculate(bumped)[target]
    assert bars[3] == bumped.values()[3]


# --- the hard-coded weights and the class hierarchy ---------------------------


@pytest.mark.parametrize(
    ("cls", "weight"),
    [(GarmanKlassSigma1, 0.5), (GarmanKlassSigma3, 0.17), (GarmanKlassSigma6, 0.012)],
)
def test_open_close_weights_are_the_published_constants(
    cls: _OpenCloseClass, weight: float
) -> None:
    """# C++ parity: garmanklass.hpp:113, :137, :176."""
    estimator = cls(1.0 / 252.0, 0.3)
    assert estimator.a() == weight
    assert estimator.market_open_fraction() == 0.3
    assert estimator.year_fraction() == 1.0 / 252.0


@pytest.mark.parametrize(
    ("derived", "base"),
    [
        (GarmanKlassSigma1, GarmanKlassSimpleSigma),
        (GarmanKlassSigma3, ParkinsonSigma),
        (GarmanKlassSigma6, GarmanKlassSigma4),
    ],
)
def test_open_close_subclasses_keep_the_cpp_is_a_relationship(
    derived: type, base: type
) -> None:
    """# C++ parity: ``GarmanKlassOpenClose<T> : public T`` (garmanklass.hpp:78).

    In C++ ``GarmanKlassSigma1`` IS-A ``GarmanKlassSimpleSigma`` because the
    template derives from its parameter. The Python port reproduces that by
    listing the intraday estimator as the second base; the MRO order is what
    decides that ``calculate`` comes from the open-close mixin while the point
    formula comes from the intraday estimator, so the order is pinned here.
    """
    assert issubclass(derived, base)
    assert issubclass(derived, GarmanKlassOpenClose)
    assert issubclass(derived, GarmanKlassAbstract)
    mro = derived.__mro__
    assert mro.index(GarmanKlassOpenClose) < mro.index(base)
    assert derived.calculate is GarmanKlassOpenClose.calculate


@pytest.mark.parametrize("cls", [GarmanKlassAbstract, GarmanKlassOpenClose])
def test_abstract_bases_cannot_be_instantiated(cls: type) -> None:
    """``GarmanKlassOpenClose`` has no ``_calculate_point``, exactly like an
    unspecialised C++ template."""
    with pytest.raises(TypeError):
        cls(1.0 / 252.0)


# --- missing OHLC components --------------------------------------------------


@pytest.mark.parametrize(
    ("missing", "cls"),
    [
        ("open", GarmanKlassSimpleSigma),
        ("close", GarmanKlassSimpleSigma),
        ("high", ParkinsonSigma),
        ("low", ParkinsonSigma),
    ],
)
def test_missing_component_is_diagnosed(missing: str, cls: type[GarmanKlassAbstract]) -> None:
    """# C++ parity divergence: C++ would feed ``Null<Real>()`` into ``log``.

    pquantlib's ``IntervalPrice`` spells that sentinel ``None``, so the
    missing field is reported instead of quietly producing nonsense.
    """
    fields: dict[str, float | None] = {
        "open": 100.0,
        "close": 101.0,
        "high": 102.0,
        "low": 99.0,
    }
    fields[missing] = None
    series: TimeSeries[IntervalPrice] = TimeSeries()
    series[Date(45000)] = IntervalPrice(
        fields["open"], fields["close"], fields["high"], fields["low"]
    )
    with pytest.raises(LibraryException, match=f"IntervalPrice.{missing}"):
        cls(1.0 / 252.0).calculate(series)


# --- independent recomputation of each point formula --------------------------


def test_point_formulas_match_the_published_expressions(
    ohlc: TimeSeries[IntervalPrice], year_fraction: float
) -> None:
    """Recompute every point formula from garmanklass.hpp, through the public API.

    Written from the header rather than from a textbook, because the C++
    weights (0.511 / 0.019 / 0.383 and ``2 ln 2 - 1``) are what the port must
    match. Each estimator's published series must equal
    ``sqrt(|point| / year_fraction)`` bar by bar.
    """
    formulas: list[tuple[GarmanKlassAbstract, list[float]]] = []
    points: dict[str, list[float]] = {"simple": [], "parkinson": [], "sigma4": [], "sigma5": []}
    for bar in ohlc.values():
        open_ = bar.open()
        close = bar.close()
        high = bar.high()
        low = bar.low()
        assert open_ is not None
        assert close is not None
        assert high is not None
        assert low is not None
        u = math.log(high / open_)
        d = math.log(low / open_)
        c = math.log(close / open_)
        points["simple"].append(c * c)
        points["parkinson"].append((u - d) * (u - d) / 4.0 / math.log(2.0))
        points["sigma4"].append(
            0.511 * (u - d) * (u - d) - 0.019 * (c * (u + d) - 2 * u * d) - 0.383 * c * c
        )
        points["sigma5"].append(0.5 * (u - d) * (u - d) - (2.0 * math.log(2.0) - 1.0) * c * c)

    formulas.append((GarmanKlassSimpleSigma(year_fraction), points["simple"]))
    formulas.append((ParkinsonSigma(year_fraction), points["parkinson"]))
    formulas.append((GarmanKlassSigma4(year_fraction), points["sigma4"]))
    formulas.append((GarmanKlassSigma5(year_fraction), points["sigma5"]))

    for estimator, expected_points in formulas:
        got = list(estimator.calculate(ohlc).values())
        for value, point in zip(got, expected_points, strict=True):
            tolerance.tight(value, math.sqrt(abs(point) / year_fraction))
