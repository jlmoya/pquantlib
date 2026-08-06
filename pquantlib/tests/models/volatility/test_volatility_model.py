"""Tests for the root-level volatility-model abstract bases.

# C++ parity: ql/volatilitymodel.hpp @ v1.43.

Structural only — the two classes have no behaviour of their own. What is
worth pinning is that they stay ABSTRACT (C++ makes every method
pure-virtual, so a compositor with nothing to fit must still say so), that
the estimator is generic over the quote type, and that each concrete class
in ql/models/volatility/ implements the base C++ gives it.
"""

from __future__ import annotations

import pytest

from pquantlib.models.volatility.constant_estimator import ConstantEstimator
from pquantlib.models.volatility.garch import Garch11
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
from pquantlib.models.volatility.simple_local_estimator import SimpleLocalEstimator
from pquantlib.time.date import Date
from pquantlib.time.time_series import TimeSeries
from pquantlib.volatility_model import LocalVolatilityEstimator, VolatilityCompositor


def test_local_volatility_estimator_cannot_be_instantiated() -> None:
    """# C++ parity: ``calculate`` is pure-virtual (volatilitymodel.hpp:37)."""
    with pytest.raises(TypeError):
        LocalVolatilityEstimator()  # type: ignore[abstract]  # pyright: ignore[reportAbstractUsage]


def test_volatility_compositor_cannot_be_instantiated() -> None:
    """# C++ parity: both methods are pure-virtual (volatilitymodel.hpp:45-46)."""
    with pytest.raises(TypeError):
        VolatilityCompositor()  # type: ignore[abstract]  # pyright: ignore[reportAbstractUsage]


def test_compositor_subclass_must_override_both_methods() -> None:
    """Overriding only ``calculate`` leaves ``calibrate`` pure-virtual."""

    class HalfDone(VolatilityCompositor):
        def calculate(self, volatility_series: TimeSeries[float]) -> TimeSeries[float]:
            return volatility_series

    with pytest.raises(TypeError):
        HalfDone()  # type: ignore[abstract]  # pyright: ignore[reportAbstractUsage]


def test_local_volatility_estimator_is_generic_over_the_quote_type() -> None:
    """# C++ parity: ``template <class T> class LocalVolatilityEstimator``.

    The estimator's input element type is arbitrary and its output is always
    ``TimeSeries[float]`` — that asymmetry is the difference from
    ``VolatilityCompositor``, whose input and output are the same type.
    """

    class CountingEstimator(LocalVolatilityEstimator[str]):
        def calculate(self, quote_series: TimeSeries[str]) -> TimeSeries[float]:
            out: TimeSeries[float] = TimeSeries()
            for date, quote in quote_series.items():
                out[date] = float(len(quote))
            return out

    quotes: TimeSeries[str] = TimeSeries[str].from_first_date(Date(45000), ["ab", "cdef"])
    result = CountingEstimator().calculate(quotes)
    assert list(result.values()) == [2.0, 4.0]


@pytest.mark.parametrize(
    "cls",
    [
        GarmanKlassAbstract,
        GarmanKlassOpenClose,
        GarmanKlassSimpleSigma,
        GarmanKlassSigma1,
        GarmanKlassSigma3,
        GarmanKlassSigma4,
        GarmanKlassSigma5,
        GarmanKlassSigma6,
        ParkinsonSigma,
        SimpleLocalEstimator,
    ],
)
def test_estimators_derive_from_local_volatility_estimator(cls: type) -> None:
    """# C++ parity: garmanklass.hpp:42, simplelocalestimator.hpp:36."""
    assert issubclass(cls, LocalVolatilityEstimator)


@pytest.mark.parametrize("cls", [ConstantEstimator, Garch11])
def test_compositors_derive_from_volatility_compositor(cls: type) -> None:
    """# C++ parity: constantestimator.hpp:35, garch.hpp:38."""
    assert issubclass(cls, VolatilityCompositor)
