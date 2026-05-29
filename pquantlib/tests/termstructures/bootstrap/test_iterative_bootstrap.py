"""Tests for IterativeBootstrap — generic piecewise bootstrapper.

# C++ parity: ql/termstructures/iterativebootstrap.hpp (v1.42.1).

Smoke tests using a minimal fake curve + traits to verify the loop
mechanics independently of the inflation-curve subclasses. End-to-end
roundtrip tests live alongside the piecewise curves.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pquantlib.math.interpolations.linear import LinearInterpolation
from pquantlib.termstructures.bootstrap.iterative_bootstrap import IterativeBootstrap
from pquantlib.termstructures.bootstrap_helper import BootstrapHelper
from pquantlib.time.date import Date


class _FakeCurve:
    """Minimal curve satisfying BootstrapCurveProtocol.

    Stores ``_dates / _times / _data / _interpolation`` arrays and
    exposes the bootstrap-internal mutators. The curve is "evaluated"
    by ``value(t)`` — a simple linear-interpolation lookup that the
    fake helpers consult to compute their implied quotes.
    """

    def __init__(self, today: Date) -> None:
        self._today = today
        self._dates: list[Date] = []
        self._times: list[float] = []
        self._data: list[float] = []
        self._interpolation: LinearInterpolation | None = None

    # -- BootstrapCurveProtocol ---------------------------------------

    def reference_date(self) -> Date:
        return self._today

    def base_date(self) -> Date:
        return self._dates[0] if self._dates else self._today

    def times(self) -> list[float]:
        return list(self._times)

    def data_live(self) -> list[float]:
        return self._data

    def set_data_at(self, i: int, level: float) -> None:
        self._data[i] = level

    def refresh_interpolation_through(self, up_to: int) -> None:
        partial_t = self._times[: up_to + 1]
        partial_d = self._data[: up_to + 1]
        self._interpolation = LinearInterpolation(
            np.asarray(partial_t, dtype=np.float64),
            np.asarray(partial_d, dtype=np.float64),
        )

    def bootstrap_install_grid(
        self, dates: list[Date], times: list[float], data: list[float]
    ) -> None:
        self._dates = list(dates)
        self._times = list(times)
        self._data = list(data)

    def time_from_reference(self, d: Date) -> float:
        # Days/365 — simple year fraction.
        return (d.serial_number() - self._today.serial_number()) / 365.0

    # -- evaluator -----------------------------------------------------

    def value(self, t: float) -> float:
        assert self._interpolation is not None
        return self._interpolation(t, allow_extrapolation=True)


class _FakeHelper(BootstrapHelper[_FakeCurve]):
    """A helper whose implied quote is ``curve.value(t_pillar)`` itself.

    Bootstrapping with these helpers solves for ``data[i] = quote[i]``
    — a trivial linear-pinning case.
    """

    def __init__(self, quote: float, pillar: Date) -> None:
        super().__init__(quote)
        self._pillar_date = pillar
        self._earliest_date = pillar
        self._latest_date = pillar
        self._curve: _FakeCurve | None = None

    def set_term_structure(self, ts: _FakeCurve) -> None:
        super().set_term_structure(ts)
        self._curve = ts

    def implied_quote(self) -> float:
        assert self._curve is not None
        t = self._curve.time_from_reference(self.pillar_date())
        return self._curve.value(t)


class _FakeTraits:
    """Minimal traits — initial_value 0, no propagation, ±0.5 bracket."""

    def initial_date(self, ts: _FakeCurve) -> Date:
        return ts.reference_date()

    def initial_value(self, ts: _FakeCurve) -> float:
        del ts
        return 0.0

    def guess(self, i: int, data: list[float], valid_data: bool) -> float:
        del i, valid_data
        return 0.05 if len(data) > 0 else 0.0

    def min_value_after(
        self, i: int, data: list[float], valid_data: bool
    ) -> float:
        del i, data, valid_data
        return -0.5

    def max_value_after(
        self, i: int, data: list[float], valid_data: bool
    ) -> float:
        del i, data, valid_data
        return 0.5

    def update_guess(self, data: list[float], level: float, i: int) -> None:
        data[i] = level

    def max_iterations(self) -> int:
        return 10


def test_iterative_bootstrap_pins_quotes_for_linear_curve() -> None:
    """3 helpers + linear interpolation → pillar data[i] == quote[i]."""
    today = Date(43000)  # arbitrary
    curve = _FakeCurve(today)
    helpers = [
        _FakeHelper(0.02, Date(43365)),  # 1Y
        _FakeHelper(0.025, Date(43730)),  # 2Y
        _FakeHelper(0.03, Date(44460)),  # 4Y
    ]
    traits = _FakeTraits()
    bootstrapper: IterativeBootstrap[_FakeCurve, _FakeTraits] = IterativeBootstrap(
        curve=curve,
        instruments=helpers,
        traits=traits,
    )
    bootstrapper.calculate()
    # Each helper's pillar data should equal its quote.
    assert math.isclose(curve.data_live()[1], 0.02, abs_tol=1e-10)
    assert math.isclose(curve.data_live()[2], 0.025, abs_tol=1e-10)
    assert math.isclose(curve.data_live()[3], 0.03, abs_tol=1e-10)


def test_iterative_bootstrap_rejects_duplicate_pillars() -> None:
    """C++ parity: ``two instruments have the same pillar date``."""
    today = Date(43000)
    curve = _FakeCurve(today)
    same_pillar = Date(43365)
    helpers = [
        _FakeHelper(0.02, same_pillar),
        _FakeHelper(0.025, same_pillar),
    ]
    traits = _FakeTraits()
    bootstrapper: IterativeBootstrap[_FakeCurve, _FakeTraits] = IterativeBootstrap(
        curve=curve,
        instruments=helpers,
        traits=traits,
    )
    with pytest.raises(Exception, match="same pillar date"):
        bootstrapper.calculate()


def test_iterative_bootstrap_requires_at_least_one_helper() -> None:
    today = Date(43000)
    curve = _FakeCurve(today)
    with pytest.raises(Exception, match="no helpers"):
        IterativeBootstrap(curve=curve, instruments=[], traits=_FakeTraits())


def test_iterative_bootstrap_sorts_helpers_by_pillar() -> None:
    """Helpers are sorted internally; out-of-order input still bootstraps."""
    today = Date(43000)
    curve = _FakeCurve(today)
    helpers = [
        _FakeHelper(0.03, Date(44460)),  # 4Y first
        _FakeHelper(0.02, Date(43365)),  # 1Y second
        _FakeHelper(0.025, Date(43730)),  # 2Y third
    ]
    traits = _FakeTraits()
    bootstrapper: IterativeBootstrap[_FakeCurve, _FakeTraits] = IterativeBootstrap(
        curve=curve,
        instruments=helpers,
        traits=traits,
    )
    bootstrapper.calculate()
    # After sort, dates are in [today, 1Y, 2Y, 4Y] order with quotes
    # [_, 0.02, 0.025, 0.03].
    assert math.isclose(curve.data_live()[1], 0.02, abs_tol=1e-10)
    assert math.isclose(curve.data_live()[2], 0.025, abs_tol=1e-10)
    assert math.isclose(curve.data_live()[3], 0.03, abs_tol=1e-10)
