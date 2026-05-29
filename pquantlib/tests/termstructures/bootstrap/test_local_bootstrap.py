"""Tests for LocalBootstrap — windowed Levenberg-Marquardt bootstrapper.

# C++ parity: ql/termstructures/localbootstrap.hpp (v1.42.1).

Reuses the smoke-test fakes pattern from
``test_iterative_bootstrap.py`` so the test exercises the loop
mechanics in isolation from the actual yield-curve subclasses.
LocalBootstrap converges to the same per-pillar fixed points as
IterativeBootstrap when the helpers reduce to a simple pin (no
nonlinear cross-pillar coupling).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pquantlib.math.interpolations.linear import LinearInterpolation
from pquantlib.termstructures.bootstrap.bootstrap_error import BootstrapError
from pquantlib.termstructures.bootstrap.local_bootstrap import LocalBootstrap
from pquantlib.termstructures.bootstrap_helper import BootstrapHelper
from pquantlib.time.date import Date


class _FakeCurve:
    """Minimal curve satisfying BootstrapCurveProtocol — same as the
    IterativeBootstrap fake.
    """

    def __init__(self, today: Date) -> None:
        self._today = today
        self._dates: list[Date] = []
        self._times: list[float] = []
        self._data: list[float] = []
        self._interpolation: LinearInterpolation | None = None

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
        self, dates: list[Date], times: list[float], data: list[float],
    ) -> None:
        self._dates = list(dates)
        self._times = list(times)
        self._data = list(data)

    def time_from_reference(self, d: Date) -> float:
        return (d.serial_number() - self._today.serial_number()) / 365.0

    def value(self, t: float) -> float:
        assert self._interpolation is not None
        return self._interpolation(t, allow_extrapolation=True)


class _FakeHelper(BootstrapHelper[_FakeCurve]):
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
    def initial_date(self, ts: _FakeCurve) -> Date:
        return ts.reference_date()

    def initial_value(self, ts: _FakeCurve) -> float:
        del ts
        return 0.0

    def guess(self, i: int, data: list[float], valid_data: bool) -> float:
        del i, valid_data
        return 0.05 if len(data) > 0 else 0.0

    def min_value_after(
        self, i: int, data: list[float], valid_data: bool,
    ) -> float:
        del i, data, valid_data
        return 0.0

    def max_value_after(
        self, i: int, data: list[float], valid_data: bool,
    ) -> float:
        del i, data, valid_data
        return 0.5

    def update_guess(self, data: list[float], level: float, i: int) -> None:
        data[i] = level

    def max_iterations(self) -> int:
        return 10


# --- happy path ---------------------------------------------------------------


def test_local_bootstrap_pins_quotes_for_linear_curve() -> None:
    """LocalBootstrap on a trivially-pinning helper set matches IterativeBootstrap.

    With 3 helpers and ``localisation=2``, each window solves a
    2-parameter system; the linearity of the fake helpers means the
    system has a unique zero-error solution at each pillar = its quote.
    """
    today = Date(43000)
    curve = _FakeCurve(today)
    helpers = [
        _FakeHelper(0.02, Date(43365)),  # 1Y
        _FakeHelper(0.025, Date(43730)),  # 2Y
        _FakeHelper(0.03, Date(44460)),  # 4Y
    ]
    traits = _FakeTraits()
    bs: LocalBootstrap[_FakeCurve, _FakeTraits] = LocalBootstrap(
        curve=curve, instruments=helpers, traits=traits,
        localisation=2, force_positive=True,
    )
    bs.calculate()
    # Each helper's pillar data should equal its quote to LOOSE
    # (scipy.optimize.least_squares + force_positive nudges vs Brent
    # root-find — LOOSE-tier divergence vs IterativeBootstrap).
    assert math.isclose(curve.data_live()[1], 0.02, abs_tol=1e-6)
    assert math.isclose(curve.data_live()[2], 0.025, abs_tol=1e-6)
    assert math.isclose(curve.data_live()[3], 0.03, abs_tol=1e-6)


def test_local_bootstrap_matches_iterative_bootstrap_at_pillars() -> None:
    """LocalBootstrap and IterativeBootstrap converge to the same pillar values.

    # C++ parity: LocalBootstrap and IterativeBootstrap solve the *same*
    # n-pillar system to a different precision profile (windowed vs
    # global). On a trivially-pinning helper set the two algorithms
    # converge to the same zero-residual solution.
    """
    from pquantlib.termstructures.bootstrap.iterative_bootstrap import (  # noqa: PLC0415
        IterativeBootstrap,
    )

    today = Date(43000)
    pillars = [Date(43365), Date(43730), Date(44460), Date(45000)]
    quotes = [0.02, 0.025, 0.03, 0.035]

    # IterativeBootstrap run.
    curve_iter = _FakeCurve(today)
    helpers_iter = [_FakeHelper(q, p) for q, p in zip(quotes, pillars, strict=True)]
    IterativeBootstrap(
        curve=curve_iter,
        instruments=helpers_iter,
        traits=_FakeTraits(),
    ).calculate()

    # LocalBootstrap run.
    curve_local = _FakeCurve(today)
    helpers_local = [_FakeHelper(q, p) for q, p in zip(quotes, pillars, strict=True)]
    LocalBootstrap(
        curve=curve_local,
        instruments=helpers_local,
        traits=_FakeTraits(),
        localisation=2,
        force_positive=True,
    ).calculate()

    for i in range(1, len(pillars) + 1):
        assert math.isclose(
            curve_local.data_live()[i],
            curve_iter.data_live()[i],
            abs_tol=1e-6,
        ), (
            f"pillar {i}: local={curve_local.data_live()[i]} "
            f"vs iter={curve_iter.data_live()[i]}"
        )


# --- validation ---------------------------------------------------------------


def test_local_bootstrap_rejects_too_few_helpers_for_window() -> None:
    """# C++ parity: ``not enough instruments`` (line 122-126)."""
    today = Date(43000)
    curve = _FakeCurve(today)
    helpers = [_FakeHelper(0.02, Date(43365))]
    with pytest.raises(Exception, match="at least 2 helpers"):
        LocalBootstrap(
            curve=curve, instruments=helpers, traits=_FakeTraits(),
            localisation=2,
        )


def test_local_bootstrap_rejects_localisation_at_or_above_n() -> None:
    """``localisation`` must be strictly less than helper count."""
    today = Date(43000)
    curve = _FakeCurve(today)
    helpers = [
        _FakeHelper(0.02, Date(43365)),
        _FakeHelper(0.025, Date(43730)),
    ]
    with pytest.raises(Exception, match="more than localisation"):
        LocalBootstrap(
            curve=curve, instruments=helpers, traits=_FakeTraits(),
            localisation=2,  # 2 helpers, localisation=2 → fails
        )


def test_local_bootstrap_rejects_duplicate_pillars() -> None:
    """# C++ parity: ``two instruments have the same pillar date``."""
    today = Date(43000)
    curve = _FakeCurve(today)
    same_pillar = Date(43365)
    helpers = [
        _FakeHelper(0.02, same_pillar),
        _FakeHelper(0.025, same_pillar),
        _FakeHelper(0.03, Date(44460)),
    ]
    bs: LocalBootstrap[_FakeCurve, _FakeTraits] = LocalBootstrap(
        curve=curve, instruments=helpers, traits=_FakeTraits(),
        localisation=2,
    )
    with pytest.raises(Exception, match="same pillar date"):
        bs.calculate()


def test_local_bootstrap_raises_bootstrap_error_on_failure() -> None:
    """When the local solver cannot converge, raises a BootstrapError.

    We simulate a non-bracketing helper by making one helper always
    report a constant large residual irrespective of the curve state.
    """

    class _BadHelper(BootstrapHelper[_FakeCurve]):
        def __init__(self, pillar: Date) -> None:
            super().__init__(0.025)
            self._pillar_date = pillar
            self._earliest_date = pillar
            self._latest_date = pillar
            self._curve: _FakeCurve | None = None

        def set_term_structure(self, ts: _FakeCurve) -> None:
            super().set_term_structure(ts)
            self._curve = ts

        def implied_quote(self) -> float:
            # Always returns 1.0 → constant residual 0.975 regardless of x.
            return 1.0

    today = Date(43000)
    curve = _FakeCurve(today)
    helpers = [
        _FakeHelper(0.02, Date(43365)),
        _BadHelper(Date(43730)),
        _FakeHelper(0.03, Date(44460)),
    ]
    bs: LocalBootstrap[_FakeCurve, _FakeTraits] = LocalBootstrap(
        curve=curve, instruments=helpers, traits=_FakeTraits(),
        localisation=2,
        accuracy=1.0e-12,
    )
    with pytest.raises(BootstrapError) as exc_info:
        bs.calculate()
    # The error should pinpoint the failing window.
    assert exc_info.value.helper_index >= 1
    assert exc_info.value.last_residual is not None
