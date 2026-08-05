"""Cross-validate AdaptiveRungeKutta against the v1.43 C++ probe.

Reference: ``migration-harness/references/v143/math/tail.json`` —
``adaptive_runge_kutta``. Three problems, chosen so agreement cannot come
from the integrator merely being accurate:

* ``y' = y`` forward and backward — every RK45 gets this;
* ``y' = -50(y - cos x)`` at two accuracy targets — moderately stiff, so the
  step controller has to shrink and the answer depends on *its* rules;
* the harmonic oscillator as a 2-vector — exercises the vector entry point,
  which the scalar one delegates to through ``OdeFctWrapper``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.math.ode.adaptive_runge_kutta import AdaptiveRungeKutta, OdeFctWrapper
from pquantlib.testing import tolerance


def test_exponential_matches_cpp_tight(v143_tail: dict[str, Any]) -> None:
    for x2, expected in v143_tail["adaptive_runge_kutta"]["exponential"]:
        rk = AdaptiveRungeKutta(1e-8, 1e-4, 1e-12)
        tolerance.tight(rk.solve_1d(lambda _x, y: y, 1.0, 0.0, x2), expected, reason=f"x2={x2}")


def test_stiff_problem_matches_cpp_tight(v143_tail: dict[str, Any]) -> None:
    """The discriminating case: a fixed-step RK45 agrees on the exponential
    above and parts company here, because the answer depends on where the
    controller chose to shrink.
    """
    for case in v143_tail["adaptive_runge_kutta"]["stiff"]:
        rk = AdaptiveRungeKutta(case["eps"], 1e-4, 1e-14)
        tolerance.tight(
            rk.solve_1d(lambda x, y: -50.0 * (y - math.cos(x)), 0.0, 0.0, case["x2"]),
            case["v"],
            reason=str(case),
        )


def test_harmonic_system_matches_cpp_tight(v143_tail: dict[str, Any]) -> None:
    for case in v143_tail["adaptive_runge_kutta"]["harmonic"]:
        rk = AdaptiveRungeKutta(1e-10, 1e-4, 1e-14)
        got = rk.solve(lambda _x, v: [v[1], -v[0]], [1.0, 0.0], 0.0, case["x2"])
        for actual, expected in zip(got, case["y"], strict=True):
            tolerance.tight(actual, expected, reason=f"x2={case['x2']}")


def test_exponential_recovers_the_analytic_solution(v143_tail: dict[str, Any]) -> None:
    """Independent check that the pinned numbers are the right numbers.

    Custom tier: the integrator's own accuracy target here is eps = 1e-8, so
    that — not a tolerance tier — is what bounds the distance to exp(x).
    """
    for x2, expected in v143_tail["adaptive_runge_kutta"]["exponential"]:
        tolerance.custom(
            expected,
            math.exp(x2),
            abs_tol=1e-7,
            rel_tol=0.0,
            reason="integrator eps = 1e-8 per step, accumulated over the interval",
        )


def test_ode_fct_wrapper_lifts_a_scalar_ode(v143_tail: dict[str, Any]) -> None:
    """The scalar entry point runs the vector controller over a length-1 state.

    Asserted by driving the vector solver through the wrapper by hand and
    matching the scalar result bit for bit.
    """
    for x2, _ in v143_tail["adaptive_runge_kutta"]["exponential"]:
        scalar = AdaptiveRungeKutta(1e-8, 1e-4, 1e-12).solve_1d(lambda _x, y: y, 1.0, 0.0, x2)
        wrapped = AdaptiveRungeKutta(1e-8, 1e-4, 1e-12).solve(OdeFctWrapper(lambda _x, y: y), [1.0], 0.0, x2)
        tolerance.exact(wrapped[0], scalar)


def test_step_size_floor_raises() -> None:
    """hmin is a hard floor: the integrator fails rather than degrading."""
    rk = AdaptiveRungeKutta(1e-14, 1e-4, 1.0)

    def stiff(x: float, y: Sequence[float]) -> list[float]:
        return [-5000.0 * (y[0] - math.cos(x))]

    with pytest.raises(LibraryException, match="too small"):
        rk.solve(stiff, [0.0], 0.0, 5.0)
