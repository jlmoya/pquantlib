"""BlackVolTimeExtrapolation, cross-validated against C++ v1.43.

Reference: migration-harness/references/v143/eqfx/timeextrap.json
Probe:     migration-harness/cpp/probes/v143_eqfx_timeextrap/probe.cpp

Four blocks:

* ``static_curve`` / ``static_surface`` — the two C++ overloads called
  directly over a piecewise-LINEAR variance functor. Here ``UseInterpolator``
  and ``LinearVariance`` necessarily coincide past the last node, so this
  block cannot tell them apart.
* ``static_curve_quadratic`` / ``static_surface_quadratic`` — same nodes, a
  QUADRATIC variance functor. This is the block that proves the two branches
  are not swapped.
* ``curve`` — the same three policies wired through ``BlackVarianceCurve``,
  end to end.

The ``LinearVariance`` policy refuses times that are not past the last node
(``linearExtrapolation`` requires ``t > times[-1]``); those appear in the
reference as ``{"raises": true}``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exceptions import LibraryException
from pquantlib.termstructures.volatility.equity_fx.black_variance_curve import (
    BlackVarianceCurve,
)
from pquantlib.termstructures.volatility.equity_fx.black_vol_time_extrapolation import (
    BlackVolTimeExtrapolation,
)
from pquantlib.testing import reference_reader, tolerance
from pquantlib.time.date import Date
from pquantlib.time.month import Month

_REF = reference_reader.load("v143/eqfx/timeextrap")

_DC = Actual365Fixed()
_REF_DATE = Date.from_ymd(15, Month.June, 2026)
_DATES = [
    Date.from_ymd(15, Month.September, 2026),
    Date.from_ymd(15, Month.December, 2026),
    Date.from_ymd(15, Month.June, 2027),
    Date.from_ymd(15, Month.June, 2028),
]
_VOLS = [0.13, 0.16, 0.20, 0.25]

_POLICIES: dict[str, BlackVolTimeExtrapolation.Type] = {
    "flat_volatility": BlackVolTimeExtrapolation.Type.FlatVolatility,
    "use_interpolator": BlackVolTimeExtrapolation.Type.UseInterpolator,
    "linear_variance": BlackVolTimeExtrapolation.Type.LinearVariance,
}

# Mirrors the probe's kQueryTimes label table.
_QUERY_TIMES: list[tuple[str, float]] = [
    ("t0p5_inside", 0.5),
    ("t1p8_inside", 1.8),
    ("t2p0_last_node", 2.0),
    ("t2p5_past", 2.5),
    ("t3p5_past", 3.5),
    ("t10p0_far_past", 10.0),
]


def _node_times() -> list[float]:
    """The node vector BlackVarianceCurve hands to the extrapolator."""
    return [0.0, *(_DC.year_fraction(_REF_DATE, d) for d in _DATES)]


def _node_variances(times: Sequence[float]) -> list[float]:
    return [0.0, *(times[j] * _VOLS[j - 1] ** 2 for j in range(1, len(times)))]


def _linear_variance(
    times: Sequence[float], variances: Sequence[float], t: float
) -> float:
    """Piecewise-linear variance, extended linearly outside the node range.

    Mirrors the probe's ``linearVariance`` helper line for line, so the
    ``static_*`` blocks test only BlackVolTimeExtrapolation and not whichever
    interpolator each language happens to use.
    """
    i = 1
    while i + 1 < len(times) and t > times[i]:
        i += 1
    t0, t1 = times[i - 1], times[i]
    return variances[i - 1] + (t - t0) * (variances[i] - variances[i - 1]) / (t1 - t0)


def _quadratic_variance(t: float) -> float:
    """The probe's ``quadCurveFn``: var(t) = 0.02 t^2 + 0.03 t."""
    return 0.02 * t * t + 0.03 * t


def _assert_matches(
    expected: Any, call: Callable[[], float], *, where: str
) -> None:
    """Assert ``call()`` reproduces ``expected``, incl. the raises sentinel."""
    if isinstance(expected, dict):
        assert expected == {"raises": True}, f"unexpected sentinel at {where}"
        with pytest.raises(LibraryException):
            call()
        return
    tolerance.tight(call(), float(expected), reason=where)


def test_node_grid_matches_reference() -> None:
    """The node vectors both sides build must agree before anything else."""
    times = _node_times()
    variances = _node_variances(times)
    for actual, expected in zip(times, _REF["nodes"]["times"], strict=True):
        tolerance.tight(actual, expected)
    for actual, expected in zip(variances, _REF["nodes"]["variances"], strict=True):
        tolerance.tight(actual, expected)


@pytest.mark.parametrize("policy_key", list(_POLICIES))
def test_static_curve_overload_linear_functor(policy_key: str) -> None:
    times = _node_times()
    variances = _node_variances(times)
    policy = _POLICIES[policy_key]
    expected = _REF["static_curve"][policy_key]
    for label, t in _QUERY_TIMES:
        _assert_matches(
            expected[label],
            lambda t=t: BlackVolTimeExtrapolation.extrapolated_variance_curve(
                policy, t, times, lambda tt: _linear_variance(times, variances, tt)
            ),
            where=f"static_curve/{policy_key}/{label}",
        )


@pytest.mark.parametrize("policy_key", list(_POLICIES))
def test_static_surface_overload_linear_functor(policy_key: str) -> None:
    times = _node_times()
    variances = _node_variances(times)
    policy = _POLICIES[policy_key]
    expected = _REF["static_surface"][policy_key]
    for label, t in _QUERY_TIMES:
        _assert_matches(
            expected[label],
            lambda t=t: BlackVolTimeExtrapolation.extrapolated_variance_surface(
                policy,
                t,
                123.0,
                times,
                lambda tt, _k: _linear_variance(times, variances, tt),
            ),
            where=f"static_surface/{policy_key}/{label}",
        )


@pytest.mark.parametrize("policy_key", list(_POLICIES))
def test_static_curve_overload_quadratic_functor(policy_key: str) -> None:
    """The block that separates UseInterpolator from LinearVariance."""
    times = _node_times()
    policy = _POLICIES[policy_key]
    expected = _REF["static_curve_quadratic"][policy_key]
    for label, t in _QUERY_TIMES:
        _assert_matches(
            expected[label],
            lambda t=t: BlackVolTimeExtrapolation.extrapolated_variance_curve(
                policy, t, times, _quadratic_variance
            ),
            where=f"static_curve_quadratic/{policy_key}/{label}",
        )


@pytest.mark.parametrize("policy_key", list(_POLICIES))
def test_static_surface_overload_quadratic_functor(policy_key: str) -> None:
    times = _node_times()
    policy = _POLICIES[policy_key]
    expected = _REF["static_surface_quadratic"][policy_key]
    for label, t in _QUERY_TIMES:
        _assert_matches(
            expected[label],
            lambda t=t: BlackVolTimeExtrapolation.extrapolated_variance_surface(
                policy, t, 123.0, times, lambda tt, _k: _quadratic_variance(tt)
            ),
            where=f"static_surface_quadratic/{policy_key}/{label}",
        )


def test_the_three_policies_actually_differ() -> None:
    """Guard the discriminating power of the reference itself.

    If a future change made two policies numerically identical past the last
    node the tests above would still pass while proving nothing, so assert
    the reference separates them by more than rounding.
    """
    far = _REF["static_curve_quadratic"]
    flat = far["flat_volatility"]["t10p0_far_past"]
    interp = far["use_interpolator"]["t10p0_far_past"]
    linear = far["linear_variance"]["t10p0_far_past"]
    assert abs(flat - interp) / interp > 0.1
    assert abs(linear - interp) / interp > 0.1
    assert abs(flat - linear) / linear > 0.1


@pytest.mark.parametrize("policy_key", list(_POLICIES))
def test_black_variance_curve_wiring(policy_key: str) -> None:
    policy = _POLICIES[policy_key]
    curve = BlackVarianceCurve(
        reference_date=_REF_DATE,
        dates=_DATES,
        black_vol_curve=_VOLS,
        day_counter=_DC,
        force_monotone_variance=True,
        time_extrapolation_type=policy,
    )
    expected = _REF["curve"][policy_key]
    assert curve.max_date().serial_number() == expected["max_date"]
    for label, t in _QUERY_TIMES:
        _assert_matches(
            expected["variance"][label],
            lambda t=t: curve.black_variance_at_time(t, 1.0, extrapolate=True),
            where=f"curve/{policy_key}/variance/{label}",
        )
        _assert_matches(
            expected["vol"][label],
            lambda t=t: curve.black_vol_at_time(t, 1.0, extrapolate=True),
            where=f"curve/{policy_key}/vol/{label}",
        )


def test_black_variance_curve_defaults_to_flat_volatility() -> None:
    """The C++ default for the new argument is FlatVolatility."""
    kwargs = {
        "reference_date": _REF_DATE,
        "dates": _DATES,
        "black_vol_curve": _VOLS,
        "day_counter": _DC,
        "force_monotone_variance": True,
    }
    default_curve = BlackVarianceCurve(**kwargs)  # pyright: ignore[reportArgumentType]
    explicit_curve = BlackVarianceCurve(
        **kwargs,  # pyright: ignore[reportArgumentType]
        time_extrapolation_type=BlackVolTimeExtrapolation.Type.FlatVolatility,
    )
    for _, t in _QUERY_TIMES:
        tolerance.exact(
            default_curve.black_variance_at_time(t, 1.0, extrapolate=True),
            explicit_curve.black_variance_at_time(t, 1.0, extrapolate=True),
        )
