"""ZabrSmileSection's four evaluation modes, cross-validated against C++ v1.43.

Reference: migration-harness/references/v143/zabr/fdsection.json
Probe:     migration-harness/cpp/probes/v143_zabr_fdsection/probe.cpp

The probe instantiates ``ZabrSmileSection<Evaluation>`` for each of the
four C++ tag types and reads ``volatility``, ``optionPrice(Call)`` and
``optionPrice(Put)`` at strikes inside the grid, at the ATM point, and
past the last grid strike where the exponential right tail takes over
from the spline.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from pquantlib.math.interpolations.zabr_formula import ZabrEvaluation
from pquantlib.termstructures.volatility.zabr_smile_section import ZabrSmileSection
from pquantlib.testing import reference_reader, tolerance

_TAU = 5.0
_FORWARD = 0.03
_PARAMS = (0.08, 0.70, 0.20, -0.30, 1.0)
_SMALL_MONEYNESS = [0.5, 0.75, 1.0, 1.5, 2.0]

# key -> (evaluation, moneyness, fd_refinement)
_CASES: dict[str, tuple[ZabrEvaluation, list[float] | None, int]] = {
    "short_maturity_lognormal": (ZabrEvaluation.ShortMaturityLognormal, None, 5),
    "short_maturity_normal": (ZabrEvaluation.ShortMaturityNormal, None, 5),
    "local_volatility": (ZabrEvaluation.LocalVolatility, None, 5),
    "local_volatility_small_grid": (ZabrEvaluation.LocalVolatility, _SMALL_MONEYNESS, 1),
    "full_fd_small_grid": (ZabrEvaluation.FullFd, _SMALL_MONEYNESS, 1),
}

_FAST_CASES = [k for k in _CASES if k != "full_fd_small_grid"]


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("v143/zabr/fdsection")


def _section(key: str) -> ZabrSmileSection:
    evaluation, moneyness, refinement = _CASES[key]
    return ZabrSmileSection(
        forward=_FORWARD,
        zabr_params=_PARAMS,
        exercise_time=_TAU,
        evaluation=evaluation,
        moneyness=moneyness,
        fd_refinement=refinement,
    )


@pytest.mark.parametrize("key", _FAST_CASES)
def test_section_metadata_matches_cpp(cpp: dict[str, Any], key: str) -> None:
    block = cpp[key]
    s = _section(key)
    tolerance.tight(s.exercise_time(), float(block["exercise_time"]))
    tolerance.tight(s.atm_level(), float(block["atm_level"]))
    tolerance.tight(s.min_strike(), float(block["min_strike"]))
    # C++ maxStrike is QL_MAX_REAL; pquantlib sections use +inf (see
    # SabrSmileSection), which behaves identically in every comparison
    # SmileSectionUtils and the pricing helpers make.
    assert s.max_strike() == math.inf


@pytest.mark.parametrize("key", _FAST_CASES)
def test_option_prices_match_cpp(cpp: dict[str, Any], key: str) -> None:
    """Call and put prices, including past the last grid strike."""
    block = cpp[key]
    s = _section(key)
    for i, strike in enumerate(block["strikes"]):
        tolerance.tight(s.option_price(strike, 1, 1.0), float(block["call_price"][i]))
        tolerance.tight(s.option_price(strike, -1, 1.0), float(block["put_price"][i]))


def test_short_maturity_lognormal_volatility_matches_cpp(cpp: dict[str, Any]) -> None:
    """The closed-form arm is exact; nothing iterative sits in this path."""
    block = cpp["short_maturity_lognormal"]
    s = _section("short_maturity_lognormal")
    for i, strike in enumerate(block["strikes"]):
        tolerance.tight(s.volatility(strike), float(block["volatility"][i]))


@pytest.mark.parametrize(
    "key", ["short_maturity_normal", "local_volatility", "local_volatility_small_grid"]
)
def test_implied_lognormal_volatility_matches_cpp(cpp: dict[str, Any], key: str) -> None:
    """The three back-solving arms.

    LOOSE tier rather than TIGHT: these do not return a formula value,
    they invert ``blackFormulaImpliedStdDev`` (a Newton-safe solve with
    C++ default ``accuracy = 1e-6`` on the total std-dev) against the
    mode's own option price. Two independent root-finders agreeing to
    better than that accuracy is not something the reference can
    guarantee; measured agreement is ~5e-10 absolute.
    """
    block = cpp[key]
    s = _section(key)
    for i, strike in enumerate(block["strikes"]):
        tolerance.loose(s.volatility(strike), float(block["volatility"][i]))


def test_normal_arm_reports_implied_lognormal_not_normal_vol() -> None:
    """``volatility()`` on the normal arm is NOT ``normalVolatility``.

    C++ ``volatilityImpl(strike, ZabrShortMaturityNormal)`` back-solves a
    lognormal vol from the Bachelier price. The two conventions are
    related at the money by ``sigma_lognormal ~ sigma_normal / F``, so
    returning the normal vol instead would be wrong by a factor of
    1/F — here 33x. Checking the ratio against 1/F pins the direction of
    the conversion, not merely that the two differ.
    """
    s = _section("short_maturity_normal")
    lognormal_ish = s.volatility(_FORWARD)
    normal_vol = s.model().normal_volatility(_FORWARD)
    ratio = lognormal_ish / normal_vol
    assert abs(ratio - 1.0 / _FORWARD) < 0.05 / _FORWARD


def test_far_otm_volatility_degrades_to_zero(cpp: dict[str, Any]) -> None:
    """The implied-vol inversion swallows its exception and returns 0.0.

    At 30x the forward the FD call price has underflowed, so
    ``blackFormulaImpliedStdDev`` has no solution; C++ catches everything
    and leaves the vol at its 0.0 initialiser.
    """
    block = cpp["local_volatility_small_grid"]
    assert block["volatility"][-1] == 0.0
    s = _section("local_volatility_small_grid")
    assert s.volatility(block["strikes"][-1]) == 0.0


def _log_price_second_difference(s: ZabrSmileSection, strikes: tuple[float, ...]) -> float:
    """Second difference of ``log(call(K))`` over three EQUALLY SPACED strikes.

    Zero exactly when the call price is ``exp(-a*K + b)`` on that span,
    which is what the right tail is and what the spline interior is not.
    """
    logs = [math.log(s.option_price(k, 1, 1.0)) for k in strikes]
    return abs(logs[0] - 2.0 * logs[1] + logs[2])


def test_exponential_tail_takes_over_past_the_last_grid_strike() -> None:
    """The grid ends at ``max(moneyness) * forward`` and a log-linear tail follows.

    Asserted through the public price only: ``log(call(K))`` is exactly
    affine in K beyond the last grid strike (the C++ ``exp(-a_*K + b_)``
    extrapolation) and is not affine across it, which locates the grid's
    right end without reading private state. For this section the last
    moneyness is 2.0 so that end is ``2.0 * forward = 0.06``.
    """
    s = _section("local_volatility_small_grid")
    k_max = 2.0 * _FORWARD

    # Wholly inside the tail: affine, to rounding.
    assert _log_price_second_difference(s, (k_max + 0.01, k_max + 0.02, k_max + 0.03)) < 1e-12
    assert _log_price_second_difference(s, (0.2, 0.4, 0.6)) < 1e-12
    # Straddling the last grid strike: the spline is not log-linear there.
    assert _log_price_second_difference(s, (k_max - 0.02, k_max - 0.01, k_max)) > 1e-3
    # And the tail decays.
    assert s.option_price(k_max + 0.01, 1, 1.0) < s.option_price(k_max, 1, 1.0)


def test_fd_refinement_changes_the_grid_and_therefore_the_prices() -> None:
    """``fd_refinement`` inserts interior points, which moves the FD prices.

    A section built with more interior points per moneyness gap prices a
    denser strike ladder off the same PDE solution, so its interpolated
    prices differ; equality would mean the argument is being ignored.
    """
    coarse = ZabrSmileSection(
        forward=_FORWARD,
        zabr_params=_PARAMS,
        exercise_time=_TAU,
        evaluation=ZabrEvaluation.LocalVolatility,
        moneyness=_SMALL_MONEYNESS,
        fd_refinement=1,
    )
    fine = ZabrSmileSection(
        forward=_FORWARD,
        zabr_params=_PARAMS,
        exercise_time=_TAU,
        evaluation=ZabrEvaluation.LocalVolatility,
        moneyness=_SMALL_MONEYNESS,
        fd_refinement=6,
    )
    assert coarse.option_price(0.025, 1, 1.0) != fine.option_price(0.025, 1, 1.0)
    # ...but only slightly: both discretise the same continuous price.
    tolerance.custom(
        coarse.option_price(0.025, 1, 1.0),
        fine.option_price(0.025, 1, 1.0),
        abs_tol=1e-5,
        rel_tol=1e-2,
        reason="two cubic-spline discretisations of the same FD price curve",
    )


def test_discount_scales_prices() -> None:
    s = _section("local_volatility_small_grid")
    for k in (0.01, 0.03, 0.06):
        tolerance.tight(
            s.option_price(k, 1, 0.75), 0.75 * s.option_price(k, 1, 1.0)
        )


@pytest.mark.slow
def test_full_fd_prices_match_cpp(cpp: dict[str, Any]) -> None:
    """Each grid point priced by its own 2-D Hundsdorfer PDE.

    LOOSE tier: the five implicit-Euler damping steps solve the full 2-D
    operator with BiCGstab at ``rel_tol = 1e-8``, so the rolled-back grids
    can only agree to that order. Measured agreement is ~4e-12 absolute
    on the prices.
    """
    block = cpp["full_fd_small_grid"]
    s = _section("full_fd_small_grid")
    for i, strike in enumerate(block["strikes"]):
        tolerance.loose(s.option_price(strike, 1, 1.0), float(block["call_price"][i]))
        tolerance.loose(s.option_price(strike, -1, 1.0), float(block["put_price"][i]))


@pytest.mark.slow
def test_full_fd_volatilities_match_cpp(cpp: dict[str, Any]) -> None:
    """FullFd implied vols.

    Bound derived rather than picked. Two error sources compose:

    1. The 2-D PDE prices agree to ~4e-12 absolute (BiCGstab ``rel_tol``
       1e-8 in the damping steps, see the price test above).
    2. Converting a price to an implied vol divides by vega. The worst
       case in this ladder is the deepest ITM strike, K = 0.0005 against
       a 0.03 forward: with sigma ~ 0.65 and T = 5, the total std-dev is
       s = 1.45, d1 = ln(F/K)/s + s/2 = 3.55, and
       dC/ds = F*phi(d1) = 0.03 * 7.4e-4 = 2.2e-5, so
       dsigma/dC = 1/(sqrt(T)*dC/ds) ~ 2.0e4. A 4e-12 price error is
       therefore worth ~1e-7 of volatility.

    On top of that ``blackFormulaImpliedStdDev`` itself only converges to
    ``accuracy = 1e-6`` on s, i.e. 4.5e-7 on sigma. 1e-6 absolute is the
    smallest bound consistent with both; measured agreement is 1.8e-8.
    """
    block = cpp["full_fd_small_grid"]
    s = _section("full_fd_small_grid")
    for i, strike in enumerate(block["strikes"]):
        tolerance.custom(
            s.volatility(strike),
            float(block["volatility"][i]),
            abs_tol=1.0e-6,
            rel_tol=1.0e-6,
            reason=(
                "deep-ITM implied-vol inversion amplifies the BiCGstab-limited "
                "price agreement by 1/vega ~ 2e4, and the root-finder's own "
                "accuracy on the total std-dev is 1e-6 (~4.5e-7 on sigma)"
            ),
        )
