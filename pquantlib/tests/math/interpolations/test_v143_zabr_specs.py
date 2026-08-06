"""Cross-validate ``ZabrSpecs`` and the ``Zabr`` factory against C++ v1.43.

Probe source: migration-harness/cpp/probes/v143_math_zabrspecs/probe.cpp
Reference:    migration-harness/references/v143/math/zabrspecs.json

``detail::ZabrSpecs<Evaluation>`` (zabrinterpolation.hpp:36) is the model
policy ``XABRInterpolation<Model>`` is instantiated with. The substantive part
is the ``direct`` / ``inverse`` bijection between the constrained 5-parameter
box and unconstrained R^5, which lets an unconstrained optimiser drive a
constrained fit. The probe sweeps **both arms of every branch** in those maps
— the saturating arms (``|x0| >= 5``, ``|x1| >= sqrt(-log(eps1))``,
``|x3| >= 2.5 pi`` on both signs) are exactly the ones a port is tempted to
drop as unreachable.

``guess`` is pinned separately because its running index over the random draws
goes beta, alpha, nu, rho, gamma — not parameter order — and a fixed parameter
consumes no draw.

Tolerance tier: TIGHT (1e-14 abs / 1e-12 rel). Every quantity here is a short
chain of libm calls on the same IEEE doubles C++ evaluates, so only last-place
differences in ``pow`` / ``atan`` / ``exp`` are expected.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pquantlib.math.interpolations.zabr_formula import ZabrEvaluation
from pquantlib.math.interpolations.zabr_interpolation import (
    NULL_REAL,
    Zabr,
    ZabrInterpolation,
    ZabrSpecs,
)
from pquantlib.termstructures.volatility.zabr_smile_section import ZabrSmileSection
from pquantlib.testing import reference_reader, tolerance


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("v143/math/zabrspecs")


@pytest.fixture
def specs() -> ZabrSpecs:
    return ZabrSpecs()


def test_null_real_matches_cpp(cpp: dict[str, Any]) -> None:
    tolerance.exact(NULL_REAL, float(cpp["null_real"]))


@pytest.mark.exact
def test_scalar_traits_match_cpp(cpp: dict[str, Any], specs: ZabrSpecs) -> None:
    assert specs.dimension() == int(cpp["dimension"])
    tolerance.exact(specs.eps(), float(cpp["eps"]))
    tolerance.exact(specs.eps1(), float(cpp["eps1"]))
    tolerance.exact(specs.eps2(), float(cpp["eps2"]))
    tolerance.exact(specs.dilation_factor(), float(cpp["dilation_factor"]))


@pytest.mark.exact
def test_zabr_factory_global_flag_matches_cpp(cpp: dict[str, Any]) -> None:
    assert Zabr.global_ is bool(cpp["zabr_factory_global"])


@pytest.mark.tight
def test_default_values_matches_cpp(cpp: dict[str, Any], specs: ZabrSpecs) -> None:
    for row in cpp["default_values"]:
        params = [NULL_REAL if float(v) == float(cpp["null_real"]) else float(v) for v in row["in"]]
        specs.default_values(params, [False] * 5, float(row["forward"]), float(row["expiry"]), [])
        for got, want in zip(params, row["out"], strict=True):
            tolerance.tight(got, float(want))


@pytest.mark.tight
def test_guess_matches_cpp(cpp: dict[str, Any], specs: ZabrSpecs) -> None:
    for row in cpp["guess"]:
        # The probe seeds distinct sentinels so untouched slots stay visible;
        # reproduce them exactly (C++ `guess` reads values[1] even when beta
        # is fixed and therefore never written by this call).
        values = np.array([-1.0, -2.0, -3.0, -4.0, -5.0], dtype=np.float64)
        specs.guess(
            values,
            [bool(f) for f in row["fixed"]],
            float(row["forward"]),
            1.0,
            [float(v) for v in row["r"]],
            [],
        )
        for got, want in zip(values, row["out"], strict=True):
            tolerance.tight(float(got), float(want))


@pytest.mark.tight
def test_direct_matches_cpp(cpp: dict[str, Any], specs: ZabrSpecs) -> None:
    for row in cpp["direct"]:
        x = np.array([float(v) for v in row["x"]], dtype=np.float64)
        y = specs.direct(x, [False] * 5, [0.0] * 5, 0.03)
        for got, want in zip(y, row["y"], strict=True):
            tolerance.tight(float(got), float(want))


@pytest.mark.tight
def test_inverse_matches_cpp(cpp: dict[str, Any], specs: ZabrSpecs) -> None:
    for row in cpp["inverse"]:
        y = np.array([float(v) for v in row["y"]], dtype=np.float64)
        x = specs.inverse(y, [False] * 5, [0.0] * 5, 0.03)
        for got, want in zip(x, row["x"], strict=True):
            tolerance.tight(float(got), float(want))


@pytest.mark.tight
def test_direct_and_inverse_round_trip(specs: ZabrSpecs) -> None:
    """``direct(inverse(y)) == y`` on the non-saturating interior of the box."""
    y = np.array([0.25, 0.5, 2.5, 0.5, 0.95], dtype=np.float64)
    back = specs.direct(specs.inverse(y, [False] * 5, [0.0] * 5, 0.03), [False] * 5, [0.0] * 5, 0.03)
    for got, want in zip(back, y, strict=True):
        tolerance.tight(float(got), float(want))


@pytest.mark.tight
def test_weight_matches_cpp(cpp: dict[str, Any], specs: ZabrSpecs) -> None:
    for row in cpp["weight"]:
        got = specs.weight(
            float(row["strike"]), float(row["forward"]), float(row["std_dev"]), []
        )
        tolerance.tight(got, float(row["out"]))


def test_instance_builds_a_zabr_smile_section(specs: ZabrSpecs) -> None:
    """C++ ``instance`` returns ``ZabrSmileSection<Evaluation>``."""
    section = specs.instance(1.0, 0.03, [0.2, 0.5, 0.6, 0.0, 1.0], [])
    assert isinstance(section, ZabrSmileSection)
    tolerance.exact(section.exercise_time(), 1.0)


def test_instance_honours_the_evaluation_template_argument() -> None:
    """``ZabrSpecs<Evaluation>`` is templated on the evaluation in C++."""
    section = ZabrSpecs(ZabrEvaluation.ShortMaturityNormal).instance(
        1.0, 0.03, [0.2, 0.5, 0.6, 0.0, 1.0], []
    )
    plain = ZabrSpecs().instance(1.0, 0.03, [0.2, 0.5, 0.6, 0.0, 1.0], [])
    assert section.volatility(0.04) != plain.volatility(0.04)


def test_zabr_factory_stamps_out_interpolations() -> None:
    """C++ ``Zabr::interpolate`` builds one ZabrInterpolation per slice."""
    strikes = [0.02, 0.025, 0.03, 0.035, 0.04]
    factory = Zabr(
        t=1.0,
        forward=0.03,
        alpha=0.2,
        beta=0.5,
        nu=0.6,
        rho=0.0,
        gamma=1.0,
        alpha_is_fixed=False,
        beta_is_fixed=True,
        nu_is_fixed=False,
        rho_is_fixed=False,
        gamma_is_fixed=True,
        max_guesses=1,
    )
    direct = ZabrInterpolation(
        strikes,
        [0.30, 0.28, 0.27, 0.28, 0.30],
        1.0,
        0.03,
        alpha=0.2,
        beta=0.5,
        nu=0.6,
        rho=0.0,
        gamma=1.0,
        beta_is_fixed=True,
        gamma_is_fixed=True,
        max_guesses=1,
    )
    fitted = factory.interpolate(strikes, [0.30, 0.28, 0.27, 0.28, 0.30])
    assert isinstance(fitted, ZabrInterpolation)
    # The factory is a pure configuration carrier: same inputs, same fit.
    tolerance.exact(fitted.alpha(), direct.alpha())
    tolerance.exact(fitted.beta(), direct.beta())
    tolerance.exact(fitted.nu(), direct.nu())
    tolerance.exact(fitted.rho(), direct.rho())
    tolerance.exact(fitted.gamma(), direct.gamma())
