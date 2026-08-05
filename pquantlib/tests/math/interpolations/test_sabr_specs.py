"""Cross-validate the SABR XABR plumbing against the v1.43 C++ probe.

Reference: ``migration-harness/references/v143/math/interp/kernel.json`` —
``sabr_specs`` and ``sabr_wrapper`` sections.

``SABRSpecs`` is the model policy the C++ ``XABRInterpolation<Model>``
template is instantiated with. The substantive part is the ``direct`` /
``inverse`` bijection between the constrained parameter box and unconstrained
R^4, which lets an unconstrained optimiser drive a constrained fit. The probe
sweeps **both arms of every branch** in those two maps — the saturating arms
(``|x0| >= 5``, ``|x1| >= sqrt(-log(eps1))``, ``|x3| >= 2.5 pi`` on both
signs) are exactly the ones a port is tempted to drop as unreachable.

``guess`` is pinned separately because its running index over the random
draws goes beta, alpha, nu, rho — not parameter order — and a fixed parameter
consumes no draw. Getting that wrong silently reshuffles a multi-start search
without ever raising.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pquantlib.math.interpolations.sabr_interpolation import (
    NULL_REAL,
    SabrInterpolation,
    SABRSpecs,
    SABRWrapper,
)
from pquantlib.math.interpolations.sabr_interpolation import (
    SABRInterpolation as SABRInterpolationAlias,
)
from pquantlib.termstructures.volatility.volatility_type import VolatilityType
from pquantlib.testing import reference_reader, tolerance


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("v143/math/interp/kernel")


def _denull(values: list[Any], null_real: float) -> list[float]:
    """Map the reference's Null sentinel back onto :data:`NULL_REAL`."""
    return [NULL_REAL if float(v) == null_real else float(v) for v in values]


def test_null_real_sentinel_matches_cpp(cpp: dict[str, Any]) -> None:
    """C++ ``Null<Real>()`` is ``numeric_limits<float>::max()``, not double max."""
    tolerance.exact(NULL_REAL, float(cpp["sabr_specs"]["null_real"]))


def test_scalar_traits_match_cpp(cpp: dict[str, Any]) -> None:
    """dimension / eps1 / eps2 / dilationFactor."""
    block = cpp["sabr_specs"]
    specs = SABRSpecs()
    assert specs.dimension() == int(block["dimension"])
    tolerance.exact(specs.eps1(), float(block["eps1"]))
    tolerance.exact(specs.eps2(), float(block["eps2"]))
    tolerance.exact(specs.dilation_factor(), float(block["dilation_factor"]))


def test_default_values_match_cpp_tight(cpp: dict[str, Any]) -> None:
    """``defaultValues`` fills Null slots in place, beta before alpha.

    The order is load-bearing: alpha's default reads the (possibly
    just-defaulted) beta, and takes the ``pow(forward+shift, 1-beta)`` arm
    only while ``beta < 0.9999``. The probe covers both arms and both the
    shifted and unshifted forward.
    """
    block = cpp["sabr_specs"]
    null_real = float(block["null_real"])
    specs = SABRSpecs()
    for case in block["default_values"]:
        params = _denull(case["in"], null_real)
        add_params = [float(case["shift"])] if case["has_add_params"] else []
        specs.default_values(
            params, [False] * 4, float(case["forward"]), float(case["expiry"]), add_params
        )
        for got, expected in zip(params, case["out"], strict=True):
            tolerance.tight(got, float(expected))


def test_guess_matches_cpp_tight(cpp: dict[str, Any]) -> None:
    """``guess`` seeds only the free parameters, consuming ``r`` in beta-first order.

    The probe seeds ``values`` with -1, -2, -3, -4 so an untouched slot is
    visible in the reference: the all-fixed case must leave all four at their
    sentinels.
    """
    block = cpp["sabr_specs"]
    specs = SABRSpecs()
    for case in block["guess"]:
        values = np.array([-1.0, -2.0, -3.0, -4.0], dtype=np.float64)
        add_params = [float(case["shift"])] if float(case["shift"]) != 0.0 else []
        specs.guess(
            values,
            [bool(f) for f in case["is_fixed"]],
            float(case["forward"]),
            1.0,
            [float(v) for v in case["r"]],
            add_params,
        )
        for got, expected in zip(values, case["out"], strict=True):
            tolerance.tight(float(got), float(expected))


def test_direct_matches_cpp_tight(cpp: dict[str, Any]) -> None:
    """R^4 -> the constrained box, including every saturating arm."""
    block = cpp["sabr_specs"]
    specs = SABRSpecs()
    for case in block["direct"]:
        y = specs.direct(
            np.asarray(case["x"], dtype=np.float64), [False] * 4, [0.0] * 4, 0.03
        )
        for got, expected in zip(y, case["y"], strict=True):
            tolerance.tight(float(got), float(expected))


def test_inverse_matches_cpp_tight(cpp: dict[str, Any]) -> None:
    """The constrained box -> R^4, including the linear arms above 25 + eps1."""
    block = cpp["sabr_specs"]
    specs = SABRSpecs()
    for case in block["inverse"]:
        x = specs.inverse(
            np.asarray(case["y"], dtype=np.float64), [False] * 4, [0.0] * 4, 0.03
        )
        for got, expected in zip(x, case["x"], strict=True):
            tolerance.tight(float(got), float(expected))


def test_direct_inverse_roundtrip(cpp: dict[str, Any]) -> None:
    """``direct(inverse(y)) == y`` on the reference's in-branch points.

    Only where the maps are actually bijective: beyond the saturation
    thresholds ``direct`` is many-to-one by construction, so a round trip
    there is meaningless and is not asserted.
    """
    block = cpp["sabr_specs"]
    specs = SABRSpecs()
    for case in block["inverse"]:
        y = np.asarray(case["y"], dtype=np.float64)
        back = specs.direct(
            specs.inverse(y, [False] * 4, [0.0] * 4, 0.03), [False] * 4, [0.0] * 4, 0.03
        )
        for got, expected in zip(back, y, strict=True):
            tolerance.custom(
                float(got),
                float(expected),
                abs_tol=1e-14,
                rel_tol=1e-10,
                reason=(
                    "the round trip composes sqrt/square and asin/sin, each of "
                    "which loses about half the significand's guard digits near "
                    "its endpoints; 1e-10 is the composed condition number of "
                    "that pair, not an empirical fit"
                ),
            )


def test_weight_matches_cpp_tight(cpp: dict[str, Any]) -> None:
    """``weight`` is ``blackFormulaStdDevDerivative(strike, fwd, sd, 1, shift)``."""
    block = cpp["sabr_specs"]
    specs = SABRSpecs()
    for case in block["weight"]:
        got = specs.weight(
            float(case["strike"]),
            float(case["forward"]),
            float(case["std_dev"]),
            [float(case["shift"])],
        )
        tolerance.tight(got, float(case["weight"]))


def test_sabr_wrapper_matches_cpp_tight(cpp: dict[str, Any]) -> None:
    """``SABRWrapper.volatility`` in both volatility types, shifted and not."""
    for case in cpp["sabr_wrapper"]["cases"]:
        add_params = [float(case["shift"])] if case["has_add_params"] else []
        wrapper = SABRWrapper(
            float(case["t"]),
            float(case["forward"]),
            [float(p) for p in case["params"]],
            add_params,
        )
        for strike, lognormal, normal in zip(
            case["strikes"], case["shifted_lognormal"], case["normal"], strict=True
        ):
            tolerance.tight(
                wrapper.volatility(float(strike), VolatilityType.ShiftedLognormal),
                float(lognormal),
            )
            tolerance.tight(
                wrapper.volatility(float(strike), VolatilityType.Normal), float(normal)
            )


def test_sabr_wrapper_rejects_non_positive_shifted_forward() -> None:
    """C++ requires ``forward + shift > 0`` before touching the formula."""
    with pytest.raises(Exception, match="forward\\+shift must be positive"):
        SABRWrapper(1.0, -0.02, [0.25, 0.5, 0.4, -0.2], [0.01])


def test_sabr_wrapper_validates_its_parameters() -> None:
    """C++ calls ``validateSabrParameters`` in the constructor."""
    with pytest.raises(Exception, match="alpha must be positive"):
        SABRWrapper(1.0, 0.03, [-0.25, 0.5, 0.4, -0.2], [])
    with pytest.raises(Exception, match="rho square"):
        SABRWrapper(1.0, 0.03, [0.25, 0.5, 0.4, 1.5], [])


def test_cpp_class_name_is_exported() -> None:
    """The coverage gate matches C++ spelling: ``SABRInterpolation``."""
    assert SABRInterpolationAlias is SabrInterpolation
