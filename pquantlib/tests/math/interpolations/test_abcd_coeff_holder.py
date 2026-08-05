"""Cross-validate AbcdCoeffHolder and the Abcd factory against the v1.43 C++ probe.

Reference: ``migration-harness/references/v143/math/interp/kernel.json`` —
``abcd_coeff_holder`` section.

``AbcdCoeffHolder``'s constructor is small and entirely made of edge cases:
a ``Null<Real>`` parameter is replaced by a hard-coded default **and** has its
``*IsFixed`` flag forced to false regardless of what the caller asked for,
while ``AbcdMathFunction::validate`` runs on the *original* arguments rather
than the defaulted ones. The probe enumerates each Null position
individually, all four at once, and the all-non-Null baseline.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pquantlib.math.interpolations.abcd_interpolation import (
    NULL_REAL,
    Abcd,
    AbcdCoeffHolder,
    AbcdInterpolation,
)
from pquantlib.testing import reference_reader, tolerance


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("v143/math/interp/kernel")


def test_null_real_sentinel_matches_cpp(cpp: dict[str, Any]) -> None:
    """C++ ``Null<Real>()`` is ``numeric_limits<float>::max()``, not double max."""
    tolerance.exact(NULL_REAL, float(cpp["abcd_coeff_holder"]["null_real"]))


def test_traits_match_cpp(cpp: dict[str, Any]) -> None:
    """``Abcd::global`` and ``Abcd::requiredPoints`` are static C++ traits."""
    block = cpp["abcd_coeff_holder"]
    assert Abcd.global_ is bool(block["abcd_global"])
    assert Abcd.required_points == int(block["abcd_required_points"])


def test_null_defaulting_matches_cpp(cpp: dict[str, Any]) -> None:
    """Coefficients and fixed-ness after the Null substitution, bit-exact.

    These are all literal assignments — no arithmetic — so EXACT is the right
    tier and anything looser would hide a wrong default.
    """
    block = cpp["abcd_coeff_holder"]
    null_real = float(block["null_real"])
    for case in block["cases"]:
        a, b, c, d = (
            NULL_REAL if float(v) == null_real else float(v) for v in case["in"]
        )
        af, bf, cf, df = (bool(f) for f in case["in_is_fixed"])
        holder = AbcdCoeffHolder(a, b, c, d, af, bf, cf, df)
        for got, expected in zip(
            (holder.a, holder.b, holder.c, holder.d), case["out"], strict=True
        ):
            tolerance.exact(got, float(expected))
        assert [
            holder.a_is_fixed,
            holder.b_is_fixed,
            holder.c_is_fixed,
            holder.d_is_fixed,
        ] == [bool(f) for f in case["out_is_fixed"]]
        # C++ ``abcdEndCriteria_`` starts at ``EndCriteria::None`` (enum 0).
        assert holder.abcd_end_criteria == int(case["end_criteria"])


def test_a_null_parameter_cannot_be_fixed(cpp: dict[str, Any]) -> None:
    """The quirk stated on its own: Null wins over the caller's isFixed flag.

    ``AbcdCoeffHolder(Null, 0.17, 0.54, 0.17, True, True, True, True)`` must
    come back with ``a_is_fixed == False`` and the other three ``True``.
    """
    holder = AbcdCoeffHolder(NULL_REAL, 0.17, 0.54, 0.17, True, True, True, True)
    tolerance.exact(holder.a, -0.06)
    assert holder.a_is_fixed is False
    assert (holder.b_is_fixed, holder.c_is_fixed, holder.d_is_fixed) == (True, True, True)


def test_validation_sees_the_original_arguments() -> None:
    """C++ validates ``(a, b, c, d)`` as passed, not as defaulted.

    A Null ``c`` therefore validates as +3.4e38 (passes ``c >= 0``) and only
    *then* becomes 0.54, whereas an explicit negative ``c`` is rejected. That
    asymmetry is the reason the check is worth pinning at all.
    """
    _ = AbcdCoeffHolder(-0.06, 0.17, NULL_REAL, 0.17, False, False, False, False)
    with pytest.raises(Exception, match="c parameter must be non-negative"):
        AbcdCoeffHolder(-0.06, 0.17, -1.0, 0.17, False, False, False, False)
    with pytest.raises(Exception, match="d parameter must be non-negative"):
        AbcdCoeffHolder(-0.06, 0.17, 0.54, -1.0, False, False, False, False)
    with pytest.raises(Exception, match="a\\+d must be non-negative"):
        AbcdCoeffHolder(-1.0, 0.17, 0.54, 0.17, False, False, False, False)


def test_interpolation_exposes_its_holder() -> None:
    """C++ ``AbcdInterpolation::coeffs()`` is the impl downcast to the holder."""
    times = np.array([0.5, 1.0, 2.0, 3.0, 5.0], dtype=np.float64)
    vols = np.array([0.20, 0.21, 0.20, 0.19, 0.18], dtype=np.float64)
    interp = AbcdInterpolation(times, vols)
    holder = interp.coeffs
    assert isinstance(holder, AbcdCoeffHolder)
    tolerance.exact(holder.a, -0.06)
    tolerance.exact(holder.b, 0.17)
    tolerance.exact(holder.c, 0.54)
    tolerance.exact(holder.d, 0.17)


def test_null_arguments_flow_through_the_interpolation() -> None:
    """Passing Null to ``AbcdInterpolation`` picks up the holder's defaults."""
    times = np.array([0.5, 1.0, 2.0, 3.0, 5.0], dtype=np.float64)
    vols = np.array([0.20, 0.21, 0.20, 0.19, 0.18], dtype=np.float64)
    interp = AbcdInterpolation(times, vols, a=NULL_REAL, c_is_fixed=True)
    tolerance.exact(interp.coeffs.a, -0.06)
    assert interp.coeffs.a_is_fixed is False
    assert interp.coeffs.c_is_fixed is True


def test_factory_matches_direct_construction() -> None:
    """``Abcd(...).interpolate(times, vols)`` is the C++ traits path."""
    times = np.array([0.5, 1.0, 2.0, 3.0, 5.0], dtype=np.float64)
    vols = np.array([0.20, 0.21, 0.20, 0.19, 0.18], dtype=np.float64)
    direct = AbcdInterpolation(
        times, vols, -0.06, 0.17, 0.54, 0.17, False, False, False, False
    )
    via_factory = Abcd(-0.06, 0.17, 0.54, 0.17, False, False, False, False).interpolate(
        times, vols
    )
    tolerance.exact(via_factory.a(), direct.a())
    tolerance.exact(via_factory.b(), direct.b())
    tolerance.exact(via_factory.c(), direct.c())
    tolerance.exact(via_factory.d(), direct.d())
    for t in (0.5, 1.25, 2.5, 5.0):
        tolerance.exact(via_factory(t), direct(t))
