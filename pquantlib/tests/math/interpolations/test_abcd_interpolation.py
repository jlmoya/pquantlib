"""Cross-validate AbcdInterpolation against the L10-C C++ probe.

Reference: ``migration-harness/references/cluster/l10c.json`` —
``abcd_interpolation`` section. Synthetic vols generated from
known (a, b, c, d) = (-0.06, 0.17, 0.54, 0.17); both implementations
recover the same fit via least-squares.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.math.interpolations.abcd_interpolation import (
    AbcdInterpolation,
    abcd_value,
    validate_abcd,
)
from pquantlib.testing import reference_reader, tolerance


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("cluster/l10c")


def _make() -> AbcdInterpolation:
    block = reference_reader.load("cluster/l10c")["abcd_interpolation"]
    times = np.array(block["times"], dtype=np.float64)
    vols = np.array(block["vols"], dtype=np.float64)
    # C++ probe uses a (-0.05, 0.15, 0.50, 0.16) starting point.
    return AbcdInterpolation(
        times, vols, a=-0.05, b=0.15, c=0.50, d=0.16,
    )


def test_recovers_truth_when_data_is_exact_abcd() -> None:
    """Python TRF recovers the synthetic truth (a, b, c, d) parameters.

    When the input data is exactly ``(a + b*t)*exp(-c*t) + d``, the
    Python ``scipy.optimize.least_squares(method='trf')`` arm converges
    to the true minimum with residuals ~1e-13. The C++ probe also
    runs LM but starts from a less-favourable initial guess and stops
    at a local minimum with rms ~1.8e-3 — see the L10-C divergence note
    in the module docstring. The Python residual is *smaller*, which
    means the Python fit is *better*; both solvers respect the
    documented 'least squares' contract.
    """
    interp = _make()
    # Truth params from the probe.
    tolerance.loose(interp.a(), -0.06,
                    reason="trf converges to truth on noiseless data")
    tolerance.loose(interp.b(), 0.17,
                    reason="trf converges to truth on noiseless data")
    tolerance.loose(interp.c(), 0.54,
                    reason="trf converges to truth on noiseless data")
    tolerance.loose(interp.d(), 0.17,
                    reason="trf converges to truth on noiseless data")
    # Tight residuals (the Python fit is essentially exact).
    assert interp.rms_error() < 1.0e-8
    assert interp.max_error() < 1.0e-8


def test_pillar_recovery_at_lsq_residual_floor(cpp: dict[str, Any]) -> None:
    """At each input time the fitted abcd reproduces the market vol.

    Custom tier — TRF's residual floor for this 4-param non-linear LSQ
    is ~1e-12 in absolute terms (xtol=ftol=1e-12 in the L9-C
    precedent). LOOSE is more generous than needed; we use ~1e-10 as
    the headroom buffer.
    """
    block = cpp["abcd_interpolation"]
    times = [float(e) for e in block["times"]]
    vols = [float(e) for e in block["vols"]]
    interp = _make()
    for t, v in zip(times, vols, strict=True):
        tolerance.custom(
            interp(t), v,
            abs_tol=1.0e-10, rel_tol=1.0e-10,
            reason="trf xtol=ftol=1e-12 leaves ~1e-12 residual floor",
        )


def test_python_fit_diverges_from_cpp_fit_documented(cpp: dict[str, Any]) -> None:
    """Documented divergence — Python TRF and C++ projected-LM converge
    to *different* (a, b, c, d) on this 4-param / 6-data system.

    The Python fit finds the global minimum (residuals ~1e-13); the C++
    fit stops at a local minimum with rms ~1.8e-3. This is captured by
    Phase 9 / L10-C divergence notes in the module docstring. We assert
    the divergence is observable — Python's pillar vols agree with
    the *input* (truth) to TIGHT, while the C++ fitted-pillar values
    differ from the input by ~1e-3.
    """
    block = cpp["abcd_interpolation"]
    times = [float(e) for e in block["times"]]
    vols = [float(e) for e in block["vols"]]
    cpp_fitted_pillars = [float(e) for e in block["fitted_at_pillars"]]
    interp = _make()
    # Python interp matches the input vols (truth) to ~1e-10 (solver
    # residual floor; see test_pillar_recovery_at_lsq_residual_floor).
    for t, v_truth in zip(times, vols, strict=True):
        tolerance.custom(
            interp(t), v_truth,
            abs_tol=1.0e-10, rel_tol=1.0e-10,
            reason="trf xtol=ftol=1e-12 leaves ~1e-12 residual floor",
        )
    # C++ converges to a local min with ~1e-3 pillar residuals.
    cpp_max_resid = max(
        abs(float(v_cpp) - float(v_truth))
        for v_cpp, v_truth in zip(cpp_fitted_pillars, vols, strict=True)
    )
    # Assert the divergence is observable — at least 1e-4 pillar
    # residual on the C++ side, confirming the two solvers genuinely
    # found different minima.
    assert cpp_max_resid > 1.0e-4, (
        f"C++ fit unexpectedly converged to the global min "
        f"(max residual: {cpp_max_resid})"
    )


def test_validate_abcd_rejects_negative_c() -> None:
    with pytest.raises(LibraryException):
        validate_abcd(0.1, 0.1, -0.1, 0.1)


def test_validate_abcd_rejects_negative_d() -> None:
    with pytest.raises(LibraryException):
        validate_abcd(0.1, 0.1, 0.1, -0.1)


def test_validate_abcd_rejects_negative_apd() -> None:
    with pytest.raises(LibraryException):
        validate_abcd(-1.0, 0.1, 0.1, 0.1)


def test_abcd_value_at_zero() -> None:
    """``f(0) = a + d``."""
    a, b, c, d = -0.06, 0.17, 0.54, 0.17
    tolerance.tight(abcd_value(0.0, a, b, c, d), a + d)


def test_abcd_value_at_infinity_long_horizon() -> None:
    """At t -> infinity the function -> d."""
    a, b, c, d = -0.06, 0.17, 0.54, 0.17
    # t = 100 — exp(-c*t) is essentially zero.
    tolerance.loose(abcd_value(100.0, a, b, c, d), d)


def test_abcd_value_negative_t_returns_zero() -> None:
    """``f(t) = 0`` for ``t < 0`` per the C++ guard."""
    tolerance.exact(abcd_value(-1.0, 0.1, 0.1, 0.1, 0.1), 0.0)


def test_fix_all_parameters() -> None:
    """When every parameter is fixed, the fit just evaluates the initial."""
    a, b, c, d = -0.06, 0.17, 0.54, 0.17
    times = np.array([0.25, 0.5, 1.0, 2.0, 5.0, 10.0])
    vols = np.array(
        [abcd_value(float(t), a, b, c, d) for t in times], dtype=np.float64
    )
    interp = AbcdInterpolation(
        times, vols, a=a, b=b, c=c, d=d,
        a_is_fixed=True, b_is_fixed=True, c_is_fixed=True, d_is_fixed=True,
    )
    # Fitted parameters equal the initial values exactly.
    tolerance.exact(interp.a(), a)
    tolerance.exact(interp.b(), b)
    tolerance.exact(interp.c(), c)
    tolerance.exact(interp.d(), d)
    # Diagnostics: zero residuals.
    tolerance.exact(interp.rms_error(), 0.0)
    tolerance.exact(interp.max_error(), 0.0)


def test_inspectors_return_python_floats() -> None:
    interp = _make()
    assert isinstance(interp.a(), float)
    assert isinstance(interp.b(), float)
    assert isinstance(interp.c(), float)
    assert isinstance(interp.d(), float)
    assert isinstance(interp.rms_error(), float)
    assert isinstance(interp.max_error(), float)
    assert isinstance(interp.converged(), bool)


def test_negative_times_raise() -> None:
    times = np.array([-1.0, 1.0, 2.0], dtype=np.float64)
    vols = np.array([0.1, 0.1, 0.1], dtype=np.float64)
    with pytest.raises(LibraryException):
        AbcdInterpolation(times, vols)
