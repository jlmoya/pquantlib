"""Cross-validate AbcdInterpolation against the L10-C C++ probe.

Reference: ``migration-harness/references/cluster/l10c.json`` —
``abcd_interpolation`` section.

NOTE ON THE PROBE'S INPUT. ``vols`` there is the INSTANTANEOUS
``AbcdMathFunction::operator()`` at ``times`` with ``(a,b,c,d) =
(-0.06, 0.17, 0.54, 0.17)``, but ``AbcdInterpolation::value`` returns
``abcdCalibrator_->value(x)`` (abcdinterpolation.hpp:126-130), which is
``abcdBlackVolatility`` — the AVERAGE vol over ``[0, x]`` (abcd.hpp:105-108).
The input is therefore NOT reproducible by the model, which is why C++'s
``*_fitted`` sit nowhere near ``*_true`` and its residual floor is ~2.6e-3.
That is a property of the probe's synthetic data, not of either optimiser.

An earlier revision read that mismatch as "Python TRF finds the global minimum
where C++ LM stops at a local one" and asserted the Python fit against
``*_true``. It passed only because ``AbcdInterpolation._value`` returned the
instantaneous ``abcd_value`` instead of ``abcd_black_volatility`` — the port
was fitting a different function. With that corrected, Python reproduces the
C++ fit to under 2e-7 absolute on every parameter, so these tests assert
against C++.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.math.interpolations.abcd_interpolation import (
    AbcdInterpolation,
    abcd_black_volatility,
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


def test_recovers_the_cpp_fitted_parameters(cpp: dict[str, Any]) -> None:
    """Python reproduces the C++ ``AbcdInterpolation`` fit on the probe's data.

    Tolerance: 1e-6 ABSOLUTE on the parameters, looser than LOOSE (1e-8 rel).
    Derivation: the two runs use different optimisers — C++
    ``LevenbergMarquardt`` (MINPACK ``lmdif``) via ``ProjectedCostFunction``
    against ``scipy.optimize.least_squares`` ``trf`` with
    ``xtol = ftol = gtol = 1e-12``. Near a minimum the cost is locally
    quadratic, so a parameter displacement ``dp`` perturbs the cost by
    ~``H dp^2``; the two agree on ``rms_error`` to ~4e-11 relative, admitting a
    parameter displacement of order ``sqrt(4e-11) ~ 6e-6``. The measured
    worst-case absolute gap is 1.9e-7, well inside that. The bound is stated
    ABSOLUTELY because ``a`` fits to 0.0108, where 1.9e-7 is 1.7e-5 relative.
    """
    block = cpp["abcd_interpolation"]
    interp = _make()
    for got, key in (
        (interp.a(), "a_fitted"),
        (interp.b(), "b_fitted"),
        (interp.c(), "c_fitted"),
        (interp.d(), "d_fitted"),
    ):
        tolerance.custom(
            got, float(block[key]), abs_tol=1e-6, rel_tol=0.0,
            reason="different optimisers at the same minimum; see docstring",
        )
    tolerance.custom(
        interp.rms_error(), float(block["rms_error"]), abs_tol=0.0, rel_tol=1e-9,
        reason="the minimised quantity itself",
    )
    tolerance.custom(
        interp.max_error(), float(block["max_error"]), abs_tol=0.0, rel_tol=1e-6,
        reason="max residual, sensitive to the parameter gap above",
    )


def test_pillar_recovery_at_lsq_residual_floor(cpp: dict[str, Any]) -> None:
    """``interp(t)`` at the input times reproduces C++'s ``fitted_at_pillars``.

    Tolerance 1e-6 relative: the fitted curve inherits the ~2e-7 parameter gap
    derived in :func:`test_recovers_the_cpp_fitted_parameters`.
    """
    block = cpp["abcd_interpolation"]
    times = [float(e) for e in block["times"]]
    interp = _make()
    for t, e in zip(times, block["fitted_at_pillars"], strict=True):
        tolerance.custom(
            interp(t), float(e), abs_tol=1e-9, rel_tol=1e-6,
            reason="inherits the optimiser-gap bound; see the parameter test",
        )


def test_probe_input_is_not_reproducible_by_the_model(cpp: dict[str, Any]) -> None:
    """The residual floor here is the DATA's, not the optimiser's.

    ``vols`` is the instantaneous abcd form; the model fits the average
    (Black) vol. The gap is structural, so BOTH implementations stop at the
    same ~2.6e-3 pillar residual. An earlier revision asserted the opposite —
    that C++'s residual proved it had found a worse local minimum — which was
    only tenable while the port evaluated the wrong function.
    """
    block = cpp["abcd_interpolation"]
    times = [float(e) for e in block["times"]]
    vols = [float(e) for e in block["vols"]]
    interp = _make()

    cpp_max_resid = max(
        abs(float(v_cpp) - float(v_truth))
        for v_cpp, v_truth in zip(block["fitted_at_pillars"], vols, strict=True)
    )
    py_max_resid = max(
        abs(interp(t) - float(v)) for t, v in zip(times, vols, strict=True)
    )
    assert cpp_max_resid > 1.0e-4
    # Same floor, to the optimiser-gap bound derived in the parameter test.
    tolerance.custom(
        py_max_resid, cpp_max_resid, abs_tol=1e-9, rel_tol=1e-6,
        reason="structural data/model mismatch, identical on both sides",
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
    # The data must be generated with the SAME function the model evaluates —
    # ``abcd_black_volatility``, not the instantaneous ``abcd_value`` — or the
    # residuals are not zero and this test says nothing about the fixed-param
    # short-circuit it exists to check.
    vols = np.array(
        [abcd_black_volatility(float(t), a, b, c, d) for t in times],
        dtype=np.float64,
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
