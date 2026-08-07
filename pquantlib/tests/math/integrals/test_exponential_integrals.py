"""Cross-validate Si / Ci / E1 / Ei against C++ QuantLib v1.43.

Probe source: ``migration-harness/cpp/probes/v143_math_expint/probe.cpp``
Reference:    ``migration-harness/references/v143/math/expint.json``

Covers ``ql/math/integrals/exponentialintegrals.{hpp,cpp}``.

Every value here is at the TIGHT tier (1e-14 abs / 1e-12 rel). The port is a
straight transcription of the C++ arithmetic, so the only divergence available
is last-bit rounding — chiefly ``complex / complex``, which libc++ evaluates
with the scaled-naive formula and CPython with Smith's algorithm. Across all
489 pinned values the worst observed discrepancy is 8e-15 relative, i.e. under
1% of the TIGHT budget, so no case needs a looser tier.
"""

from __future__ import annotations

import math
from typing import Any, Final

import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.math.integrals import exponential_integrals as expint
from pquantlib.math.integrals.exponential_integrals import (
    M_EULER_MASCHERONI,
    ci,
    e1,
    ei,
    si,
)
from pquantlib.testing import reference_reader, tolerance

_REFERENCE_KEY: Final[str] = "v143/math/expint"

# Loaded at import time as well as through the fixture: the parametrisation
# below takes its ids straight from the probe, so adding a case to the probe
# adds a test without touching this file.
_REFERENCE: Final[dict[str, Any]] = reference_reader.load(_REFERENCE_KEY)

# JSON has no literal for the non-finite doubles the probe genuinely produces
# (Ci(0) is -inf, Ei(0) is -inf, E1(0) is +inf), so it emits these sentinels.
_NON_FINITE: Final[dict[str, float]] = {
    "nan": math.nan,
    "inf": math.inf,
    "-inf": -math.inf,
}


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return _REFERENCE


# --- reference decoding --------------------------------------------------


def _num(value: Any) -> float:
    """Reference scalar as a float, decoding the non-finite string tokens."""
    if isinstance(value, str):
        return _NON_FINITE[value]
    return float(value)


def _cnum(value: Any) -> complex:
    """Reference complex, carried by the probe as a ``[re, im]`` pair."""
    return complex(_num(value[0]), _num(value[1]))


def _names(prefix: str) -> list[str]:
    return sorted(k for k in _REFERENCE if k.startswith(prefix))


def _tight(actual: float, expected: float, *, reason: str) -> None:
    """TIGHT, with ``+/-inf`` and NaN required to match exactly."""
    if math.isnan(expected):
        assert math.isnan(actual), f"{reason}: expected nan, got {actual!r}"
        return
    if math.isinf(expected):
        assert actual == expected, f"{reason}: expected {expected!r}, got {actual!r}"
        return
    tolerance.tight(actual, expected, reason=reason)


def _tight_complex(actual: complex, expected: complex, *, reason: str) -> None:
    _tight(actual.real, expected.real, reason=f"{reason}.re")
    _tight(actual.imag, expected.imag, reason=f"{reason}.im")


def _z(inputs: dict[str, Any]) -> complex:
    return complex(_num(inputs["re"]), _num(inputs["im"]))


# --- 0. the thresholds every branch keys on ------------------------------


def test_branch_thresholds_are_derived_not_typed(cpp: dict[str, Any]) -> None:
    """The port must arrive at C++'s constants from the same expressions.

    ``z_asym`` and the ``1.1*z_asym`` asymptotic cut-off are computed from
    ``QL_EPSILON`` in the C++ source; a port that hard-codes a rounded 41.4
    would take the wrong branch for arguments in the gap.
    """
    # pyright: ignore[reportPrivateUsage] below -- white-box probe of the
    # module-private thresholds; they mirror function-local constants in C++,
    # so they are deliberately not public API here either.
    expected = cpp["_meta"]["expected"]
    tolerance.exact(expint.QL_EPSILON, _num(expected["QL_EPSILON"]))
    tolerance.exact(expint._MAX_ERROR, _num(expected["MAX_ERROR"]))  # pyright: ignore[reportPrivateUsage]
    tolerance.exact(expint._DIST, _num(expected["DIST"]))  # pyright: ignore[reportPrivateUsage]
    tolerance.exact(expint._Z_INF, _num(expected["z_inf"]))  # pyright: ignore[reportPrivateUsage]
    tolerance.exact(expint._Z_ASYM, _num(expected["z_asym"]))  # pyright: ignore[reportPrivateUsage]
    tolerance.exact(
        1.1 * expint._Z_ASYM,  # pyright: ignore[reportPrivateUsage]
        _num(expected["asymptotic_threshold"]),
    )
    tolerance.exact(M_EULER_MASCHERONI, _num(expected["euler_mascheroni"]))


# --- 1. Si(Real) ---------------------------------------------------------


@pytest.mark.parametrize("name", _names("si_real_"))
def test_si_real(cpp: dict[str, Any], name: str) -> None:
    """``Si(Real)``: the ``x < 0`` mirror, the ``x <= 4`` rational series and
    the ``x > 4`` ``pi/2 - f cos - g sin`` asymptotic form.

    The reference straddles ``x == 4.0`` with the two adjacent doubles, so a
    port that writes ``<`` for ``<=`` fails on ``si_real_at4``.
    """
    case = cpp[name]
    x = _num(case["inputs"]["x"])
    _tight(si(x), _num(case["expected"]["si"]), reason=name)


# --- 2. Ci(Real) ---------------------------------------------------------


@pytest.mark.parametrize("name", _names("ci_real_"))
def test_ci_real(cpp: dict[str, Any], name: str) -> None:
    """``Ci(Real)``, including its ``QL_REQUIRE(x >= 0)`` guard.

    ``x == 0`` is the log pole and must return ``-inf`` rather than raising —
    C++ ``std::log(0.0)`` is ``-inf`` where Python's ``math.log`` raises, so
    this is a place the port has to diverge textually to converge numerically.
    """
    case = cpp[name]
    x = _num(case["inputs"]["x"])
    expected = case["expected"]["ci"]

    if expected == "raises":
        with pytest.raises(LibraryException):
            ci(x)
        return

    _tight(ci(x), _num(expected), reason=name)


# --- 3. Si(complex) ------------------------------------------------------


@pytest.mark.parametrize("name", _names("si_cplx_"))
def test_si_complex(cpp: dict[str, Any], name: str) -> None:
    """``Si(complex)``: the ``|z| <= 0.2`` Taylor series and the ``E1`` form.

    The recorded ``plus_pi`` flag is re-derived here rather than trusted,
    because the branch-cut sign is the one piece of this function that is not
    continuous: ``(0, -y)`` takes ``-pi`` while ``(x, -y)`` with ``x > 0``
    takes ``+pi``, and both are in the reference.
    """
    case = cpp[name]
    inputs = case["inputs"]
    z = _z(inputs)

    if "plus_pi" in inputs:
        plus_pi = (z.real >= 0 and z.imag >= 0) or (z.real > 0 and z.imag < 0)
        assert plus_pi is bool(inputs["plus_pi"]), f"{name}: branch-cut sign disagrees with C++"

    _tight_complex(si(z), _cnum(case["expected"]["si"]), reason=name)


# --- 4. Ci(complex) ------------------------------------------------------


@pytest.mark.parametrize("name", _names("ci_cplx_"))
def test_ci_complex(cpp: dict[str, Any], name: str) -> None:
    """``Ci(complex)``, including the ``+/- i*pi`` accumulator selection.

    Same asymmetry as ``Si``: ``(0, +y)`` takes no accumulator while
    ``(0, -y)`` takes ``-i*pi``.
    """
    case = cpp[name]
    inputs = case["inputs"]
    z = _z(inputs)

    if z.real < 0.0 and z.imag >= 0.0:
        acc = "plus_i_pi"
    elif z.real <= 0.0 and z.imag <= 0.0:
        acc = "minus_i_pi"
    else:
        acc = "zero"
    assert acc == inputs["acc"], f"{name}: accumulator branch disagrees with C++"

    _tight_complex(ci(z), _cnum(case["expected"]["ci"]), reason=name)


# --- 5. E1 / Ei ----------------------------------------------------------


@pytest.mark.parametrize("name", _names("expint_"))
def test_e1_and_ei(cpp: dict[str, Any], name: str) -> None:
    """``E1`` and ``Ei`` directly, which is where all three ``Ei`` algorithms
    (asymptotic series, 47-level continued fraction, power series) and the
    ``E1`` branch-cut offset live.

    Pinned on their own rather than only through ``Si``/``Ci`` because the two
    callers cancel some of the accumulator away.
    """
    case = cpp[name]
    z = _z(case["inputs"])
    expected = case["expected"]

    _tight_complex(e1(z), _cnum(expected["e1"]), reason=f"{name}.e1")
    _tight_complex(ei(z), _cnum(expected["ei"]), reason=f"{name}.ei")


# --- 6. the production caller's arguments --------------------------------


@pytest.mark.parametrize("name", _names("heston_"))
def test_heston_control_variate_arguments(cpp: dict[str, Any], name: str) -> None:
    """The exact ``z`` that ``AP_Helper::controlVariateValue`` feeds in.

    ``AnalyticHestonEngine`` with ``ComplexLogFormula.AsymptoticChF`` calls
    ``ci(-0.5*phi_freq)`` and ``si(0.5*phi_freq)``. The five parameter sets
    span ``|z|`` from 0.08 to 91, which reaches the Taylor branch, the ``Ei``
    power series, the continued fraction and the asymptotic series in turn.
    """
    case = cpp[name]
    inputs = case["inputs"]
    expected = case["expected"]

    _tight_complex(si(_cnum(inputs["z_si"])), _cnum(expected["si"]), reason=f"{name}.si")
    _tight_complex(ci(_cnum(inputs["z_ci"])), _cnum(expected["ci"]), reason=f"{name}.ci")


# --- 7. closed-form identities -------------------------------------------


def test_identities_against_cpp(cpp: dict[str, Any]) -> None:
    """Facts that hold independently of the pinned tables.

    A probe that emitted self-consistent garbage would still have to satisfy
    these, so they are asserted for the port *and* checked against the values
    the probe recorded for the same expressions.
    """
    expected = cpp["_identities"]["expected"]

    # Si is odd and vanishes at the origin — exactly, since B1 is a literal
    # negation and the series starts at x.
    tolerance.exact(si(0.0), 0.0)
    tolerance.exact(si(0.0), _num(expected["si_0"]))
    tolerance.exact(si(-3.0) + si(3.0), 0.0)
    tolerance.exact(si(-9.0) + si(9.0), 0.0)
    tolerance.exact(si(-3.0) + si(3.0), _num(expected["si_symmetry_residual_3"]))
    tolerance.exact(si(-9.0) + si(9.0), _num(expected["si_symmetry_residual_9"]))

    # Si(x) -> pi/2. The O(1/x) remainder is -cos(x)/x, ~9.4e-7 at x = 1e6.
    half_pi = _num(expected["half_pi"])
    _tight(si(1e6) - half_pi, _num(expected["si_1e6_minus_half_pi"]), reason="si_1e6")
    assert abs(si(1e6) - half_pi) < 1e-6

    # Ci has a log singularity at 0 and its first positive zero at 0.61650549.
    assert ci(0.0) == -math.inf
    tolerance.exact(ci(0.0), _num(expected["ci_0"]))
    assert abs(ci(0.6165054856207162)) < 1e-15
    _tight(ci(0.6165054856207162), _num(expected["ci_first_zero"]), reason="ci_first_zero")

    # E1(x) == -Ei(-x) on the positive real axis.
    _tight_complex(e1(complex(2.0, 0.0)), _cnum(expected["e1_2"]), reason="e1_2")
    _tight_complex(ei(complex(-2.0, 0.0)), _cnum(expected["ei_minus_2"]), reason="ei_minus_2")
    _tight_complex(e1(complex(2.0, 0.0)), -ei(complex(-2.0, 0.0)), reason="e1_vs_ei")


@pytest.mark.parametrize("x", [3.0, 9.0])
def test_real_and_complex_overloads_agree_on_the_positive_axis(cpp: dict[str, Any], x: float) -> None:
    """``Si(x)`` and ``Si(x + 0i)`` are different algorithms.

    For ``x > 0.2`` the complex overload routes through ``E1``/``Ei`` while the
    real one uses the Pade/asymptotic pair, so agreement here is an independent
    check on both. They agree to a few ULP, which TIGHT's 1e-14 absolute floor
    covers; the imaginary part must be exactly zero (Ei's ``imag() == 0`` tail
    substitutes ``acc.imag()``).
    """
    expected = cpp["_identities"]["expected"]
    key = "3" if x == 3.0 else "9"

    si_c = si(complex(x, 0.0))
    ci_c = ci(complex(x, 0.0))

    _tight_complex(si_c, _cnum(expected[f"si_cplx_on_axis_{key}"]), reason=f"si_cplx_{key}")
    _tight_complex(ci_c, _cnum(expected[f"ci_cplx_on_axis_{key}"]), reason=f"ci_cplx_{key}")
    _tight(si(x), _num(expected[f"si_real_{key}"]), reason=f"si_real_{key}")
    _tight(ci(x), _num(expected[f"ci_real_{key}"]), reason=f"ci_real_{key}")

    tolerance.exact(si_c.imag, 0.0)
    tolerance.exact(ci_c.imag, 0.0)
    _tight(si_c.real, si(x), reason=f"si overloads at {x}")
    _tight(ci_c.real, ci(x), reason=f"ci overloads at {x}")


# --- 8. dispatch + coverage guards ---------------------------------------


def test_dispatch_follows_the_argument_type() -> None:
    """C++ overloads on ``Real`` vs ``std::complex<Real>``; Python dispatches
    on the runtime type, so the return type has to follow the argument.
    """
    assert isinstance(si(1.0), float)
    assert isinstance(ci(1.0), float)
    assert isinstance(si(complex(1.0, 0.0)), complex)
    assert isinstance(ci(complex(1.0, 0.0)), complex)

    # Ci(-x) is only defined on the complex overload.
    with pytest.raises(LibraryException):
        ci(-1.0)
    assert isinstance(ci(complex(-1.0, 0.0)), complex)


def test_reference_straddles_every_branch_threshold() -> None:
    """Guard against the reference silently losing a branch.

    Each of the four thresholds in this file (``x == 4`` for the real pair,
    ``|z| <= 0.2`` for Si, ``|z| > 4.5`` for Ei's continued fraction and
    ``|z| > 1.1*z_asym`` for its asymptotic series) must have pinned cases on
    both sides, otherwise the suite is green for the wrong reason.
    """

    def moduli(prefix: str) -> list[float]:
        return [
            _num(_REFERENCE[k]["inputs"]["abs"]) for k in _names(prefix) if "abs" in _REFERENCE[k]["inputs"]
        ]

    real_xs = [_num(_REFERENCE[k]["inputs"]["x"]) for k in _names("si_real_")]
    assert any(0 <= x <= 4.0 for x in real_xs)
    assert any(x > 4.0 for x in real_xs)
    assert any(x < 0 for x in real_xs)
    assert 4.0 in real_xs

    ci_xs = [_num(_REFERENCE[k]["inputs"]["x"]) for k in _names("ci_real_")]
    assert any(x < 0 for x in ci_xs)
    assert 0.0 in ci_xs

    si_r = moduli("si_cplx_")
    assert any(r <= 0.2 for r in si_r)
    assert any(r > 0.2 for r in si_r)

    dist: float = expint._DIST  # pyright: ignore[reportPrivateUsage]
    asymptotic: float = 1.1 * expint._Z_ASYM  # pyright: ignore[reportPrivateUsage]

    ei_r = moduli("expint_") + moduli("si_cplx_") + moduli("ci_cplx_")
    assert any(r <= dist for r in ei_r)
    assert any(dist < r <= asymptotic for r in ei_r)
    assert any(r > asymptotic for r in ei_r)
