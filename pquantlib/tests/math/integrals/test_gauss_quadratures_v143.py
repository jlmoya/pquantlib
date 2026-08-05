"""Cross-validate the Gauss quadrature family against C++ QuantLib v1.43.

Probe source: ``migration-harness/cpp/probes/v143_math_integrals/probe.cpp``
Reference:    ``migration-harness/references/v143/math/integrals.json``

Covers ``gaussianorthogonalpolynomial.{hpp,cpp}``, ``gaussianquadratures.{hpp,cpp}``
and ``gausslaguerrecosinepolynomial.hpp``.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from fractions import Fraction
from typing import Any

import numpy as np
import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.math.array import Array
from pquantlib.math.integrals.gauss_laguerre_cosine_polynomial import (
    GaussLaguerreCosinePolynomial,
    GaussLaguerreSinePolynomial,
    GaussLaguerreTrigonometricBase,
)
from pquantlib.math.integrals.gaussian_orthogonal_polynomial import (
    GaussChebyshev2ndPolynomial,
    GaussChebyshevPolynomial,
    GaussGegenbauerPolynomial,
    GaussHermitePolynomial,
    GaussHyperbolicPolynomial,
    GaussianOrthogonalPolynomial,
    GaussJacobiPolynomial,
    GaussLaguerrePolynomial,
    GaussLegendrePolynomial,
)
from pquantlib.math.integrals.gaussian_quadrature import (
    GaussChebyshev2ndIntegration,
    GaussChebyshev2ndIntegrator,
    GaussChebyshevIntegration,
    GaussChebyshevIntegrator,
    GaussGegenbauerIntegration,
    GaussHermiteIntegration,
    GaussHyperbolicIntegration,
    GaussianQuadrature,
    GaussianQuadratureIntegrator,
    GaussJacobiIntegration,
    GaussLaguerreIntegration,
    GaussLegendreIntegration,
    GaussLegendreIntegrator,
)
from pquantlib.testing import reference_reader, tolerance

_EPS = 2.220446049250313e-16


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("v143/math/integrals")


# --- the shared integrand table, mirroring the probe's ------------------


def _sqrt_abs(x: float) -> float:
    return math.sqrt(abs(x))


INTEGRANDS: dict[str, Callable[[float], float]] = {
    "one": lambda x: 1.0,
    "x": lambda x: x,
    "x2": lambda x: x * x,
    "x4": lambda x: x * x * x * x,
    "inv_1px2": lambda x: 1.0 / (1.0 + x * x),
    "sin_x": math.sin,
    "abs_x": abs,
    "sqrt_abs_x": _sqrt_abs,
}


# --- 1. orthogonal polynomials -----------------------------------------

POLYNOMIALS: dict[str, Callable[[], GaussianOrthogonalPolynomial]] = {
    "laguerre_s0": lambda: GaussLaguerrePolynomial(0.0),
    "laguerre_s1_5": lambda: GaussLaguerrePolynomial(1.5),
    "hermite_mu0": lambda: GaussHermitePolynomial(0.0),
    "hermite_mu0_3": lambda: GaussHermitePolynomial(0.3),
    "jacobi_0_5_1_5": lambda: GaussJacobiPolynomial(0.5, 1.5),
    "jacobi_m0_25_0_75": lambda: GaussJacobiPolynomial(-0.25, 0.75),
    "legendre": GaussLegendrePolynomial,
    "chebyshev": GaussChebyshevPolynomial,
    "chebyshev2nd": GaussChebyshev2ndPolynomial,
    "gegenbauer_1_5": lambda: GaussGegenbauerPolynomial(1.5),
    "gegenbauer_0_5": lambda: GaussGegenbauerPolynomial(0.5),
    "hyperbolic": GaussHyperbolicPolynomial,
}


def test_polynomial_names_cover_the_probe(cpp: dict[str, Any]) -> None:
    assert {e["name"] for e in cpp["orthogonal_polynomials"]} == set(POLYNOMIALS)


@pytest.mark.parametrize("case", range(12))
def test_orthogonal_polynomial(cpp: dict[str, Any], case: int) -> None:
    """mu_0, alpha, beta, w, value and weightedValue at TIGHT.

    Every one of these is a closed-form expression in the polynomial's
    parameters plus ``GammaFunction::logValue``; there is no iteration and no
    cancellation beyond a handful of operations, so the only difference
    available is the last-bit rounding of ``pow``/``exp``.
    """
    block = cpp["orthogonal_polynomials"][case]
    poly = POLYNOMIALS[block["name"]]()
    indices = [int(v) for v in block["indices"]]
    xs = [float(v) for v in block["xs"]]

    tolerance.tight(poly.mu_0(), float(block["mu_0"]))

    for i, expected in zip(indices, block["alpha"], strict=True):
        if expected is None:
            # GaussJacobiPolynomial::alpha raises when both the numerator and
            # the l'Hospital-corrected denominator vanish.
            with pytest.raises(LibraryException):
                poly.alpha(i)
        else:
            tolerance.tight(poly.alpha(i), float(expected))

    for i, expected in zip(indices, block["beta"], strict=True):
        if expected is None:
            with pytest.raises(LibraryException):
                poly.beta(i)
        else:
            tolerance.tight(poly.beta(i), float(expected))

    for x, expected in zip(xs, block["w"], strict=True):
        tolerance.tight(poly.w(x), float(expected))

    base = poly
    assert hasattr(base, "value")
    for n, row in zip(indices, block["value"], strict=True):
        for x, expected in zip(xs, row, strict=True):
            tolerance.tight(base.value(n, x), float(expected))  # type: ignore[attr-defined]
    for n, row in zip(indices, block["weighted_value"], strict=True):
        for x, expected in zip(xs, row, strict=True):
            tolerance.tight(base.weighted_value(n, x), float(expected))  # type: ignore[attr-defined]


# --- 2. the quadrature rules -------------------------------------------

QUADRATURES: dict[str, Callable[[], GaussianQuadrature]] = {
    "laguerre_8": lambda: GaussLaguerreIntegration(8),
    "laguerre_16_s1_5": lambda: GaussLaguerreIntegration(16, 1.5),
    "hermite_8": lambda: GaussHermiteIntegration(8),
    "hermite_6_mu0_3": lambda: GaussHermiteIntegration(6, 0.3),
    "hermite_5": lambda: GaussHermiteIntegration(5),
    "jacobi_8": lambda: GaussJacobiIntegration(8, 0.5, 1.5),
    "legendre_8": lambda: GaussLegendreIntegration(8),
    "legendre_16": lambda: GaussLegendreIntegration(16),
    "chebyshev_8": lambda: GaussChebyshevIntegration(8),
    "chebyshev2nd_8": lambda: GaussChebyshev2ndIntegration(8),
    "gegenbauer_8": lambda: GaussGegenbauerIntegration(8, 1.5),
    "hyperbolic_8": lambda: GaussHyperbolicIntegration(8),
    "legendre_3": lambda: GaussLegendreIntegration(3),
}


def test_quadrature_names_cover_the_probe(cpp: dict[str, Any]) -> None:
    assert {e["name"] for e in cpp["quadratures"]} == set(QUADRATURES)


@pytest.mark.parametrize("case", range(13))
def test_quadrature_nodes_and_weights(cpp: dict[str, Any], case: int) -> None:
    """Nodes and weights element-by-element, *in C++ order*.

    This is the assertion that catches an eigen-solver delegation that hands
    the spectrum back ascending: C++ ``TqrEigenDecomposition`` sorts
    descending, LAPACK ascending, and ``x()`` / ``weights()`` are public API.

    TIGHT: both sides solve the same symmetric tridiagonal eigenproblem with a
    backward-stable method, so the eigenvalues agree to ``O(eps * ||T||)`` and
    the weights — built from the squared first eigenvector component — to
    ``O(eps * ||T|| / gap)``. Both are far inside 1e-12 relative for these
    orders.
    """
    block = cpp["quadratures"][case]
    quad = QUADRATURES[block["name"]]()

    assert quad.order() == int(block["order"])
    assert len(quad.x()) == len(block["x"])

    for got, expected in zip(quad.x(), block["x"], strict=True):
        tolerance.tight(float(got), float(expected))
    for got, expected in zip(quad.weights(), block["weights"], strict=True):
        tolerance.tight(float(got), float(expected))


def _integral_bound(block: dict[str, Any], name: str, l1: float) -> float:
    """Absolute bound on the quadrature-sum difference for integrand ``name``.

    Two terms, both derived rather than fitted:

    * ``1e-12 * l1`` — the sums differ because the nodes and weights differ.
      Nodes and weights are separately asserted to agree within 1e-12
      *relative*, so the sums agree within 1e-12 of the L1 norm of the sum,
      ``sum_i |w_i f(x_i)|``. For an odd integrand on a symmetric rule the
      result cancels from O(1) terms to ~1e-14 and this is the only bound with
      any content; comparing relative to the *result* there is meaningless.
    * for ``sqrt_abs_x`` only, an extra term at any node that sits at a
      computed zero. ``sqrt|x|`` is Holder-1/2, not Lipschitz, at the origin:
      ``|sqrt|x1| - sqrt|x2|| <= sqrt(|x1-x2|)``. The node agreement there is
      absolute (1e-14), not relative, so the integrand values can differ by
      ``sqrt(1e-14) = 1e-7`` even though the nodes agree to the last bit that
      is meaningful.
    """
    bound = 1e-12 * l1 + 1e-14
    if name == "sqrt_abs_x":
        for x, w in zip(block["x"], block["weights"], strict=True):
            if abs(float(x)) < 1e-6:
                bound += abs(float(w)) * math.sqrt(1e-14 + 1e-12 * abs(float(x)))
    return bound


@pytest.mark.parametrize("case", range(13))
def test_quadrature_integrals(cpp: dict[str, Any], case: int) -> None:
    """Integrals of eight integrands, including two the rules cannot do well.

    ``abs_x`` and ``sqrt_abs_x`` are not polynomials, so a Gauss rule gets them
    badly wrong — which is the point: they pin *the rule*, not the answer. A
    port that quietly swapped in a different quadrature would still get
    ``x2`` right and would miss here.
    """
    block = cpp["quadratures"][case]
    quad = QUADRATURES[block["name"]]()
    names = list(block["integrand_names"])

    for name, expected, l1 in zip(names, block["integrals"], block["integrals_l1"], strict=True):
        got = quad(INTEGRANDS[name])
        tolerance.custom(
            got,
            float(expected),
            abs_tol=_integral_bound(block, name, float(l1)),
            rel_tol=1e-12,
            reason=(
                "node/weight agreement propagated through the quadrature sum; "
                "see _integral_bound"
            ),
        )


# --- 3. MultiDimGaussianIntegration ------------------------------------


def _sum_sq(v: Array) -> float:
    return float(np.sum(v * v))


def _exp_sum(v: Array) -> float:
    return math.exp(float(np.sum(v)))


@pytest.mark.parametrize("case", range(3))
def test_multi_dim_gaussian_integration(cpp: dict[str, Any], case: int) -> None:
    """Tensor-product weights, the full node table and two sums, at TIGHT.

    The node table is where the ``spacing``/``partial_sum`` index algebra
    lives; a transposed or mis-strided tensor product still produces the right
    *set* of nodes and the right sum for a symmetric integrand, so the table
    itself is pinned entry by entry.
    """
    block = cpp["multi_dim"][case]
    from pquantlib.math.integrals.gaussian_quadrature import (  # noqa: PLC0415
        MultiDimGaussianIntegration,
    )

    mdi = MultiDimGaussianIntegration([int(n) for n in block["ns"]], GaussLegendreIntegration)

    for got, expected in zip(mdi.weights(), block["weights"], strict=True):
        tolerance.tight(float(got), float(expected))
    for got_row, expected_row in zip(mdi.x(), block["x"], strict=True):
        for got, expected in zip(got_row, expected_row, strict=True):
            tolerance.tight(float(got), float(expected))

    tolerance.tight(mdi(_sum_sq), float(block["integral_sum_sq"]))
    tolerance.tight(mdi(_exp_sum), float(block["integral_exp_prod"]))


# --- 4. GaussianQuadratureIntegrator -----------------------------------

QUADRATURE_INTEGRATORS: dict[str, Callable[[int], GaussianQuadratureIntegrator]] = {
    "GaussLegendreIntegrator": GaussLegendreIntegrator,
    "GaussChebyshevIntegrator": GaussChebyshevIntegrator,
    "GaussChebyshev2ndIntegrator": GaussChebyshev2ndIntegrator,
}


@pytest.mark.parametrize("case", range(8))
def test_gaussian_quadrature_integrator(cpp: dict[str, Any], case: int) -> None:
    """The affine remap of a fixed rule onto ``[a, b]``, at TIGHT.

    ``absolute_accuracy`` is also pinned: the C++ constructor passes
    ``Null<Real>()`` (``numeric_limits<float>::max()``), which is only *just*
    above ``QL_EPSILON`` in the base-class precondition and would be easy to
    replace with something else that happens to pass.
    """
    block = cpp["quadrature_integrators"][case]
    integrator = QUADRATURE_INTEGRATORS[block["kind"]](int(block["n"]))

    got = integrator(INTEGRANDS[block["integrand"]], float(block["a"]), float(block["b"]))
    tolerance.tight(got, float(block["value"]))
    assert integrator.max_evaluations() == int(block["max_evaluations"])
    tolerance.exact(integrator.absolute_accuracy(), float(block["absolute_accuracy"]))
    assert integrator.get_integration().order() == int(block["n"])


# --- 5. Laguerre trigonometric polynomials -----------------------------


class _ExactTrigonometricRecursion:
    """The Laguerre-trigonometric recursion evaluated over ``Fraction``.

    ``u`` is exactly representable for every probed case, so every quantity in
    the moment recursion and in the Hankel-determinant recursion below it is a
    rational number. Evaluating them exactly gives the *true* value of the
    algorithm C++ and Python both run in double precision, which is what makes
    the accuracy claims in :func:`test_laguerre_trigonometric_recurrence`
    statements about accuracy rather than about agreement.
    """

    def __init__(self, u: float, *, cosine: bool) -> None:
        self.u = Fraction(u)
        self.cosine = cosine
        self.norm = (
            1 + 1 / (1 + self.u**2) if cosine else 1 + self.u / (1 + self.u**2)
        )
        self._m: dict[int, Fraction] = {}
        self._f: dict[int, Fraction] = {}
        self._b: dict[int, Fraction] = {}
        self._c: dict[int, Fraction] = {}
        self._z_tab: dict[tuple[int, int], Fraction] = {}

    def _m0(self) -> Fraction:
        return 1 / (1 + self.u**2) if self.cosine else self.u / (1 + self.u**2)

    def _m1(self) -> Fraction:
        if self.cosine:
            return (1 - self.u**2) / (1 + self.u**2) ** 2
        return 2 * self.u / (1 + self.u**2) ** 2

    def _raw(self, n: int) -> Fraction:
        if n not in self._m:
            if n == 0:
                self._m[n] = self._m0()
            elif n == 1:
                self._m[n] = self._m1()
            else:
                self._m[n] = (
                    2 * n * self._raw(n - 1) - n * (n - 1) * self._raw(n - 2)
                ) / (1 + self.u**2)
        return self._m[n]

    def _fact(self, n: int) -> Fraction:
        if n not in self._f:
            self._f[n] = Fraction(1) if n == 0 else n * self._fact(n - 1)
        return self._f[n]

    def moment(self, i: int) -> Fraction:
        return (self._raw(i) + self._fact(i)) / self.norm

    def _z(self, k: int, i: int) -> Fraction:
        if k == -1:
            return Fraction(0)
        if (k, i) not in self._z_tab:
            if k == 0:
                self._z_tab[(k, i)] = self.moment(i)
            else:
                self._z_tab[(k, i)] = (
                    self._z(k - 1, i + 1)
                    - self.alpha(k - 1) * self._z(k - 1, i)
                    - self.beta(k - 1) * self._z(k - 2, i)
                )
        return self._z_tab[(k, i)]

    def alpha(self, u: int) -> Fraction:
        if u not in self._b:
            if u == 0:
                self._b[u] = self.moment(1)
            else:
                self._b[u] = -self._z(u - 1, u) / self._z(u - 1, u - 1) + self._z(
                    u, u + 1
                ) / self._z(u, u)
        return self._b[u]

    def beta(self, u: int) -> Fraction:
        if u == 0:
            return Fraction(1)
        if u not in self._c:
            self._c[u] = self._z(u, u) / self._z(u - 1, u - 1)
        return self._c[u]


LAGUERRE_TRIG: dict[str, Callable[[float], GaussLaguerreTrigonometricBase]] = {
    "GaussLaguerreCosinePolynomial": GaussLaguerreCosinePolynomial,
    "GaussLaguerreSinePolynomial": GaussLaguerreSinePolynomial,
}

# Forward error the double-precision Hankel-determinant recursion carries by
# index 7. Measured against exact rational arithmetic in
# test_laguerre_trigonometric_recurrence: the C++ result for alpha_7 of the
# u = 2 cosine polynomial is 1.2e-10 relative from the true value, and the
# Python result is the same order. This is not a cross-validation slack — both
# implementations are asserted against the *exact* value at this bound — and it
# is the reason the C++ class is a template over mp_real.
_ILL_CONDITIONED_REL = 1e-9
# Below this the recursion is still well conditioned at double precision and
# the exact value must be reproduced to TIGHT.
_WELL_CONDITIONED_REL = 1e-13


def _n8(name: str) -> bool:
    return name.endswith("_n8")


@pytest.mark.parametrize("case", range(9))
def test_laguerre_trigonometric_moments(cpp: dict[str, Any], case: int) -> None:
    """Moments, mu_0 and the weight function, at TIGHT.

    The moment recursion itself is benign — a two-term linear recursion with
    growing, same-signed terms — so it reproduces to the last bits.
    """
    block = cpp["laguerre_trigonometric"][case]
    poly = LAGUERRE_TRIG[block["kind"]](float(block["u"]))

    for i, expected in enumerate(block["moments"]):
        tolerance.tight(poly.moment(i), float(expected))
    tolerance.tight(poly.mu_0(), float(block["mu_0"]))
    for x, expected in zip(block["xs"], block["w"], strict=True):
        tolerance.tight(poly.w(float(x)), float(expected))


@pytest.mark.parametrize("case", range(9))
def test_laguerre_trigonometric_recurrence(cpp: dict[str, Any], case: int) -> None:
    """alpha / beta asserted against **exact rational** values.

    Both implementations run the identical double-precision recursion, so in
    principle they should agree bit for bit — and they do once the C++ is built
    with ``-ffp-contract=off``. At ``-O2`` Apple clang contracts
    ``z0 - a*z1 - b*z2`` into two FMAs; CPython never contracts. That single
    saved rounding per step is then amplified by the Hankel-determinant
    recursion, whose condition number grows with the index.

    So agreement between the two doubles is not the interesting property;
    accuracy is. Each index is compared against the exact rational value of the
    same recursion, at TIGHT wherever the C++ result shows the recursion is
    still well conditioned there, and at the measured forward error of the
    algorithm (1e-9 relative) where it is not.
    """
    block = cpp["laguerre_trigonometric"][case]
    poly = LAGUERRE_TRIG[block["kind"]](float(block["u"]))
    exact = _ExactTrigonometricRecursion(
        float(block["u"]), cosine=block["kind"].endswith("CosinePolynomial")
    )

    for label, expecteds, exact_fn, py_fn in (
        ("alpha", block["alpha"], exact.alpha, poly.alpha),
        ("beta", block["beta"], exact.beta, poly.beta),
    ):
        for i, cpp_value in enumerate(expecteds):
            truth = float(exact_fn(i))
            cpp_error = abs(float(cpp_value) - truth)
            well_conditioned = cpp_error <= _WELL_CONDITIONED_REL * abs(truth)
            if well_conditioned:
                tolerance.tight(py_fn(i), truth)
            else:
                tolerance.custom(
                    py_fn(i),
                    truth,
                    abs_tol=_EPS,
                    rel_tol=_ILL_CONDITIONED_REL,
                    reason=(
                        f"{label}[{i}]: the Hankel-determinant recursion is "
                        "ill conditioned here; the bound is the forward error "
                        "the double-precision algorithm itself carries, and it "
                        "is asserted against exact rational arithmetic, not "
                        "against C++"
                    ),
                )


@pytest.mark.parametrize("case", range(9))
def test_laguerre_trigonometric_quadrature(cpp: dict[str, Any], case: int) -> None:
    """Nodes, weights and two integrals of the moment-driven Gauss rule.

    The ``n = 5`` and ``n = 6`` rules only reach recurrence indices where the
    recursion is well conditioned, so they reproduce C++ at TIGHT and pin the
    whole pipeline (moments -> recurrence -> eigenproblem -> weights).

    The ``n = 8`` rules feed the eigensolver coefficients that already carry
    the ~1e-10 forward error established above, so the bound is that error
    propagated: Weyl gives ``|dlambda| <= ||dT||_2`` for the nodes, and the
    weights pick up a further ``1/gap`` from the eigenvector perturbation.
    ``d_T`` is computed here from the *pinned C++* coefficients, so the bound
    is data, not a guess.
    """
    block = cpp["laguerre_trigonometric"][case]
    poly = LAGUERRE_TRIG[block["kind"]](float(block["u"]))
    n = int(block["n"])
    quad = GaussianQuadrature(n, poly)

    if not _n8(block["name"]):
        for got, expected in zip(quad.x(), block["nodes"], strict=True):
            tolerance.tight(float(got), float(expected))
        for got, expected in zip(quad.weights(), block["quad_weights"], strict=True):
            tolerance.tight(float(got), float(expected))
        tolerance.tight(quad(INTEGRANDS["one"]), float(block["integral_one"]))
        tolerance.tight(quad(INTEGRANDS["inv_1px2"]), float(block["integral_inv_1px2"]))
        return

    # ||T_py - T_cpp||_2 <= max_i (|dd_i| + |de_{i-1}| + |de_i|), with
    # d_i = alpha(i) and e_i = sqrt(beta(i+1)).
    d_alpha = [abs(poly.alpha(i) - float(block["alpha"][i])) for i in range(n)]
    d_e = [
        abs(math.sqrt(poly.beta(i)) - math.sqrt(float(block["beta"][i])))
        for i in range(1, n)
    ]
    d_t = max(d_alpha) + 2 * max(d_e)

    for got, expected in zip(quad.x(), block["nodes"], strict=True):
        tolerance.custom(
            float(got),
            float(expected),
            abs_tol=d_t + 1e-14,
            rel_tol=1e-12,
            reason=(
                "Weyl: an eigenvalue of a symmetric matrix moves by at most "
                "the spectral norm of the perturbation, and the perturbation "
                "here is the pinned difference in the Jacobi coefficients"
            ),
        )

    # The eigenvector — hence the weight, which is its squared first component
    # — picks up an extra 1/gap. The smallest gap among these nodes is O(0.5)
    # and the smallest first component is O(1e-2), so the amplification over
    # d_T is at most ~1e2.
    weight_bound = 1e2 * d_t + 1e-14
    for got, expected in zip(quad.weights(), block["quad_weights"], strict=True):
        tolerance.custom(
            float(got),
            float(expected),
            abs_tol=weight_bound,
            rel_tol=1e-8,
            reason="eigenvector perturbation 1/gap amplification of d_T",
        )
    for key, integrand in (
        ("integral_one", "one"),
        ("integral_inv_1px2", "inv_1px2"),
    ):
        tolerance.custom(
            quad(INTEGRANDS[integrand]),
            float(block[key]),
            abs_tol=weight_bound * n,
            rel_tol=1e-8,
            reason="node and weight perturbation summed over the n terms",
        )
