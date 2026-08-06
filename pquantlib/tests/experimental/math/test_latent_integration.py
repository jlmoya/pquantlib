"""Cross-validate the latent-model integration and sampling machinery.

Probe source: migration-harness/cpp/probes/v143_experimental_latent/probe.cpp
Reference:    migration-harness/references/v143/experimental/latent.json

Covers the five v1.43 ``experimental`` classes ported in this wave:
``VectorIntegrator``, ``IntegrationBase`` (both full specialisations),
``IntegrationFactory``, ``FactorSampler`` (generic plus both partial
specialisations) and the ``Ziggurat`` RNG traits.

.. rubric:: Why some assertions carry a derived bound rather than TIGHT

``IntegrationFactory`` asks for a **25-point** Gauss-Hermite rule. At that
order the outermost weight is built from an eigenvector component of size
~4e-9, so Golub-Welsch in double precision resolves it to about eight
significant digits and its exact value depends on the last-bit details of the
eigen-solver (C++ ``TqrEigenDecomposition``, here LAPACK via
``scipy.linalg.eigh_tridiagonal``). :func:`_golub_welsch_bounds` turns that
into a per-node/per-weight bound from first-order perturbation theory, and
:func:`_quad_sum_bound` propagates it through a quadrature sum. Everything the
class is actually *used* for — a probability density times a payoff, which is
~1e-9 at the extreme node — is unaffected and is asserted at TIGHT.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Callable, Sequence
from typing import Any, cast

import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.experimental.math.gaussian_copula_policy import GaussianCopulaPolicy
from pquantlib.experimental.math.latent_model import (
    GaussianQuadIntegrationBase,
    IntegrationBase,
    LatentModel,
    LatentModelIntegrationType,
    LMIntegration,
    MultidimIntegralIntegrationBase,
)
from pquantlib.experimental.math.multidim_quadrature import GaussianQuadMultidimIntegrator
from pquantlib.experimental.math.polar_student_t_rng import PolarStudentTRng
from pquantlib.experimental.math.t_copula_policy import TCopulaPolicy
from pquantlib.experimental.math.ziggurat_rng import Ziggurat, ZigguratRng
from pquantlib.math.integrals.gaussian_orthogonal_polynomial import GaussHermitePolynomial
from pquantlib.math.integrals.gaussian_quadrature import GaussHermiteIntegration
from pquantlib.math.integrals.trapezoid import TrapezoidIntegral
from pquantlib.math.randomnumbers.mersenne_twister import MersenneTwisterUniformRng
from pquantlib.math.randomnumbers.random_sequence_generator import RandomSequenceGenerator
from pquantlib.testing import reference_reader, tolerance

_EPS = 2.220446049250313e-16

_W1 = [[0.5], [0.4], [0.3]]
_W2 = [[0.5, 0.3], [0.4, 0.2], [0.3, 0.1]]
_WT = [[0.5], [0.4]]
_T_ORDERS = [3, 5]


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("v143/experimental/latent")


# --- derived bounds -----------------------------------------------------


def _golub_welsch_bounds(order: int, mu: float) -> tuple[float, list[float]]:
    """First-order perturbation bounds on one Gauss-Hermite rule.

    Returns ``(node_abs_bound, weight_abs_bound[i])``.

    Both sides run Golub-Welsch on the *same* symmetric tridiagonal Jacobi
    matrix ``T`` with a backward-stable method, so each side returns the exact
    spectrum of a matrix within ``c * eps * ||T||`` of ``T``. Wilkinson's
    constant for a tridiagonal QR sweep is O(n), so ``c = order`` is used:

    * eigenvalues (nodes) move by at most ``order * eps * ||T||`` *absolutely*;
    * the weight is ``w_i = mu_0 * v_i^2 / w(x_i)`` where ``v_i`` is the first
      component of the i-th normalised eigenvector, so a perturbation of
      ``order * eps * ||T||`` in ``v_i`` moves ``w_i`` by ``2 * w_i *
      order * eps * ||T|| / |v_i|``. For the extreme node of a 25-point rule
      ``|v_i| ~ 4e-9``, which is where the ~1e-7 relative slack comes from;
      for the central nodes ``|v_i| ~ 0.5`` and the bound is ~1e-14.
    """
    gh = GaussHermiteIntegration(order, mu)
    poly = GaussHermitePolynomial(mu)
    x = gh.x()
    w = gh.weights()
    norm_t = 2.0 * max(abs(float(v)) for v in x)
    delta = order * _EPS * norm_t
    weight_bounds: list[float] = []
    for i in range(order):
        # |v_i| recovered from the weight itself: v_i^2 = w_i * w(x_i) / mu_0.
        v_i = math.sqrt(abs(float(w[i])) * poly.w(float(x[i])) / poly.mu_0())
        weight_bounds.append(2.0 * abs(float(w[i])) * delta / v_i)
    return delta, weight_bounds


def _quad_sum_bound(
    order: int,
    mu: float,
    cpp_nodes: Sequence[float],
    cpp_weights: Sequence[float],
    g: Callable[[float], float],
    *,
    leading_unweighted: bool = False,
) -> float:
    """Absolute bound on ``sum_i w_i g(x_i)`` from the rule disagreement.

    Triangle inequality on each term, using this port's own rule and the
    reference rule:

        |w_i^py g(x_i^py) - w_i^cpp g(x_i^cpp)|
            <= |w_i^py - w_i^cpp| |g(x_i^py)|
               + |w_i^cpp| |g(x_i^py) - g(x_i^cpp)|

    ``leading_unweighted`` covers ``VectorIntegrator``, whose highest-index
    term carries no weight at all (see the defect note in
    :mod:`pquantlib.experimental.math.multidim_quadrature`), so only the
    node-difference part of that term survives.
    """
    gh = GaussHermiteIntegration(order, mu)
    x = gh.x()
    w = gh.weights()
    total = 0.0
    for i in range(order):
        xp = float(x[i])
        xc = float(cpp_nodes[i])
        node_term = abs(g(xp) - g(xc))
        if leading_unweighted and i == order - 1:
            total += node_term
            continue
        total += abs(float(w[i]) - float(cpp_weights[i])) * abs(g(xp))
        total += abs(float(cpp_weights[i])) * node_term
    return total


def _abs_mass(order: int, mu: float, g: Callable[[float], float]) -> float:
    """``|g(x[n-1])| + sum_{i<n-1} |w_i g(x_i)|`` for this port's rule.

    The absolute mass of one axis of the tensor sum — the leading term is
    unweighted, matching ``VectorIntegrator``.
    """
    gh = GaussHermiteIntegration(order, mu)
    x = gh.x()
    w = gh.weights()
    total = abs(g(float(x[order - 1])))
    for i in range(order - 1):
        total += abs(float(w[i])) * abs(g(float(x[i])))
    return total


def _tensor_quad_bound(
    order: int,
    mu: float,
    cpp_nodes: Sequence[float],
    cpp_weights: Sequence[float],
    axes: Sequence[Sequence[Callable[[float], float]]],
) -> float:
    """Absolute bound on a tensor-product ``VectorIntegrator`` component.

    ``axes`` is the component written as a sum of separable terms, each term
    given as one ``g`` per axis (innermost axis first). ``integrate_v`` on a
    separable integrand is exactly the product of the per-axis defected
    quadratures, so replacing one axis at a time and bounding the rest by
    their absolute masses ``A_k`` telescopes to

        |prod_k S_k^py - prod_k S_k^cpp|
            <= sum_k (prod_{i<k} A_i) * delta_k * (prod_{i>k} A_i)

    with ``delta_k`` the per-axis bound already derived in
    :func:`_quad_sum_bound`. A second term covers the accumulation itself:
    the reference build compiles ``VectorIntegrator``'s ``w_[i]*x + y``
    (multidimquadrature.hpp:69) with clang's default ``-ffp-contract=on``,
    which fuses it into one FMA, while Python rounds twice. Each of the
    ``order`` accumulation steps at each of the ``D`` levels therefore differs
    by at most ``2 * eps`` times the magnitude passing through it, and the
    total magnitude passing through the tensor sum is ``prod_k A_k``:

        + 2 * D * eps * prod_k A_k

    Both terms are worst-case: they are bounds, not estimates. The companion
    test ``test_vector_integrator_order9_divergence_is_rule_plus_contraction``
    shows the *actual* divergence is nothing but those two effects, by
    reproducing C++ bit for bit once both are substituted.
    """
    total = 0.0
    for term in axes:
        masses = [_abs_mass(order, mu, g) for g in term]
        deltas = [
            _quad_sum_bound(order, mu, cpp_nodes, cpp_weights, g, leading_unweighted=True)
            for g in term
        ]
        for k in range(len(term)):
            contribution = deltas[k]
            for i, m in enumerate(masses):
                if i != k:
                    contribution *= m
            total += contribution
        product_mass = 1.0
        for m in masses:
            product_mass *= m
        total += 2.0 * len(term) * _EPS * product_mass
    return total


def _one(_x: float) -> float:
    return 1.0


def _ident(x: float) -> float:
    return x


def _cpp_rule_vector_integrate(
    f: Callable[[Sequence[float]], list[float]],
    dimension: int,
    order: int,
    nodes: Sequence[float],
    weights: Sequence[float],
) -> list[float]:
    """``GaussianQuadMultidimIntegrator.integrate_v`` with two substitutions.

    Verbatim transcription of multidimquadrature.hpp:55-71 + :149-157 + the
    ``<1>`` terminal at :226-234, except that (a) the quadrature rule is the
    reference build's rather than this port's, and (b) ``w_[i]*x + y`` is
    evaluated with :func:`math.fma`, i.e. contracted, as the reference build
    compiles it. Used only to attribute the order-9 divergence.
    """

    def vec_int(g: Callable[[float], list[float]]) -> list[float]:
        total = list(g(float(nodes[order - 1])))
        for i in range(order - 2, -1, -1):
            term = g(float(nodes[i]))
            w_i = float(weights[i])
            for j in range(len(total)):
                total[j] = math.fma(w_i, term[j], total[j])
        return total

    def level(buf: list[float], depth: int, m_fctr: float) -> list[float]:
        buf[depth] = m_fctr
        if depth == 0:
            return f(buf)
        return vec_int(lambda x: level(buf, depth - 1, x))

    buf = [0.0] * dimension
    return vec_int(lambda x: level(buf, dimension - 1, x))


# --- integrands (mirroring the probe's) ---------------------------------


def _std_normal_density(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _dens(v: Sequence[float]) -> float:
    return _std_normal_density(v[0])


def _fv(v: Sequence[float]) -> list[float]:
    return [1.0, v[0], v[0] * v[0], math.cos(v[0])]


def _fvd(v: Sequence[float]) -> list[float]:
    d = _std_normal_density(v[0])
    return [d, d * v[0], d * v[0] * v[0], d * math.cos(v[0])]


# =======================================================================
# A. the 1-D Gauss-Hermite rule everything below is built from
# =======================================================================


def test_gauss_hermite_25_order_and_direction(cpp: dict[str, Any]) -> None:
    gh = GaussHermiteIntegration(25, 0.0)
    assert gh.order() == int(cpp["A_gh25_order"])
    # C++ TqrEigenDecomposition sorts with std::greater<>; VectorIntegrator
    # indexes x_ and w_ directly, so this order is load-bearing.
    assert (float(gh.x()[0]) > float(gh.x()[-1])) is bool(cpp["A_gh25_descending"])


def test_gauss_hermite_25_nodes_and_weights(cpp: dict[str, Any]) -> None:
    """Nodes and weights element by element, against a derived bound."""
    gh = GaussHermiteIntegration(25, 0.0)
    node_bound, weight_bounds = _golub_welsch_bounds(25, 0.0)
    for i, expected in enumerate(cpp["A_gh25_nodes"]):
        tolerance.custom(
            float(gh.x()[i]),
            float(expected),
            abs_tol=node_bound,
            rel_tol=0.0,
            reason="eigenvalue backward-error bound; see _golub_welsch_bounds",
        )
    for i, expected in enumerate(cpp["A_gh25_weights"]):
        tolerance.custom(
            float(gh.weights()[i]),
            float(expected),
            abs_tol=weight_bounds[i],
            rel_tol=0.0,
            reason="first-eigenvector-component bound; see _golub_welsch_bounds",
        )


def test_gauss_hermite_25_integrals(cpp: dict[str, Any]) -> None:
    """Gaussian-damped integrands: the extreme-weight slack cannot reach these."""
    gh = GaussHermiteIntegration(25, 0.0)
    tolerance.tight(gh(_std_normal_density), cpp["A_gh25_int_gaussdens"])
    tolerance.tight(
        gh(lambda x: x * x * _std_normal_density(x)), cpp["A_gh25_int_x2gaussdens"]
    )


# =======================================================================
# B. VectorIntegrator
# =======================================================================


def test_vector_integrator_leading_term_is_unweighted(cpp: dict[str, Any]) -> None:
    """The defect itself: pinned as arithmetic, on both sides.

    The C++ probe reconstructs both readings of ``VectorIntegrator::operator()``
    and reports how far the library's own output is from each. This test does
    the same in Python and then checks that the two ports agree about which
    reading is the real one.
    """
    order = 25
    gh = GaussHermiteIntegration(order, 0.0)
    x = gh.x()
    w = gh.weights()
    got = GaussianQuadMultidimIntegrator(1, order).integrate_v(_fv)

    bugged = _fv([float(x[order - 1])])  # leading term, NOT scaled by w[24]
    for i in range(order - 2, -1, -1):
        term = _fv([float(x[i])])
        w_i = float(w[i])
        for j in range(len(bugged)):
            bugged[j] = w_i * term[j] + bugged[j]

    textbook = [0.0] * 4
    for i in range(order - 1, -1, -1):
        term = _fv([float(x[i])])
        for j in range(len(textbook)):
            textbook[j] += float(w[i]) * term[j]

    # same expression, same order -> bit-identical inside one interpreter
    for a, b in zip(got, bugged, strict=True):
        tolerance.exact(a, b)

    # ...and C++ says the same about its own output
    for r in cpp["B_vecint_dim1_reldiff_bugged"]:
        assert float(r) < 1e-14
    # the textbook reading is wrong by percent, not by rounding
    assert max(float(r) for r in cpp["B_vecint_dim1_reldiff_textbook"]) > 1e-2
    assert max(abs(a - b) for a, b in zip(got, textbook, strict=True)) > 1e-3


def test_vector_integrator_dim1_order25_raw(cpp: dict[str, Any]) -> None:
    """Undamped integrand at the factory's order: derived bound, not TIGHT."""
    got = GaussianQuadMultidimIntegrator(1, 25).integrate_v(_fv)
    expected = cpp["B_vecint_dim1_order25"]
    components: list[Callable[[float], float]] = [
        lambda x: 1.0,
        lambda x: x,
        lambda x: x * x,
        math.cos,
    ]
    for j, g in enumerate(components):
        tolerance.custom(
            got[j],
            float(expected[j]),
            abs_tol=_quad_sum_bound(
                25, 0.0, cpp["A_gh25_nodes"], cpp["A_gh25_weights"], g,
                leading_unweighted=True,
            ),
            rel_tol=0.0,
            reason="rule disagreement propagated term by term; see _quad_sum_bound",
        )


def test_vector_integrator_dim1_order25_damped(cpp: dict[str, Any]) -> None:
    """The realistic case — and the defect is still visible in the last digits.

    Component 0 is ``1.0000000003070459``: the ``3.07e-10`` excess *is* the
    unweighted leading term, ``(1 - w[24]) * density(x[24])``. A port that
    "fixed" the defect would return 1.0000000000 and fail here at TIGHT.
    """
    got = GaussianQuadMultidimIntegrator(1, 25).integrate_v(_fvd)
    for a, b in zip(got, cpp["B_vecintd_dim1_order25"], strict=True):
        tolerance.tight(a, float(b))


def test_gauss_hermite_order9_rule(cpp: dict[str, Any]) -> None:
    """The rule the two order-9 tensor tests below are built from.

    Neither side resolves it to the last bit: both run Golub-Welsch on the
    same Jacobi matrix but with different eigen-solvers, so the nodes agree
    to ~1e-15 absolute and the weights to ~2.5e-14 absolute. That is inside
    :func:`_golub_welsch_bounds` and is *the* reason the undamped tensor
    products below cannot be asserted at TIGHT.
    """
    gh = GaussHermiteIntegration(9, 0.0)
    node_bound, weight_bounds = _golub_welsch_bounds(9, 0.0)
    for i, expected in enumerate(cpp["B_gh9_nodes"]):
        tolerance.custom(
            float(gh.x()[i]),
            float(expected),
            abs_tol=node_bound,
            rel_tol=0.0,
            reason="eigen-solver disagreement; see _golub_welsch_bounds",
        )
    for i, expected in enumerate(cpp["B_gh9_weights"]):
        tolerance.custom(
            float(gh.weights()[i]),
            float(expected),
            abs_tol=weight_bounds[i],
            rel_tol=0.0,
            reason="eigen-solver disagreement; see _golub_welsch_bounds",
        )


def test_vector_integrator_dim2_order9(cpp: dict[str, Any]) -> None:
    """Undamped monomials in 2-D: derived bound, not TIGHT.

    ``[1, v0, v1, v0*v1]`` are the four separable products of the two axis
    quadratures ``T = Q(1) ~ 7.44`` and ``S = Q(x) ~ 0.15``. ``S`` is a sum of
    terms of magnitude ~3.3 that cancels to 0.15 — a condition number of ~90 —
    so the last-bit rule disagreement pinned by
    :func:`test_gauss_hermite_order9_rule` lands at ~1e-12 relative on ``S``
    and doubles again on ``S**2``. Nothing here is a porting difference; see
    the attribution test below.
    """
    got = GaussianQuadMultidimIntegrator(2, 9).integrate_v(
        lambda v: [1.0, v[0], v[1], v[0] * v[1]]
    )
    terms: list[list[list[Callable[[float], float]]]] = [
        [[_one, _one]],
        [[_ident, _one]],
        [[_one, _ident]],
        [[_ident, _ident]],
    ]
    for j, (a, b) in enumerate(zip(got, cpp["B_vecint_dim2_order9"], strict=True)):
        tolerance.custom(
            a,
            float(b),
            abs_tol=_tensor_quad_bound(
                9, 0.0, cpp["B_gh9_nodes"], cpp["B_gh9_weights"], terms[j]
            ),
            rel_tol=0.0,
            reason="rule disagreement + FMA contraction; see _tensor_quad_bound",
        )


def test_vector_integrator_dim2_order25_damped(cpp: dict[str, Any]) -> None:
    def f(v: Sequence[float]) -> list[float]:
        d = math.exp(-0.5 * (v[0] * v[0] + v[1] * v[1])) / (2.0 * math.pi)
        return [d, d * v[0], d * v[1], d * v[0] * v[1]]

    got = GaussianQuadMultidimIntegrator(2, 25).integrate_v(f)
    for a, b in zip(got, cpp["B_vecintd_dim2_order25"], strict=True):
        tolerance.tight(a, float(b))


def test_vector_integrator_dim3_order9(cpp: dict[str, Any]) -> None:
    """Same story one dimension up — ``S**3`` triples the amplification."""
    got = GaussianQuadMultidimIntegrator(3, 9).integrate_v(
        lambda v: [1.0, v[0] + v[1] + v[2], v[0] * v[1] * v[2]]
    )
    terms: list[list[list[Callable[[float], float]]]] = [
        [[_one, _one, _one]],
        [[_ident, _one, _one], [_one, _ident, _one], [_one, _one, _ident]],
        [[_ident, _ident, _ident]],
    ]
    for j, (a, b) in enumerate(zip(got, cpp["B_vecint_dim3_order9"], strict=True)):
        tolerance.custom(
            a,
            float(b),
            abs_tol=_tensor_quad_bound(
                9, 0.0, cpp["B_gh9_nodes"], cpp["B_gh9_weights"], terms[j]
            ),
            rel_tol=0.0,
            reason="rule disagreement + FMA contraction; see _tensor_quad_bound",
        )


def test_vector_integrator_order9_divergence_is_rule_plus_contraction(
    cpp: dict[str, Any],
) -> None:
    """Attribution: the two order-9 gaps are *only* the rule and the FMA.

    Feeding this port's own recursion the reference build's nodes and weights
    and contracting ``w_[i]*x + y`` into a single ``math.fma`` reproduces
    every component of both C++ vectors **bit for bit**. There is no third
    term — no reordering, no missing weight, no different recursion. That is
    what licenses the derived bounds above instead of TIGHT.
    """
    nodes = cpp["B_gh9_nodes"]
    weights = cpp["B_gh9_weights"]
    got2 = _cpp_rule_vector_integrate(
        lambda v: [1.0, v[0], v[1], v[0] * v[1]], 2, 9, nodes, weights
    )
    for a, b in zip(got2, cpp["B_vecint_dim2_order9"], strict=True):
        tolerance.exact(a, float(b))
    got3 = _cpp_rule_vector_integrate(
        lambda v: [1.0, v[0] + v[1] + v[2], v[0] * v[1] * v[2]], 3, 9, nodes, weights
    )
    for a, b in zip(got3, cpp["B_vecint_dim3_order9"], strict=True):
        tolerance.exact(a, float(b))


def test_vector_integrator_dim1_order4(cpp: dict[str, Any]) -> None:
    """Order 4: small enough that the rule agrees to the last bits."""
    got = GaussianQuadMultidimIntegrator(1, 4).integrate_v(_fv)
    for a, b in zip(got, cpp["B_vecint_dim1_order4"], strict=True):
        tolerance.tight(a, float(b))
    gh = GaussHermiteIntegration(4, 0.0)
    for i, expected in enumerate(cpp["B_gh4_nodes"]):
        tolerance.tight(float(gh.x()[i]), float(expected))
    for i, expected in enumerate(cpp["B_gh4_weights"]):
        tolerance.tight(float(gh.weights()[i]), float(expected))


def test_vector_integrator_mu(cpp: dict[str, Any]) -> None:
    """``mu != 0`` reaches the generalized-Hermite weight ``|x|^(2 mu) e^-x^2``."""
    quad = GaussianQuadMultidimIntegrator(1, 8, 0.3)
    assert quad.order() == int(cpp["B_vecint_mu_order"])
    for a, b in zip(quad.integrate_v(_fv), cpp["B_vecint_dim1_order8_mu0_3"], strict=True):
        tolerance.tight(a, float(b))
    gh = GaussHermiteIntegration(8, 0.3)
    for i, expected in enumerate(cpp["B_gh8_mu0_3_nodes"]):
        tolerance.tight(float(gh.x()[i]), float(expected))
    for i, expected in enumerate(cpp["B_gh8_mu0_3_weights"]):
        tolerance.tight(float(gh.weights()[i]), float(expected))


# =======================================================================
# C. the scalar path the vector one sits beside
# =======================================================================


def test_scalar_quadrature(cpp: dict[str, Any]) -> None:
    q1 = GaussianQuadMultidimIntegrator(1, 25)
    assert q1.order() == int(cpp["C_quad_order"])
    tolerance.tight(
        q1.integrate(lambda v: math.exp(-0.5 * v[0] * v[0])), cpp["C_quad_dim1_one"]
    )
    tolerance.tight(
        q1.integrate(lambda v: v[0] * v[0] * math.exp(-0.5 * v[0] * v[0])),
        cpp["C_quad_dim1_sqr"],
    )
    # operator() and integrate<Real> are the same expression in C++
    assert q1(lambda v: math.exp(-0.5 * v[0] * v[0])) == q1.integrate(
        lambda v: math.exp(-0.5 * v[0] * v[0])
    )
    q2 = GaussianQuadMultidimIntegrator(2, 25)
    tolerance.tight(
        q2.integrate(
            lambda v: v[0] * v[1] * math.exp(-0.5 * (v[0] * v[0] + v[1] * v[1]))
        ),
        cpp["C_quad_dim2_prod"],
    )
    tolerance.tight(
        q2.integrate(
            lambda v: (v[0] * v[0] + v[1] * v[1])
            * math.exp(-0.5 * (v[0] * v[0] + v[1] * v[1]))
        ),
        cpp["C_quad_dim2_sqr"],
    )


# =======================================================================
# D / E. IntegrationBase and its two specialisations
# =======================================================================


def test_integration_base_primary_template_refuses_construction() -> None:
    """# C++ parity: the primary template's ``private`` default ctor."""
    with pytest.raises(LibraryException, match="fully specialised"):
        IntegrationBase()


def test_integration_base_quadrature(cpp: dict[str, Any]) -> None:
    ib = GaussianQuadIntegrationBase(1, 25)
    assert isinstance(ib, LMIntegration)
    assert isinstance(ib, GaussianQuadMultidimIntegrator)
    assert ib.order() == int(cpp["D_ibquad_order"])
    tolerance.tight(
        ib.integrate(lambda v: _std_normal_density(v[0])),
        cpp["D_ibquad_integrate_gaussdens"],
    )
    got = ib.integrate_v(
        lambda v: [
            _std_normal_density(v[0]),
            _std_normal_density(v[0]) * v[0],
            _std_normal_density(v[0]) * v[0] * v[0],
        ]
    )
    for a, b in zip(got, cpp["D_ibquad_integrateV"], strict=True):
        tolerance.tight(a, float(b))
    # dispatched through the abstract base, as C++ does through the vtable
    lmi: LMIntegration = ib
    tolerance.tight(
        lmi.integrate(lambda v: _std_normal_density(v[0])), cpp["D_ibquad_via_base"]
    )
    ib2 = GaussianQuadIntegrationBase(2, 25)
    tolerance.tight(
        ib2.integrate(
            lambda v: math.exp(-0.5 * (v[0] * v[0] + v[1] * v[1])) / (2.0 * math.pi)
        ),
        cpp["D_ibquad_dim2"],
    )


def test_integration_base_trapezoid(cpp: dict[str, Any]) -> None:
    ib = MultidimIntegralIntegrationBase([TrapezoidIntegral(1.0e-4, 20)], -35.0, 35.0)
    assert isinstance(ib, LMIntegration)
    assert ib.a == [float(v) for v in cpp["E_ibtrap_a"]]
    assert ib.b == [float(v) for v in cpp["E_ibtrap_b"]]
    tolerance.tight(
        ib.integrate(lambda v: _std_normal_density(v[0])), cpp["E_ibtrap_gaussdens"]
    )
    tolerance.tight(
        ib.integrate(lambda v: v[0] * v[0] * _std_normal_density(v[0])),
        cpp["E_ibtrap_x2gaussdens"],
    )
    tolerance.tight(
        ib.integrate(lambda v: abs(v[0]) * _std_normal_density(v[0])),
        cpp["E_ibtrap_absxgaussdens"],
    )
    tolerance.tight(
        ib.integrate(lambda v: math.cos(v[0]) * _std_normal_density(v[0])),
        cpp["E_ibtrap_cosgaussdens"],
    )


def test_integration_base_trapezoid_has_no_vector_integration(cpp: dict[str, Any]) -> None:
    """C++ leaves ``integrateV`` unimplemented here; the base default fails."""
    assert bool(cpp["E_ibtrap_integrateV_throws"])
    assert bool(cpp["E_ibtrap_integrateV_msg_has_novector"])
    ib = MultidimIntegralIntegrationBase([TrapezoidIntegral(1.0e-4, 20)], -35.0, 35.0)
    with pytest.raises(LibraryException, match="No vector integration provided"):
        ib.integrate_v(lambda v: [1.0])


def test_integration_base_trapezoid_dim2(cpp: dict[str, Any]) -> None:
    ib = MultidimIntegralIntegrationBase(
        [TrapezoidIntegral(1.0e-4, 20), TrapezoidIntegral(1.0e-4, 20)], -8.0, 8.0
    )
    assert ib.a == [float(v) for v in cpp["E_ibtrap2_a"]]
    assert ib.b == [float(v) for v in cpp["E_ibtrap2_b"]]
    tolerance.tight(
        ib.integrate(
            lambda v: math.exp(-0.5 * (v[0] * v[0] + v[1] * v[1])) / (2.0 * math.pi)
        ),
        cpp["E_ibtrap2_gaussdens"],
    )


# =======================================================================
# F. IntegrationFactory
# =======================================================================


def test_integration_type_enum_values(cpp: dict[str, Any]) -> None:
    assert int(LatentModelIntegrationType.GAUSSIAN_QUADRATURE) == int(
        cpp["F_enum_GaussianQuadrature"]
    )
    assert int(LatentModelIntegrationType.TRAPEZOID) == int(cpp["F_enum_Trapezoid"])


def test_integration_factory_hardcoded_parameters(cpp: dict[str, Any]) -> None:
    """Order 25 / trapezoid(1e-4, 20) over [-35, 35], pinned by behaviour.

    The C++ probe proves the same thing on its side by requiring the factory
    product and a hand-built ``IntegrationBase`` to agree bit-for-bit.
    """
    assert bool(cpp["F_factory_quad_is_order25"])
    assert bool(cpp["F_factory_trap_is_1e4_20_m35_35"])

    factory = LatentModel.IntegrationFactory

    quad = factory.create_lm_integration(1, LatentModelIntegrationType.GAUSSIAN_QUADRATURE)
    trap = factory.create_lm_integration(1, LatentModelIntegrationType.TRAPEZOID)
    tolerance.tight(quad.integrate(_dens), cpp["F_factory_quad_gaussdens"])
    tolerance.tight(trap.integrate(_dens), cpp["F_factory_trap_gaussdens"])

    assert quad.integrate(_dens) == GaussianQuadIntegrationBase(1, 25).integrate(_dens)
    assert trap.integrate(_dens) == MultidimIntegralIntegrationBase(
        [TrapezoidIntegral(1.0e-4, 20)], -35.0, 35.0
    ).integrate(_dens)


def test_integration_factory_default_is_quadrature(cpp: dict[str, Any]) -> None:
    assert bool(cpp["F_factory_default_is_quadrature"])
    factory = LatentModel.IntegrationFactory
    assert factory.create_lm_integration(1).integrate(_dens) == factory.create_lm_integration(
        1, LatentModelIntegrationType.GAUSSIAN_QUADRATURE
    ).integrate(_dens)


def test_integration_factory_dim2(cpp: dict[str, Any]) -> None:
    lmi = LatentModel.IntegrationFactory.create_lm_integration(
        2, LatentModelIntegrationType.GAUSSIAN_QUADRATURE
    )
    tolerance.tight(
        lmi.integrate(
            lambda v: math.exp(-0.5 * (v[0] * v[0] + v[1] * v[1])) / (2.0 * math.pi)
        ),
        cpp["F_factory_quad_dim2_gaussdens"],
    )


def test_integration_factory_rejects_unknown_type(cpp: dict[str, Any]) -> None:
    """The ``default: QL_FAIL(...)`` arm (latentmodel.hpp:481-483).

    C++ reaches it by passing an out-of-range value of the *unscoped* enum
    ``LatentModelIntegrationType::LatentModelIntegrationType``, which is a
    plain C enum and accepts any integer. ``LatentModelIntegrationType(99)``
    is not constructible in Python — ``enum`` rejects it before the call —
    so the analogue is to pass the bare ``99`` the C++ caller would have
    passed. The annotation is not enforced at runtime, which is exactly the
    latitude the C++ enum has.
    """
    assert bool(cpp["F_factory_unknown_type_throws"])
    assert bool(cpp["F_factory_unknown_type_msg"])
    with pytest.raises(ValueError, match="99 is not a valid LatentModelIntegrationType"):
        LatentModelIntegrationType(99)
    with pytest.raises(LibraryException, match="Unknown latent model integration type"):
        LatentModel.IntegrationFactory.create_lm_integration(
            1,
            cast("LatentModelIntegrationType", 99),
        )


def test_integration_factory_is_a_static_factory() -> None:
    """# C++ parity: ``private: IntegrationFactory() = default;``."""
    with pytest.raises(LibraryException, match="static factory"):
        LatentModel.IntegrationFactory()


# =======================================================================
# G. LatentModel.integrated_expected_value end to end
# =======================================================================


def _gauss_model(
    weights: Sequence[Sequence[float]], integration_type: LatentModelIntegrationType
) -> LatentModel:
    return LatentModel(
        weights,
        GaussianCopulaPolicy(weights),
        LatentModel.IntegrationFactory.create_lm_integration(
            len(weights[0]), integration_type
        ),
    )


_SCALAR_INTEGRANDS: dict[str, Callable[[Sequence[float]], float]] = {
    "one": lambda v: 1.0,
    "lin": lambda v: v[0],
    "sqr": lambda v: v[0] * v[0],
    "gauss": lambda v: math.exp(-0.25 * v[0] * v[0]),
    "cos": lambda v: math.cos(v[0]),
}


@pytest.mark.parametrize("name", list(_SCALAR_INTEGRANDS))
@pytest.mark.parametrize("kind", ["quad", "trap"])
def test_gaussian_integrated_expected_value(
    cpp: dict[str, Any], kind: str, name: str
) -> None:
    integration_type = (
        LatentModelIntegrationType.GAUSSIAN_QUADRATURE
        if kind == "quad"
        else LatentModelIntegrationType.TRAPEZOID
    )
    lm = _gauss_model(_W1, integration_type)
    tolerance.tight(
        lm.integrated_expected_value(_SCALAR_INTEGRANDS[name]),
        cpp[f"G_gauss_{kind}_{name}"],
    )


def test_gaussian_integrated_expected_value_matches_analytics(cpp: dict[str, Any]) -> None:
    """The pinned numbers are also the right numbers, to quadrature accuracy."""
    tolerance.loose(
        float(cpp["G_gauss_quad_gauss"]),
        float(cpp["G_analytic_gauss"]),
        reason="25-point Gauss-Hermite of exp(-x^2/4) against sqrt(2/3).",
    )
    tolerance.loose(
        float(cpp["G_gauss_trap_cos"]),
        float(cpp["G_analytic_cos"]),
        reason="trapezoid over [-35,35] of cos(x) phi(x) against exp(-1/2).",
    )


def test_gaussian_integrated_expected_value_vector(cpp: dict[str, Any]) -> None:
    lm = _gauss_model(_W1, LatentModelIntegrationType.GAUSSIAN_QUADRATURE)
    got = lm.integrated_expected_value_v(_fv)
    for a, b in zip(got, cpp["G_gauss_quad_V"], strict=True):
        tolerance.tight(a, float(b))


def test_trapezoid_facility_has_no_vector_integration(cpp: dict[str, Any]) -> None:
    assert bool(cpp["G_gauss_trap_V_throws"])
    lm = _gauss_model(_W1, LatentModelIntegrationType.TRAPEZOID)
    with pytest.raises(LibraryException, match="No vector integration provided"):
        lm.integrated_expected_value_v(lambda v: [1.0])


def test_two_systemic_factors(cpp: dict[str, Any]) -> None:
    lm = _gauss_model(_W2, LatentModelIntegrationType.GAUSSIAN_QUADRATURE)
    tolerance.tight(lm.integrated_expected_value(lambda v: 1.0), cpp["G_gauss2_quad_one"])
    tolerance.tight(
        lm.integrated_expected_value(lambda v: v[0] * v[1]), cpp["G_gauss2_quad_prod"]
    )
    tolerance.tight(
        lm.integrated_expected_value(lambda v: v[0] * v[0] + v[1] * v[1]),
        cpp["G_gauss2_quad_sqr"],
    )


def _t_model(integration_type: LatentModelIntegrationType) -> LatentModel:
    return LatentModel(
        _WT,
        TCopulaPolicy(_WT, _T_ORDERS),
        LatentModel.IntegrationFactory.create_lm_integration(1, integration_type),
    )


def test_t_copula_density_and_variance_factors(cpp: dict[str, Any]) -> None:
    """The scipy Student-t delegation, checked against C++ boost::math."""
    policy = TCopulaPolicy(_WT, _T_ORDERS)
    for a, b in zip(policy.variance_factors(), cpp["G_t_varianceFactors"], strict=True):
        tolerance.tight(a, float(b))
    for point, expected in zip(
        cpp["G_t_density_pts"], cpp["G_t_density_vals"], strict=True
    ):
        tolerance.tight(policy.density([float(point)]), float(expected))


def test_t_copula_trapezoid_integration(cpp: dict[str, Any]) -> None:
    lm = _t_model(LatentModelIntegrationType.TRAPEZOID)
    tolerance.tight(lm.integrated_expected_value(lambda v: 1.0), cpp["G_t_trap_one"])
    tolerance.tight(
        lm.integrated_expected_value(lambda v: v[0] * v[0]), cpp["G_t_trap_sqr"]
    )
    tolerance.tight(
        lm.integrated_expected_value(lambda v: math.exp(-0.25 * v[0] * v[0])),
        cpp["G_t_trap_gauss"],
    )


@pytest.mark.parametrize(
    ("key", "power"),
    [("G_t_quad_one", 0), ("G_t_quad_sqr", 2)],
)
def test_t_copula_quadrature_integration(cpp: dict[str, Any], key: str, power: int) -> None:
    """Gauss-Hermite over a Student-t density: derived bound, not TIGHT.

    A Student-t density does *not* decay like ``exp(-x^2)``, so it is still
    ~4e-4 at the extreme 25-point node, and the ~5e-8 absolute slack in that
    node's weight reaches the answer. ``_quad_sum_bound`` measures exactly
    that, from this port's rule against the reference rule.
    """
    lm = _t_model(LatentModelIntegrationType.GAUSSIAN_QUADRATURE)
    policy = lm.copula()
    got = lm.integrated_expected_value(lambda v: v[0] ** power)
    bound = _quad_sum_bound(
        25,
        0.0,
        cpp["A_gh25_nodes"],
        cpp["A_gh25_weights"],
        lambda x: policy.density([x]) * x**power,
    )
    tolerance.custom(
        got,
        float(cpp[key]),
        abs_tol=bound,
        rel_tol=0.0,
        reason="rule disagreement propagated term by term; see _quad_sum_bound",
    )


# =======================================================================
# H. FactorSampler
# =======================================================================


def _drain(sampler: Any, draws: int) -> list[float]:
    flat: list[float] = []
    for _ in range(draws):
        flat.extend(float(v) for v in sampler.next_sequence().value)
    return flat


def test_factor_sampler_generic_gaussian(cpp: dict[str, Any]) -> None:
    policy = GaussianCopulaPolicy(_W1)
    assert policy.num_factors() == int(cpp["H_gauss_numFactors"])
    sampler = LatentModel.FactorSampler(policy, 42)
    first = sampler.next_sequence()
    tolerance.exact(first.weight, float(cpp["H1_generic_weight0"]))
    flat = [float(v) for v in first.value] + _drain(sampler, 4)
    for a, b in zip(flat, cpp["H1_generic_seed42_5x4"], strict=True):
        tolerance.tight(a, float(b))


def test_factor_sampler_box_muller(cpp: dict[str, Any]) -> None:
    policy = GaussianCopulaPolicy(_W1)
    sampler = LatentModel.BoxMullerFactorSampler(policy, 42)
    first = sampler.next_sequence()
    tolerance.exact(first.weight, float(cpp["H2_boxmuller_weight0"]))
    flat = [float(v) for v in first.value] + _drain(sampler, 4)
    for a, b in zip(flat, cpp["H2_boxmuller_seed42_5x4"], strict=True):
        tolerance.tight(a, float(b))


def test_factor_sampler_box_muller_differs_from_generic(cpp: dict[str, Any]) -> None:
    """The specialisation bypasses the copula inversion, so it is a different
    stream off the same seed — that is the whole point of specialising."""
    generic = cpp["H1_generic_seed42_5x4"]
    box_muller = cpp["H2_boxmuller_seed42_5x4"]
    assert all(abs(a - b) > 1e-3 for a, b in zip(generic, box_muller, strict=True))


def test_factor_sampler_polar_student_t(cpp: dict[str, Any]) -> None:
    policy = TCopulaPolicy(_WT, _T_ORDERS)
    assert policy.num_factors() == int(cpp["H3_t_numFactors"])
    for a, b in zip(policy.variance_factors(), cpp["H3_t_varianceFactors"], strict=True):
        tolerance.tight(a, float(b))
    sampler = LatentModel.PolarStudentTFactorSampler(policy, 42)
    first = sampler.next_sequence()
    tolerance.exact(first.weight, float(cpp["H3_polart_weight0"]))
    flat = [float(v) for v in first.value] + _drain(sampler, 4)
    for a, b in zip(flat, cpp["H3_polart_seed42_5x3"], strict=True):
        tolerance.tight(a, float(b))


def test_factor_sampler_polar_student_t_shares_one_uniform_stream(
    cpp: dict[str, Any],
) -> None:
    """The C++ defect, isolated.

    ``FactorSampler``'s ctor (latentmodel.hpp:775-783) hands every per-factor
    ``PolarStudentTRng`` a **copy** of the sampler's own just-seeded uniform
    generator::

        for (Real i : varF)
            trng_.push_back(PolarStudentTRng<urng_type>(2. / (1. - i * i), urng_));

    so every generator replays the same uniforms. ``PolarStudentTRng::next()``
    (polarstudenttrng.hpp:77-86) draws the pair ``(v, u)`` and rejects on
    ``r^2 = v*v + u*u >= 1``, a test that does **not** involve ``degFreedom_``:
    the generators therefore accept and reject in lockstep, and their k-th
    draws are ``u * sqrt(nu (r^-2/nu - 1) / r^2)`` off the *same* ``u``. The
    sign is ``sign(u)``, so the k-th draws never disagree about it.

    C++'s own 32-draw streams are the proof; the assertion below is made
    against them before it is made against this port.
    """
    policy = TCopulaPolicy(_WT, _T_ORDERS)
    dfs = [2.0 / (1.0 - v * v) for v in policy.variance_factors()]
    for a, b in zip(dfs, cpp["H3_derived_dfs"], strict=True):
        tolerance.tight(a, float(b))

    urng = MersenneTwisterUniformRng(42)
    rng3 = PolarStudentTRng(dfs[0], copy.deepcopy(urng))
    rng5 = PolarStudentTRng(dfs[1], copy.deepcopy(urng))
    for a, b in zip(
        [rng3.next().value for _ in range(4)], cpp["H3_standalone_df3"], strict=True
    ):
        tolerance.tight(a, float(b))
    for a, b in zip(
        [rng5.next().value for _ in range(4)], cpp["H3_standalone_df5"], strict=True
    ):
        tolerance.tight(a, float(b))

    # the dependence itself, over 32 draws — asserted on C++'s own output first
    cpp3 = [float(v) for v in cpp["H3_standalone_df3_32"]]
    cpp5 = [float(v) for v in cpp["H3_standalone_df5_32"]]
    assert all((a > 0.0) == (b > 0.0) for a, b in zip(cpp3, cpp5, strict=True))

    urng = MersenneTwisterUniformRng(42)
    py3 = PolarStudentTRng(dfs[0], copy.deepcopy(urng))
    py5 = PolarStudentTRng(dfs[1], copy.deepcopy(urng))
    got3 = [py3.next().value for _ in range(32)]
    got5 = [py5.next().value for _ in range(32)]
    for a, b in zip(got3, cpp3, strict=True):
        tolerance.tight(a, float(b))
    for a, b in zip(got5, cpp5, strict=True):
        tolerance.tight(a, float(b))
    assert all((a > 0.0) == (b > 0.0) for a, b in zip(got3, got5, strict=True))


def test_polar_student_t_sign_sharing_does_not_reach_the_sample_vector(
    cpp: dict[str, Any],
) -> None:
    """...because ``nextSequence()`` drains the generators at different rates.

    latentmodel.hpp:784-791::

        for(; i<trng_.size(); i++)              // one draw per systemic factor
            sequence_.value[i] = trng_[i].next().value;
        for(; i<sequence_.value.size(); i++)    // the rest, all off the LAST one
            sequence_.value[i] = trng_.back().next().value;

    With ``numFactors() == 3`` and two variance factors, ``trng_.back()``
    advances **twice** per row while ``trng_[0]`` advances once, so after the
    first row the two columns are reading different uniform pairs and the
    lockstep of the previous test is gone. Guarding it here keeps anyone from
    re-deriving the (false) claim that the sample vector inherits the sign
    agreement.
    """
    rows = [
        [float(v) for v in cpp["H3_polart_seed42_5x3"][3 * k : 3 * k + 3]] for k in range(5)
    ]
    assert any((r[0] > 0.0) != (r[1] > 0.0) for r in rows)

    # column 0 IS trng_[0]'s k-th draw, bit for bit.
    cpp3 = [float(v) for v in cpp["H3_standalone_df3_32"]]
    for k, r in enumerate(rows):
        tolerance.exact(r[0], cpp3[k])

    # ...and this port reproduces the same rows.
    policy = TCopulaPolicy(_WT, _T_ORDERS)
    sampler = LatentModel.PolarStudentTFactorSampler(policy, 42)
    for k in range(5):
        got = [float(v) for v in sampler.next_sequence().value]
        for a, b in zip(got, rows[k], strict=True):
            tolerance.tight(a, b)


def test_factor_sampler_generic_over_t_copula(cpp: dict[str, Any]) -> None:
    """For contrast: the generic sampler does go through the copula inversion,
    so its draws are the variance-normalised ones."""
    policy = TCopulaPolicy(_WT, _T_ORDERS)
    sampler = LatentModel.FactorSampler(policy, 42)
    for a, b in zip(_drain(sampler, 3), cpp["H4_generic_t_seed42_3x3"], strict=True):
        tolerance.tight(a, float(b))


# =======================================================================
# I. Ziggurat RNG traits
# =======================================================================


def test_ziggurat_traits(cpp: dict[str, Any]) -> None:
    assert Ziggurat.allows_error_estimate == int(cpp["I_ziggurat_allowsErrorEstimate"])
    assert Ziggurat.rng_type is ZigguratRng
    assert Ziggurat.rsg_type is RandomSequenceGenerator


def test_ziggurat_sequence_generator(cpp: dict[str, Any]) -> None:
    rsg = Ziggurat.make_sequence_generator(4, 42)
    assert rsg.dimension() == int(cpp["I_ziggurat_rsg_dimension"])
    first = rsg.next_sequence()
    tolerance.exact(first.weight, float(cpp["I_ziggurat_weight0"]))
    flat = [float(v) for v in first.value] + _drain(rsg, 4)
    # the MT uniform stream, the ziggurat tables and the tail quantile are all
    # bit-identical to C++, so the whole sequence is.
    for a, b in zip(flat, cpp["I_ziggurat_seed42_5x4"], strict=True):
        tolerance.exact(a, float(b))


def test_ziggurat_factory_is_rsg_over_ziggurat_rng(cpp: dict[str, Any]) -> None:
    assert bool(cpp["I_ziggurat_factory_is_rsg_of_zigguratrng"])
    hand = RandomSequenceGenerator(4, ZigguratRng(42))
    factory = Ziggurat.make_sequence_generator(4, 42)
    for a, b in zip(_drain(hand, 5), _drain(factory, 5), strict=True):
        tolerance.exact(a, b)
    scalar = ZigguratRng(42)
    for a, b in zip(
        [scalar.next().value for _ in range(8)], cpp["I_zigguratrng_seed42_8"], strict=True
    ):
        tolerance.exact(a, float(b))
