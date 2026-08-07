"""GaussianQuadMultidimIntegrator — tensor-product Gauss-Hermite quadrature.

# C++ parity: ql/experimental/math/multidimquadrature.{hpp,cpp} @ v1.43.
# class GaussianQuadMultidimIntegrator, its private nested class
# VectorIntegrator (multidimquadrature.hpp:49-72) and the four template
# specialisations of ``integrate``/``operator()`` at :180-235.

Integrates a scalar **or vector** function of a vector domain over ``R^dim``
using a Gauss-Hermite quadrature along each axis. The C++ class uses template
recursion over the dimensions; the Python port expresses the same nested
cross-section recursion directly, level by level, in the same order.

The 1-D building block is QuantLib's ``GaussHermiteIntegration(order, mu)`` —
the generalised Gauss-Hermite rule with weight ``w(x) = |x|^(2 mu) exp(-x^2)``.
Crucially, QuantLib *divides the quadrature weights by* ``w(x_i)``
(``w_i = mu_0 * ev[0,i]^2 / w(x_i)``), so the rule approximates the plain
Lebesgue integral ``int f(x) dx`` (the Gaussian decay is assumed to be carried
by the integrand).

.. rubric:: Where the nodes come from (align, v1.43)

Earlier revisions of this module re-derived the Gauss-Hermite nodes and
weights inline with a private Golub-Welsch helper over ``numpy.linalg.eigh``.
C++ does not: ``GaussianQuadMultidimIntegrator`` holds a
``GaussHermiteIntegration integral_`` and a ``VectorIntegrator integralV_``
(which *is-a* ``GaussHermiteIntegration``), so the rule is the one already
ported in :mod:`pquantlib.math.integrals.gaussian_quadrature`. This module now
uses that class, which (a) removes a duplicated eigen-decomposition, and
(b) — the part that matters — gives the nodes in the same **descending** index
order as C++'s ``TqrEigenDecomposition``. ``VectorIntegrator`` indexes ``x_``
and ``w_`` directly, so its result depends on that order.

.. rubric:: Accuracy note for high orders

``LatentModel``'s integration factory asks for order 25. At that order the
outermost Gauss-Hermite weight is built from an eigenvector component of size
~4e-9, so Golub-Welsch in double precision resolves it to only about 8
significant digits, and its exact value depends on the last-bit details of the
eigen-solver. C++ and this port therefore agree on ``weights()[0]`` to ~6e-8
relative, not to the last bit; that is inherent to the method, not a porting
artefact. It is invisible for the integrands this class exists to serve (a
probability density times a payoff decays to ~1e-9 at the extreme node) but it
does bound how tightly a *raw* order-25 quadrature can be cross-validated.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from pquantlib import qassert
from pquantlib.math.integrals.gaussian_quadrature import GaussHermiteIntegration

_MAX_DIMENSIONS = 15

#: An integrand of a vector argument returning a scalar.
ScalarIntegrand = Callable[[Sequence[float]], float]
#: An integrand of a vector argument returning a vector.
VectorIntegrand = Callable[[Sequence[float]], list[float]]


class GaussianQuadMultidimIntegrator:
    """Tensor-product Gauss-Hermite quadrature over ``R^dimension``.

    # C++ parity: ``class GaussianQuadMultidimIntegrator``
    # (multidimquadrature.hpp:47-177, ctor at multidimquadrature.cpp:26-37).
    """

    class VectorIntegrator(GaussHermiteIntegration):
        """Gauss-Hermite quadrature of a **vector**-valued function of a scalar.

        # C++ parity: ``class GaussianQuadMultidimIntegrator::VectorIntegrator``
        # (multidimquadrature.hpp:49-72). ``private:`` in C++ — it exists only
        # to give ``integralV_`` a type — but it carries the whole vector
        # quadrature, so it is ported rather than folded away.

        # C++ parity note (DEFECT, reproduced verbatim): the leading term is
        # **not** weighted. multidimquadrature.hpp:58-62 reads
        #
        #     std::vector<Real> term = f(x_[i]);                  // i = n-1
        #     std::for_each(term.begin(), term.end(),
        #                   [&](Real x) -> Real { return x * w_[i]; });
        #     std::vector<Real> sum = term;
        #
        # ``std::for_each`` passes ``Real x`` **by value** and discards the
        # lambda's return, so ``term`` is never scaled: the highest-index node
        # enters the sum with weight 1 instead of ``w_[n-1]``. Every vector
        # integral this class produces therefore carries an extra
        # ``(1 - w_[n-1]) * f(x_[n-1])``. Do not "fix" it — the port's job is
        # to reproduce v1.43, and the defect is pinned by the C++ probe
        # (``B_vecint_dim1_reldiff_bugged`` ~ 4e-16 against the bugged reading,
        # ~1e-2 against the textbook one).
        """

        __slots__ = ()

        def __call__(self, f: Callable[[float], list[float]]) -> list[float]:  # type: ignore[override]
            """``sum_i w_i f(x_i)`` — except for the unweighted leading term.

            # C++ parity: ``VectorIntegrator::operator()``
            # (multidimquadrature.hpp:55-71).
            """
            x = self.x()
            w = self.weights()
            i = self.order() - 1
            # the leading term, deliberately unweighted (see the class note)
            total = list(f(float(x[i])))
            for i in range(self.order() - 2, -1, -1):
                term = f(float(x[i]))
                w_i = float(w[i])
                for j in range(len(total)):
                    total[j] = w_i * term[j] + total[j]
            return total

    __slots__ = ("_dimension", "_integral", "_integral_v")

    def __init__(self, dimension: int, quad_order: int, mu: float = 0.0) -> None:
        # C++ parity note: the C++ ctor does NOT range-check ``dimension``; it
        # sizes ``integrationEntries_`` at ``maxDimensions_`` and then indexes
        # it with ``dimension_-1``, so ``dimension > 15`` is undefined
        # behaviour there. The check below turns that into an error; it is the
        # only place this port refuses an input C++ accepts.
        qassert.require(
            dimension <= _MAX_DIMENSIONS,
            "Too many dimensions in integration.",
        )
        self._dimension = dimension
        # C++ parity: multidimquadrature.cpp:28-29 — two independent rules of
        # the same order, one scalar and one vector.
        self._integral = GaussHermiteIntegration(quad_order, mu)
        self._integral_v = GaussianQuadMultidimIntegrator.VectorIntegrator(quad_order, mu)

    def order(self) -> int:
        """Quadrature order.

        # C++ parity: ``order()`` (multidimquadrature.hpp:81) — reads it off
        # ``integralV_``, not ``integral_``.
        """
        return self._integral_v.order()

    # ---- scalar integrand -------------------------------------------------

    def _scalar_integrator(
        self, f: ScalarIntegrand, buf: list[float], depth: int, m_fctr: float
    ) -> float:
        # C++ parity: ``scalarIntegrator<intgDepth>`` (multidimquadrature.hpp:
        # 139-147) and its ``<1>`` terminal specialisation (:216-224).
        buf[depth] = m_fctr
        if depth == 0:
            return f(buf)
        return self._integral(lambda x: self._scalar_integrator(f, buf, depth - 1, x))

    def integrate(self, f: ScalarIntegrand) -> float:
        """Integrate scalar ``f`` over ``R^dimension``.

        # C++ parity: ``integrate<Real>`` (multidimquadrature.hpp:194-202).
        """
        # C++ parity note: C++ shares one ``mutable varBuffer_`` across calls;
        # the buffer is allocated per call here, which is observationally
        # identical unless the integrand keeps the vector it is handed.
        buf = [0.0] * self._dimension
        return self._integral(
            lambda x: self._scalar_integrator(f, buf, self._dimension - 1, x)
        )

    def __call__(self, f: ScalarIntegrand) -> float:
        """Integrate scalar ``f`` over ``R^dimension``.

        # C++ parity: ``operator()<Real>`` (multidimquadrature.hpp:182-188) —
        # the same expression as ``integrate<Real>``.
        """
        return self.integrate(f)

    # ---- vector integrand -------------------------------------------------

    def _vector_integrator_vr(
        self, f: VectorIntegrand, buf: list[float], depth: int, m_fctr: float
    ) -> list[float]:
        # C++ parity: ``vectorIntegratorVR<intgDepth>``
        # (multidimquadrature.hpp:149-157) and its ``<1>`` terminal
        # specialisation (:226-234).
        buf[depth] = m_fctr
        if depth == 0:
            return f(buf)
        return self._integral_v(lambda x: self._vector_integrator_vr(f, buf, depth - 1, x))

    def integrate_v(self, f: VectorIntegrand) -> list[float]:
        """Integrate vector-valued ``f`` over ``R^dimension``.

        # C++ parity: ``integrate<std::vector<Real> >``
        # (multidimquadrature.hpp:208-214). Carries the ``VectorIntegrator``
        # leading-term defect once per recursion level.
        """
        buf = [0.0] * self._dimension
        return self._integral_v(
            lambda x: self._vector_integrator_vr(f, buf, self._dimension - 1, x)
        )


# The class carries the C++ spelling; this is the descriptive name it used to be
# declared under, kept so existing call sites keep working.
MultidimGaussianQuadrature = GaussianQuadMultidimIntegrator


__all__ = ["GaussianQuadMultidimIntegrator", "MultidimGaussianQuadrature"]
