"""LatentModel — the generic multifactor latent variable model.

# C++ parity: ql/experimental/math/latentmodel.hpp @ v1.43.

The latent model describes a set of latent (random) variables

    Y_i = sum_k a_{i,k} M_k + sqrt(1 - sum_k a_{i,k}^2) Z_i

where the systemic factors ``M_k`` and the idiosyncratic factors ``Z_i`` have
independent zero-mean unit-variance distributions fixed by a *copula policy*
(:class:`~pquantlib.experimental.math.gaussian_copula_policy.GaussianCopulaPolicy`
or :class:`~pquantlib.experimental.math.t_copula_policy.TCopulaPolicy`). All the
``N`` idiosyncratic variables share the same law (but are independent). The
correlation between ``Y_i`` and ``Y_j`` is ``sum_k a_{i,k} a_{j,k}``.

This module carries the whole ``latentmodel.hpp`` header:

* :class:`LMIntegration` — the abstract integration interface;
* :class:`IntegrationBase` and its two specialisations, adapting
  ``GaussianQuadMultidimIntegrator`` and ``MultidimIntegral`` onto it;
* :class:`LatentModelIntegrationType` — the algorithm selector;
* :class:`LatentModel`, with the nested :class:`LatentModel.IntegrationFactory`
  and the three :class:`LatentModel.FactorSampler` forms.

Relationship to the credit latent models
----------------------------------------
This is the GENERAL math template. The W3-B credit cluster ports the
domain-specialised
:class:`~pquantlib.experimental.credit.default_probability_latent_model.DefaultProbabilityLatentModel`
and its loss-model relatives under ``experimental.credit`` — those are
specialisations of this template to the joint-default setting (the transfer
function ``F_i`` maps the latent variable to a default indicator coupled to a
``Basket``). This class keeps the copula-/factor-loading core without the
credit-domain bookkeeping.

# C++ parity divergence: the C++ class is a template ``LatentModel<Impl>``
# parameterised on the copula policy, and its random-sample machinery is a
# nested class template ``FactorSampler<USNG>`` with two partial
# specialisations. Python has neither, so the three C++ types become three
# sibling nested classes selected by the caller rather than by a template
# argument — which is what they are in C++ too, since specialisations of a
# class template share no base. The four C++ constructors (full matrix /
# single-factor vector / scalar correlation / Quote-handle) collapse to a
# single constructor plus the :meth:`LatentModel.single_factor` /
# :meth:`LatentModel.from_correlation` factory helpers.
"""

from __future__ import annotations

import copy
import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from enum import IntEnum
from typing import Protocol

import numpy as np

from pquantlib import qassert
from pquantlib.experimental.math.multidim_integrator import MultidimIntegral
from pquantlib.experimental.math.multidim_quadrature import GaussianQuadMultidimIntegrator
from pquantlib.experimental.math.polar_student_t_rng import PolarStudentTRng
from pquantlib.math.integrals.integrator import Integrator
from pquantlib.math.integrals.trapezoid import TrapezoidIntegral
from pquantlib.math.randomnumbers.box_muller_gaussian import BoxMullerGaussianRng
from pquantlib.math.randomnumbers.mersenne_twister import MersenneTwisterUniformRng
from pquantlib.math.randomnumbers.random_number_generator import SequenceSample
from pquantlib.math.randomnumbers.random_sequence_generator import RandomSequenceGenerator


def _multiply_v(d: float, v: Sequence[float]) -> list[float]:
    """Scale every component of ``v`` by ``d``.

    # C++ parity: ``struct QuantLib::detail::multiplyV``
    # (latentmodel.hpp:42-51) — "havent figured out how to do this in-place".
    """
    return [x * d for x in v]


class CopulaPolicy(Protocol):
    """Structural type satisfied by the Gaussian / Student-t copula policies."""

    def num_factors(self) -> int: ...
    def cumulative_y(self, val: float, i_variable: int) -> float: ...
    def cumulative_z(self, z: float) -> float: ...
    def inverse_cumulative_y(self, p: float, i_variable: int) -> float: ...
    def inverse_cumulative_z(self, p: float) -> float: ...
    def inverse_cumulative_density(self, p: float, i_factor: int) -> float: ...
    def density(self, m: Sequence[float]) -> float: ...
    def all_factor_cumul_inverter(self, probs: Sequence[float]) -> list[float]: ...


class TCopulaLike(Protocol):
    """The extra surface the Student-t factor sampler needs from its copula.

    # C++ parity: the ``TCopulaPolicy`` members the
    # ``FactorSampler<RandomSequenceGenerator<PolarStudentTRng<URNG> > >``
    # specialisation touches (latentmodel.hpp:775-786). Declared structurally
    # so this module does not have to import the (scipy-backed) policy.
    """

    def num_factors(self) -> int: ...
    def variance_factors(self) -> list[float]: ...


class UniformSequenceGeneratorLike(Protocol):
    """What the generic factor sampler needs from its ``USNG``."""

    def next_sequence(self) -> SequenceSample: ...


class LMIntegration(ABC):
    """Common interface over QuantLib's two unrelated integrator families.

    # C++ parity: ``class LMIntegration`` (latentmodel.hpp:63-84).

    The C++ comment explains why this exists: the grid integrators and the
    quadratures do not hang off a common base, and the factory below needs one
    pointer type for both.
    """

    __slots__ = ()

    @abstractmethod
    def integrate(self, f: Callable[[Sequence[float]], float]) -> float:
        """Integral of a scalar function of the factor vector.

        # C++ parity: the pure virtual ``integrate`` (latentmodel.hpp:67-68).
        """

    def integrate_v(self, f: Callable[[Sequence[float]], list[float]]) -> list[float]:
        """Integral of a vector function of the factor vector.

        # C++ parity: ``integrateV`` (latentmodel.hpp:78-83) — a virtual with
        # a failing default, so a facility that has no vector quadrature
        # simply does not override it. C++ needs the separate *name* because
        # MSVC could not resolve the overload; Python inherits the name.
        """
        qassert.fail("No vector integration provided")


class IntegrationBase(LMIntegration):
    """Primary template — always fully specialised.

    # C++ parity: ``template <class I_T> class IntegrationBase``
    # (latentmodel.hpp:87-93). The primary template has an empty body and a
    # ``private`` default constructor: "this class template always to be fully
    # specialized". The two C++ full specialisations are the two concrete
    # subclasses below; instantiating this class directly is refused, which is
    # what the private constructor achieves in C++.
    """

    __slots__ = ()

    def __init__(self) -> None:
        # C++ parity: ``private: IntegrationBase() = default;``.
        qassert.fail("IntegrationBase must be fully specialised")

    def integrate(self, f: Callable[[Sequence[float]], float]) -> float:
        """Unreachable — the primary template has no integrator."""
        qassert.fail("IntegrationBase must be fully specialised")


class LatentModelIntegrationType(IntEnum):
    """Which integration algorithm an integrable latent model should build.

    # C++ parity: ``namespace LatentModelIntegrationType { enum ... }``
    # (latentmodel.hpp:96-105). The C++ comment explains the namespace: gcc
    # reported a value collision with the Heston engine. Values are pinned
    # because ``IntegrationFactory`` switches on them.
    """

    GAUSSIAN_QUADRATURE = 0
    TRAPEZOID = 1


class GaussianQuadIntegrationBase(IntegrationBase, GaussianQuadMultidimIntegrator):
    """Gauss-Hermite multidimensional quadrature as an :class:`LMIntegration`.

    # C++ parity: ``template<> class IntegrationBase<
    # GaussianQuadMultidimIntegrator>`` (latentmodel.hpp:112-127). The C++
    # class inherits from both ``GaussianQuadMultidimIntegrator`` and
    # ``LMIntegration`` ("diamond on 'integrate'"), so it also exposes the
    # quadrature's own surface — ``order()`` in particular.

    Both directions of the vector integral carry the ``VectorIntegrator``
    leading-term defect documented in
    :mod:`pquantlib.experimental.math.multidim_quadrature`.
    """

    __slots__ = ()

    def __init__(self, dimension: int, order: int) -> None:
        # C++ parity: latentmodel.hpp:115-116.
        GaussianQuadMultidimIntegrator.__init__(self, dimension, order)

    def integrate(self, f: Callable[[Sequence[float]], float]) -> float:
        """# C++ parity: latentmodel.hpp:117-119."""
        return GaussianQuadMultidimIntegrator.integrate(self, f)

    def integrate_v(self, f: Callable[[Sequence[float]], list[float]]) -> list[float]:
        """# C++ parity: latentmodel.hpp:120-125."""
        return GaussianQuadMultidimIntegrator.integrate_v(self, f)


class MultidimIntegralIntegrationBase(IntegrationBase, MultidimIntegral):
    """Tensor-product grid integration as an :class:`LMIntegration`.

    # C++ parity: ``template<> class IntegrationBase<MultidimIntegral>``
    # (latentmodel.hpp:130-144). The integration domain is a *cube*: the
    # single ``a``/``b`` passed in are replicated across every axis.
    """

    __slots__ = ("a", "b")

    def __init__(self, integrators: Sequence[Integrator], a: float, b: float) -> None:
        # C++ parity: latentmodel.hpp:133-137.
        MultidimIntegral.__init__(self, integrators)
        #: # C++ parity: the public ``const std::vector<Real> a_, b_``
        #: (latentmodel.hpp:143) — public data members in C++, so public
        #: attributes here.
        self.a: list[float] = [a] * len(integrators)
        self.b: list[float] = [b] * len(integrators)

    def integrate(self, f: Callable[[Sequence[float]], float]) -> float:
        """# C++ parity: latentmodel.hpp:138-140."""
        return MultidimIntegral.__call__(self, f, self.a, self.b)

    # C++ parity note: ``integrateV`` is deliberately NOT overridden here —
    # "vector version here...." is an unfulfilled C++ to-do (latentmodel.hpp:
    # 141), so the trapezoid facility inherits LMIntegration's failing
    # default. Reproduced.


class LatentModel:
    """Generic multifactor latent variable model parameterised on a copula.

    # C++ parity: ``template <class copulaPolicyImpl> class LatentModel``
    # (latentmodel.hpp:277-611).

    :param factor_weights: the factor-loading matrix
        ``factor_weights[i_variable][i_factor]``.
    :param copula: the copula policy instance.
    :param integration: optional integration facility used by
        :meth:`integrated_expected_value`. C++ supplies this through the
        protected virtual ``integration()`` that integrable models override;
        Python takes it as a constructor argument.
    """

    # ---- nested: random factor sample generation --------------------------

    class FactorSampler:
        """Generic factor sampler: uniform sequence, then copula inversion.

        # C++ parity: ``LatentModel<...>::FactorSampler<USNG, bool>``
        # (latentmodel.hpp:403-437). The default implementation inverts the
        # copula's cumulative distributions, so ``USNG`` must be a *uniform*
        # sequence generator.

        The C++ template argument becomes ``usng_factory`` here: a callable
        ``(dimension, seed) -> sequence generator``, defaulting to
        ``RandomSequenceGenerator`` over the Mersenne Twister, which is the
        combination C++ users instantiate.
        """

        __slots__ = ("_copula", "_sequence_gen")

        def __init__(
            self,
            copula: CopulaPolicy,
            seed: int = 0,
            usng_factory: Callable[[int, int], UniformSequenceGeneratorLike] | None = None,
        ) -> None:
            # C++ parity: latentmodel.hpp:410-414.
            factory = usng_factory if usng_factory is not None else _default_usng_factory
            self._sequence_gen: UniformSequenceGeneratorLike = factory(
                copula.num_factors(), seed
            )
            self._copula = copula

        def next_sequence(self) -> SequenceSample:
            """One sample of the full factor set ``M_k, Z_i``.

            # C++ parity: ``nextSequence`` (latentmodel.hpp:425-430). Note the
            # C++ keeps ``x_.weight`` at the 1.0 it was constructed with and
            # discards the generator's own weight; reproduced.
            """
            sample = self._sequence_gen.next_sequence()
            values = self._copula.all_factor_cumul_inverter(
                [float(v) for v in sample.value]
            )
            return SequenceSample(value=np.asarray(values, dtype=np.float64), weight=1.0)

    class BoxMullerFactorSampler:
        """Direct Gaussian sampling, bypassing the copula inversion.

        # C++ parity: the partial specialisation ``LatentModel<TC>::
        # FactorSampler<RandomSequenceGenerator<BoxMullerGaussianRng<URNG> >,
        # dummy>`` (latentmodel.hpp:735-757).

        The C++ warning applies verbatim: QuantLib's Box-Muller is the
        rejection (polar) variant, so do not use this inside a multithreaded
        simulation. This is a sibling of, not a subclass of,
        :class:`LatentModel.FactorSampler` — the two are unrelated types in
        C++ as well, being distinct specialisations of one template.
        """

        __slots__ = ("_box_mull_rng",)

        def __init__(self, copula: CopulaPolicy, seed: int = 0) -> None:
            # C++ parity: latentmodel.hpp:745-748.
            self._box_mull_rng = RandomSequenceGenerator(
                copula.num_factors(),
                BoxMullerGaussianRng(MersenneTwisterUniformRng(seed)),
            )

        def next_sequence(self) -> SequenceSample:
            """# C++ parity: ``nextSequence`` (latentmodel.hpp:749-751)."""
            return self._box_mull_rng.next_sequence()

    class PolarStudentTFactorSampler:
        """Direct Student-t sampling, bypassing the copula inversion.

        # C++ parity: the partial specialisation ``LatentModel<TC>::
        # FactorSampler<RandomSequenceGenerator<PolarStudentTRng<URNG> >,
        # dummy>`` (latentmodel.hpp:766-800).

        # C++ parity note (DEFECT, reproduced verbatim): every per-factor
        # generator receives a **copy of the same just-seeded uniform RNG**
        # (``trng_.push_back(PolarStudentTRng<urng_type>(..., urng_))`` at
        # latentmodel.hpp:786, where ``PolarStudentTRng`` stores its generator
        # by value). All the factors therefore consume the *same* uniform
        # stream and come out perfectly dependent — deterministic monotone
        # functions of one another — rather than independent. This port makes
        # the copy explicitly, because the ported ``PolarStudentTRng`` stores
        # its uniform generator by reference.

        # C++ parity note: the degrees of freedom are recovered from the
        # copula's variance factors as ``2 / (1 - v^2)``; since
        # ``v = sqrt((nu-2)/nu)`` this is ``nu`` again, up to rounding. The
        # samples are therefore raw Student-t draws, NOT divided by ``v`` — so
        # unlike the generic sampler (which goes through
        # ``all_factor_cumul_inverter``) they are not normalised to unit
        # variance. Reproduced as-is.
        """

        __slots__ = ("_n_factors", "_trng")

        def __init__(self, copula: TCopulaLike, seed: int = 0) -> None:
            # C++ parity: latentmodel.hpp:775-787.
            self._n_factors = copula.num_factors()
            urng = MersenneTwisterUniformRng(seed)
            self._trng: list[PolarStudentTRng] = [
                PolarStudentTRng(2.0 / (1.0 - v * v), copy.deepcopy(urng))
                for v in copula.variance_factors()
            ]

        def next_sequence(self) -> SequenceSample:
            """# C++ parity: ``nextSequence`` (latentmodel.hpp:788-795).

            The first ``len(trng)`` entries (systemic factors plus one
            idiosyncratic) come from their own generator; every remaining
            idiosyncratic entry comes from the last one.
            """
            values: list[float] = [t.next().value for t in self._trng]
            last = self._trng[-1]
            while len(values) < self._n_factors:
                values.append(last.next().value)
            return SequenceSample(value=np.asarray(values, dtype=np.float64), weight=1.0)

    # ---- nested: integration algorithm selection --------------------------

    class IntegrationFactory:
        """Builds the integration facility an integrable latent model needs.

        # C++ parity: ``class LatentModel<...>::IntegrationFactory``
        # (latentmodel.hpp:439-486) — ``protected`` and with a ``private``
        # default constructor, i.e. a static factory that is never
        # instantiated. The C++ to-do note applies: the quadrature order and
        # the integration domain really belong to the copula, which is why
        # there is nominally one factory per copula policy.
        """

        __slots__ = ()

        def __init__(self) -> None:
            # C++ parity: ``private: IntegrationFactory() = default;``.
            qassert.fail("IntegrationFactory is a static factory")

        @staticmethod
        def create_lm_integration(
            dimension: int,
            integration_type: LatentModelIntegrationType = (
                LatentModelIntegrationType.GAUSSIAN_QUADRATURE
            ),
        ) -> LMIntegration:
            """# C++ parity: ``createLMIntegration`` (latentmodel.hpp:442-483).

            The parameters are hard-coded in C++ and pinned here: quadrature
            order 25, or one ``TrapezoidIntegral<Default>(1e-4, 20)`` per
            dimension over the cube ``[-35, 35]``.
            """
            if integration_type == LatentModelIntegrationType.GAUSSIAN_QUADRATURE:
                return GaussianQuadIntegrationBase(dimension, 25)
            if integration_type == LatentModelIntegrationType.TRAPEZOID:
                integrals: list[Integrator] = [
                    TrapezoidIntegral(1.0e-4, 20) for _ in range(dimension)
                ]
                # C++ parity note: "This integration domain is tailored for the
                # T distribution; it is too wide for normals or Ts of high
                # order." (latentmodel.hpp:469-477.)
                return MultidimIntegralIntegrationBase(integrals, -35.0, 35.0)
            qassert.fail("Unknown latent model integration type.")

    __slots__ = (
        "_copula",
        "_factor_weights",
        "_idiosync_fctrs",
        "_integration",
        "_n_factors",
        "_n_variables",
    )

    def __init__(
        self,
        factor_weights: Sequence[Sequence[float]],
        copula: CopulaPolicy,
        integration: LMIntegration | None = None,
    ) -> None:
        self._factor_weights = [list(row) for row in factor_weights]
        self._n_factors = len(self._factor_weights[0])
        self._n_variables = len(self._factor_weights)
        self._idiosync_fctrs: list[float] = []
        for i, row in enumerate(self._factor_weights):
            if len(row) != self._n_factors:
                msg = f"Name {i} provides a different number of factors"
                raise ValueError(msg)
            self._idiosync_fctrs.append(
                math.sqrt(1.0 - sum(w * w for w in row))
            )
        self._copula = copula
        self._integration = integration

    # ---- factory helpers (C++ single-factor / scalar constructors) ----

    @classmethod
    def single_factor(
        cls,
        factor_weights: Sequence[float],
        copula_factory: Callable[[Sequence[Sequence[float]]], CopulaPolicy],
        integration: LMIntegration | None = None,
    ) -> LatentModel:
        """One-factor model from a per-variable loading vector (column form)."""
        matrix = [[w] for w in factor_weights]
        return cls(matrix, copula_factory(matrix), integration)

    @classmethod
    def from_correlation(
        cls,
        correl_sqr: float,
        n_variables: int,
        copula_factory: Callable[[Sequence[Sequence[float]]], CopulaPolicy],
        integration: LMIntegration | None = None,
    ) -> LatentModel:
        """One-factor model with a single common-factor loading for all names."""
        matrix = [[correl_sqr] for _ in range(n_variables)]
        return cls(matrix, copula_factory(matrix), integration)

    # ---- inspectors ----

    def size(self) -> int:
        """Number of modelled (latent) variables."""
        return self._n_variables

    def num_factors(self) -> int:
        """Number of systemic factors."""
        return self._n_factors

    def num_total_factors(self) -> int:
        """Total factors = variables (idiosyncratic) + systemic factors."""
        return self._n_variables + self._n_factors

    def factor_weights(self) -> list[list[float]]:
        """The factor-loading matrix."""
        return self._factor_weights

    def idiosync_fctrs(self) -> list[float]:
        """The per-variable idiosyncratic loadings ``sqrt(1 - sum a^2)``."""
        return self._idiosync_fctrs

    def copula(self) -> CopulaPolicy:
        """The copula policy."""
        return self._copula

    def integration(self) -> LMIntegration:
        """The integration facility.

        # C++ parity: the protected virtual ``integration()``
        # (latentmodel.hpp:600-604), whose base implementation fails with
        # "Integration non implemented in Latent model."
        """
        if self._integration is None:
            msg = "no integration facility provided"
            raise ValueError(msg)
        return self._integration

    # ---- copula passthroughs ----

    def cumulative_y(self, val: float, i_variable: int) -> float:
        """Cumulative of latent variable ``i_variable`` at ``val``."""
        return self._copula.cumulative_y(val, i_variable)

    def cumulative_z(self, z: float) -> float:
        """Cumulative of the idiosyncratic factor at ``z``."""
        return self._copula.cumulative_z(z)

    def inverse_cumulative_y(self, p: float, i_variable: int) -> float:
        """Inverse cumulative of latent variable ``i_variable`` at ``p``."""
        return self._copula.inverse_cumulative_y(p, i_variable)

    def inverse_cumulative_z(self, p: float) -> float:
        """Inverse cumulative of the idiosyncratic factor at ``p``."""
        return self._copula.inverse_cumulative_z(p)

    def inverse_cumulative_density(self, p: float, i_factor: int) -> float:
        """Inverse cumulative of systemic factor ``i_factor`` at ``p``."""
        return self._copula.inverse_cumulative_density(p, i_factor)

    def all_factor_cumul_inverter(self, probs: Sequence[float]) -> list[float]:
        """Map a vector of uniforms to factor draws via the copula."""
        return self._copula.all_factor_cumul_inverter(probs)

    # ---- latent-variable evaluation ----

    def latent_var_value(self, all_factors: Sequence[float], i_var: int) -> float:
        """Evaluate ``Y_{i_var}`` from a full factor sample.

        ``all_factors`` holds all independent factors (systemic first, then the
        idiosyncratic factors), so the idiosyncratic value for variable ``i_var``
        is at index ``num_factors() + i_var``.
        """
        row = self._factor_weights[i_var]
        systemic = sum(row[k] * all_factors[k] for k in range(len(row)))
        idiosync = (
            all_factors[self._n_factors + i_var] * self._idiosync_fctrs[i_var]
        )
        return systemic + idiosync

    def latent_variable_correl(self, i_var1: int, i_var2: int) -> float:
        """Correlation between latent variables ``i_var1`` and ``i_var2``."""
        ini = (
            self._idiosync_fctrs[i_var1] * self._idiosync_fctrs[i_var1]
            if i_var1 == i_var2
            else 0.0
        )
        row1 = self._factor_weights[i_var1]
        row2 = self._factor_weights[i_var2]
        return ini + sum(row1[k] * row2[k] for k in range(len(row1)))

    # ---- integration ----

    def integrated_expected_value(
        self, f: Callable[[Sequence[float]], float]
    ) -> float:
        """``E[f(M)] = integral copula.density(x) * f(x) dx`` over the factors.

        # C++ parity: ``integratedExpectedValue`` (latentmodel.hpp:580-587) —
        # function composition: the integrand is composed with the density
        # through a product.
        """
        return self.integration().integrate(
            lambda x: self._copula.density(x) * f(x)
        )

    def integrated_expected_value_v(
        self, f: Callable[[Sequence[float]], list[float]]
    ) -> list[float]:
        """Vector-valued ``E[f(M)]`` over the factor density.

        # C++ parity: ``integratedExpectedValueV`` (latentmodel.hpp:588-597).
        Only the Gauss-Hermite facility implements vector integration; the
        trapezoid one fails with "No vector integration provided", as in C++.
        """
        return self.integration().integrate_v(
            lambda x: _multiply_v(self._copula.density(x), f(x))
        )


def _default_usng_factory(dimension: int, seed: int) -> UniformSequenceGeneratorLike:
    """``RandomSequenceGenerator<MersenneTwisterUniformRng>(dimension, seed)``."""
    return RandomSequenceGenerator.from_seed(dimension, seed)


__all__ = [
    "CopulaPolicy",
    "GaussianQuadIntegrationBase",
    "IntegrationBase",
    "LMIntegration",
    "LatentModel",
    "LatentModelIntegrationType",
    "MultidimIntegralIntegrationBase",
    "TCopulaLike",
]
