"""LatentModel — the generic multifactor latent variable model.

# C++ parity: ql/experimental/math/latentmodel.hpp @ v1.42.1 (099987f0).

The latent model describes a set of latent (random) variables

    Y_i = sum_k a_{i,k} M_k + sqrt(1 - sum_k a_{i,k}^2) Z_i

where the systemic factors ``M_k`` and the idiosyncratic factors ``Z_i`` have
independent zero-mean unit-variance distributions fixed by a *copula policy*
(:class:`~pquantlib.experimental.math.gaussian_copula_policy.GaussianCopulaPolicy`
or :class:`~pquantlib.experimental.math.t_copula_policy.TCopulaPolicy`). All the
``N`` idiosyncratic variables share the same law (but are independent). The
correlation between ``Y_i`` and ``Y_j`` is ``sum_k a_{i,k} a_{j,k}``.

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

# C++ parity divergence: the C++ class is a template ``LatentModel<Impl>`` whose
# ``Impl`` supplies the random-sample machinery (the ``FactorSampler`` partial
# specialisations). The Python port keeps the *integration / inspector* surface
# that every downstream user touches (factor loadings, per-variable
# correlation, cumulative/inverse copula queries, latent-variable evaluation,
# and ``integrated_expected_value``); the random-sampling
# ``FactorSampler`` specialisations are a deliberate carve-out (the copula RNGs
# in this same package provide direct sample generation when needed). The four
# C++ constructors (full matrix / single-factor vector / scalar correlation /
# Quote-handle) collapse to a single constructor plus the
# :meth:`single_factor` / :meth:`from_correlation` factory helpers.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Protocol


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


class LMIntegration(Protocol):
    """Structural type for a latent-model integration facility."""

    def integrate(self, f: Callable[[Sequence[float]], float]) -> float: ...


class LatentModel:
    """Generic multifactor latent variable model parameterised on a copula.

    :param factor_weights: the factor-loading matrix
        ``factor_weights[i_variable][i_factor]``.
    :param copula: the copula policy instance.
    :param integration: optional integration facility used by
        :meth:`integrated_expected_value`.
    """

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

        Requires an integration facility to have been supplied at construction.
        """
        if self._integration is None:
            msg = "no integration facility provided"
            raise ValueError(msg)
        return self._integration.integrate(
            lambda x: self._copula.density(x) * f(x)
        )
