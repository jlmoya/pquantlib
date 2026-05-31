"""TCopulaPolicy — Student-t copula policy for the latent model.

# C++ parity: ql/experimental/math/tcopulapolicy.{hpp,cpp} @ v1.42.1 (099987f0).

The Student-t copula policy describes a set of normalised independent Student-t
factors fed into a :class:`~pquantlib.experimental.math.latent_model.LatentModel`.
The latent model requires unit-variance independent variables, so the policy
normalises each T variable by the square root of its variance ``nu/(nu-2)`` (the
``varianceFactors_`` array stores ``sqrt((nu-2)/nu)``).

Unlike the Gaussian policy, the cumulative / inverse of a *latent* variable
involves the convolution of the factor distributions; this is handled by
:class:`~pquantlib.experimental.math.convolved_student_t.CumulativeBehrensFisher`
and its inverse. The idiosyncratic and per-systemic-factor queries use the plain
Student-t distribution (via :mod:`scipy.stats`).

The ``init_traits`` is the list of T orders (degrees of freedom). It must have
length ``num_factors + 1`` (one per systemic factor plus the shared
idiosyncratic factor), and every order must be ``> 2`` (finite variance) and
odd (required by the Behrens-Fisher convolution).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from scipy.stats import t as _student_t  # pyright: ignore[reportMissingTypeStubs, reportUnknownVariableType]

from pquantlib import qassert
from pquantlib.experimental.math.convolved_student_t import (
    CumulativeBehrensFisher,
    InverseCumulativeBehrensFisher,
)


class TCopulaPolicy:
    """Student-t copula policy.

    :param factor_weights: per-name factor-loading matrix
        ``factor_weights[i_variable][i_factor]``.
    :param t_orders: degrees of freedom; length ``num_factors + 1``, each odd
        and ``> 2``.
    """

    __slots__ = (
        "_distributions_df",
        "_latent_cumul",
        "_latent_inverters",
        "_variance_factors",
    )

    def __init__(
        self,
        factor_weights: Sequence[Sequence[float]] | None = None,
        t_orders: Sequence[int] | None = None,
    ) -> None:
        weights = list(factor_weights) if factor_weights is not None else []
        orders = list(t_orders) if t_orders is not None else []

        self._distributions_df: list[int] = []
        self._variance_factors: list[float] = []
        for t_order in orders:
            qassert.require(
                t_order > 2, "Non finite variance T in latent model."
            )
            self._distributions_df.append(t_order)
            self._variance_factors.append(math.sqrt((t_order - 2.0) / t_order))

        self._latent_cumul: list[CumulativeBehrensFisher] = []
        self._latent_inverters: list[InverseCumulativeBehrensFisher] = []
        for factor_weight in weights:
            qassert.require(
                len(orders) == len(factor_weight) + 1,
                "Incompatible number of T functions and number of factors.",
            )
            factors_norm = sum(w * w for w in factor_weight)
            qassert.require(
                factors_norm < 1.0, "Non normal random factor combination."
            )
            idiosync_fctr = math.sqrt(1.0 - factors_norm)
            norm_factor_weights = [
                factor_weight[i] * self._variance_factors[i]
                for i in range(len(factor_weight))
            ]
            norm_factor_weights.append(idiosync_fctr * self._variance_factors[-1])
            self._latent_cumul.append(
                CumulativeBehrensFisher(orders, norm_factor_weights)
            )
            self._latent_inverters.append(
                InverseCumulativeBehrensFisher(orders, norm_factor_weights)
            )

    def num_factors(self) -> int:
        """Number of independent random factors (systemic + idiosyncratic)."""
        return len(self._latent_inverters) + len(self._variance_factors) - 1

    def variance_factors(self) -> list[float]:
        """The per-factor variance-normalisation factors ``sqrt((nu-2)/nu)``."""
        return self._variance_factors

    # ---- cumulative / inverse on the latent variable ----

    def cumulative_y(self, val: float, i_variable: int) -> float:
        """Cumulative of latent variable ``i_variable`` (Behrens-Fisher)."""
        return self._latent_cumul[i_variable](val)

    def cumulative_z(self, z: float) -> float:
        """Cumulative of the idiosyncratic factor (normalised Student-t)."""
        df = self._distributions_df[-1]
        return float(_student_t.cdf(z / self._variance_factors[-1], df))  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]

    def density(self, m: Sequence[float]) -> float:
        """Joint density of the independent systemic-factor realisations ``m``."""
        prod = 1.0
        for i in range(len(m)):
            vf = self._variance_factors[i]
            pdf = float(_student_t.pdf(m[i] / vf, self._distributions_df[i]))  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]
            prod *= pdf / vf
        return prod

    def inverse_cumulative_y(self, p: float, i_variable: int) -> float:
        """Inverse cumulative of latent variable ``i_variable`` (root-solved)."""
        return self._latent_inverters[i_variable](p)

    def inverse_cumulative_z(self, p: float) -> float:
        """Inverse cumulative of the idiosyncratic factor."""
        df = self._distributions_df[-1]
        return float(_student_t.ppf(p, df)) * self._variance_factors[-1]  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]

    def inverse_cumulative_density(self, p: float, i_factor: int) -> float:
        """Inverse cumulative of systemic factor ``i_factor``."""
        df = self._distributions_df[i_factor]
        return float(_student_t.ppf(p, df)) * self._variance_factors[i_factor]  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]

    def all_factor_cumul_inverter(self, probs: Sequence[float]) -> list[float]:
        """Map a vector of uniforms to normalised Student-t factor draws."""
        n_systemic = len(self._variance_factors) - 1
        result: list[float] = []
        for i in range(n_systemic):
            result.append(self.inverse_cumulative_density(probs[i], i))
        for i in range(n_systemic, len(probs)):
            result.append(self.inverse_cumulative_z(probs[i]))
        return result
