"""GaussianCopulaPolicy — Gaussian copula policy for the latent model.

# C++ parity: ql/experimental/math/gaussiancopulapolicy.hpp @ v1.42.1 (099987f0).

The Gaussian copula policy is the simplest copula policy a
:class:`~pquantlib.experimental.math.latent_model.LatentModel` can take. The
convolution stability of the Gaussian distribution makes every cumulative /
inverse-cumulative query a plain standard-normal lookup, independent of the
factor structure: the latent variable, the idiosyncratic factor and each
systemic factor all follow the unit normal law.

The policy stores the per-name factor-weight matrix only to validate that each
row is normalised (sum of squares < 1) and to report the total number of
independent factors (``len(rows) + len(rows[0])`` — systemic + per-name
idiosyncratic). The C++ class caches a single static ``NormalDistribution`` /
``CumulativeNormalDistribution``; the Python port uses the module-level
distribution objects directly.
"""

from __future__ import annotations

from collections.abc import Sequence

from pquantlib import qassert
from pquantlib.math.distributions.cumulative_normal_distribution import (
    CumulativeNormalDistribution,
)
from pquantlib.math.distributions.inverse_cumulative_normal import (
    InverseCumulativeNormal,
)
from pquantlib.math.distributions.normal_distribution import NormalDistribution

_density = NormalDistribution()
_cumulative = CumulativeNormalDistribution()


class GaussianCopulaPolicy:
    """Gaussian copula policy.

    :param factor_weights: per-name factor-loading matrix
        ``factor_weights[i_variable][i_factor]``. Each row must satisfy
        ``sum(w_k^2) < 1`` (a convex / normalised combination).
    """

    __slots__ = ("_num_factors",)

    def __init__(
        self, factor_weights: Sequence[Sequence[float]] | None = None
    ) -> None:
        if factor_weights is None or len(factor_weights) == 0:
            # C++ parity: the default ctor indexes factorWeights[0]; an empty
            # matrix is therefore not a valid runtime state. We mirror the
            # "rows + columns" factor count with a zero-factor empty model.
            self._num_factors = 0
            return
        # numFactors_ = factorWeights.size() + factorWeights[0].size()
        self._num_factors = len(factor_weights) + len(factor_weights[0])
        for factor_weight in factor_weights:
            factors_norm = sum(w * w for w in factor_weight)
            qassert.require(
                factors_norm < 1.0, "Non normal random factor combination."
            )

    def num_factors(self) -> int:
        """Number of independent random factors (systemic + idiosyncratic)."""
        return self._num_factors

    # ---- cumulative / inverse on the latent variable ----

    def cumulative_y(self, val: float, i_variable: int) -> float:
        """Cumulative probability of latent variable ``i_variable`` at ``val``."""
        return _cumulative(val)

    def cumulative_z(self, z: float) -> float:
        """Cumulative probability of the (common-law) idiosyncratic factor."""
        return _cumulative(z)

    def density(self, m: Sequence[float]) -> float:
        """Joint density of the independent systemic factor realisations ``m``."""
        prod = 1.0
        for y in m:
            prod *= _density(y)
        return prod

    def inverse_cumulative_y(self, p: float, i_variable: int) -> float:
        """Inverse cumulative of latent variable ``i_variable`` at probability ``p``."""
        return InverseCumulativeNormal.standard_value(p)

    def inverse_cumulative_z(self, p: float) -> float:
        """Inverse cumulative of the idiosyncratic factor at probability ``p``."""
        return InverseCumulativeNormal.standard_value(p)

    def inverse_cumulative_density(self, p: float, i_factor: int) -> float:
        """Inverse cumulative of systemic factor ``i_factor`` at probability ``p``."""
        return InverseCumulativeNormal.standard_value(p)

    def all_factor_cumul_inverter(self, probs: Sequence[float]) -> list[float]:
        """Map a vector of uniforms to standard-normal factor draws."""
        return [InverseCumulativeNormal.standard_value(p) for p in probs]
