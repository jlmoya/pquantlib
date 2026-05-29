"""HomogeneousPool — pool with uniform notional + uniform recovery.

# C++ parity: ql/experimental/credit/homogeneouspooldef.hpp (v1.42.1).

The C++ header defines ``HomogeneousPoolLossModel`` — a templated loss
model parameterised by a one-factor copula. That full loss-model class
templated on Gaussian / T-Student copulas belongs to the W3-B / W3-C
clusters (basket + correlation latent models + loss models).

This module ships the foundation-level abstraction: a ``HomogeneousPool``
attached to a base ``Pool`` that records the uniform notional and uniform
recovery rate that every issuer in the basket shares. Downstream loss-model
code can read these two scalars instead of indexing per-name arrays.
"""

from __future__ import annotations

from pquantlib.experimental.credit.pool import Pool


class HomogeneousPool(Pool):
    """Pool where every issuer shares one notional and one recovery rate.

    # C++ parity divergence: the C++ HomogeneousPoolLossModel is a
    # full loss-distribution computation primed with these scalars +
    # a copula. The Python foundation strips that out — the loss-model
    # computation lives in W3-B/C as ``HomogeneousPoolLossModel``
    # under ``pquantlib.experimental.credit.loss_models``.

    Adds two scalar attributes:

      - ``uniform_notional``: the notional each issuer contributes
        (e.g. 1.0 for a normalised basket).
      - ``uniform_recovery_rate``: the recovery rate applied uniformly
        across issuers on default.
    """

    __slots__ = ("_uniform_notional", "_uniform_recovery_rate")

    def __init__(
        self,
        uniform_notional: float = 1.0,
        uniform_recovery_rate: float = 0.4,
    ) -> None:
        super().__init__()
        self._uniform_notional = uniform_notional
        self._uniform_recovery_rate = uniform_recovery_rate

    def uniform_notional(self) -> float:
        """Common notional per issuer."""
        return self._uniform_notional

    def uniform_recovery_rate(self) -> float:
        """Common recovery rate applied across issuers."""
        return self._uniform_recovery_rate

    def notionals(self) -> list[float]:
        """Return per-name notionals (uniform for homogeneous pool)."""
        return [self._uniform_notional] * self.size()

    def recovery_rates(self) -> list[float]:
        """Return per-name recovery rates (uniform for homogeneous pool)."""
        return [self._uniform_recovery_rate] * self.size()
