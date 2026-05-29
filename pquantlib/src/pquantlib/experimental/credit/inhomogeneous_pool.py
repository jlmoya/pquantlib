"""InhomogeneousPool — pool with per-issuer notionals + recoveries.

# C++ parity: ql/experimental/credit/inhomogeneouspooldef.hpp (v1.42.1).

Like ``HomogeneousPool``, the C++ header defines a templated loss model
parameterised by a one-factor copula. The full loss-model class lives in
the W3-B / W3-C cluster scope (basket + correlation latent models +
loss models).

This module ships the foundation-level abstraction: an
``InhomogeneousPool`` recording per-issuer notional + per-issuer recovery
rate. Downstream loss-model code reads these arrays to size buckets and
weight the convolution.
"""

from __future__ import annotations

from pquantlib.experimental.credit.default_probability_key import DefaultProbKey
from pquantlib.experimental.credit.issuer import Issuer
from pquantlib.experimental.credit.pool import Pool


class InhomogeneousPool(Pool):
    """Pool where each issuer carries its own notional + recovery rate.

    The two arrays (``notionals`` and ``recovery_rates``) must be kept
    in sync with ``names()``. They grow when ``add_with_attributes`` is
    used; ``add`` (inherited from ``Pool``) requires the caller to push
    matching entries explicitly.

    # C++ parity divergence: the C++ InhomogeneousPoolLossModel is a
    # full loss-distribution computation. The Python foundation strips
    # that out — the loss-model computation lives in W3-B/C as
    # ``InhomogeneousPoolLossModel``.
    """

    __slots__ = ("_notionals", "_recovery_rates")

    def __init__(self) -> None:
        super().__init__()
        self._notionals: list[float] = []
        self._recovery_rates: list[float] = []

    def notionals(self) -> list[float]:
        return list(self._notionals)

    def recovery_rates(self) -> list[float]:
        return list(self._recovery_rates)

    def add_with_attributes(
        self,
        name: str,
        issuer: Issuer,
        notional: float,
        recovery_rate: float,
        contract_trigger: DefaultProbKey | None = None,
    ) -> None:
        """Add an issuer with explicit notional + recovery.

        If the name is already present, this is a no-op (mirrors
        ``Pool.add``). Otherwise the issuer is registered AND the
        parallel attribute arrays are extended.
        """
        already_present = self.has(name)
        super().add(name, issuer, contract_trigger)
        if not already_present:
            self._notionals.append(notional)
            self._recovery_rates.append(recovery_rate)

    def clear(self) -> None:
        super().clear()
        self._notionals.clear()
        self._recovery_rates.clear()
