"""ConstantLossLatentModel — deterministic per-issuer LGD on top of the
default-probability latent model.

# C++ parity: ql/experimental/credit/constantlosslatentmodel.hpp @ v1.42.1.

The C++ template adds a fixed per-issuer recovery vector to
``DefaultLatentModel<CP>``; the Python port keeps the same surface but
takes the recovery vector directly.

The conditional recovery is deterministic — the loss given default is
simply ``1 - recoveries_[i]`` regardless of the latent factor draw.
This model serves both as a foundation for the binomial / recursive /
saddlepoint loss models and as a stand-alone model for NTD pricing.
"""

from __future__ import annotations

from collections.abc import Sequence

from pquantlib import qassert
from pquantlib.experimental.credit.default_probability_latent_model import (
    DefaultProbabilityLatentModel,
)
from pquantlib.experimental.credit.one_factor_copula import OneFactorCopula


class ConstantLossLatentModel(DefaultProbabilityLatentModel):
    """Default-prob latent model plus a per-name constant recovery vector.

    # C++ parity: constantlosslatentmodel.hpp ConstantLossLatentmodel.
    The C++ template stays single-class for both the loss-latent-model
    and the loss-model interface; the Python port keeps the surface
    minimal — downstream loss models (Binomial / Recursive / Saddlepoint)
    consume ``recoveries()`` directly.
    """

    __slots__ = ("_recoveries",)

    def __init__(
        self,
        copula: OneFactorCopula,
        recoveries: Sequence[float],
    ) -> None:
        super().__init__(copula, pool_size=len(recoveries))
        qassert.require(
            len(recoveries) > 0,
            "recoveries must be non-empty",
        )
        for i, r in enumerate(recoveries):
            qassert.require(
                0.0 <= r <= 1.0,
                f"recoveries[{i}] = {r} not in [0, 1]",
            )
        self._recoveries = list(recoveries)

    def recoveries(self) -> list[float]:
        """Return a copy of the per-name recovery vector."""
        return list(self._recoveries)

    def recovery(self, i_name: int) -> float:
        qassert.require(
            0 <= i_name < len(self._recoveries),
            f"i_name {i_name} out of range",
        )
        return self._recoveries[i_name]

    def conditional_recovery(
        self,
        i_name: int,
        m: float | None = None,  # kept for signature parity with C++
    ) -> float:
        """Constant LGD model — recovery is the stored value regardless of ``m``.

        # C++ parity: constantlosslatentmodel.hpp:72-90.
        """
        _ = m  # the constant model ignores the factor draw
        return self.recovery(i_name)

    def expected_loss(self, i_name: int, prob: float) -> float:
        """Single-name expected loss at unconditional default probability ``prob``.

        Reproduces ``prob * (1 - recoveries_[i])`` up to integration noise
        and provides a sanity check on the latent-model implementation.

        # C++ parity: defaultprobabilitylatentmodel.hpp:probOfDefault times
        # the constant LGD = 1 - recoveries_[i].
        """
        return self.prob_of_default(i_name, prob) * (
            1.0 - self.recovery(i_name)
        )
