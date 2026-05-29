"""GaussianLHPLossModel — Vasicek Large Homogeneous Pool closed form.

# C++ parity: ql/experimental/credit/gaussianlhplossmodel.{hpp,cpp}
# @ v1.42.1 (099987f0).

The model implements the Vasicek 2002 / Kalemanova-Schmid-Werner (2007)
LHP analytical expected tranche loss formula. The underlying basket is
assumed to be:

  * large (so the unconditional default rate of the pool equals the
    per-name unconditional default prob — the law of large numbers
    applied to the conditional pool default rate)
  * homogeneous (uniform per-name notional and recovery rate; the model
    accepts an average recovery rate)
  * Gaussian one-factor (loadings ``sqrt(rho)`` on the common factor M).

Under these assumptions:

  ETL(K) = remainingNotional * { K phi(g(K)) - K' phi(g(K'))
                                  + (1-RR) [biphi(ip, -g(K)) - biphi(ip, -g(K'))] }

where:

  K, K' = min(1, attach/(1-RR)), min(1, detach/(1-RR))
  g(K)  = (Phi^-1(p) - sqrt(1-rho) Phi^-1(K)) / sqrt(rho)
  ip    = Phi^-1(p)

The percentile-loss-fraction has a similarly compact closed form via
the inverse Gaussian copula identity.
"""

from __future__ import annotations

import numpy as np

from pquantlib import qassert
from pquantlib.experimental.credit.default_loss_model import DefaultLossModel
from pquantlib.math.distributions.bivariate_normal_distribution import (
    BivariateCumulativeNormalDistribution,
)
from pquantlib.math.distributions.cumulative_normal_distribution import (
    CumulativeNormalDistribution,
)
from pquantlib.math.distributions.inverse_cumulative_normal import (
    InverseCumulativeNormal,
)

# QL_EPSILON in C++ — double precision epsilon — used by the LHP formula.
_QL_EPSILON: float = 2.2204460492503131e-16


class GaussianLHPLossModel(DefaultLossModel):
    """Vasicek large-homogeneous-pool Gaussian one-factor loss model.

    # C++ parity: gaussianlhplossmodel.hpp/.cpp.

    Constructor takes (correlation, recovery_rate). Both are scalars in
    the LHP setup (homogeneous pool); for a heterogeneous basket the
    caller computes the average recovery and passes it in.
    """

    __slots__ = ("_beta", "_correlation", "_phi", "_recovery_rate", "_sqrt1minus")

    def __init__(self, correlation: float, recovery_rate: float = 0.4) -> None:
        super().__init__()
        qassert.require(
            0.0 < correlation < 1.0,
            f"correlation {correlation} must be in (0, 1)",
        )
        qassert.require(
            0.0 <= recovery_rate <= 1.0,
            f"recovery_rate {recovery_rate} not in [0, 1]",
        )
        self._correlation = correlation
        self._recovery_rate = recovery_rate
        self._beta: float = float(np.sqrt(correlation))
        self._sqrt1minus: float = float(np.sqrt(1.0 - correlation))
        # # C++ parity: gaussianlhplossmodel.hpp:82 — biphi initialized
        # with rho = -beta = -sqrt(correlation).
        self._phi = CumulativeNormalDistribution()

    def correlation(self) -> float:
        return self._correlation

    def recovery_rate(self) -> float:
        return self._recovery_rate

    def expected_tranche_loss(
        self,
        remaining_notional: float,
        prob: float,
        average_rr: float,
        attach: float,
        detach: float,
    ) -> float:
        """Closed-form ETL in the (attach, detach) tranche.

        ``attach`` and ``detach`` are in basket-notional units (e.g.
        ``attach=0.03`` for a 3% attachment point). The model expects
        them as fractions of the *remaining* notional.

        # C++ parity: gaussianlhplossmodel.cpp:85-116
        # expectedTrancheLossImpl.
        """
        if attach >= detach:
            return 0.0
        if remaining_notional == 0.0:
            return 0.0
        # # C++ parity: gaussianlhplossmodel.cpp:97-101 — clip-to-one to
        # avoid the inverse CDF blowing up at K=1.
        one = 1.0 - 1.0e-12
        k1 = min(one, attach / (1.0 - average_rr)) + _QL_EPSILON
        k2 = min(one, detach / (1.0 - average_rr)) + _QL_EPSILON
        if prob > 0.0:
            ip = InverseCumulativeNormal.standard_value(prob)
            inv_k1 = InverseCumulativeNormal.standard_value(k1)
            inv_k2 = InverseCumulativeNormal.standard_value(k2)
            inv_flight_k1 = (ip - self._sqrt1minus * inv_k1) / self._beta
            inv_flight_k2 = (ip - self._sqrt1minus * inv_k2) / self._beta
            # The C++ biphi is constructed with rho = -beta;
            # BivariateCumulativeNormalDistribution(-beta).
            biphi = BivariateCumulativeNormalDistribution(-self._beta)
            return remaining_notional * (
                detach * self._phi(inv_flight_k2)
                - attach * self._phi(inv_flight_k1)
                + (1.0 - average_rr) * (
                    biphi(ip, -inv_flight_k2) - biphi(ip, -inv_flight_k1)
                )
            )
        return 0.0

    def percentile_portfolio_loss_fraction(
        self,
        prob: float,
        average_rr: float,
        perctl: float,
    ) -> float:
        """Untranched percentile loss as a fraction of the live portfolio.

        Closed-form Gaussian-copula percentile (Vasicek 2002, eq. 5):

          q(p_perctl) = (1 - RR) * Phi(
              (Phi^-1(p) + beta * Phi^-1(p_perctl)) / sqrt(1-rho)
          )

        # C++ parity: gaussianlhplossmodel.cpp:187-201
        # percentilePortfolioLossFraction.
        """
        qassert.require(
            0.0 <= perctl <= 1.0,
            f"perctl {perctl} out of range [0, 1]",
        )
        if perctl == 0.0:
            return 0.0
        if perctl == 1.0:
            perctl = 1.0 - _QL_EPSILON
        ip = InverseCumulativeNormal.standard_value(prob)
        inv_perctl = InverseCumulativeNormal.standard_value(perctl)
        return (1.0 - average_rr) * self._phi(
            (ip + self._beta * inv_perctl) / self._sqrt1minus
        )

    def percentile(
        self,
        remaining_notional: float,
        prob: float,
        average_rr: float,
        attach: float,
        detach: float,
        perctl: float,
    ) -> float:
        """Tranche-loss percentile.

        # C++ parity: gaussianlhplossmodel.hpp:142 percentile.
        """
        ptfl = self.percentile_portfolio_loss_fraction(prob, average_rr, perctl)
        return remaining_notional * min(max(ptfl - attach, 0.0), detach - attach)

    def prob_over_loss(
        self,
        remaining_notional: float,
        prob: float,
        average_rr: float,
        attach: float,
        detach: float,
        remaining_loss_fraction: float,
    ) -> float:
        """Probability of losing >= remaining_loss_fraction in the tranche.

        # C++ parity: gaussianlhplossmodel.cpp:118-154 probOverLoss.
        """
        qassert.require(
            0.0 <= remaining_loss_fraction <= 1.0,
            f"remaining_loss_fraction {remaining_loss_fraction} out of range [0, 1]",
        )
        _ = remaining_notional
        # # C++ parity: gaussianlhplossmodel.cpp:134-135.
        ptfl_fract = attach + remaining_loss_fraction * (detach - attach)
        max_loss_fract = 1.0 - average_rr
        if ptfl_fract > max_loss_fract:
            return 0.0
        if ptfl_fract <= _QL_EPSILON:
            return 1.0
        ip = InverseCumulativeNormal.standard_value(prob)
        inv_flight_k = (
            ip
            - self._sqrt1minus
            * InverseCumulativeNormal.standard_value(
                ptfl_fract / (1.0 - average_rr)
            )
        ) / self._beta
        return float(self._phi(inv_flight_k))

    def expected_shortfall(
        self,
        remaining_notional: float,
        prob: float,
        average_rr: float,
        attach: float,
        detach: float,
        perctl: float,
    ) -> float:
        """Expected shortfall at ``perctl``.

        # C++ parity: gaussianlhplossmodel.cpp:156-185 expectedShortfall.
        """
        ptfl_loss_perc = self.percentile_portfolio_loss_fraction(
            prob, average_rr, perctl
        )
        if ptfl_loss_perc >= detach - _QL_EPSILON:
            return remaining_notional * (detach - attach)
        max_loss_level = max(attach, ptfl_loss_perc)
        val_a = self.expected_tranche_loss(
            remaining_notional, prob, average_rr, max_loss_level, detach
        )
        val_b = self.prob_over_loss(
            remaining_notional,
            prob,
            average_rr,
            attach,
            detach,
            min(max((max_loss_level - attach) / (detach - attach), 0.0), 1.0),
        )
        return (val_a + (max_loss_level - attach) * remaining_notional * val_b) / (
            1.0 - perctl
        )
