"""RiskyAssetSwapOption — option on a risky asset-swap.

# C++ parity: ql/experimental/credit/riskyassetswapoption.{hpp,cpp} (v1.42.1).

The option value is computed in a Bachelier-normal-model fashion on
the spread:

* Let ``w = -1`` if the underlying is fixed-payer (asset-swap call =
  spread put) and ``w = +1`` otherwise.
* ``d = (asw.spread() - market_spread) / stdDev`` where
  ``stdDev = spread_volatility * sqrt(T)``, ``T`` is the year fraction
  from today to ``expiry`` in Actual/365 Fixed (matches C++).
* ``A0 = nominal * float_annuity`` of the underlying.
* ``NPV = A0 * stdDev * (w*d * Phi(w*d) + phi(d))``.

Where ``Phi`` is the cumulative normal distribution and ``phi`` is the
normal density.
"""

from __future__ import annotations

import math

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.experimental.credit.risky_asset_swap import RiskyAssetSwap
from pquantlib.instruments.instrument import Instrument
from pquantlib.math.distributions.cumulative_normal_distribution import (
    CumulativeNormalDistribution,
)
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.time.date import Date

_PHI: CumulativeNormalDistribution = CumulativeNormalDistribution()


def _normal_density(x: float) -> float:
    """Standard normal PDF.

    # C++ parity: ``NormalDistribution()(x)`` with mean=0, sigma=1.
    """
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


class RiskyAssetSwapOption(Instrument):
    """Option on a risky asset-swap.

    # C++ parity: ``RiskyAssetSwapOption`` class.

    Construction binds the option to a risky asset-swap, an expiry
    date, a market spread, and a spread volatility. The option NPV
    is computed by the closed-form Bachelier-spread formula.
    """

    def __init__(
        self,
        underlying: RiskyAssetSwap,
        expiry: Date,
        market_spread: float,
        spread_volatility: float,
    ) -> None:
        super().__init__()
        self._asw: RiskyAssetSwap = underlying
        self._expiry: Date = expiry
        self._market_spread: float = market_spread
        self._spread_volatility: float = spread_volatility

    # ---- Instrument interface --------------------------------------

    def is_expired(self) -> bool:
        """The expiry date is in the past.

        # C++ parity: riskyassetswapoption.cpp:34-36.
        """
        today = ObservableSettings().evaluation_date_or_today()
        return self._expiry <= today

    def perform_calculations(self) -> None:
        """Closed-form Bachelier spread-option value.

        # C++ parity: riskyassetswapoption.cpp:39-54.
        """
        w = -1.0 if self._asw.fixed_payer() else 1.0
        today = ObservableSettings().evaluation_date_or_today()
        expiry_time = Actual365Fixed().year_fraction(today, self._expiry)
        std_dev = self._spread_volatility * math.sqrt(expiry_time)
        d = (self._asw.spread() - self._market_spread) / std_dev
        a0 = self._asw.nominal() * self._asw.float_annuity()

        self._npv = a0 * std_dev * (
            w * d * _PHI(w * d) + _normal_density(d)
        )

    def _perform_calculations(self) -> None:
        self.perform_calculations()


__all__ = ["RiskyAssetSwapOption"]
