"""BSMRNDCalculator — lognormal terminal density of a GBSM process.

# C++ parity: ql/methods/finitedifferences/utilities/bsmrndcalculator.{hpp,cpp}
# (v1.43).

The density is taken in ``x = ln(S)``: a normal with

    stdDev = blackVol(t, exp(x)) * sqrt(t)
    mean   = ln(S0) - stdDev^2/2 + ln(q(t)/r(t))

Note the volatility is read *at the strike ``exp(x)`` being evaluated*, so on
a skewed surface this is not a single fixed normal — mean and stddev both
move with ``x``. That is what C++ does and it is deliberate.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, final

from pquantlib.math.distributions.cumulative_normal_distribution import (
    CumulativeNormalDistribution,
)
from pquantlib.math.distributions.inverse_cumulative_normal import (
    InverseCumulativeNormal,
)
from pquantlib.math.distributions.normal_distribution import NormalDistribution
from pquantlib.methods.finitedifferences.utilities.risk_neutral_density_calculator import (
    RiskNeutralDensityCalculator,
)

if TYPE_CHECKING:
    from pquantlib.processes.generalized_black_scholes_process import (
        GeneralizedBlackScholesProcess,
    )


@final
class BSMRNDCalculator(RiskNeutralDensityCalculator):
    """Risk-neutral terminal density of ``ln(S)`` under a GBSM process.

    # C++ parity: ``class BSMRNDCalculator : public RiskNeutralDensityCalculator``.
    """

    __slots__ = ("_process",)

    def __init__(self, process: GeneralizedBlackScholesProcess) -> None:
        self._process: GeneralizedBlackScholesProcess = process

    def _distribution_params(self, x: float, t: float) -> tuple[float, float]:
        """# C++ parity: ``BSMRNDCalculator::distributionParams``."""
        std_dev = self._process.black_volatility().black_vol_at_time(t, math.exp(x)) * math.sqrt(t)
        mean = (
            math.log(self._process.x0())
            - 0.5 * std_dev * std_dev
            + math.log(
                self._process.dividend_yield().discount(t)
                / self._process.risk_free_rate().discount(t)
            )
        )
        return mean, std_dev

    def pdf(self, x: float, t: float, /) -> float:
        """# C++ parity: ``BSMRNDCalculator::pdf`` — ``x = ln(S)``."""
        mean, std_dev = self._distribution_params(x, t)
        return NormalDistribution(mean, std_dev)(x)

    def cdf(self, x: float, t: float, /) -> float:
        """# C++ parity: ``BSMRNDCalculator::cdf`` — ``x = ln(S)``."""
        mean, std_dev = self._distribution_params(x, t)
        return CumulativeNormalDistribution(mean, std_dev)(x)

    def invcdf(self, p: float, t: float, /) -> float:
        """# C++ parity: ``BSMRNDCalculator::invcdf``.

        C++ passes the *probability* into ``distributionParams`` as if it were
        an ``x``, so the volatility is read at strike ``exp(p)``. That looks
        like a slip, but it is v1.43 behaviour and the source of truth, so it
        is reproduced exactly — including the argument name ``x`` in the .cpp.
        """
        mean, std_dev = self._distribution_params(p, t)
        return InverseCumulativeNormal(mean, std_dev)(p)


__all__ = ["BSMRNDCalculator"]
