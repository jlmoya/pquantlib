"""CEVRNDCalculator — terminal density of a CEV process with absorption at 0.

# C++ parity: ql/methods/finitedifferences/utilities/cevrndcalculator.{hpp,cpp}
# (v1.43).

``df = alpha f^beta dW`` with an absorbing boundary at ``f = 0``. Writing
``X(f) = f^{2(1-beta)} / (alpha(1-beta))^2`` and ``delta = (1-2beta)/(1-beta)``,
the law of ``X(f_t)/t`` is a non-central chi-square, with the roles of
argument and non-centrality *swapped* between the ``delta < 2`` and
``delta >= 2`` regimes. That swap is the whole content of the class and is
reproduced literally.

Reference: D.R. Brecher, A.E. Lindsay, *Results on the CEV Process, Past and
Present*.

**Delegation, cross-validated.** C++ uses Boost's
``non_central_chi_squared_distribution`` and ``gamma_p``. Python delegates to
``scipy.stats.ncx2`` / ``scipy.special.gammainc``, and every branch — pdf,
cdf, ``massAtZero``, both ``invcdf`` regimes and the Sankaran-approximation
warm start — is pinned against the C++ probe
(``references/v143/methods/rnd.json``).
"""

from __future__ import annotations

import math
from typing import final

from scipy.special import gammainc  # pyright: ignore[reportMissingTypeStubs, reportUnknownVariableType]
from scipy.stats import ncx2  # pyright: ignore[reportMissingTypeStubs]

from pquantlib import qassert
from pquantlib.math.constants import QL_EPSILON
from pquantlib.math.distributions.inverse_cumulative_normal import (
    InverseCumulativeNormal,
)
from pquantlib.math.solvers1d.brent import Brent
from pquantlib.methods.finitedifferences.utilities.risk_neutral_density_calculator import (
    InvCDFHelper,
    RiskNeutralDensityCalculator,
)

_INV_CUM_NORMAL = InverseCumulativeNormal()


@final
class CEVRNDCalculator(RiskNeutralDensityCalculator):
    """Constant-elasticity-of-variance terminal density.

    # C++ parity: ``class CEVRNDCalculator : public RiskNeutralDensityCalculator``.
    """

    __slots__ = ("_alpha", "_beta", "_delta", "_f0", "_x0")

    def __init__(self, f0: float, alpha: float, beta: float) -> None:
        qassert.require(beta != 1.0, "beta can not be one")
        self._f0: float = f0
        self._alpha: float = alpha
        self._beta: float = beta
        self._delta: float = (1.0 - 2.0 * beta) / (1.0 - beta)
        self._x0: float = self._x(f0)

    # --- internal transforms -------------------------------------------

    def _x(self, f: float) -> float:
        """# C++ parity: ``CEVRNDCalculator::X``."""
        return math.pow(f, 2.0 * (1.0 - self._beta)) / (
            self._alpha * (1.0 - self._beta)
        ) ** 2

    def _inv_x(self, x: float) -> float:
        """# C++ parity: ``CEVRNDCalculator::invX``."""
        return math.pow(
            x * (self._alpha * (1.0 - self._beta)) ** 2,
            1.0 / (2.0 * (1.0 - self._beta)),
        )

    def _sankaran_approx(self, c: float, t: float, x: float) -> float:
        """# C++ parity: ``CEVRNDCalculator::sankaranApprox``."""
        a = self._x0 / t
        b = 2.0 - self._delta

        c = max(c, -0.45 * b)

        h = 1.0 - 2.0 * (b + c) * (b + 3.0 * c) / (3.0 * (b + 2.0 * c) ** 2)
        p = (b + 2.0 * c) / (b + c) ** 2
        m = (h - 1.0) * (1.0 - 3.0 * h)

        u = (
            math.pow(a / (b + c), h)
            - (1.0 + h * p * (h - 1.0 - 0.5 * (2.0 - h) * m * p))
        ) / (h * math.sqrt(2.0 * p) * (1.0 + 0.5 * m * p))

        return u - x

    # --- RiskNeutralDensityCalculator ----------------------------------

    def mass_at_zero(self, t: float) -> float:
        """# C++ parity: ``CEVRNDCalculator::massAtZero``."""
        if self._delta < 2.0:
            return 1.0 - float(gammainc(-0.5 * self._delta + 1.0, self._x0 / (2.0 * t)))
        return 0.0

    def pdf(self, f: float, t: float, /) -> float:
        """# C++ parity: ``CEVRNDCalculator::pdf``."""
        y = self._x(f)
        if self._delta < 2.0:
            base = float(ncx2.pdf(self._x0 / t, 4.0 - self._delta, y / t))  # pyright: ignore[reportUnknownMemberType]
            return base / t * 2.0 * (1.0 - self._beta) * y / f
        base = float(ncx2.pdf(y / t, self._delta, self._x0 / t))  # pyright: ignore[reportUnknownMemberType]
        return base / t * 2.0 * (self._beta - 1.0) * y / f

    def cdf(self, f: float, t: float, /) -> float:
        """# C++ parity: ``CEVRNDCalculator::cdf``."""
        y = self._x(f)
        if self._delta < 2.0:
            return 1.0 - float(ncx2.cdf(self._x0 / t, 2.0 - self._delta, y / t))  # pyright: ignore[reportUnknownMemberType]
        return 1.0 - float(ncx2.cdf(y / t, self._delta, self._x0 / t))  # pyright: ignore[reportUnknownMemberType]

    def invcdf(self, p: float, t: float, /) -> float:
        """# C++ parity: ``CEVRNDCalculator::invcdf``.

        For ``delta < 2`` C++ warm-starts Brent from the Sankaran normal
        approximation of the non-central chi-square quantile, then refines with
        ``InvCDFHelper``; if the warm start throws, it falls back to starting
        the refinement at ``f0``. Both paths are reproduced, including the
        bare ``catch (...)``.
        """
        if self._delta < 2.0:
            if self._f0 < QL_EPSILON or p < self.mass_at_zero(t):
                return 0.0

            x = _INV_CUM_NORMAL(1.0 - p)
            y0 = self._x(self._f0) / t

            try:
                brent = Brent()
                brent.set_max_evaluations(20)
                guess = self._inv_x(
                    brent.solve(
                        lambda c: self._sankaran_approx(c, t, x), 1e-8, y0, 0.02 * y0
                    )
                    * t
                )
                return InvCDFHelper(self, guess, 1e-8, 100).inverse_cdf(p, t)
            except Exception:
                return InvCDFHelper(self, self._f0, 1e-8, 100).inverse_cdf(p, t)
        x = t * float(ncx2.ppf(1.0 - p, self._delta, self._x0 / t))  # pyright: ignore[reportUnknownMemberType]
        return self._inv_x(x)


__all__ = ["CEVRNDCalculator"]
