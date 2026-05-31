"""Concrete alpha(t) forms.

# C++ parity: ql/models/marketmodels/models/alphaformconcrete.{hpp,cpp}
# (v1.42.1).

Two parametric forms used by the coterminal-swap-market-model caplet
calibration:

* ``AlphaFormInverseLinear``: ``1 / (1 + alpha*t)``.
* ``AlphaFormLinearHyperbolic``: ``sqrt(1 + a*t*(atan(a*t) - pi/2))`` with
  ``a = alpha``.
"""

from __future__ import annotations

import math

from pquantlib.models.marketmodels.models.alpha_form import AlphaForm


class AlphaFormInverseLinear(AlphaForm):
    """``alpha(i) = 1 / (1 + alpha*t_i)``.

    # C++ parity: AlphaFormInverseLinear.
    """

    def __init__(self, times: list[float], alpha: float = 0.0) -> None:
        # C++ parity: AlphaFormInverseLinear(std::vector<Time> times, Real alpha).
        self._times = list(times)
        self._alpha = alpha

    def __call__(self, i: int) -> float:
        # C++ parity: 1.0/(1.0+alpha_*times_[i]).
        return 1.0 / (1.0 + self._alpha * self._times[i])

    def set_alpha(self, alpha: float) -> None:
        self._alpha = alpha


class AlphaFormLinearHyperbolic(AlphaForm):
    """``alpha(i) = sqrt(1 + a*t_i*(atan(a*t_i) - pi/2))`` with ``a = alpha``.

    # C++ parity: AlphaFormLinearHyperbolic.
    """

    def __init__(self, times: list[float], alpha: float = 0.0) -> None:
        # C++ parity: AlphaFormLinearHyperbolic(std::vector<Time> times, Real alpha).
        self._times = list(times)
        self._alpha = alpha

    def __call__(self, i: int) -> float:
        # C++ parity: alphaformconcrete.cpp AlphaFormLinearHyperbolic::operator():
        #   at = alpha*times[i]; res = atan(at)-0.5*pi; res *= at; res += 1;
        #   res = sqrt(res).
        at = self._alpha * self._times[i]  # C++ parity: `at`
        res = math.atan(at) - 0.5 * math.pi
        res *= at
        res += 1.0
        return math.sqrt(res)

    def set_alpha(self, alpha: float) -> None:
        self._alpha = alpha
