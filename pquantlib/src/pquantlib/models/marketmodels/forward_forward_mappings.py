"""ForwardForwardMappings — forward-tenor-to-forward-tenor mapping helpers.

# C++ parity: ql/models/marketmodels/forwardforwardmappings.{hpp,cpp} (v1.42.1).

Static utilities mapping between forward rates of a long tenor (a multiple of
the base tenor) and the base-tenor forward rates:

- ``forward_forward_jacobian`` — ``dg[i]/df[j]`` jacobian.
- ``y_matrix`` — Y-matrix switching base between long- and short-tenor
  forwards.
- ``restrict_curve_state`` — build an ``LMMCurveState`` on the periodic
  subset of times.

C++ exposes these as a ``namespace``; pquantlib mirrors ``SwapForwardMappings``
and wraps them in a ``ForwardForwardMappings`` class with static methods.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pquantlib import qassert
from pquantlib.models.marketmodels.curvestates.lmm_curve_state import LMMCurveState

if TYPE_CHECKING:
    from pquantlib.math.matrix import Matrix
    from pquantlib.models.marketmodels.curve_state import CurveState


class ForwardForwardMappings:
    """Forward-tenor <-> forward-tenor mapping helpers (all static).

    # C++ parity: forwardforwardmappings.hpp ForwardForwardMappings (namespace).
    """

    @staticmethod
    def forward_forward_jacobian(cs: CurveState, multiplier: int, offset: int) -> Matrix:
        """``dg[i]/df[j]`` jacobian: long-tenor vs base-tenor forwards.

        # C++ parity: forwardforwardmappings.cpp
        ForwardForwardMappings::ForwardForwardJacobian.
        """
        n = cs.number_of_rates()
        qassert.require(
            offset < multiplier,
            "offset  must be less than period in  forward forward mappings",
        )
        k = (n - offset) // multiplier
        tau = cs.rate_taus()
        jacobian = np.zeros((k, n), dtype=np.float64)
        m = offset
        for ell in range(k):
            df = cs.discount_ratio(m, m + multiplier)
            big_tau = cs.rate_times()[m + multiplier] - cs.rate_times()[m]
            for _r in range(multiplier):
                value = df * tau[m] * cs.discount_ratio(m + 1, m) - 1
                value /= big_tau
                jacobian[ell, m] = -value
                m += 1
        return jacobian

    @staticmethod
    def y_matrix(
        cs: CurveState,
        short_displacements: list[float],
        long_displacements: list[float],
        multiplier: int,
        offset: int,
    ) -> Matrix:
        """Y-matrix switching base between long- and short-tenor forwards.

        # C++ parity: forwardforwardmappings.cpp ForwardForwardMappings::YMatrix.
        """
        n = cs.number_of_rates()
        qassert.require(
            offset < multiplier,
            "offset  must be less than period in  forward forward mappings",
        )
        k = (n - offset) // multiplier
        qassert.require(
            len(short_displacements) == n,
            "shortDisplacements must be of size equal to number of rates",
        )
        qassert.require(
            len(long_displacements) == k,
            "longDisplacements must be of size equal to (number of rates minus offset) "
            "divided by multiplier",
        )
        jacobian = ForwardForwardMappings.forward_forward_jacobian(cs, multiplier, offset)
        for i in range(k):
            tau = (
                cs.rate_times()[(i + 1) * multiplier + offset]
                - cs.rate_times()[i * multiplier + offset]
            )
            long_forward = (
                cs.discount_ratio((i + 1) * multiplier + offset, i * multiplier + offset)
                - 1.0
            ) / tau
            long_forward_displaced = long_forward + long_displacements[i]
            for j in range(n):
                short_forward = cs.forward_rate(j)
                short_forward_displaced = short_forward + short_displacements[j]
                jacobian[i, j] *= short_forward_displaced / long_forward_displaced
        return jacobian

    @staticmethod
    def restrict_curve_state(cs: CurveState, multiplier: int, offset: int) -> LMMCurveState:
        """Build an ``LMMCurveState`` on the periodic subset of times.

        # C++ parity: forwardforwardmappings.cpp
        ForwardForwardMappings::RestrictCurveState.
        """
        n = cs.number_of_rates()
        qassert.require(
            offset < multiplier,
            "offset  must be less than period in  forward forward mappings",
        )
        k = (n - offset) // multiplier
        times = [0.0] * (k + 1)
        disc_ratios = [0.0] * (k + 1)
        for i in range(k + 1):
            times[i] = cs.rate_times()[i * multiplier + offset]
            disc_ratios[i] = cs.discount_ratio(i * multiplier + offset, 0)
        new_state = LMMCurveState(times)
        new_state.set_on_discount_ratios(disc_ratios)
        return new_state
