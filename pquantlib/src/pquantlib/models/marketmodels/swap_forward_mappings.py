"""SwapForwardMappings — swap<->forward jacobian + Z-matrix helpers.

# C++ parity: ql/models/marketmodels/swapforwardmappings.{hpp,cpp} (v1.42.1).

Static utility functions mapping between swap rates and forward rates over a
``CurveState``:

- ``annuity`` / ``swap_derivative`` — building blocks.
- ``coterminal_swap_forward_jacobian`` / ``coinitial_swap_forward_jacobian``
  / ``cm_swap_forward_jacobian`` — ``dsr[i]/df[j]`` jacobians.
- ``coterminal_swap_zed_matrix`` / ``coinitial_swap_zed_matrix`` /
  ``cm_swap_zed_matrix`` — Z-matrices switching base from forward to swap
  rates.

Divergence: ``swaptionImpliedVolatility`` (freezing-coefficient swaption
implied vol) is deferred — it needs a concrete ``MarketModel`` with a
``pseudo_root``, which lands with the W9-B/C model concretes. The static
jacobian/Z-matrix helpers it builds on ARE ported here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pquantlib.math.matrix import Matrix
    from pquantlib.models.marketmodels.curve_state import CurveState


class SwapForwardMappings:
    """Swap-rate <-> forward-rate mapping helpers (all static).

    # C++ parity: swapforwardmappings.hpp SwapForwardMappings.
    """

    @staticmethod
    def annuity(
        cs: CurveState, start_index: int, end_index: int, numeraire_index: int
    ) -> float:
        """Annuity of the swap spanning ``[start_index, end_index)``.

        # C++ parity: swapforwardmappings.cpp SwapForwardMappings::annuity.
        """
        taus = cs.rate_taus()
        result = 0.0
        for i in range(start_index, end_index):
            result += taus[i] * cs.discount_ratio(i + 1, numeraire_index)
        return result

    @staticmethod
    def swap_derivative(
        cs: CurveState, start_index: int, end_index: int, forward_index: int
    ) -> float:
        """Derivative of the swap rate w.r.t. forward rate ``forward_index``.

        # C++ parity: swapforwardmappings.cpp SwapForwardMappings::swapDerivative.
        """
        if forward_index < start_index:
            return 0.0
        if forward_index >= end_index:
            return 0.0
        numerator = cs.discount_ratio(start_index, end_index) - 1
        swap_annuity = SwapForwardMappings.annuity(cs, start_index, end_index, end_index)
        ratio = cs.rate_taus()[forward_index] / (
            1 + cs.rate_taus()[forward_index] * cs.forward_rate(forward_index)
        )
        part1 = ratio * (numerator + 1) / swap_annuity
        part2 = numerator / (swap_annuity * swap_annuity)
        if forward_index >= 1:
            part2 *= ratio * SwapForwardMappings.annuity(
                cs, start_index, forward_index, end_index
            )
        else:
            part2 = 0.0
        return part1 - part2

    @staticmethod
    def coterminal_swap_forward_jacobian(cs: CurveState) -> Matrix:
        """``dsr[i]/df[j]`` jacobian between coterminal swap and forward rates.

        # C++ parity: swapforwardmappings.cpp
        SwapForwardMappings::coterminalSwapForwardJacobian.
        """
        n = cs.number_of_rates()
        f = cs.forward_rates()
        tau = cs.rate_taus()
        # coterminal floating-leg values a[k] = P(k)/P(n) - 1
        a = [cs.discount_ratio(k, n) - 1.0 for k in range(n)]
        jacobian = np.zeros((n, n), dtype=np.float64)
        for i in range(n):  # i = swap rate index
            for j in range(i, n):  # j = forward rate index
                bi = cs.coterminal_swap_annuity(n, i)
                bj = cs.coterminal_swap_annuity(n, j)
                jacobian[i, j] = tau[j] / cs.coterminal_swap_annuity(j + 1, i) + tau[j] / (
                    1.0 + f[j] * tau[j]
                ) * (-a[j] * bi + a[i] * bj) / (bi * bi)
        return jacobian

    @staticmethod
    def coterminal_swap_zed_matrix(cs: CurveState, displacement: float) -> Matrix:
        """Z-matrix switching base from forward to coterminal swap rates.

        # C++ parity: swapforwardmappings.cpp
        SwapForwardMappings::coterminalSwapZedMatrix.
        """
        n = cs.number_of_rates()
        z_matrix = SwapForwardMappings.coterminal_swap_forward_jacobian(cs)
        f = cs.forward_rates()
        sr = cs.coterminal_swap_rates()
        for i in range(n):
            for j in range(i, n):
                z_matrix[i, j] *= (f[j] + displacement) / (sr[i] + displacement)
        return z_matrix

    @staticmethod
    def coinitial_swap_forward_jacobian(cs: CurveState) -> Matrix:
        """``dsr[i]/df[j]`` jacobian between coinitial swap and forward rates.

        # C++ parity: swapforwardmappings.cpp
        SwapForwardMappings::coinitialSwapForwardJacobian.
        """
        n = cs.number_of_rates()
        jacobian = np.zeros((n, n), dtype=np.float64)
        for i in range(n):  # i = swap rate index
            for j in range(n):  # j = forward rate index
                jacobian[i, j] = SwapForwardMappings.swap_derivative(cs, 0, i + 1, j)
        return jacobian

    @staticmethod
    def cm_swap_forward_jacobian(cs: CurveState, spanning_forwards: int) -> Matrix:
        """``dsr[i]/df[j]`` jacobian between CM swap and forward rates.

        # C++ parity: swapforwardmappings.cpp
        SwapForwardMappings::cmSwapForwardJacobian.
        """
        n = cs.number_of_rates()
        jacobian = np.zeros((n, n), dtype=np.float64)
        for i in range(n):  # i = swap rate index
            for j in range(n):  # j = forward rate index
                jacobian[i, j] = SwapForwardMappings.swap_derivative(
                    cs, i, min(n, i + spanning_forwards), j
                )
        return jacobian

    @staticmethod
    def coinitial_swap_zed_matrix(cs: CurveState, displacement: float) -> Matrix:
        """Z-matrix switching base from forward to coinitial swap rates.

        # C++ parity: swapforwardmappings.cpp
        SwapForwardMappings::coinitialSwapZedMatrix.
        """
        n = cs.number_of_rates()
        z_matrix = SwapForwardMappings.coinitial_swap_forward_jacobian(cs)
        f = cs.forward_rates()
        sr = [cs.cm_swap_rate(0, i + 1) for i in range(n)]
        for i in range(n):
            for j in range(i, n):
                z_matrix[i, j] *= (f[j] + displacement) / (sr[i] + displacement)
        return z_matrix

    @staticmethod
    def cm_swap_zed_matrix(
        cs: CurveState, spanning_forwards: int, displacement: float
    ) -> Matrix:
        """Z-matrix switching base from forward to CM swap rates.

        # C++ parity: swapforwardmappings.cpp
        SwapForwardMappings::cmSwapZedMatrix.
        """
        n = cs.number_of_rates()
        z_matrix = SwapForwardMappings.cm_swap_forward_jacobian(cs, spanning_forwards)
        f = cs.forward_rates()
        sr = [cs.cm_swap_rate(i, spanning_forwards) for i in range(n)]
        for i in range(n):
            for j in range(i, n):
                z_matrix[i, j] *= (f[j] + displacement) / (sr[i] + displacement)
        return z_matrix
