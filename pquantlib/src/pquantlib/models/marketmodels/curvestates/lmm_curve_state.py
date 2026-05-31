"""LMMCurveState — LIBOR-market-model curve state.

# C++ parity: ql/models/marketmodels/curvestates/lmmcurvestate.{hpp,cpp} (v1.42.1).

Stores the yield-curve state for a LIBOR market model. Driven by either
forward rates (``set_on_forward_rates``) or discount ratios
(``set_on_discount_ratios``); all other geometry (coterminal/CM swap rates,
annuities) is derived lazily.

Divergence from C++: the ``first_cot_annuity_comped_`` incremental
coterminal-annuity cache and all ``mutable`` buffers are plain instance
attributes here.
"""

from __future__ import annotations

from pquantlib import qassert
from pquantlib.models.marketmodels.curve_state import (
    CurveState,
    constant_maturity_from_discount_ratios,
    coterminal_from_discount_ratios,
)


class LMMCurveState(CurveState):
    """Curve state for LIBOR market models.

    # C++ parity: lmmcurvestate.hpp LMMCurveState.
    """

    def __init__(self, rate_times: list[float]) -> None:
        super().__init__(rate_times)
        n = self._number_of_rates
        self._first = n
        self._disc_ratios = [1.0] * (n + 1)
        self._forward_rates = [0.0] * n
        last_tau = self._rate_taus[n - 1]
        self._cm_swap_rates = [0.0] * n
        self._cm_swap_annuities = [last_tau] * n
        self._cot_swap_rates = [0.0] * n
        self._cot_annuities = [last_tau] * n
        self._first_cot_annuity_comped = n

    def set_on_forward_rates(self, rates: list[float], first_valid_index: int = 0) -> None:
        """Set the state from forward rates; recompute discount ratios.

        # C++ parity: lmmcurvestate.cpp setOnForwardRates.
        """
        n = self._number_of_rates
        qassert.require(
            len(rates) == n,
            f"rates mismatch: {n} required, {len(rates)} provided",
        )
        qassert.require(
            first_valid_index < n,
            f"first valid index must be less than {n}: {first_valid_index} not allowed",
        )
        self._first = first_valid_index
        for i in range(self._first, n):
            self._forward_rates[i] = rates[i]
        for i in range(self._first, n):
            self._disc_ratios[i + 1] = self._disc_ratios[i] / (
                1.0 + self._forward_rates[i] * self._rate_taus[i]
            )
        self._first_cot_annuity_comped = n

    def set_on_discount_ratios(
        self, disc_ratios: list[float], first_valid_index: int = 0
    ) -> None:
        """Set the state from discount ratios; recompute forward rates.

        # C++ parity: lmmcurvestate.cpp setOnDiscountRatios.
        """
        n = self._number_of_rates
        qassert.require(
            len(disc_ratios) == n + 1,
            f"too many discount ratios: {n + 1} required, {len(disc_ratios)} provided",
        )
        qassert.require(
            first_valid_index < n,
            f"first valid index must be less than {n + 1}: {first_valid_index} not allowed",
        )
        self._first = first_valid_index
        for i in range(self._first, n + 1):
            self._disc_ratios[i] = disc_ratios[i]
        for i in range(self._first, n):
            self._forward_rates[i] = (
                self._disc_ratios[i] / self._disc_ratios[i + 1] - 1.0
            ) / self._rate_taus[i]
        self._first_cot_annuity_comped = n

    def _check_initialized(self) -> None:
        qassert.require(self._first < self._number_of_rates, "curve state not initialized yet")

    def discount_ratio(self, i: int, j: int) -> float:
        self._check_initialized()
        qassert.require(min(i, j) >= self._first, "invalid index")
        qassert.require(max(i, j) <= self._number_of_rates, "invalid index")
        return self._disc_ratios[i] / self._disc_ratios[j]

    def forward_rate(self, i: int) -> float:
        self._check_initialized()
        qassert.require(self._first <= i <= self._number_of_rates, "invalid index")
        return self._forward_rates[i]

    def coterminal_swap_annuity(self, numeraire: int, i: int) -> float:
        n = self._number_of_rates
        self._check_initialized()
        qassert.require(self._first <= numeraire <= n, "invalid numeraire")
        qassert.require(self._first <= i <= n, "invalid index")
        # Incremental lazy computation of coterminal annuities (C++ parity).
        if self._first_cot_annuity_comped <= i:
            return self._cot_annuities[i] / self._disc_ratios[numeraire]
        if self._first_cot_annuity_comped == n:
            self._cot_annuities[n - 1] = self._rate_taus[n - 1] * self._disc_ratios[n]
            self._first_cot_annuity_comped -= 1
        for j in range(self._first_cot_annuity_comped - 1, i - 1, -1):
            self._cot_annuities[j] = (
                self._cot_annuities[j + 1] + self._rate_taus[j] * self._disc_ratios[j + 1]
            )
        self._first_cot_annuity_comped = i
        return self._cot_annuities[i] / self._disc_ratios[numeraire]

    def coterminal_swap_rate(self, i: int) -> float:
        n = self._number_of_rates
        self._check_initialized()
        qassert.require(self._first <= i <= n, "invalid index")
        return (self._disc_ratios[i] / self._disc_ratios[n] - 1.0) / self.coterminal_swap_annuity(
            n, i
        )

    def cm_swap_annuity(self, numeraire: int, i: int, spanning_forwards: int) -> float:
        n = self._number_of_rates
        self._check_initialized()
        qassert.require(self._first <= numeraire <= n, "invalid numeraire")
        qassert.require(self._first <= i <= n, "invalid index")
        constant_maturity_from_discount_ratios(
            spanning_forwards,
            self._first,
            self._disc_ratios,
            self._rate_taus,
            self._cm_swap_rates,
            self._cm_swap_annuities,
        )
        return self._cm_swap_annuities[i] / self._disc_ratios[numeraire]

    def cm_swap_rate(self, i: int, spanning_forwards: int) -> float:
        self._check_initialized()
        qassert.require(self._first <= i <= self._number_of_rates, "invalid index")
        constant_maturity_from_discount_ratios(
            spanning_forwards,
            self._first,
            self._disc_ratios,
            self._rate_taus,
            self._cm_swap_rates,
            self._cm_swap_annuities,
        )
        return self._cm_swap_rates[i]

    def forward_rates(self) -> list[float]:
        self._check_initialized()
        return self._forward_rates

    def coterminal_swap_rates(self) -> list[float]:
        self._check_initialized()
        coterminal_from_discount_ratios(
            self._first,
            self._disc_ratios,
            self._rate_taus,
            self._cot_swap_rates,
            self._cot_annuities,
        )
        return self._cot_swap_rates

    def cm_swap_rates(self, spanning_forwards: int) -> list[float]:
        self._check_initialized()
        constant_maturity_from_discount_ratios(
            spanning_forwards,
            self._first,
            self._disc_ratios,
            self._rate_taus,
            self._cm_swap_rates,
            self._cm_swap_annuities,
        )
        return self._cm_swap_rates

    def clone(self) -> LMMCurveState:
        new = LMMCurveState(self._rate_times)
        new._first = self._first
        new._disc_ratios = list(self._disc_ratios)
        new._forward_rates = list(self._forward_rates)
        new._cm_swap_rates = list(self._cm_swap_rates)
        new._cm_swap_annuities = list(self._cm_swap_annuities)
        new._cot_swap_rates = list(self._cot_swap_rates)
        new._cot_annuities = list(self._cot_annuities)
        new._first_cot_annuity_comped = self._first_cot_annuity_comped
        return new
