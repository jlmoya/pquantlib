"""CoterminalSwapCurveState — coterminal-swap market-model curve state.

# C++ parity: ql/models/marketmodels/curvestates/coterminalswapcurvestate.{hpp,cpp} (v1.42.1).

Stores the yield-curve state parametrized by coterminal swap rates (all
swaps share the same final payment date). Driven by
``set_on_coterminal_swap_rates``; discount ratios + coterminal annuities are
computed directly. Forward / CM swap rates are derived lazily.
"""

from __future__ import annotations

from pquantlib import qassert
from pquantlib.models.marketmodels.curve_state import (
    CurveState,
    constant_maturity_from_discount_ratios,
    forwards_from_discount_ratios,
)


class CoterminalSwapCurveState(CurveState):
    """Curve state for coterminal-swap market models.

    # C++ parity: coterminalswapcurvestate.hpp CoterminalSwapCurveState.
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

    def set_on_coterminal_swap_rates(
        self, rates: list[float], first_valid_index: int = 0
    ) -> None:
        """Set the state from coterminal swap rates; recompute discount ratios.

        # C++ parity: coterminalswapcurvestate.cpp setOnCoterminalSwapRates.
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
            self._cot_swap_rates[i] = rates[i]
        # reference discount bond = P(n); discRatios_[n] = 1.0 by construction
        self._cot_annuities[n - 1] = self._rate_taus[n - 1]
        for i in range(n - 1, self._first, -1):
            self._disc_ratios[i] = 1.0 + self._cot_swap_rates[i] * self._cot_annuities[i]
            self._cot_annuities[i - 1] = (
                self._cot_annuities[i] + self._rate_taus[i - 1] * self._disc_ratios[i]
            )
        self._disc_ratios[self._first] = (
            1.0 + self._cot_swap_rates[self._first] * self._cot_annuities[self._first]
        )

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
        forwards_from_discount_ratios(
            self._first, self._disc_ratios, self._rate_taus, self._forward_rates
        )
        return self._forward_rates[i]

    def coterminal_swap_annuity(self, numeraire: int, i: int) -> float:
        n = self._number_of_rates
        self._check_initialized()
        qassert.require(self._first <= numeraire <= n, "invalid numeraire")
        qassert.require(self._first <= i <= n, "invalid index")
        return self._cot_annuities[i] / self._disc_ratios[numeraire]

    def coterminal_swap_rate(self, i: int) -> float:
        self._check_initialized()
        qassert.require(self._first <= i <= self._number_of_rates, "invalid index")
        return self._cot_swap_rates[i]

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
        forwards_from_discount_ratios(
            self._first, self._disc_ratios, self._rate_taus, self._forward_rates
        )
        return self._forward_rates

    def coterminal_swap_rates(self) -> list[float]:
        self._check_initialized()
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

    def clone(self) -> CoterminalSwapCurveState:
        new = CoterminalSwapCurveState(self._rate_times)
        new._first = self._first
        new._disc_ratios = list(self._disc_ratios)
        new._forward_rates = list(self._forward_rates)
        new._cm_swap_rates = list(self._cm_swap_rates)
        new._cm_swap_annuities = list(self._cm_swap_annuities)
        new._cot_swap_rates = list(self._cot_swap_rates)
        new._cot_annuities = list(self._cot_annuities)
        return new
