"""CurveState — abstract yield-curve geometry over a rate-time grid.

# C++ parity: ql/models/marketmodels/curvestate.{hpp,cpp} (v1.42.1).

``CurveState`` is the abstract base for the discounting object passed to
market-model products. It stores ``rate_times`` / ``rate_taus`` and the
``number_of_rates``, and declares the abstract geometry interface:

    discount_ratio(i, j)              P(i) / P(j)
    forward_rate(i)
    coterminal_swap_rate(i)
    coterminal_swap_annuity(numeraire, i)
    cm_swap_rate(i, spanning_forwards)
    cm_swap_annuity(numeraire, i, spanning_forwards)
    forward_rates() / coterminal_swap_rates() / cm_swap_rates(spanning)
    clone()

plus the concrete ``swap_rate(begin, end)`` helper.

The module also hosts the three free conversion functions
(``forwards_from_discount_ratios``, ``coterminal_from_discount_ratios``,
``constant_maturity_from_discount_ratios``) used by the concrete curve
states.

Divergences from C++:

- The free conversion functions mutate caller-supplied output lists
  in-place (matching the C++ out-parameter contract), because the
  concrete curve states drive lazy evaluation by repeatedly recomputing
  into preallocated buffers. They are *not* pure functions returning new
  lists.
- ``clone()`` returns a ``CurveState`` (C++ ``std::unique_ptr<CurveState>``);
  Python has no unique ownership, so it is a plain deep copy.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from pquantlib import qassert
from pquantlib.models.marketmodels.utilities import (
    check_increasing_times_and_calculate_taus,
)

# C++ parity: ``Rate`` / ``Real`` / ``DiscountFactor`` are all ``Real``
# (a typedef for ``double``) in QuantLib; in Python they are all ``float``.


class CurveState(ABC):
    """Abstract yield-curve state for market-model simulations.

    # C++ parity: curvestate.hpp CurveState.
    """

    def __init__(self, rate_times: list[float]) -> None:
        self._number_of_rates = 0 if not rate_times else len(rate_times) - 1
        self._rate_times = list(rate_times)
        self._rate_taus = check_increasing_times_and_calculate_taus(self._rate_times)

    def number_of_rates(self) -> int:
        """Number of forward rates ``n``."""
        return self._number_of_rates

    def rate_times(self) -> list[float]:
        """The ``n + 1`` rate times."""
        return self._rate_times

    def rate_taus(self) -> list[float]:
        """The ``n`` rate-time differences."""
        return self._rate_taus

    @abstractmethod
    def discount_ratio(self, i: int, j: int) -> float:
        """Discount-bond ratio ``P(rate_times[i]) / P(rate_times[j])``."""

    @abstractmethod
    def forward_rate(self, i: int) -> float:
        """Forward rate ``f_i`` for the period ``[t_i, t_{i+1}]``."""

    @abstractmethod
    def coterminal_swap_annuity(self, numeraire: int, i: int) -> float:
        """Coterminal swap annuity for swap ``i``, rebased to ``numeraire``."""

    @abstractmethod
    def coterminal_swap_rate(self, i: int) -> float:
        """Coterminal swap rate for swap starting at index ``i``."""

    @abstractmethod
    def cm_swap_annuity(self, numeraire: int, i: int, spanning_forwards: int) -> float:
        """Constant-maturity swap annuity for swap ``i`` (rebased to numeraire)."""

    @abstractmethod
    def cm_swap_rate(self, i: int, spanning_forwards: int) -> float:
        """Constant-maturity swap rate for swap ``i`` spanning ``spanning_forwards``."""

    @abstractmethod
    def forward_rates(self) -> list[float]:
        """All forward rates."""

    @abstractmethod
    def coterminal_swap_rates(self) -> list[float]:
        """All coterminal swap rates."""

    @abstractmethod
    def cm_swap_rates(self, spanning_forwards: int) -> list[float]:
        """All constant-maturity swap rates for the given span."""

    @abstractmethod
    def clone(self) -> CurveState:
        """A newly-allocated copy of this curve state."""

    def swap_rate(self, begin: int, end: int) -> float:
        """Par swap rate spanning forward indices ``[begin, end)``.

        # C++ parity: curvestate.cpp CurveState::swapRate.
        """
        qassert.require(end > begin, "empty range specified")
        qassert.require(end <= self._number_of_rates, "taus/end mismatch")
        total = 0.0
        for i in range(begin, end):
            total += self._rate_taus[i] * self.discount_ratio(i + 1, self._number_of_rates)
        return (
            self.discount_ratio(begin, self._number_of_rates)
            - self.discount_ratio(end, self._number_of_rates)
        ) / total


# --- Free conversion functions (C++ parity: curvestate.cpp) ------------------
# These mutate the supplied output lists in place, matching the C++
# out-parameter signatures (the concrete curve states pass preallocated
# buffers and re-run these for lazy evaluation).


def forwards_from_discount_ratios(
    first_valid_index: int,
    ds: list[float],
    taus: list[float],
    fwds: list[float],
) -> None:
    """Fill ``fwds[i] = (ds[i] - ds[i+1]) / (ds[i+1] * taus[i])``.

    # C++ parity: curvestate.cpp forwardsFromDiscountRatios.
    """
    qassert.require(len(taus) == len(fwds), "taus.size()!=fwds.size()")
    qassert.require(len(ds) == len(fwds) + 1, "ds.size()!=fwds.size()+1")
    for i in range(first_valid_index, len(fwds)):
        fwds[i] = (ds[i] - ds[i + 1]) / (ds[i + 1] * taus[i])


def coterminal_from_discount_ratios(
    first_valid_index: int,
    discount_factors: list[float],
    taus: list[float],
    cot_swap_rates: list[float],
    cot_swap_annuities: list[float],
) -> None:
    """Fill coterminal swap rates + annuities from discount ratios.

    # C++ parity: curvestate.cpp coterminalFromDiscountRatios.
    """
    n = len(cot_swap_rates)
    qassert.require(len(taus) == n, "taus.size()!=cotSwapRates.size()")
    qassert.require(len(cot_swap_annuities) == n, "cotSwapAnnuities.size()!=cotSwapRates.size()")
    qassert.require(
        len(discount_factors) == n + 1, "discountFactors.size()!=cotSwapRates.size()+1"
    )
    cot_swap_annuities[n - 1] = taus[n - 1] * discount_factors[n]
    cot_swap_rates[n - 1] = (
        discount_factors[n - 1] - discount_factors[n]
    ) / cot_swap_annuities[n - 1]
    for i in range(n - 1, first_valid_index, -1):
        cot_swap_annuities[i - 1] = cot_swap_annuities[i] + taus[i - 1] * discount_factors[i]
        cot_swap_rates[i - 1] = (
            discount_factors[i - 1] - discount_factors[n]
        ) / cot_swap_annuities[i - 1]


def constant_maturity_from_discount_ratios(
    spanning_forwards: int,
    first_valid_index: int,
    ds: list[float],
    taus: list[float],
    const_mat_swap_rates: list[float],
    const_mat_swap_annuities: list[float],
) -> None:
    """Fill constant-maturity swap rates + annuities from discount ratios.

    # C++ parity: curvestate.cpp constantMaturityFromDiscountRatios.
    """
    n = len(const_mat_swap_rates)
    qassert.require(len(taus) == n, "taus.size()!=nConstMatSwapRates")
    qassert.require(len(const_mat_swap_annuities) == n, "constMatSwapAnnuities.size()!=nConstMatSwapRates")
    qassert.require(len(ds) == n + 1, "ds.size()!=nConstMatSwapRates+1")
    # first cms rate + annuity
    const_mat_swap_annuities[first_valid_index] = 0.0
    last_index = min(first_valid_index + spanning_forwards, n)
    for i in range(first_valid_index, last_index):
        const_mat_swap_annuities[first_valid_index] += taus[i] * ds[i + 1]
    const_mat_swap_rates[first_valid_index] = (
        ds[first_valid_index] - ds[last_index]
    ) / const_mat_swap_annuities[first_valid_index]
    old_last_index = last_index
    # all the other cms rates + annuities
    for i in range(first_valid_index + 1, n):
        last_index = min(i + spanning_forwards, n)
        const_mat_swap_annuities[i] = const_mat_swap_annuities[i - 1] - taus[i - 1] * ds[i]
        if last_index != old_last_index:
            const_mat_swap_annuities[i] += taus[last_index - 1] * ds[last_index]
        const_mat_swap_rates[i] = (ds[i] - ds[last_index]) / const_mat_swap_annuities[i]
        old_last_index = last_index
