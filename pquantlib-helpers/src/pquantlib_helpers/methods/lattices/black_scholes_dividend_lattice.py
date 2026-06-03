"""BlackScholesDividendLattice — binomial BSM lattice with discrete dividends.

# Retired-API compat layer — NOT a port of C++ QuantLib v1.42.1.

Java parity:
``org.jquantlib.methods.lattices.BlackScholesDividendLattice`` (jquantlib-helpers).

Extends pquantlib core's
:class:`~pquantlib.methods.lattices.bsm_lattice.BlackScholesLattice` by shifting
every node's underlying price down by the accumulated escrow of future
dividends. The escrow is built once in the constructor from the dividend
schedule and a per-time-grid index ``map`` so ``underlying(i, index)`` is a
single array lookup, exactly as in the Java original.

# Java-bug parity (DELIBERATE — required for the primary cross-validation gate)
#
# Two faithful reproductions of quirks in the Java source:
#
# 1. The escrow accumulator ``_list[k]`` sums ``exp(-r * t_k)`` — the *bare
#    discount factor* of each dividend date — and does NOT multiply by the
#    dividend cash amount.  The Java line is
#        list[i] = list[i-1] + Math.exp(-riskFreeRate * time);
#    (BlackScholesDividendLattice.java:67).  Economically this under-subtracts
#    the escrow for dividends whose amount != 1.0, so the resulting NPV is a
#    JQuantLib-specific approximation, NOT the C++ analytic escrowed-dividend
#    value.  We reproduce it verbatim because pquantlib-helpers' job is to be a
#    behaviour-identical compat layer for the retired JQuantLib API.
#
# 2. ``underlying(i, index)`` subtracts ``_list[_map[index]]`` — indexed by the
#    node ``index`` rather than the time slice ``i``.  The Java line is
#        return tree.underlying(i, index) - list[map[index]];
#    (BlackScholesDividendLattice.java:81).  We keep the same indexing so the
#    grid the discretized option reads (and the Greek-extraction probes) match
#    Java bit-for-bit.
#
# These are documented, intentional divergences from economically-correct
# behaviour; the v1.42.1 ``AnalyticDividendEuropeanEngine`` is the correct
# model and is what pquantlib core ships for new code.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from pquantlib.methods.lattices.bsm_lattice import BlackScholesLattice

if TYPE_CHECKING:
    from pquantlib.cashflows.dividend import Dividend
    from pquantlib.daycounters.day_counter import DayCounter
    from pquantlib.methods.lattices.binomial_tree import BinomialTree
    from pquantlib.time.date import Date
    from pquantlib.time.time_grid import TimeGrid


class BlackScholesDividendLattice(BlackScholesLattice):
    """Binomial BSM lattice with a discrete-dividend escrow adjustment.

    Java parity: ``BlackScholesDividendLattice<T>``.
    """

    def __init__(
        self,
        tree: BinomialTree,
        risk_free_rate: float,
        end: float,
        steps: int,
        day_counter: DayCounter,
        grid: TimeGrid,
        reference_date: Date,
        cash_flow: list[Dividend],
    ) -> None:
        # Java parity: ``super(tree, riskFreeRate, end, steps)``.
        super().__init__(tree, risk_free_rate, end, steps)
        self._div_tree: BinomialTree = tree

        # Java parity (BlackScholesDividendLattice.java:55-77):
        # ``map`` maps each time-grid element to an index into ``list``;
        # ``list`` accumulates the (bare) discount factor of each dividend.
        grid_size = len(grid)
        # map from TimeGrid element to internal list of dividend amounts.
        self._map: list[int] = [0] * grid_size
        # list of total amount of dividends to be discounted; list[0] = 0.
        self._list: list[float] = [0.0] * (len(cash_flow) + 1)

        last_idx = 0
        for i in range(1, len(self._list)):
            # Java parity: time = dc.yearFraction(referenceDate, cashFlow[i-1].date()).
            time = day_counter.year_fraction(reference_date, cash_flow[i - 1].date())
            # Java parity: list[i] = list[i-1] + exp(-r * time).  NOTE the
            # missing dividend amount — see module-level Java-bug parity note.
            self._list[i] = self._list[i - 1] + math.exp(-risk_free_rate * time)
            # grid element immediately greater than current dividend time.
            curr_idx = grid.closest_index(time) + 1
            for j in range(last_idx, curr_idx):
                self._map[j] = i - 1
            last_idx = curr_idx
        for j in range(last_idx, len(self._map)):
            self._map[j] = len(self._list) - 1

    def underlying(self, i: int, index: int) -> float:
        """Tree underlying minus the accumulated dividend escrow.

        Java parity: ``tree.underlying(i, index) - list[map[index]]``
        (note the ``index`` — not ``i`` — lookup; see module-level note).
        """
        return self._div_tree.underlying(i, index) - self._list[self._map[index]]


__all__ = ["BlackScholesDividendLattice"]
