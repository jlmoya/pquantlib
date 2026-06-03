"""BlackScholesDividendLattice — binomial BSM lattice with discrete dividends.

# Retired-API compat layer — NOT a port of C++ QuantLib v1.42.1, but
# functionally equivalent: it converges to the same escrowed-dividend value
# that v1.42.1 ``AnalyticDividendEuropeanEngine`` computes in closed form.

Java parity:
``org.jquantlib.methods.lattices.BlackScholesDividendLattice`` (jquantlib-helpers).

Escrowed-spot model
-------------------
The discrete-dividend option is priced by running a plain CRR binomial tree on
an *escrowed* initial spot ``S0' = S0 - D``, where ``D`` is the present value of
all dividends paid in ``(referenceDate, expiry]``::

    D = sum_i amount_i * riskFreeDiscount(t_i) / dividendYieldDiscount(t_i)
      = sum_i amount_i * exp(-(r_rate - q_rate) * t_i)   (continuous flat rates).

A CRR tree is multiplicative — every node value is ``S0 * u^a * d^b`` — so
scaling every node by ``scale = (S0 - D) / S0`` yields ``(S0 - D) * u^a * d^b``,
i.e. exactly the tree that would be built starting from the escrowed spot
``S0 - D``. The escrow adjustment therefore stays entirely inside
:meth:`underlying` as a single multiplicative scale, with no per-node
``map``/``list`` machinery.

For European options this converges (as ``steps -> infinity``) to the C++
``AnalyticDividendEuropeanEngine`` (plain Black on the escrowed spot). For
American options it is the consistent escrowed approximation (no closed-form
oracle exists).

This corrects the previous implementation, which (a) dropped the dividend cash
amount from the escrow accumulator and (b) subtracted a node-``index``-keyed
escrow rather than a uniform spot shift — neither of which converged to the
analytic escrowed-dividend value. The fix is mirrored in JQuantLib's
``BlackScholesDividendLattice``.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from pquantlib import qassert
from pquantlib.methods.lattices.bsm_lattice import BlackScholesLattice

if TYPE_CHECKING:
    from pquantlib.cashflows.dividend import Dividend
    from pquantlib.daycounters.day_counter import DayCounter
    from pquantlib.methods.lattices.binomial_tree import BinomialTree
    from pquantlib.time.date import Date


class BlackScholesDividendLattice(BlackScholesLattice):
    """Binomial BSM lattice with a discrete-dividend escrowed-spot adjustment.

    Java parity: ``BlackScholesDividendLattice<T>``.
    """

    def __init__(
        self,
        tree: BinomialTree,
        risk_free_rate: float,
        q_rate: float,
        end: float,
        steps: int,
        day_counter: DayCounter,
        reference_date: Date,
        cash_flow: list[Dividend],
    ) -> None:
        # Java parity: ``super(tree, riskFreeRate, end, steps)``.
        super().__init__(tree, risk_free_rate, end, steps)

        # Escrowed-dividend PV over dividends in (referenceDate, expiry].
        d = 0.0
        for dividend in cash_flow:
            time = day_counter.year_fraction(reference_date, dividend.date())
            # keep only dividends strictly after the reference date, up to maturity
            if time > 0.0 and time <= end:
                d += dividend.amount() * math.exp(-(risk_free_rate - q_rate) * time)

        # Root node = initial spot (CRR tree is centred: underlying(0,0) == S0).
        s0 = tree.underlying(0, 0)
        escrowed_spot = s0 - d
        qassert.require(
            escrowed_spot > 0.0, "negative underlying after subtracting dividends"
        )
        self._scale: float = escrowed_spot / s0

    def underlying(self, i: int, index: int) -> float:
        """Tree underlying scaled by the multiplicative dividend escrow.

        Java parity: ``tree.underlying(i, index) * scale`` where
        ``scale = (S0 - D) / S0``. Scaling every CRR node reproduces the tree
        built from the escrowed spot ``S0 - D``.
        """
        return super().underlying(i, index) * self._scale


__all__ = ["BlackScholesDividendLattice"]
