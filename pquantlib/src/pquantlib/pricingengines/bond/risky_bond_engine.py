"""RiskyBondEngine — bond pricer with default risk and recovery.

# C++ parity: ql/pricingengines/bond/riskybondengine.{hpp,cpp} (v1.43),
#             ``class RiskyBondEngine : public Bond::engine``.

Every cashflow is contingent on survival, and each coupon period contributes a
recovery payment assumed to occur at the period's midpoint::

    NPV = sum_i CF_i P(t, T_i) Q(T_i)
        + sum_i Rec * N_i * P(t, T_i^mid) * (Q(T_{i-1}) - Q(T_i))

where ``Q`` is the survival probability and ``T_i^mid = T_{i-1} + (T_i - T_{i-1})/2``
in **days**, integer-divided (C++ ``d1 + (d2 - d1) / 2`` on ``Date``).

Details a port would plausibly get wrong, all cross-validated:

* ``d1`` starts at ``max(yieldTS.referenceDate(), CashFlows::startDate(cashflows))``
  and is advanced to ``d2`` **only when the cashflow is a Coupon** — a
  redemption between two coupons does not move the recovery window.
* the survival-weighted coupon is discounted at the *payment* date but the
  recovery leg at the *midpoint* date.
* ``results.value`` is the NPV as of the yield curve's reference date, while
  ``results.settlement_value`` sums only the cashflows strictly after the
  settlement date and then divides by ``discount(settlementDate)``. The two
  are therefore not simply related.
* a zero hazard rate makes every survival probability 1 and every default
  probability 0, so the engine must return exactly the risk-free discounted
  value — pinned against ``DiscountingBondEngine`` on the same curve.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pquantlib.cashflows.cash_flows import CashFlows
from pquantlib.cashflows.coupon import Coupon
from pquantlib.instruments.bond import BondArguments, BondResults
from pquantlib.pricingengines.generic_engine import GenericEngine

if TYPE_CHECKING:
    from pquantlib.termstructures.credit.default_probability_term_structure import (
        DefaultProbabilityTermStructure,
    )
    from pquantlib.termstructures.protocols import YieldTermStructureProtocol


class RiskyBondEngine(GenericEngine[BondArguments, BondResults]):
    """Risky pricing engine for bonds.

    # C++ parity: ``class RiskyBondEngine`` (riskybondengine.hpp:70-83).
    """

    def __init__(
        self,
        default_ts: DefaultProbabilityTermStructure,
        recovery_rate: float,
        yield_ts: YieldTermStructureProtocol,
    ) -> None:
        super().__init__(BondArguments(), BondResults())
        self._default_ts: DefaultProbabilityTermStructure = default_ts
        self._recovery_rate: float = float(recovery_rate)
        self._yield_ts: YieldTermStructureProtocol = yield_ts
        # C++ parity: riskybondengine.cpp:35-36 — registerWith both handles.
        default_ts.register_with(self)
        register = getattr(yield_ts, "register_with", None)
        if register is not None:
            register(self)

    # --- inspectors --------------------------------------------------------

    def default_ts(self) -> DefaultProbabilityTermStructure:
        """C++ parity: riskybondengine.hpp:85-87."""
        return self._default_ts

    def recovery_rate(self) -> float:
        """C++ parity: riskybondengine.hpp:89."""
        return self._recovery_rate

    def yield_ts(self) -> YieldTermStructureProtocol:
        """C++ parity: riskybondengine.hpp:91."""
        return self._yield_ts

    # --- engine ------------------------------------------------------------

    def calculate(self) -> None:
        # C++ parity: riskybondengine.cpp:39-74.
        args = self._arguments
        results = self._results
        results.reset()

        npv_date = self._yield_ts.reference_date()
        settlement_date = args.settlement_date
        start_date = CashFlows.start_date(args.cashflows)
        d1 = max(npv_date, start_date)

        npv = 0.0
        settlement_value = 0.0
        for cf in args.cashflows:
            d2 = cf.date()
            if d2 <= npv_date:
                continue

            weighted_coupon_amount = cf.amount() * self._default_ts.survival_probability(d2)
            npv += weighted_coupon_amount * self._yield_ts.discount(d2)
            if d2 > settlement_date:
                settlement_value += weighted_coupon_amount * self._yield_ts.discount(d2)

            if isinstance(cf, Coupon):
                # C++ ``Date defaultDate = d1 + (d2 - d1) / 2;`` — integer
                # day arithmetic, truncating.
                default_date = d1 + (d2 - d1) // 2
                weighted_recovery = (
                    cf.nominal()
                    * self._recovery_rate
                    * (
                        self._default_ts.survival_probability(d1)
                        - self._default_ts.survival_probability(d2)
                    )
                )
                npv += weighted_recovery * self._yield_ts.discount(default_date)
                if d2 > settlement_date:
                    settlement_value += weighted_recovery * self._yield_ts.discount(default_date)
                # Only a Coupon advances the recovery window.
                d1 = d2

        results.value = npv
        results.settlement_value = settlement_value / self._yield_ts.discount(settlement_date)
        results.valuation_date = npv_date


__all__ = ["RiskyBondEngine"]
