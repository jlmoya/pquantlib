"""DiscountingSwapEngine — NPV-per-leg pricing engine for Swap instruments.

# C++ parity: ql/pricingengines/swap/discountingswapengine.{hpp,cpp} (v1.42.1).

The engine discounts every cashflow on every leg of a Swap against a
single yield curve (``discount_curve``), computes per-leg NPV/BPS,
applies the payer-sign multiplier, sums to the swap NPV, and emits the
start/end discount factors per leg.

C++ accepts an ``includeSettlementDateFlows`` Tri-state (optional<bool>)
and falls back to ``Settings.includeReferenceDateEvents()`` when null.
The Python port simplifies to a bool default of ``False`` (matches the
C++ behaviour when both the engine flag and Settings flag are null/false).
``settlement_date`` and ``npv_date`` default to ``curve.reference_date()``.
"""

from __future__ import annotations

from pquantlib import qassert
from pquantlib.cashflows.cash_flows import CashFlows
from pquantlib.instruments.swap import (
    SwapEngine,
    leg_maturity_date,
    leg_start_date,
)
from pquantlib.termstructures.protocols import YieldTermStructureProtocol
from pquantlib.time.date import Date


class DiscountingSwapEngine(SwapEngine):
    """Discount-curve-based swap pricing engine.

    Discounts each leg's cashflows to the curve's reference date,
    applies the payer-sign multiplier, sums into the swap NPV, and
    emits per-leg NPV / BPS / start-discount / end-discount.

    # C++ parity: ``DiscountingSwapEngine::calculate`` (discountingswapengine.cpp:40-112).
    """

    def __init__(
        self,
        discount_curve: YieldTermStructureProtocol,
        include_settlement_date_flows: bool = False,
        settlement_date: Date | None = None,
        npv_date: Date | None = None,
    ) -> None:
        super().__init__()
        self._discount_curve: YieldTermStructureProtocol = discount_curve
        self._include_settlement_date_flows: bool = include_settlement_date_flows
        # Use None to mean "fall back to curve reference date" (C++ uses
        # ``Date()`` sentinel — the Date() null-date interface in PQuantLib
        # is also valid but Optional is clearer at the engine API level).
        self._settlement_date: Date | None = settlement_date
        self._npv_date: Date | None = npv_date
        # Observer wiring: subscribe to the discount-curve so that
        # curve updates invalidate downstream Swap caches.
        register = getattr(discount_curve, "register_with", None)
        if register is not None:
            register(self)

    def discount_curve(self) -> YieldTermStructureProtocol:
        return self._discount_curve

    def calculate(self) -> None:
        """Compute results from arguments + discount curve.

        # C++ parity: ``DiscountingSwapEngine::calculate``
        # (discountingswapengine.cpp:40-112).
        """
        args = self._arguments
        results = self._results

        ref_date = self._discount_curve.reference_date()
        settle = self._settlement_date if self._settlement_date is not None else ref_date
        qassert.require(
            settle >= ref_date,
            f"settlement date ({settle}) before discount curve reference date ({ref_date})",
        )
        npv_d = self._npv_date if self._npv_date is not None else ref_date
        qassert.require(
            npv_d >= ref_date,
            f"npv date ({npv_d}) before discount curve reference date ({ref_date})",
        )

        results.value = 0.0
        results.error_estimate = None
        results.valuation_date = npv_d
        results.npv_date_discount = self._discount_curve.discount(npv_d)

        n = len(args.legs)
        results.leg_npv = [0.0] * n
        results.leg_bps = [0.0] * n
        results.start_discounts = [0.0] * n
        results.end_discounts = [0.0] * n

        total_npv = 0.0
        for i in range(n):
            leg = args.legs[i]
            try:
                # C++ uses CashFlows::npvbps for a single-pass joint compute;
                # we call npv_curve + bps separately. Slight performance cost,
                # numerically identical (both are linear in the leg).
                leg_npv_val = CashFlows.npv_curve(
                    leg,
                    self._discount_curve,
                    self._include_settlement_date_flows,
                    settle,
                    npv_d,
                )
                leg_bps_val = CashFlows.bps(
                    leg,
                    self._discount_curve,
                    self._include_settlement_date_flows,
                    settle,
                    npv_d,
                )
            except Exception as e:
                msg = f"leg #{i + 1}: {e}"
                raise type(e)(msg) from e

            results.leg_npv[i] = leg_npv_val * args.payer[i]
            results.leg_bps[i] = leg_bps_val * args.payer[i]

            if len(leg) > 0:
                d1 = leg_start_date(leg)
                d2 = leg_maturity_date(leg)
                results.start_discounts[i] = (
                    self._discount_curve.discount(d1) if d1 >= ref_date else 0.0
                )
                results.end_discounts[i] = (
                    self._discount_curve.discount(d2) if d2 >= ref_date else 0.0
                )

            total_npv += results.leg_npv[i]

        results.value = total_npv


__all__ = ["DiscountingSwapEngine"]
