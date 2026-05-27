"""DiscountingBondEngine — discounts bond cashflows to the settlement date.

# C++ parity: ql/pricingengines/bond/discountingbondengine.{hpp,cpp} (v1.42.1).

The engine discounts every cashflow back to the valuation date via the
supplied yield curve, then computes a second NPV at the settlement date
(to handle the case where the bond's cashflow on settlement is excluded
by the default ``include_settlement_date_flows=False``).
"""

from __future__ import annotations

from typing import cast

from pquantlib.cashflows.cash_flows import CashFlows
from pquantlib.instruments.bond import BondArguments, BondResults
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.pricingengines.generic_engine import GenericEngine
from pquantlib.termstructures.protocols import YieldTermStructureProtocol
from pquantlib.termstructures.yield_term_structure import YieldTermStructure


class DiscountingBondEngine(GenericEngine[BondArguments, BondResults]):
    """Discounting engine for Bond.

    # C++ parity: ``class DiscountingBondEngine : public Bond::engine``.

    Construct with a ``YieldTermStructure`` discount curve and an
    optional ``include_settlement_date_flows`` override (defaults to
    ``ObservableSettings().include_reference_date_events`` — same C++
    Settings semantic).

    # C++ parity divergence — type widening:
    # the C++ engine takes ``Handle<YieldTermStructure>``. The Python
    # port takes the abstract base ``YieldTermStructure`` directly
    # (Python has no ``Handle`` wrapper). The earlier draft used
    # ``YieldTermStructureProtocol`` but that Protocol's narrower
    # ``zero_rate`` return type (float) breaks structural typing of
    # FlatForward (whose ``zero_rate`` returns ``InterestRate``). The
    # engine only ever calls ``discount`` and ``reference_date``, so
    # widening to the abstract base is the smallest fix.
    """

    def __init__(
        self,
        discount_curve: YieldTermStructure,
        include_settlement_date_flows: bool | None = None,
    ) -> None:
        super().__init__(BondArguments(), BondResults())
        self._discount_curve: YieldTermStructure = discount_curve
        self._include_settlement_date_flows: bool | None = include_settlement_date_flows
        discount_curve.register_with(self)

    def discount_curve(self) -> YieldTermStructure:
        return self._discount_curve

    def calculate(self) -> None:
        """Compute NPV + settlement value via discounted cashflows.

        # C++ parity: discountingbondengine.cpp:36-67.

        # C++ parity divergence: the C++ engine guards against an empty
        # ``Handle<YieldTermStructure>``. The Python port takes a non-
        # optional protocol-typed curve at construction time, so the
        # runtime null check is dropped (pyright flags it as
        # ``reportUnnecessaryComparison``).
        """
        results = self._results
        args = self._arguments

        results.valuation_date = self._discount_curve.reference_date()

        if self._include_settlement_date_flows is not None:
            include_ref_date_flows = self._include_settlement_date_flows
        else:
            include_ref_date_flows = ObservableSettings().include_reference_date_events

        # YieldTermStructure → YieldTermStructureProtocol cast is safe
        # at runtime (the abstract class has reference_date / discount).
        # See class docstring: pyright's structural check fails on the
        # return-type difference of ``zero_rate`` (InterestRate vs the
        # Protocol's float), which we don't call here.
        curve = cast("YieldTermStructureProtocol", self._discount_curve)

        results.value = CashFlows.npv_curve(
            args.cashflows,
            curve,
            include_ref_date_flows,
            results.valuation_date,
            results.valuation_date,
        )
        results.error_estimate = 0.0

        # The bond's settlement-date cashflow is never taken into account;
        # recompute if the settlement date differs from the valuation date.
        if not include_ref_date_flows and results.valuation_date == args.settlement_date:
            results.settlement_value = results.value
        else:
            results.settlement_value = CashFlows.npv_curve(
                args.cashflows,
                curve,
                False,
                args.settlement_date,
                args.settlement_date,
            )


__all__ = ["DiscountingBondEngine"]
