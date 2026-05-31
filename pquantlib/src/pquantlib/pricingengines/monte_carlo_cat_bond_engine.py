"""MonteCarloCatBondEngine — MC pricing engine for cat bonds.

# C++ parity: ql/experimental/catbonds/montecarlocatbondengine.{hpp,cpp}
#             (v1.42.1).

Prices a ``CatBond`` by Monte-Carlo over the attached ``CatRisk``:

1. Draw scenarios of (event_date, loss) pairs from the cat risk's
   simulation over [effective_date, maturity].
2. Map each scenario onto a ``NotionalPath`` via the bond's
   ``NotionalRisk``.
3. Value the (notional-eroded) cashflows on each path; average and
   discount.

Also accumulates the loss probability, exhaustion probability, and
expected loss across the simulated paths.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pquantlib.instruments.cat_bond import CatBondArguments, CatBondResults
from pquantlib.instruments.risky_notional import NotionalPath
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.pricingengines.generic_engine import GenericEngine
from pquantlib.time.date import Date

if TYPE_CHECKING:
    from pquantlib.cashflows.cash_flow import CashFlow
    from pquantlib.instruments.cat_risk import CatRisk, EventPath
    from pquantlib.termstructures.yield_term_structure import YieldTermStructure

_NULL_DATE: Date = Date()
# C++ ``MAX_PATHS`` (montecarlocatbondengine.cpp:77).
_MAX_PATHS: int = 10000


class MonteCarloCatBondEngine(GenericEngine[CatBondArguments, CatBondResults]):
    """Monte-Carlo cat-bond engine.

    # C++ parity: ``class MonteCarloCatBondEngine`` (montecarlocatbondengine.{hpp,cpp}).
    """

    def __init__(
        self,
        cat_risk: CatRisk,
        discount_curve: YieldTermStructure,
        include_settlement_date_flows: bool | None = None,
    ) -> None:
        super().__init__(CatBondArguments(), CatBondResults())
        self._cat_risk: CatRisk = cat_risk
        self._discount_curve: YieldTermStructure = discount_curve
        self._include_settlement_date_flows: bool | None = include_settlement_date_flows

    def discount_curve(self) -> YieldTermStructure:
        return self._discount_curve

    def calculate(self) -> None:
        # C++ parity: montecarlocatbondengine.cpp:37-73.
        results = self._results
        results.reset()
        # C++ checks ``discountCurve_.empty()``; PQuantLib requires a concrete
        # curve at construction, so the empty-handle guard has no analogue.

        results.valuation_date = self._discount_curve.reference_date()

        include_ref_date_flows = (
            self._include_settlement_date_flows
            if self._include_settlement_date_flows is not None
            else ObservableSettings().include_reference_date_events
        )

        value, loss_p, exhaustion_p, expected_loss = self._npv(
            include_ref_date_flows, results.valuation_date, results.valuation_date
        )
        results.value = value
        results.loss_probability = loss_p
        results.exhaustion_probability = exhaustion_p
        results.expected_loss = expected_loss

        # A bond's cashflow on the settlement date is never taken into
        # account, so recompute the settlement value if needed.
        if not include_ref_date_flows and results.valuation_date == self._arguments.settlement_date:
            results.settlement_value = value
        else:
            sv, _, _, _ = self._npv(
                include_ref_date_flows,
                self._arguments.settlement_date,
                self._arguments.settlement_date,
            )
            results.settlement_value = sv

    def _npv(
        self,
        include_settlement_date_flows: bool,
        settlement_date: Date,
        npv_date: Date,
    ) -> tuple[float, float, float, float]:
        # C++ parity: montecarlocatbondengine.cpp:75-116.
        loss_probability = 0.0
        exhaustion_probability = 0.0
        expected_loss = 0.0
        cashflows = self._arguments.cashflows
        if not cashflows:
            return 0.0, loss_probability, exhaustion_probability, expected_loss

        settle = settlement_date
        if settle == _NULL_DATE:
            settle = ObservableSettings().evaluation_date_or_today()
        npv_d = npv_date if npv_date != _NULL_DATE else settle

        total_npv = 0.0
        effective_date = max(self._arguments.start_date, settle)
        maturity_date = cashflows[-1].date()
        notional_risk = self._arguments.notional_risk
        assert notional_risk is not None
        cat_simulation = self._cat_risk.new_simulation(effective_date, maturity_date)

        events_path: EventPath = []
        notional_path = NotionalPath()
        risk_free_npv = self._path_npv(include_settlement_date_flows, settle, notional_path)
        path_count = 0
        while cat_simulation.next_path(events_path) and path_count < _MAX_PATHS:
            notional_risk.update_path(events_path, notional_path)
            if notional_path.loss() > 0:  # most paths have no loss
                total_npv += self._path_npv(include_settlement_date_flows, settle, notional_path)
                loss_probability += 1
                if notional_path.loss() == 1:
                    exhaustion_probability += 1
                expected_loss += notional_path.loss()
            else:
                total_npv += risk_free_npv
            path_count += 1

        if path_count > 0:
            loss_probability /= path_count
            exhaustion_probability /= path_count
            expected_loss /= path_count
            total_npv = total_npv / (path_count * self._discount_curve.discount(npv_d))
        return total_npv, loss_probability, exhaustion_probability, expected_loss

    def _path_npv(
        self,
        include_settlement_date_flows: bool,
        settlement_date: Date,
        notional_path: NotionalPath,
    ) -> float:
        # C++ parity: montecarlocatbondengine.cpp:118-129.
        total_npv = 0.0
        for cashflow in self._arguments.cashflows:
            if not cashflow.has_occurred(settlement_date, include_settlement_date_flows):
                amount = self._cash_flow_risky_value(cashflow, notional_path)
                total_npv += amount * self._discount_curve.discount(cashflow.date())
        return total_npv

    def _cash_flow_risky_value(self, cf: CashFlow, notional_path: NotionalPath) -> float:
        # C++ parity: montecarlocatbondengine.cpp:131-134.
        return cf.amount() * notional_path.notional_rate(cf.date())


__all__ = ["MonteCarloCatBondEngine"]
