"""DiscountingConstNotionalCrossCurrencySwapEngine — two-currency discounting engine.

# C++ parity: ql/pricingengines/swap/discountingconstnotionalcrosscurrencyswapengine.{hpp,cpp}
  — new in v1.43.

Prices a :class:`~pquantlib.instruments.const_notional_cross_currency_swap.ConstNotionalCrossCurrencySwap`
whose legs involve exactly two currencies. Every leg is discounted on its own
currency's curve; legs in the foreign currency are then converted into the
domestic one, in which the NPV is expressed. Both curves must share a reference
date.

``spot_fx`` is quoted in units of ``domestic_ccy`` per unit of ``foreign_ccy``,
for settlement on ``spot_fx_settle_date``. When that date is not the curves'
reference date, the quote is carried to it through the discount-factor parity
relation ``fx(T1)/fx(T2) = FwdDF_quote(T1->T2) / FwdDF_base(T1->T2)``.

Python port notes:

- PQuantLib has no ``Handle``: the curves and the FX quote are passed directly,
  as in ``DiscountingFwdEngine`` and ``DiscountingSwapEngine``.
- C++'s ``ext::optional<bool> includeSettlementDateFlows`` falls back to
  ``Settings::instance().includeReferenceDateEvents()``, whose default is
  ``false``. PQuantLib's cashflow layer does not read Settings, so the flag is
  a plain ``bool`` defaulting to ``False`` — the same behaviour, consistent
  with the sibling engines.
- Null dates map to ``None``: ``settlement_date`` / ``spot_fx_settle_date``
  default to the curve reference date and ``npv_date`` to the same, mirroring
  the C++ ``Date()`` sentinels.
"""

from __future__ import annotations

from pquantlib import qassert
from pquantlib.cashflows.cash_flows import CashFlows
from pquantlib.currencies.currency import Currency
from pquantlib.instruments.const_notional_cross_currency_swap import (
    ConstNotionalCrossCurrencySwapEngine,
)
from pquantlib.instruments.swap import leg_maturity_date, leg_start_date
from pquantlib.quotes.quote import Quote
from pquantlib.termstructures.protocols import YieldTermStructureProtocol
from pquantlib.time.date import Date


class DiscountingConstNotionalCrossCurrencySwapEngine(
    ConstNotionalCrossCurrencySwapEngine
):
    """Discounting engine for constant-notional cross-currency swaps."""

    def __init__(
        self,
        domestic_ccy: Currency,
        domestic_ccy_discount_curve: YieldTermStructureProtocol,
        foreign_ccy: Currency,
        foreign_ccy_discount_curve: YieldTermStructureProtocol,
        spot_fx: Quote,
        include_settlement_date_flows: bool = False,
        settlement_date: Date | None = None,
        npv_date: Date | None = None,
        spot_fx_settle_date: Date | None = None,
    ) -> None:
        """# C++ parity: the sole ``DiscountingConstNotionalCrossCurrencySwapEngine`` ctor.

        :param domestic_ccy: currency the NPV is expressed in.
        :param domestic_ccy_discount_curve: discount curve for domestic-currency flows.
        :param foreign_ccy: the other currency any leg may be denominated in.
        :param foreign_ccy_discount_curve: discount curve for foreign-currency flows.
        :param spot_fx: market spot rate, domestic per foreign, quoted for
            settlement on ``spot_fx_settle_date``.
        :param include_settlement_date_flows: whether flows falling exactly on
            the settlement date count towards the NPV.
        :param settlement_date: flows before it are dropped; ``None`` means the
            curve reference date.
        :param npv_date: date the NPV is discounted to; ``None`` means the curve
            reference date.
        :param spot_fx_settle_date: date the FX conversion applies as of;
            ``None`` means the curve reference date.
        """
        super().__init__()
        self._domestic_ccy: Currency = domestic_ccy
        self._domestic_ccy_discount_curve: YieldTermStructureProtocol = (
            domestic_ccy_discount_curve
        )
        self._foreign_ccy: Currency = foreign_ccy
        self._foreign_ccy_discount_curve: YieldTermStructureProtocol = (
            foreign_ccy_discount_curve
        )
        self._spot_fx: Quote = spot_fx
        self._include_settlement_date_flows: bool = include_settlement_date_flows
        self._settlement_date: Date | None = settlement_date
        self._npv_date: Date | None = npv_date
        self._spot_fx_settle_date: Date | None = spot_fx_settle_date

        # Observer wiring: curve / quote updates invalidate the instrument.
        for observable in (domestic_ccy_discount_curve, foreign_ccy_discount_curve):
            register = getattr(observable, "register_with", None)
            if register is not None:
                register(self)
        spot_fx.register_with(self)

    # --- inspectors --------------------------------------------------------

    def domestic_currency(self) -> Currency:
        return self._domestic_ccy

    def domestic_ccy_discount_curve(self) -> YieldTermStructureProtocol:
        return self._domestic_ccy_discount_curve

    def foreign_currency(self) -> Currency:
        return self._foreign_ccy

    def foreign_ccy_discount_curve(self) -> YieldTermStructureProtocol:
        return self._foreign_ccy_discount_curve

    def spot_fx(self) -> Quote:
        return self._spot_fx

    # --- helpers -----------------------------------------------------------

    @staticmethod
    def _resolve_date(candidate: Date | None, reference_date: Date, label: str) -> Date:
        """A null / omitted date means the curve reference date; else validate it.

        # C++ parity: the three ``if (x_ == Date()) ... else QL_REQUIRE(x_ >= referenceDate)``
        # blocks at the top of ``calculate``.
        """
        if candidate is None or candidate == Date():
            return reference_date
        qassert.require(
            candidate >= reference_date,
            f"{label} ({candidate}) cannot be before discount curve "
            f"reference date ({reference_date})",
        )
        return candidate

    def _spot_fx_rate(self, fx_settle: Date, reference_date: Date) -> float:
        """Spot FX, carried to ``fx_settle`` when that is not the reference date.

        Uses the parity relation between discount factors and FX rates,
        ``fx(T1)/fx(T2) = FwdDF_quote(T1->T2) / FwdDF_base(T1->T2)``, where
        ``fx`` is the currency ratio base/quote.
        """
        rate = self._spot_fx.value()
        if fx_settle == reference_date:
            return rate
        domestic_df = self._domestic_ccy_discount_curve.discount(fx_settle)
        foreign_df = self._foreign_ccy_discount_curve.discount(fx_settle)
        qassert.require(
            foreign_df != 0.0,
            f"discount factor associated with currency {self._foreign_ccy} "
            f"at maturity {fx_settle} cannot be zero",
        )
        return rate * domestic_df / foreign_df

    # --- engine interface --------------------------------------------------

    def calculate(self) -> None:
        """# C++ parity: ``DiscountingConstNotionalCrossCurrencySwapEngine::calculate``."""
        args = self._arguments
        results = self._results

        dom_curve = self._domestic_ccy_discount_curve
        for_curve = self._foreign_ccy_discount_curve
        reference_date = dom_curve.reference_date()
        qassert.require(
            reference_date == for_curve.reference_date(),
            "term structures should have the same reference date",
        )

        settlement = self._resolve_date(
            self._settlement_date, reference_date, "settlement date"
        )
        results.valuation_date = self._resolve_date(
            self._npv_date, reference_date, "NPV date"
        )
        fx_settle = self._resolve_date(
            self._spot_fx_settle_date, reference_date, "FX settlement date"
        )

        num_legs = len(args.legs)
        results.value = 0.0
        results.error_estimate = None
        results.leg_npv = [0.0] * num_legs
        results.leg_bps = [0.0] * num_legs
        start_discounts: list[float | None] = [0.0] * num_legs
        end_discounts: list[float | None] = [0.0] * num_legs
        results.start_discounts = start_discounts
        results.end_discounts = end_discounts
        results.in_ccy_leg_npv = [0.0] * num_legs
        results.in_ccy_leg_bps = [0.0] * num_legs
        results.npv_date_discounts = [0.0] * num_legs

        total_npv = 0.0
        for leg_no in range(num_legs):
            try:
                leg_currency = args.currencies[leg_no]
                if leg_currency == self._domestic_ccy:
                    leg_discount_curve = dom_curve
                else:
                    qassert.require(
                        leg_currency == self._foreign_ccy,
                        f"leg ccy ({leg_currency}) must be domesticCcy "
                        f"({self._domestic_ccy}) or foreignCcy ({self._foreign_ccy})",
                    )
                    leg_discount_curve = for_curve

                leg = args.legs[leg_no]

                results.npv_date_discounts[leg_no] = leg_discount_curve.discount(
                    results.valuation_date
                )

                npv, bps = CashFlows.npvbps(
                    leg,
                    leg_discount_curve,
                    self._include_settlement_date_flows,
                    settlement,
                    results.valuation_date,
                )
                results.in_ccy_leg_npv[leg_no] = npv * args.payer[leg_no]
                results.in_ccy_leg_bps[leg_no] = bps * args.payer[leg_no]

                results.leg_npv[leg_no] = results.in_ccy_leg_npv[leg_no]
                results.leg_bps[leg_no] = results.in_ccy_leg_bps[leg_no]

                if leg_currency != self._domestic_ccy:
                    spot_fx_rate = self._spot_fx_rate(fx_settle, reference_date)
                    results.leg_npv[leg_no] *= spot_fx_rate
                    results.leg_bps[leg_no] *= spot_fx_rate

                # C++ reports Null<DiscountFactor>() — ``None`` here — for a
                # date preceding the curve reference date.
                start_date = leg_start_date(leg)
                results.start_discounts[leg_no] = (
                    leg_discount_curve.discount(start_date)
                    if start_date >= reference_date
                    else None
                )

                maturity_date = leg_maturity_date(leg)
                results.end_discounts[leg_no] = (
                    leg_discount_curve.discount(maturity_date)
                    if maturity_date >= reference_date
                    else None
                )
            except Exception as e:
                msg = f"leg #{leg_no + 1}: {e}"
                raise type(e)(msg) from e

            total_npv += results.leg_npv[leg_no]

        results.value = total_npv


__all__ = ["DiscountingConstNotionalCrossCurrencySwapEngine"]
