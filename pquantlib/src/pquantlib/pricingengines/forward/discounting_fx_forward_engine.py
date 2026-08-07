"""DiscountingFxForwardEngine — discount-curve FX-forward pricer.

# C++ parity: ql/pricingengines/forward/discountingfxforwardengine.{hpp,cpp}
#             (v1.43), ``class DiscountingFxForwardEngine : public FxForward::engine``.

Discounts the two legs of an :class:`~pquantlib.instruments.fx_forward.FxForward`
on their own currency curves. Everything is measured **from the settlement
date**: the discount factors used for the legs are
``curve.discount(T) / curve.discount(settlement)``, and the settlement-date NPV
is then pushed back to the curve reference date.

Formulae (C++ discountingfxforwardengine.cpp:37-121)::

    dfSource = D_src(T) / D_src(tau)          dfTarget = D_tgt(T) / D_tgt(tau)
    fairForwardRate = S * dfSource / dfTarget
    pvSource = N_src * dfSource               pvTarget = N_tgt * dfTarget
    npvAtSettlement = -/+ pvSource +/- pvTarget / S       (sign from paySourceCurrency)
    npvSourceCurrency = npvAtSettlement * D_src(tau)
    npvTargetCurrency = npvAtSettlement * S * D_tgt(tau)

Three things a port gets wrong if it reasons from first principles instead of
reading the source:

* the fair forward rate is ``S * dfSource / dfTarget``, not the reciprocal —
  v1.43 inverted this ratio relative to the older engine;
* ``npvTargetCurrency`` is **not** ``npvSourceCurrency * S``: the two use
  different settlement discount factors (``D_src(tau)`` vs ``D_tgt(tau)``);
* seven ``additionalResults`` are published under exact C++ keys, and there are
  exactly seven — they are pinned by name and count.

Guards: both curve reference dates must be on or before the settlement date,
and the spot quote must be strictly positive. There is deliberately **no**
guard on maturity preceding settlement — C++ has none, the settlement-
normalised discount factors simply exceed 1, and that behaviour is pinned.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pquantlib import qassert
from pquantlib.instruments.fx_forward import FxForwardArguments, FxForwardResults
from pquantlib.pricingengines.generic_engine import GenericEngine
from pquantlib.time.date import Date

if TYPE_CHECKING:
    from pquantlib.quotes.quote import Quote
    from pquantlib.termstructures.protocols import YieldTermStructureProtocol


class DiscountingFxForwardEngine(GenericEngine[FxForwardArguments, FxForwardResults]):
    """Discounting engine for ``FxForward`` instruments.

    # C++ parity: ``class DiscountingFxForwardEngine``
    # (discountingfxforwardengine.hpp:66-93).
    """

    def __init__(
        self,
        source_currency_discount_curve: YieldTermStructureProtocol,
        target_currency_discount_curve: YieldTermStructureProtocol,
        spot_fx: Quote,
    ) -> None:
        super().__init__(FxForwardArguments(), FxForwardResults())
        self._source_discount: YieldTermStructureProtocol = source_currency_discount_curve
        self._target_discount: YieldTermStructureProtocol = target_currency_discount_curve
        self._spot_fx: Quote = spot_fx
        # C++ parity: discountingfxforwardengine.cpp:33-35 — registerWith all three.
        for curve in (source_currency_discount_curve, target_currency_discount_curve):
            register = getattr(curve, "register_with", None)
            if register is not None:
                register(self)
        spot_fx.register_with(self)

    # --- inspectors --------------------------------------------------------

    def source_currency_discount_curve(self) -> YieldTermStructureProtocol:
        return self._source_discount

    def target_currency_discount_curve(self) -> YieldTermStructureProtocol:
        return self._target_discount

    def spot_fx(self) -> Quote:
        return self._spot_fx

    # --- engine interface --------------------------------------------------

    def calculate(self) -> None:
        """C++ parity: ``DiscountingFxForwardEngine::calculate`` (cpp:37-121)."""
        args = self._arguments
        results = self._results

        results.value = 0.0
        results.error_estimate = None

        settlement_date = args.settlement_date
        qassert.require(settlement_date != Date(), "settlement date not set by instrument")
        maturity_date = args.maturity_date
        qassert.require(maturity_date != Date(), "maturity date not set")

        source_ref = self._source_discount.reference_date()
        target_ref = self._target_discount.reference_date()
        qassert.require(
            source_ref <= settlement_date,
            f"source currency discount curve reference date ({source_ref}) "
            f"must be on or before settlement date ({settlement_date})",
        )
        qassert.require(
            target_ref <= settlement_date,
            f"target currency discount curve reference date ({target_ref}) "
            f"must be on or before settlement date ({settlement_date})",
        )

        spot_fx_rate = self._spot_fx.value()
        qassert.require(spot_fx_rate > 0.0, "spot FX rate must be positive")

        df_source_settlement = self._source_discount.discount(settlement_date)
        df_target_settlement = self._target_discount.discount(settlement_date)
        df_source = self._source_discount.discount(maturity_date) / df_source_settlement
        df_target = self._target_discount.discount(maturity_date) / df_target_settlement

        results.fair_forward_rate = spot_fx_rate * df_source / df_target

        qassert.require(args.source_nominal is not None, "source nominal missing")
        qassert.require(args.target_nominal is not None, "target nominal missing")
        assert args.source_nominal is not None
        assert args.target_nominal is not None
        pv_source = args.source_nominal * df_source
        pv_target = args.target_nominal * df_target
        pv_target_in_source = pv_target / spot_fx_rate

        if args.pay_source_currency:
            npv_at_settlement_in_source = -pv_source + pv_target_in_source
        else:
            npv_at_settlement_in_source = pv_source - pv_target_in_source

        npv_in_source = npv_at_settlement_in_source * df_source_settlement
        npv_in_target = npv_at_settlement_in_source * spot_fx_rate * df_target_settlement

        results.value = npv_in_source
        results.npv_source_currency = npv_in_source
        results.npv_target_currency = npv_in_target
        results.additional_results = {
            "spotFx": spot_fx_rate,
            "sourceCurrencyDiscountFactor": df_source,
            "targetCurrencyDiscountFactor": df_target,
            "sourceCurrencySettlementDiscountFactor": df_source_settlement,
            "targetCurrencySettlementDiscountFactor": df_target_settlement,
            "sourceCurrencyPV": pv_source,
            "targetCurrencyPV": pv_target,
        }


__all__ = ["DiscountingFxForwardEngine"]
