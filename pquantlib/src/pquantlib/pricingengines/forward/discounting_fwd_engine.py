"""DiscountingFwdEngine — discount-curve FX-forward pricer.

# C++ parity: ql/pricingengines/forward/discountingfxforwardengine.{hpp,cpp}
#             (v1.42.1).  Renamed ``Fx`` → ``Fwd`` in PQuantLib per the
#             L3-E spec — at this layer the engine is FX-specific but
#             carry the more generic name to match downstream callers
#             that may extend it.

Pricing formula (paySourceCurrency=True, NPV in source-currency terms):

    NPV  = -sourceNominal * df_source(T) + (targetNominal * df_target(T)) / spotFx

with df_source(t) and df_target(t) normalised to the settlement date
(i.e. ``df(t) = curve.discount(t) / curve.discount(settlementDate)``).

Fair forward rate:

    F = spotFx * df_target(T) / df_source(T)
"""

from __future__ import annotations

from pquantlib import qassert
from pquantlib.instruments.fx_forward import FxForwardArguments, FxForwardResults
from pquantlib.pricingengines.generic_engine import GenericEngine
from pquantlib.quotes.quote import Quote
from pquantlib.termstructures.protocols import YieldTermStructureProtocol
from pquantlib.time.date import Date


class DiscountingFwdEngine(GenericEngine[FxForwardArguments, FxForwardResults]):
    """Discounting engine for FxForward instruments.

    # C++ parity: ``class DiscountingFxForwardEngine : public
    # FxForward::engine``.
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
        # Observer wiring: the engine notifies the attached instrument
        # when the underlying spot quote changes.
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
        """Compute fair forward + NPV.

        # C++ parity: ``DiscountingFxForwardEngine::calculate``.
        """
        args = self._arguments
        results = self._results

        # The settlement date should have been pre-set by the
        # instrument's setup_arguments (FxForward forwards its own
        # settlement_date computed from the payment calendar).
        settlement_date = args.settlement_date
        qassert.require(
            settlement_date != Date(),
            "settlement date not set by instrument",
        )

        maturity_date = args.maturity_date
        qassert.require(maturity_date != Date(), "maturity date not set")

        # Curve reference-date validation (mirrors C++ checks).
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

        # Discount factors normalised to settlement date.
        df_source = self._source_discount.discount(
            maturity_date
        ) / self._source_discount.discount(settlement_date)
        df_target = self._target_discount.discount(
            maturity_date
        ) / self._target_discount.discount(settlement_date)

        # Fair forward rate.
        results.fair_forward_rate = spot_fx_rate * df_target / df_source

        # PV in each currency.
        qassert.require(args.source_nominal is not None, "source nominal missing")
        qassert.require(args.target_nominal is not None, "target nominal missing")
        assert args.source_nominal is not None
        assert args.target_nominal is not None
        pv_source = args.source_nominal * df_source
        pv_target = args.target_nominal * df_target
        pv_target_in_source = pv_target / spot_fx_rate

        if args.pay_source_currency:
            npv_in_source = -pv_source + pv_target_in_source
        else:
            npv_in_source = pv_source - pv_target_in_source

        results.value = npv_in_source
        results.error_estimate = None
        results.npv_source_currency = npv_in_source
        results.npv_target_currency = npv_in_source * spot_fx_rate
        # Additional results for inspection.
        results.additional_results = {
            "spotFx": spot_fx_rate,
            "sourceCurrencyDiscountFactor": df_source,
            "targetCurrencyDiscountFactor": df_target,
            "sourceCurrencyPV": pv_source,
            "targetCurrencyPV": pv_target,
        }


__all__ = ["DiscountingFwdEngine"]
