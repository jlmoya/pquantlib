"""LfmSwaptionEngine — Black swaption pricing off a LiborForwardModel.

# C++ parity: ql/legacy/libormarketmodels/lfmswaptionengine.{hpp,cpp} (v1.43).

Reads the model's Rebonato swaption volatility matrix, looks up the volatility
at the swaption's (exercise time, swap length) and applies the Black formula
scaled by the fixed leg's annuity (``fixedLegBPS / 1bp``).

Handle indirection: C++ takes ``Handle<YieldTermStructure> discountCurve`` and
``registerWith``s it. This port has no ``Handle``; the curve is threaded
directly, and the engine registers with it when the object supports observer
registration (matching what other PQuantLib engines do).
"""

from __future__ import annotations

import math

from pquantlib import qassert
from pquantlib.instruments.swap import SwapType
from pquantlib.instruments.swaption import (
    SettlementMethod,
    SwaptionArguments,
    SwaptionResults,
)
from pquantlib.legacy.libormarketmodels.libor_forward_model import LiborForwardModel
from pquantlib.payoffs import OptionType
from pquantlib.pricingengines.black_formula import black_formula
from pquantlib.pricingengines.generic_engine import GenericEngine
from pquantlib.pricingengines.swap.discounting_swap_engine import DiscountingSwapEngine
from pquantlib.termstructures.protocols import YieldTermStructureProtocol
from pquantlib.time.date import Date

# C++ parity: ``static const Spread basisPoint = 1.0e-4;``
# (lfmswaptionengine.cpp:40).
_BASIS_POINT = 1.0e-4


class LfmSwaptionEngine(GenericEngine[SwaptionArguments, SwaptionResults]):
    """LIBOR forward model swaption engine based on the Black formula.

    # C++ parity: ``class LfmSwaptionEngine : public
    # GenericModelEngine<LiborForwardModel, Swaption::arguments,
    # Swaption::results>`` (lfmswaptionengine.hpp:35-45).

    # C++ parity divergence: PQuantLib has no ``GenericModelEngine``; the
    # model is held as a plain attribute on a ``GenericEngine`` subclass,
    # which is what every other model-driven engine in this port does
    # (e.g. ``JamshidianSwaptionEngine``).
    """

    def __init__(
        self,
        model: LiborForwardModel,
        discount_curve: YieldTermStructureProtocol,
    ) -> None:
        # C++ parity: lfmswaptionengine.cpp:27-32.
        super().__init__(SwaptionArguments(), SwaptionResults())
        self._model: LiborForwardModel = model
        self._discount_curve: YieldTermStructureProtocol = discount_curve
        # C++ ``registerWith(discountCurve_)``; also registers with the model
        # via GenericModelEngine's ctor.
        register = getattr(discount_curve, "register_with", None)
        if register is not None:
            register(self)
        model.register_with(self)

    # --- inspectors -------------------------------------------------------

    def model(self) -> LiborForwardModel:
        """The pricing model.

        # C++ parity: ``GenericModelEngine::model_`` (protected).
        """
        return self._model

    def discount_curve(self) -> YieldTermStructureProtocol:
        """The discounting curve.

        # C++ parity: ``LfmSwaptionEngine::discountCurve_`` (private).
        """
        return self._discount_curve

    # --- pricing ----------------------------------------------------------

    def calculate(self) -> None:
        """# C++ parity: ``LfmSwaptionEngine::calculate``
        (lfmswaptionengine.cpp:35-70).
        """
        args = self._arguments

        qassert.require(
            args.settlement_method != SettlementMethod.ParYieldCurve,
            "cash settled (ParYieldCurve) swaptions not priced with Lfm engine",
        )

        swap = args.swap
        qassert.require(swap is not None, "swap not set")
        assert swap is not None
        swap.set_pricing_engine(DiscountingSwapEngine(self._discount_curve, False))

        correction = swap.spread() * abs(swap.floating_leg_bps() / swap.fixed_leg_bps())
        fixed_rate = swap.fixed_rate() - correction
        fair_rate = swap.fair_rate() - correction

        volatility = self._model.get_swaption_volatility_matrix()

        reference_date = volatility.reference_date()
        day_counter = volatility.day_counter()

        exercise_obj = args.exercise
        qassert.require(exercise_obj is not None, "exercise not set")
        assert exercise_obj is not None
        exercise = day_counter.year_fraction(reference_date, exercise_obj.date(0))
        # ``FixedVsFloatingSwapArguments`` types its date lists as
        # ``list[object]``; narrow here rather than reach outside legacy/.
        last_pay = args.fixed_pay_dates[-1]
        first_reset = args.fixed_reset_dates[0]
        assert isinstance(last_pay, Date)
        assert isinstance(first_reset, Date)
        swap_length = day_counter.year_fraction(
            reference_date, last_pay
        ) - day_counter.year_fraction(reference_date, first_reset)

        w = OptionType.Call if args.swap_type == SwapType.Payer else OptionType.Put
        vol = volatility.volatility(exercise, swap_length, fair_rate, True)
        self._results.value = (swap.fixed_leg_bps() / _BASIS_POINT) * black_formula(
            w, fixed_rate, fair_rate, vol * math.sqrt(exercise)
        )


__all__ = ["LfmSwaptionEngine"]
