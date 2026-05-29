"""BlackCDSOptionEngine — Black 1976 engine for ``CDSOption``.

# C++ parity: ql/experimental/credit/blackcdsoptionengine.{hpp,cpp} (v1.42.1).

Computes the Black-76 value of an option on a running-spread CDS using
the underlying's fair spread as the forward and its coupon-leg PV as
the annuity. For a non-knock-out payer option the engine adds the
front-end-protection contribution (``notional * (1 - recovery) *
P_default(exercise) * D(exercise)``).

Inputs:

* ``probability``: default-probability term structure used for the
  front-end-protection contribution.
* ``recovery_rate``: float in [0,1].
* ``discount_curve``: yield curve used to discount the option payoff
  and to read the day-count convention for the option-tenor.
* ``volatility``: lognormal volatility quote (the engine reads
  ``.value()`` at each calculation).
"""

from __future__ import annotations

import math

from pquantlib import qassert
from pquantlib.instruments.cds_option import (
    CDSOptionArguments,
    CDSOptionResults,
)
from pquantlib.instruments.credit_default_swap import ProtectionSide
from pquantlib.payoffs import OptionType
from pquantlib.pricingengines.black_formula import black_formula
from pquantlib.pricingengines.generic_engine import GenericEngine
from pquantlib.quotes.quote import Quote
from pquantlib.termstructures.credit.default_probability_term_structure import (
    DefaultProbabilityTermStructure,
)
from pquantlib.termstructures.yield_term_structure import YieldTermStructure


class BlackCDSOptionEngine(
    GenericEngine[CDSOptionArguments, CDSOptionResults],
):
    """Black-76 engine for ``CDSOption``."""

    def __init__(
        self,
        probability: DefaultProbabilityTermStructure,
        recovery_rate: float,
        discount_curve: YieldTermStructure,
        volatility: Quote,
    ) -> None:
        super().__init__(CDSOptionArguments(), CDSOptionResults())
        self._probability: DefaultProbabilityTermStructure = probability
        self._recovery_rate: float = recovery_rate
        self._discount_curve: YieldTermStructure = discount_curve
        self._volatility: Quote = volatility
        probability.register_with(self)
        discount_curve.register_with(self)
        volatility.register_with(self)

    def calculate(self) -> None:
        """Compute the option NPV + risky-annuity result.

        # C++ parity: blackcdsoptionengine.cpp:42-80.
        """
        args = self._arguments
        results = self._results
        assert args.swap is not None
        assert args.exercise is not None
        assert args.side is not None
        assert args.notional is not None
        assert args.spread is not None

        maturity_date = args.swap.coupons()[0].date()
        exercise_date = args.exercise.dates()[0]
        qassert.require(
            maturity_date > exercise_date,
            "Underlying CDS should start after option maturity",
        )
        settlement = self._discount_curve.reference_date()
        ts_dc = self._discount_curve.day_counter()

        spot_fwd_spread = args.swap.fair_spread()
        swap_spread = args.swap.running_spread()

        # Risky annuity: absolute coupon-leg NPV per unit spread (the
        # sign of the underlying is encoded by `side`, but the annuity
        # is sign-free).
        risky_annuity = abs(args.swap.coupon_leg_npv() / swap_spread)
        results.risky_annuity = risky_annuity

        time_to_exercise = ts_dc.year_fraction(settlement, exercise_date)
        std_dev = self._volatility.value() * math.sqrt(time_to_exercise)

        call_put = (
            OptionType.Call if args.side == ProtectionSide.Buyer
            else OptionType.Put
        )

        results.value = black_formula(
            call_put, swap_spread, spot_fwd_spread, std_dev, risky_annuity
        )

        # Non-knock-out payer option: front-end protection
        # contribution. Sign tracks Option::Type integer (Call=+1, Put=-1).
        if args.side == ProtectionSide.Buyer and not args.knocks_out:
            sign = int(call_put)
            front_end_protection = (
                sign
                * args.notional
                * (1.0 - self._recovery_rate)
                * self._probability.default_probability(exercise_date)
                * self._discount_curve.discount(exercise_date)
            )
            results.value += front_end_protection

        # Populate the inherited InstrumentResults fields read by
        # Instrument.fetch_results.
        results.error_estimate = None


__all__ = ["BlackCDSOptionEngine"]
