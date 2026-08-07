"""Analytic engines for American digital (binary) options.

# C++ parity: ql/pricingengines/vanilla/analyticdigitalamericanengine.{hpp,cpp}
# (v1.43) — ``class AnalyticDigitalAmericanEngine`` and
# ``class AnalyticDigitalAmericanKOEngine``.

Both engines wrap the two closed-form calculators:

* ``exercise.payoff_at_expiry()`` is ``True``  -> :class:`AmericanPayoffAtExpiry`
* ``exercise.payoff_at_expiry()`` is ``False`` -> :class:`AmericanPayoffAtHit`

and the knock-in/knock-out distinction is nothing but the ``knock_in()``
virtual: ``True`` for :class:`AnalyticDigitalAmericanEngine`, ``False`` for
:class:`AnalyticDigitalAmericanKOEngine`.

The trap worth stating out loud
-------------------------------
``knock_in()`` is consulted **only on the at-expiry path**.
:class:`AmericanPayoffAtHit` has no knock-in parameter at all, so with
``payoff_at_expiry() == False`` the KO engine returns exactly the knock-in
price.  That is C++ v1.43 behaviour and the cross-validation pins it
(``digital_hit_cash_*_ko_equals_ki`` in the reference).

Which results are filled
------------------------
* at-expiry: **only** ``value``.  Every greek stays unset.
* at-hit: ``value``, ``delta``, ``gamma``, ``rho`` — and nothing else.  No
  vega, no theta, no dividend rho.  The C++ header's ``\\todo add more greeks
  (as of now only delta and rho available)`` is still open in v1.43.
"""

from __future__ import annotations

from pquantlib import qassert
from pquantlib.exercise import AmericanExercise
from pquantlib.instruments.one_asset_option import OneAssetOptionResults
from pquantlib.option import OptionArguments
from pquantlib.payoffs import StrikedTypePayoff
from pquantlib.pricingengines.american_payoff_at_expiry import AmericanPayoffAtExpiry
from pquantlib.pricingengines.american_payoff_at_hit import AmericanPayoffAtHit
from pquantlib.pricingengines.generic_engine import GenericEngine
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)


class AnalyticDigitalAmericanEngine(GenericEngine[OptionArguments, OneAssetOptionResults]):
    """Analytic engine for American vanilla options with a digital payoff.

    # C++ parity: ``class AnalyticDigitalAmericanEngine : public
    # VanillaOption::engine``.
    """

    def __init__(self, process: GeneralizedBlackScholesProcess) -> None:
        super().__init__(OptionArguments(), OneAssetOptionResults())
        self._process: GeneralizedBlackScholesProcess = process
        process.register_with(self)

    def knock_in(self) -> bool:
        """Whether the barrier knocks the option *in*.

        # C++ parity: ``virtual bool knock_in() const { return true; }``.
        """
        return True

    def calculate(self) -> None:
        """Dispatch to the at-hit or at-expiry closed form.

        # C++ parity: ``AnalyticDigitalAmericanEngine::calculate``.
        """
        args = self._arguments
        results = self._results

        qassert.require(args.exercise is not None, "no exercise given")
        assert args.exercise is not None
        qassert.require(
            isinstance(args.exercise, AmericanExercise),
            "non-American exercise given",
        )
        assert isinstance(args.exercise, AmericanExercise)
        exercise: AmericanExercise = args.exercise

        process = self._process
        qassert.require(
            exercise.dates()[0] <= process.black_volatility().reference_date(),
            "American option with window exercise not handled yet",
        )

        qassert.require(args.payoff is not None, "no payoff given")
        assert args.payoff is not None
        qassert.require(isinstance(args.payoff, StrikedTypePayoff), "non-striked payoff given")
        assert isinstance(args.payoff, StrikedTypePayoff)
        payoff: StrikedTypePayoff = args.payoff

        spot = process.state_variable().value()
        qassert.require(spot > 0.0, "negative or null underlying given")

        last_date = exercise.last_date()
        variance = process.black_volatility().black_variance(
            last_date, payoff.strike(), extrapolate=True
        )
        dividend_discount = process.dividend_yield().discount(last_date)
        risk_free_discount = process.risk_free_rate().discount(last_date)

        if exercise.payoff_at_expiry():
            at_expiry = AmericanPayoffAtExpiry(
                spot,
                risk_free_discount,
                dividend_discount,
                variance,
                payoff,
                self.knock_in(),
            )
            results.value = at_expiry.value()
        else:
            at_hit = AmericanPayoffAtHit(
                spot, risk_free_discount, dividend_discount, variance, payoff
            )
            results.value = at_hit.value()
            results.delta = at_hit.delta()
            results.gamma = at_hit.gamma()

            rfdc = process.risk_free_rate().day_counter()
            t = rfdc.year_fraction(process.risk_free_rate().reference_date(), last_date)
            results.rho = at_hit.rho(t)


class AnalyticDigitalAmericanKOEngine(AnalyticDigitalAmericanEngine):
    """Analytic engine for American *knock-out* digital options.

    # C++ parity: ``class AnalyticDigitalAmericanKOEngine :
    # public AnalyticDigitalAmericanEngine`` — the whole subclass is the
    # single overridden ``knock_in()``.
    """

    def knock_in(self) -> bool:
        """Always ``False``.

        # C++ parity: ``bool knock_in() const override { return false; }``.
        """
        return False


__all__ = ["AnalyticDigitalAmericanEngine", "AnalyticDigitalAmericanKOEngine"]
