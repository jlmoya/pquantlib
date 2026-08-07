"""AnalyticBSMHullWhiteEngine — European vanilla under Black-Scholes + Hull-White rates.

# C++ parity: ql/pricingengines/vanilla/analyticbsmhullwhiteengine.{hpp,cpp}
# (v1.43) — ``class AnalyticBSMHullWhiteEngine : public
# GenericModelEngine<HullWhite, VanillaOption::arguments, VanillaOption::results>``.

When the short rate follows Hull-White

    dr_t = (theta(t) - a r_t) dt + sigma dW_r

and the equity Brownian is correlated with ``dW_r`` at a constant ``rho``, the
T-forward-measure variance of ``ln S_T`` picks up a closed-form offset. The
engine therefore does not integrate anything: it shifts the Black *variance* by

    v  = sigma^2/a^2 * (t + 2/a e^{-a t} - 1/(2a) e^{-2 a t} - 3/(2a))
    mu = 2 rho sigma eta / a * (t - 1/a (1 - e^{-a t}))

(``eta`` being the input Black vol at the option's strike and maturity), wraps
the process's vol surface in one that adds ``v + mu`` to every variance, and
delegates to :class:`~pquantlib.pricingengines.vanilla.analytic_european_engine.AnalyticEuropeanEngine`.

Below ``a*t <= QL_EPSILON**0.25`` (which is exactly ``2**-13``) C++ switches to
the algebraic small-``a`` limit

    v  = sigma^2 t^3 (1/3 - a t/4 + 7 a^2 t^2/60)
    mu = rho sigma eta t^2 (1 - a t/3 + a^2 t^2/12)

The test is a strict ``>``, so ``a*t`` exactly equal to the threshold takes the
*low* branch. Both sides are pinned bit-for-bit by
``bsmhw_a_at_threshold`` / ``bsmhw_a_just_above_threshold``.

Reference: Brigo, Mercurio, *Interest Rate Models*.

Consequences of the delegation, which a port must reproduce
-----------------------------------------------------------
* **The full Greek block is produced**, because C++ assigns
  ``results_ = *dynamic_cast<const OneAssetOption::results*>(
  bsmEngine->getResults())`` — the whole result object of the inner engine,
  Greeks and ``additionalResults`` included. ``vega`` and the
  ``"volatility"`` additional result are therefore taken against the
  *shifted* surface, not the input one.
* **The engine has no exercise-type check of its own.** An American exercise
  still fails, but with the inner engine's "not a European option" message.
* ``rho`` genuinely moves the price: it enters ``mu`` linearly.
"""

from __future__ import annotations

import math

from pquantlib import qassert
from pquantlib.instruments.one_asset_option import OneAssetOptionResults
from pquantlib.instruments.vanilla_option import VanillaOption
from pquantlib.math.constants import QL_EPSILON
from pquantlib.models.shortrate.onefactor.hull_white import HullWhite
from pquantlib.option import OptionArguments
from pquantlib.payoffs import StrikedTypePayoff
from pquantlib.pricingengines.generic_engine import GenericEngine
from pquantlib.pricingengines.vanilla.analytic_european_engine import (
    AnalyticEuropeanEngine,
)
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)
from pquantlib.termstructures.volatility.equity_fx.black_vol_term_structure import (
    BlackVolTermStructure,
)
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.date import Date


class _ShiftedBlackVolTermStructure(BlackVolTermStructure):
    """A Black vol surface with a constant offset added to every variance.

    # C++ parity: ``class ShiftedBlackVolTermStructure`` in the anonymous
    # namespace at analyticbsmhullwhiteengine.cpp:33-66. Anonymous-namespace
    # type with no public C++ name, so it is private here too.

    The C++ constructor forces ``Following`` as the business-day convention and
    otherwise mirrors the wrapped surface's reference date, calendar and day
    counter.
    """

    __slots__ = ("_variance_offset", "_vol_ts")

    def __init__(self, variance_offset: float, vol_ts: BlackVolTermStructure) -> None:
        super().__init__(
            business_day_convention=BusinessDayConvention.Following,
            reference_date=vol_ts.reference_date(),
            calendar=vol_ts.calendar(),
            day_counter=vol_ts.day_counter(),
        )
        self._variance_offset: float = variance_offset
        self._vol_ts: BlackVolTermStructure = vol_ts

    def min_strike(self) -> float:
        return self._vol_ts.min_strike()

    def max_strike(self) -> float:
        return self._vol_ts.max_strike()

    def max_date(self) -> Date:
        return self._vol_ts.max_date()

    def _black_variance_impl(self, t: float, strike: float) -> float:
        # C++ parity: cpp:53-55 — note extrapolate=true on the inner lookup.
        return (
            self._vol_ts.black_variance_at_time(t, strike, extrapolate=True)
            + self._variance_offset
        )

    def _black_vol_impl(self, t: float, strike: float) -> float:
        # C++ parity: cpp:56-60 — the t == 0 substitution is 1e-5, not QL_EPSILON.
        non_zero_maturity = 0.00001 if t == 0.0 else t
        var = self._black_variance_impl(non_zero_maturity, strike)
        return math.sqrt(var / non_zero_maturity)


class AnalyticBSMHullWhiteEngine(GenericEngine[OptionArguments, OneAssetOptionResults]):
    """European vanilla under Black-Scholes with a Hull-White short rate.

    # C++ parity: ``class AnalyticBSMHullWhiteEngine`` in
    # analyticbsmhullwhiteengine.hpp:41-56.
    """

    def __init__(
        self,
        equity_short_rate_correlation: float,
        process: GeneralizedBlackScholesProcess,
        model: HullWhite,
    ) -> None:
        """# C++ parity: analyticbsmhullwhiteengine.cpp:68-77."""
        super().__init__(OptionArguments(), OneAssetOptionResults())
        # C++ parity: cpp:74-75 — QL_REQUIRE on both the process and the model.
        # The annotations say neither can be None, but annotations are not
        # enforced at runtime and C++'s shared_ptr can be null, so the checks are
        # kept: they turn a later AttributeError into the C++ message.
        qassert.require(
            process is not None,  # pyright: ignore[reportUnnecessaryComparison]
            "no Black-Scholes process specified",
        )
        qassert.require(
            model is not None,  # pyright: ignore[reportUnnecessaryComparison]
            "no Hull-White model specified",
        )
        self._rho: float = float(equity_short_rate_correlation)
        self._process: GeneralizedBlackScholesProcess = process
        self._model: HullWhite = model
        model.register_with(self)
        # C++ parity: cpp:76 — registerWith(process_).
        process.register_with(self)

    # --- inspectors -----------------------------------------------------

    def model(self) -> HullWhite:
        """The Hull-White model (``model_`` in the GenericModelEngine base)."""
        return self._model

    def rho(self) -> float:
        """Equity / short-rate correlation supplied at construction."""
        return self._rho

    # --- calculate ------------------------------------------------------

    def calculate(self) -> None:
        """# C++ parity: ``AnalyticBSMHullWhiteEngine::calculate`` at
        # analyticbsmhullwhiteengine.cpp:79-131.
        """
        args = self._arguments
        results = self._results

        process = self._process
        # C++ parity: cpp:81 — QL_REQUIRE(process_->x0() > 0.0, ...).
        qassert.require(process.x0() > 0.0, "negative or null underlying given")

        qassert.require(args.exercise is not None, "no exercise given")
        qassert.require(args.payoff is not None, "no payoff given")
        assert args.exercise is not None
        assert args.payoff is not None
        qassert.require(
            isinstance(args.payoff, StrikedTypePayoff), "non-striked payoff given"
        )
        assert isinstance(args.payoff, StrikedTypePayoff)
        payoff: StrikedTypePayoff = args.payoff
        exercise = args.exercise
        last_date = exercise.last_date()

        risk_free = process.risk_free_rate()
        # C++ parity: cpp:90-92 — the risk-free curve's own day counter and
        # reference date.
        t = risk_free.day_counter().year_fraction(risk_free.reference_date(), last_date)

        a = self._model.a()
        sigma = self._model.sigma()
        eta = process.black_volatility().black_vol(last_date, payoff.strike())

        # C++ parity: cpp:101 — strict `>` against pow(QL_EPSILON, 0.25) == 2**-13.
        if a * t > math.pow(QL_EPSILON, 0.25):
            v = (
                sigma
                * sigma
                / (a * a)
                * (
                    t
                    + 2.0 / a * math.exp(-a * t)
                    - 1.0 / (2.0 * a) * math.exp(-2.0 * a * t)
                    - 3.0 / (2.0 * a)
                )
            )
            mu = (
                2.0
                * self._rho
                * sigma
                * eta
                / a
                * (t - 1.0 / a * (1.0 - math.exp(-a * t)))
            )
        else:
            # C++ parity: cpp:110-114 — low-a algebraic limit.
            v = sigma * sigma * t * t * t * (1.0 / 3.0 - 0.25 * a * t + 7.0 / 60.0 * a * a * t * t)
            mu = self._rho * sigma * eta * t * t * (1.0 - a * t / 3.0 + a * a * t * t / 12.0)

        variance_offset = v + mu

        vol_ts = _ShiftedBlackVolTermStructure(
            variance_offset, process.black_volatility()
        )
        adj_process = GeneralizedBlackScholesProcess(
            x0=process.state_variable(),
            dividend_ts=process.dividend_yield(),
            risk_free_ts=process.risk_free_rate(),
            black_vol_ts=vol_ts,
        )

        bsm_engine = AnalyticEuropeanEngine(adj_process)
        # C++ parity: cpp:126-129 — a fresh VanillaOption pushes the payoff and
        # exercise into the inner engine's arguments, then calculate() is called
        # directly (the inner engine, not the option, is what is driven).
        VanillaOption(payoff, exercise).setup_arguments(bsm_engine.get_arguments())
        bsm_engine.calculate()

        # C++ parity: cpp:130-131 — the WHOLE OneAssetOption::results block is
        # copied, Greeks and additionalResults included.
        inner = bsm_engine.get_results()
        results.reset()
        results.value = inner.value
        results.error_estimate = inner.error_estimate
        results.valuation_date = inner.valuation_date
        results.additional_results = dict(inner.additional_results)
        results.delta = inner.delta
        results.gamma = inner.gamma
        results.theta = inner.theta
        results.vega = inner.vega
        results.rho = inner.rho
        results.dividend_rho = inner.dividend_rho
        results.itm_cash_probability = inner.itm_cash_probability
        results.delta_forward = inner.delta_forward
        results.elasticity = inner.elasticity
        results.theta_per_day = inner.theta_per_day
        results.strike_sensitivity = inner.strike_sensitivity


__all__ = ["AnalyticBSMHullWhiteEngine"]
