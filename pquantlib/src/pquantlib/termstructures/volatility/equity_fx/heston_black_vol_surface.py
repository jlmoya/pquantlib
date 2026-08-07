"""HestonBlackVolSurface — the Black vol surface implied by a Heston model.

# C++ parity: ql/termstructures/volatility/equityfx/hestonblackvolsurface.hpp +
#             hestonblackvolsurface.cpp (v1.43).

Prices a European vanilla under the Heston dynamics and inverts the Black
formula to get the implied volatility, so a Heston model can stand in
anywhere a ``BlackVolTermStructure`` is wanted. Two details that matter:

* The option type is chosen per query — ``Put`` when the strike is below the
  forward, ``Call`` when it is above — so the inversion always works on the
  out-of-the-money option, where vega is largest and the root is best
  conditioned.
* A non-positive Heston price means the strike is so far out of the money that
  the Fourier inversion has underflowed; C++ then returns ``sqrt(theta)``, the
  long-run Heston volatility, rather than failing.

The constructor takes C++'s two engine tuning knobs —
``AnalyticHestonEngine::ComplexLogFormula`` (default ``AngledContour``) and
``AnalyticHestonEngine::Integration`` (default ``gaussLaguerre(160)``) — and
forwards them verbatim. An earlier revision omitted both and built the engine
from its default constructor, because ``AnalyticHestonEngine`` then implemented
only the Gatheral complex-log branch over ``scipy.integrate.quad`` and had
nothing for the arguments to select. The engine now implements the full set, so
the divergence is closed: this surface picks the same contour and the same
quadrature as C++, not merely a comparably accurate one.
"""

from __future__ import annotations

import math
import sys

from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.math.solvers1d.brent import Brent
from pquantlib.models.equity.heston_model import HestonModel
from pquantlib.payoffs import OptionType, PlainVanillaPayoff
from pquantlib.pricingengines.black_formula import black_formula
from pquantlib.pricingengines.vanilla.analytic_heston_engine import (
    AnalyticHestonEngine,
    ComplexLogFormula,
    Integration,
)
from pquantlib.termstructures.volatility.equity_fx.black_vol_term_structure import (
    BlackVolTermStructure,
)
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.null_calendar import NullCalendar
from pquantlib.time.date import Date


class HestonBlackVolSurface(BlackVolTermStructure):
    """Black volatility surface backed by a Heston model."""

    def __init__(
        self,
        heston_model: HestonModel,
        cpx_log_formula: ComplexLogFormula = ComplexLogFormula.AngledContour,
        integration: Integration | None = None,
    ) -> None:
        """# C++ parity: ``HestonBlackVolSurface`` ctor
        # (hestonblackvolsurface.hpp:36-41, .cpp:46-57).

        ``integration`` defaults to ``Integration::gaussLaguerre(160)``, spelled
        as ``None`` here only because a mutable default cannot be a class-level
        expression; C++'s default argument is that same 160-point rule.
        """
        process = heston_model.process()
        super().__init__(
            business_day_convention=BusinessDayConvention.Following,
            reference_date=process.risk_free_rate().reference_date(),
            calendar=NullCalendar(),
            day_counter=process.risk_free_rate().day_counter(),
        )
        self._heston_model: HestonModel = heston_model
        self._cpx_log_formula: ComplexLogFormula = cpx_log_formula
        self._integration: Integration = (
            Integration.gauss_laguerre(160) if integration is None else integration
        )
        self._heston_model.register_with(self)

    # --- TermStructure interface -------------------------------------------

    def day_counter(self) -> DayCounter:
        return self._heston_model.process().risk_free_rate().day_counter()

    def max_date(self) -> Date:
        return Date.max_date()

    # --- VolatilityTermStructure interface ---------------------------------

    def min_strike(self) -> float:
        return 0.0

    def max_strike(self) -> float:
        # C++ parity: ``std::numeric_limits<Real>::max()``.
        return sys.float_info.max

    # --- BlackVolTermStructure interface -----------------------------------

    def atm_level(self, t: float) -> float:
        """Forward at time ``t``.

        # C++ parity: ``HestonBlackVolSurface::atmLevel``.
        """
        process = self._heston_model.process()
        return (
            process.s0().value()
            * process.dividend_yield().discount(t, True)
            / process.risk_free_rate().discount(t, True)
        )

    def _black_variance_impl(self, t: float, strike: float) -> float:
        vol = self._black_vol_impl(t, strike)
        return vol * vol * t

    def _black_vol_impl(self, t: float, strike: float) -> float:
        # C++ parity: hestonblackvolsurface.cpp:84-85 — the engine is rebuilt
        # per call from the stored formula and integration rule.
        heston_engine = AnalyticHestonEngine.with_integration(
            self._heston_model, self._cpx_log_formula, self._integration
        )
        process = self._heston_model.process()

        df = process.risk_free_rate().discount(t, True)
        fwd = process.s0().value() * process.dividend_yield().discount(t, True) / df

        payoff = PlainVanillaPayoff(
            OptionType.Put if fwd > strike else OptionType.Call, strike
        )
        npv = heston_engine.price_vanilla_payoff(payoff=payoff, maturity=t)

        theta = self._heston_model.theta()
        if npv <= 0.0:
            return math.sqrt(theta)

        solver = Brent()
        solver.set_max_evaluations(10000)
        guess = math.sqrt(theta)
        # C++ parity: ``std::numeric_limits<double>::epsilon()``; Solver1D
        # floors the requested accuracy at QL_EPSILON anyway.
        accuracy = sys.float_info.epsilon

        def objective(v: float) -> float:
            # C++ parity: the anonymous-namespace ``blackValue`` helper.
            return (
                black_formula(
                    payoff.option_type(),
                    strike,
                    fwd,
                    max(0.0, v) * math.sqrt(t),
                    df,
                )
                - npv
            )

        return solver.solve(objective, accuracy, guess, 0.01)


__all__ = ["HestonBlackVolSurface"]
