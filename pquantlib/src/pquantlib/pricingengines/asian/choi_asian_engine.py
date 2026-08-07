"""ChoiAsianEngine — discrete arithmetic Asian option via the Choi basket.

# C++ parity:
# ql/pricingengines/asian/choiasianengine.{hpp,cpp} (v1.43),
# ``QuantLib::ChoiAsianEngine``.

Jaehyuk Choi, *Sum of all Black-Scholes-Merton Models: An efficient Pricing
Method for Spread, Basket and Asian Options*,
https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2913048

The arithmetic average of ``m`` future fixings of one asset is an equally
weighted basket of ``m`` perfectly-cointegrated assets, so the option is
replicated as a :class:`ChoiBasketEngine` basket. Three branches:

``future_fixings == 0``
    Everything is already fixed: the discounted intrinsic on
    ``running_accumulator / past_fixings``.
``future_fixings == 1``
    One unknown fixing — a plain :func:`black_formula`, no basket at all.
``future_fixings > 1``
    The Choi basket, with per-fixing "assets" whose forwards are
    ``x0 * qDF(t_i) / rDF(t_i)``, whose vols are
    ``blackVol(t_i, K_orig) * sqrt(t_i / t_last)``, and whose correlation is
    ``rho[i][j] = var[min(i, j)] / sqrt(var[i] var[j])`` — the Brownian
    covariance structure of a single asset sampled at ``m`` times.

Details a port gets wrong easily, all pinned by the probe:

* a fixing at ``t == 0`` is folded into the past: ``future_fixings`` drops by
  one, ``past_fixings`` rises by one and ``running_accumulator`` picks up
  ``x0``. The fixing dates are also **sorted** first, so an unsorted vector
  must price identically;
* the effective strike subtracts ``running_accumulator / (past + future)`` —
  divided by the **total** fixing count, not by ``past_fixings`` — and the
  basket weights are ``1 / (future + past)``, so a seasoned option is not
  simply a fresh option on the remaining fixings;
* the two strikes are used *differently*: the per-fixing variances that build
  the correlation matrix are read at the **effective** strike, while the
  per-fixing volatilities that build the basket processes are read at the
  **original** payoff strike;
* the basket legs use a **zero** rate curve for both the risk-free and the
  dividend leg (the discounting is applied once, at the end, from the option's
  own exercise date — which need not be the last fixing date).
"""

from __future__ import annotations

import math

import numpy as np

from pquantlib import qassert
from pquantlib.exercise import EuropeanExercise, Exercise
from pquantlib.instruments.asian_option import DiscreteAveragingAsianOptionArguments
from pquantlib.instruments.average_type import AverageType
from pquantlib.instruments.basket_option import AverageBasketPayoff, BasketOption
from pquantlib.instruments.one_asset_option import OneAssetOptionResults
from pquantlib.math.matrix import Matrix
from pquantlib.payoffs import PlainVanillaPayoff
from pquantlib.pricingengines.basket.choi_basket_engine import ChoiBasketEngine
from pquantlib.pricingengines.black_formula import black_formula
from pquantlib.pricingengines.generic_engine import GenericEngine
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.volatility.equity_fx.black_constant_vol import (
    BlackConstantVol,
)
from pquantlib.termstructures.yield_.flat_forward import FlatForward

# C++ default: 2 << 21.
_DEFAULT_MAX_NR_INTEGRATION_STEPS = 2 << 21


class ChoiAsianEngine(
    GenericEngine[DiscreteAveragingAsianOptionArguments, OneAssetOptionResults]
):
    """Arithmetic discrete-average Asian engine built on the Choi basket.

    # C++ parity: ``ChoiAsianEngine`` (choiasianengine.hpp:49-62,
    # choiasianengine.cpp:31-155).

    Args:
        process: the single underlying process.
        lambda_: forwarded to :class:`ChoiBasketEngine`. C++ default ``15``.
        max_nr_integration_steps: forwarded to :class:`ChoiBasketEngine`.
            C++ default ``2 << 21``.
    """

    def __init__(
        self,
        process: GeneralizedBlackScholesProcess,
        lambda_: float = 15.0,
        max_nr_integration_steps: int = _DEFAULT_MAX_NR_INTEGRATION_STEPS,
    ) -> None:
        # C++ parity: choiasianengine.cpp:31-40.
        super().__init__(
            DiscreteAveragingAsianOptionArguments(), OneAssetOptionResults()
        )
        self._process: GeneralizedBlackScholesProcess = process
        self._lambda: float = lambda_
        self._max_nr_integration_steps: int = max_nr_integration_steps
        process.register_with(self)

    def calculate(self) -> None:  # noqa: PLR0915  (one long C++ function)
        """Price the Asian option.

        # C++ parity: ``ChoiAsianEngine::calculate`` (choiasianengine.cpp:42-155).
        """
        args = self._arguments
        results = self._results
        results.reset()

        qassert.require(
            args.average_type == AverageType.Arithmetic,
            "must be Average::Type Arithmetic ",
        )
        exercise = args.exercise
        qassert.require(exercise is not None, "not a European Option")
        assert exercise is not None
        qassert.require(
            exercise.type() == Exercise.Type.European, "not a European Option"
        )

        payoff = args.payoff
        qassert.require(
            isinstance(payoff, PlainVanillaPayoff), "non plain vanilla payoff given"
        )
        assert isinstance(payoff, PlainVanillaPayoff)

        fixing_dates = sorted(args.fixing_dates)
        future_fixings = len(fixing_dates)
        assert args.past_fixings is not None
        assert args.running_accumulator is not None
        past_fixings = args.past_fixings
        running_accumulator = args.running_accumulator

        exercise_date = exercise.last_date()
        r_ts = self._process.risk_free_rate()

        # A fixing at t == 0 is folded into the past fixings.
        if future_fixings > 0 and self._process.time(fixing_dates[0]) == 0.0:
            fixing_dates = fixing_dates[1:]
            future_fixings -= 1
            past_fixings += 1
            running_accumulator += self._process.x0()

        if future_fixings == 0:
            qassert.require(past_fixings > 0, "no past fixings given")
            results.value = payoff(
                running_accumulator / past_fixings
            ) * r_ts.discount(exercise_date)
            return

        qassert.require(
            fixing_dates[-1] <= exercise_date,
            "last fixing date must be before exercise date",
        )
        qassert.require(
            self._process.time(fixing_dates[0]) >= 0.0,
            "first fixing date is in the past",
        )
        qassert.require(
            all(fixing_dates[i] != fixing_dates[i + 1] for i in range(future_fixings - 1)),
            "two fixing dates are the same",
        )

        accrued_average = (
            running_accumulator / (past_fixings + future_fixings)
            if past_fixings != 0
            else 0.0
        )

        strike = payoff.strike() - accrued_average
        qassert.require(strike >= 0.0, "effective strike should to be positive")

        q_ts = self._process.dividend_yield()
        vol_ts = self._process.black_volatility()
        vol_ref_date = vol_ts.reference_date()
        vol_dc = vol_ts.day_counter()

        if future_fixings > 1:
            fixing_times = [
                vol_dc.year_fraction(vol_ref_date, d) for d in fixing_dates
            ]
            # NB: variances at the EFFECTIVE strike ...
            variances = [
                self._process.black_volatility().black_variance(d, strike)
                for d in fixing_dates
            ]

            rho: Matrix = np.zeros((future_fixings, future_fixings), dtype=np.float64)
            for i in range(future_fixings):
                for j in range(i, future_fixings):
                    val = variances[min(i, j)] / math.sqrt(variances[i] * variances[j])
                    rho[i, j] = val
                    rho[j, i] = val

            zero_ts = FlatForward.from_rate(
                reference_date=r_ts.reference_date(),
                forward_rate=0.0,
                day_counter=r_ts.day_counter(),
            )

            processes: list[GeneralizedBlackScholesProcess] = []
            for i, fixing_date in enumerate(fixing_dates):
                # ... but volatilities at the ORIGINAL payoff strike.
                sig = vol_ts.black_vol(fixing_date, payoff.strike()) * math.sqrt(
                    fixing_times[i] / fixing_times[-1]
                )
                processes.append(
                    GeneralizedBlackScholesProcess(
                        x0=SimpleQuote(
                            self._process.x0()
                            * q_ts.discount(fixing_date)
                            / r_ts.discount(fixing_date)
                        ),
                        dividend_ts=zero_ts,
                        risk_free_ts=zero_ts,
                        black_vol_ts=BlackConstantVol(
                            reference_date=vol_ref_date,
                            calendar=vol_ts.calendar(),
                            day_counter=vol_dc,
                            volatility=SimpleQuote(sig),
                        ),
                    )
                )

            basket_option = BasketOption(
                AverageBasketPayoff(
                    PlainVanillaPayoff(payoff.option_type(), strike),
                    [1.0 / (future_fixings + past_fixings)] * future_fixings,
                ),
                EuropeanExercise(fixing_dates[-1]),
            )
            basket_option.set_pricing_engine(
                ChoiBasketEngine(
                    processes,
                    rho,
                    self._lambda,
                    self._max_nr_integration_steps,
                )
            )
            results.value = basket_option.npv() * r_ts.discount(exercise_date)
        else:
            last = fixing_dates[-1]
            results.value = black_formula(
                payoff.option_type(),
                strike,
                self._process.x0()
                / (past_fixings + future_fixings)
                * q_ts.discount(last)
                / r_ts.discount(last),
                math.sqrt(vol_ts.black_variance(last, strike)),
                r_ts.discount(exercise_date),
            )

    def update(self) -> None:
        self.notify_observers()


__all__ = ["ChoiAsianEngine"]
