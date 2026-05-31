"""Swaption / cap pseudo-root derivatives for market vegas.

# C++ parity: ql/models/marketmodels/pathwisegreeks/swaptionpseudojacobian.
# {hpp,cpp} (v1.42.1).

``SwaptionPseudoDerivative`` gives the derivative of a coterminal swaption's
implied vol (and variance) with respect to each pseudo-root element.
``CapPseudoDerivative`` does the same for a cap, where the cap implied vol has a
non-trivial (sum-of-caplets) relationship to the individual caplet vols and is
recovered by a Brent solve.

The C++ ``SwaptionPseudoJacobian`` named in the migration plan is this pair of
``*PseudoDerivative`` classes (the translation unit is
``swaptionpseudojacobian``).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from pquantlib import qassert
from pquantlib.math.solvers1d.brent import Brent
from pquantlib.models.marketmodels.curvestates.lmm_curve_state import LMMCurveState
from pquantlib.models.marketmodels.swap_forward_mappings import SwapForwardMappings
from pquantlib.payoffs import OptionType
from pquantlib.pricingengines.black_formula import (
    black_formula,
    black_formula_vol_derivative,
)

if TYPE_CHECKING:
    from pquantlib.math.matrix import Matrix
    from pquantlib.models.marketmodels.market_model import MarketModel


class SwaptionPseudoDerivative:
    """d(swaption implied vol / variance) / d(pseudo-root).

    # C++ parity: SwaptionPseudoDerivative.
    """

    def __init__(
        self,
        input_model: MarketModel,
        start_index: int,
        end_index: int,
    ) -> None:
        # C++ parity: SwaptionPseudoDerivative ctor.
        evolution = input_model.evolution()
        rate_times = evolution.rate_times()
        initial_rates = input_model.initial_rates()

        sub_rate_times = list(rate_times[start_index : end_index + 1])
        sub_forwards = list(initial_rates[start_index:end_index])

        cs = LMMCurveState(sub_rate_times)
        cs.set_on_forward_rates(sub_forwards)

        zed: Matrix = SwapForwardMappings.coterminal_swap_zed_matrix(
            cs, input_model.displacements()[0]
        )
        factors = input_model.number_of_factors()

        # first compute variance and implied vol
        variance = 0.0
        index = 0
        first_alive = evolution.first_alive_rate()
        while (
            index < evolution.number_of_steps()
            and first_alive[index] <= start_index
        ):
            this_pseudo = input_model.pseudo_root(index)
            this_variance = 0.0
            for j in range(start_index, end_index):
                for k in range(start_index, end_index):
                    for f in range(factors):
                        this_variance += (
                            zed[0][j - start_index]
                            * this_pseudo[j][f]
                            * this_pseudo[k][f]
                            * zed[0][k - start_index]
                        )
            variance += this_variance
            index += 1

        stop_index = index

        self._expiry = sub_rate_times[0]
        self._variance = variance
        self._implied_volatility = math.sqrt(variance / self._expiry)

        scale = 0.5 * (1.0 / self._expiry) / self._implied_volatility

        number_rates = evolution.number_of_rates()
        null_derivative = np.zeros((number_rates, factors), dtype=np.float64)

        self._variance_derivatives: list[Matrix] = []
        self._volatility_derivatives: list[Matrix] = []

        index = 0
        while index < stop_index:
            this_pseudo = input_model.pseudo_root(index)
            this_derivative = np.zeros((number_rates, factors), dtype=np.float64)
            for rate in range(start_index, end_index):
                z_index = rate - start_index
                for f in range(factors):
                    total = 0.0
                    for rate2 in range(start_index, end_index):
                        z_index2 = rate2 - start_index
                        total += zed[0][z_index2] * this_pseudo[rate2][f]
                    total *= 2.0 * zed[0][z_index]
                    this_derivative[rate][f] = total

            # variance derivative is this_derivative; volatility derivative is
            # the same scaled by `scale`. Copy before scaling (C++ pushes the
            # variance Matrix, then scales the same buffer in place).
            self._variance_derivatives.append(this_derivative.copy())
            for rate in range(start_index, end_index):
                for f in range(factors):
                    this_derivative[rate][f] *= scale
            self._volatility_derivatives.append(this_derivative)
            index += 1

        while index < evolution.number_of_steps():
            self._variance_derivatives.append(null_derivative.copy())
            self._volatility_derivatives.append(null_derivative.copy())
            index += 1

    def variance_derivative(self, i: int) -> Matrix:
        # C++ parity: SwaptionPseudoDerivative::varianceDerivative.
        return self._variance_derivatives[i]

    def volatility_derivative(self, i: int) -> Matrix:
        # C++ parity: SwaptionPseudoDerivative::volatilityDerivative.
        return self._volatility_derivatives[i]

    def implied_volatility(self) -> float:
        # C++ parity: SwaptionPseudoDerivative::impliedVolatility.
        return self._implied_volatility

    def variance(self) -> float:
        # C++ parity: SwaptionPseudoDerivative::variance.
        return self._variance

    def expiry(self) -> float:
        # C++ parity: SwaptionPseudoDerivative::expiry.
        return self._expiry


class _QuickCap:
    """Sum-of-caplets cap pricer used by the cap implied-vol root find.

    # C++ parity: anonymous-namespace ``QuickCap`` in swaptionpseudojacobian.cpp.
    """

    def __init__(
        self,
        strike: float,
        annuities: list[float],
        current_rates: list[float],
        expiries: list[float],
        price: float,
    ) -> None:
        self._strike = strike
        self._annuities = annuities
        self._current_rates = current_rates
        self._expiries = expiries
        self._price = price

    def __call__(self, volatility: float) -> float:
        # returns difference from input price
        price = 0.0
        for i in range(len(self._annuities)):
            price += black_formula(
                OptionType.Call,
                self._strike,
                self._current_rates[i],
                volatility * math.sqrt(self._expiries[i]),
                self._annuities[i],
            )
        return price - self._price

    def vega(self, volatility: float) -> float:
        # returns vol derivative
        vega = 0.0
        for i in range(len(self._annuities)):
            vega += black_formula_vol_derivative(
                self._strike,
                self._current_rates[i],
                volatility * math.sqrt(self._expiries[i]),
                self._expiries[i],
                self._annuities[i],
                0.0,
            )
        return vega


class CapPseudoDerivative:
    """d(cap implied vol / price) / d(pseudo-root).

    # C++ parity: CapPseudoDerivative.
    """

    def __init__(  # noqa: PLR0915  (faithful C++ ctor — one flat numeric kernel)
        self,
        input_model: MarketModel,
        strike: float,
        start_index: int,
        end_index: int,
        first_df: float,
    ) -> None:
        # C++ parity: CapPseudoDerivative ctor.
        self._first_df = first_df
        qassert.require(
            start_index < end_index,
            "for a cap pseudo derivative the start of the cap must be before "
            "the end",
        )
        qassert.require(
            end_index <= input_model.number_of_rates(),
            "for a cap pseudo derivative the end of the cap must before the end "
            "of the rates",
        )

        number_caplets = end_index - start_index
        number_rates = input_model.number_of_rates()
        factors = input_model.number_of_factors()
        evolution = input_model.evolution()
        curve = LMMCurveState(evolution.rate_times())
        curve.set_on_forward_rates(input_model.initial_rates())

        total_covariance = input_model.total_covariance(
            input_model.number_of_steps() - 1
        )

        displaced_implied_vols = [0.0] * number_caplets
        annuities = [0.0] * number_caplets
        initial_rates = [0.0] * number_caplets
        expiries = [0.0] * number_caplets

        cap_price = 0.0
        guess = 0.0
        min_vol = 1e10
        max_vol = 0.0

        rate_times = evolution.rate_times()
        rate_taus = evolution.rate_taus()
        displacements = input_model.displacements()
        model_initial = input_model.initial_rates()

        for j in range(start_index, end_index):
            caplet_index = j - start_index
            reset_time = rate_times[j]
            expiries[caplet_index] = reset_time

            sd = math.sqrt(total_covariance[j][j])
            displaced_implied_vols[caplet_index] = math.sqrt(
                total_covariance[j][j] / reset_time
            )

            forward = model_initial[j]
            initial_rates[caplet_index] = forward

            annuity = curve.discount_ratio(j + 1, 0) * rate_taus[j] * first_df
            annuities[caplet_index] = annuity

            displacement = displacements[j]

            guess += (
                displaced_implied_vols[caplet_index]
                * (forward + displacement)
                / forward
            )
            min_vol = min(min_vol, displaced_implied_vols[caplet_index])
            max_vol = max(
                max_vol,
                displaced_implied_vols[caplet_index]
                * (forward + displacement)
                / forward,
            )

            caplet_price = black_formula(
                OptionType.Call, strike, forward, sd, annuity, displacement
            )
            cap_price += caplet_price

        guess /= number_caplets

        first_alive = evolution.first_alive_rate()

        self._price_derivatives: list[Matrix] = []
        for step in range(evolution.number_of_steps()):
            this_derivative = np.zeros((number_rates, factors), dtype=np.float64)
            for rate in range(max(first_alive[step], start_index), end_index):
                for f in range(factors):
                    expiry = rate_times[rate]
                    vol_derivative = input_model.pseudo_root(step)[rate][f] / (
                        displaced_implied_vols[rate - start_index] * expiry
                    )
                    caplet_vega = black_formula_vol_derivative(
                        strike,
                        model_initial[rate],
                        displaced_implied_vols[rate - start_index]
                        * math.sqrt(expiry),
                        expiry,
                        annuities[rate - start_index],
                        displacements[rate],
                    )
                    # the cap derivative is one of the caplet ones (lose a loop)
                    this_derivative[rate][f] = vol_derivative * caplet_vega
            self._price_derivatives.append(this_derivative)

        cap_pricer = _QuickCap(strike, annuities, initial_rates, expiries, cap_price)

        max_evaluations = 1000
        accuracy = 1e-6
        solver = Brent()
        solver.set_max_evaluations(max_evaluations)
        self._implied_volatility = solver.solve(
            cap_pricer, accuracy, guess, min_vol * 0.99, max_vol * 1.01
        )

        self._vega = cap_pricer.vega(self._implied_volatility)

        self._volatility_derivatives: list[Matrix] = []
        for step in range(evolution.number_of_steps()):
            this_derivative = np.zeros((number_rates, factors), dtype=np.float64)
            for rate in range(max(first_alive[step], start_index), end_index):
                for f in range(factors):
                    this_derivative[rate][f] = (
                        self._price_derivatives[step][rate][f] / self._vega
                    )
            self._volatility_derivatives.append(this_derivative)

    def price_derivative(self, i: int) -> Matrix:
        # C++ parity: CapPseudoDerivative::priceDerivative.
        return self._price_derivatives[i]

    def volatility_derivative(self, i: int) -> Matrix:
        # C++ parity: CapPseudoDerivative::volatilityDerivative.
        return self._volatility_derivatives[i]

    def implied_volatility(self) -> float:
        # C++ parity: CapPseudoDerivative::impliedVolatility.
        return self._implied_volatility
