"""ReplicationPathPricer — P&L of a discrete Black-Scholes hedging strategy.

Port of QuantLib's ``Examples/DiscreteHedging`` ``ReplicationPathPricer``
(the Java ``util/ReplicationPathPricer.java`` was a "Work in progress" stub
that threw; this follows the complete C++ original).

For a single simulated stock path, the pricer:

* sells one European option, banking its Black-Scholes premium;
* delta-hedges by buying ``delta`` shares, funded from the money account;
* re-hedges at each discrete step (the money account accrues at ``r``);
* at expiry, delivers the option payoff and unwinds the stock position.

The returned number is the final profit & loss of the strategy on that path.
"""

from __future__ import annotations

import math

from pquantlib.methods.montecarlo.path import Path
from pquantlib.methods.montecarlo.path_pricer import PathPricer
from pquantlib.payoffs import OptionType, PlainVanillaPayoff
from pquantlib.pricingengines.black_calculator import BlackCalculator


class ReplicationPathPricer(PathPricer[Path]):
    """Compute the discrete-hedging P&L for one stock path.

    # C++ parity: ``ReplicationPathPricer::operator()`` in DiscreteHedging.cpp.
    """

    def __init__(
        self,
        option_type: OptionType,
        strike: float,
        r: float,
        maturity: float,
        sigma: float,
    ) -> None:
        if strike <= 0.0:
            raise ValueError("strike must be positive")
        if r < 0.0:
            raise ValueError("risk free rate (r) must be positive or zero")
        if maturity <= 0.0:
            raise ValueError("maturity must be positive")
        if sigma < 0.0:
            raise ValueError("volatility (sigma) must be positive or zero")
        self._type = option_type
        self._strike = strike
        self._r = r
        self._maturity = maturity
        self._sigma = sigma

    def __call__(self, path: Path) -> float:
        n = path.length() - 1
        if n <= 0:
            raise ValueError("the path cannot be empty")

        dt = self._maturity / n
        stock_dividend_yield = 0.0
        t = 0.0
        stock = path.front()
        money_account = 0.0

        payoff = PlainVanillaPayoff(self._type, self._strike)

        # initial deal: sell the option, delta-hedge.
        r_discount = math.exp(-self._r * self._maturity)
        q_discount = math.exp(-stock_dividend_yield * self._maturity)
        forward = stock * q_discount / r_discount
        std_dev = math.sqrt(self._sigma * self._sigma * self._maturity)
        black = BlackCalculator(payoff, forward, std_dev, r_discount)
        money_account += black.value()
        delta = black.delta(stock)
        stock_amount = delta
        money_account -= stock_amount * stock

        # re-hedge through the option's life.
        for step in range(n - 1):
            t += dt
            money_account *= math.exp(self._r * dt)
            stock = path[step + 1]
            tau = self._maturity - t
            r_discount = math.exp(-self._r * tau)
            q_discount = math.exp(-stock_dividend_yield * tau)
            forward = stock * q_discount / r_discount
            std_dev = math.sqrt(self._sigma * self._sigma * tau)
            black = BlackCalculator(payoff, forward, std_dev, r_discount)
            delta = black.delta(stock)
            money_account -= (delta - stock_amount) * stock
            stock_amount = delta

        # expiration.
        money_account *= math.exp(self._r * dt)
        stock = path[n]
        option_payoff = PlainVanillaPayoff(self._type, self._strike)(stock)
        money_account -= option_payoff
        money_account += stock_amount * stock

        return money_account
