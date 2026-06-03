"""DiscreteHedging sample — replication error of a discrete hedging strategy.

Port of QuantLib's ``Examples/DiscreteHedging/DiscreteHedging.cpp`` (the Java
``DiscreteHedging.java`` delegated to the "Work in progress" ReplicationError
stub; this follows the complete C++ original). Computes the profit & loss of a
discrete-interval Black-Scholes hedging strategy via Monte Carlo and compares
the realized P&L standard deviation with Derman & Kamal's closed-form
approximation, for two re-hedging frequencies (21 and 84 trades).

The number of scenarios is reduced from the C++ 50 000 to a smaller default so
the sample runs quickly under the smoke suite; pass a larger ``scenarios`` to
:func:`compute` for the full-resolution figures. The pseudo-random sequence is
seeded deterministically (a fixed nonzero seed), so the output is reproducible.
"""

from __future__ import annotations

from pquantlib.payoffs import OptionType
from pquantlib_samples.util.replication_error import HedgeStats, ReplicationError
from pquantlib_samples.util.stop_clock import StopClock


def compute(scenarios: int = 2000) -> tuple[float, tuple[HedgeStats, ...]]:
    """Return the option value and the per-frequency replication-error rows."""
    maturity = 1.0 / 12.0  # 1 month
    strike = 100.0
    underlying = 100.0
    volatility = 0.20  # 20%
    risk_free_rate = 0.05  # 5%

    rp = ReplicationError(OptionType.Call, maturity, strike, underlying, volatility, risk_free_rate)

    rows = (
        rp.compute(21, scenarios),
        rp.compute(84, scenarios),
    )
    return rp.option_value, rows


def run() -> None:
    print("::::: DiscreteHedging :::::")

    clock = StopClock()
    clock.start_clock()

    option_value, rows = compute()
    print(f"Option value: {option_value}")
    print()
    header = (
        f"{'samples':>8} | {'trades':>8} | {'P&L mean':>10} | {'P&L stddev':>11} | "
        f"{'D&K formula':>12} | {'skew':>8} | {'kurtosis':>8}"
    )
    print(header)
    print("-" * len(header))
    for s in rows:
        print(
            f"{s.samples:>8} | {s.trades:>8} | {s.pl_mean:>10.3f} | "
            f"{s.pl_std_dev:>11.2f} | {s.derman_kamal:>12.2f} | "
            f"{s.pl_skew:>8.2f} | {s.pl_kurt:>8.2f}"
        )

    clock.stop_clock()
    clock.log()


if __name__ == "__main__":
    run()
