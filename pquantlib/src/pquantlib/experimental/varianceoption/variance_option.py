"""Variance option (forward fair-variance derivative).

# C++ parity: ql/experimental/varianceoption/varianceoption.{hpp,cpp}
# (v1.42.1).

A variance option pays out as a function of the realised integrated
variance of the underlying over the period ``[startDate,
maturityDate]``. The payoff is a function of the realised variance,
scaled by a notional.

Caveat: the class does not currently handle seasoned variance options
(C++ ``warning: This class does not manage seasoned variance
options.``).
"""

from __future__ import annotations

from pquantlib import qassert
from pquantlib.instruments.instrument import Instrument, InstrumentResults
from pquantlib.payoffs import Payoff
from pquantlib.pricingengines.pricing_engine import (
    PricingEngineArguments,
)
from pquantlib.time.date import Date


class VarianceOptionArguments(PricingEngineArguments):
    """Arguments for forward fair-variance calculation.

    # C++ parity: ``VarianceOption::arguments``.
    """

    def __init__(self) -> None:
        super().__init__()
        self.payoff: Payoff | None = None
        self.notional: float | None = None
        self.start_date: Date | None = None
        self.maturity_date: Date | None = None

    def validate(self) -> None:
        qassert.require(self.payoff is not None, "no strike given")
        qassert.require(self.notional is not None, "no notional given")
        assert self.notional is not None
        qassert.require(
            self.notional > 0.0, "negative or null notional given"
        )
        qassert.require(self.start_date is not None, "null start date given")
        qassert.require(
            self.maturity_date is not None, "null maturity date given"
        )


class VarianceOptionResults(InstrumentResults):
    """Results carrier for variance options (matches C++).

    # C++ parity: ``VarianceOption::results`` is an empty subclass of
    # ``Instrument::results`` — kept distinct for engine dispatch.
    """


class VarianceOption(Instrument):
    """Variance option (one asset, realised-variance payoff).

    # C++ parity: ``VarianceOption(payoff, notional, startDate,
    # maturityDate)``.
    """

    def __init__(
        self,
        payoff: Payoff,
        notional: float,
        start_date: Date,
        maturity_date: Date,
    ) -> None:
        super().__init__()
        self._payoff: Payoff = payoff
        self._notional: float = notional
        self._start_date: Date = start_date
        self._maturity_date: Date = maturity_date

    def start_date(self) -> Date:
        return self._start_date

    def maturity_date(self) -> Date:
        return self._maturity_date

    def notional(self) -> float:
        return self._notional

    def payoff(self) -> Payoff:
        return self._payoff

    def is_expired(self) -> bool:
        # Phase 1 carve-out: Settings.evaluation_date isn't wired through
        # the instrument graph for is_expired checks; the engine always
        # runs. Matches existing pattern (VanillaOption.is_expired
        # also returns False).
        return False

    def setup_arguments(self, args: PricingEngineArguments) -> None:
        qassert.require(
            isinstance(args, VarianceOptionArguments),
            "wrong argument type (expected VarianceOptionArguments)",
        )
        assert isinstance(args, VarianceOptionArguments)
        args.payoff = self._payoff
        args.notional = self._notional
        args.start_date = self._start_date
        args.maturity_date = self._maturity_date


__all__ = [
    "VarianceOption",
    "VarianceOptionArguments",
    "VarianceOptionResults",
]
