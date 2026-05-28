"""Cliquet (Ratchet) option.

# C++ parity: ql/instruments/cliquetoption.{hpp,cpp} (v1.42.1).

A cliquet is a series of forward-starting (deferred-strike) options
where each period's strike resets to a fixed percentage of the spot
at the period start. The Python port mirrors the C++ ``arguments``
class with the same six configurable fields (accrued_coupon,
last_fixing, local_cap/floor, global_cap/floor) and ``reset_dates``.

Local/global caps + floors and the accrued coupon are typically
``None`` for a fresh ratchet; they're populated by the pricing
engine.
"""

from __future__ import annotations

from collections.abc import Sequence

from pquantlib import qassert
from pquantlib.exercise import EuropeanExercise, Exercise
from pquantlib.instruments.one_asset_option import OneAssetOption
from pquantlib.option import OptionArguments
from pquantlib.payoffs import PercentageStrikePayoff
from pquantlib.pricingengines.pricing_engine import PricingEngineArguments
from pquantlib.time.date import Date


class CliquetOptionArguments(OptionArguments):
    """Engine arguments for cliquet options.

    # C++ parity: ``CliquetOption::arguments``. The six float fields
    # default to ``None`` (= C++ ``Null<Real>()``) — engines that
    # support them write through ``setup_arguments``.
    """

    def __init__(self) -> None:
        super().__init__()
        self.accrued_coupon: float | None = None
        self.last_fixing: float | None = None
        self.local_cap: float | None = None
        self.local_floor: float | None = None
        self.global_cap: float | None = None
        self.global_floor: float | None = None
        self.reset_dates: list[Date] = []

    def validate(self) -> None:
        super().validate()
        qassert.require(
            isinstance(self.payoff, PercentageStrikePayoff),
            "wrong payoff type (expected PercentageStrikePayoff)",
        )
        assert isinstance(self.payoff, PercentageStrikePayoff)
        qassert.require(
            self.payoff.strike() > 0.0,
            f"negative or zero moneyness given ({self.payoff.strike()})",
        )
        qassert.require(
            self.accrued_coupon is None or self.accrued_coupon >= 0.0,
            "negative accrued coupon",
        )
        qassert.require(
            self.local_cap is None or self.local_cap >= 0.0, "negative local cap"
        )
        qassert.require(
            self.local_floor is None or self.local_floor >= 0.0, "negative local floor"
        )
        qassert.require(
            self.global_cap is None or self.global_cap >= 0.0, "negative global cap"
        )
        qassert.require(
            self.global_floor is None or self.global_floor >= 0.0,
            "negative global floor",
        )
        qassert.require(len(self.reset_dates) > 0, "no reset dates given")
        assert self.exercise is not None
        last_date = self.exercise.last_date()
        for i, d in enumerate(self.reset_dates):
            qassert.require(
                last_date > d, "reset date greater or equal to maturity"
            )
            if i > 0:
                qassert.require(
                    d > self.reset_dates[i - 1], "unsorted reset dates"
                )


class CliquetOption(OneAssetOption):
    """Cliquet (ratchet) option — series of forward-starting strikes.

    # C++ parity: ``CliquetOption(ext::shared_ptr<PercentageStrikePayoff>,
    # ext::shared_ptr<EuropeanExercise>, std::vector<Date> resetDates)``.
    """

    def __init__(
        self,
        payoff: PercentageStrikePayoff,
        maturity: EuropeanExercise,
        reset_dates: Sequence[Date],
    ) -> None:
        super().__init__(payoff, maturity)
        self._reset_dates: list[Date] = list(reset_dates)

    def reset_dates(self) -> list[Date]:
        return self._reset_dates

    def is_expired(self) -> bool:
        # See VanillaOption.is_expired — Settings.evaluation_date is a
        # Phase 1 carve-out, so engines always run.
        return False

    def setup_arguments(self, args: PricingEngineArguments) -> None:
        super().setup_arguments(args)
        qassert.require(
            isinstance(args, CliquetOptionArguments),
            "wrong argument type (expected CliquetOptionArguments)",
        )
        assert isinstance(args, CliquetOptionArguments)
        args.reset_dates = list(self._reset_dates)


# Re-export Exercise so callers that import from this module can use
# the European-exercise constructor.
_ = Exercise


__all__ = [
    "CliquetOption",
    "CliquetOptionArguments",
]
