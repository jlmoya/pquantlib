"""DiscreteAveragingAsianOption — discrete-average Asian option.

# C++ parity: ql/instruments/asianoption.{hpp,cpp} (v1.42.1).

A discrete-average Asian option pays
``max(opt * (A_n - K), 0)`` where ``A_n`` is the arithmetic (or
geometric) mean of the underlying asset at ``n`` fixing dates.

The instrument supports two construction modes:

* ``DiscreteAveragingAsianOption(average_type, running_accumulator,
  past_fixings, fixing_dates, payoff, exercise)`` — the traditional
  interface used by C++ test suite: caller provides a
  ``running_accumulator`` (running sum for arithmetic, running
  product for geometric) and the number of past fixings already
  observed.  If ``past_fixings == 0`` the accumulator is forced to
  the algebraic identity (1 for geometric, 0 for arithmetic), even
  if a non-default value was passed.

* ``DiscreteAveragingAsianOption.with_all_past_fixings(...)`` —
  caller provides *all* fixing dates plus a complete vector of
  past observations.  The instrument splits past vs future at
  ``Settings.evaluationDate``.  The Python port DOES NOT implement
  this branch because pquantlib has no global ``Settings`` /
  ``evaluation_date`` yet (deferred per L1 carve-out).  Use the
  primary constructor instead.

The Python port keeps the same engine-argument bundle
(``DiscreteAveragingAsianOptionArguments``) so MC engines can pull
``average_type`` / ``running_accumulator`` / ``past_fixings`` /
``fixing_dates`` per the same pattern as C++ ``arguments_``.
"""

from __future__ import annotations

from pquantlib import qassert
from pquantlib.exercise import Exercise
from pquantlib.instruments.average_type import AverageType
from pquantlib.instruments.one_asset_option import OneAssetOption
from pquantlib.option import OptionArguments
from pquantlib.payoffs import StrikedTypePayoff
from pquantlib.pricingengines.pricing_engine import PricingEngineArguments
from pquantlib.time.date import Date


class DiscreteAveragingAsianOptionArguments(OptionArguments):
    """Engine argument bundle for discrete-average Asian options.

    # C++ parity: ``DiscreteAveragingAsianOption::arguments``
    # (asianoption.hpp:115-126).
    """

    def __init__(self) -> None:
        super().__init__()
        # average_type defaults to None (C++ uses -1 as sentinel).
        self.average_type: AverageType | None = None
        self.running_accumulator: float | None = None
        self.past_fixings: int | None = None
        self.fixing_dates: list[Date] = []

    def validate(self) -> None:
        # OneAssetOption::arguments::validate -> OptionArguments::validate
        super().validate()
        qassert.require(self.average_type is not None, "unspecified average type")
        qassert.require(self.past_fixings is not None, "null past-fixing number")
        qassert.require(self.running_accumulator is not None, "null running product")
        assert self.running_accumulator is not None
        if self.average_type == AverageType.Arithmetic:
            qassert.require(
                self.running_accumulator >= 0.0,
                f"non negative running sum required: {self.running_accumulator} not allowed",
            )
        elif self.average_type == AverageType.Geometric:
            qassert.require(
                self.running_accumulator > 0.0,
                f"positive running product required: {self.running_accumulator} not allowed",
            )
        else:
            qassert.fail("invalid average type")


class DiscreteAveragingAsianOption(OneAssetOption):
    """Discrete-average Asian option on a single underlying asset.

    # C++ parity: ``DiscreteAveragingAsianOption`` (asianoption.{hpp,cpp}).
    """

    def __init__(
        self,
        average_type: AverageType,
        running_accumulator: float,
        past_fixings: int,
        fixing_dates: list[Date],
        payoff: StrikedTypePayoff,
        exercise: Exercise,
    ) -> None:
        super().__init__(payoff, exercise)
        self._average_type: AverageType = average_type
        self._past_fixings: int = past_fixings
        # Per C++ ctor: sort the fixing-dates list.
        self._fixing_dates: list[Date] = sorted(fixing_dates)
        # Per C++ ctor: if past_fixings == 0 force the algebraic identity
        # even if caller passed a non-default accumulator.
        if past_fixings == 0:
            if average_type == AverageType.Geometric:
                self._running_accumulator: float = 1.0
            elif average_type == AverageType.Arithmetic:
                self._running_accumulator = 0.0
            else:
                qassert.fail(
                    "Unrecognised average type, must be AverageType.Arithmetic "
                    "or AverageType.Geometric"
                )
        else:
            self._running_accumulator = running_accumulator

    # --- inspectors ------------------------------------------------------

    def average_type(self) -> AverageType:
        return self._average_type

    def running_accumulator(self) -> float:
        return self._running_accumulator

    def past_fixings(self) -> int:
        return self._past_fixings

    def fixing_dates(self) -> list[Date]:
        return list(self._fixing_dates)

    def is_expired(self) -> bool:
        """Return ``False`` (Settings.evaluationDate not wired yet).

        # C++ parity: ``Instrument::isExpired`` defers to
        # ``Settings::evaluationDate``. Until ``Settings`` is ported,
        # we return ``False`` so the engine always runs — consistent
        # with the ``VanillaOption`` carve-out in this codebase.
        """
        return False

    # --- argument setup --------------------------------------------------

    def setup_arguments(self, args: PricingEngineArguments) -> None:
        """Populate the engine arguments bundle with our extra fields.

        # C++ parity: ``DiscreteAveragingAsianOption::setupArguments``
        # (asianoption.cpp:65-116). The Python port omits the
        # ``allPastFixingsProvided_`` split-on-evaluation-date branch
        # (Settings.evaluationDate is not yet wired).
        """
        super().setup_arguments(args)
        qassert.require(
            isinstance(args, DiscreteAveragingAsianOptionArguments),
            "wrong argument type",
        )
        assert isinstance(args, DiscreteAveragingAsianOptionArguments)
        args.average_type = self._average_type
        args.running_accumulator = self._running_accumulator
        args.past_fixings = self._past_fixings
        args.fixing_dates = list(self._fixing_dates)


__all__ = [
    "DiscreteAveragingAsianOption",
    "DiscreteAveragingAsianOptionArguments",
]
