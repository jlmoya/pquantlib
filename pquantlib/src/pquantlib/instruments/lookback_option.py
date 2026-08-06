"""Lookback options (continuous monitoring).

# C++ parity: ql/instruments/lookbackoption.{hpp,cpp} (v1.43).

Four concrete variants:

* ``ContinuousFloatingLookbackOption(minmax, FloatingTypePayoff,
  exercise)`` — payoff at exercise = ``S_T - min`` (Call) or
  ``max - S_T`` (Put). ``minmax`` is the realized extremum to date
  (= spot at issuance for an unseasoned option).
* ``ContinuousFixedLookbackOption(minmax, StrikedTypePayoff, exercise)``
  — payoff = ``max(max - K, 0)`` (Call on running max) or
  ``max(K - min, 0)`` (Put on running min).
* ``ContinuousPartialFloatingLookbackOption(minmax, lambda_,
  lookback_period_end, FloatingTypePayoff, exercise)`` — Heynen-Kat
  (1994): the lookback window starts at time zero and ENDS at
  ``lookback_period_end``, before expiry; the realized extremum over
  that window is scaled by ``lambda_`` when the strike is struck.
* ``ContinuousPartialFixedLookbackOption(lookback_period_start,
  StrikedTypePayoff, exercise)`` — Heynen-Kat (1994): the lookback
  window STARTS at ``lookback_period_start``, after inception, and runs
  to expiry. It takes no running extremum at all: the C++ constructor
  forwards a hard zero to the base class.

Both partial-time variants are strictly cheaper than the corresponding
full-window lookback, and both collapse onto it in the degenerate corner
(``lambda_ == 1`` with the window running to expiry, respectively a
window starting at inception).

Discrete-monitoring lookbacks remain deferred.
"""

from __future__ import annotations

from pquantlib import qassert
from pquantlib.exercise import Exercise
from pquantlib.instruments.one_asset_option import OneAssetOption
from pquantlib.option import OptionArguments
from pquantlib.payoffs import FloatingTypePayoff, OptionType, StrikedTypePayoff, TypePayoff
from pquantlib.pricingengines.pricing_engine import PricingEngineArguments
from pquantlib.time.date import Date


class ContinuousFloatingLookbackOptionArguments(OptionArguments):
    """Engine arguments for floating-strike continuous lookback.

    # C++ parity: ``ContinuousFloatingLookbackOption::arguments`` —
    # holds the running extremum (min for calls, max for puts).
    """

    def __init__(self) -> None:
        super().__init__()
        self.minmax: float | None = None

    def validate(self) -> None:
        super().validate()
        qassert.require(self.minmax is not None, "null prior extremum")
        assert self.minmax is not None
        qassert.require(
            self.minmax >= 0.0,
            f"nonnegative prior extremum required: {self.minmax} not allowed",
        )


class ContinuousFixedLookbackOptionArguments(OptionArguments):
    """Engine arguments for fixed-strike continuous lookback.

    # C++ parity: ``ContinuousFixedLookbackOption::arguments`` — same
    # ``minmax`` field as floating but a striked payoff.
    """

    def __init__(self) -> None:
        super().__init__()
        self.minmax: float | None = None

    def validate(self) -> None:
        super().validate()
        qassert.require(self.minmax is not None, "null prior extremum")
        assert self.minmax is not None
        qassert.require(
            self.minmax >= 0.0,
            f"nonnegative prior extremum required: {self.minmax} not allowed",
        )


class ContinuousFloatingLookbackOption(OneAssetOption):
    """Continuous-monitoring floating-strike lookback option.

    # C++ parity: ``ContinuousFloatingLookbackOption(minmax, payoff,
    # exercise)``. The payoff is a ``FloatingTypePayoff``; the strike is
    # fixed by the realized extremum at exercise.
    """

    def __init__(
        self,
        minmax: float,
        payoff: FloatingTypePayoff,
        exercise: Exercise,
    ) -> None:
        # C++ accepts ``TypePayoff``; floating-type is the only useful
        # subclass for a lookback. Constructor narrows to
        # ``FloatingTypePayoff`` for type safety.
        super().__init__(payoff, exercise)
        self._minmax: float = minmax

    def minmax(self) -> float:
        return self._minmax

    def is_expired(self) -> bool:
        # See VanillaOption.is_expired — Settings.evaluation_date is a
        # Phase 1 carve-out, so engines always run.
        return False

    def setup_arguments(self, args: PricingEngineArguments) -> None:
        super().setup_arguments(args)
        qassert.require(
            isinstance(args, ContinuousFloatingLookbackOptionArguments),
            "wrong argument type (expected ContinuousFloatingLookbackOptionArguments)",
        )
        assert isinstance(args, ContinuousFloatingLookbackOptionArguments)
        args.minmax = self._minmax


class ContinuousFixedLookbackOption(OneAssetOption):
    """Continuous-monitoring fixed-strike lookback option.

    # C++ parity: ``ContinuousFixedLookbackOption(minmax, payoff,
    # exercise)``. The payoff is a ``StrikedTypePayoff`` (typically
    # PlainVanilla); engine valuates against ``max`` (Call) or ``min``
    # (Put) over the lookback window.
    """

    def __init__(
        self,
        minmax: float,
        payoff: StrikedTypePayoff,
        exercise: Exercise,
    ) -> None:
        super().__init__(payoff, exercise)
        self._minmax: float = minmax

    def minmax(self) -> float:
        return self._minmax

    def is_expired(self) -> bool:
        # See VanillaOption.is_expired — Settings.evaluation_date is a
        # Phase 1 carve-out, so engines always run.
        return False

    def setup_arguments(self, args: PricingEngineArguments) -> None:
        super().setup_arguments(args)
        qassert.require(
            isinstance(args, ContinuousFixedLookbackOptionArguments),
            "wrong argument type (expected ContinuousFixedLookbackOptionArguments)",
        )
        assert isinstance(args, ContinuousFixedLookbackOptionArguments)
        args.minmax = self._minmax


class ContinuousPartialFloatingLookbackOptionArguments(ContinuousFloatingLookbackOptionArguments):
    """Engine arguments for the partial-time floating-strike lookback.

    # C++ parity: ``ContinuousPartialFloatingLookbackOption::arguments``
    # — adds ``lambda`` and ``lookbackPeriodEnd`` on top of the
    # ``minmax`` carried by the full-window arguments.
    """

    def __init__(self) -> None:
        super().__init__()
        # ``lambda`` is a Python keyword; the trailing underscore is the
        # repo-wide convention for that clash.
        self.lambda_: float | None = None
        self.lookback_period_end: Date = Date()

    def validate(self) -> None:
        """Validate the partial-time floating arguments.

        # C++ parity: ``ContinuousPartialFloatingLookbackOption::arguments::validate``
        # (lookbackoption.cpp:99-118). Note the asymmetric lambda
        # constraint: >= 1 for calls, <= 1 for puts, both non-strict, so
        # lambda == 1 is admissible for either type.
        #
        # C++ ``dynamic_pointer_cast``es the exercise to EuropeanExercise
        # and the payoff to FloatingTypePayoff and then dereferences both
        # WITHOUT a null check — a non-European exercise or a striked
        # payoff is undefined behaviour there. This port reads
        # ``last_date()`` off the generic Exercise (identical for a
        # European one) and raises on a non-floating payoff instead of
        # crashing.
        """
        super().validate()

        exercise = self.exercise
        assert exercise is not None
        qassert.require(
            self.lookback_period_end <= exercise.last_date(),
            "lookback start date must be earlier than exercise date",
        )

        payoff = self.payoff
        qassert.require(
            isinstance(payoff, FloatingTypePayoff),
            "wrong payoff type (expected FloatingTypePayoff)",
        )
        assert isinstance(payoff, FloatingTypePayoff)
        assert self.lambda_ is not None

        if payoff.option_type() == OptionType.Call:
            qassert.require(
                self.lambda_ >= 1.0,
                "lambda should be greater than or equal to 1 for calls",
            )
        if payoff.option_type() == OptionType.Put:
            qassert.require(
                self.lambda_ <= 1.0,
                "lambda should be smaller than or equal to 1 for puts",
            )


class ContinuousPartialFixedLookbackOptionArguments(ContinuousFixedLookbackOptionArguments):
    """Engine arguments for the partial-time fixed-strike lookback.

    # C++ parity: ``ContinuousPartialFixedLookbackOption::arguments`` —
    # adds ``lookbackPeriodStart`` on top of the (always zero) ``minmax``
    # carried by the full-window arguments.
    """

    def __init__(self) -> None:
        super().__init__()
        self.lookback_period_start: Date = Date()

    def validate(self) -> None:
        """Validate the partial-time fixed arguments.

        # C++ parity: ``ContinuousPartialFixedLookbackOption::arguments::validate``
        # (lookbackoption.cpp:132-140). As above, C++ dereferences an
        # unchecked ``dynamic_pointer_cast<EuropeanExercise>``; this port
        # reads ``last_date()`` off the generic Exercise.
        """
        super().validate()

        exercise = self.exercise
        assert exercise is not None
        qassert.require(
            self.lookback_period_start <= exercise.last_date(),
            "lookback start date must be earlier than exercise date",
        )


class ContinuousPartialFloatingLookbackOption(ContinuousFloatingLookbackOption):
    """Partial-time floating-strike lookback (Heynen-Kat 1994).

    # C++ parity: ``ContinuousPartialFloatingLookbackOption(minmax,
    # lambda, lookbackPeriodEnd, payoff, exercise)``.

    The lookback window runs from inception to ``lookback_period_end``,
    which must not be later than the exercise date; the strike struck at
    expiry is ``lambda_`` times the realized extremum over that window.
    ``lambda_`` must be >= 1 for calls and <= 1 for puts, so that the
    scaling always works against the holder and the option stays cheaper
    than the full-window one.
    """

    def __init__(
        self,
        minmax: float,
        lambda_: float,
        lookback_period_end: Date,
        payoff: FloatingTypePayoff,
        exercise: Exercise,
    ) -> None:
        super().__init__(minmax, payoff, exercise)
        self._lambda: float = lambda_
        self._lookback_period_end: Date = lookback_period_end

    def lambda_(self) -> float:
        return self._lambda

    def lookback_period_end(self) -> Date:
        return self._lookback_period_end

    def setup_arguments(self, args: PricingEngineArguments) -> None:
        super().setup_arguments(args)
        qassert.require(
            isinstance(args, ContinuousPartialFloatingLookbackOptionArguments),
            "wrong argument type (expected ContinuousPartialFloatingLookbackOptionArguments)",
        )
        assert isinstance(args, ContinuousPartialFloatingLookbackOptionArguments)
        args.lambda_ = self._lambda
        args.lookback_period_end = self._lookback_period_end


class ContinuousPartialFixedLookbackOption(ContinuousFixedLookbackOption):
    """Partial-time fixed-strike lookback (Heynen-Kat 1994).

    # C++ parity: ``ContinuousPartialFixedLookbackOption(lookbackPeriodStart,
    # payoff, exercise)``.

    The lookback window starts at ``lookback_period_start`` — which must
    not be later than the exercise date — and runs to expiry. There is no
    running-extremum argument: the C++ constructor forwards a hard ``0``
    to ``ContinuousFixedLookbackOption`` (lookbackoption.cpp:121), so
    ``minmax()`` is always ``0.0`` and only satisfies the base class's
    ``minmax >= 0`` check.
    """

    def __init__(
        self,
        lookback_period_start: Date,
        payoff: StrikedTypePayoff,
        exercise: Exercise,
    ) -> None:
        super().__init__(0.0, payoff, exercise)
        self._lookback_period_start: Date = lookback_period_start

    def lookback_period_start(self) -> Date:
        return self._lookback_period_start

    def setup_arguments(self, args: PricingEngineArguments) -> None:
        super().setup_arguments(args)
        qassert.require(
            isinstance(args, ContinuousPartialFixedLookbackOptionArguments),
            "wrong argument type (expected ContinuousPartialFixedLookbackOptionArguments)",
        )
        assert isinstance(args, ContinuousPartialFixedLookbackOptionArguments)
        args.lookback_period_start = self._lookback_period_start


# Re-export TypePayoff for callers that want to type the C++-style
# generic exercise + payoff.
_ = TypePayoff


__all__ = [
    "ContinuousFixedLookbackOption",
    "ContinuousFixedLookbackOptionArguments",
    "ContinuousFloatingLookbackOption",
    "ContinuousFloatingLookbackOptionArguments",
    "ContinuousPartialFixedLookbackOption",
    "ContinuousPartialFixedLookbackOptionArguments",
    "ContinuousPartialFloatingLookbackOption",
    "ContinuousPartialFloatingLookbackOptionArguments",
]
