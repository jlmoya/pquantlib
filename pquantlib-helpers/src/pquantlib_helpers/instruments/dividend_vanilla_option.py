"""DividendVanillaOption — single-asset vanilla option with discrete dividends.

# Retired-API compat layer — NOT a port of C++ QuantLib v1.42.1.

This class was retired from C++ QuantLib (v1.42.1 prices discrete-dividend
options with a plain :class:`~pquantlib.instruments.vanilla_option.VanillaOption`
plus a ``DividendSchedule`` passed to ``AnalyticDividendEuropeanEngine`` /
``FdBlackScholesVanillaEngine``). JQuantLib still carries it because it forked
QuantLib ~2008. PQuantLib hosts the compat layer here, in pquantlib-helpers,
built on the v1.42.1 primitives that pquantlib core ships (period-specific
extensions belong in the helpers package — exactly as the Java repo puts
``BinomialDividendVanillaEngine`` / ``BlackScholesDividendLattice`` in
jquantlib-helpers, not core).

Java parity:
``org.jquantlib.instruments.DividendVanillaOption`` (jquantlib core, fork era).

A ``DividendVanillaOption`` is a
:class:`~pquantlib.instruments.vanilla_option.VanillaOption` that additionally
carries a schedule of discrete :class:`~pquantlib.cashflows.dividend.Dividend`
cash flows. The dividend schedule (built from parallel date/amount lists via
:func:`~pquantlib.cashflows.dividend.dividend_vector`) is copied into the
engine arguments by :meth:`setup_arguments`, where the
:class:`~pquantlib_helpers.pricingengines.vanilla.binomial_dividend_vanilla_engine.BinomialDividendVanillaEngine`
reads it.
"""

from __future__ import annotations

from pquantlib import qassert
from pquantlib.cashflows.dividend import Dividend, dividend_vector
from pquantlib.exercise import Exercise
from pquantlib.instruments.one_asset_option import OneAssetOptionResults
from pquantlib.instruments.vanilla_option import VanillaOption
from pquantlib.option import OptionArguments
from pquantlib.payoffs import StrikedTypePayoff
from pquantlib.pricingengines.pricing_engine import PricingEngineArguments
from pquantlib.time.date import Date


class DividendVanillaOptionArguments(OptionArguments):
    """Engine argument bundle for a dividend vanilla option.

    Java parity: ``DividendVanillaOption.ArgumentsImpl`` — extends the
    plain option arguments with the dividend ``cashFlow`` list.
    """

    def __init__(self) -> None:
        super().__init__()
        self.cash_flow: list[Dividend] = []

    def validate(self) -> None:
        """Validate base arguments + every dividend date precedes expiry.

        Java parity: ``DividendVanillaOption.ArgumentsImpl.validate`` —
        ``QL.require(d.le(exerciseDate), ...)`` for every dividend.
        """
        super().validate()
        assert self.exercise is not None
        exercise_date = self.exercise.last_date()
        for div in self.cash_flow:
            qassert.require(
                div.date() <= exercise_date,
                "dividend date later than the exercise date",
            )


class DividendVanillaOptionResults(OneAssetOptionResults):
    """Result carrier for a dividend vanilla option.

    Java parity: ``DividendVanillaOption.ResultsImpl`` (a marker subclass).
    Identical to the plain :class:`OneAssetOptionResults`; named separately
    only to mirror the Java type hierarchy.
    """


class DividendVanillaOption(VanillaOption):
    """Vanilla single-asset option carrying a discrete-dividend schedule.

    Java parity: ``org.jquantlib.instruments.DividendVanillaOption``.

    Construct from a payoff + exercise + parallel lists of dividend dates and
    cash amounts; the lists are turned into a sequence of
    :class:`~pquantlib.cashflows.dividend.FixedDividend` (Java
    ``Dividend.DividendVector``).
    """

    def __init__(
        self,
        payoff: StrikedTypePayoff,
        exercise: Exercise,
        dividend_dates: list[Date],
        dividends: list[float],
    ) -> None:
        super().__init__(payoff, exercise)
        # Java parity: ``cashFlow = Dividend.DividendVector(dates, dividends)``.
        self._cash_flow: list[Dividend] = dividend_vector(dividend_dates, dividends)

    def cash_flow(self) -> list[Dividend]:
        """The dividend schedule (list of :class:`Dividend`)."""
        return self._cash_flow

    def setup_arguments(self, args: PricingEngineArguments) -> None:
        """Copy payoff + exercise + dividend schedule into the engine args.

        Java parity: ``DividendVanillaOption.setupArguments`` —
        ``arguments.cashFlow = cashFlow`` on top of the base copy.
        """
        super().setup_arguments(args)
        qassert.require(
            isinstance(args, DividendVanillaOptionArguments),
            "wrong argument type (expected DividendVanillaOptionArguments)",
        )
        assert isinstance(args, DividendVanillaOptionArguments)
        args.cash_flow = self._cash_flow


__all__ = [
    "DividendVanillaOption",
    "DividendVanillaOptionArguments",
    "DividendVanillaOptionResults",
]
