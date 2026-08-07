"""Jamshidian decomposition swaption engine under a Gaussian1d model.

# C++ parity: ql/pricingengines/swaption/gaussian1djamshidianswaptionengine.{hpp,cpp}
# @ v1.43 (6b57206e0).

A European swaption on a fixed-vs-floating swap is a portfolio of
options on the individual fixed-leg zero bonds, provided the short rate
is a monotone function of the single state variable. The critical state
``y*`` is the one at which the remaining fixed flows (plus notional) are
worth exactly the notional as seen from the swap's value date; the
per-flow strikes follow from ``y*``.

C++ inverts the option type: a PAYER swaption becomes a PUT on the bond
portfolio and a RECEIVER a CALL.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pquantlib import qassert
from pquantlib.exercise import Exercise
from pquantlib.instruments.swap import SwapType
from pquantlib.instruments.swaption import (
    SettlementMethod,
    SwaptionArguments,
    SwaptionResults,
)
from pquantlib.math.solvers1d.brent import Brent
from pquantlib.payoffs import OptionType
from pquantlib.pricingengines.generic_engine import GenericEngine

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pquantlib.models.shortrate.gaussian1d_model import Gaussian1dModel
    from pquantlib.time.date import Date

_MIN_STRIKE: float = -8.0
_MAX_STRIKE: float = 8.0
_ACCURACY: float = 1e-8
_MAX_EVALUATIONS: int = 10000


class _RStarFinder:
    """Objective whose root is the critical state ``y*``.

    # C++ parity: ``Gaussian1dJamshidianSwaptionEngine::rStarFinder`` in
    # gaussian1djamshidianswaptionengine.cpp:27-58 (v1.43). Note the C++
    # member is called ``strike_`` but is initialised from the NOMINAL,
    # and ``times_`` holds DATES, not times — both names are historical.
    """

    def __init__(
        self,
        model: Gaussian1dModel,
        nominal: float,
        maturity_date: Date,
        value_date: Date,
        fixed_pay_dates: Sequence[Date],
        amounts: Sequence[float],
        start_index: int,
    ) -> None:
        self._model = model
        self._nominal = nominal
        self._maturity_date = maturity_date
        self._value_date = value_date
        self._dates = list(fixed_pay_dates)
        self._amounts = list(amounts)
        self._start_index = start_index

    def __call__(self, y: float) -> float:
        # # C++ parity: gaussian1djamshidianswaptionengine.cpp:40-49.
        value = self._nominal
        for i in range(self._start_index, len(self._dates)):
            db_value = self._model.zerobond_date(
                self._dates[i], self._maturity_date, y
            ) / self._model.zerobond_date(self._value_date, self._maturity_date, y)
            value -= self._amounts[i] * db_value
        return value


class Gaussian1dJamshidianSwaptionEngine(
    GenericEngine[SwaptionArguments, SwaptionResults]
):
    """Jamshidian swaption engine for a one-factor Gaussian model.

    # C++ parity: ``class Gaussian1dJamshidianSwaptionEngine`` in
    # gaussian1djamshidianswaptionengine.hpp:37-52 (v1.43).
    """

    def __init__(self, model: Gaussian1dModel) -> None:
        super().__init__(SwaptionArguments(), SwaptionResults())
        self._model: Gaussian1dModel = model

    def calculate(self) -> None:
        # # C++ parity: gaussian1djamshidianswaptionengine.cpp:60-125.
        args = self._arguments
        results = self._results
        results.reset()

        qassert.require(
            args.settlement_method != SettlementMethod.ParYieldCurve,
            "cash settled (ParYieldCurve) swaptions not priced with "
            "Gaussian1dJamshidianSwaptionEngine",
        )
        qassert.require(args.exercise is not None, "no exercise given")
        exercise = args.exercise
        assert isinstance(exercise, Exercise)
        qassert.require(
            exercise.type() == Exercise.Type.European,
            "cannot use the Jamshidian decomposition on exotic swaptions",
        )
        swap = args.swap
        assert swap is not None
        qassert.require(
            swap.spread() == 0.0,
            f"non zero spread ({swap.spread()}) not allowed",
        )
        qassert.require(
            args.nominal is not None, "non-constant nominals are not supported yet"
        )
        nominal = args.nominal
        assert nominal is not None

        amounts = list(args.fixed_coupons)
        amounts[-1] += nominal

        expiry = exercise.date(0)
        # Only consider coupons whose accrual starts on or after the
        # exercise date; the "- 1" makes the upper_bound inclusive.
        reset_dates: list[Date] = list(args.fixed_reset_dates)  # type: ignore[arg-type]
        cutoff = expiry - 1
        start_index = 0
        while start_index < len(reset_dates) and reset_dates[start_index] <= cutoff:
            start_index += 1

        finder = _RStarFinder(
            self._model,
            nominal,
            expiry,
            reset_dates[start_index],
            args.fixed_pay_dates,  # type: ignore[arg-type]
            amounts,
            start_index,
        )
        solver = Brent()
        solver.set_max_evaluations(_MAX_EVALUATIONS)
        solver.set_lower_bound(_MIN_STRIKE)
        solver.set_upper_bound(_MAX_STRIKE)
        # This is actually y*, not a rate; the C++ variable name is historical.
        r_star = solver.solve(finder, _ACCURACY, 0.00, _MIN_STRIKE, _MAX_STRIKE)

        # C++ inverts the option type: Payer -> Put on the bond.
        w = OptionType.Put if args.swap_type == SwapType.Payer else OptionType.Call

        value = 0.0
        value_date = reset_dates[start_index]
        for i in range(start_index, len(args.fixed_coupons)):
            pay_date: Date = args.fixed_pay_dates[i]  # type: ignore[assignment]
            strike = self._model.zerobond_date(
                pay_date, expiry, r_star
            ) / self._model.zerobond_date(value_date, expiry, r_star)
            dbo_value = self._model.zerobond_option(
                w, expiry, value_date, pay_date, strike
            )
            value += amounts[i] * dbo_value
        results.value = value


__all__ = ["Gaussian1dJamshidianSwaptionEngine"]
