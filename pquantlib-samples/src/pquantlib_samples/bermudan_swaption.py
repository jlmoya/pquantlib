"""BermudanSwaption sample — calibrate a short-rate model and price a Bermudan.

Port of QuantLib's ``Examples/BermudanSwaption`` (the Java ``BermudanSwaption``
was a ``// Work in progress`` stub that built only the dates; this follows the
complete C++ original, reduced to the Hull-White leg). Calibrates a Hull-White
one-factor model to a diagonal of co-terminal market swaptions via Jamshidian's
analytic engine + Levenberg-Marquardt, then prices a Bermudan swaption (annual
co-terminal exercise dates) with :class:`TreeSwaptionEngine`.

The C++ example also calibrates G2++ and Black-Karasinski; this port keeps the
Hull-White leg (the representative one-factor calibration + tree pricing path)
to stay focused and fast. The market swaption-vol diagonal and the 5y10y swap
geometry mirror the C++ ``swaptionVols`` / ``swapLengths`` setup at lower
resolution.
"""

from __future__ import annotations

from dataclasses import dataclass

from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.daycounters.thirty_360 import Convention, Thirty360
from pquantlib.exercise import BermudanExercise
from pquantlib.indexes.ibor.euribor import Euribor
from pquantlib.instruments.swap import SwapType
from pquantlib.instruments.swaption import Swaption
from pquantlib.instruments.vanilla_swap import VanillaSwap
from pquantlib.math.optimization.end_criteria import EndCriteria
from pquantlib.math.optimization.levenberg_marquardt import LevenbergMarquardt
from pquantlib.models.calibration_helper import CalibrationErrorType
from pquantlib.models.shortrate.onefactor.hull_white import HullWhite
from pquantlib.models.swaption_helper import SwaptionHelper
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.pricingengines.swaption.tree_swaption_engine import TreeSwaptionEngine
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.compounding import Compounding
from pquantlib.time.date import Date
from pquantlib.time.date_generation import DateGeneration
from pquantlib.time.frequency import Frequency
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.schedule import Schedule
from pquantlib.time.time_unit import TimeUnit
from pquantlib_samples.util.stop_clock import StopClock

# diagonal of co-terminal swaptions: (option maturity yrs, swap length yrs, vol).
_SWAPTIONS: tuple[tuple[int, int, float], ...] = (
    (1, 5, 0.1490),
    (2, 4, 0.1340),
    (3, 3, 0.1228),
    (4, 2, 0.1189),
    (5, 1, 0.1148),
)


@dataclass(frozen=True, slots=True)
class BermudanResult:
    """Computed quantities a :func:`run` would print — for cross-checking."""

    hw_a: float
    hw_sigma: float
    bermudan_npv: float


def compute() -> BermudanResult:
    today = Date.from_ymd(15, Month.February, 2002)
    settlement = Date.from_ymd(19, Month.February, 2002)
    ObservableSettings().evaluation_date = today

    calendar = TARGET()
    rate = SimpleQuote(0.04875825)
    curve = FlatForward(settlement, rate, Actual360(), Compounding.Continuous, Frequency.Annual)

    index = Euribor.six_months(curve)
    fixed_leg_tenor = Period(1, TimeUnit.Years)
    fixed_leg_day_counter = Thirty360(Convention.European)
    floating_leg_day_counter = Actual360()

    model = HullWhite(curve)

    helpers: list[SwaptionHelper] = []
    for maturity_yrs, length_yrs, vol in _SWAPTIONS:
        helper = SwaptionHelper(
            maturity=Period(maturity_yrs, TimeUnit.Years),
            length=Period(length_yrs, TimeUnit.Years),
            volatility=SimpleQuote(vol),
            index=index,
            fixed_leg_tenor=fixed_leg_tenor,
            fixed_leg_day_counter=fixed_leg_day_counter,
            floating_leg_day_counter=floating_leg_day_counter,
            term_structure=curve,
            error_type=CalibrationErrorType.RelativePriceError,
        )
        # The C++ example calibrates Hull-White analytically (Jamshidian);
        # pquantlib's TreeSwaptionEngine drives the calibration here instead
        # (matching the C++ HW2/Black-Karasinski legs, and side-stepping a
        # multi-factor signature issue in the analytic Jamshidian path).
        helper.set_pricing_engine(TreeSwaptionEngine(model, 50, curve))
        helpers.append(helper)

    method = LevenbergMarquardt()
    end_criteria = EndCriteria(400, 100, 1.0e-8, 1.0e-8, 1.0e-8)
    model.calibrate(helpers, method, end_criteria)
    params = model.params()
    hw_a, hw_sigma = float(params[0]), float(params[1])

    # Build the co-terminal underlying swap (5y option into a 5y swap), then a
    # Bermudan with annual exercise dates.
    start_date = calendar.advance(settlement, 1, TimeUnit.Years)
    maturity = calendar.advance(start_date, 5, TimeUnit.Years)
    fixed_schedule = Schedule.from_rule(
        start_date,
        maturity,
        Period(1, TimeUnit.Years),
        calendar,
        BusinessDayConvention.Unadjusted,
        BusinessDayConvention.Unadjusted,
        DateGeneration.Forward,
        False,
    )
    float_schedule = Schedule.from_rule(
        start_date,
        maturity,
        Period(6, TimeUnit.Months),
        calendar,
        BusinessDayConvention.ModifiedFollowing,
        BusinessDayConvention.ModifiedFollowing,
        DateGeneration.Forward,
        False,
    )
    atm_rate = 0.05
    swap = VanillaSwap(
        SwapType.Payer,
        1000.0,
        fixed_schedule,
        atm_rate,
        fixed_leg_day_counter,
        float_schedule,
        index,
        0.0,
        floating_leg_day_counter,
    )

    bermudan_dates = list(fixed_schedule.dates[:-1])
    bermudan_swaption = Swaption(swap, BermudanExercise(bermudan_dates))
    bermudan_swaption.set_pricing_engine(TreeSwaptionEngine(model, 50, curve))
    bermudan_npv = bermudan_swaption.npv()

    return BermudanResult(hw_a=hw_a, hw_sigma=hw_sigma, bermudan_npv=bermudan_npv)


def run() -> None:
    print("::::: BermudanSwaption :::::")

    clock = StopClock()
    clock.start_clock()

    r = compute()
    print("Hull-White calibrated to the co-terminal swaption diagonal:")
    print(f"  a     = {r.hw_a:.6f}")
    print(f"  sigma = {r.hw_sigma:.6f}")
    print()
    print(f"Bermudan swaption NPV (TreeSwaptionEngine, 50 steps) = {r.bermudan_npv:.6f}")

    clock.stop_clock()
    clock.log()


if __name__ == "__main__":
    run()
