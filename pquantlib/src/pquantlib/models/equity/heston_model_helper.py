"""HestonModelHelper — calibration helper for the Heston model.

# C++ parity: ql/models/equity/hestonmodelhelper.{hpp,cpp} (v1.42.1).

Concrete subclass of ``BlackCalibrationHelper`` that holds a single
European vanilla option (Call or Put depending on the at-or-OTM
selection rule) for cross-checking against a model-implied price.

The helper builds its option in ``_perform_calculations`` (lazy
LazyObject hook):

1. Compute the exercise date by advancing ``maturity_period`` on
   ``calendar`` starting from the risk-free curve's reference date.
2. Compute ``tau`` (year fraction) via the curve's day counter.
3. Pick option type: ``Call`` if strike * df_r >= spot * df_q else
   ``Put`` (i.e. the OTM choice — preferred for vol calibration).
4. Construct a ``VanillaOption`` with a ``PlainVanillaPayoff`` and
   ``EuropeanExercise``, then chain to the parent to populate the
   Black market value.

``black_price(vol)`` uses the standard Black-on-forward formula::

    black_formula(type, strike * df_r, spot * df_q, vol * sqrt(tau))

with ``df_r = risk_free.discount(tau)`` and ``df_q = dividend.discount(tau)``.

Divergences from C++:

* C++ has two ctors — one taking ``Real s0`` (wrapped in a fresh
  SimpleQuote) and one taking ``Handle<Quote> s0`` (registered as an
  observer). Python collapses to a single ctor that takes a ``Quote``;
  callers wrap a scalar via ``SimpleQuote(s0)`` if needed.
* The ``Handle<>`` indirection is removed throughout (pquantlib
  convention — observability comes from the Quote / YieldTermStructure
  base classes directly).
* ``addTimesTo`` is a no-op (matches the C++ override — Heston model
  doesn't need to know about any specific calibration times).
"""

from __future__ import annotations

import math
from typing import override

from pquantlib.exercise import EuropeanExercise
from pquantlib.instruments.vanilla_option import VanillaOption
from pquantlib.models.calibration_helper import (
    BlackCalibrationHelper,
    CalibrationErrorType,
)
from pquantlib.payoffs import OptionType, PlainVanillaPayoff
from pquantlib.pricingengines.black_formula import black_formula
from pquantlib.quotes.quote import Quote
from pquantlib.termstructures.volatility.volatility_type import VolatilityType
from pquantlib.termstructures.yield_term_structure import YieldTermStructure
from pquantlib.time.calendar import Calendar
from pquantlib.time.date import Date
from pquantlib.time.period import Period


class HestonModelHelper(BlackCalibrationHelper):
    """European-vanilla calibration helper for HestonModel.

    # C++ parity: ``class HestonModelHelper : public BlackCalibrationHelper``
    # in ql/models/equity/hestonmodelhelper.hpp:34-72 (v1.42.1).
    """

    __slots__ = (
        "_calendar",
        "_dividend_yield",
        "_exercise_date",
        "_maturity_period",
        "_option",
        "_risk_free_rate",
        "_s0",
        "_strike_price",
        "_tau",
        "_type",
    )

    def __init__(
        self,
        *,
        maturity: Period,
        calendar: Calendar,
        s0: Quote,
        strike_price: float,
        volatility: Quote,
        risk_free_rate: YieldTermStructure,
        dividend_yield: YieldTermStructure,
        calibration_error_type: CalibrationErrorType = CalibrationErrorType.RelativePriceError,
    ) -> None:
        super().__init__(
            volatility=volatility,
            calibration_error_type=calibration_error_type,
            volatility_type=VolatilityType.ShiftedLognormal,
            shift=0.0,
        )
        self._maturity_period: Period = maturity
        self._calendar: Calendar = calendar
        self._s0: Quote = s0
        self._strike_price: float = strike_price
        self._risk_free_rate: YieldTermStructure = risk_free_rate
        self._dividend_yield: YieldTermStructure = dividend_yield

        # Will be populated by ``_perform_calculations`` on first call.
        self._exercise_date: Date | None = None
        self._tau: float = float("nan")
        self._type: OptionType = OptionType.Call  # placeholder
        self._option: VanillaOption | None = None

        # C++ parity: hestonmodelhelper.cpp:59-61 — observer registration.
        s0.register_with(self)
        risk_free_rate.register_with(self)
        dividend_yield.register_with(self)

    # --- LazyObject hook -------------------------------------------------

    @override
    def _perform_calculations(self) -> None:
        """Build the VanillaOption + populate market value.

        # C++ parity: ``HestonModelHelper::performCalculations`` in
        # hestonmodelhelper.cpp:64-78.
        """
        rf = self._risk_free_rate
        div = self._dividend_yield
        self._exercise_date = self._calendar.advance_period(
            rf.reference_date(), self._maturity_period
        )
        # ``time_from_reference`` returns the year-fraction from rf's
        # reference_date to the exercise_date in rf's day counter.
        self._tau = rf.time_from_reference(self._exercise_date)

        # OTM-preferring type selection: Call if K*df_r >= S*df_q.
        if (
            self._strike_price * rf.discount(self._tau)
            >= self._s0.value() * div.discount(self._tau)
        ):
            self._type = OptionType.Call
        else:
            self._type = OptionType.Put

        payoff = PlainVanillaPayoff(self._type, self._strike_price)
        exercise = EuropeanExercise(self._exercise_date)
        self._option = VanillaOption(payoff, exercise)

        # Chain to parent — populates market value from black_price(vol).
        super()._perform_calculations()

    # --- inspectors ------------------------------------------------------

    def maturity(self) -> float:
        """Time to expiry in year-fraction units.

        # C++ parity: ``HestonModelHelper::maturity`` in
        # hestonmodelhelper.hpp:60.
        """
        self.calculate()
        return self._tau

    @property
    def strike_price(self) -> float:
        """The option strike price."""
        return self._strike_price

    @property
    def option_type(self) -> OptionType:
        """The OTM-selected option type (Call or Put).

        Requires ``calculate()`` to have run first (lazy initialization
        in ``_perform_calculations``).
        """
        self.calculate()
        return self._type

    # --- abstract overrides ---------------------------------------------

    @override
    def add_times_to(self, times: list[float]) -> None:
        """No-op — Heston doesn't need explicit calibration times.

        # C++ parity: hestonmodelhelper.hpp:56 — ``addTimesTo(std::list<Time>&) override {}``.
        """

    @override
    def model_value(self) -> float:
        """Model-implied option price.

        # C++ parity: ``HestonModelHelper::modelValue`` in
        # hestonmodelhelper.cpp:80-84.

        Requires :meth:`set_pricing_engine` to have attached an engine
        capable of pricing a European vanilla under Heston (e.g.
        ``AnalyticHestonEngine``).
        """
        self.calculate()
        assert self._option is not None, "perform_calculations did not build option"
        engine = self._engine
        if engine is None:
            raise RuntimeError("no pricing engine attached to HestonModelHelper")
        self._option.set_pricing_engine(engine)
        return self._option.npv()

    @override
    def black_price(self, volatility: float) -> float:
        """Black-on-forward price at the given vol.

        # C++ parity: ``HestonModelHelper::blackPrice`` in
        # hestonmodelhelper.cpp:86-92.
        """
        self.calculate()
        std_dev = volatility * math.sqrt(self._tau)
        rf = self._risk_free_rate
        div = self._dividend_yield
        return black_formula(
            option_type=self._type,
            strike=self._strike_price * rf.discount(self._tau),
            forward=self._s0.value() * div.discount(self._tau),
            std_dev=std_dev,
            discount=1.0,
            displacement=0.0,
        )

__all__ = ["HestonModelHelper"]
