"""Calibration helper abstract base classes.

# C++ parity: ql/models/calibrationhelper.{hpp,cpp} (v1.42.1).

C++ defines two layers:

- ``CalibrationHelper`` — abstract ``calibrationError()``. Used by
  ``CalibratedModel::calibrate``.
- ``BlackCalibrationHelper`` — concrete-able subclass that adds
  Black76 market instrument semantics: holds a vol Quote, a
  ``CalibrationErrorType`` (PriceError / RelativePriceError /
  ImpliedVolError), a ``VolatilityType`` (ShiftedLognormal / Normal),
  and a shift. Concrete subclasses (SwaptionHelper / CapHelper /
  HestonModelHelper) land in L4-C/E.

The pquantlib port mirrors this hierarchy verbatim. Both classes
remain abstract — concrete fields (``modelValue``, ``blackPrice``,
``addTimesTo``) are pure-virtual in C++ and ``@abstractmethod`` here.

Divergences from C++:

- ``Handle<Quote>`` collapses to a direct ``Quote`` reference (the
  pquantlib convention — Quote is itself Observable, so the
  intermediate Handle is unnecessary).
- ``setPricingEngine`` becomes a plain setter ``set_pricing_engine``
  for the optional engine. Concrete subclasses use it during
  ``modelValue``.
- ``addTimesTo`` takes the times list by mutable reference in C++;
  in Python it's a plain ``list[float]`` (append in-place).
- ``BlackCalibrationHelper`` inherits ``LazyObject`` AND
  ``CalibrationHelper`` in C++. Python's MRO handles this fine; the
  ``LazyObject`` mixin provides ``calculate()`` / ``update()`` and
  the subclass overrides ``_perform_calculations`` to populate
  ``marketValue_`` from ``blackPrice(volatility.value())``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import IntEnum
from typing import TYPE_CHECKING

from pquantlib.math.solvers1d.brent import Brent
from pquantlib.patterns.lazy_object import LazyObject
from pquantlib.termstructures.volatility.volatility_type import VolatilityType

if TYPE_CHECKING:
    from pquantlib.pricingengines.pricing_engine import PricingEngine
    from pquantlib.quotes.quote import Quote


class CalibrationHelper(ABC):
    """Abstract base for calibration helpers.

    # C++ parity: ``class CalibrationHelper`` in
    # ql/models/calibrationhelper.hpp:40-45 (v1.42.1).

    The only obligation is ``calibration_error()``, which the model's
    calibration cost function evaluates per iteration.
    """

    @abstractmethod
    def calibration_error(self) -> float:
        """Return the error resulting from the model valuation."""
        ...


class CalibrationErrorType(IntEnum):
    """How to measure the calibration error.

    # C++ parity: ``BlackCalibrationHelper::CalibrationErrorType`` in
    # ql/models/calibrationhelper.hpp:50-52 (v1.42.1).

    - ``RelativePriceError`` (default): ``|market - model| / market``
    - ``PriceError``: ``market - model``
    - ``ImpliedVolError``: ``impliedVol(model_price) - market_vol``
    """

    RelativePriceError = 0
    PriceError = 1
    ImpliedVolError = 2


class BlackCalibrationHelper(LazyObject, CalibrationHelper):
    """Liquid Black76 market instrument used during calibration.

    # C++ parity: ``class BlackCalibrationHelper`` in
    # ql/models/calibrationhelper.hpp:48-106 (v1.42.1).

    Concrete subclasses (Swaption / Cap / Heston model helpers) land
    in L4-C/E. They implement ``model_value`` (price under the model),
    ``black_price(vol)`` (Black-or-Bachelier price for a given vol),
    and ``add_times_to(times)`` (which calibration times the model
    needs to evaluate at).
    """

    __slots__ = (
        "_calibration_error_type",
        "_engine",
        "_market_value",
        "_shift",
        "_volatility",
        "_volatility_type",
    )

    def __init__(
        self,
        volatility: Quote,
        calibration_error_type: CalibrationErrorType = CalibrationErrorType.RelativePriceError,
        volatility_type: VolatilityType = VolatilityType.ShiftedLognormal,
        shift: float = 0.0,
    ) -> None:
        super().__init__()
        self._volatility: Quote = volatility
        self._calibration_error_type: CalibrationErrorType = calibration_error_type
        self._volatility_type: VolatilityType = volatility_type
        self._shift: float = shift
        # C++ parity: cap.hpp:97 — ``mutable Real marketValue_``;
        # populated by ``_perform_calculations``. NaN sentinel until
        # ``calculate()`` runs.
        self._market_value: float = float("nan")
        self._engine: PricingEngine | None = None
        # C++ parity: calibrationhelper.hpp:60 — ``registerWith(volatility_)``.
        self._volatility.register_with(self)

    # --- LazyObject hook ----------------------------------------------

    def _perform_calculations(self) -> None:
        # C++ parity: calibrationhelper.hpp:62-64 — populates marketValue_
        # from the current vol quote.
        self._market_value = self.black_price(self._volatility.value())

    # --- inspectors ---------------------------------------------------

    @property
    def volatility(self) -> Quote:
        """The volatility quote driving the market value.

        # C++ parity: calibrationhelper.hpp:67 — ``volatility()``.
        """
        return self._volatility

    @property
    def volatility_type(self) -> VolatilityType:
        """ShiftedLognormal or Normal vol convention.

        # C++ parity: calibrationhelper.hpp:70 — ``volatilityType()``.
        """
        return self._volatility_type

    @property
    def shift(self) -> float:
        """Lognormal shift (used when ``volatility_type == ShiftedLognormal``)."""
        return self._shift

    def market_value(self) -> float:
        """Return the actual instrument price (cached, triggers calculate).

        # C++ parity: calibrationhelper.hpp:73 — ``marketValue()``.
        """
        self.calculate()
        return self._market_value

    # --- concrete-by-subclass -----------------------------------------

    @abstractmethod
    def model_value(self) -> float:
        """Return the price of the instrument under the model.

        # C++ parity: calibrationhelper.hpp:76 — ``modelValue() = 0``.
        """
        ...

    @abstractmethod
    def add_times_to(self, times: list[float]) -> None:
        """Append every time the model needs to know about to ``times``.

        # C++ parity: calibrationhelper.hpp:81 — ``addTimesTo(list<Time>&) = 0``.

        Concrete subclasses (SwaptionHelper / CapHelper) compute the
        relevant calibration grid from their schedule.
        """
        ...

    @abstractmethod
    def black_price(self, volatility: float) -> float:
        """Black-or-Bachelier price given a vol.

        # C++ parity: calibrationhelper.hpp:91 — ``blackPrice(Volatility) = 0``.

        Concrete subclasses know whether to use ``black_formula``
        (ShiftedLognormal) or ``bachelier_black_formula`` (Normal).
        """
        ...

    # --- calibration error -------------------------------------------

    def calibration_error(self) -> float:
        """Return the calibration error per ``calibration_error_type``.

        # C++ parity: calibrationhelper.cpp:38-72.
        """
        cet = self._calibration_error_type

        if cet == CalibrationErrorType.RelativePriceError:
            m = self.market_value()
            return abs(m - self.model_value()) / m

        if cet == CalibrationErrorType.PriceError:
            return self.market_value() - self.model_value()

        # ImpliedVolError — solve for the vol that the model price
        # implies, compare with the market vol.
        min_vol = 0.0010 if self._volatility_type == VolatilityType.ShiftedLognormal else 0.00005
        max_vol = 10.0 if self._volatility_type == VolatilityType.ShiftedLognormal else 0.50
        lower_price = self.black_price(min_vol)
        upper_price = self.black_price(max_vol)
        model_price = self.model_value()

        if model_price <= lower_price:
            implied = min_vol
        elif model_price >= upper_price:
            implied = max_vol
        else:
            implied = self.implied_volatility(model_price, 1e-12, 5000, min_vol, max_vol)
        return implied - self._volatility.value()

    def implied_volatility(
        self,
        target_value: float,
        accuracy: float,
        max_evaluations: int,
        min_vol: float,
        max_vol: float,
    ) -> float:
        """Black volatility implied by the model price.

        # C++ parity: calibrationhelper.cpp:26-36.

        Brent-solver inversion of ``black_price``.
        """

        def error(x: float) -> float:
            return target_value - self.black_price(x)

        solver = Brent()
        solver.set_max_evaluations(max_evaluations)
        return solver.solve(error, accuracy, self._volatility.value(), min_vol, max_vol)

    # --- pricing engine ----------------------------------------------

    def set_pricing_engine(self, engine: PricingEngine) -> None:
        """Attach a pricing engine for ``model_value`` evaluation.

        # C++ parity: calibrationhelper.hpp:93-95 — ``setPricingEngine``.
        """
        self._engine = engine

    @property
    def pricing_engine(self) -> PricingEngine | None:
        """The currently attached pricing engine, if any."""
        return self._engine
