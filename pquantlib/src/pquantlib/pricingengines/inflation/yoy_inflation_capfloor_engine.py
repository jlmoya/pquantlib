"""YoY inflation cap/floor base engine + Bachelier / Black / UnitDisplaced concretes.

# C++ parity: ql/pricingengines/inflation/inflationcapfloorengines.{hpp,cpp}
   (v1.42.1).

For each optionlet in the YoY cap/floor leg the engine:
1. fetches the forward YoY rate from ``index.yoy_inflation_term_structure().yoy_rate(fixing_date)``,
2. fetches the discount factor on the nominal curve at ``pay_date``,
3. multiplies into ``discounted_accrual = nominal * gearing * accrual_time * df``,
4. fetches ``std_dev = sqrt(volatility.total_variance(fixing_date, strike, Period(0, Days)))``,
5. calls ``optionlet_impl(call/put, strike, forward, std_dev, discounted_accrual)``.

Subclasses (Black / Bachelier / UnitDisplacedBlack) provide ``optionlet_impl``.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from pquantlib import qassert
from pquantlib.instruments.yoy_inflation_capfloor import (
    YoYInflationCapFloorArguments,
    YoYInflationCapFloorResults,
    YoYInflationCapFloorType,
)
from pquantlib.payoffs import OptionType
from pquantlib.pricingengines.black_formula import (
    bachelier_black_formula,
    black_formula,
)
from pquantlib.pricingengines.generic_engine import GenericEngine
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit

if TYPE_CHECKING:
    from pquantlib.indexes.inflation.inflation_index import YoYInflationIndex
    from pquantlib.termstructures.protocols import YieldTermStructureProtocol
    from pquantlib.termstructures.volatility.inflation.yoy_optionlet_volatility_surface import (
        YoYOptionletVolatilitySurface,
    )

# Sentinel used by the YoY surface "use my default lag" path.
_ZERO_LAG = Period(0, TimeUnit.Days)


class YoYInflationCapFloorEngine(
    GenericEngine[YoYInflationCapFloorArguments, YoYInflationCapFloorResults]
):
    """Base YoY inflation cap/floor engine.

    # C++ parity: ``YoYInflationCapFloorEngine`` (inflationcapfloorengines.hpp:44-67).

    The index must be linked to a YoY-inflation term structure (the
    forward path uses it); the nominal term structure is used for
    discounting only.
    """

    def __init__(
        self,
        index: YoYInflationIndex,
        volatility: YoYOptionletVolatilitySurface | None,
        nominal_term_structure: YieldTermStructureProtocol,
    ) -> None:
        # # C++ parity: the vol argument is a ``Handle`` which may be empty —
        # # the optionlet stripper constructs the engine with no vol surface
        # # and sets it during the per-K bootstrap via ``set_volatility``.
        super().__init__(
            YoYInflationCapFloorArguments(), YoYInflationCapFloorResults()
        )
        self._index: YoYInflationIndex = index
        self._volatility: YoYOptionletVolatilitySurface | None = volatility
        self._nominal_term_structure: YieldTermStructureProtocol = nominal_term_structure
        # # C++ parity: registerWith(index/volatility/nominalTermStructure).
        # # PQuantLib's Engine base doesn't auto-register; concrete callers
        # # may explicitly set up observer wiring if needed.

    # --- inspectors ----------------------------------------------------

    def index(self) -> YoYInflationIndex:
        return self._index

    def volatility(self) -> YoYOptionletVolatilitySurface:
        qassert.require(self._volatility is not None, "volatility surface not set")
        assert self._volatility is not None
        return self._volatility

    def nominal_term_structure(self) -> YieldTermStructureProtocol:
        return self._nominal_term_structure

    def set_volatility(self, vol: YoYOptionletVolatilitySurface) -> None:
        """Replace the vol surface.

        # C++ parity: YoYInflationCapFloorEngine::setVolatility
        # (inflationcapfloorengines.cpp:41-48).
        """
        self._volatility = vol
        self.update()

    # --- engine --------------------------------------------------------

    def calculate(self) -> None:
        # # C++ parity: YoYInflationCapFloorEngine::calculate
        # # (inflationcapfloorengines.cpp:51-128).
        args = self._arguments
        results = self._results
        results.reset()
        qassert.require(
            self._volatility is not None,
            "YoY cap/floor engine: volatility surface not set",
        )
        assert self._volatility is not None

        n = len(args.start_dates)
        values: list[float] = [0.0] * n
        std_devs: list[float] = [0.0] * n
        forwards: list[float] = [0.0] * n

        cap_type = args.type
        ref_date = self._nominal_term_structure.reference_date()

        # YoY-rate forecast curve sits on the index handle.
        yoy_ts_obj = self._index.yoy_inflation_term_structure()
        qassert.require(
            yoy_ts_obj is not None,
            "YoY index needs a YoY-inflation term-structure for cap/floor pricing",
        )
        # The Protocol slot is typed object | None on the index; cast for
        # the engine path (concrete types come from L7-B at merge).
        yoy_ts = yoy_ts_obj
        assert yoy_ts is not None

        total_value = 0.0
        for i in range(n):
            payment_date = args.pay_dates[i]
            if payment_date <= ref_date:
                # Expired — skip (matches C++ "discard expired caplets").
                continue

            df = self._nominal_term_structure.discount(payment_date)
            discounted_accrual = (
                args.nominals[i] * args.gearings[i] * df * args.accrual_times[i]
            )

            # # C++ parity: forwards[i] = yoyTS->yoyRate(arguments_.fixingDates[i]);
            forwards[i] = yoy_ts.yoy_rate(args.fixing_dates[i])  # type: ignore[attr-defined]
            forward = forwards[i]

            fixing_date = args.fixing_dates[i]
            # # C++ parity: sqrtTime = sqrt(volatility->timeFromBase(fixingDate))
            # # but we use totalVariance() directly to get vol^2 * time.
            sqrt_time_check = fixing_date > self._volatility.base_date()

            # Cap leg.
            if cap_type in (
                YoYInflationCapFloorType.Cap,
                YoYInflationCapFloorType.Collar,
            ):
                strike = args.cap_rates[i]
                if sqrt_time_check:
                    std_devs[i] = math.sqrt(
                        self._volatility.total_variance(fixing_date, strike, _ZERO_LAG)
                    )
                # std_dev = 0 for already-fixed → intrinsic value.
                values[i] = self._optionlet_impl(
                    OptionType.Call, strike, forward, std_devs[i], discounted_accrual
                )

            # Floor leg.
            if cap_type in (
                YoYInflationCapFloorType.Floor,
                YoYInflationCapFloorType.Collar,
            ):
                strike = args.floor_rates[i]
                if sqrt_time_check:
                    std_devs[i] = math.sqrt(
                        self._volatility.total_variance(fixing_date, strike, _ZERO_LAG)
                    )
                floorlet = self._optionlet_impl(
                    OptionType.Put, strike, forward, std_devs[i], discounted_accrual
                )
                if cap_type == YoYInflationCapFloorType.Floor:
                    values[i] = floorlet
                else:
                    # Collar: long cap, short floor → subtract floorlet.
                    values[i] -= floorlet

            total_value += values[i]

        results.value = total_value
        results.additional_results["optionletsPrice"] = values
        results.additional_results["optionletsAtmForward"] = forwards
        if cap_type != YoYInflationCapFloorType.Collar:
            results.additional_results["optionletsStdDev"] = std_devs

    # --- subclass hook -------------------------------------------------

    def _optionlet_impl(
        self,
        option_type: OptionType,
        strike: float,
        forward: float,
        std_dev: float,
        d: float,
    ) -> float:
        """Per-optionlet pricing. Subclasses override.

        # C++ parity: ``optionletImpl`` pure virtual (inflationcapfloorengines.hpp:60-62).
        """
        del option_type, strike, forward, std_dev, d
        qassert.fail("subclass must implement _optionlet_impl")


class YoYInflationBlackCapFloorEngine(YoYInflationCapFloorEngine):
    """Black-formula YoY inflation cap/floor engine.

    # C++ parity: ``YoYInflationBlackCapFloorEngine``
    # (inflationcapfloorengines.hpp:72-81 + .cpp:135-148).
    """

    def _optionlet_impl(
        self,
        option_type: OptionType,
        strike: float,
        forward: float,
        std_dev: float,
        d: float,
    ) -> float:
        # # C++ parity: blackFormula(type, strike, forward, stdDev, d).
        return black_formula(option_type, strike, forward, std_dev, d, 0.0)


class YoYInflationUnitDisplacedBlackCapFloorEngine(YoYInflationCapFloorEngine):
    """Unit-displaced Black YoY cap/floor engine.

    # C++ parity: ``YoYInflationUnitDisplacedBlackCapFloorEngine``
    # (inflationcapfloorengines.hpp:85-95 + .cpp:152-168).

    Implements ``blackFormula(type, strike + 1, forward + 1, stdDev, d)``,
    which is the algebraically-equivalent +1.0 displacement form. We
    follow C++'s explicit "clearer" formulation rather than passing
    ``displacement=1`` to ``black_formula``.
    """

    def _optionlet_impl(
        self,
        option_type: OptionType,
        strike: float,
        forward: float,
        std_dev: float,
        d: float,
    ) -> float:
        # # C++ parity: blackFormula(type, strike+1.0, forward+1.0, stdDev, d).
        return black_formula(option_type, strike + 1.0, forward + 1.0, std_dev, d, 0.0)


class YoYInflationBachelierCapFloorEngine(YoYInflationCapFloorEngine):
    """Bachelier (Normal) YoY inflation cap/floor engine.

    # C++ parity: ``YoYInflationBachelierCapFloorEngine``
    # (inflationcapfloorengines.hpp:98-109 + .cpp:171-184).
    """

    def _optionlet_impl(
        self,
        option_type: OptionType,
        strike: float,
        forward: float,
        std_dev: float,
        d: float,
    ) -> float:
        # # C++ parity: bachelierBlackFormula(type, strike, forward, stdDev, d).
        return bachelier_black_formula(option_type, strike, forward, std_dev, d)


__all__ = [
    "YoYInflationBachelierCapFloorEngine",
    "YoYInflationBlackCapFloorEngine",
    "YoYInflationCapFloorEngine",
    "YoYInflationUnitDisplacedBlackCapFloorEngine",
]
