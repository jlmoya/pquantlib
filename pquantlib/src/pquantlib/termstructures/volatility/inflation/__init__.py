"""Inflation volatility term-structures (CPI + YoY).

# C++ parity: ql/termstructures/volatility/inflation/ (v1.42.1).
"""

from pquantlib.termstructures.volatility.inflation.constant_cpi_volatility import (
    ConstantCPIVolatility,
)
from pquantlib.termstructures.volatility.inflation.constant_yoy_optionlet_volatility import (
    ConstantYoYOptionletVolatility,
)
from pquantlib.termstructures.volatility.inflation.cpi_volatility_surface import (
    CPIVolatilitySurface,
)
from pquantlib.termstructures.volatility.inflation.yoy_optionlet_volatility_surface import (
    YoYOptionletVolatilitySurface,
)

__all__ = [
    "CPIVolatilitySurface",
    "ConstantCPIVolatility",
    "ConstantYoYOptionletVolatility",
    "YoYOptionletVolatilitySurface",
]
