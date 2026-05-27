"""Bachelier (normal) cap/floor engine — re-export from black_capfloor_engine.

# C++ parity: ql/pricingengines/capfloor/bacheliercapfloorengine.{hpp,cpp}
# (v1.42.1). PQuantLib keeps the implementation in
# :mod:`black_capfloor_engine` (factored loop) and re-exports the
# concrete class here so imports mirror the C++ filename convention.
"""

from __future__ import annotations

from pquantlib.pricingengines.capfloor.black_capfloor_engine import (
    BachelierCapFloorEngine,
)

__all__ = ["BachelierCapFloorEngine"]
