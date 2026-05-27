"""Bachelier (normal) swaption engine — re-export from black_swaption_engine.

# C++ parity: ql/pricingengines/swaption/blackswaptionengine.{hpp,cpp}
# also defines ``BachelierSwaptionEngine``. PQuantLib keeps the
# implementation in :mod:`black_swaption_engine` (matches the C++
# template hierarchy) and re-exports the concrete class here so
# imports can mirror the C++ filename convention.
"""

from __future__ import annotations

from pquantlib.pricingengines.swaption.black_swaption_engine import (
    BachelierSwaptionEngine,
)

__all__ = ["BachelierSwaptionEngine"]
