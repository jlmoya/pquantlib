"""Backwards-compatible alias for :mod:`discounting_fx_forward_engine`.

# C++ parity: ql/pricingengines/forward/discountingfxforwardengine.{hpp,cpp}
#             (v1.43) — the C++ class is ``DiscountingFxForwardEngine``.

ALIGN (v1.43 bondswap wave): this module used to hold the engine itself under
the renamed class ``DiscountingFwdEngine``. Renaming a ported class away from
its C++ name hides it from the coverage gate, which matches on the C++ name
character-for-character, so the implementation now lives in
``discounting_fx_forward_engine.py`` (the snake_case of the C++ header) under
the C++ name. ``DiscountingFwdEngine`` remains here as a plain alias so that
existing call sites keep working; it is the *same object*, not a subclass.
"""

from __future__ import annotations

from pquantlib.pricingengines.forward.discounting_fx_forward_engine import (
    DiscountingFxForwardEngine,
)

#: Legacy name. Identical to :class:`DiscountingFxForwardEngine`.
DiscountingFwdEngine = DiscountingFxForwardEngine

__all__ = ["DiscountingFwdEngine", "DiscountingFxForwardEngine"]
