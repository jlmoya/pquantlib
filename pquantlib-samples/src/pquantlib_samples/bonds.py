"""Bonds sample — PENDING (blocked on a functional ``FixedRateBondHelper``).

Port target: QuantLib's ``Examples/Bonds`` — bootstraps a depo-bond discounting
curve and a depo-swap forecasting curve, then prices a zero-coupon, a
fixed-rate and a floating-rate bond against them.

This sample is genuinely blocked on the bond-helper bootstrap path: pquantlib
ships ``BondHelper`` but its ``implied_quote()`` is an explicit deferred stub
(``"requires L3 Bond + DiscountingBondEngine (deferred to L3)"`` — never
closed), and there is no functional ``FixedRateBondHelper``. The C++ sample's
curve is bootstrapped *from* fixed-rate bond helpers, so it cannot be
reproduced. (The deposit/swap helper legs do bootstrap, but the depo-bond curve
— the spine of this sample — does not.)

The Java AllSamples also kept ``Bonds`` in its ``pending`` bucket. See
``docs/carve-outs.md`` (samples section). When ``BondHelper.implied_quote`` /
``FixedRateBondHelper`` are completed in pquantlib core, this module can be
filled in following the C++ original.
"""

from __future__ import annotations


def run() -> None:
    raise NotImplementedError(
        "pending: Bonds needs a functional FixedRateBondHelper / "
        "BondHelper.implied_quote (deferred stub in pquantlib core) — "
        "see docs/carve-outs.md"
    )


if __name__ == "__main__":
    run()
