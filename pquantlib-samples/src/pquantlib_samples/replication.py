"""Replication sample — PENDING (blocked on ``CompositeInstrument``).

Port target: QuantLib's ``Examples/Replication`` (Mark Joshi, "The Concepts and
Practice of Mathematical Finance" §10.2) — a static replication of a
down-and-out barrier option built with the ``CompositeInstrument`` class.

This sample is genuinely blocked: pquantlib core does **not** port
``CompositeInstrument`` (verified — no class of that name exists under any
module). The Java original was itself an incomplete stub that trailed off after
the column headers without building the stochastic process or the composite.

See ``docs/carve-outs.md`` (samples section). When ``CompositeInstrument`` lands
in pquantlib core, this module can be filled in following the C++ original.
"""

from __future__ import annotations


def run() -> None:
    raise NotImplementedError(
        "pending: Replication needs CompositeInstrument (not ported to "
        "pquantlib core) — see docs/carve-outs.md"
    )


if __name__ == "__main__":
    run()
