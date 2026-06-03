"""CoxRossWithHullWhite sample — PENDING (blocked on engine wiring + HullWhiteProcess).

Port target: QuantLib's equity-option example variant that prices European /
Bermudan / American options with a ``BinomialVanillaEngine<ExtendedCoxRossRubinstein>``
built over a ``HullWhiteProcess`` (a stochastic-rate binomial tree).

This sample is genuinely blocked. The extended binomial-tree family
(``ExtendedCoxRossRubinstein``, ``ExtendedJarrowRudd``, etc.) IS ported at
``pquantlib.experimental.lattices.extended_binomial_tree``. However:

* No engine wires the extended binomial trees to a diffusion process. The ported
  ``BinomialVanillaEngine`` only accepts a :class:`GeneralizedBlackScholesProcess`
  (constant-coefficient trees), not the extended trees or a short-rate process.
* There is no equity-style ``HullWhiteProcess`` usable as the binomial diffusion.
  Only ``HullWhiteForwardProcess`` and the short-rate ``HullWhite`` model are
  ported.

The Java original threw ``UnsupportedOperationException`` for the same reason
(its binomial-with-Hull-White lines were commented out).

See ``docs/carve-outs.md`` (samples section).
"""

from __future__ import annotations


def run() -> None:
    raise NotImplementedError(
        "pending: CoxRossWithHullWhite needs the extended binomial-tree family "
        "(ExtendedCoxRossRubinstein) over a HullWhiteProcess (not ported to "
        "pquantlib core) — see docs/carve-outs.md"
    )


if __name__ == "__main__":
    run()
