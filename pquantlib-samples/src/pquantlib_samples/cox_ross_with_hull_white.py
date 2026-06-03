"""CoxRossWithHullWhite sample — PENDING (blocked on extended binomial trees).

Port target: QuantLib's equity-option example variant that prices European /
Bermudan / American options with a ``BinomialVanillaEngine<ExtendedCoxRossRubinstein>``
built over a ``HullWhiteProcess`` (a stochastic-rate binomial tree).

This sample is genuinely blocked. pquantlib core does **not** port:

* the *extended* binomial-tree family (``ExtendedCoxRossRubinstein`` /
  ``ExtendedJarrowRudd`` / etc.) that accept a time-varying drift; the ported
  ``BinomialVanillaEngine`` only accepts a :class:`GeneralizedBlackScholesProcess`
  (constant-coefficient trees), not a short-rate ``HullWhiteProcess``;
* an equity-style ``HullWhiteProcess`` usable as the binomial diffusion (only
  ``HullWhiteForwardProcess`` and the short-rate ``HullWhite`` model exist).

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
