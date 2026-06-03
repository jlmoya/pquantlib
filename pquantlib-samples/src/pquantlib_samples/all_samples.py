"""Sample-program registry — mirrors Java ``org.jquantlib.samples.AllSamples``.

Each entry is the *module name* of a sample under ``pquantlib_samples`` whose
``run()`` performs the computation and prints to stdout (parity with the Java
``Runnable.run()`` originals).

Three buckets mirror the Java arrays:

* ``COMPLETE``   — fully ported samples that run to completion (Java ``complete``).
* ``INCOMPLETE`` — samples that run but whose port is not yet exhaustive
  (Java ``incomplete``).
* ``PENDING``    — samples not yet ported; smoke-tested via a skipped test that
  mirrors Java ``@Ignore`` (the Java ``pending`` array).

A module name MUST NOT appear in ``COMPLETE`` / ``INCOMPLETE`` unless its module
file exists and ``run()`` succeeds — the smoke suite parametrizes over these and
would otherwise fail. As later W-S5 clusters land their samples, they move the
corresponding name out of ``PENDING`` into ``COMPLETE`` / ``INCOMPLETE``.
"""

from __future__ import annotations

# Fully-ported samples whose ``run()`` executes to completion (Java ``complete``
# plus everything W-S5-B verified runs cleanly in pquantlib).
COMPLETE: tuple[str, ...] = (
    "calendars",
    "dates",
    "swap",
    "repo",
    "convertible_bonds",
    "processes",
    "yield_curve_term_structures",
    "fra",
    "discrete_hedging",
    "sobol_chart_sample",
    "volatility_term_structures",
    "bermudan_swaption",
)

# Samples that run but whose port is not exhaustive (mirrors Java ``incomplete``).
# ``equity_options``: the American analytic approximations (Barone-Adesi/Whaley,
# Bjerksund/Stensland, Ju Quadratic, Integral) are not ported in pquantlib, so
# those rows print N/A; everything else (Black-Scholes, 4 binomial trees, FD,
# Monte Carlo) runs.
INCOMPLETE: tuple[str, ...] = ("equity_options",)

# Samples genuinely blocked by a core class that does NOT exist in pquantlib
# (verified by re-inspection). Their modules exist but ``run()`` raises
# NotImplementedError; the smoke suite skips them (see ``docs/carve-outs.md``):
#   * bonds                   — needs a functional FixedRateBondHelper /
#                               BondHelper.implied_quote (deferred core stub).
#   * replication             — needs CompositeInstrument (not ported).
#   * cox_ross_with_hull_white — needs the extended binomial-tree family
#                               (ExtendedCoxRossRubinstein) over a HullWhiteProcess.
PENDING: tuple[str, ...] = (
    "bonds",
    "replication",
    "cox_ross_with_hull_white",
)

__all__: list[str] = ["COMPLETE", "INCOMPLETE", "PENDING"]
