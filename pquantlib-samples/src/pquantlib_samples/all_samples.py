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

# --- W-S5-A: foundation + simplest samples (this cluster) -----------------
COMPLETE: tuple[str, ...] = (
    "calendars",
    "dates",
    "swap",
    "repo",
)

INCOMPLETE: tuple[str, ...] = ()

# Names of samples slated for later W-S5 clusters. Kept as forward references
# so the runner mirrors Java ``AllSamples.pending``; they are skipped (not
# imported) by the smoke suite until their module files land.
PENDING: tuple[str, ...] = (
    "bonds",
    "yield_curve_term_structures",
    "bermudan_swaption",
    "fra",
    "processes",
    "replication",
    "discrete_hedging",
    "cox_ross_with_hull_white",
    "sobol_chart_sample",
    "equity_options",
    "volatility_term_structures",
    "convertible_bonds",
)

__all__: list[str] = ["COMPLETE", "INCOMPLETE", "PENDING"]
