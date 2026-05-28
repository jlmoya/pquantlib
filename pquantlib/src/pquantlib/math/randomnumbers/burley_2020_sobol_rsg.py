"""Burley (2020) Owen-scrambled Sobol' low-discrepancy sequence.

# C++ parity: ql/math/randomnumbers/burley2020sobolrsg.{hpp,cpp} (v1.42.1)
#             ``class Burley2020SobolRsg``.

C++ ``Burley2020SobolRsg`` implements the variant proposed by Burley
("Practical Hash-based Owen Scrambling", JCGT 9 (4), 2020), which
sits atop an unscrambled ``SobolRsg`` and applies a per-draw
Owen-style hash scramble.

The Python port maps this to ``scipy.stats.qmc.Sobol(scramble=True)``,
which performs Owen scrambling internally. scipy 1.7+ uses Matousek's
LMS+shift implementation by default; the resulting sequence is **not
bit-identical** to the C++ Burley2020 sequence (different hash, but
same statistical properties).

This is documented as a divergence: the C++ Burley sequence is
deterministic given (dim, seed); scipy's scrambled sequence is also
deterministic given (dim, seed). Both yield well-equidistributed
sequences with the same low-discrepancy properties — only the exact
floating-point values differ. Cross-validation tests focus on the
**properties** of the sequence (range, deterministic given seed,
non-origin start, etc.) rather than bit-exact match against the C++
probe.
"""

from __future__ import annotations

from typing import Any

from scipy.stats import qmc  # type: ignore[import-untyped]

from pquantlib.math.randomnumbers.sobol_rsg import SobolRsg


class Burley2020SobolRsg(SobolRsg):
    """Owen-scrambled Sobol' sequence (Burley 2020 / scipy LMS+shift).

    # C++ parity: ``Burley2020SobolRsg``. The Python port subclasses
    # ``SobolRsg`` and flips scipy's ``scramble`` flag on; everything
    # else (skip_to, reset, dimension, next_sequence, last_sequence) is
    # inherited from the base class.
    """

    def _make_engine(self, *, scramble: bool) -> Any:
        # C++ parity: Burley2020 always scrambles; the ``scramble`` arg
        # passed by SobolRsg.__init__ is ignored.
        # scipy uses Matousek's LMS+shift Owen scrambling under the
        # hood; well-defined for any seed.
        del scramble
        return qmc.Sobol(d=self._dim, scramble=True, rng=self._seed)
