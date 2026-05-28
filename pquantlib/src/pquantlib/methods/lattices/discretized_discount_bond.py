"""DiscretizedDiscountBond — pays 1 unit at maturity.

# C++ parity: ql/discretizedasset.hpp (v1.42.1) — ``class
#             DiscretizedDiscountBond``.

Trivial subclass of ``DiscretizedAsset``: ``reset(size)`` produces a
constant ``Array(size, 1.0)`` (the bond's terminal payoff is 1.0 in
every state), and ``mandatory_times()`` returns the empty list (the
bond has no intermediate cash flows). The lattice's ``rollback``
machinery handles the discounting back to ``t = 0``.
"""

from __future__ import annotations

import numpy as np

from pquantlib.methods.lattices.discretized_asset import DiscretizedAsset


class DiscretizedDiscountBond(DiscretizedAsset):
    """Zero-coupon discount bond with terminal value 1.

    # C++ parity: ``DiscretizedDiscountBond`` (discretizedasset.hpp:147-152).
    """

    def reset(self, size: int) -> None:
        # C++ parity: ``values_ = Array(size, 1.0)`` (discretizedasset.hpp:150).
        self._values = np.ones(size, dtype=np.float64)

    def mandatory_times(self) -> list[float]:
        # C++ parity: returns empty ``std::vector<Time>``.
        return []
