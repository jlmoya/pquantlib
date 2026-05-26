"""Array — rank-1 dense float64 vector type alias.

# C++ parity: ql/math/array.hpp (v1.42.1) — ``QuantLib::Array``.

C++ uses a hand-rolled ``Array`` class wrapping a ``std::valarray`` (and
historically ``boost::numeric::ublas::vector``). JQuantLib reimplemented
this as a custom ``Array`` class because the JVM lacked vectorized
primitive arrays at the time of porting.

PQuantLib diverges: ``Array`` is a **type alias** for
``numpy.typing.NDArray[numpy.float64]``. The rationale:

- numpy provides every operation QuantLib needs (elementwise arithmetic,
  dot products, slicing) with BLAS-backed performance for free.
- No need for a wrapper class — pyright can statically narrow ``Array``
  to a 1-D float64 vector everywhere, and runtime is plain numpy.
- The C++ API surface that returned ``Array`` (e.g. ``Curve::data()``)
  becomes a function returning ``Array``; callers use numpy semantics.

The rank-1 invariant is documented but not enforced at type-checker
level — pyright cannot distinguish 1-D from 2-D ``NDArray``. Callers
that mix ranks must validate at runtime via ``array.ndim == 1``.

Construction helpers (when needed) should use ``numpy.asarray(..., dtype=float64)``
or ``numpy.array(..., dtype=float64)`` at the call site; no factory
class is exported here.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

# RANK INVARIANT: 1-D. Pyright cannot enforce this at the type level
# (NDArray has no rank parameter in current numpy stubs); callers MUST
# treat ``Array`` as a rank-1 ``float64`` vector. Use ``Matrix`` for rank-2.
type Array = npt.NDArray[np.float64]
