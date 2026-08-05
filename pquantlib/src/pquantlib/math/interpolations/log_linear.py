"""Log-linear interpolation between discrete points.

# C++ parity: ql/math/interpolations/loginterpolation.hpp (v1.43) —
# ``class LogLinearInterpolation``.

The class itself now lives in
:mod:`pquantlib.math.interpolations.log_interpolation`, alongside the rest
of the ``loginterpolation.hpp`` family, because they all share the one
``detail::LogInterpolationImpl`` wrapper and duplicating it here would mean
two copies of ``exp(inner(log(y)))`` that could drift apart. This module
stays as the import path the term-structure code already uses.
"""

from __future__ import annotations

from pquantlib.math.interpolations.log_interpolation import LogLinearInterpolation

__all__ = ["LogLinearInterpolation"]
