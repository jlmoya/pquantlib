"""Legacy quantitative-finance models that C++ QuantLib still ships and supports.

# C++ parity: ql/legacy/ (v1.43).

C++ keeps these under ``ql/legacy/`` because the API predates later
conventions, not because the code is unmaintained: it is compiled, tested and
released with every version. The port mirrors the directory so the C++ header
path maps one-to-one onto the Python module path.

Currently: ``libormarketmodels`` — the LIBOR Market Model (a.k.a. BGM / LFM)
volatility and correlation parameterizations, the LFM stochastic process, the
LiborForwardModel and its Black swaption engine.
"""

from __future__ import annotations

__all__: list[str] = []
