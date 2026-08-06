"""Historical volatility estimators and the GARCH(1,1) compositor.

# C++ parity: ql/models/volatility/ @ v1.43.

Everything here implements one of the two abstract bases in
``pquantlib.volatility_model``:

* ``LocalVolatilityEstimator`` — ``SimpleLocalEstimator`` (close-to-close)
  and the whole Garman-Klass family (OHLC bars).
* ``VolatilityCompositor`` — ``ConstantEstimator`` (rolling window) and
  ``Garch11`` (fitted GARCH(1,1)).
"""
