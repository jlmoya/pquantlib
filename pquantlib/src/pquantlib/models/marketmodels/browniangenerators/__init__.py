"""MarketModels Brownian generators (W9-C).

# C++ parity: ql/models/marketmodels/browniangenerators/ (v1.42.1).

Concrete ``BrownianGenerator`` implementations driving the BGM evolution:

- ``mt_brownian_generator.MTBrownianGenerator`` / ``MTBrownianGeneratorFactory``
  — Mersenne-Twister uniform stream + inverse-cumulative Gaussian.
- ``sobol_brownian_generator.SobolBrownianGenerator`` /
  ``SobolBrownianGeneratorFactory`` (+ ``SobolBrownianGeneratorBase``) — Sobol
  low-discrepancy stream + inverse-cumulative Gaussian + Brownian bridge with
  Factors / Steps / Diagonal ordering.
"""
