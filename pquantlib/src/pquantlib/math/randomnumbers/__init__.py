"""Random number generators (uniform + Gaussian).

# C++ parity: ql/math/randomnumbers/* (v1.42.1).

L1-D cluster scope (this batch):

- ``RandomNumberGenerator`` Protocol + ``Sample`` dataclass
- 5 uniform RNGs: ``MersenneTwisterUniformRng``, ``KnuthUniformRng``,
  ``LecuyerUniformRng``, ``Ranlux3UniformRng``, ``Xoshiro256StarStarUniformRng``
- 1 Gaussian wrapper: ``BoxMullerGaussianRng``

Carve-outs (deferred to follow-up clusters): SobolRsg, Burley2020,
Halton, Faure, InverseCumulativeRng, SeedGenerator.
"""
