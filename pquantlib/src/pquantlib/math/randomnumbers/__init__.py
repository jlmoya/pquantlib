"""Random number generators (uniform, Gaussian, low-discrepancy).

# C++ parity: ql/math/randomnumbers/* (v1.43).

Everything in the C++ directory is here:

- ``RandomNumberGenerator`` / ``UniformSequenceGenerator`` protocols plus the
  ``Sample`` / ``SequenceSample`` dataclasses (C++ ``Sample<T>``);
- uniform RNGs: ``MersenneTwisterUniformRng``, ``KnuthUniformRng``,
  ``LecuyerUniformRng``, ``Ranlux64UniformRng`` (with the ``Ranlux3`` /
  ``Ranlux4`` luxury levels), ``Xoshiro256StarStarUniformRng``;
- Gaussian wrappers: ``BoxMullerGaussianRng``, ``CLGaussianRng``,
  ``ZigguratGaussianRng``, ``InverseCumulativeRng``;
- sequence generators: ``RandomSequenceGenerator``, ``InverseCumulativeRsg``,
  ``HaltonRsg``, ``SobolRsg``, ``Burley2020SobolRsg``, ``FaureRsg``,
  ``LatticeRsg``, ``RandomizedLDS``, ``SobolBrownianBridgeRsg``,
  ``Burley2020SobolBrownianBridgeRsg``;
- support: ``SeedGenerator``, ``LatticeRule``, the ``rng_traits`` policies
  (``PseudoRandom``, ``PoissonPseudoRandom``, ``LowDiscrepancy``) and
  ``StochasticCollocationInvCDF``.

Bulk data (Sobol direction integers, primitive polynomials, lattice-rule
generating vectors) lives under ``data/`` as text resources generated from the
C++ by ``migration-harness/generate_sobol_tables.py``; ``sobol_tables`` loads
it lazily.
"""
