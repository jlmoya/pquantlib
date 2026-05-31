"""Pathwise-greeks Jacobians for the BGM/LIBOR market model.

# C++ parity: ql/models/marketmodels/pathwisegreeks/ (v1.42.1).

The analytic building blocks of pathwise market vegas:

- ``RatePseudoRootJacobian`` / ``RatePseudoRootJacobianAllElements`` /
  ``RatePseudoRootJacobianNumerical`` — the Jacobian of the one-step
  log-Euler forward evolution with respect to the pseudo-root elements.
- ``SwaptionPseudoDerivative`` / ``CapPseudoDerivative`` — the derivative of a
  swaption / cap implied vol (and variance / price) with respect to the
  pseudo-root.
- ``VegaBumpCluster`` / ``VegaBumpCollection`` — deterministic clustering of
  pseudo-root elements into bump groups.
- ``VolatilityBumpInstrumentJacobian`` / ``OrthogonalizedBumpFinder`` — the
  instrument-level vega-bump Jacobian and the orthogonalised bump construction
  consumed by the pathwise-vegas accounting engine.
"""
