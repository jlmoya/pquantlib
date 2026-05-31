"""Experimental volatility surfaces.

# C++ parity: ql/experimental/volatility/* (v1.42.1).

Black / equity-FX / interest-rate volatility surface bases plus a few
concrete fits:

- :class:`BlackAtmVolCurve` — ATM (no-smile) Black vol curve base.
- :class:`BlackVolSurface` — Black smile surface base (per-expiry
  :class:`~pquantlib.termstructures.volatility.smile_section.SmileSection`).
- :class:`EquityFXVolSurface` — equity/FX smile surface base
  (adds ATM-forward vol/variance).
- :class:`InterestRateVolSurface` — IR smile surface base
  (index-driven optionlet date conversion).
- :class:`VolatilityCube` — generic IR vol-cube aggregation base.
- :class:`AbcdAtmVolCurve` — Abcd-parametric ATM IR vol curve (concrete).
- :class:`ExtendedBlackVarianceCurve` /
  :class:`ExtendedBlackVarianceSurface` — Quote-backed Black variance
  term structures with a selectable interpolator (concrete).
- :class:`SabrVolSurface` — interpolated SABR vol surface from market
  vol spreads (concrete).
- :class:`SABRVolTermStructure` — implied-vol surface backed by a single
  SABR parameter set (concrete).
"""
