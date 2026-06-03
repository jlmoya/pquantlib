"""Heston stochastic volatility: the implied-vol smile that Black-Scholes can't make."""

from __future__ import annotations

import numpy as np
import streamlit as st

from pquantlib_showcase import charts, ui
from pquantlib_showcase.quant import heston

ui.setup_page("Heston Model", icon="🌊")
ui.about_sidebar()
ui.hero(
    "Heston stochastic volatility",
    "A mean-reverting variance process produces a realistic volatility smile.",
    pills=["HestonProcess", "HestonModel", "AnalyticHestonEngine"],
)

with st.sidebar:
    st.subheader("Market")
    spot = st.slider("Spot", 50.0, 150.0, 100.0, 1.0)
    r = st.slider("Risk-free rate", 0.0, 0.10, 0.05, 0.0025, format="%.4f")
    q = st.slider("Dividend yield", 0.0, 0.08, 0.0, 0.0025, format="%.4f")
    t = st.slider("Maturity (years)", 0.1, 3.0, 1.0, 0.1)
    st.subheader("Heston parameters")
    v0 = st.slider("v₀  (initial variance)", 0.005, 0.20, 0.04, 0.005)
    kappa = st.slider("κ  (mean reversion)", 0.1, 10.0, 2.0, 0.1)
    theta = st.slider("θ  (long-run variance)", 0.005, 0.20, 0.04, 0.005)
    sigma = st.slider("σ  (vol of vol)", 0.05, 1.50, 0.30, 0.05)
    rho = st.slider("ρ  (spot/vol correlation)", -0.95, 0.95, -0.70, 0.05)

strikes = list(np.linspace(70.0, 130.0, 25))
sm = heston.heston_smile(spot, r, q, v0, kappa, theta, sigma, rho, t, strikes)

atm_vol = float(np.sqrt(v0))
ui.metric_cards(
    [
        ("ATM implied vol", f"{sm.implied_vols[len(sm.implied_vols) // 2]:.2%}", None),
        ("√v₀ (short-vol)", f"{atm_vol:.2%}", None),
        ("√θ (long-vol)", f"{np.sqrt(theta):.2%}", None),
        ("Feller 2κθ > σ²", "satisfied" if sm.feller_satisfied else "violated", None),
    ]
)

st.subheader("Heston smile vs. flat Black-Scholes")
bs_flat = [atm_vol] * len(strikes)
fig = charts.smile(strikes, None, sm.implied_vols, model_name="Heston implied vol", spot=spot)
fig.add_scatter(
    x=strikes,
    y=bs_flat,
    mode="lines",
    name="Black-Scholes (flat)",
    line=dict(color=charts.MUTED, width=2, dash="dash"),
)
st.plotly_chart(fig, use_container_width=True)
ui.note(
    "Black-Scholes assumes a single flat volatility (dashed). Heston's correlated, mean-reverting variance "
    "bends it into a <b>smile/skew</b> — negative ρ tilts it down to the left, matching equity markets. "
    "σ (vol of vol) controls the curvature."
)

st.subheader("Heston call prices across strikes")
st.plotly_chart(
    charts.line(strikes, sm.prices, xtitle="Strike", ytitle="Call price", name="Heston", fill=True),
    use_container_width=True,
)
