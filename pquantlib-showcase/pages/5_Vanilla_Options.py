"""Vanilla options priced four independent ways — the headline demonstration."""

from __future__ import annotations

import numpy as np
import streamlit as st

from pquantlib_showcase import charts, ui
from pquantlib_showcase.quant import options

ui.setup_page("Vanilla Options", icon="🎯")
ui.about_sidebar()
ui.hero(
    "Vanilla options, four engines",
    "Closed-form, binomial tree, Monte Carlo and finite differences — one contract.",
    pills=[
        "AnalyticEuropeanEngine",
        "BinomialVanillaEngine",
        "MCEuropeanEngine",
        "FdBlackScholesVanillaEngine",
    ],
)

with st.sidebar:
    kind = st.radio("Type", ["Call", "Put"], horizontal=True)
    spot = st.slider("Spot", 50.0, 150.0, 100.0, 1.0)
    strike = st.slider("Strike", 50.0, 150.0, 100.0, 1.0)
    r = st.slider("Risk-free rate", 0.0, 0.10, 0.05, 0.0025, format="%.4f")
    q = st.slider("Dividend yield", 0.0, 0.08, 0.02, 0.0025, format="%.4f")
    vol = st.slider("Volatility", 0.05, 0.80, 0.20, 0.01)
    t = st.slider("Maturity (years)", 0.1, 3.0, 1.0, 0.1)
    steps = st.slider("Binomial steps", 10, 1000, 500, 10)
    samples = st.select_slider("MC samples", [5_000, 10_000, 20_000, 50_000, 100_000], value=20_000)

res = options.price_vanilla(kind, spot, strike, r, q, vol, t, binomial_steps=steps, mc_samples=samples)

st.subheader("Same option, four engines")
ui.metric_cards(
    [
        ("Closed-form (Black-Scholes)", f"{res.analytic:.4f}", None),
        ("Binomial tree", f"{res.binomial:.4f}", f"{res.binomial - res.analytic:+.4f}"),
        ("Monte Carlo", f"{res.mc:.4f}", f"±{res.mc_error:.3f} (1σ)"),
        ("Finite differences", f"{res.fd:.4f}", f"{res.fd - res.analytic:+.4f}"),
    ]
)
ui.note(
    "Four numerically independent methods converge on the same value — the strongest "
    "possible evidence that the pricing stack is correct. Greeks below come from the closed-form engine."
)
ui.metric_cards(
    [
        ("Delta", f"{res.delta:.4f}", None),
        ("Gamma", f"{res.gamma:.4f}", None),
        ("Vega", f"{res.vega:.4f}", None),
        ("Theta", f"{res.theta:.4f}", None),
        ("Rho", f"{res.rho:.4f}", None),
    ]
)

st.divider()
left, right = st.columns(2)
with left:
    st.subheader("Binomial convergence")
    ns, prices, target = options.binomial_convergence(kind, spot, strike, r, q, vol, t, max_steps=200)
    st.plotly_chart(charts.convergence(ns, prices, target), use_container_width=True)
    ui.note("The classic oscillating convergence of a Cox-Ross-Rubinstein tree onto the closed-form price.")

with right:
    st.subheader("Payoff vs. value today")
    spots = list(np.linspace(spot * 0.5, spot * 1.5, 80))
    payoff = options.payoff_at_expiry(kind, strike, spots)
    value = [
        options.price_vanilla(kind, s, strike, r, q, vol, t, binomial_steps=50, mc_samples=5000).analytic
        for s in spots
    ]
    st.plotly_chart(
        charts.payoff_and_value(spots, payoff, value, spot=spot, strike=strike), use_container_width=True
    )
    ui.note("Time value is the gap between the option's worth today and its payoff at expiry.")
