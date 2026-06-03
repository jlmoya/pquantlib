"""Calibrate the 5 Heston parameters to a market volatility smile."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from pquantlib_showcase import charts, ui
from pquantlib_showcase.quant import heston

ui.setup_page("Heston Calibration", icon="🛠")
ui.about_sidebar()
ui.hero(
    "Heston calibration",
    "Fit the model to the market: Levenberg-Marquardt over a quoted vol smile.",
    pills=["HestonModelHelper", "LevenbergMarquardt", "model.calibrate(...)"],
)

with st.sidebar:
    spot = st.slider("Spot", 50.0, 150.0, 100.0, 1.0)
    r = st.slider("Risk-free rate", 0.0, 0.10, 0.05, 0.0025, format="%.4f")
    q = st.slider("Dividend yield", 0.0, 0.08, 0.0, 0.0025, format="%.4f")
    t = st.slider("Maturity (years)", 0.25, 3.0, 1.0, 0.25)
    st.subheader("Synthetic market smile")
    atm = st.slider("ATM level", 0.10, 0.40, 0.20, 0.01)
    skew = st.slider("Skew", 0.0, 0.003, 0.0008, 0.0001, format="%.4f")
    convex = st.slider("Convexity", 0.0, 0.0001, 0.00002, 0.000005, format="%.6f")

strikes, market_vols = heston.synthetic_market_smile(atm, skew, convex)

st.subheader("Market quotes")
st.caption("Edit any cell to define your own smile, then the model is re-calibrated to it.")
edited = st.data_editor(
    pd.DataFrame({"Strike": strikes, "Market vol": market_vols}),
    use_container_width=True,
    hide_index=True,
    column_config={"Market vol": st.column_config.NumberColumn(format="%.4f")},
)
strikes = [float(x) for x in edited["Strike"].tolist()]
market_vols = [float(x) for x in edited["Market vol"].tolist()]

with st.spinner("Calibrating Heston to the smile…"):
    cal = heston.calibrate_heston(spot, r, q, t, strikes, market_vols)

st.subheader("Calibrated parameters")
ui.metric_cards(
    [
        ("v₀", f"{cal.v0:.4f}", None),
        ("κ", f"{cal.kappa:.3f}", None),
        ("θ", f"{cal.theta:.4f}", None),
        ("σ", f"{cal.sigma:.3f}", None),
        ("ρ", f"{cal.rho:.3f}", None),
    ]
)
ui.metric_cards(
    [
        ("Fit quality (RMSE)", f"{cal.rmse_bps:.1f} bps", None),
        ("Feller 2κθ > σ²", "satisfied" if cal.feller_satisfied else "violated", None),
    ]
)

st.subheader("Market vs. calibrated model")
fig = charts.smile(cal.strikes, cal.market_vols, cal.model_vols, model_name="Calibrated Heston", spot=spot)
st.plotly_chart(fig, use_container_width=True)
ui.note(
    "The optimiser adjusts all five parameters until the model's implied-vol smile (line) sits on top of the "
    "market quotes (dots). A few-basis-point RMSE means PQuantLib's analytic engine, calibration helpers and "
    "Levenberg-Marquardt optimiser are all working together correctly."
)
