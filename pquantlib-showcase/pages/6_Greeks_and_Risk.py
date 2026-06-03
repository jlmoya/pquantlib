"""Greek profiles across spot, and the implied-volatility inversion."""

from __future__ import annotations

import streamlit as st

from pquantlib_showcase import charts, ui
from pquantlib_showcase.quant import options

ui.setup_page("Greeks & Risk", icon="📐")
ui.about_sidebar()
ui.hero(
    "Greeks & risk",
    "How option sensitivities evolve across the underlying — and implied vol, inverted.",
    pills=["delta · gamma · vega · theta", "implied_volatility"],
)

with st.sidebar:
    kind = st.radio("Type", ["Call", "Put"], horizontal=True)
    strike = st.slider("Strike", 50.0, 150.0, 100.0, 1.0)
    r = st.slider("Risk-free rate", 0.0, 0.10, 0.05, 0.0025, format="%.4f")
    q = st.slider("Dividend yield", 0.0, 0.08, 0.0, 0.0025, format="%.4f")
    vol = st.slider("Volatility", 0.05, 0.80, 0.20, 0.01)
    t = st.slider("Maturity (years)", 0.1, 3.0, 1.0, 0.1)

prof = options.greeks_vs_spot(kind, strike, r, q, vol, t, strike * 0.5, strike * 1.5)

st.subheader("Greek profiles across spot")
c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(
        charts.line(prof.spots, prof.delta, xtitle="Spot", ytitle="Delta", name="Delta", color=charts.ACCENT),
        use_container_width=True,
    )
    st.plotly_chart(
        charts.line(prof.spots, prof.vega, xtitle="Spot", ytitle="Vega", name="Vega", color=charts.ACCENT3),
        use_container_width=True,
    )
with c2:
    st.plotly_chart(
        charts.line(
            prof.spots, prof.gamma, xtitle="Spot", ytitle="Gamma", name="Gamma", color=charts.ACCENT2
        ),
        use_container_width=True,
    )
    st.plotly_chart(
        charts.line(prof.spots, prof.theta, xtitle="Spot", ytitle="Theta", name="Theta", color="#9333ea"),
        use_container_width=True,
    )
ui.note(
    "Gamma and vega peak around the at-the-money strike — where the option's value is most "
    "sensitive to spot and to volatility. Delta sweeps from 0 to 1 (calls) as the option goes in the money."
)

st.divider()
st.subheader("Implied volatility — inverting the price")
ui.note(
    "Enter a market price; PQuantLib's Brent solver recovers the Black-Scholes volatility that reproduces it."
)
cc1, cc2, cc3 = st.columns(3)
with cc1:
    spot = st.number_input("Spot", 50.0, 150.0, 100.0)
with cc2:
    true_vol = st.slider("True volatility (to generate a price)", 0.05, 0.80, 0.25, 0.01)
price = options.price_vanilla(kind, spot, strike, r, q, true_vol, t).analytic
recovered = options.implied_vol(kind, spot, strike, r, q, t, price)
with cc3:
    st.metric("Market price", f"{price:.4f}")
ui.metric_cards(
    [
        ("Implied volatility (recovered)", f"{recovered:.4%}", f"input was {true_vol:.2%}"),
        ("Inversion error", f"{abs(recovered - true_vol) * 1e4:.2f} bps", None),
    ]
)
