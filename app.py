"""AI Supply Chain Lab - Streamlit dashboard.

AI in Operations - lab activity and exercise:
  * Bullwhip-effect mini-simulation and AI redesign of the network (forecasting, routing)
  * Exercise: AI-enhanced reorder point - ML demand forecasting, dynamic safety stock with
    Monte Carlo, and Bayesian optimisation of the reorder point under constraints.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import sys
from pathlib import Path

# The engine modules live next to app.py; put this folder first on the import path so
# they are found on any host (and take priority over any same-named installed package).
sys.path.insert(0, str(Path(__file__).resolve().parent))

import bullwhip as bw  # noqa: E402
import data as sd  # noqa: E402
import forecasting as fc  # noqa: E402
import inventory as inv  # noqa: E402
import routing as rt  # noqa: E402
from optimization import bayesian_optimize  # noqa: E402

st.set_page_config(page_title="AI Supply Chain Lab", page_icon="📦", layout="wide")

# --------------------------------------------------------------------------------------
# Styling helpers
# --------------------------------------------------------------------------------------
TIER_COLORS = {"Customer": "#1f77b4", "Retailer": "#ff7f0e",
               "Distributor": "#2ca02c", "Manufacturer": "#d62728"}
BASE_COLOR, AI_COLOR = "#8c8c8c", "#6f42c1"
WARM_HISTORY = 104  # weeks of pre-simulation history used to train / initialise


def show(fig: go.Figure, height: int = 380, title: str | None = None):
    # The title is rendered by Streamlit above the chart (not inside Plotly) so it can never
    # collide with a legend that wraps onto several rows on narrow screens. The legend sits
    # above the plot area and Plotly auto-expands the top margin to fit however many rows it needs.
    if title:
        st.markdown(f"**{title}**")
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
                      hovermode="x unified" if fig.data and fig.data[0].type == "scatter" else None)
    # Plotly's floating toolbar would cover the legend on hover; drag-to-zoom, double-click
    # reset and tooltips still work without it.
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False, "displayModeBar": False})


def table(df: pd.DataFrame, fmt: dict | None = None):
    st.dataframe(df.style.format(fmt or {}, precision=2), width="stretch", hide_index=True)


# --------------------------------------------------------------------------------------
# Cached computations
# --------------------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def customer_demand(n_total: int, mean: float, pattern: str, seed: int) -> pd.DataFrame:
    return sd.generate_customer_demand(n_total, mean, pattern, seed, step_week=WARM_HISTORY + 10)


@st.cache_data(show_spinner="Training AI demand forecaster...")
def ai_forecast(full: pd.DataFrame, model_name: str, seed: int):
    return bw.ai_customer_forecast(full, WARM_HISTORY, model_name, seed)


@st.cache_data(show_spinner="Running walk-forward backtest (model retrains as data arrives)...")
def backtest(df: pd.DataFrame, model_name: str, n_test: int, refit_every: int, seed: int):
    return fc.walk_forward(df, model_name, n_test, refit_every, seed)


@st.cache_data(show_spinner="Training on all data and forecasting ahead...")
def future_forecast(df: pd.DataFrame, model_name: str, horizon: int, promo_weeks: tuple, seed: int):
    model = fc.build_model(model_name, seed).fit(df)
    fut = sd.future_exogenous(df["date"].iloc[-1], horizon, set(promo_weeks))
    fut["forecast"] = model.forecast(df, fut)
    explain = model.explain() if hasattr(model, "explain") else None
    return fut, explain


@st.cache_data(show_spinner=False)
def routing_run(policy: str, supply: tuple, penalty: float, cv: float, weeks: int, seed: int):
    net = rt.Network(weekly_supply=np.array(supply, dtype=float))
    return rt.simulate_routing(net, policy, weeks, penalty, demand_cv=cv, seed=seed)


# --------------------------------------------------------------------------------------
# Sidebar - supply-chain environment (shared by the bullwhip tabs)
# --------------------------------------------------------------------------------------
with st.sidebar:
    st.title("📦 AI Supply Chain Lab")
    st.caption("AI in Operations · AI Applications in Reorder Point & Inventory Management")
    st.header("⚙️ Simulation environment")
    st.caption("Used by the **Bullwhip** and **AI-Redesigned Network** tabs.")
    weeks = st.slider("Simulation horizon (weeks)", 20, 104, 52, 2)
    pattern = st.selectbox("Customer demand pattern", list(sd.DEMAND_PATTERNS),
                           format_func=sd.DEMAND_PATTERNS.get)
    mean_demand = st.number_input("Mean customer demand (units/week)", 5.0, 500.0, 20.0, 5.0)
    seed = st.number_input("Random seed", 0, 9999, 7, 1)
    with st.expander("Stages: lead times, inventory & costs", expanded=False):
        manual_inv = st.toggle("Set starting inventory manually",
                               help="Off = each stage starts in steady state (cycle stock + safety stock).")
        lead_times, h_costs, b_costs, init_inv = [], [], [], []
        for i, (stage, lt0, h0, b0) in enumerate(zip(bw.STAGES, (2, 2, 3), (1.0, 0.75, 0.5), (5.0, 2.5, 1.5))):
            st.markdown(f"**{stage}**")
            c1, c2 = st.columns(2)
            lead_times.append(c1.number_input("Lead time (wk)", 1, 8, lt0, key=f"lt{i}"))
            init_inv.append(c2.number_input("Starting inventory", 0.0, 1e6, float(round(mean_demand * 2)),
                                            float(max(1, round(mean_demand / 4))), key=f"inv{i}",
                                            disabled=not manual_inv))
            c3, c4 = st.columns(2)
            h_costs.append(c3.number_input("Hold $/unit·wk", 0.0, 10.0, h0, 0.25, key=f"h{i}"))
            b_costs.append(c4.number_input("Backlog $/unit·wk", 0.0, 50.0, b0, 0.5, key=f"b{i}"))
    with st.expander("Baseline ordering policy (no AI)", expanded=False):
        base_method = st.radio("Forecast of incoming orders", ["moving_average", "exp_smoothing"],
                               format_func={"moving_average": "Moving average",
                                            "exp_smoothing": "Exponential smoothing"}.get)
        base_window = st.slider("Moving-average window (weeks)", 2, 12, 4)
        base_es = st.slider("Smoothing constant", 0.05, 0.95, 0.4, 0.05)
        base_z = st.slider("Safety factor z", 0.0, 3.0, 1.65, 0.05, key="base_z")
    st.divider()
    st.caption("Built with Streamlit · scikit-learn · PyTorch · SciPy · Plotly")

cfg = bw.ChainConfig(lead_times=tuple(lead_times), holding_costs=tuple(h_costs),
                     backlog_costs=tuple(b_costs),
                     initial_inventory=tuple(init_inv) if manual_inv else None)
full_demand = customer_demand(WARM_HISTORY + weeks + 1, mean_demand, pattern, int(seed))
history = full_demand["demand"].to_numpy()[:WARM_HISTORY]
demand = full_demand["demand"].to_numpy()[WARM_HISTORY:WARM_HISTORY + weeks]
base_policy = bw.OrderingPolicy(forecast_method=base_method, ma_window=base_window,
                                es_alpha=base_es, safety_factor=base_z)
baseline = bw.simulate_supply_chain(demand, history, cfg, base_policy)

# Apply tuned AI-policy parameters (set by the Bayesian optimiser) before widgets render.
st.session_state.setdefault("ai_alpha", 0.5)
st.session_state.setdefault("ai_z", 1.65)
if "pending_tune" in st.session_state:
    st.session_state["ai_alpha"], st.session_state["ai_z"] = st.session_state.pop("pending_tune")

# --------------------------------------------------------------------------------------
# Tabs
# --------------------------------------------------------------------------------------
st.title("AI in Operations — Supply Chain Lab")
tabs = st.tabs(["🏠 Overview", "🌊 Bullwhip Simulation", "🤖 AI-Redesigned Network",
                "📈 Demand Forecasting", "🛡️ Dynamic Safety Stock", "🎯 Bayesian ROP Optimisation"])

# ======================================================================================
# 0. Overview
# ======================================================================================
with tabs[0]:
    c1, c2 = st.columns([1.15, 1])
    with c1:
        st.subheader("What this dashboard covers")
        st.markdown(
            """
**Lab activity — mini-simulation of the bullwhip effect, redesigned with AI levers**

| Step from the brief | Where |
|---|---|
| Define the simulation environment (stages, lead times, order quantities, inventory) | Sidebar |
| Base simulation **without AI** — demand, ordering, inventory at each stage | 🌊 Bullwhip Simulation |
| AI **forecasting** used by the retailer to order more accurately | 🤖 AI-Redesigned Network |
| AI **routing / optimisation** of orders and inventory allocation | 🤖 AI-Redesigned Network |

**Exercise — AI-enhanced reorder-point calculator**

| Step from the brief | Where |
|---|---|
| ML demand forecasting (time series, regression, LSTM) with seasonality, promotions, external factors; add new data; continuously update | 📈 Demand Forecasting |
| Dynamic safety stock — probabilistic model **and** Monte Carlo risk scenarios | 🛡️ Dynamic Safety Stock |
| Sensitivity analysis via **Bayesian optimisation** of the reorder point under constraints | 🎯 Bayesian ROP Optimisation |
"""
        )
        st.info("**Bullwhip effect:** small fluctuations in customer demand cause increasingly "
                "larger swings in orders and inventory as you move upstream — retailer → "
                "distributor → manufacturer — leading to over-stocking, stock-outs and inflated costs.")
    with c2:
        st.subheader("Network")
        st.graphviz_chart(
            """
digraph {
  rankdir=LR; bgcolor="transparent";
  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, color="#555555"];
  C [label="Customer", fillcolor="#cfe3f5"];
  R [label="Retailer", fillcolor="#ffe0c2"];
  D [label="Distributor", fillcolor="#d3f0d3"];
  M [label="Manufacturer", fillcolor="#f7cccc"];
  S [label="Raw supply", fillcolor="#eeeeee"];
  C -> R [label=" demand", fontsize=9];
  R -> D [label=" orders", fontsize=9];
  D -> M [label=" orders", fontsize=9];
  M -> S [label=" production", fontsize=9];
  AI [label="AI levers\\n• ML forecasting\\n• POS info sharing\\n• order smoothing (BO)\\n• LP routing", shape=note, fillcolor="#e6dcf7"];
  AI -> R [style=dashed, color="#6f42c1"];
  AI -> D [style=dashed, color="#6f42c1"];
  AI -> M [style=dashed, color="#6f42c1"];
}
"""
        )
        st.subheader("Key formulas")
        st.latex(r"\text{Safety Stock} = Z\cdot\sqrt{\sigma_D^2\,L + D^2\,\sigma_L^2}")
        st.latex(r"\text{ROP} = D\cdot L + \text{Safety Stock}")
        st.latex(r"\text{Bullwhip ratio}_i = \frac{\operatorname{Var}(\text{orders}_i)}{\operatorname{Var}(\text{customer demand})}")
    st.caption("Tip: every chart is interactive — hover, zoom, and click legend items to toggle series.")

# ======================================================================================
# 1. Bullwhip simulation (no AI)
# ======================================================================================
with tabs[1]:
    st.subheader("A · Classic bullwhip model (lab slide)")
    st.caption("Each tier copies the downstream signal with a delay and over-orders by an amplification "
               "factor (default: 1-week delay, 20 % amplification). Customer demand is Poisson.")
    c1, c2, c3 = st.columns([1, 1, 3])
    delay = c1.slider("Delay (weeks)", 0, 4, 1)
    amp = c2.slider("Amplification", 1.0, 2.0, 1.2, 0.05)
    tiers = bw.classic_bullwhip(demand, delay, amp)
    fig = go.Figure()
    for lab, series in zip(bw.TIER_LABELS, tiers):
        fig.add_trace(go.Scatter(y=series, name=lab, mode="lines+markers", marker_size=4,
                                 line=dict(color=TIER_COLORS[lab], width=2)))
    fig.update_xaxes(title="Week")
    fig.update_yaxes(title="Demand / orders (units)")
    show(fig, 380, "Bullwhip Effect in Supply Chain")
    with c3:
        st.markdown(f"Peak manufacturer signal: **{tiers[-1].max():.0f}** vs peak customer demand "
                    f"**{tiers[0].max():.0f}** (×{amp ** 3:.2f} cumulative amplification).")

    st.divider()
    st.subheader("B · Inventory simulation — baseline network without AI")
    st.caption("Every stage holds inventory, carries backlog, waits for its replenishment lead time, "
               "forecasts **the orders it receives** and places order-up-to orders. Nobody sees "
               "end-customer demand except the retailer — the textbook recipe for bullwhip.")
    ratios = baseline.bullwhip_ratios()
    m = st.columns(4)
    m[0].metric("Manufacturer bullwhip ratio", f"{ratios[-1]:.1f}×",
                help="Variance of manufacturer orders ÷ variance of customer demand")
    m[1].metric("Peak manufacturer order", f"{baseline.orders[-1].max():.0f}",
                f"vs peak demand {demand.max():.0f}", delta_color="off")
    m[2].metric("Retailer fill rate", f"{baseline.fill_rate:.1%}")
    m[3].metric("Total supply-chain cost", f"${baseline.total_cost:,.0f}")

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure(go.Scatter(y=demand, name="Customer", line=dict(color=TIER_COLORS["Customer"], width=3)))
        for i, stg in enumerate(bw.STAGES):
            fig.add_trace(go.Scatter(y=baseline.orders[i], name=f"{stg} orders",
                                     line=dict(color=TIER_COLORS[stg])))
        fig.update_xaxes(title="Week")
        fig.update_yaxes(title="Units")
        show(fig, 380, "Orders placed at each stage")
    with c2:
        fig = go.Figure()
        for i, stg in enumerate(bw.STAGES):
            fig.add_trace(go.Scatter(y=baseline.inventory[i] - baseline.backlog[i], name=stg,
                                     line=dict(color=TIER_COLORS[stg])))
        fig.add_hline(y=0, line_dash="dot", line_color="black")
        fig.update_xaxes(title="Week")
        fig.update_yaxes(title="Net inventory (on-hand − backlog)")
        show(fig, 380, "Net inventory — below zero means stock-out / backlog")

    c1, c2 = st.columns([1, 1.4])
    with c1:
        fig = go.Figure(go.Bar(x=list(bw.STAGES), y=ratios, marker_color=[TIER_COLORS[s] for s in bw.STAGES],
                               text=[f"{r:.1f}×" for r in ratios], textposition="outside"))
        fig.add_hline(y=1, line_dash="dash", annotation_text="no amplification")
        fig.update_yaxes(title="Var(orders) / Var(demand)")
        show(fig, 340, "Variance amplification by stage")
    with c2:
        st.markdown("**Stage summary**")
        table(baseline.summary(), {"Total cost": "${:,.0f}"})
        st.markdown(
            "**Why it happens here:** demand-signal processing (each stage re-forecasts a noisy order "
            "stream), lead-time-driven over-reaction of order-up-to policies, safety stocks sized on "
            "amplified variance, and no visibility of true customer demand.")

# ======================================================================================
# 2. AI-redesigned network
# ======================================================================================
with tabs[2]:
    st.subheader("Redesign the network with AI levers")
    lc = st.columns([1.1, 1.1, 1, 1])
    use_ai = lc[0].toggle("AI demand forecasting (retailer)", True)
    ai_model = lc[0].selectbox("Forecasting model", list(bw.AI_MODELS), disabled=not use_ai)
    share = lc[1].toggle("Information sharing (POS data upstream)", True,
                         help="Every stage forecasts on end-customer demand instead of the orders it receives.")
    alpha = lc[2].slider("Order smoothing α", 0.05, 1.0, step=0.05, key="ai_alpha",
                         help="Fraction of the inventory-position gap closed each week (1 = classic order-up-to).")
    z_ai = lc[3].slider("Safety factor z (AI)", 0.0, 3.0, step=0.05, key="ai_z")

    fc_arr, fc_sigma = ai_forecast(full_demand, ai_model, int(seed)) if use_ai else (None, None)
    ai_policy = bw.OrderingPolicy(forecast_method=base_method, ma_window=base_window, es_alpha=base_es,
                                  safety_factor=z_ai, smoothing=alpha, info_sharing=share,
                                  ai_forecast=fc_arr, ai_sigma=fc_sigma)
    ai_res = bw.simulate_supply_chain(demand, history, cfg, ai_policy)

    with st.expander("🎯 Auto-tune α and z with Bayesian optimisation", expanded=False):
        t1, t2, t3 = st.columns([1, 1, 1])
        fill_target = t1.slider("Min retailer fill rate", 0.80, 1.0, 0.98, 0.01)
        bo_iters = t2.slider("BO iterations", 5, 40, 20)
        t3.write("")
        if t3.button("Run Bayesian optimisation", type="primary"):
            # Robust tuning: evaluate each candidate on the current demand path AND two extra
            # sample paths of the same pattern, so the policy is not over-fitted to one path.
            with st.spinner("Preparing demand scenarios..."):
                scenarios = []
                for k_s, s_seed in enumerate((int(seed), int(seed) + 101, int(seed) + 202)):
                    fd = full_demand if k_s == 0 else customer_demand(WARM_HISTORY + weeks + 1, mean_demand,
                                                                      pattern, s_seed)
                    f_arr, f_sig = ai_forecast(fd, ai_model, s_seed) if use_ai else (None, None)
                    dv = fd["demand"].to_numpy()
                    scenarios.append((dv[WARM_HISTORY:WARM_HISTORY + weeks], dv[:WARM_HISTORY], f_arr, f_sig))

            def chain_objective(x):
                costs, fills = [], []
                for d_s, h_s, f_arr, f_sig in scenarios:
                    pol = bw.OrderingPolicy(**{**ai_policy.__dict__, "smoothing": float(x[0]),
                                               "safety_factor": float(x[1]), "ai_forecast": f_arr,
                                               "ai_sigma": f_sig})
                    r = bw.simulate_supply_chain(d_s, h_s, cfg, pol)
                    costs.append(r.total_cost)
                    fills.append(r.fill_rate)
                worst = min(fills)
                return float(np.mean(costs)) * (1 + 20 * max(0.0, fill_target - worst)), {"fill": worst}
            with st.spinner("Optimising ordering policy..."):
                res = bayesian_optimize(chain_objective, [(0.05, 1.0), (0.0, 3.0)], n_init=6,
                                        n_iter=bo_iters, seed=int(seed), x0=[[alpha, z_ai]])
            snap = lambda v, lo, hi: float(np.clip(round(round(float(v) / 0.05) * 0.05, 2), lo, hi))
            st.session_state["pending_tune"] = (snap(res.best_x[0], 0.05, 1.0), snap(res.best_x[1], 0.0, 3.0))
            st.session_state["tune_msg"] = (f"Tuned α = {res.best_x[0]:.2f}, z = {res.best_x[1]:.2f} "
                                            f"after {len(res.y)} evaluations × 3 demand scenarios "
                                            f"(worst-case fill rate {res.info[res.best_index]['fill']:.1%}).")
            st.rerun()
        if "tune_msg" in st.session_state:
            st.success(st.session_state["tune_msg"] + " Values applied to the sliders above (rounded to 0.05).")
        st.caption("Objective: mean total chain cost over 3 demand scenarios, penalised when the worst-case retailer fill rate "
                   "falls below the target. Gaussian-process surrogate + Expected Improvement.")

    b_r, a_r = baseline.bullwhip_ratios(), ai_res.bullwhip_ratios()
    m = st.columns(4)
    m[0].metric("Manufacturer bullwhip ratio", f"{a_r[-1]:.2f}×", f"{a_r[-1] - b_r[-1]:+.1f} vs baseline",
                delta_color="inverse")
    m[1].metric("Total supply-chain cost", f"${ai_res.total_cost:,.0f}",
                f"{(ai_res.total_cost / baseline.total_cost - 1):+.1%} vs baseline", delta_color="inverse")
    m[2].metric("Retailer fill rate", f"{ai_res.fill_rate:.1%}", f"{(ai_res.fill_rate - baseline.fill_rate) * 100:+.1f} pp")
    m[3].metric("Peak manufacturer order", f"{ai_res.orders[-1].max():.0f}",
                f"{ai_res.orders[-1].max() - baseline.orders[-1].max():+.0f}", delta_color="inverse")

    c1, c2 = st.columns(2)
    ymax = max(baseline.orders.max(), ai_res.orders.max(), demand.max()) * 1.05
    for col, res_, ttl in ((c1, baseline, "Baseline (no AI)"), (c2, ai_res, "AI-redesigned")):
        with col:
            fig = go.Figure(go.Scatter(y=demand, name="Customer", line=dict(color=TIER_COLORS["Customer"], width=3)))
            for i, stg in enumerate(bw.STAGES):
                fig.add_trace(go.Scatter(y=res_.orders[i], name=stg, line=dict(color=TIER_COLORS[stg])))
            fig.update_yaxes(range=[0, ymax], title="Orders (units)")
            fig.update_xaxes(title="Week")
            show(fig, 360, f"Orders — {ttl}")

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure([go.Bar(name="Baseline", x=list(bw.STAGES), y=b_r, marker_color=BASE_COLOR,
                                text=[f"{v:.1f}" for v in b_r], textposition="outside"),
                         go.Bar(name="AI", x=list(bw.STAGES), y=a_r, marker_color=AI_COLOR,
                                text=[f"{v:.2f}" for v in a_r], textposition="outside")])
        fig.add_hline(y=1, line_dash="dash")
        fig.update_layout(barmode="group")
        fig.update_yaxes(title="Bullwhip ratio", type="log")
        show(fig, 360, "Bullwhip ratio by stage (log scale)")
    with c2:
        fig = go.Figure(go.Scatter(y=demand, name="Actual customer demand",
                                   line=dict(color=TIER_COLORS["Customer"], width=2)))
        # forecasts made at week t refer to week t+1 -> plot them one week ahead
        fig.add_trace(go.Scatter(x=np.arange(1, weeks + 1), y=baseline.forecasts[0], name="Baseline forecast",
                                 line=dict(color=BASE_COLOR, dash="dash")))
        if fc_arr is not None:
            fig.add_trace(go.Scatter(x=np.arange(1, weeks + 1), y=fc_arr[:weeks], name=f"AI forecast ({ai_model})",
                                     line=dict(color=AI_COLOR, width=2)))
        fig.update_xaxes(title="Week")
        fig.update_yaxes(title="Units")
        show(fig, 360, "Retailer's view of demand")

    # ---- lever ladder --------------------------------------------------------------------
    st.markdown("#### Contribution of each AI lever")
    st.caption(f"Levers are added one at a time (model: {ai_model}, α = {alpha:.2f}, z = {z_ai:.2f}).")
    l_fc, l_sig = ai_forecast(full_demand, ai_model, int(seed))
    ladder = [("Baseline", base_policy)]
    common = dict(forecast_method=base_method, ma_window=base_window, es_alpha=base_es, safety_factor=z_ai)
    ladder.append(("+ AI forecasting", bw.OrderingPolicy(**common, ai_forecast=l_fc, ai_sigma=l_sig)))
    ladder.append(("+ Info sharing", bw.OrderingPolicy(**common, ai_forecast=l_fc, ai_sigma=l_sig, info_sharing=True)))
    ladder.append(("+ Order smoothing", bw.OrderingPolicy(**common, ai_forecast=l_fc, ai_sigma=l_sig,
                                                          info_sharing=True, smoothing=alpha)))
    rows = []
    for name, pol in ladder:
        r = bw.simulate_supply_chain(demand, history, cfg, pol)
        rows.append({"Scenario": name, "Manufacturer bullwhip": r.bullwhip_ratios()[-1],
                     "Total cost": r.total_cost, "Fill rate": r.fill_rate})
    lad = pd.DataFrame(rows)
    fig = make_subplots(rows=1, cols=3, subplot_titles=("Manufacturer bullwhip ratio", "Total cost ($)", "Retailer fill rate"))
    colors = [BASE_COLOR] + [AI_COLOR] * 3
    for j, col in enumerate(["Manufacturer bullwhip", "Total cost", "Fill rate"]):
        fig.add_trace(go.Bar(x=lad["Scenario"], y=lad[col], marker_color=colors, showlegend=False,
                             text=[f"{v:.1%}" if col == "Fill rate" else (f"{v:,.0f}" if col == "Total cost" else f"{v:.2f}")
                                   for v in lad[col]], textposition="outside"), 1, j + 1)
    fig.update_yaxes(range=[0, 1.1], row=1, col=3)
    show(fig, 360)

    # ---- routing / allocation --------------------------------------------------------------
    st.divider()
    st.subheader("AI-based order routing & inventory allocation")
    st.caption("Two DCs supply four retailers. Inbound supply is regionally unbalanced. The baseline "
               "routes each retailer to its nearest DC; the optimised policy solves a min-cost "
               "transportation **linear program** every week (shipping cost vs. lost-sale penalty).")
    rc = st.columns(4)
    sup_n = rc[0].number_input("DC North weekly supply", 0, 1000, 240, 10)
    sup_s = rc[1].number_input("DC South weekly supply", 0, 1000, 160, 10)
    penalty = rc[2].number_input("Lost-sale penalty ($/unit)", 0.0, 100.0, 10.0, 1.0)
    cv = rc[3].slider("Retailer demand variability (CV)", 0.0, 0.6, 0.2, 0.05)
    r_base = routing_run("nearest", (sup_n, sup_s), penalty, cv, 26, int(seed))
    r_lp = routing_run("lp", (sup_n, sup_s), penalty, cv, 26, int(seed))
    m = st.columns(4)
    tb, tl = r_base["totals"], r_lp["totals"]
    m[0].metric("Fill rate (optimised)", f"{tl['fill_rate']:.1%}", f"{(tl['fill_rate'] - tb['fill_rate']) * 100:+.1f} pp")
    m[1].metric("Total cost (optimised)", f"${tl['total_cost']:,.0f}",
                f"{tl['total_cost'] / max(tb['total_cost'], 1e-9) - 1:+.1%}", delta_color="inverse")
    m[2].metric("Shipping cost", f"${tl['shipping_cost']:,.0f}",
                f"{tl['shipping_cost'] - tb['shipping_cost']:+,.0f}", delta_color="inverse")
    m[3].metric("Lost-sales cost", f"${tl['lost_sales_cost']:,.0f}",
                f"{tl['lost_sales_cost'] - tb['lost_sales_cost']:+,.0f}", delta_color="inverse")

    net = rt.Network()

    def sankey(flows, title):
        nd = len(net.dcs)
        src, tgt, val, col = [], [], [], []
        for i in range(nd):
            for j in range(len(net.retailers)):
                if flows[i, j] > 0.5:
                    src.append(i); tgt.append(nd + j); val.append(flows[i, j])
                    col.append("rgba(31,119,180,0.45)" if i == 0 else "rgba(214,39,40,0.45)")
        fig = go.Figure(go.Sankey(node=dict(label=net.dcs + net.retailers, pad=18,
                                            color=["#1f77b4", "#d62728"] + ["#999999"] * len(net.retailers)),
                                  link=dict(source=src, target=tgt, value=val, color=col)))
        show(fig, 300, title)

    c1, c2, c3 = st.columns([1, 1, 1.1])
    with c1:
        sankey(r_base["flows"], "Nearest-DC routing")
    with c2:
        sankey(r_lp["flows"], "LP-optimised routing")
    with c3:
        fig = go.Figure([go.Scatter(x=r_base["weekly"]["week"], y=r_base["weekly"]["fill_rate"], name="Nearest DC",
                                    line=dict(color=BASE_COLOR)),
                         go.Scatter(x=r_lp["weekly"]["week"], y=r_lp["weekly"]["fill_rate"], name="LP-optimised",
                                    line=dict(color=AI_COLOR, width=3))])
        fig.update_yaxes(tickformat=".0%", title="Weekly fill rate")
        fig.update_xaxes(title="Week")
        show(fig, 300, "Service over time")
    per = r_base["per_retailer"][["retailer", "demand"]].copy()
    per["Fill rate — nearest DC"] = r_base["per_retailer"]["fill_rate"]
    per["Fill rate — LP"] = r_lp["per_retailer"]["fill_rate"]
    table(per.rename(columns={"retailer": "Retailer", "demand": "Total demand"}),
          {"Fill rate — nearest DC": "{:.1%}", "Fill rate — LP": "{:.1%}", "Total demand": "{:,.0f}"})
    st.caption("Extension idea: replace the weekly LP with a reinforcement-learning agent that also "
               "decides DC replenishment, learning from the same simulator.")

# ======================================================================================
# 3. Demand forecasting
# ======================================================================================
with tabs[3]:
    st.subheader("ML demand forecasting — continuously updated")
    st.session_state.setdefault("demand_df", sd.generate_demand_history(156, sd.DemandParams(), 42))
    st.session_state.setdefault("gen_params", sd.DemandParams())
    st.session_state.setdefault("update_log", [])

    dc1, dc2 = st.columns([1, 2.2])
    with dc1:
        with st.container(border=True):
            st.markdown("**① Data**")
            source = st.radio("Source", ["Synthetic history", "Upload CSV"], horizontal=True)
            if source == "Upload CSV":
                up = st.file_uploader("CSV with columns: date, demand [, promotion, holiday, temperature]",
                                      type="csv")
                if up is not None and st.session_state.get("upload_id") != (up.name, up.size):
                    try:
                        st.session_state["demand_df"] = sd.validate_demand_frame(pd.read_csv(up))
                        st.session_state["upload_id"] = (up.name, up.size)
                        lvl = float(st.session_state["demand_df"]["demand"].tail(52).mean())
                        st.session_state["gen_params"] = sd.DemandParams(base_level=lvl, trend_per_week=0.0)
                        st.session_state["update_log"] = []
                        st.success(f"Loaded {len(st.session_state['demand_df'])} weeks.")
                    except Exception as e:  # show validation problems to the user
                        st.error(f"Could not load file: {e}")
            else:
                with st.expander("Generator settings"):
                    g1, g2 = st.columns(2)
                    gp = sd.DemandParams(
                        base_level=g1.number_input("Base level", 10.0, 5000.0, 200.0, 10.0),
                        trend_per_week=g2.number_input("Trend / week", -5.0, 5.0, 0.5, 0.1),
                        seasonal_amplitude=g1.slider("Seasonality", 0.0, 0.6, 0.25, 0.05),
                        promo_probability=g2.slider("Promo frequency", 0.0, 0.4, 0.12, 0.02),
                        promo_lift=g1.slider("Promo lift", 0.0, 1.0, 0.35, 0.05),
                        temp_sensitivity=g2.slider("Weather sensitivity", 0.0, 0.08, 0.03, 0.01),
                        noise_cv=g1.slider("Noise (CV)", 0.0, 0.3, 0.08, 0.01),
                    )
                    n_hist = g2.number_input("Weeks of history", 60, 520, 156, 4)
                    if st.button("Regenerate history"):
                        st.session_state["demand_df"] = sd.generate_demand_history(int(n_hist), gp, 42)
                        st.session_state["gen_params"] = gp
                        st.session_state["update_log"] = []
                        st.session_state.pop("upload_id", None)

            df = st.session_state["demand_df"]
            st.caption(f"**{len(df)} weeks** · {df['date'].min():%d %b %Y} → {df['date'].max():%d %b %Y}")

        with st.container(border=True):
            st.markdown("**② New data arrives**")
            with st.form("add_obs", clear_on_submit=False):
                last = df["date"].iloc[-1]
                f1, f2 = st.columns(2)
                new_date = f1.date_input("Week", (last + pd.DateOffset(weeks=1)).date())
                new_dem = f2.number_input("Actual demand", 0.0, 1e6, float(round(df["demand"].tail(4).mean())))
                f3, f4 = st.columns(2)
                new_promo = f3.checkbox("Promotion")
                new_hol = f4.checkbox("Holiday")
                new_temp = st.number_input("Temperature °C", -30.0, 50.0,
                                           float(np.round(sd.climatology(pd.DatetimeIndex([pd.Timestamp(new_date)]))[0], 1)))
                if st.form_submit_button("➕ Add observation", type="primary"):
                    row = pd.DataFrame([{"date": pd.Timestamp(new_date), "demand": new_dem,
                                         "promotion": int(new_promo), "holiday": int(new_hol),
                                         "temperature": new_temp}])
                    st.session_state["demand_df"] = sd.validate_demand_frame(pd.concat([df, row]))
                    st.rerun()
            a1, a2 = st.columns([1, 1])
            n_new = a1.number_input("Weeks to stream", 1, 26, 4)
            if a2.button("⏩ Simulate arrival", help="Append the next weeks from the demand process"):
                new = sd.simulate_new_weeks(df, int(n_new), st.session_state["gen_params"], seed=42)
                st.session_state["demand_df"] = pd.concat([df, new], ignore_index=True)
                st.rerun()
            d1, d2 = st.columns(2)
            d1.download_button("⬇️ Download data", df.to_csv(index=False).encode(), "demand_history.csv",
                               "text/csv", width="stretch")
            if d2.button("↺ Reset data", width="stretch"):
                for k in ("demand_df", "gen_params", "update_log", "upload_id"):
                    st.session_state.pop(k, None)
                st.rerun()

        with st.container(border=True):
            st.markdown("**③ Model**")
            model_name = st.selectbox("Forecasting model", fc.available_models(),
                                      index=fc.available_models().index("Ridge regression"))
            st.caption(fc.MODEL_INFO[model_name])
            m1, m2 = st.columns(2)
            n_test = m1.slider("Backtest weeks", 8, 52, 26)
            refit_every = m2.slider("Retrain every N weeks", 1, 13, 4)
            horizon = m1.slider("Forecast horizon (weeks)", 4, 26, 12)
            promo_plan = m2.multiselect("Planned promo weeks", list(range(1, horizon + 1)), default=[3])

    df = st.session_state["demand_df"]
    wf = backtest(df, model_name, n_test, refit_every, 0)
    naive = backtest(df, "Naive (last week)", n_test, refit_every, 0)
    met, met_naive = fc.forecast_metrics(wf["actual"], wf["forecast"]), fc.forecast_metrics(naive["actual"], naive["forecast"])
    sigma_week = float(wf["error"].std(ddof=1))
    fut, explain = future_forecast(df, model_name, horizon, tuple(sorted(promo_plan)), 0)

    log = st.session_state["update_log"]
    key = (len(df), model_name)
    if not log or (log[-1]["Observations"], log[-1]["Model"]) != key:
        log.append({"Observations": len(df), "Last week": df["date"].iloc[-1].date(), "Model": model_name,
                    "WAPE %": met["WAPE %"], "MAPE %": met["MAPE %"], "Error σ": sigma_week})

    with dc2:
        k = st.columns(4)
        k[0].metric("WAPE", f"{met['WAPE %']:.1f}%", f"{met['WAPE %'] - met_naive['WAPE %']:+.1f} pp vs naive",
                    delta_color="inverse")
        k[1].metric("MAPE", f"{met['MAPE %']:.1f}%", f"{met['MAPE %'] - met_naive['MAPE %']:+.1f} pp vs naive",
                    delta_color="inverse")
        k[2].metric("RMSE", f"{met['RMSE']:.1f}", f"{met['RMSE'] - met_naive['RMSE']:+.1f}", delta_color="inverse")
        k[3].metric("Bias", f"{met['Bias %']:+.1f}%", help="Positive = over-forecasting")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["date"], y=df["demand"], name="Actual demand",
                                 line=dict(color=TIER_COLORS["Customer"], width=2)))
        promo_hist = df[df["promotion"] == 1]
        fig.add_trace(go.Scatter(x=promo_hist["date"], y=promo_hist["demand"], mode="markers", name="Promotion week",
                                 marker=dict(symbol="star", size=9, color="#ff7f0e")))
        fig.add_trace(go.Scatter(x=wf["date"], y=wf["forecast"], name="Walk-forward forecast (backtest)",
                                 line=dict(color=AI_COLOR, dash="dot", width=2)))
        rt_pts = wf[wf["retrained"]]
        fig.add_trace(go.Scatter(x=rt_pts["date"], y=rt_pts["forecast"], mode="markers", name="Model retrained",
                                 marker=dict(symbol="diamond", size=8, color=AI_COLOR)))
        band = 1.2816 * sigma_week
        fig.add_trace(go.Scatter(x=list(fut["date"]) + list(fut["date"][::-1]),
                                 y=list(fut["forecast"] + band) + list((fut["forecast"] - band).clip(lower=0)[::-1]),
                                 fill="toself", fillcolor="rgba(111,66,193,0.15)", line=dict(width=0),
                                 name="80% interval", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=fut["date"], y=fut["forecast"], name="Future forecast",
                                 line=dict(color=AI_COLOR, width=3)))
        fp = fut[fut["promotion"] == 1]
        fig.add_trace(go.Scatter(x=fp["date"], y=fp["forecast"], mode="markers", name="Planned promo",
                                 marker=dict(symbol="star", size=12, color="#ff7f0e", line=dict(width=1, color="black"))))
        fig.add_vline(x=df["date"].iloc[-1], line_dash="dash", line_color="gray")
        fig.update_xaxes(range=[df["date"].iloc[max(0, len(df) - 104)], fut["date"].iloc[-1]])
        fig.update_yaxes(title="Units / week")
        show(fig, 430, f"Demand, walk-forward backtest and {horizon}-week forecast — {model_name}")
        st.caption("Walk-forward: each week the model forecasts the next week using only data available "
                   f"at that time, and is retrained every {refit_every} week(s) as new data arrives. "
                   "The 80% band uses the backtest error σ.")

        e1, e2 = st.columns(2)
        with e1:
            if explain is not None:
                fig = go.Figure(go.Bar(x=explain["importance"], y=explain["feature"], orientation="h",
                                       marker_color=AI_COLOR))
                show(fig, 330, "What drives the forecast " + ("(|z|-statistic of coefficients)"
                                                              if model_name.startswith("SARIMAX")
                                                              else "(feature importance)"))
            else:
                st.info("Driver importance is available for SARIMAX, Ridge, Random Forest and Gradient Boosting.")
        with e2:
            lg = pd.DataFrame(log)
            if len(lg) > 1:
                fig = go.Figure(go.Scatter(x=lg["Observations"], y=lg["WAPE %"], mode="lines+markers",
                                           text=lg["Model"], line=dict(color=AI_COLOR)))
                fig.update_xaxes(title="Observations available")
                fig.update_yaxes(title="Backtest WAPE %")
                show(fig, 330, "Accuracy as new data arrives")
            else:
                st.info("Add observations or click **Simulate arrival** — every update retrains the model "
                        "and the accuracy history appears here.")
        with st.expander("Model update log", expanded=False):
            table(pd.DataFrame(log), {"WAPE %": "{:.2f}", "MAPE %": "{:.2f}", "Error σ": "{:.1f}"})

        if st.toggle("🏁 Compare all models (leaderboard)"):
            rows = []
            for nm in fc.available_models():
                w = backtest(df, nm, n_test, refit_every, 0)
                rows.append({"Model": nm, **fc.forecast_metrics(w["actual"], w["forecast"])})
            lb = pd.DataFrame(rows).sort_values("WAPE %")
            table(lb, {c: "{:.2f}" for c in lb.columns if c != "Model"})

        with st.expander("Forecast table"):
            table(fut.assign(date=fut["date"].dt.date), {"forecast": "{:.1f}", "temperature": "{:.1f}"})

# ======================================================================================
# 4. Dynamic safety stock
# ======================================================================================
with tabs[4]:
    st.subheader("Dynamic safety stock — probabilistic model + Monte Carlo")
    st.latex(r"SS = Z\cdot\sqrt{\sigma_D^2\cdot L + D^2\cdot\sigma_L^2}\qquad ROP = D\cdot L + SS")
    p1, p2, p3 = st.columns(3)
    with p1:
        with st.container(border=True):
            st.markdown("**Demand inputs**")
            src = st.radio("Source", ["AI forecast (📈 tab)", "Manual"], horizontal=True)
            if src == "Manual":
                D = st.number_input("Mean daily demand D", 0.1, 1e5, 100.0)
                sigma_d = st.number_input("Daily demand std σD", 0.0, 1e5, 20.0)
            else:
                D = float(fut["forecast"].mean() / 7)
                sigma_d = sigma_week / np.sqrt(7)
                st.metric("D (units/day)", f"{D:.1f}", help=f"Mean of the {horizon}-week {model_name} forecast ÷ 7")
                st.metric("σD (units/day)", f"{sigma_d:.1f}", help="Std of walk-forward forecast errors ÷ √7")
                st.caption("σD is the **forecast-error** std — the uncertainty that remains after the model, "
                           "which is what safety stock must cover.")
    with p2:
        with st.container(border=True):
            st.markdown("**Lead time & service**")
            L = st.number_input("Mean lead time L (days)", 0.5, 120.0, 7.0, 0.5)
            sigma_l = st.number_input("Lead-time std σL (days)", 0.0, 60.0, 2.0, 0.5)
            sl = st.slider("Target cycle service level", 0.80, 0.995, 0.95, 0.005)
            unit_h = st.number_input("Holding cost ($/unit/year)", 0.0, 1000.0, 18.0, 1.0)
    with p3:
        with st.container(border=True):
            st.markdown("**Monte Carlo risk scenario**")
            n_sims = st.select_slider("Simulations", [2000, 5000, 10000, 20000, 50000, 100000], 20000)
            demand_dist = st.selectbox("Demand distribution", ["Normal", "Gamma (right-skewed)"])
            lt_dist = st.selectbox("Lead-time distribution", ["Normal", "Gamma (right-skewed)", "Lognormal (heavy tail)"])
            dis_p = st.slider("Supply-disruption probability", 0.0, 0.3, 0.05, 0.01)
            dis_d = st.slider("Extra delay if disrupted (days)", 0.0, 30.0, 5.0, 0.5)

    z = inv.z_value(sl)
    ss_f = float(inv.safety_stock(z, sigma_d, L, D, sigma_l))
    rop_f = float(inv.reorder_point(D, L, ss_f))
    mc = inv.monte_carlo_lead_time_demand(D, sigma_d, L, sigma_l, n_sims, demand_dist, lt_dist, dis_p, dis_d, int(seed))
    mc_r = inv.monte_carlo_safety_stock(mc["ltd"], D, L, sl)
    ach = inv.achieved_service_level(mc["ltd"], rop_f)

    k = st.columns(5)
    k[0].metric("Z", f"{z:.3f}")
    k[1].metric("Safety stock — formula", f"{ss_f:,.0f}")
    k[2].metric("Safety stock — Monte Carlo", f"{mc_r['ss']:,.0f}", f"{mc_r['ss'] - ss_f:+,.0f} vs formula",
                delta_color="off")
    k[3].metric("Reorder point (MC)", f"{mc_r['rop']:,.0f}", f"formula {rop_f:,.0f}", delta_color="off")
    k[4].metric("Service actually achieved by formula ROP", f"{ach:.1%}", f"{(ach - sl) * 100:+.1f} pp vs target")
    if ach < sl - 0.005:
        st.warning(f"Under this risk scenario the closed-form ROP only protects **{ach:.1%}** of cycles "
                   f"(target {sl:.1%}). Monte Carlo sizes the buffer at **{mc_r['ss']:,.0f}** units instead.")

    c1, c2 = st.columns([1.6, 1])
    with c1:
        fig = go.Figure(go.Histogram(x=mc["ltd"], nbinsx=80, marker_color="#9ecae1", name="Lead-time demand",
                                     histnorm="probability density"))
        fig.add_vline(x=D * L, line_dash="dot", annotation_text="Expected D·L", annotation_position="top left")
        fig.add_vline(x=rop_f, line_color=BASE_COLOR, line_width=3, annotation_text="ROP formula")
        fig.add_vline(x=mc_r["rop"], line_color=AI_COLOR, line_width=3, annotation_text="ROP Monte Carlo",
                      annotation_position="bottom right")
        fig.update_xaxes(title="Demand during lead time (units)")
        show(fig, 360, f"Monte Carlo: {n_sims:,} simulated replenishment cycles")
    with c2:
        fig = go.Figure(go.Histogram(x=mc["lead_times"], nbinsx=60, marker_color="#fdae6b"))
        fig.update_xaxes(title="Simulated lead time (days)")
        show(fig, 360, "Lead-time scenarios")

    # Dynamic SS over the forecast horizon
    if src != "Manual":
        st.markdown("#### Dynamic safety stock over the forecast horizon")
        hist_D = float(df["demand"].tail(52).mean() / 7)
        hist_sd = float(df["demand"].tail(52).std() / np.sqrt(7))
        dyn = fut[["date", "forecast", "promotion"]].copy()
        dyn["D"] = dyn["forecast"] / 7
        dyn["SS"] = inv.safety_stock(z, sigma_d, L, dyn["D"], sigma_l)
        dyn["ROP"] = dyn["D"] * L + dyn["SS"]
        static_ss = float(inv.safety_stock(z, hist_sd, L, hist_D, sigma_l))
        static_rop = hist_D * L + static_ss
        fig = go.Figure()
        fig.add_trace(go.Bar(x=dyn["date"], y=dyn["SS"], name="Dynamic safety stock", marker_color="rgba(111,66,193,0.35)"))
        fig.add_trace(go.Scatter(x=dyn["date"], y=dyn["ROP"], name="Dynamic ROP (AI forecast)", mode="lines+markers",
                                 line=dict(color=AI_COLOR, width=3)))
        fig.add_hline(y=static_rop, line_color=BASE_COLOR, line_dash="dash",
                      annotation_text=f"Static ROP (52-wk avg & raw std): {static_rop:,.0f}")
        fig.add_hline(y=static_ss, line_color=BASE_COLOR, line_dash="dot",
                      annotation_text=f"Static SS: {static_ss:,.0f}", annotation_position="bottom right")
        pr = dyn[dyn["promotion"] == 1]
        fig.add_trace(go.Scatter(x=pr["date"], y=pr["ROP"], mode="markers", name="Planned promo",
                                 marker=dict(symbol="star", size=14, color="#ff7f0e", line=dict(width=1, color="black"))))
        fig.update_yaxes(title="Units")
        show(fig, 380, "Reorder point re-computed every week from the forecast")
        st.caption("The static approach sizes buffers on raw historical variability (seasonality and promotions "
                   "look like noise). The dynamic approach moves the ROP with forecast demand and only buffers "
                   "the residual forecast error — typically **less stock and better timing**.")

    st.markdown("#### Sensitivity analysis")
    s1, s2 = st.columns(2)
    with s1:
        tr = inv.service_level_tradeoff(D, sigma_d, L, sigma_l, unit_h)
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=tr["service_level"], y=tr["safety_stock"], name="Safety stock",
                                 line=dict(color="#1f77b4", width=3)))
        fig.add_trace(go.Scatter(x=tr["service_level"], y=tr["holding_cost_per_year"], name="Holding cost $/yr",
                                 line=dict(color="#2ca02c", width=3, dash="dash")), secondary_y=True)
        fig.add_vline(x=sl, line_dash="dot")
        fig.update_xaxes(tickformat=".0%", title="Service level")
        fig.update_yaxes(title="Safety stock (units)")
        fig.update_yaxes(title="Holding cost ($/yr)", secondary_y=True)
        show(fig, 360, "Service-level trade-off (cost rises steeply above ~95%)")
    with s2:
        tor, base_rop = inv.tornado(D, sigma_d, L, sigma_l, sl, 0.2)
        fig = go.Figure([go.Bar(y=tor["driver"], x=tor["low"], orientation="h", name="−20%", marker_color="#6baed6"),
                         go.Bar(y=tor["driver"], x=tor["high"], orientation="h", name="+20%", marker_color="#fd8d3c")])
        fig.update_layout(barmode="overlay")
        fig.update_xaxes(title=f"Change in ROP (base {base_rop:,.0f} units)")
        show(fig, 360, "Tornado: ROP sensitivity to ±20% in each driver")

    with st.expander("ROP sensitivity curves (notebook analysis, now with dynamic safety stock)"):
        fig = make_subplots(rows=1, cols=3, subplot_titles=("Lead time", "Daily demand", "Lead-time variability"))
        lts = np.linspace(max(0.5, L * 0.25), L * 2, 30)
        ds = np.linspace(D * 0.5, D * 1.5, 30)
        sls = np.linspace(0, max(sigma_l * 2, 1), 30)
        for j, (xs, ys) in enumerate([
            (lts, inv.reorder_point(D, lts, inv.safety_stock(z, sigma_d, lts, D, sigma_l))),
            (ds, inv.reorder_point(ds, L, inv.safety_stock(z, sigma_d, L, ds, sigma_l))),
            (sls, inv.reorder_point(D, L, inv.safety_stock(z, sigma_d, L, D, sls))),
        ]):
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", marker_size=4, showlegend=False,
                                     line=dict(color=["#1f77b4", "#2ca02c", "#d62728"][j])), 1, j + 1)
        fig.update_yaxes(title="Reorder point", col=1)
        show(fig, 320)
        st.caption("Unlike the fixed-safety-stock sweep in the notebook, the safety stock here is recomputed "
                   "for every parameter value, so curves are non-linear (e.g. ROP grows faster than linearly "
                   "with σL).")

# ======================================================================================
# 5. Bayesian optimisation of the reorder point
# ======================================================================================
with tabs[5]:
    st.subheader("Bayesian optimisation of reorder point (s) and order quantity (Q)")
    st.caption("Instead of a manual parameter sweep, a Gaussian-process surrogate learns the cost surface of a "
               "stochastic (s, Q) inventory simulation and Expected Improvement chooses where to evaluate next. "
               f"Demand and lead-time inputs come from the 🛡️ tab: D = {D:.1f}/day, σD = {sigma_d:.1f}, "
               f"L = {L:.1f} d, σL = {sigma_l:.1f} d, {lt_dist.lower()} lead times, disruption p = {dis_p:.2f}.")
    o1, o2, o3 = st.columns(3)
    with o1:
        with st.container(border=True):
            st.markdown("**Costs**")
            h_day = st.number_input("Holding $/unit/day", 0.0, 100.0, round(unit_h / 365, 3), 0.01, format="%.3f")
            b_day = st.number_input("Backorder penalty $/unit/day", 0.0, 1000.0, 1.0, 0.1)
            K = st.number_input("Ordering cost $/order", 0.0, 1e5, 150.0, 10.0)
    with o2:
        with st.container(border=True):
            st.markdown("**Constraints**")
            min_fill = st.slider("Minimum fill rate", 0.80, 0.999, 0.95, 0.005)
            eoq_q = inv.eoq(D, K, h_day)
            cap = st.number_input("Max average on-hand (warehouse capacity)", 1.0, 1e7,
                                  float(round(max(ss_f + eoq_q, 1) * 1.2)), 10.0)
    with o3:
        with st.container(border=True):
            st.markdown("**Optimiser**")
            b1, b2 = st.columns(2)
            n_init = b1.number_input("Initial points", 3, 20, 8)
            n_iter = b2.number_input("BO iterations", 5, 60, 22)
            reps = b1.number_input("Replications", 5, 100, 20, help="Simulated years per evaluation")
            do_grid = b2.checkbox("Grid benchmark", True, help="15×15 brute-force sweep for comparison")
            run = st.button("🚀 Run optimisation", type="primary", width="stretch")

    s_hi = max(rop_f * 2.5, D * L * 2, 10.0)
    q_lo, q_hi = max(1.0, D), max(D * 30, eoq_q * 3)
    bounds = [(0.0, s_hi), (q_lo, q_hi)]

    def sq_eval(x):
        r = inv.simulate_sQ(float(x[0]), float(x[1]), D, sigma_d, L, sigma_l, h_day, b_day, K, 365, int(reps),
                            lt_dist, dis_p, dis_d, seed=123)
        short = max(0.0, min_fill - r["fill_rate"])
        over = max(0.0, r["avg_on_hand"] - cap) / cap
        r["feasible"] = short == 0 and over == 0
        return r["annual_cost"] * (1 + 10 * short + 5 * over), r

    sig = (D, sigma_d, L, sigma_l, lt_dist, dis_p, dis_d, h_day, b_day, K, min_fill, cap, n_init, n_iter, reps, do_grid)
    if run:
        prog = st.progress(0.0, "Bayesian optimisation...")
        textbook = [rop_f, max(q_lo, min(q_hi, eoq_q))]
        res = bayesian_optimize(sq_eval, bounds, int(n_init), int(n_iter), seed=int(seed), names=["s", "Q"],
                                x0=[textbook],
                                callback=lambda i, n, b: prog.progress(i / n, f"Evaluation {i}/{n} · best ${b:,.0f}"))
        grid = None
        if do_grid:
            prog.progress(0.0, "Grid benchmark...")
            gs, gq = np.linspace(*bounds[0], 15), np.linspace(*bounds[1], 15)
            pts = [(s_, q_) for s_ in gs for q_ in gq]
            vals = []
            for i, p in enumerate(pts):
                vals.append(sq_eval(p))
                prog.progress((i + 1) / len(pts), f"Grid point {i + 1}/{len(pts)}")
            grid = {"pts": np.array(pts), "y": np.array([v[0] for v in vals]), "info": [v[1] for v in vals]}
        prog.empty()
        st.session_state["bo"] = {"res": res, "grid": grid, "sig": sig, "textbook": (textbook, sq_eval(textbook))}

    if "bo" not in st.session_state:
        st.info("Set costs and constraints, then click **Run optimisation**.")
    else:
        bo = st.session_state["bo"]
        if bo["sig"] != sig:
            st.warning("Inputs changed since the last run — results below are stale. Click **Run optimisation**.")
        res, grid = bo["res"], bo["grid"]
        (tb_x, (tb_y, tb_info)) = bo["textbook"]
        best_info = res.info[res.best_index]
        rows = [{"Method": "Textbook (ROP formula + EOQ)", "s (ROP)": tb_x[0], "Q": tb_x[1], "Evaluations": 1,
                 "Annual cost": tb_info["annual_cost"], "Fill rate": tb_info["fill_rate"],
                 "Avg on-hand": tb_info["avg_on_hand"], "Feasible": tb_info["feasible"], "Objective": tb_y}]
        if grid is not None:
            gi = int(np.argmin(grid["y"]))
            g = grid["info"][gi]
            rows.append({"Method": "Grid search (manual sweep)", "s (ROP)": grid["pts"][gi][0], "Q": grid["pts"][gi][1],
                         "Evaluations": len(grid["y"]), "Annual cost": g["annual_cost"], "Fill rate": g["fill_rate"],
                         "Avg on-hand": g["avg_on_hand"], "Feasible": g["feasible"], "Objective": grid["y"][gi]})
        rows.append({"Method": "Bayesian optimisation", "s (ROP)": res.best_x[0], "Q": res.best_x[1],
                     "Evaluations": len(res.y), "Annual cost": best_info["annual_cost"],
                     "Fill rate": best_info["fill_rate"], "Avg on-hand": best_info["avg_on_hand"],
                     "Feasible": best_info["feasible"], "Objective": res.best_y})
        comp = pd.DataFrame(rows)

        k = st.columns(4)
        k[0].metric("Optimal reorder point s*", f"{res.best_x[0]:,.0f}", f"textbook {tb_x[0]:,.0f}", delta_color="off")
        k[1].metric("Optimal order quantity Q*", f"{res.best_x[1]:,.0f}", f"EOQ {tb_x[1]:,.0f}", delta_color="off")
        k[2].metric("Annual cost", f"${best_info['annual_cost']:,.0f}",
                    f"{best_info['annual_cost'] / tb_info['annual_cost'] - 1:+.1%} vs textbook", delta_color="inverse")
        k[3].metric("Fill rate", f"{best_info['fill_rate']:.2%}", f"target ≥ {min_fill:.1%}", delta_color="off")
        table(comp, {"s (ROP)": "{:,.0f}", "Q": "{:,.0f}", "Annual cost": "${:,.0f}", "Fill rate": "{:.2%}",
                     "Avg on-hand": "{:,.0f}", "Objective": "{:,.0f}"})

        c1, c2 = st.columns([1, 1.25])
        with c1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=np.arange(1, len(res.y) + 1), y=res.y, mode="markers", name="Evaluation",
                                     marker=dict(color="#bbbbbb")))
            fig.add_trace(go.Scatter(x=np.arange(1, len(res.y) + 1), y=res.best_so_far(), name="BO best so far",
                                     line=dict(color=AI_COLOR, width=3, shape="hv")))
            fig.add_hline(y=tb_y, line_dash="dot", line_color=BASE_COLOR, annotation_text="Textbook")
            if grid is not None:
                fig.add_hline(y=float(grid["y"].min()), line_dash="dash", line_color="#2ca02c",
                              annotation_text=f"Grid best ({len(grid['y'])} evals)", annotation_position="bottom right")
            fig.update_xaxes(title="Evaluation #")
            fig.update_yaxes(title="Penalised annual cost ($)",
                             range=[res.y.min() * 0.97, max(tb_y, res.y.min()) * 1.25])
            show(fig, 420, "Convergence")
        with c2:
            gs, gq = np.linspace(*bounds[0], 60), np.linspace(*bounds[1], 60)
            S, Qm = np.meshgrid(gs, gq)
            mu, _ = res.predict(np.column_stack([S.ravel(), Qm.ravel()]))
            fig = go.Figure(go.Contour(x=gs, y=gq, z=mu.reshape(S.shape), colorscale="Viridis", ncontours=25,
                                       colorbar=dict(title="Cost"), zmin=float(mu.min()),
                                       zmax=float(np.percentile(mu, 70))))
            feas = np.array([i["feasible"] for i in res.info])
            fig.add_trace(go.Scatter(x=res.X[feas, 0], y=res.X[feas, 1], mode="markers", name="Evaluated (feasible)",
                                     marker=dict(color="white", size=7, line=dict(color="black", width=1))))
            fig.add_trace(go.Scatter(x=res.X[~feas, 0], y=res.X[~feas, 1], mode="markers", name="Evaluated (infeasible)",
                                     marker=dict(color="white", symbol="x", size=7, line=dict(color="black", width=1))))
            fig.add_trace(go.Scatter(x=[tb_x[0]], y=[tb_x[1]], mode="markers", name="Textbook",
                                     marker=dict(symbol="diamond", size=13, color="orange", line=dict(color="black", width=1))))
            fig.add_trace(go.Scatter(x=[res.best_x[0]], y=[res.best_x[1]], mode="markers", name="BO optimum",
                                     marker=dict(symbol="star", size=18, color="red", line=dict(color="black", width=1))))
            fig.update_xaxes(title="Reorder point s")
            fig.update_yaxes(title="Order quantity Q")
            fig.update_layout(hovermode="closest")
            show(fig, 420, "GP surrogate of the cost surface (posterior mean)")

        with st.expander("Cost breakdown of the BO optimum"):
            bd = pd.DataFrame({"Component": ["Holding", "Backorder", "Ordering"],
                               "Annual $": [best_info["holding"], best_info["backorder"], best_info["ordering"]]})
            c1, c2 = st.columns([1, 1.5])
            with c1:
                table(bd, {"Annual $": "${:,.0f}"})
                st.caption(f"{best_info['orders_per_year']:.1f} orders/year · avg on-hand "
                           f"{best_info['avg_on_hand']:,.0f} (capacity {cap:,.0f})")
            with c2:
                fig = go.Figure(go.Pie(labels=bd["Component"], values=bd["Annual $"], hole=0.5))
                show(fig, 260)
        st.caption("How it works: 1) Latin-hypercube initial design (+ the textbook point as a warm start); "
                   "2) fit a Matern-5/2 Gaussian process; 3) maximise Expected Improvement; 4) simulate the "
                   "proposed (s, Q) for the configured number of years; repeat. Constraints are handled with "
                   "a multiplicative penalty on fill-rate shortfall and capacity excess.")
