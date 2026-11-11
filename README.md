# 📦 AI Supply Chain Lab — Bullwhip Effect & AI-Enhanced Reorder Point

Interactive Streamlit dashboard for **Session II – AI in Operations** (lab activity + exercise).

## What it does

| Tab | Lab requirement | Techniques |
|---|---|---|
| 🌊 **Bullwhip Simulation** | Define the environment and build the base simulation **without AI** | Classic delay/amplification model (lab slide) + full multi-echelon inventory simulation (Retailer → Distributor → Manufacturer) with per-stage lead times, starting inventory, costs, backlog and order-up-to policies |
| 🤖 **AI-Redesigned Network** | Integrate AI for **forecasting** and **routing/optimisation** | ML forecast of customer demand, POS information sharing, order smoothing auto-tuned by **Bayesian optimisation**, and **LP-optimised order routing** across distribution centres |
| 📈 **Demand Forecasting** | ML forecasting with seasonality, promotions and external factors; add new data; update continuously | Time series (**Holt-Winters**, **SARIMAX** with drivers), regression (Ridge), Random Forest, Gradient Boosting, **LSTM (PyTorch)**; walk-forward backtest with periodic retraining; manual entry, CSV upload and streamed data |
| 🛡️ **Dynamic Safety Stock** | Probabilistic model + **Monte Carlo** risk scenarios | `SS = Z·√(σD²·L + D²·σL²)`; lead-time distributions with disruptions; dynamic ROP over the forecast horizon; tornado and service-level sensitivity |
| 🎯 **Bayesian ROP Optimisation** | Bayesian optimisation of reorder points under constraints | Gaussian-process surrogate + Expected Improvement over a stochastic (s, Q) simulation, with fill-rate and capacity constraints; compared against the textbook formula and a grid sweep |

### Main results with the default settings
- The baseline chain amplifies demand variance **3.8× → 15× → 55×** from retailer to manufacturer; with all AI levers on, the manufacturer's ratio falls to about **1.5×** and total cost drops about **30–75 %**, depending on the demand pattern.
- ML and time-series models roughly **halve forecast error** compared with the naïve forecast (WAPE 17 % → 7–8 % for SARIMAX and Ridge; Holt-Winters, which uses history only, reaches 12 %).
- Under supply-disruption risk, the closed-form safety stock delivers only **about 92 %** service against a 95 % target; Monte Carlo sizes the buffer correctly.
- Bayesian optimisation finds an (s, Q) policy about **7 % cheaper** than ROP + EOQ in about 30 simulations. That is as good as or better than a 225-point grid search.

## Project structure

```
app.py                     Streamlit dashboard (UI only)
supply_chain/
  data.py                  synthetic demand generators, CSV validation
  bullwhip.py              classic + multi-echelon bullwhip simulation, AI forecaster
  forecasting.py           ML / LSTM forecasters, walk-forward evaluation, metrics
  inventory.py             safety stock, Monte Carlo, sensitivity, (s, Q) simulator
  optimization.py          Gaussian-process Bayesian optimisation (Expected Improvement)
  routing.py               LP-based order routing / inventory allocation
requirements.txt
.streamlit/config.toml
```

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Push this folder to a GitHub repository (`app.py` must be at the repository root).
2. Go to https://share.streamlit.io → **Create app** → choose the repository, branch `main`, main file `app.py`.
3. (Optional) Under **Advanced settings**, choose Python 3.12.
4. Click **Deploy**. The first build takes a few minutes because PyTorch is installed.

### Uploading your own data
In **📈 Demand Forecasting → Upload CSV**, provide a CSV with the columns `date, demand` and, optionally, `promotion, holiday, temperature`. Missing drivers are filled in automatically.
