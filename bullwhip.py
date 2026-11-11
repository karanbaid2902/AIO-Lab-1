"""Bullwhip-effect simulation.

Two models are provided:

1. :func:`classic_bullwhip` - the stylised model from the lab slide / notebook: every tier
   copies the downstream signal with a fixed delay and an amplification factor.

2. :func:`simulate_supply_chain` - a proper multi-echelon inventory simulation
   (Customer -> Retailer -> Distributor -> Manufacturer -> unlimited raw supply).
   Each stage holds on-hand inventory, carries a backlog, has replenishment lead time,
   forecasts its demand and places orders with a (proportional) order-up-to policy::

       S_t   = F_t * (L + R) + z * sigma_t * sqrt(L + R)          (target inventory position)
       O_t   = max(0, F_t + alpha * (F_t * L + SS_t - IP_t))         (POUT policy)

   With ``alpha = 1`` this is the classic order-up-to policy that *causes* bullwhip when the
   forecast F_t is a moving average of the local (already distorted) order stream.

   The AI-redesigned network switches on three levers:

   * **AI demand forecasting** - the retailer orders on an ML forecast of end-customer demand.
   * **Information sharing** - every stage forecasts on end-customer (POS) demand instead of
     the orders it receives (the "redesign the network" lever).
   * **Order smoothing** - ``alpha < 1``; tuned automatically with Bayesian optimisation.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

STAGES = ("Retailer", "Distributor", "Manufacturer")
TIER_LABELS = ("Customer",) + STAGES


# --------------------------------------------------------------------------------------
# 1. Classic (slide) model
# --------------------------------------------------------------------------------------
def classic_bullwhip(demand: np.ndarray, delay: int = 1, amplification: float = 1.2,
                     n_tiers: int = 3) -> np.ndarray:
    """Return array (n_tiers + 1, T): customer demand followed by each upstream tier.

    Same logic as the lab notebook, but kept in floats (the notebook's ``zeros_like`` on an
    integer Poisson array silently truncated the amplified values).
    """
    demand = np.asarray(demand, dtype=float)
    tiers = [demand]
    for _ in range(n_tiers):
        prev = tiers[-1]
        nxt = np.zeros_like(prev)
        nxt[delay:] = amplification * prev[: len(prev) - delay]
        tiers.append(nxt)
    return np.vstack(tiers)


# --------------------------------------------------------------------------------------
# 2. Multi-echelon simulation
# --------------------------------------------------------------------------------------
@dataclass
class ChainConfig:
    lead_times: tuple[int, int, int] = (2, 2, 3)          # replenishment lead time (weeks)
    holding_costs: tuple[float, float, float] = (1.0, 0.75, 0.5)   # per unit-week
    backlog_costs: tuple[float, float, float] = (5.0, 2.5, 1.5)    # per unit-week
    review_period: int = 1
    # Starting on-hand inventory per stage; None = steady state (cycle stock + safety stock)
    initial_inventory: tuple[float, float, float] | None = None


@dataclass
class OrderingPolicy:
    forecast_method: str = "moving_average"   # "moving_average" | "exp_smoothing"
    ma_window: int = 4
    es_alpha: float = 0.4
    safety_factor: float = 1.65               # z
    smoothing: float = 1.0                    # alpha (1 = classic order-up-to)
    info_sharing: bool = False
    ai_forecast: np.ndarray | None = None     # forecast of demand[t+1] available at week t
    ai_sigma: float | None = None             # std of the AI forecast error


@dataclass
class SimulationResult:
    demand: np.ndarray
    orders: np.ndarray        # (n_stages, T) orders placed upstream
    inventory: np.ndarray     # (n_stages, T) end-of-week on-hand
    backlog: np.ndarray       # (n_stages, T) end-of-week backlog
    forecasts: np.ndarray     # (n_stages, T)
    costs: np.ndarray         # (n_stages, T)
    fill_rate: float          # share of customer demand served from stock in the same week
    stages: tuple[str, ...] = field(default=STAGES)

    @property
    def total_cost(self) -> float:
        return float(self.costs.sum())

    def bullwhip_ratios(self) -> np.ndarray:
        var_d = np.var(self.demand, ddof=1)
        return np.var(self.orders, axis=1, ddof=1) / max(var_d, 1e-9)

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame({
            "Stage": list(self.stages),
            "Bullwhip ratio (Var orders / Var demand)": self.bullwhip_ratios(),
            "Std of orders": self.orders.std(axis=1, ddof=1),
            "Peak order": self.orders.max(axis=1),
            "Avg on-hand": self.inventory.mean(axis=1),
            "Avg backlog": self.backlog.mean(axis=1),
            "Total cost": self.costs.sum(axis=1),
        })


def simulate_supply_chain(demand: np.ndarray, history: np.ndarray, cfg: ChainConfig,
                          policy: OrderingPolicy) -> SimulationResult:
    """Run the beer-game-style simulation.

    ``history`` is pre-simulation customer demand used to initialise forecasts and to start
    the chain in steady state (pipelines pre-filled with the average demand).
    """
    demand = np.asarray(demand, dtype=float)
    history = np.asarray(history, dtype=float)
    T, n = len(demand), len(cfg.lead_times)
    L = np.asarray(cfg.lead_times, dtype=int)
    R = cfg.review_period
    z = policy.safety_factor
    alpha = policy.smoothing
    sd_window = max(8, 2 * policy.ma_window)

    mu0 = float(history[-26:].mean())
    sigma0 = float(history[-26:].std(ddof=1)) if len(history) > 2 else np.sqrt(mu0)

    # state
    if cfg.initial_inventory is not None:
        on_hand = np.asarray(cfg.initial_inventory, dtype=float).copy()
    else:
        on_hand = np.array([mu0 * R + z * sigma0 * np.sqrt(L[i] + R) for i in range(n)])
    backlog = np.zeros(n)
    arrivals = np.zeros((n, T + int(L.max()) + 2))
    for i in range(n):
        arrivals[i, : L[i]] = mu0                   # steady-state pipeline
    on_order = L * mu0
    seen = [list(history[-52:]) for _ in range(n)]   # demand history observed by each stage
    customer_seen = list(history[-52:])
    es_state = np.full(n, mu0)

    orders = np.zeros((n, T))
    inv_hist = np.zeros((n, T))
    back_hist = np.zeros((n, T))
    fc_hist = np.zeros((n, T))
    cost_hist = np.zeros((n, T))
    served_new_total = 0.0

    for t in range(T):
        on_hand += arrivals[:, t]
        on_order -= arrivals[:, t]
        customer_seen.append(demand[t])
        incoming = demand[t]
        for i in range(n):
            # ---- fulfil demand (backlog first) -------------------------------------
            if i == 0:
                served_new_total += float(np.clip(on_hand[0] - backlog[0], 0, incoming))
            required = incoming + backlog[i]
            ship = min(on_hand[i], required)
            on_hand[i] -= ship
            backlog[i] = required - ship
            if i > 0:
                arrivals[i - 1, t + L[i - 1]] += ship
            seen[i].append(incoming)

            # ---- forecast ----------------------------------------------------------
            use_ai = policy.ai_forecast is not None and (i == 0 or policy.info_sharing)
            if use_ai:
                F = float(policy.ai_forecast[t])
                sigma = float(policy.ai_sigma if policy.ai_sigma is not None else sigma0)
            else:
                basis = customer_seen if policy.info_sharing else seen[i]
                if policy.forecast_method == "exp_smoothing":
                    es_state[i] = policy.es_alpha * basis[-1] + (1 - policy.es_alpha) * es_state[i]
                    F = float(es_state[i])
                else:
                    F = float(np.mean(basis[-policy.ma_window:]))
                sigma = float(np.std(basis[-sd_window:], ddof=1))

            # ---- order (proportional order-up-to, POUT) -------------------------------
            # Replace expected consumption over the review period (F * R) and close a
            # fraction alpha of the gap between the target (F*L + SS) and the inventory
            # position. alpha = 1 gives the classic order-up-to policy O = S - IP.
            safety = z * sigma * np.sqrt(L[i] + R)
            inv_position = on_hand[i] - backlog[i] + on_order[i]
            order = max(0.0, F * R + alpha * (F * L[i] + safety - inv_position))
            orders[i, t] = order
            on_order[i] += order
            if i == n - 1:  # manufacturer: production with lead time, unlimited capacity
                arrivals[i, t + L[i]] += order

            fc_hist[i, t] = F
            inv_hist[i, t] = on_hand[i]
            back_hist[i, t] = backlog[i]
            cost_hist[i, t] = cfg.holding_costs[i] * on_hand[i] + cfg.backlog_costs[i] * backlog[i]
            incoming = order  # becomes the demand seen by the next stage upstream

    fill = served_new_total / max(demand.sum(), 1e-9)
    return SimulationResult(demand, orders, inv_hist, back_hist, fc_hist, cost_hist, fill)


# --------------------------------------------------------------------------------------
# 3. AI forecaster for end-customer demand
# --------------------------------------------------------------------------------------
AI_MODELS = {
    "Random Forest": lambda seed: RandomForestRegressor(
        n_estimators=200, min_samples_leaf=3, random_state=seed, n_jobs=-1),
    "Ridge Regression": lambda seed: make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
    "Gradient Boosting": lambda seed: GradientBoostingRegressor(
        n_estimators=200, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=seed),
}


def _customer_features(demand: np.ndarray, promo: np.ndarray) -> pd.DataFrame:
    s = pd.Series(demand)
    t = np.arange(len(demand))
    return pd.DataFrame({
        "lag_1": s.shift(1), "lag_2": s.shift(2), "lag_3": s.shift(3), "lag_4": s.shift(4),
        "roll_mean_4": s.shift(1).rolling(4).mean(),
        "roll_mean_12": s.shift(1).rolling(12).mean(),
        "season_sin": np.sin(2 * np.pi * t / 52.0),
        "season_cos": np.cos(2 * np.pi * t / 52.0),
        "promotion": promo,
    })


def ai_customer_forecast(full: pd.DataFrame, n_history: int, model_name: str = "Gradient Boosting",
                         seed: int = 0) -> tuple[np.ndarray, float]:
    """Train on the first ``n_history`` weeks, then produce rolling one-step-ahead forecasts.

    ``full`` must contain ``n_history + T + 1`` rows (the extra row supplies the known promo
    calendar for the final look-ahead). Returns (forecast array of length T, error sigma)
    where forecast[t] is the prediction of demand at simulation week t+1 made at week t.
    """
    demand = full["demand"].to_numpy(float)
    X = _customer_features(demand, full["promotion"].to_numpy())
    train = X.iloc[:n_history].dropna()
    y_train = demand[train.index]
    model = AI_MODELS[model_name](seed)

    # Out-of-sample error estimate (expanding-window time-series CV) -> sigma for safety stock.
    errors = []
    for tr_idx, te_idx in TimeSeriesSplit(n_splits=5).split(train):
        m = AI_MODELS[model_name](seed).fit(train.iloc[tr_idx], y_train[tr_idx])
        errors.append(y_train[te_idx] - m.predict(train.iloc[te_idx]))
    sigma = float(np.std(np.concatenate(errors), ddof=1))

    model.fit(train, y_train)
    T = len(full) - n_history - 1
    target_rows = X.iloc[n_history + 1: n_history + 1 + T]   # target week t+1 for each t
    return np.clip(model.predict(target_rows), 0, None), sigma
