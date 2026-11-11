"""Reorder point, dynamic safety stock and Monte Carlo risk analysis.

Safety stock with demand AND lead-time uncertainty (lab formula)::

    SS = Z * sqrt(sigma_D^2 * L + D^2 * sigma_L^2)
    ROP = D * L + SS

All quantities in this module use a *daily* time unit unless stated otherwise.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


# --------------------------------------------------------------------------------------
# Analytical (probabilistic) model
# --------------------------------------------------------------------------------------
def z_value(service_level: float) -> float:
    return float(norm.ppf(service_level))


def safety_stock(z: float, sigma_d: float, lead_time: float, demand: float, sigma_l: float):
    """Works element-wise on numpy arrays as well as scalars."""
    return z * np.sqrt(np.square(sigma_d) * lead_time + np.square(demand) * np.square(sigma_l))


def reorder_point(demand, lead_time, ss):
    return demand * lead_time + ss


# --------------------------------------------------------------------------------------
# Monte Carlo
# --------------------------------------------------------------------------------------
def sample_lead_times(rng: np.random.Generator, n: int, mean: float, sd: float, dist: str,
                      disruption_prob: float = 0.0, disruption_days: float = 0.0) -> np.ndarray:
    if sd <= 0:
        lt = np.full(n, float(mean))
    elif dist == "Gamma (right-skewed)":
        k = (mean / sd) ** 2
        lt = rng.gamma(k, sd ** 2 / mean, n)
    elif dist == "Lognormal (heavy tail)":
        s2 = np.log(1 + (sd / mean) ** 2)
        lt = rng.lognormal(np.log(mean) - s2 / 2, np.sqrt(s2), n)
    else:  # Normal, truncated at a small positive value
        lt = np.clip(rng.normal(mean, sd, n), 0.1, None)
    if disruption_prob > 0 and disruption_days > 0:
        lt = lt + (rng.random(n) < disruption_prob) * disruption_days
    return lt


def monte_carlo_lead_time_demand(D: float, sigma_d: float, L: float, sigma_l: float,
                                 n_sims: int = 20000, demand_dist: str = "Normal",
                                 lt_dist: str = "Normal", disruption_prob: float = 0.0,
                                 disruption_days: float = 0.0, seed: int = 0) -> dict:
    """Sample demand during lead time (LTD).

    Lead time L_k is drawn first; then LTD | L_k has mean D*L_k and variance sigma_d^2 * L_k
    (sum of L_k i.i.d. daily demands), using a Normal or moment-matched Gamma distribution.
    """
    rng = np.random.default_rng(seed)
    lt = sample_lead_times(rng, n_sims, L, sigma_l, lt_dist, disruption_prob, disruption_days)
    mean, sd = D * lt, sigma_d * np.sqrt(lt)
    if demand_dist == "Gamma (right-skewed)" and sigma_d > 0:
        k = (mean / sd) ** 2
        ltd = rng.gamma(k, sd ** 2 / mean)
    else:
        ltd = np.clip(rng.normal(mean, sd), 0, None)
    return {"lead_times": lt, "ltd": ltd}


def monte_carlo_safety_stock(ltd: np.ndarray, D: float, L: float, service_level: float) -> dict:
    rop = float(np.quantile(ltd, service_level))
    return {"rop": rop, "ss": rop - D * L}


def achieved_service_level(ltd: np.ndarray, rop: float) -> float:
    """Probability of no stock-out during a replenishment cycle (cycle service level)."""
    return float(np.mean(ltd <= rop))


# --------------------------------------------------------------------------------------
# Sensitivity analysis
# --------------------------------------------------------------------------------------
def service_level_tradeoff(D, sigma_d, L, sigma_l, unit_holding_cost_per_year,
                           levels=np.round(np.arange(0.80, 0.9951, 0.005), 3)) -> pd.DataFrame:
    z = norm.ppf(levels)
    ss = safety_stock(z, sigma_d, L, D, sigma_l)
    return pd.DataFrame({"service_level": levels, "z": z, "safety_stock": ss,
                         "holding_cost_per_year": ss * unit_holding_cost_per_year,
                         "reorder_point": D * L + ss})


def tornado(D, sigma_d, L, sigma_l, service_level, swing: float = 0.2) -> pd.DataFrame:
    """One-at-a-time +/- swing on each driver of the dynamic reorder point."""
    base_args = dict(D=D, sigma_d=sigma_d, L=L, sigma_l=sigma_l, sl=service_level)

    def rop(D, sigma_d, L, sigma_l, sl):
        return reorder_point(D, L, safety_stock(z_value(sl), sigma_d, L, D, sigma_l))

    base = rop(**base_args)
    labels = {"D": "Daily demand D", "sigma_d": "Demand std sigma_D", "L": "Lead time L",
              "sigma_l": "Lead-time std sigma_L", "sl": "Service level"}
    rows = []
    for k, label in labels.items():
        lo_args, hi_args = dict(base_args), dict(base_args)
        if k == "sl":  # service level: move the shortfall (1 - SL) by +/- swing
            lo_args[k] = 1 - (1 - base_args[k]) * (1 + swing)
            hi_args[k] = min(0.9999, 1 - (1 - base_args[k]) * (1 - swing))
        else:
            lo_args[k] = base_args[k] * (1 - swing)
            hi_args[k] = base_args[k] * (1 + swing)
        rows.append({"driver": label, "low": rop(**lo_args) - base, "high": rop(**hi_args) - base})
    out = pd.DataFrame(rows)
    out["range"] = (out["high"] - out["low"]).abs()
    return out.sort_values("range"), base


# --------------------------------------------------------------------------------------
# Stochastic (s, Q) inventory simulation - objective for Bayesian optimisation
# --------------------------------------------------------------------------------------
def simulate_sQ(s: float, Q: float, D: float, sigma_d: float, L: float, sigma_l: float,
                holding_cost: float, backorder_cost: float, order_cost: float,
                days: int = 365, reps: int = 30, lt_dist: str = "Normal",
                disruption_prob: float = 0.0, disruption_days: float = 0.0,
                seed: int = 123) -> dict:
    """Continuous-review (s, nQ) policy simulated day-by-day over ``reps`` replications.

    Common random numbers (fixed ``seed``) make the objective deterministic in (s, Q), which
    gives the Bayesian optimiser a smooth surface to learn. Backorders are allowed.
    """
    rng = np.random.default_rng(seed)
    demand = np.clip(rng.normal(D, sigma_d, (reps, days)), 0, None)
    lt = np.maximum(1, np.round(sample_lead_times(rng, reps * days, L, sigma_l, lt_dist,
                                                  disruption_prob, disruption_days))).astype(int)
    lt = lt.reshape(reps, days)
    arrivals = np.zeros((reps, days + int(lt.max()) + 1))
    on_hand = np.full(reps, s + Q, dtype=float)
    backlog = np.zeros(reps)
    on_order = np.zeros(reps)
    hold = np.zeros(reps)
    back = np.zeros(reps)
    n_orders = np.zeros(reps)
    served = np.zeros(reps)
    rows = np.arange(reps)
    for t in range(days):
        on_hand += arrivals[:, t]
        on_order -= arrivals[:, t]
        d = demand[:, t]
        served += np.clip(on_hand - backlog, 0, d)
        need = backlog + d
        ship = np.minimum(on_hand, need)
        on_hand -= ship
        backlog = need - ship
        ip = on_hand - backlog + on_order
        k = np.where(ip <= s, np.floor((s - ip) / Q) + 1, 0)
        if k.any():
            m = k > 0
            np.add.at(arrivals, (rows[m], t + lt[m, t]), k[m] * Q)
            on_order += k * Q
            n_orders += m
        hold += on_hand
        back += backlog
    total = holding_cost * hold + backorder_cost * back + order_cost * n_orders
    scale = 365.0 / days
    return {
        "annual_cost": float(total.mean() * scale),
        "holding": float((holding_cost * hold).mean() * scale),
        "backorder": float((backorder_cost * back).mean() * scale),
        "ordering": float((order_cost * n_orders).mean() * scale),
        "fill_rate": float(served.sum() / max(demand.sum(), 1e-9)),
        "avg_on_hand": float(hold.mean() / days),
        "orders_per_year": float(n_orders.mean() * scale),
    }


def eoq(D_daily: float, order_cost: float, holding_cost_daily: float) -> float:
    return float(np.sqrt(2 * D_daily * order_cost / max(holding_cost_daily, 1e-12)))
