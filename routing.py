"""Order routing / inventory allocation across a two-echelon network.

Two distribution centres (DCs) replenish four retailers every week. DC inbound supply is
fixed and *regionally unbalanced*, so the allocation rule matters:

* **Static nearest-DC routing** (baseline): each retailer is served only by its closest DC;
  when a DC is short it rations proportionally to demand.
* **Optimised routing (LP)**: each week solve a min-cost transportation problem::

      min  sum_dr c[d,r] x[d,r]  +  p * sum_r u[r]
      s.t. sum_r x[d,r]          <= inventory[d]       (DC availability)
           sum_d x[d,r] + u[r]   =  demand[r]          (serve or lose demand)
           x, u >= 0

  so cross-region shipments are used whenever the lost-sale penalty exceeds the extra cost.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import linprog


@dataclass
class Network:
    dcs: list[str] = field(default_factory=lambda: ["DC North", "DC South"])
    retailers: list[str] = field(default_factory=lambda: ["Retailer A", "Retailer B",
                                                           "Retailer C", "Retailer D"])
    # shipping cost per unit (rows: DCs, cols: retailers); A, B are northern, C, D southern
    ship_cost: np.ndarray = field(default_factory=lambda: np.array([[1.0, 1.5, 4.0, 4.5],
                                                                   [4.5, 4.0, 1.2, 1.0]]))
    weekly_supply: np.ndarray = field(default_factory=lambda: np.array([240.0, 160.0]))
    demand_means: np.ndarray = field(default_factory=lambda: np.array([90.0, 80.0, 110.0, 120.0]))


def simulate_routing(net: Network, policy: str, weeks: int = 26, lost_sale_penalty: float = 10.0,
                     holding_cost: float = 0.5, demand_cv: float = 0.2, seed: int = 11) -> dict:
    rng = np.random.default_rng(seed)
    nd, nr = net.ship_cost.shape
    demand = np.clip(np.round(rng.normal(net.demand_means, demand_cv * net.demand_means,
                                         (weeks, nr))), 0, None)
    nearest = net.ship_cost.argmin(axis=0)
    inv = net.weekly_supply.copy()
    flows = np.zeros((nd, nr))
    weekly = []
    for w in range(weeks):
        d = demand[w]
        if policy == "lp":
            c = np.concatenate([net.ship_cost.ravel(), np.full(nr, lost_sale_penalty)])
            A_ub = np.zeros((nd, nd * nr + nr))
            for i in range(nd):
                A_ub[i, i * nr:(i + 1) * nr] = 1
            A_eq = np.zeros((nr, nd * nr + nr))
            for r in range(nr):
                A_eq[r, [i * nr + r for i in range(nd)]] = 1
                A_eq[r, nd * nr + r] = 1
            res = linprog(c, A_ub=A_ub, b_ub=inv, A_eq=A_eq, b_eq=d, bounds=(0, None), method="highs")
            x = res.x[: nd * nr].reshape(nd, nr)
        else:
            x = np.zeros((nd, nr))
            for i in range(nd):
                mine = nearest == i
                need = d[mine].sum()
                ratio = 1.0 if need <= inv[i] else (inv[i] / need if need > 0 else 0.0)
                x[i, mine] = d[mine] * ratio
        shipped = x.sum(axis=1)
        served = x.sum(axis=0)
        lost = d - served
        inv = inv - shipped
        ship_cost = float((x * net.ship_cost).sum())
        weekly.append({"week": w + 1, "demand": d.sum(), "served": served.sum(),
                       "fill_rate": served.sum() / max(d.sum(), 1e-9),
                       "shipping_cost": ship_cost, "lost_sales_cost": lost.sum() * lost_sale_penalty,
                       "holding_cost": inv.sum() * holding_cost, "dc_inventory": inv.sum()})
        flows += x
        inv = inv + net.weekly_supply  # next week's inbound supply
    wk = pd.DataFrame(weekly)
    per_ret = pd.DataFrame({"retailer": net.retailers, "demand": demand.sum(axis=0),
                            "served": flows.sum(axis=0)})
    per_ret["fill_rate"] = per_ret["served"] / per_ret["demand"]
    totals = {k: float(wk[k].sum()) for k in ("shipping_cost", "lost_sales_cost", "holding_cost")}
    totals["total_cost"] = sum(totals.values())
    totals["fill_rate"] = float(wk["served"].sum() / wk["demand"].sum())
    return {"weekly": wk, "flows": flows, "per_retailer": per_ret, "totals": totals}
