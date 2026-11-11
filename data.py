"""Synthetic data generation and validation.

Two kinds of data are produced:

* A rich **weekly demand history** (trend, yearly seasonality, promotions, holidays and an
  external factor - temperature) used by the forecasting / safety-stock / optimisation tabs.
* A simple **end-customer demand stream** for the bullwhip simulation (Poisson, as on the
  lab slide, plus seasonal and step-change variants).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

HOLIDAY_WEEKS = frozenset({47, 51, 52})  # ISO weeks with a holiday uplift
REQUIRED_COLUMNS = ("date", "demand")
OPTIONAL_COLUMNS = ("promotion", "holiday", "temperature")


@dataclass(frozen=True)
class DemandParams:
    """Structural parameters of the synthetic weekly demand process."""

    base_level: float = 200.0
    trend_per_week: float = 0.5
    seasonal_amplitude: float = 0.25
    promo_probability: float = 0.12
    promo_lift: float = 0.35
    holiday_lift: float = 0.25
    temp_sensitivity: float = 0.03  # +3 % demand per degC above the seasonal norm
    noise_cv: float = 0.08


# --------------------------------------------------------------------------------------
# Weekly demand history (forecasting tabs)
# --------------------------------------------------------------------------------------
def climatology(dates: pd.DatetimeIndex | pd.Series) -> np.ndarray:
    """Seasonal 'normal' temperature (degC) for each date - peaks in late July."""
    doy = pd.DatetimeIndex(dates).dayofyear.to_numpy()
    return 15.0 + 10.0 * np.sin(2 * np.pi * (doy - 110) / 365.25)


def iso_week(dates: pd.DatetimeIndex | pd.Series) -> np.ndarray:
    return pd.DatetimeIndex(dates).isocalendar().week.to_numpy().astype(int)


def _expected_demand(t_idx: np.ndarray, dates: pd.DatetimeIndex, promo: np.ndarray,
                     holiday: np.ndarray, temp: np.ndarray, p: DemandParams) -> np.ndarray:
    woy = iso_week(dates)
    seasonal = 1.0 + p.seasonal_amplitude * np.sin(2 * np.pi * (woy - 13) / 52.0)
    level = p.base_level + p.trend_per_week * t_idx
    temp_effect = 1.0 + p.temp_sensitivity * (temp - climatology(dates))
    return level * seasonal * (1 + p.promo_lift * promo) * (1 + p.holiday_lift * holiday) * temp_effect


def _simulate_block(dates: pd.DatetimeIndex, t_idx: np.ndarray, p: DemandParams,
                    rng: np.random.Generator) -> pd.DataFrame:
    n = len(dates)
    promo = (rng.random(n) < p.promo_probability).astype(int)
    holiday = np.isin(iso_week(dates), list(HOLIDAY_WEEKS)).astype(int)
    temp = climatology(dates) + rng.normal(0, 3.0, n)
    mu = _expected_demand(t_idx, dates, promo, holiday, temp, p)
    demand = np.clip(np.round(rng.normal(mu, p.noise_cv * mu)), 0, None)
    return pd.DataFrame({
        "date": dates,
        "demand": demand.astype(float),
        "promotion": promo,
        "holiday": holiday,
        "temperature": np.round(temp, 1),
    })


def generate_demand_history(n_weeks: int = 156, params: DemandParams = DemandParams(),
                            seed: int = 42, start: str = "2023-01-02") -> pd.DataFrame:
    """Generate ``n_weeks`` of weekly demand with trend, seasonality, promos and weather."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start, periods=n_weeks, freq="W-MON")
    return _simulate_block(dates, np.arange(n_weeks), params, rng)


def simulate_new_weeks(history: pd.DataFrame, n_new: int, params: DemandParams = DemandParams(),
                       seed: int = 0) -> pd.DataFrame:
    """Simulate the next ``n_new`` weeks *arriving* after the last row of ``history``."""
    rng = np.random.default_rng(seed + 7919 * len(history))
    last = pd.Timestamp(history["date"].iloc[-1])
    dates = pd.date_range(last + pd.DateOffset(weeks=1), periods=n_new, freq="W-MON")
    t_idx = np.arange(len(history), len(history) + n_new)
    return _simulate_block(dates, t_idx, params, rng)


def future_exogenous(last_date: pd.Timestamp, horizon: int,
                     promo_weeks: set[int] | None = None) -> pd.DataFrame:
    """Known-in-advance drivers for the forecast horizon.

    ``promo_weeks`` holds 1-based horizon steps with a planned promotion. Temperature is
    set to the seasonal norm (a weather forecast would be plugged in here in production).
    """
    promo_weeks = promo_weeks or set()
    dates = pd.date_range(pd.Timestamp(last_date) + pd.DateOffset(weeks=1), periods=horizon, freq="W-MON")
    return pd.DataFrame({
        "date": dates,
        "promotion": [1 if (i + 1) in promo_weeks else 0 for i in range(horizon)],
        "holiday": np.isin(iso_week(dates), list(HOLIDAY_WEEKS)).astype(int),
        "temperature": np.round(climatology(dates), 1),
    })


def validate_demand_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Clean a user-supplied frame. Requires ``date`` and ``demand``; fills optional drivers."""
    df = raw.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(missing)}. "
                         f"Expected at least: {', '.join(REQUIRED_COLUMNS)}.")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["demand"] = pd.to_numeric(df["demand"], errors="coerce")
    df = df.dropna(subset=["date", "demand"])
    if df.empty:
        raise ValueError("No valid rows after parsing dates and demand values.")
    df = df.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    if "promotion" in df.columns:
        df["promotion"] = pd.to_numeric(df["promotion"], errors="coerce").fillna(0).clip(0, 1).astype(int)
    else:
        df["promotion"] = 0
    if "holiday" in df.columns:
        df["holiday"] = pd.to_numeric(df["holiday"], errors="coerce").fillna(0).clip(0, 1).astype(int)
    else:  # infer from the calendar
        df["holiday"] = np.isin(iso_week(df["date"]), list(HOLIDAY_WEEKS)).astype(int)
    if "temperature" in df.columns:
        temp = pd.to_numeric(df["temperature"], errors="coerce")
        df["temperature"] = temp.fillna(pd.Series(climatology(df["date"]), index=df.index))
    else:
        df["temperature"] = np.round(climatology(df["date"]), 1)
    df["demand"] = df["demand"].clip(lower=0).astype(float)
    if len(df) < 30:
        raise ValueError(f"Need at least 30 weekly observations to train models (got {len(df)}).")
    return df[["date", "demand", "promotion", "holiday", "temperature"]]


# --------------------------------------------------------------------------------------
# End-customer demand for the bullwhip simulation
# --------------------------------------------------------------------------------------
DEMAND_PATTERNS = {
    "poisson": "Stationary Poisson (lab slide)",
    "seasonal_promo": "Seasonal + promotions",
    "step": "Step change (+50 % shock)",
}


def generate_customer_demand(n_weeks: int, mean: float = 20.0, pattern: str = "poisson",
                             seed: int = 7, step_week: int | None = None) -> pd.DataFrame:
    """End-customer weekly demand. Returns columns ``week``, ``demand``, ``promotion``."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_weeks)
    promo = np.zeros(n_weeks, dtype=int)
    if pattern == "poisson":
        lam = np.full(n_weeks, mean, dtype=float)
    elif pattern == "seasonal_promo":
        promo = (rng.random(n_weeks) < 0.10).astype(int)
        lam = mean * (1 + 0.30 * np.sin(2 * np.pi * t / 52.0)) * (1 + 0.6 * promo)
    elif pattern == "step":
        k = step_week if step_week is not None else n_weeks // 2
        lam = np.where(t >= k, 1.5 * mean, mean)
    else:
        raise ValueError(f"Unknown demand pattern: {pattern}")
    return pd.DataFrame({"week": t, "demand": rng.poisson(lam).astype(float), "promotion": promo})
