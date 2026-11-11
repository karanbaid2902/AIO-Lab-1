"""ML demand forecasting with continuous (walk-forward) updating.

Models share one interface::

    model.fit(history_df)                       # (re)train on all data seen so far
    model.predict_next(history_df, exog_row)    # one-step-ahead forecast
    model.forecast(history_df, future_exog)     # recursive multi-step forecast

Drivers used: lags / rolling means of demand (time-series signal), week-of-year
seasonality, promotions, holidays and an external factor (temperature).
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# LSTM is optional - PyTorch is imported lazily (only when the LSTM is used) to keep the
# dashboard start-up fast, and the model list degrades gracefully without it.
TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None

EXOG_COLUMNS = ["promotion", "holiday", "temperature"]
FEATURES = ["trend", "woy_sin", "woy_cos", "promotion", "holiday", "temperature",
            "lag_1", "lag_2", "lag_3", "lag_4", "roll_mean_4", "roll_mean_12"]

MODEL_INFO = {
    "Naive (last week)": "Baseline: next week = this week (the notebook's naive forecast).",
    "Moving average (4 wk)": "Baseline: average of the last 4 weeks.",
    "Holt-Winters (time series)": "Classical exponential smoothing with additive trend and 52-week "
                                  "seasonality (history only - ignores promotions and weather).",
    "SARIMAX (time series + drivers)": "ARIMA(1,0,1) errors with Fourier seasonality, trend, promotions, "
                                       "holidays and temperature as exogenous regressors.",
    "Ridge regression": "Regularised linear regression on lags, seasonality, promotions and weather.",
    "Random Forest": "Ensemble of decision trees (as in the lab notebook) on the same features.",
    "Gradient Boosting": "Boosted shallow trees - strong non-linear tabular learner.",
    "LSTM (PyTorch)": "Recurrent neural network over the last 8 weeks + next week's known drivers.",
}


def available_models() -> list[str]:
    names = list(MODEL_INFO)
    return names if TORCH_AVAILABLE else [n for n in names if not n.startswith("LSTM")]


# --------------------------------------------------------------------------------------
# Feature engineering
# --------------------------------------------------------------------------------------
def make_features(df: pd.DataFrame) -> pd.DataFrame:
    dates = pd.DatetimeIndex(df["date"])
    woy = dates.isocalendar().week.to_numpy().astype(float)
    y = df["demand"].astype(float).reset_index(drop=True)
    out = pd.DataFrame({
        "trend": np.arange(len(df), dtype=float),
        "woy_sin": np.sin(2 * np.pi * woy / 52.0),
        "woy_cos": np.cos(2 * np.pi * woy / 52.0),
        "promotion": df["promotion"].to_numpy(float),
        "holiday": df["holiday"].to_numpy(float),
        "temperature": df["temperature"].to_numpy(float),
    })
    for k in (1, 2, 3, 4):
        out[f"lag_{k}"] = y.shift(k)
    out["roll_mean_4"] = y.shift(1).rolling(4).mean()
    out["roll_mean_12"] = y.shift(1).rolling(12, min_periods=4).mean()
    return out[FEATURES]


def _append_exog(history: pd.DataFrame, exog_row: pd.DataFrame | pd.Series) -> pd.DataFrame:
    row = exog_row.to_frame().T if isinstance(exog_row, pd.Series) else exog_row
    row = row[["date"] + EXOG_COLUMNS].copy()
    row["demand"] = np.nan
    return pd.concat([history[["date", "demand"] + EXOG_COLUMNS], row], ignore_index=True)


# --------------------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------------------
class BaseForecaster:
    name = "base"

    def fit(self, history: pd.DataFrame) -> "BaseForecaster":
        return self

    def predict_next(self, history: pd.DataFrame, exog_row) -> float:
        raise NotImplementedError

    def forecast(self, history: pd.DataFrame, future_exog: pd.DataFrame) -> np.ndarray:
        """Recursive multi-step forecast: each prediction becomes a lag for the next step."""
        hist = history[["date", "demand"] + EXOG_COLUMNS].copy()
        preds = []
        for i in range(len(future_exog)):
            row = future_exog.iloc[[i]]
            p = self.predict_next(hist, row)
            preds.append(p)
            new = row[["date"] + EXOG_COLUMNS].copy()
            new["demand"] = p
            hist = pd.concat([hist, new], ignore_index=True)
        return np.asarray(preds)


class NaiveForecaster(BaseForecaster):
    name = "Naive (last week)"

    def predict_next(self, history, exog_row) -> float:
        return float(history["demand"].iloc[-1])


class MovingAverageForecaster(BaseForecaster):
    name = "Moving average (4 wk)"

    def __init__(self, window: int = 4):
        self.window = window

    def predict_next(self, history, exog_row) -> float:
        return float(history["demand"].iloc[-self.window:].mean())


class TabularForecaster(BaseForecaster):
    def __init__(self, name: str, seed: int = 0):
        self.name = name
        if name == "Ridge regression":
            self.model = make_pipeline(StandardScaler(), Ridge(alpha=3.0))
        elif name == "Random Forest":
            self.model = RandomForestRegressor(n_estimators=150, min_samples_leaf=2,
                                               random_state=seed, n_jobs=-1)
        elif name == "Gradient Boosting":
            self.model = GradientBoostingRegressor(n_estimators=250, max_depth=3, learning_rate=0.05,
                                                   subsample=0.85, random_state=seed)
        else:
            raise ValueError(name)

    def fit(self, history):
        X = make_features(history)
        mask = X.notna().all(axis=1).to_numpy()
        self.model.fit(X[mask], history["demand"].to_numpy(float)[mask])
        return self

    def predict_next(self, history, exog_row) -> float:
        X = make_features(_append_exog(history, exog_row)).iloc[[-1]]
        return float(max(0.0, self.model.predict(X)[0]))

    def explain(self) -> pd.DataFrame:
        """Feature importances (trees) or standardised coefficients (Ridge)."""
        if hasattr(self.model, "feature_importances_"):
            vals = self.model.feature_importances_
        else:
            vals = np.abs(self.model[-1].coef_)
            vals = vals / vals.sum()
        return (pd.DataFrame({"feature": FEATURES, "importance": vals})
                .sort_values("importance", ascending=True))


class HoltWintersForecaster(BaseForecaster):
    """Additive Holt-Winters. Parameters are estimated at each retrain; between retrains the
    level/trend/season states are updated with the frozen parameters as new weeks arrive."""

    name = "Holt-Winters (time series)"
    season = 52

    def _model(self, y: np.ndarray):
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        seasonal = "add" if len(y) >= 2 * self.season + 10 else None  # needs two full cycles
        return ExponentialSmoothing(y, trend="add", damped_trend=True, seasonal=seasonal,
                                    seasonal_periods=self.season if seasonal else None,
                                    initialization_method="heuristic" if seasonal else "estimated")

    def fit(self, history):
        import warnings
        y = history["demand"].to_numpy(float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = self._model(y).fit(optimized=True)
        self.params = {k: res.params[k] for k in ("smoothing_level", "smoothing_trend",
                                                  "smoothing_seasonal", "damping_trend")
                       if res.params.get(k) is not None and not np.isnan(res.params[k])}
        return self

    def _refilter(self, history):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return self._model(history["demand"].to_numpy(float)).fit(optimized=False, **self.params)

    def predict_next(self, history, exog_row) -> float:
        return float(max(0.0, self._refilter(history).forecast(1)[0]))

    def forecast(self, history, future_exog):
        return np.clip(self._refilter(history).forecast(len(future_exog)), 0, None)


class SARIMAXForecaster(BaseForecaster):
    """Regression with ARIMA(1,0,1) errors; drivers enter as exogenous regressors."""

    name = "SARIMAX (time series + drivers)"

    @staticmethod
    def _exog(df: pd.DataFrame, t0: int = 0) -> np.ndarray:
        woy = pd.DatetimeIndex(df["date"]).isocalendar().week.to_numpy().astype(float)
        t = np.arange(t0, t0 + len(df), dtype=float)
        cols = [t / 52.0, df["promotion"].to_numpy(float), df["holiday"].to_numpy(float),
                (df["temperature"].to_numpy(float) - 15.0) / 10.0]
        for k in (1, 2):  # Fourier terms for yearly seasonality
            cols += [np.sin(2 * np.pi * k * woy / 52.0), np.cos(2 * np.pi * k * woy / 52.0)]
        return np.column_stack(cols)

    def fit(self, history):
        import warnings
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.res = SARIMAX(history["demand"].to_numpy(float), exog=self._exog(history),
                               order=(1, 0, 1), trend="c").fit(disp=False, maxiter=200)
        return self

    def _applied(self, history):
        import warnings
        with warnings.catch_warnings():  # re-filter new data with the estimated parameters
            warnings.simplefilter("ignore")
            return self.res.apply(history["demand"].to_numpy(float), exog=self._exog(history))

    def predict_next(self, history, exog_row) -> float:
        row = exog_row if isinstance(exog_row, pd.DataFrame) else exog_row.to_frame().T
        nxt = self._exog(row, t0=len(history))
        return float(max(0.0, self._applied(history).forecast(1, exog=nxt)[0]))

    def forecast(self, history, future_exog):
        fut = self._exog(future_exog, t0=len(history))
        return np.clip(self._applied(history).forecast(len(future_exog), exog=fut), 0, None)

    def explain(self) -> pd.DataFrame:
        names = ["const", "trend", "promotion", "holiday", "temperature",
                 "season_sin_1", "season_cos_1", "season_sin_2", "season_cos_2"]
        p = self.res.params[:len(names)]
        se = self.res.bse[:len(names)]
        out = pd.DataFrame({"feature": names, "importance": np.abs(p / np.where(se > 0, se, np.nan))})
        return out[out["feature"] != "const"].sort_values("importance")


def _torch():
    import torch
    from torch import nn

    class _LSTMNet(nn.Module):
        def __init__(self, n_seq: int, n_exog: int, hidden: int = 32):
            super().__init__()
            self.lstm = nn.LSTM(n_seq, hidden, batch_first=True)
            self.head = nn.Sequential(nn.Linear(hidden + n_exog, 16), nn.ReLU(), nn.Linear(16, 1))

        def forward(self, seq, exog):
            _, (h, _) = self.lstm(seq)
            return self.head(torch.cat([h[-1], exog], dim=1)).squeeze(-1)

    return torch, nn, _LSTMNet


class LSTMForecaster(BaseForecaster):
    """Sequence model; supports warm-start fine-tuning when new data arrives (online learning)."""

    name = "LSTM (PyTorch)"

    def __init__(self, seed: int = 0, window: int = 8, hidden: int = 32, epochs: int = 300,
                 finetune_epochs: int = 80, lr: float = 0.01):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is not installed.")
        self.torch, self.nn, self._Net = _torch()
        self.seed, self.window, self.hidden = seed, window, hidden
        self.epochs, self.finetune_epochs, self.lr = epochs, finetune_epochs, lr
        self.net = None

    def _arrays(self, df: pd.DataFrame):
        dates = pd.DatetimeIndex(df["date"])
        woy = dates.isocalendar().week.to_numpy().astype(float)
        exog = np.column_stack([
            df["promotion"].to_numpy(float), df["holiday"].to_numpy(float),
            (df["temperature"].to_numpy(float) - 15.0) / 10.0,
            np.sin(2 * np.pi * woy / 52.0), np.cos(2 * np.pi * woy / 52.0),
        ])
        y_s = (df["demand"].to_numpy(float) - self.y_mean) / self.y_std
        return y_s, exog, np.column_stack([y_s, exog])

    def fit(self, history):
        torch, nn = self.torch, self.nn
        warm = self.net is not None
        if not warm:
            torch.manual_seed(self.seed)
            self.y_mean = float(history["demand"].mean())
            self.y_std = float(history["demand"].std() or 1.0)
            self.net = self._Net(6, 5, self.hidden)
            self.opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        y_s, exog, seq = self._arrays(history)
        idx = np.arange(self.window, len(history))
        X_seq = torch.tensor(np.stack([seq[i - self.window:i] for i in idx]), dtype=torch.float32)
        X_ex = torch.tensor(exog[idx], dtype=torch.float32)
        y = torch.tensor(y_s[idx], dtype=torch.float32)
        self.net.train()
        for _ in range(self.finetune_epochs if warm else self.epochs):
            self.opt.zero_grad()
            loss = nn.functional.mse_loss(self.net(X_seq, X_ex), y)
            loss.backward()
            self.opt.step()
        return self

    def predict_next(self, history, exog_row) -> float:
        tmp = _append_exog(history, exog_row)
        tmp["demand"] = tmp["demand"].fillna(0.0)
        _, exog, seq = self._arrays(tmp)
        torch = self.torch
        self.net.eval()
        with torch.no_grad():
            s = torch.tensor(seq[-self.window - 1:-1][None], dtype=torch.float32)
            e = torch.tensor(exog[-1:], dtype=torch.float32)
            pred = float(self.net(s, e).item()) * self.y_std + self.y_mean
        return max(0.0, pred)


def build_model(name: str, seed: int = 0) -> BaseForecaster:
    if name == "Naive (last week)":
        return NaiveForecaster()
    if name == "Moving average (4 wk)":
        return MovingAverageForecaster(4)
    if name == "LSTM (PyTorch)":
        return LSTMForecaster(seed=seed)
    if name == "Holt-Winters (time series)":
        return HoltWintersForecaster()
    if name == "SARIMAX (time series + drivers)":
        return SARIMAXForecaster()
    return TabularForecaster(name, seed=seed)


# --------------------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------------------
def forecast_metrics(actual, pred) -> dict[str, float]:
    a, p = np.asarray(actual, float), np.asarray(pred, float)
    err = a - p
    nz = a != 0
    return {
        "MAPE %": float(np.mean(np.abs(err[nz] / a[nz])) * 100),
        "WAPE %": float(np.abs(err).sum() / max(np.abs(a).sum(), 1e-9) * 100),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "Bias %": float(-err.sum() / max(a.sum(), 1e-9) * 100),   # + = over-forecast
    }


def walk_forward(df: pd.DataFrame, model_name: str, n_test: int = 26, refit_every: int = 4,
                 seed: int = 0) -> pd.DataFrame:
    """Simulate continuous operation: each week the model forecasts the next week using all
    data available so far; it is re-trained every ``refit_every`` weeks as new data arrives
    (the LSTM fine-tunes its existing weights instead of starting from scratch)."""
    df = df.reset_index(drop=True)
    n = len(df)
    n_test = int(min(n_test, n - 20))
    model = build_model(model_name, seed)
    rows = []
    for k, idx in enumerate(range(n - n_test, n)):
        hist = df.iloc[:idx]
        refit = k % refit_every == 0
        if refit:
            model.fit(hist)
        pred = model.predict_next(hist, df.iloc[[idx]])
        rows.append({"date": df["date"].iloc[idx], "actual": df["demand"].iloc[idx],
                     "forecast": pred, "retrained": refit, "n_train": idx})
    out = pd.DataFrame(rows)
    out["error"] = out["actual"] - out["forecast"]
    return out
