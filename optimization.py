"""Gaussian-process Bayesian optimisation (minimisation) with Expected Improvement.

Implemented from first principles on scikit-learn's GP so every step is inspectable:

1. Latin-hypercube initial design.
2. Fit GP surrogate (Matern 5/2 + white noise) on normalised inputs / standardised outputs.
3. Maximise Expected Improvement: dense random candidates + L-BFGS-B polish.
4. Evaluate the true (expensive, noisy) objective at the proposal; repeat.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm, qmc
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel


@dataclass
class BOResult:
    X: np.ndarray                 # evaluated points (original scale)
    y: np.ndarray                 # objective values
    info: list[dict]              # extra outputs returned by the objective
    bounds: np.ndarray
    gp: GaussianProcessRegressor
    y_mean: float
    y_std: float
    names: list[str] = field(default_factory=list)
    log_y: bool = True

    @property
    def best_index(self) -> int:
        return int(np.argmin(self.y))

    @property
    def best_x(self) -> np.ndarray:
        return self.X[self.best_index]

    @property
    def best_y(self) -> float:
        return float(self.y[self.best_index])

    def best_so_far(self) -> np.ndarray:
        return np.minimum.accumulate(self.y)

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Posterior median / std of the surrogate, mapped back to the original objective scale."""
        U = (np.atleast_2d(X) - self.bounds[:, 0]) / (self.bounds[:, 1] - self.bounds[:, 0])
        mu, sd = self.gp.predict(U, return_std=True)
        mu, sd = mu * self.y_std + self.y_mean, sd * self.y_std
        if self.log_y:
            return np.exp(mu), np.exp(mu) * sd   # delta-method std
        return mu, sd


def expected_improvement(mu: np.ndarray, sd: np.ndarray, best: float, xi: float = 0.01) -> np.ndarray:
    sd = np.maximum(sd, 1e-12)
    imp = best - mu - xi
    z = imp / sd
    return imp * norm.cdf(z) + sd * norm.pdf(z)


def bayesian_optimize(objective: Callable[[np.ndarray], tuple[float, dict]], bounds,
                      n_init: int = 8, n_iter: int = 22, xi: float = 0.01, seed: int = 0,
                      names: list[str] | None = None, x0: list | None = None, log_y: bool = True,
                      callback: Callable[[int, int, float], None] | None = None) -> BOResult:
    """Minimise ``objective(x) -> (value, info_dict)`` over the box ``bounds`` [(lo, hi), ...].

    ``x0`` optionally warm-starts the search with known good points (e.g. the textbook
    ROP/EOQ solution), so the optimiser can only improve on them.
    ``log_y`` fits the GP to log(cost): penalty cliffs at infeasible corners otherwise
    dominate the standardised targets and flatten the region near the optimum.
    """
    bounds = np.asarray(bounds, dtype=float)
    dim = len(bounds)
    rng = np.random.default_rng(seed)
    lo, span = bounds[:, 0], bounds[:, 1] - bounds[:, 0]

    U = qmc.LatinHypercube(d=dim, seed=seed).random(n_init)
    if x0:
        U = np.vstack([np.clip((np.asarray(x0, float) - lo) / span, 0, 1), U])
    X_list, y_list, info_list = [], [], []
    total = len(U) + n_iter

    def evaluate(u):
        x = lo + u * span
        val, info = objective(x)
        X_list.append(x)
        y_list.append(float(val))
        info_list.append(info)
        if callback:
            callback(len(y_list), total, float(min(y_list)))

    for u in U:
        evaluate(u)

    kernel = (ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=[0.3] * dim,
                                                       length_scale_bounds=(1e-2, 1e2), nu=2.5)
              + WhiteKernel(1e-3, (1e-9, 1e-1)))
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=False, n_restarts_optimizer=3,
                                  random_state=seed)

    def transformed():
        y = np.asarray(y_list)
        return np.log(np.maximum(y, 1e-9)) if log_y else y

    def fit_gp():
        y = transformed()
        y_mean, y_std = y.mean(), y.std() or 1.0
        Uobs = (np.asarray(X_list) - lo) / span
        with warnings.catch_warnings():  # hyper-parameter bound hits are harmless here
            warnings.simplefilter("ignore", ConvergenceWarning)
            gp.fit(Uobs, (y - y_mean) / y_std)
        return y_mean, y_std

    for _ in range(n_iter):
        y_mean, y_std = fit_gp()
        best = (transformed().min() - y_mean) / y_std
        cand = rng.random((4000, dim))
        mu, sd = gp.predict(cand, return_std=True)
        ei = expected_improvement(mu, sd, best, xi)
        starts = cand[np.argsort(ei)[-3:]]
        best_u, best_ei = starts[-1], ei.max()
        for s0 in starts:  # polish the most promising candidates
            res = minimize(lambda u: -expected_improvement(*gp.predict(u[None], return_std=True), best, xi)[0],
                           s0, bounds=[(0, 1)] * dim, method="L-BFGS-B")
            if res.success and -res.fun > best_ei:
                best_u, best_ei = res.x, -res.fun
        evaluate(np.clip(best_u, 0, 1))

    y_mean, y_std = fit_gp()
    return BOResult(np.asarray(X_list), np.asarray(y_list), info_list, bounds, gp,
                    float(y_mean), float(y_std), names or [f"x{i}" for i in range(dim)], log_y)
