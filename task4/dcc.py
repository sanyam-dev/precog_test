"""
DCC-GARCH(1,1) with strictly backward-looking rolling-window estimation.

Two-step Engle (2002) DCC:
  1. Univariate variance-targeting GARCH(1,1) on each asset.
  2. Correlation-targeting DCC(1,1) on the standardized residuals.

At origin t the model is fit on returns in [t - window, t) only, then walked
forward with frozen parameters. R_t therefore uses information through t-1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize


DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "anonymized_data"

# Work in percent returns so GARCH scales like typical daily equity series.
_RETURN_SCALE = 100.0
_A_B_MAX = 0.999
_RIDGE = 1e-8
_EPS_CLIP = 4.0
_QBAR_SHRINK = 0.10


def load_panel(data_dir: Path | str | None = None) -> pd.DataFrame:
    """Load the 100-asset OHLCV panel (MultiIndex columns: Asset, Metric)."""
    data_dir = Path(data_dir) if data_dir is not None else DATA_DIR
    frames = {}
    for i in range(1, 101):
        path = data_dir / f"Asset_{i:03d}.csv"
        df = pd.read_csv(path, parse_dates=["Date"]).set_index("Date")
        frames[f"Asset_{i}"] = df
    panel = pd.concat(frames, axis=1)
    panel.columns.names = ["Asset", "Metric"]
    panel = panel.sort_index()
    return panel


def log_returns(panel: pd.DataFrame) -> pd.DataFrame:
    close = panel.xs("Close", axis=1, level="Metric")
    return np.log(close / close.shift(1)).dropna(how="all")


def _to_corr(S: np.ndarray, ridge: float = _RIDGE) -> np.ndarray:
    S = 0.5 * (S + S.T)
    S = S + ridge * np.eye(S.shape[0])
    d = np.sqrt(np.clip(np.diag(S), _RIDGE, None))
    R = S / np.outer(d, d)
    R = np.clip(R, -0.999999, 0.999999)
    np.fill_diagonal(R, 1.0)
    return 0.5 * (R + R.T)


def _shrink_cov(eps: np.ndarray, delta: float = _QBAR_SHRINK) -> np.ndarray:
    S = np.cov(eps, rowvar=False)
    n = S.shape[0]
    mu = float(np.trace(S) / n)
    delta = float(np.clip(delta, 0.0, 1.0))
    return (1.0 - delta) * S + delta * mu * np.eye(n)


def _garch_variance(r: np.ndarray, omega: float, alpha: float, beta: float) -> np.ndarray:
    T = r.shape[0]
    var = np.empty(T, dtype=float)
    var0 = omega / max(1.0 - alpha - beta, 1e-8)
    var[0] = max(var0, _RIDGE)
    r2 = r * r
    for t in range(1, T):
        var[t] = omega + alpha * r2[t - 1] + beta * var[t - 1]
    return np.maximum(var, _RIDGE)


def _garch_nll(params: np.ndarray, r: np.ndarray, var_target: float) -> float:
    alpha, beta = params
    if alpha < 0.0 or beta < 0.0 or alpha + beta >= _A_B_MAX:
        return 1e12
    omega = var_target * (1.0 - alpha - beta)
    var = _garch_variance(r, omega, alpha, beta)
    if not np.isfinite(var).all():
        return 1e12
    return 0.5 * float(np.sum(np.log(var) + (r * r) / var))


def fit_garch11(
    r: np.ndarray,
    x0: tuple[float, float] = (0.05, 0.90),
) -> dict:
    """Variance-targeting GARCH(1,1). `r` is already demeaned, in percent."""
    r = np.asarray(r, dtype=float)
    var_target = float(np.mean(r * r))
    var_target = max(var_target, _RIDGE)

    res = minimize(
        _garch_nll,
        x0=np.array(x0, dtype=float),
        args=(r, var_target),
        method="L-BFGS-B",
        bounds=((1e-8, 0.35), (1e-8, 0.989)),
        options={"maxiter": 50, "ftol": 1e-7},
    )
    alpha, beta = (res.x if np.isfinite(res.x).all() else np.array(x0, dtype=float))
    alpha = float(np.clip(alpha, 1e-8, 0.35))
    beta = float(np.clip(beta, 1e-8, 0.989))
    if alpha + beta >= _A_B_MAX:
        alpha, beta = 0.05, 0.90
    omega = var_target * (1.0 - alpha - beta)
    var = _garch_variance(r, omega, alpha, beta)
    sigma = np.sqrt(var)
    eps = r / sigma
    return {
        "omega": omega,
        "alpha": alpha,
        "beta": beta,
        "sigma": sigma,
        "eps": eps,
        "success": bool(res.success),
    }


def _dcc_filter_RQ(
    eps: np.ndarray,
    a: float,
    b: float,
    Qbar: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (R[t], last Q). R[t] uses eps[t-1] (causal)."""
    T, n = eps.shape
    R = np.empty((T, n, n), dtype=float)
    Q = Qbar.copy()
    ab = 1.0 - a - b
    for t in range(T):
        if t > 0:
            e = eps[t - 1]
            Q = ab * Qbar + a * np.outer(e, e) + b * Q
        R[t] = _to_corr(Q)
    return R, Q


def _dcc_composite_nll(params: np.ndarray, eps: np.ndarray, Qbar: np.ndarray, iu) -> float:
    """Pairwise composite DCC likelihood (stable for large n)."""
    a, b = params
    if a < 0.0 or b < 0.0 or a + b >= _A_B_MAX:
        return 1e12
    T = eps.shape[0]
    Q = Qbar.copy()
    ab = 1.0 - a - b
    nll = 0.0
    for t in range(T):
        if t > 0:
            e_prev = eps[t - 1]
            Q = ab * Qbar + a * np.outer(e_prev, e_prev) + b * Q
        d = np.sqrt(np.clip(np.diag(Q), _RIDGE, None))
        rho = np.clip((Q / np.outer(d, d))[iu], -0.999, 0.999)
        e = eps[t]
        e_i = e[iu[0]]
        e_j = e[iu[1]]
        one_m = 1.0 - rho * rho
        quad = (e_i * e_i + e_j * e_j - 2.0 * rho * e_i * e_j) / one_m
        nll += 0.5 * float(np.sum(np.log(one_m) + quad))
        if not np.isfinite(nll):
            return 1e12
    return nll


def fit_dcc(
    eps: np.ndarray,
    x0: tuple[float, float] = (0.05, 0.40),
) -> dict:
    """Correlation-targeting DCC(1,1) on standardized residuals (T x n)."""
    eps = np.clip(np.asarray(eps, dtype=float), -_EPS_CLIP, _EPS_CLIP)
    Qbar = _shrink_cov(eps)
    n = eps.shape[1]
    iu = np.triu_indices(n, k=1)
    res = minimize(
        _dcc_composite_nll,
        x0=np.array(x0, dtype=float),
        args=(eps, Qbar, iu),
        method="L-BFGS-B",
        bounds=((0.0, 0.25), (0.0, 0.989)),
        options={"maxiter": 15, "ftol": 1e-5},
    )
    if np.isfinite(res.fun) and np.isfinite(res.x).all():
        a, b = float(res.x[0]), float(res.x[1])
        success = bool(res.success)
        nll = float(res.fun)
    else:
        a, b = 0.0, 0.0
        success = False
        nll = np.nan
    if a + b >= _A_B_MAX:
        a, b = 0.02, 0.95
    R, Q_last = _dcc_filter_RQ(eps, a, b, Qbar)
    return {
        "a": a,
        "b": b,
        "Qbar": Qbar,
        "R": R,
        "Q_last": Q_last,
        "success": success,
        "nll": nll,
    }


def _mean_offdiag(R: np.ndarray) -> float:
    n = R.shape[0]
    return float((R.sum() - np.trace(R)) / (n * (n - 1)))


@dataclass
class DCCState:
    """Frozen-parameter filter that can be walked forward one day at a time."""

    omega: np.ndarray
    garch_a: np.ndarray
    garch_b: np.ndarray
    dcc_a: float
    dcc_b: float
    Qbar: np.ndarray
    mean: np.ndarray
    last_r: np.ndarray
    last_var: np.ndarray
    last_eps: np.ndarray
    last_Q: np.ndarray

    def forecast_R(self) -> np.ndarray:
        """One-step-ahead correlation using information through last_r."""
        Q_next = (
            (1.0 - self.dcc_a - self.dcc_b) * self.Qbar
            + self.dcc_a * np.outer(self.last_eps, self.last_eps)
            + self.dcc_b * self.last_Q
        )
        return _to_corr(Q_next)

    def update(self, r_pct: np.ndarray) -> np.ndarray:
        """
        Incorporate a newly observed (demeaned, percent) return vector and
        return the next one-step-ahead R.
        """
        var_next = self.omega + self.garch_a * (self.last_r ** 2) + self.garch_b * self.last_var
        var_next = np.maximum(var_next, _RIDGE)
        Q_next = (
            (1.0 - self.dcc_a - self.dcc_b) * self.Qbar
            + self.dcc_a * np.outer(self.last_eps, self.last_eps)
            + self.dcc_b * self.last_Q
        )
        eps_new = np.clip(r_pct / np.sqrt(var_next), -_EPS_CLIP, _EPS_CLIP)
        self.last_r = r_pct
        self.last_var = var_next
        self.last_eps = eps_new
        self.last_Q = Q_next
        return self.forecast_R()


def fit_state(returns_pct: np.ndarray, x0_garch=None, x0_dcc=None) -> DCCState:
    """Fit GARCH + DCC on a backward window of demeaned percent returns (T x n)."""
    T, n = returns_pct.shape
    omega = np.empty(n)
    garch_a = np.empty(n)
    garch_b = np.empty(n)
    eps = np.empty_like(returns_pct)
    last_var = np.empty(n)

    for i in range(n):
        start = (0.05, 0.90) if x0_garch is None else (float(x0_garch[0][i]), float(x0_garch[1][i]))
        fit = fit_garch11(returns_pct[:, i], x0=start)
        omega[i] = fit["omega"]
        garch_a[i] = fit["alpha"]
        garch_b[i] = fit["beta"]
        eps[:, i] = fit["eps"]
        last_var[i] = fit["sigma"][-1] ** 2

    dcc = fit_dcc(eps, x0=(0.05, 0.40) if x0_dcc is None else x0_dcc)
    return DCCState(
        omega=omega,
        garch_a=garch_a,
        garch_b=garch_b,
        dcc_a=dcc["a"],
        dcc_b=dcc["b"],
        Qbar=dcc["Qbar"],
        mean=np.zeros(n),
        last_r=returns_pct[-1],
        last_var=last_var,
        last_eps=np.clip(eps[-1], -_EPS_CLIP, _EPS_CLIP),
        last_Q=dcc["Q_last"],
    )


@dataclass
class RollingDCCResult:
    dates: pd.DatetimeIndex
    assets: list[str]
    R: np.ndarray
    params: pd.DataFrame
    window: int
    step: int
    extra: dict = field(default_factory=dict)

    def pair(self, a: str, b: str) -> pd.Series:
        i = self.assets.index(a)
        j = self.assets.index(b)
        return pd.Series(self.R[:, i, j], index=self.dates, name=f"{a}__{b}")

    def mean_corr(self) -> pd.Series:
        vals = np.array([_mean_offdiag(self.R[t]) for t in range(self.R.shape[0])])
        return pd.Series(vals, index=self.dates, name="mean_dcc_corr")

    def matrix_on(self, date) -> pd.DataFrame:
        loc = self.dates.get_indexer([pd.Timestamp(date)], method="ffill")[0]
        return pd.DataFrame(self.R[loc], index=self.assets, columns=self.assets)

    def save(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            dates=self.dates.to_numpy().astype("datetime64[ns]"),
            assets=np.array(self.assets),
            R=self.R.astype(np.float32),
            window=np.array([self.window]),
            step=np.array([self.step]),
        )
        self.params.to_csv(path.with_name(path.stem + "_params.csv"), index=False)
        self.mean_corr().to_csv(path.with_name(path.stem + "_mean_corr.csv"), header=True)
        return path

    @classmethod
    def load(cls, path: Path | str) -> "RollingDCCResult":
        path = Path(path)
        data = np.load(path, allow_pickle=True)
        dates = pd.DatetimeIndex(data["dates"], name="Date")
        assets = [str(a) for a in data["assets"].tolist()]
        params_path = path.with_name(path.stem + "_params.csv")
        params = (
            pd.read_csv(params_path, parse_dates=["refit_date", "forecast_from"])
            if params_path.exists()
            else pd.DataFrame()
        )
        return cls(
            dates=dates,
            assets=assets,
            R=data["R"].astype(np.float64),
            params=params,
            window=int(data["window"][0]),
            step=int(data["step"][0]),
        )


def rolling_dcc(
    returns: pd.DataFrame,
    window: int = 252,
    step: int = 21,
    n_assets: int | None = None,
    verbose: bool = True,
) -> RollingDCCResult:
    """
    Estimate DCC on a rolling backward window and record one-step-ahead R_t.

    Parameters
    ----------
    returns : DataFrame
        Daily log returns, dates x assets. No look-ahead: at date t the
        correlation uses only returns strictly before t for parameter
        estimation, and R_t is the one-step forecast from t-1.
    window : int
        Length of the estimation window in trading days.
    step : int
        Re-estimate GARCH/DCC parameters every `step` days. Between refits
        the filter is updated with newly observed returns (still causal).
    n_assets : int, optional
        If set, use the first n columns only.
    """
    rets = returns.dropna(how="any").sort_index()
    if n_assets is not None:
        rets = rets.iloc[:, :n_assets]
    assets = [str(c) for c in rets.columns]
    n = len(assets)
    if window <= n:
        raise ValueError(f"window ({window}) must be > n_assets ({n}) so Qbar is PD")

    idx = rets.index
    raw = rets.to_numpy(dtype=float)
    T = raw.shape[0]
    n_out = T - window
    if n_out <= 0:
        raise ValueError("not enough rows for the chosen window")

    R_out = np.empty((n_out, n, n), dtype=np.float32)
    out_dates = idx[window:]
    rows = []

    origin = window
    x0_garch = None
    x0_dcc = None
    out_i = 0
    n_refits = 0

    while origin < T:
        train = raw[origin - window : origin]
        mu = train.mean(axis=0)
        train_pct = (train - mu) * _RETURN_SCALE

        state = fit_state(train_pct, x0_garch=x0_garch, x0_dcc=x0_dcc)
        state.mean = mu
        x0_garch = (state.garch_a.copy(), state.garch_b.copy())
        x0_dcc = (state.dcc_a, state.dcc_b)
        n_refits += 1

        rows.append(
            {
                "refit_date": idx[origin - 1],
                "forecast_from": idx[origin],
                "a": state.dcc_a,
                "b": state.dcc_b,
                "a_plus_b": state.dcc_a + state.dcc_b,
                "mean_garch_alpha": float(state.garch_a.mean()),
                "mean_garch_beta": float(state.garch_b.mean()),
            }
        )
        if verbose:
            print(
                f"refit {n_refits:3d}  window {idx[origin - window].date()} -> "
                f"{idx[origin - 1].date()}  DCC(a={state.dcc_a:.4f}, b={state.dcc_b:.4f})"
            )

        R_out[out_i] = state.forecast_R()
        out_i += 1

        last = min(origin + step, T)
        for t in range(origin + 1, last):
            r_pct = (raw[t - 1] - mu) * _RETURN_SCALE
            R_next = state.update(r_pct)
            R_out[out_i] = R_next
            out_i += 1

        origin = last

    assert out_i == n_out, f"wrote {out_i} matrices, expected {n_out}"
    params = pd.DataFrame(rows)
    return RollingDCCResult(
        dates=out_dates,
        assets=assets,
        R=R_out,
        params=params,
        window=window,
        step=step,
        extra={"n_refits": n_refits},
    )
