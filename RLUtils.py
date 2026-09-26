import shutil
from pathlib import Path

import pandas as pd
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Chart colors: one fixed color per series, blue <-> gray <-> red for signed values.
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
MUTED = "#52514e"
DIVERGING = LinearSegmentedColormap.from_list("blue_gray_red", ["#104281", "#f0efec", "#b3261e"])

class RLUtils:
    """Helpers for the PortfolioOptimizationEnv experiments in task3."""

    ENV_KWARGS = dict(
        initial_amount=10000,
        order_df=True,
        comission_fee_pct=0.001,
        comission_fee_model="trf",
        return_last_action=True,
        normalize_df=None,
    )
    STRATEGY_COLORS = {
        "Equal weight": SERIES_COLORS[0],
        "PPO": SERIES_COLORS[1],
        "DDPG": SERIES_COLORS[2],
        "A2C": SERIES_COLORS[3],
        "Ensemble": SERIES_COLORS[4],
    }

    # Scale-free observation. Raw close and volume are left out; they saturate the networks.
    FEATURE_LIST = [
        "boll_pctb",
        "volume_stad",
        "rsi_6_scale",
        "macd_stad",
        "macds_stad",
        "macdh_stad",
        "adx_scale",
        "pvo_stad",
        "tema_stad",
        "stochrsi_3_scale",
        "log-ret_stad",
    ]
    # Dedicated RL file: raw OHLCV, day of week, the scaled features and the risk-free columns.
    RL_PANEL_PATH = "data/rl_features.csv"
    RF_PATH = "data/risk_free.csv"  # FRED DTB3 cache, so a rebuild needs no network
    RF_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DTB3&cosd={start}&coed={end}"
    RAW_COLUMNS = ["open", "high", "low", "close", "volume"]
    # Observation for ShareTradingEnv: raw OHLCV and day of week (Monday = 0) are observed unscaled.
    RL_FEATURE_LIST = RAW_COLUMNS + ["day"] + FEATURE_LIST + ["rf_return", "excess_ret"]
    SHARE_ENV_KWARGS = dict(
        initial_amount=1_000_000,
        hmax=100,
        buy_cost_pct=0.001,
        sell_cost_pct=0.001,
        verbose=False,
    )
    TUNE_N_ASSETS = 25
    TOTAL_TIMESTEPS = 20_000
    N_TRIALS = 5
    EW_PENALTY = 1e-4
    PPO_CURVES = [
        "rollout/ep_rew_mean", "train/approx_kl", "train/value_loss", "train/policy_gradient_loss",
        "train/entropy_loss", "train/explained_variance", "train/clip_fraction", "train/std",
    ]
    DDPG_CURVES = ["rollout/ep_rew_mean", "train/actor_loss", "train/critic_loss"]

    @staticmethod
    def make_env(df, features, **overrides):
        # Imported here: poe switches matplotlib to Agg, which other notebooks don't want.
        from poe import PortfolioOptimizationEnv
        return PortfolioOptimizationEnv(df=df, features=features, **{**RLUtils.ENV_KWARGS, **overrides})

    @staticmethod
    def make_vec_env(df, features, gamma, **overrides):
        """SB3 vec env with reward normalization and episode metrics, as used for training."""
        vec_env, _ = RLUtils.make_env(df, features, **overrides).get_sb_env()
        vec_env = VecMonitor(vec_env)  # adds rollout/ep_rew_mean, ep_len_mean
        return VecNormalize(vec_env, norm_obs=False, norm_reward=True, clip_reward=10.0, gamma=gamma)

    @staticmethod
    def make_share_env(df, features, **overrides):
        from trading_env import ShareTradingEnv
        return ShareTradingEnv(df=df, features=features, **{**RLUtils.SHARE_ENV_KWARGS, **overrides})

    @staticmethod
    def make_share_vec_env(df, features, gamma, norm_obs=False, **overrides):
        """make_vec_env for ShareTradingEnv: reward normalization and episode metrics.

        `norm_obs=True` also z-scores every (feature, asset) cell of the state with running
        statistics from this env's data, clipped at +-10. Raw OHLCV and volume need it:
        unscaled, they pin the policy's actions at +-1. Pass the fitted wrapper
        (model.get_vec_normalize_env()) to DRLAgent.DRL_prediction at test time.
        """
        vec_env, _ = RLUtils.make_share_env(df, features, **overrides).get_sb_env()
        vec_env = VecMonitor(vec_env)
        return VecNormalize(
            vec_env,
            norm_obs=norm_obs,
            norm_obs_keys=["state"] if norm_obs else None,  # the weights are already in [0, 1]
            norm_reward=True,
            clip_reward=10.0,
            gamma=gamma,
        )

    @staticmethod
    def load_share_normalizer(path, df, features):
        """Observation statistics saved from a training VecNormalize, frozen for testing."""
        from stable_baselines3.common.vec_env import DummyVecEnv
        normalizer = VecNormalize.load(path, DummyVecEnv([lambda: RLUtils.make_share_env(df, features)]))
        normalizer.training = False
        normalizer.norm_reward = False
        return normalizer

    @staticmethod
    def load_risk_free(start, end, path=None):
        """Annualized 3-month T-bill rate (FRED DTB3) as a decimal, indexed by date.

        Downloaded once and cached at `path`. Bond-market holidays are NaN here.
        """
        path = Path(RLUtils.RF_PATH if path is None else path)
        if not path.exists():
            raw = pd.read_csv(RLUtils.RF_URL.format(start=start, end=end))
            path.parent.mkdir(parents=True, exist_ok=True)
            raw.to_csv(path, index=False)
        raw = pd.read_csv(path, parse_dates=["observation_date"])
        rate = pd.to_numeric(raw["DTB3"], errors="coerce") / 100
        return pd.Series(rate.to_numpy(), index=raw["observation_date"], name="rf_rate")

    @staticmethod
    def build_rl_panel(panel_path="data/features_panel.csv", prices_path="data/created_df.csv", out_path=None):
        """Write the RL file: date, tic, RL_FEATURE_LIST, rf_return and excess_ret.

        rf_return is the daily T-bill return, the same for every asset on a date.
        excess_ret is each asset's close-to-close return minus the previous session's
        rf_return, the rate in force over that holding day.
        """
        out_path = RLUtils.RL_PANEL_PATH if out_path is None else out_path
        panel = pd.read_csv(panel_path, usecols=["date", "tic"] + RLUtils.FEATURE_LIST, parse_dates=["date"])
        prices = pd.read_csv(prices_path, usecols=["date", "tic"] + RLUtils.RAW_COLUMNS, parse_dates=["date"])
        prices = prices.sort_values(["tic", "date"])
        prices["ret"] = prices.groupby("tic")["close"].pct_change()

        # Built and shifted on the date index, before merging, so nothing moves across tickers.
        sessions = pd.DatetimeIndex(prices["date"].unique()).sort_values()
        start = sessions[0] - pd.Timedelta(days=10)  # so the first session has a rate to carry forward
        rate = RLUtils.load_risk_free(f"{start:%Y-%m-%d}", f"{sessions[-1]:%Y-%m-%d}")
        rate = rate.reindex(rate.index.union(sessions)).ffill().reindex(sessions)
        rf = pd.DataFrame({"rf_return": (1 + rate) ** (1 / 252) - 1}, index=sessions)
        rf["rf_prev"] = rf["rf_return"].shift(1)
        rf.index.name = "date"

        out = panel.merge(prices, on=["date", "tic"], how="left").merge(rf.reset_index(), on="date", how="left")
        out["excess_ret"] = out["ret"] - out["rf_prev"]
        out["day"] = out["date"].dt.dayofweek
        out = out[["date", "tic"] + RLUtils.RL_FEATURE_LIST]
        out = out.sort_values(["date", "tic"]).reset_index(drop=True)

        missing = out.isna().sum()
        assert not missing.any(), f"NaN in the RL file:\n{missing[missing > 0]}"
        per_date = out.groupby("date")[["rf_return", "day"]].nunique().max()
        assert (per_date == 1).all(), f"date-level columns vary across assets:\n{per_date}"
        out.to_csv(out_path, index=False)
        print(out_path, out.shape, "tickers:", out["tic"].nunique(), "sessions:", out["date"].nunique())
        return out

    @staticmethod
    def value_metrics(values, turnover=None):
        """Sharpe, drawdowns, turnover and return from a daily portfolio-value series."""
        values = np.asarray(values, dtype=float)
        rets = values[1:] / values[:-1] - 1
        drawdown = values / np.maximum.accumulate(values) - 1
        metrics = {
            "final value": values[-1],
            "total return": values[-1] / values[0] - 1,
            "Sharpe": np.sqrt(252) * rets.mean() / rets.std(ddof=1),
            "max drawdown": drawdown.min(),
            "avg drawdown": drawdown.mean(),
        }
        if turnover is not None:
            metrics["daily turnover"] = float(np.nanmean(np.asarray(turnover, dtype=float)))
        return metrics

    @staticmethod
    def subset_assets(df, n):
        """First `n` tickers, in sorted ticker order, for the tuning subset."""
        tics = sorted(df["tic"].unique())[:n]
        return df[df["tic"].isin(tics)].reset_index(drop=True)

    @staticmethod
    def trial_logger(algo, trial, root="logs"):
        """CSV + TensorBoard logger at logs/<algo>/trial_<n>, one folder per Optuna trial."""
        path = Path(root) / algo / f"trial_{trial.number:02d}"
        return str(path), configure(str(path), ["csv", "tensorboard"])

    @staticmethod
    def attach_final_metrics(trial, log_dir):
        """Store the last value of every train/ and rollout/ metric on the Optuna trial."""
        prog = pd.read_csv(Path(log_dir) / "progress.csv")
        for col in prog.columns:
            if col.startswith(("train/", "rollout/")) and prog[col].notna().any():
                trial.set_user_attr(col, float(prog[col].dropna().iloc[-1]))

    @staticmethod
    def plot_training_curves(algo, metrics, root="logs"):
        """One panel per metric, one line per trial, from logs/<algo>/trial_*/progress.csv."""
        runs = {}
        for f in sorted(Path(root, algo).glob("trial_*/progress.csv")):
            try:
                runs[f.parent.name] = pd.read_csv(f)
            except pd.errors.EmptyDataError:
                pass
        metrics = [m for m in metrics if any(m in d.columns for d in runs.values())]
        if not metrics:
            print("No metrics logged yet under", Path(root, algo))
            return
        ncol = 2
        nrow = (len(metrics) + ncol - 1) // ncol
        fig, axes = plt.subplots(nrow, ncol, figsize=(13, 3.2 * nrow), squeeze=False)
        for ax, metric in zip(axes.flat, metrics):
            for name, frame in runs.items():
                if metric in frame.columns and frame[metric].notna().any():
                    sub = frame[["time/total_timesteps", metric]].dropna()
                    ax.plot(sub["time/total_timesteps"], sub[metric], marker=".", lw=1, label=name)
            ax.set_title(metric)
            ax.set_xlabel("timesteps")
            ax.grid(True, alpha=0.3)
        for ax in list(axes.flat)[len(metrics):]:
            ax.axis("off")
        axes.flat[0].legend(fontsize=7, ncol=2)
        fig.suptitle(f"{algo.upper()} training metrics per Optuna trial")
        fig.tight_layout()
        plt.show()

    @staticmethod
    def rollout_value(model, env):
        """Final portfolio value of a deterministic run of `model` on an existing env."""
        vec_env, obs = env.get_sb_env()  # get_sb_env resets the env
        value = float(env._initial_amount)
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _ = vec_env.step(action)
            if done[0]:
                break
            value = float(env._portfolio_value)
        return value

    @staticmethod
    def describe_observation(df, features, **overrides):
        """Print one observation's shape, scale and NaN count. Returns the last-step frame."""
        env = RLUtils.make_env(df, features, **overrides)
        obs, info = env.reset()
        state = obs["state"] if isinstance(obs, dict) else obs
        print(env.observation_space)
        print("state", state.shape, state.dtype)
        print("features", env._features)
        print("date", info["end_time"])
        frame = pd.DataFrame(state[:, :, -1], index=env._features, columns=list(env._tic_list)).T
        print(frame.describe().T)
        print(frame.iloc[:5, :5])
        print("nan", np.isnan(state).sum(), "inf", np.isinf(state).sum())
        return frame

    @staticmethod
    def run_episode(df, features, model=None, **overrides):
        """Run one full episode on df, with equal weights when model is None, and return the env.

        Stops before the env's final bookkeeping call, so its memories are still intact.
        """
        env = RLUtils.make_env(df, features, save_plots=False, **overrides)
        obs, _ = env.reset()
        equal = np.r_[0.0, np.full(env._stock_dim, 1.0 / env._stock_dim)].astype(np.float32)
        while env._time_index < len(env._sorted_times) - 1:
            action = equal if model is None else model.predict(obs, deterministic=True)[0]
            obs, *_ = env.step(action)
        return env

    @staticmethod
    def episode_metrics(env):
        """Sharpe, drawdown, turnover and distance from equal weight for a finished episode."""
        values = np.asarray(env._asset_memory["final"], dtype=float)
        rets = np.asarray(env._portfolio_return_memory[1:], dtype=float)
        chosen = np.asarray(env._actions_memory[1:], dtype=float)  # weights picked each day
        held = np.asarray(env._final_weights[:-1], dtype=float)  # weights going into that day
        turnover = 0.5 * np.abs(chosen - held).sum(axis=1)
        return {
            "final value": values[-1],
            "total return": values[-1] / values[0] - 1,
            "Sharpe": np.sqrt(252) * rets.mean() / rets.std(ddof=1),
            "max drawdown": (values / np.maximum.accumulate(values) - 1).min(),
            "daily turnover": turnover[1:].mean(),  # skips the first move out of cash
            "gap to equal weight": float(np.mean(env._ew_deviation_memory[1:])),
        }

    @staticmethod
    def compare_strategies(windows, models, features, **overrides):
        """Plot cumulative return for each window and return the episode-metric table.

        `windows` is {label: dataframe}. `models` is {label: model}, with None for equal weight.
        `features` is one list for every model, or {label: list} when models were trained on
        different columns.
        """
        features = features if isinstance(features, dict) else dict.fromkeys(models, features)
        episodes = {
            (window, name): RLUtils.run_episode(df, features[name], model, **overrides)
            for window, df in windows.items()
            for name, model in models.items()
        }
        fig, axes = plt.subplots(1, len(windows), figsize=(6 * len(windows), 4), squeeze=False)
        for ax, (window, df) in zip(axes.flat, windows.items()):
            for name in models:
                env = episodes[(window, name)]
                values = np.asarray(env._asset_memory["final"], dtype=float)
                color = RLUtils.STRATEGY_COLORS.get(name)
                ax.plot(
                    pd.to_datetime(env._date_memory),
                    values / values[0] - 1,
                    lw=1.2,
                    label=name,
                    color=color,
                )
            span = pd.to_datetime(df["date"])
            ax.axhline(0, color="k", lw=0.8)
            ax.set_title(f"{window}\n{span.min():%Y-%m-%d} to {span.max():%Y-%m-%d}")
            ax.set_xlabel("Date")
            ax.legend()
        axes.flat[0].set_ylabel("Cumulative return")
        fig.autofmt_xdate()
        fig.tight_layout()
        plt.show()
        metrics = pd.DataFrame({key: RLUtils.episode_metrics(env) for key, env in episodes.items()}).T
        metrics.index.names = ["window", "strategy"]
        return metrics

    @staticmethod
    def tune_ppo(
        train,
        features=None,
        n_trials=None,
        total_timesteps=None,
        tune_n_assets=None,
        ew_penalty=None,
        log_root="logs",
        save_path="ppo_portfolio",
    ):
        """Optuna search on a ticker subset, then one retrain of the best settings on all assets.

        Returns (model, study). The model is also written to `save_path`.
        """
        import optuna
        from stable_baselines3 import PPO

        features = list(RLUtils.FEATURE_LIST if features is None else features)
        n_trials = RLUtils.N_TRIALS if n_trials is None else n_trials
        total_timesteps = RLUtils.TOTAL_TIMESTEPS if total_timesteps is None else total_timesteps
        tune_n_assets = RLUtils.TUNE_N_ASSETS if tune_n_assets is None else tune_n_assets
        ew_penalty = RLUtils.EW_PENALTY if ew_penalty is None else ew_penalty

        tune_train = RLUtils.subset_assets(train, tune_n_assets)
        print(
            f"tuning on {tune_train['tic'].nunique()} of {train['tic'].nunique()} assets, "
            f"{tune_train['date'].nunique()} sessions"
        )
        shutil.rmtree(Path(log_root, "ppo"), ignore_errors=True)

        def objective(trial):
            params = dict(
                n_steps=trial.suggest_categorical("n_steps", [512, 1024, 2048]),
                batch_size=trial.suggest_categorical("batch_size", [64, 128, 256, 512]),
                gamma=trial.suggest_float("gamma", 0.9, 0.999),
                clip_range=trial.suggest_float("clip_range", 0.1, 0.3),
                learning_rate=trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
                log_std_init=trial.suggest_float("log_std_init", -2.0, -1.0),
            )
            vec_env = RLUtils.make_vec_env(
                tune_train, features, params["gamma"], ew_penalty=ew_penalty, save_plots=False,
            )
            model = PPO(
                "MultiInputPolicy",
                vec_env,
                verbose=0,
                seed=42,
                n_steps=params["n_steps"],
                batch_size=params["batch_size"],
                clip_range=params["clip_range"],
                gamma=params["gamma"],
                device="cpu",
                learning_rate=params["learning_rate"],
                policy_kwargs=dict(log_std_init=params["log_std_init"]),
            )
            log_dir, logger = RLUtils.trial_logger("ppo", trial, log_root)
            model.set_logger(logger)
            model.learn(total_timesteps=total_timesteps, log_interval=1)
            RLUtils.attach_final_metrics(trial, log_dir)
            return RLUtils.rollout_value(model, RLUtils.make_env(tune_train, features, save_plots=False))

        study = optuna.create_study(
            direction="maximize", sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=2),
        )
        study.optimize(objective, n_trials=n_trials)
        print("best PPO params:", study.best_params, "tuning score:", study.best_value)

        best = study.best_params
        vec_env = RLUtils.make_vec_env(
            train, features, best["gamma"], ew_penalty=ew_penalty, save_plots=False,
        )
        model = PPO(
            "MultiInputPolicy",
            vec_env,
            verbose=0,
            seed=42,
            n_steps=best["n_steps"],
            batch_size=best["batch_size"],
            clip_range=best["clip_range"],
            gamma=best["gamma"],
            device="cpu",
            learning_rate=best["learning_rate"],
            policy_kwargs=dict(log_std_init=best["log_std_init"]),
        )
        model.set_logger(configure(str(Path(log_root, "ppo", "final")), ["csv", "tensorboard"]))
        model.learn(total_timesteps=total_timesteps, log_interval=1)
        model.save(save_path)
        print("final PPO train value:", RLUtils.rollout_value(model, RLUtils.make_env(train, features)))
        return model, study

    @staticmethod
    def tune_ddpg(
        train,
        features=None,
        n_trials=None,
        total_timesteps=None,
        tune_n_assets=None,
        ew_penalty=None,
        log_root="logs",
        save_path="ddpg_portfolio",
    ):
        """Optuna search on a ticker subset, then one retrain of the best settings on all assets.

        Returns (model, study). The model is also written to `save_path`.
        """
        import optuna
        from stable_baselines3 import DDPG
        from stable_baselines3.common.noise import NormalActionNoise

        features = list(RLUtils.FEATURE_LIST if features is None else features)
        n_trials = RLUtils.N_TRIALS if n_trials is None else n_trials
        total_timesteps = RLUtils.TOTAL_TIMESTEPS if total_timesteps is None else total_timesteps
        tune_n_assets = RLUtils.TUNE_N_ASSETS if tune_n_assets is None else tune_n_assets
        ew_penalty = RLUtils.EW_PENALTY if ew_penalty is None else ew_penalty

        tune_train = RLUtils.subset_assets(train, tune_n_assets)
        print(
            f"tuning on {tune_train['tic'].nunique()} of {train['tic'].nunique()} assets, "
            f"{tune_train['date'].nunique()} sessions"
        )
        shutil.rmtree(Path(log_root, "ddpg"), ignore_errors=True)

        def objective(trial):
            params = dict(
                gamma=trial.suggest_float("gamma", 0.95, 0.999),
                sigma=trial.suggest_float("sigma", 0.05, 0.2),
                learning_rate=trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
                learning_starts=trial.suggest_categorical("learning_starts", [500, 1000, 2000]),
                tau=trial.suggest_float("tau", 1e-3, 1e-2, log=True),
                batch_size=trial.suggest_categorical("batch_size", [64, 128, 256]),
            )
            vec_env = RLUtils.make_vec_env(
                tune_train, features, params["gamma"], ew_penalty=ew_penalty, save_plots=False,
            )
            n_actions = vec_env.action_space.shape[-1]
            model = DDPG(
                "MultiInputPolicy",
                vec_env,
                verbose=0,
                seed=42,
                learning_rate=params["learning_rate"],
                buffer_size=100_000,
                learning_starts=params["learning_starts"],
                tau=params["tau"],
                gamma=params["gamma"],
                train_freq=1,
                gradient_steps=1,
                batch_size=params["batch_size"],
                device="cuda",
                action_noise=NormalActionNoise(
                    mean=np.zeros(n_actions), sigma=params["sigma"] * np.ones(n_actions),
                ),
            )
            log_dir, logger = RLUtils.trial_logger("ddpg", trial, log_root)
            model.set_logger(logger)
            model.learn(total_timesteps=total_timesteps, log_interval=1)
            RLUtils.attach_final_metrics(trial, log_dir)
            return RLUtils.rollout_value(model, RLUtils.make_env(tune_train, features, save_plots=False))

        study = optuna.create_study(
            direction="maximize", sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=2),
        )
        study.optimize(objective, n_trials=n_trials)
        print("best DDPG params:", study.best_params, "tuning score:", study.best_value)

        best = study.best_params
        vec_env = RLUtils.make_vec_env(
            train, features, best["gamma"], ew_penalty=ew_penalty, save_plots=False,
        )
        n_actions = vec_env.action_space.shape[-1]
        model = DDPG(
            "MultiInputPolicy",
            vec_env,
            verbose=0,
            seed=42,
            learning_rate=best["learning_rate"],
            buffer_size=100_000,
            learning_starts=best["learning_starts"],
            tau=best["tau"],
            gamma=best["gamma"],
            train_freq=1,
            gradient_steps=1,
            batch_size=best["batch_size"],
            device="cuda",
            action_noise=NormalActionNoise(
                mean=np.zeros(n_actions), sigma=best["sigma"] * np.ones(n_actions),
            ),
        )
        model.set_logger(configure(str(Path(log_root, "ddpg", "final")), ["csv", "tensorboard"]))
        model.learn(total_timesteps=total_timesteps, log_interval=1)
        model.save(save_path)
        print("final DDPG train value:", RLUtils.rollout_value(model, RLUtils.make_env(train, features)))
        return model, study

    @staticmethod
    def last_n_sessions(df, n):
        sessions = pd.Index(pd.to_datetime(df["date"]).unique()).sort_values()
        recent = df[pd.to_datetime(df["date"]).isin(sessions[-n:])]
        return pd.DataFrame(recent).reset_index()

    @staticmethod
    def run_equal_weight(env):
        """Step `env` to the end with equal weights on every asset (no cash). Returns rewards."""
        n = env._stock_dim
        weights = np.zeros(n + 1, dtype=np.float32)
        weights[1:] = 1 / n
        rewards = []
        while True:
            _, reward, terminated, _, _ = env.step(weights)
            rewards.append(reward)
            if terminated:
                break
        return rewards

    @staticmethod
    def run_policy(model, df, features):
        """Roll out a trained SB3 model deterministically. Returns (dates, portfolio values)."""
        env = RLUtils.make_env(df, features)
        vec_env, obs = env.get_sb_env()
        values, dates = list(env._asset_memory["final"]), list(env._date_memory)
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _ = vec_env.step(action)
            if done[0]:
                break
            values, dates = list(env._asset_memory["final"]), list(env._date_memory)
        return pd.to_datetime(dates), np.asarray(values, dtype=float)

    @staticmethod
    def env_values(env):
        """(dates, portfolio values) recorded by an env that has been stepped."""
        return pd.to_datetime(env._date_memory), np.asarray(env._asset_memory["final"], dtype=float)

    @staticmethod
    def equal_weight_curve(df, features, **overrides):
        env = RLUtils.make_env(df, features, **overrides)
        RLUtils.run_equal_weight(env)
        dates, values = RLUtils.env_values(env)
        return dates, values / values[0] - 1

    @staticmethod
    def policy_curve(model, df, features):
        dates, values = RLUtils.run_policy(model, df, features)
        return dates, values / values[0] - 1

    @staticmethod
    def final_value(model, df, features):
        return float(RLUtils.run_policy(model, df, features)[1][-1])

    @staticmethod
    def learn_and_score(model, best, total_timesteps, tb_log_name, df, features):
        """Train `model`, score it by final portfolio value on `df`, keep it in `best` if it wins."""
        model.learn(total_timesteps=total_timesteps, tb_log_name=tb_log_name)
        score = RLUtils.final_value(model, df, features)
        if score > best["score"]:
            best["model"], best["score"] = model, score
        return score

    @staticmethod
    def plot_curves(curves, title, ax=None):
        """Plot {label: (dates, cumulative return)} on one axis."""
        own = ax is None
        if own:
            _, ax = plt.subplots(figsize=(10, 4))
        for i, (label, (dates, cumulative)) in enumerate(curves.items()):
            color = RLUtils.STRATEGY_COLORS.get(label, SERIES_COLORS[i % len(SERIES_COLORS)])
            ax.plot(dates, cumulative, lw=1.6, color=color, label=label)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_title(title)
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative return")
        if len(curves) > 1:
            ax.legend()
        if own:
            plt.tight_layout()
            plt.show()
        return ax