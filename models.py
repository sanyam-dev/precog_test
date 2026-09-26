# DRL models from Stable Baselines 3
"""Ported from FinRL-0.3.8/finrl/agents/stablebaselines3/models.py (MIT).

Changes from FinRL:
- no finrl imports: `config` and `data_split` are defined here
- trading_env.ShareTradingEnv (FinRL's StockTradingEnv trading rules) replaces StockTradingEnv,
  built through RLUtils.make_share_env / make_share_vec_env
- the default policy is MultiInputPolicy, because the observation is a Dict
- model kwargs are copied before the action noise is set. FinRL wrote the noise object into
  the shared defaults, so a second get_model call failed
- predictions read the env's own memories instead of StockTradingEnv's save_* methods
- DRLEnsembleAgent drops the turbulence threshold (the data has no turbulence column) and the
  previous-state hand-over (every window starts from cash)
"""
from __future__ import annotations

import statistics
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Type

import numpy as np
import pandas as pd
from stable_baselines3 import A2C
from stable_baselines3 import DDPG
from stable_baselines3 import PPO
from stable_baselines3 import SAC
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise

from RLUtils import RLUtils

# What FinRL read from finrl/config.py. The *_PARAMS are FinRL's defaults, plus:
# - PPO and A2C on the CPU. The off-policy models keep SB3's "auto" device (CUDA when present)
# - Normal action noise for DDPG (sigma 0.1 in get_model). FinRL's default DDPG has none
# TD3's 1,000,000-step buffer holds two Dict observations per step, about 14 GB at 19 features
# and 100 assets, so pass a smaller buffer_size before using TD3.
config = SimpleNamespace(
    TRAINED_MODEL_DIR="trained_models",
    TENSORBOARD_LOG_DIR="tensorboard_log",
    RESULTS_DIR="results",
    A2C_PARAMS={"n_steps": 5, "ent_coef": 0.01, "learning_rate": 0.0007, "device": "cpu"},
    PPO_PARAMS={
        "n_steps": 2048,
        "ent_coef": 0.01,
        "learning_rate": 0.00025,
        "batch_size": 64,
        "device": "cpu",
    },
    DDPG_PARAMS={
        "batch_size": 128,
        "buffer_size": 50000,
        "learning_rate": 0.001,
        "action_noise": "normal",
    },
    TD3_PARAMS={"batch_size": 100, "buffer_size": 1000000, "learning_rate": 0.001},
    SAC_PARAMS={
        "batch_size": 64,
        "buffer_size": 100000,
        "learning_rate": 0.0001,
        "learning_starts": 100,
        "ent_coef": "auto_0.1",
    },
)

MODELS = {"a2c": A2C, "ddpg": DDPG, "td3": TD3, "sac": SAC, "ppo": PPO}

MODEL_KWARGS = {x: getattr(config, f"{x.upper()}_PARAMS") for x in MODELS.keys()}

NOISE = {
    "normal": NormalActionNoise,
    "ornstein_uhlenbeck": OrnsteinUhlenbeckActionNoise,
}


def data_split(df, start, end, target_date_col="date"):
    """
    split the dataset into training or testing using date
    :param data: (df) pandas dataframe, start, end
    :return: (df) pandas dataframe
    """
    data = df[(df[target_date_col] >= start) & (df[target_date_col] < end)]
    data = data.sort_values([target_date_col, "tic"], ignore_index=True)
    data.index = data[target_date_col].factorize()[0]
    return data


def _with_noise(model_kwargs, env):
    """Copy of `model_kwargs` with a named action noise replaced by the noise object."""
    model_kwargs = dict(model_kwargs)
    if isinstance(model_kwargs.get("action_noise"), str):
        n_actions = env.action_space.shape[-1]
        model_kwargs["action_noise"] = NOISE[model_kwargs["action_noise"]](
            mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions)
        )
    return model_kwargs


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        try:
            self.logger.record(key="train/reward", value=self.locals["rewards"][0])

        except BaseException as error:
            try:
                self.logger.record(key="train/reward", value=self.locals["reward"][0])

            except BaseException as inner_error:
                # Handle the case where neither "rewards" nor "reward" is found
                self.logger.record(key="train/reward", value=None)
                # Print the original error and the inner error for debugging
                print("Original Error:", error)
                print("Inner Error:", inner_error)
        return True

    def _on_rollout_end(self) -> bool:
        # Off-policy models (DDPG, TD3, SAC) have no rollout buffer and end a "rollout"
        # every step. FinRL printed a logging error each time.
        if "rollout_buffer" not in self.locals:
            return True
        try:
            rollout_buffer_rewards = self.locals["rollout_buffer"].rewards.flatten()
            self.logger.record(
                key="train/reward_min", value=min(rollout_buffer_rewards)
            )
            self.logger.record(
                key="train/reward_mean", value=statistics.mean(rollout_buffer_rewards)
            )
            self.logger.record(
                key="train/reward_max", value=max(rollout_buffer_rewards)
            )
        except BaseException as error:
            # Handle the case where "rewards" is not found
            self.logger.record(key="train/reward_min", value=None)
            self.logger.record(key="train/reward_mean", value=None)
            self.logger.record(key="train/reward_max", value=None)
            print("Logging Error:", error)
        return True


class DRLAgent:
    """Provides implementations for DRL algorithms

    Attributes
    ----------
        env: gym environment class
            SB3 vec env on ShareTradingEnv, e.g. RLUtils.make_share_vec_env(...)

    Methods
    -------
        get_model()
            setup DRL algorithms
        train_model()
            train DRL algorithms in a train dataset
            and output the trained model
        DRL_prediction()
            make a prediction in a test dataset and get results
    """

    def __init__(self, env):
        self.env = env

    def get_model(
        self,
        model_name,
        policy="MultiInputPolicy",
        policy_kwargs=None,
        model_kwargs=None,
        verbose=1,
        seed=None,
        tensorboard_log=None,
    ):
        if model_name not in MODELS:
            raise ValueError(
                f"Model '{model_name}' not found in MODELS."
            )  # this is more informative than NotImplementedError("NotImplementedError")

        if model_kwargs is None:
            model_kwargs = MODEL_KWARGS[model_name]
        model_kwargs = _with_noise(model_kwargs, self.env)
        print(model_kwargs)
        return MODELS[model_name](
            policy=policy,
            env=self.env,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            policy_kwargs=policy_kwargs,
            seed=seed,
            **model_kwargs,
        )

    @staticmethod
    def train_model(
        model,
        tb_log_name,
        total_timesteps=5000,
        callbacks: Type[BaseCallback] = None,
    ):  # this function is static method, so it can be called without creating an instance of the class
        model = model.learn(
            total_timesteps=total_timesteps,
            tb_log_name=tb_log_name,
            callback=(
                CallbackList(
                    [TensorboardCallback()] + [callback for callback in callbacks]
                )
                if callbacks is not None
                else TensorboardCallback()
            ),
        )
        return model

    @staticmethod
    def DRL_prediction(model, environment, deterministic=True, obs_normalizer=None):
        """make a prediction and get results

        `environment` is a ShareTradingEnv. `obs_normalizer` is the VecNormalize the model
        was trained with, when it normalized observations; its statistics are applied, not
        updated. Returns (account_memory, actions_memory): date, account_value and turnover
        for each session, and the shares traded in each asset on each date.
        """
        test_env, test_obs = environment.get_sb_env()
        # DummyVecEnv resets the env on the terminal step, which clears its memories,
        # so they are copied before every step (as RLUtils.run_policy does)
        memories = DRLAgent._copy_memories(environment)
        for _ in range(len(environment._sorted_times)):
            if obs_normalizer is not None:
                test_obs = obs_normalizer.normalize_obs(test_obs)
            action, _states = model.predict(test_obs, deterministic=deterministic)
            test_obs, rewards, dones, info = test_env.step(action)
            if dones[0]:
                print("hit end!")
                break
            memories = DRLAgent._copy_memories(environment)
        return DRLAgent._memory_frames(environment, *memories)

    @staticmethod
    def _copy_memories(env):
        return (
            list(env._date_memory),
            list(env._asset_memory["final"]),
            list(env._turnover_memory),
            list(env._actions_memory),
        )

    @staticmethod
    def _memory_frames(env, dates, values, turnover, actions):
        dates = pd.to_datetime(dates)
        account_memory = pd.DataFrame(
            {
                "date": dates,
                "account_value": values,
                "turnover": [np.nan] + turnover,  # no trade before the first session
            }
        )
        actions_memory = pd.DataFrame(
            np.asarray(actions).reshape(len(actions), env._stock_dim),
            index=pd.Index(dates[:-1], name="date"),  # trades placed at each session's close
            columns=list(env._tic_list),
        )
        return account_memory, actions_memory

    @staticmethod
    def DRL_prediction_load_from_file(
        model_name, environment, cwd, deterministic=True, obs_normalizer=None
    ):
        if model_name not in MODELS:
            raise ValueError(
                f"Model '{model_name}' not found in MODELS."
            )  # this is more informative than NotImplementedError("NotImplementedError")
        try:
            # load agent
            model = MODELS[model_name].load(cwd)
            print("Successfully load model", cwd)
        except BaseException as error:
            raise ValueError(f"Failed to load agent. Error: {str(error)}") from error

        # test on the testing env
        account_memory, _ = DRLAgent.DRL_prediction(
            model, environment, deterministic, obs_normalizer
        )
        episode_total_assets = account_memory["account_value"].tolist()
        print("episode_return", episode_total_assets[-1] / episode_total_assets[0])
        print("Test Finished!")
        return episode_total_assets


class DRLEnsembleAgent:
    """FinRL's ensemble strategy on ShareTradingEnv.

    Every `rebalance_window` sessions, each model with kwargs is trained from scratch on an
    expanding window, scored by Sharpe on the next `validation_window` sessions, and the
    best one trades the `rebalance_window` sessions after that.

    Differences from FinRL: no turbulence threshold; validation and trading windows each
    start from cash (ShareTradingEnv cannot take over the previous window's holdings), so
    chain the windows' returns, not their account values; predictions are deterministic.
    Each trading window's account values go to results/account_value_trade_ensemble_<i>.csv.
    """

    @staticmethod
    def get_model(
        model_name,
        env,
        policy="MultiInputPolicy",
        policy_kwargs=None,
        model_kwargs=None,
        seed=None,
        verbose=1,
    ):
        if model_name not in MODELS:
            raise ValueError(
                f"Model '{model_name}' not found in MODELS."
            )  # this is more informative than NotImplementedError("NotImplementedError")

        if model_kwargs is None:
            model_kwargs = MODEL_KWARGS[model_name]
        temp_model_kwargs = _with_noise(model_kwargs, env)
        print(temp_model_kwargs)
        return MODELS[model_name](
            policy=policy,
            env=env,
            tensorboard_log=f"{config.TENSORBOARD_LOG_DIR}/{model_name}",
            verbose=verbose,
            policy_kwargs=policy_kwargs,
            seed=seed,
            **temp_model_kwargs,
        )

    @staticmethod
    def train_model(
        model,
        model_name,
        tb_log_name,
        iter_num,
        total_timesteps=5000,
        callbacks: Type[BaseCallback] = None,
    ):
        model = model.learn(
            total_timesteps=total_timesteps,
            tb_log_name=tb_log_name,
            callback=(
                CallbackList(
                    [TensorboardCallback()] + [callback for callback in callbacks]
                )
                if callbacks is not None
                else TensorboardCallback()
            ),
        )
        model.save(
            f"{config.TRAINED_MODEL_DIR}/{model_name.upper()}_{total_timesteps // 1000}k_{iter_num}"
        )
        return model

    @staticmethod
    def get_validation_sharpe(account_value):
        """Calculate Sharpe ratio based on validation results"""
        daily_return = pd.Series(account_value, dtype=float).pct_change().dropna()
        # If the agent did not make any transaction
        if daily_return.var() == 0:
            if daily_return.mean() > 0:
                return np.inf
            else:
                return 0.0
        else:
            return (252**0.5) * daily_return.mean() / daily_return.std()

    def __init__(
        self,
        df,
        features,
        train_period,
        val_test_period,
        rebalance_window,
        validation_window,
        env_kwargs=None,
        gamma=0.99,
        norm_obs=False,
    ):
        self.df = df
        self.features = list(features)
        self.train_period = train_period
        self.val_test_period = val_test_period

        self.unique_trade_date = df[
            (df.date > val_test_period[0]) & (df.date <= val_test_period[1])
        ].date.unique()
        self.rebalance_window = rebalance_window
        self.validation_window = validation_window

        # ShareTradingEnv settings (initial_amount, hmax, costs, ew_penalty, ...)
        self.env_kwargs = dict(env_kwargs or {})
        self.gamma = gamma
        self.norm_obs = norm_obs  # normalize observations with the training window's statistics
        self.train_env = None  # defined in train_validation() function

    def DRL_validation(self, model, test_env):
        """validation process: account values of one deterministic run"""
        account_memory, _ = DRLAgent.DRL_prediction(
            model, test_env, obs_normalizer=model.get_vec_normalize_env()
        )
        return account_memory["account_value"]

    def DRL_prediction(self, model, name, iter_num):
        """make a prediction based on trained model"""

        # trading env
        trade_data = data_split(
            self.df,
            start=self.unique_trade_date[iter_num - self.rebalance_window],
            end=self.unique_trade_date[iter_num],
        )
        trade_env = RLUtils.make_share_env(trade_data, self.features, **self.env_kwargs)
        account_memory, _ = DRLAgent.DRL_prediction(
            model, trade_env, obs_normalizer=model.get_vec_normalize_env()
        )
        Path(config.RESULTS_DIR).mkdir(parents=True, exist_ok=True)
        account_memory.to_csv(
            f"{config.RESULTS_DIR}/account_value_trade_{name}_{iter_num}.csv", index=False
        )
        return account_memory

    def _train_window(
        self,
        model_name,
        model_kwargs,
        sharpe_list,
        validation_start_date,
        validation_end_date,
        timesteps_dict,
        i,
        validation,
    ):
        """
        Train the model for a single window.
        """
        if model_kwargs is None:
            # nan keeps the summary table aligned; -inf so a skipped model is never picked
            sharpe_list.append(np.nan)
            return None, sharpe_list, -np.inf

        print(f"======{model_name} Training========")
        model = self.get_model(
            model_name, self.train_env, policy="MultiInputPolicy", model_kwargs=model_kwargs
        )
        model = self.train_model(
            model,
            model_name,
            tb_log_name=f"{model_name}_{i}",
            iter_num=i,
            total_timesteps=timesteps_dict[model_name],
        )  # 100_000
        print(
            f"======{model_name} Validation from: ",
            validation_start_date,
            "to ",
            validation_end_date,
        )
        val_env = RLUtils.make_share_env(validation, self.features, **self.env_kwargs)
        account_value = self.DRL_validation(model=model, test_env=val_env)
        sharpe = self.get_validation_sharpe(account_value)
        print(f"{model_name} Sharpe Ratio: ", sharpe)
        sharpe_list.append(sharpe)
        return model, sharpe_list, sharpe

    def run_ensemble_strategy(
        self,
        A2C_model_kwargs,
        PPO_model_kwargs,
        DDPG_model_kwargs,
        SAC_model_kwargs=None,
        TD3_model_kwargs=None,
        timesteps_dict=None,
    ):
        """Pass `timesteps_dict` by keyword. Models whose kwargs are None are skipped."""
        # Model Parameters
        kwargs = {
            "a2c": A2C_model_kwargs,
            "ppo": PPO_model_kwargs,
            "ddpg": DDPG_model_kwargs,
            "sac": SAC_model_kwargs,
            "td3": TD3_model_kwargs,
        }
        # Model Sharpe Ratios
        model_dct = {k: {"sharpe_list": [], "sharpe": -1} for k in MODELS.keys()}

        """Ensemble Strategy that combines A2C, PPO, DDPG, SAC, and TD3"""
        print("============Start Ensemble Strategy============")
        model_use = []
        validation_start_date_list = []
        validation_end_date_list = []
        iteration_list = []

        start = time.time()
        for i in range(
            self.rebalance_window + self.validation_window,
            len(self.unique_trade_date),
            self.rebalance_window,
        ):
            validation_start_date = self.unique_trade_date[
                i - self.rebalance_window - self.validation_window
            ]
            validation_end_date = self.unique_trade_date[i - self.rebalance_window]

            validation_start_date_list.append(validation_start_date)
            validation_end_date_list.append(validation_end_date)
            iteration_list.append(i)

            print("============================================")
            # Environment Setup starts
            # training env
            train = data_split(
                self.df,
                start=self.train_period[0],
                end=self.unique_trade_date[
                    i - self.rebalance_window - self.validation_window
                ],
            )
            self.train_env = RLUtils.make_share_vec_env(
                train, self.features, self.gamma, norm_obs=self.norm_obs, **self.env_kwargs
            )

            validation = data_split(
                self.df,
                start=self.unique_trade_date[
                    i - self.rebalance_window - self.validation_window
                ],
                end=self.unique_trade_date[i - self.rebalance_window],
            )
            # Environment Setup ends

            # Training and Validation starts
            print(
                "======Model training from: ",
                self.train_period[0],
                "to ",
                self.unique_trade_date[
                    i - self.rebalance_window - self.validation_window
                ],
            )
            # Train Each Model
            for model_name in MODELS.keys():
                # Train The Model
                model, sharpe_list, sharpe = self._train_window(
                    model_name,
                    kwargs[model_name],
                    model_dct[model_name]["sharpe_list"],
                    validation_start_date,
                    validation_end_date,
                    timesteps_dict,
                    i,
                    validation,
                )
                # Save the model's sharpe ratios, and the model itself
                model_dct[model_name]["sharpe_list"] = sharpe_list
                model_dct[model_name]["model"] = model
                model_dct[model_name]["sharpe"] = sharpe

            # Model Selection based on sharpe ratio
            # Same order as MODELS: {"a2c": A2C, "ddpg": DDPG, "td3": TD3, "sac": SAC, "ppo": PPO}
            sharpes = [model_dct[k]["sharpe"] for k in MODELS.keys()]
            # Find the model with the highest sharpe ratio
            max_mod = list(MODELS.keys())[np.argmax(sharpes)]
            model_use.append(max_mod.upper())
            model_ensemble = model_dct[max_mod]["model"]
            # Training and Validation ends

            # Trading starts
            print(
                "======Trading from: ",
                self.unique_trade_date[i - self.rebalance_window],
                "to ",
                self.unique_trade_date[i],
            )
            self.DRL_prediction(model=model_ensemble, name="ensemble", iter_num=i)
            # Trading ends

        end = time.time()
        print("Ensemble Strategy took: ", (end - start) / 60, " minutes")

        df_summary = pd.DataFrame(
            [
                iteration_list,
                validation_start_date_list,
                validation_end_date_list,
                model_use,
                model_dct["a2c"]["sharpe_list"],
                model_dct["ppo"]["sharpe_list"],
                model_dct["ddpg"]["sharpe_list"],
                model_dct["sac"]["sharpe_list"],
                model_dct["td3"]["sharpe_list"],
            ]
        ).T
        df_summary.columns = [
            "Iter",
            "Val Start",
            "Val End",
            "Model Used",
            "A2C Sharpe",
            "PPO Sharpe",
            "DDPG Sharpe",
            "SAC Sharpe",
            "TD3 Sharpe",
        ]

        return df_summary
