# Portfolio Optimization Environment 

"""From FinRL https://github.com/AI4Finance-LLC/FinRL/tree/master/finrl/env"""

import math
from typing import cast

import gym
import matplotlib
import numpy as np
import pandas as pd
from gym import spaces
from gym.utils import seeding

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import DummyVecEnv
from pathlib import Path

try:
    import quantstats as qs
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        """QuantStats module not found, environment can't plot results and calculate indicadors.
        This module is not installed with FinRL. Install by running one of the options:
        pip install quantstats --upgrade --no-cache-dir
        conda install -c ranaroussi quantstats
        """
    )


class PortfolioOptimizationEnv(gym.Env):
    """A portfolio allocantion environment for OpenAI gym.

    This environment simulates the interactions between an agent and the financial market
    based on data provided by a dataframe. The dataframe contains the time series of
    features defined by the user (such as closing, high and low prices) and must have
    a time and a tic column with a list of datetimes and ticker symbols respectively.
    An example of dataframe is shown below::

            date        high            low             close           tic
        0   2020-12-23  0.157414        0.127420        0.136394        ADA-USD
        1   2020-12-23  34.381519       30.074295       31.097898       BNB-USD
        2   2020-12-23  24024.490234    22802.646484    23241.345703    BTC-USD
        3   2020-12-23  0.004735        0.003640        0.003768        DOGE-USD
        4   2020-12-23  637.122803      560.364258      583.714600      ETH-USD
        ... ...         ...             ...             ...             ...

    Based on this dataframe, the environment will create an observation space that can
    be a Dict or a Box. The Box observation space is a three-dimensional array of shape
    (f, n, t), where f is the number of features, n is the number of stocks in the
    portfolio and t is the user-defined time window. If the environment is created with
    the parameter return_last_action set to True, the observation space is a Dict with
    the following keys::

        {
        "state": three-dimensional Box (f, n, t) representing the time series,
        "last_action": one-dimensional Box (n+1,) representing the portfolio weights
        }

    Note that the action space of this environment is an one-dimensional Box with size
    n + 1 because the portfolio weights must contains the weights related to all the
    stocks in the portfolio and to the remaining cash.

    Attributes:
        action_space: Action space.
        observation_space: Observation space.
        episode_length: Number of timesteps of an episode.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        df,
        initial_amount,
        order_df=True,
        return_last_action=False,
        normalize_df="by_previous_time",
        reward_scaling=1,
        comission_fee_model="trf",
        comission_fee_pct=0.0,
        features=["close", "high", "low"],
        valuation_feature="close",
        time_column="date",
        time_format="%Y-%m-%d",
        tic_column="tic",
        time_window=1,
        cwd="./",
        new_gym_api=False,
        vol_window=20,
        vol_penalty=0.0,
        ew_penalty=0.0,
        action_temperature=1.0,
        save_plots=True,
    ):
        """Initializes environment's instance.

        Args:
            df: Dataframe with market information over a period of time.
            initial_amount: Initial amount of cash available to be invested.
            order_df: If True input dataframe is ordered by time.
            return_last_action: If True, observations also return the last performed
                action. Note that, in that case, the observation space is a Dict.
            normalize_df: Defines the normalization method applied to input dataframe.
                Possible values are "by_previous_time", "by_fist_time_window_value",
                "by_COLUMN_NAME" (where COLUMN_NAME must be changed to a real column
                name) and a custom function. If None no normalization is done.
            reward_scaling: A scaling factor to multiply the reward function. This
                factor can help training.
            comission_fee_model: Model used to simulate comission fee. Possible values
                are "trf" (for transaction remainder factor model) and "wvm" (for weights
                vector modifier model). If None, commission fees are not considered.
            comission_fee_pct: Percentage to be used in comission fee. It must be a value
                between 0 and 1.
            features: List of features to be considered in the observation space. The
                items
                of the list must be names of columns of the input dataframe.
            valuation_feature: Feature to be considered in the portfolio value calculation.
            time_column: Name of the dataframe's column that contain the datetimes that
                index the dataframe.
            time_format: Formatting string of time column.
            tic_name: Name of the dataframe's column that contain ticker symbols.
            time_window: Size of time window.
            cwd: Local repository in which resulting graphs will be saved.
            new_gym_api: If True, the environment will use the new gym api standard for
                step and reset methods.
            vol_window: Trailing window, in steps, for the portfolio volatility estimate.
            vol_penalty: Reward penalty per unit of daily portfolio volatility (0 = off).
            ew_penalty: Reward penalty for staying close to equal weight. The penalty is
                ew_penalty at exactly equal weight and 0 with everything in one asset.
            action_temperature: Multiplier applied to raw actions before the softmax.
                Higher values let the agent concentrate its weights more.
            save_plots: If True, save end-of-episode charts and a quantstats snapshot
                to results/rl. Set False for training runs.
        """
        # super(StockEnv, self).__init__()
        # money = 10 , scope = 1
        self._time_window = time_window
        self._time_index = time_window - 1
        self._time_column = time_column
        self._time_format = time_format
        self._tic_column = tic_column
        self._df : pd.DataFrame = df
        self._initial_amount = initial_amount
        self._return_last_action = return_last_action
        self._reward_scaling = reward_scaling
        self._comission_fee_pct = comission_fee_pct
        self._comission_fee_model = comission_fee_model
        self._features = features
        self._valuation_feature = valuation_feature
        self._cwd = Path(cwd)
        self._new_gym_api = new_gym_api
        # volatility: trailing window for the covariance, and reward penalty weight
        self._vol_window = vol_window
        self._vol_penalty = vol_penalty
        # reward penalty for staying close to equal weight (0 = off)
        self._ew_penalty = ew_penalty
        # softmax sharpness applied to raw actions (1 = plain softmax)
        self._action_temperature = action_temperature
        # save end-of-episode charts and the quantstats snapshot
        self._save_plots = save_plots
        # columns that need a day-over-day variation: the features plus the valuation price,
        # so portfolio values stay correct when the price itself is not an observed feature
        self._price_columns = list(dict.fromkeys(list(features) + [valuation_feature]))

        # results file
        self._results_file = self._cwd / "results" / "rl"
        self._results_file.mkdir(parents=True, exist_ok=True)

        # price variation (filled in _preprocess_data before any read)
        self._df_price_variation: pd.DataFrame

        # preprocess data
        self._preprocess_data(order_df, normalize_df)

        # dims and spaces
        self._tic_list = self._df[self._tic_column].unique()
        self._stock_dim = len(self._tic_list)
        action_space = 1 + self._stock_dim

        # sort datetimes and define episode length
        self._sorted_times = sorted(set(self._df[time_column]))
        self.episode_length = len(self._sorted_times) - time_window + 1

        # build the data once as arrays, so each step is a slice instead of a dataframe filter
        self._build_arrays()

        # define action space
        self.action_space = spaces.Box(low=0, high=1, shape=(action_space,))

        # define observation state
        if self._return_last_action:
            # if  last action must be returned, a dict observation
            # is defined
            self.observation_space = spaces.Dict(
                {
                    "state": spaces.Box(
                        low=-np.inf,
                        high=np.inf,
                        shape=(len(self._features), self._stock_dim, self._time_window),
                    ),
                    "last_action": spaces.Box(low=0, high=1, shape=(action_space,)),
                }
            )
        else:
            # if information about last action is not relevant,
            # a 3D observation space is defined
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(len(self._features), self._stock_dim, self._time_window),
            )

        self._reset_memory()

        self._portfolio_value = self._initial_amount
        self._terminal = False

    def step(self, action):
        """Performs a simulation step.

        Args:
            actions: An unidimensional array containing the new portfolio
                weights.

        Note:
            If the environment was created with "return_last_action" set to
            True, the next state returned will be a Dict. If it's set to False,
            the next state will be a Box. You can check the observation state
            through the attribute "observation_space".

        Returns:
            If "new_gym_api" is set to True, the following tuple is returned:
            (state, reward, terminal, truncated, info). If it's set to False,
            the following tuple is returned: (state, reward, terminal, info).

            state: Next simulation state.
            reward: Reward related to the last performed action.
            terminal: If True, the environment is in a terminal state.
            truncated: If True, the environment has passed it's simulation
                time limit. Currently, it's always False.
            info: A dictionary containing informations about the last state.
        """
        self._terminal = self._time_index >= len(self._sorted_times) - 1

        if self._terminal:
            metrics_df = pd.DataFrame(
                {
                    "date": self._date_memory,
                    "returns": self._portfolio_return_memory,
                    "rewards": self._portfolio_reward_memory,
                    "portfolio_values": self._asset_memory["final"],
                    "vol": self._vol_memory,
                }
            )
            metrics_df.set_index("date", inplace=True)

            if self._save_plots:
                self._save_episode_charts(metrics_df)

            print("=================================")
            print("Initial portfolio value:{}".format(self._asset_memory["final"][0]))
            print("Final portfolio value: {}".format(self._portfolio_value))
            print(
                "Final accumulative portfolio value: {}".format(
                    self._portfolio_value / self._asset_memory["final"][0]
                )
            )

            equity = metrics_df["portfolio_values"]
            max_dd = (equity / equity.cummax() - 1).min()
            print("Maximum DrawDown: {}".format(max_dd))

            print("Sharpe ratio: {}".format(qs.stats.sharpe(metrics_df["returns"])))
            print("Mean ex-ante daily vol: {}".format(float(np.mean(self._vol_memory[1:] or [0.0]))))
            print("Mean deviation from equal weight: {}".format(float(np.mean(self._ew_deviation_memory[1:] or [0.0]))))
            print("=================================")

            if self._save_plots:
                qs.plots.snapshot(
                    metrics_df["returns"],
                    show=False,
                    savefig=str(self._results_file / "portfolio_summary.png"),
                    fontname="DejaVu Sans",
                )
            return self._state, self._reward, self._terminal, False, self._info

        else:
            # transform action to numpy array (if it's a list)
            actions = np.array(action, dtype=np.float32)

            # if necessary, normalize weights
            if math.isclose(np.sum(actions), 1, abs_tol=1e-6) and np.min(actions) >= 0:
                weights = actions
            else:
                weights = self._softmax_normalization(actions)

            # save initial portfolio weights for this time step
            self._actions_memory.append(weights)

            # get last step final weights and portfolio_value
            last_weights = self._final_weights[-1]

            # load next state
            self._time_index += 1
            self._state, self._info = self._get_state_and_info_from_time_index(
                self._time_index
            )

            # if using weights vector modifier, we need to modify weights vector
            if self._comission_fee_model == "wvm":
                delta_weights = weights - last_weights
                delta_assets = delta_weights[1:]  # disconsider
                # calculate fees considering weights modification
                fees = np.sum(np.abs(delta_assets * self._portfolio_value))
                if fees > weights[0] * self._portfolio_value:
                    weights = last_weights
                    # maybe add negative reward
                else:
                    portfolio = weights * self._portfolio_value
                    portfolio[0] -= fees
                    self._portfolio_value = np.sum(portfolio)  # new portfolio value
                    weights = portfolio / self._portfolio_value  # new weights
            elif self._comission_fee_model == "trf":
                last_mu = 1
                mu = 1 - 2 * self._comission_fee_pct + self._comission_fee_pct**2
                while abs(mu - last_mu) > 1e-10:
                    last_mu = mu
                    mu = (
                        1
                        - self._comission_fee_pct * weights[0]
                        - (2 * self._comission_fee_pct - self._comission_fee_pct**2)
                        * np.sum(np.maximum(last_weights[1:] - mu * weights[1:], 0))
                    ) / (1 - self._comission_fee_pct * weights[0])
                self._portfolio_value = mu * self._portfolio_value

            # save initial portfolio value of this time step
            self._asset_memory["initial"].append(self._portfolio_value)

            # time passes and time variation changes the portfolio distribution
            portfolio = self._portfolio_value * (weights * self._price_variation)

            # calculate new portfolio value and weights
            self._portfolio_value = np.sum(portfolio)
            weights = portfolio / self._portfolio_value

            # save final portfolio value and weights of this time step
            self._asset_memory["final"].append(self._portfolio_value)
            weights = np.array(weights, dtype=np.float32)
            self._final_weights.append(weights)

            # save date memory
            self._date_memory.append(self._info["end_time"])

            # define portfolio return
            rate_of_return = (
                self._asset_memory["final"][-1] / self._asset_memory["final"][-2]
            )
            portfolio_return = rate_of_return - 1
            ew_rate = float(self._price_variation[1:].mean())

            # volatility of the chosen weights over the trailing window
            # ex-ante: sqrt(w' Σ w) with Σ from the last `vol_window` asset returns (cash has no variance)
            self._asset_return_memory.append(self._price_variation[1:] - 1)
            hist = np.asarray(self._asset_return_memory[-self._vol_window:], dtype=np.float64)
            w = np.asarray(self._actions_memory[-1][1:], dtype=np.float64)
            if len(hist) >= 2:
                cov = np.atleast_2d(np.cov(hist, rowvar=False))
                port_vol = float(np.sqrt(max(w @ cov @ w, 0.0)))
                recent = self._portfolio_return_memory[1:][-(self._vol_window - 1):] + [portfolio_return]
                realized_vol = float(np.std(recent, ddof=1)) if len(recent) >= 2 else 0.0
            else:
                port_vol = realized_vol = 0.0
            vol_change = port_vol - self._vol_memory[-1]
            self._vol_memory.append(port_vol)
            self._realized_vol_memory.append(realized_vol)
            self._info.update(portfolio_vol=port_vol, vol_change=vol_change, realized_vol=realized_vol)

            # distance of the chosen asset weights from equal weight:
            # 0 = exactly 1/n in every asset, 1 = everything in one asset
            n = self._stock_dim
            ew_deviation = (
                float(np.abs(w - 1.0 / n).sum() / (2.0 * (1.0 - 1.0 / n))) if n > 1 else 0.0
            )
            self._ew_deviation_memory.append(ew_deviation)
            self._info.update(ew_deviation=ew_deviation)

            portfolio_reward = (
                np.log(rate_of_return)
                - np.log(ew_rate)
                - self._vol_penalty * port_vol
                - self._ew_penalty * (1.0 - ew_deviation)
            )

            # save portfolio return memory
            self._portfolio_return_memory.append(portfolio_return)
            self._portfolio_reward_memory.append(portfolio_reward)

            # Define portfolio return
            self._reward = portfolio_reward
            self._reward = self._reward * self._reward_scaling

            return self._state, self._reward, self._terminal, False, self._info

    def reset(self, *, seed=None, options=None):
        """Resets the environment and returns it to its initial state (the
        fist date of the dataframe).

        Note:
            If the environment was created with "return_last_action" set to
            True, the initial state will be a Dict. If it's set to False,
            the initial state will be a Box. You can check the observation
            state through the attribute "observation_space".

        Returns:
            If "new_gym_api" is set to True, the following tuple is returned:
            (state, info). If it's set to False, only the initial state is
            returned.

            state: Initial state.
            info: Initial state info.
        """
        # time_index must start a little bit in the future to implement lookback
        self._time_index = self._time_window - 1
        self._reset_memory()

        self._state, self._info = self._get_state_and_info_from_time_index(
            self._time_index
        )
        self._portfolio_value = self._initial_amount
        self._terminal = False

        return self._state, self._info

    def _get_state_and_info_from_time_index(self, time_index):
        """Gets state and information given a time index. It also updates "data"
        attribute with information about the current simulation step.

        Args:
            time_index: An integer that represents the index of a specific datetime.
                The initial datetime of the dataframe is given by 0.

        Note:
            If the environment was created with "return_last_action" set to
            True, the returned state will be a Dict. If it's set to False,
            the returned state will be a Box. You can check the observation
            state through the attribute "observation_space".

        Returns:
            A tuple with the following form: (state, info).

            state: The state of the current time index. It can be a Box or a Dict.
            info: A dictionary with some informations about the current simulation
                step. The dict has the following keys::

                {
                "tics": List of ticker symbols,
                "start_time": Start time of current time window,
                "start_time_index": Index of start time of current time window,
                "end_time": End time of current time window,
                "end_time_index": Index of end time of current time window,
                "price_variation": Price variation of current time step
                }
        """
        # returns state in form (channels, tics, timesteps)
        end_time = self._sorted_times[time_index]
        start_time = self._sorted_times[time_index - (self._time_window - 1)]

        # price variation of this time step (index 0 is cash)
        self._price_variation = np.insert(self._price_array[time_index], 0, 1)

        # state (features, tics, time_window) sliced from the precomputed array
        start_index = time_index - (self._time_window - 1)
        state = self._state_array[start_index : time_index + 1].transpose((1, 2, 0))
        info = {
            "tics": self._tic_list,
            "start_time": start_time,
            "start_time_index": time_index - (self._time_window - 1),
            "end_time": end_time,
            "end_time_index": time_index,
            "price_variation": self._price_variation,
        }
        return self._standardize_state(state), info

    def _build_arrays(self):
        """Precompute the state and price-variation arrays once.

        _state_array has shape (time, features, tics) and _price_array has shape
        (time, tics), both in the order of _sorted_times and _tic_list.
        """
        times = pd.Index(self._sorted_times)
        tics = pd.Index(self._tic_list)
        per_feature = [
            self._df.pivot(index=self._time_column, columns=self._tic_column, values=f)
            .reindex(index=times, columns=tics)
            .to_numpy(dtype=np.float32)
            for f in self._features
        ]
        self._state_array = np.stack(per_feature, axis=1)
        self._price_array = (
            self._df_price_variation.pivot(
                index=self._time_column, columns=self._tic_column, values=self._valuation_feature
            )
            .reindex(index=times, columns=tics)
            .to_numpy(dtype=np.float32)
        )
        missing_state = int(np.isnan(self._state_array).sum())
        missing_price = int(np.isnan(self._price_array).sum())
        if missing_state or missing_price:
            raise ValueError(
                f"{missing_state} NaN feature values and {missing_price} NaN prices after aligning "
                "dates and tickers. Every tic needs a row on every date, with no NaN features."
            )

    def _save_episode_charts(self, metrics_df):
        """Save the end-of-episode charts, each on its own figure.

        Explicit figures keep these charts off any figure the caller has open,
        such as a notebook plot that is being built around a rollout.
        """
        fig, ax = plt.subplots()
        ax.plot(metrics_df["portfolio_values"], "r")
        ax.set(title="Portfolio Value Over Time", xlabel="Time", ylabel="Portfolio value")
        fig.savefig(self._results_file / "portfolio_value.png")
        plt.close(fig)

        fig, ax = plt.subplots()
        ax.plot(self._portfolio_reward_memory, "r")
        ax.set(title="Reward Over Time", xlabel="Time", ylabel="Reward")
        fig.savefig(self._results_file / "reward.png")
        plt.close(fig)

        fig, ax = plt.subplots()
        ax.plot(self._actions_memory)
        ax.set(title="Actions performed", xlabel="Time", ylabel="Weight")
        fig.savefig(self._results_file / "actions.png")
        plt.close(fig)

    def render(self, mode="human"):
        """Renders the environment.

        Returns:
            Observation of current simulation step.
        """
        return self._state

    def _softmax_normalization(self, actions):
        """Normalizes the action vector using a softmax scaled by action_temperature.

        Returns:
            Normalized action vector (portfolio vector).
        """
        logits = self._action_temperature * actions
        if logits.max() > 50:  # keep exp() finite for large inputs
            logits = logits - logits.max()
        numerator = np.exp(logits)
        return numerator / np.sum(numerator)

    def enumerate_portfolio(self):
        """Enumerates the current porfolio by showing the ticker symbols
        of all the investments considered in the portfolio.
        """
        print("Index: 0. Tic: Cash")
        for index, tic in enumerate(self._tic_list):
            print("Index: {}. Tic: {}".format(index + 1, tic))

    def _preprocess_data(self, order, normalize):
        """Orders and normalizes the environment's dataframe.

        Args:
            order: If true, the dataframe will be ordered by ticker list
                and datetime.
            normalize: Defines the normalization method applied to the dataframe.
                Possible values are "by_previous_time", "by_fist_time_window_value",
                "by_COLUMN_NAME" (where COLUMN_NAME must be changed to a real column
                name) and a custom function. If None no normalization is done.
        """
        # order time dataframe by tic and time
        if order:
            self._df = self._df.sort_values(by=[self._tic_column, self._time_column])
        # defining price variation after ordering dataframe
        self._df_price_variation = self._temporal_variation_df(columns=self._price_columns)
        # apply normalization
        if normalize:
            self._normalize_dataframe(normalize)
        # transform str to datetime
        self._df[self._time_column] = pd.to_datetime(self._df[self._time_column])
        self._df_price_variation[self._time_column] = pd.to_datetime(
            self._df_price_variation[self._time_column]
        )
        # transform numeric variables to float32 (compatibility with pytorch)
        self._df[self._features] = self._df[self._features].astype("float32")
        self._df_price_variation[self._price_columns] = self._df_price_variation[
            self._price_columns
        ].astype("float32")

    def _reset_memory(self):
        """Resets the environment's memory."""
        date_time = self._sorted_times[self._time_index]
        # memorize portfolio value each step
        self._asset_memory = {
            "initial": [self._initial_amount],
            "final": [self._initial_amount],
        }
        # memorize portfolio return and reward each step
        self._portfolio_return_memory = [0]
        self._portfolio_reward_memory = [0]
        # initial action: all money is allocated in cash
        self._actions_memory = [np.array([1] + [0] * self._stock_dim, dtype=np.float32)]
        # memorize portfolio weights at the ending of time step
        self._final_weights = [np.array([1] + [0] * self._stock_dim, dtype=np.float32)]
        # memorize datetimes
        self._date_memory = [date_time]
        # memorize asset returns and portfolio volatility (ex-ante and realized)
        self._asset_return_memory = []
        self._vol_memory = [0.0]
        self._realized_vol_memory = [0.0]
        # memorize how far the chosen weights are from equal weight
        self._ew_deviation_memory = [0.0]

    def _standardize_state(self, state):
        """Standardize the state given the observation space. If "return_last_action"
        is set to False, a three-dimensional box is returned. If it's set to True, a
        dictionary is returned. The dictionary follows the standard below::

            {
            "state": Three-dimensional box representing the current state,
            "last_action": One-dimensional box representing the last action
            }
        """
        last_action = self._actions_memory[-1]
        if self._return_last_action:
            return {"state": state, "last_action": last_action}
        else:
            return state

    def _normalize_dataframe(self, normalize):
        """ "Normalizes the environment's dataframe.

        Args:
            normalize: Defines the normalization method applied to the dataframe.
                Possible values are "by_previous_time", "by_fist_time_window_value",
                "by_COLUMN_NAME" (where COLUMN_NAME must be changed to a real column
                name) and a custom function. If None no normalization is done.

        Note:
            If a custom function is used in the normalization, it must have an
            argument representing the environment's dataframe.
        """
        if type(normalize) == str:
            if normalize == "by_fist_time_window_value":
                print(
                    "Normalizing {} by first time window value...".format(
                        self._features
                    )
                )
                self._df = self._temporal_variation_df(self._time_window - 1)
            elif normalize == "by_previous_time":
                print("Normalizing {} by previous time...".format(self._features))
                self._df = self._temporal_variation_df()
            elif normalize.startswith("by_"):
                normalizer_column = normalize[3:]
                print("Normalizing {} by {}".format(self._features, normalizer_column))
                for column in self._features:
                    self._df[column] = self._df[column] / self._df[normalizer_column]
        elif callable(normalize):
            print("Applying custom normalization function...")
            self._df = cast(pd.DataFrame, normalize(self._df))
        else:
            print("No normalization was performed.")

    def _temporal_variation_df(self, periods=1, columns=None):
        """Calculates the temporal variation dataframe. For each feature, this
        dataframe contains the rate of the current feature's value and the last
        feature's value given a period. It's used to normalize the dataframe.

        Args:
            periods: Periods (in time indexes) to calculate temporal variation.
            columns: Columns to vary. Defaults to the observed features.

        Returns:
            Temporal variation dataframe.
        """
        columns = self._features if columns is None else columns
        df_temporal_variation = self._df.copy()
        prev_columns = []
        for column in columns:
            prev_column = "prev_{}".format(column)
            prev_columns.append(prev_column)
            df_temporal_variation[prev_column] = df_temporal_variation.groupby(
                self._tic_column
            )[column].shift(periods=periods)
            df_temporal_variation[column] = (
                df_temporal_variation[column] / df_temporal_variation[prev_column]
            )
        df_temporal_variation = (
            df_temporal_variation.drop(columns=prev_columns)
            .fillna(1)
            .reset_index(drop=True)
        )
        return df_temporal_variation

    def _seed(self, seed=None):
        """Seeds the sources of randomness of this environment to guarantee
        reproducibility.

        Args:
            seed: Seed value to be applied.

        Returns:
            Seed value applied.
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_sb_env(self, env_number=1):
        """Generates an environment compatible with Stable Baselines 3. The
        generated environment is a vectorized version of the current one.

        Returns:
            A tuple with the generated environment and an initial observation.
        """
        e = DummyVecEnv([lambda: self] * env_number)  # type: ignore[arg-type]
        obs = e.reset()
        return e, obs