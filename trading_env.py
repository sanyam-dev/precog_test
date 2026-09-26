"""Share-trading environment: the agent trades whole shares of each asset.

The trading rules are ported from FinRL's StockTradingEnv
(FinRL-0.3.8/finrl/meta/env_stock_trading/env_stocktrading.py):

* the action is one number per asset in [-1, 1], multiplied by ``hmax`` and rounded
  to whole shares, so with ``hmax=100`` an action of +0.02 buys 2 shares and -0.02 sells 2
  (FinRL truncates instead. That loses a share on 4 of the 201 whole-share targets from
  -100 to 100 in float32, and on 100 of them once the action is converted to float64)
* sells run first and are capped at the shares held
* buying and selling cost a percentage of the traded value

One rule differs from FinRL on purpose. FinRL fills buys largest request first until the
cash runs out, which leaves most of the smaller requests with nothing. Here, when the buys
cost more than the cash left after selling, every buy is scaled by the same fraction and
rounded down to whole shares. Leftover cash then buys one share at a time, first for the
assets furthest below their proportional amount, until no requested share fits. The
executed trades keep the agent's relative preferences, and no buy exceeds its request.

The reward and the observation differ from FinRL. The reward is the portfolio's log
return, minus the optional volatility and equal-weight penalties. Unlike
PortfolioOptimizationEnv in poe.py, it does not subtract the equal-weight portfolio's
return, which the agent cannot influence. The observation holds scale-free features plus
the current portfolio weights, never raw cash, prices or share counts.
"""

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv


class ShareTradingEnv(gym.Env):
    """Trade whole shares of ``n`` assets from a starting cash balance.

    Observation (dict):
        state: features over the lookback window, shape (features, assets, time_window)
        weights: current portfolio weights, cash first, shape (assets + 1,)

    Reward per step:
        log(portfolio return)
        - vol_penalty * volatility of the held weights
        - ew_penalty * (1 - deviation from equal weight)
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        df,
        features,
        initial_amount=1_000_000,
        hmax=100,
        buy_cost_pct=0.001,
        sell_cost_pct=0.001,
        reward_scaling=1.0,
        vol_window=20,
        vol_penalty=0.0,
        ew_penalty=0.0,
        time_window=1,
        price_column="close",
        time_column="date",
        tic_column="tic",
        verbose=True,
    ):
        """
        Args:
            df: Panel with a date column, a tic column, the price column and every feature.
                Every tic needs a row on every date.
            features: Observation columns. Use scale-free columns (z-scores, 0-1 scales).
            initial_amount: Starting cash. With whole shares it has to be large enough for an
                equal-weight budget to buy shares of every asset: at 10,000 across 100 assets,
                21 to 83 of them cost more than their $100 budget, so the default is 1,000,000.
            hmax: Most shares traded per asset per step. The action is scaled by it.
            buy_cost_pct, sell_cost_pct: Trading cost as a fraction of the traded value.
            reward_scaling: Multiplier applied to the reward.
            vol_window: Trailing window, in steps, for the portfolio volatility estimate.
            vol_penalty: Reward penalty per unit of daily portfolio volatility (0 = off).
            ew_penalty: Reward penalty for staying close to equal weight. The penalty is
                ew_penalty at exactly equal weight and 0 with everything in one asset.
            time_window: Number of days of features in each observation.
            verbose: Print a summary at the end of each episode.
        """
        super().__init__()
        self._features = list(features)
        self._initial_amount = float(initial_amount)
        self._hmax = int(hmax)
        self._buy_cost_pct = float(buy_cost_pct)
        self._sell_cost_pct = float(sell_cost_pct)
        self._reward_scaling = float(reward_scaling)
        self._vol_window = int(vol_window)
        self._vol_penalty = float(vol_penalty)
        self._ew_penalty = float(ew_penalty)
        self._time_window = int(time_window)
        self._verbose = verbose

        frame = pd.DataFrame(df).copy()
        frame[time_column] = pd.to_datetime(frame[time_column])
        times = pd.Index(frame[time_column].unique()).sort_values()
        tics = pd.Index(sorted(frame[tic_column].unique()))
        self._sorted_times = list(times)
        self._tic_list = np.asarray(tics)
        self._stock_dim = len(tics)

        def grid(column):
            return (
                frame.pivot(index=time_column, columns=tic_column, values=column)
                .reindex(index=times, columns=tics)
                .to_numpy(dtype=np.float64)
            )

        # (time, features, tics) and (time, tics), built once
        self._state_array = np.stack([grid(f) for f in self._features], axis=1).astype(np.float32)
        self._price_array = grid(price_column)
        missing = int(np.isnan(self._state_array).sum()) + int(np.isnan(self._price_array).sum())
        if missing:
            raise ValueError(
                f"{missing} NaN values after aligning dates and tickers. "
                "Every tic needs a row on every date, with no NaN features or prices."
            )
        if (self._price_array <= 0).any():
            raise ValueError("Prices must be positive.")
        if len(self._sorted_times) < self._time_window + 1:
            raise ValueError("Need at least time_window + 1 dates.")

        n = self._stock_dim
        self.episode_length = len(self._sorted_times) - self._time_window + 1
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(n,), dtype=np.float32)
        self.observation_space = spaces.Dict(
            {
                "state": spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=(len(self._features), n, self._time_window), dtype=np.float32,
                ),
                "weights": spaces.Box(low=0.0, high=1.0, shape=(n + 1,), dtype=np.float32),
            }
        )
        self.reset()

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        n = self._stock_dim
        self._time_index = self._time_window - 1
        self._cash = self._initial_amount
        self._holdings = np.zeros(n, dtype=np.int64)
        self._portfolio_value = self._initial_amount
        self._weights = np.r_[1.0, np.zeros(n)]
        self._cost = 0.0
        self._trades = 0
        self._reward = 0.0
        self._last_trades = np.zeros(n, dtype=np.int64)
        self._last_step = {}
        # memories, named as in PortfolioOptimizationEnv so the same notebook helpers work
        self._asset_memory = {"final": [self._initial_amount]}
        self._date_memory = [self._sorted_times[self._time_index]]
        self._portfolio_return_memory = [0.0]
        self._portfolio_reward_memory = [0.0]
        self._actions_memory = []              # executed trades, in shares
        self._weights_memory = [self._weights.copy()]
        self._turnover_memory = []             # traded value / portfolio value, per step
        self._asset_return_memory = []
        self._vol_memory = [0.0]
        self._ew_deviation_memory = [0.0]
        return self._observation(), self._info()

    def step(self, action):
        if self._time_index >= len(self._sorted_times) - 1:
            if self._verbose:
                self._print_summary()
            self._reward = 0.0
            return self._observation(), 0.0, True, False, self._info()

        prices = self._price_array[self._time_index]
        value_before = self._cash + float(self._holdings @ prices)

        # +k buys k shares, -k sells k shares
        requested = np.rint(np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0) * self._hmax).astype(np.int64)
        executed = self._execute_trades(requested, prices)

        # allocation chosen for the coming day, at today's prices
        value_after_trades = self._cash + float(self._holdings @ prices)
        chosen = np.r_[self._cash, self._holdings * prices] / value_after_trades
        turnover = float(np.abs(executed) @ prices) / value_before

        # the market moves to the next day
        self._time_index += 1
        new_prices = self._price_array[self._time_index]
        self._portfolio_value = self._cash + float(self._holdings @ new_prices)
        self._weights = np.r_[self._cash, self._holdings * new_prices] / self._portfolio_value

        rate_of_return = self._portfolio_value / value_before
        price_relatives = new_prices / prices

        # volatility of the chosen asset weights over the trailing window (cash has none)
        w = chosen[1:]
        self._asset_return_memory.append(price_relatives - 1.0)
        hist = np.asarray(self._asset_return_memory[-self._vol_window:])
        port_vol = float(np.std(hist @ w, ddof=1)) if len(hist) >= 2 else 0.0

        # distance of the chosen asset weights from equal weight:
        # 0 = exactly 1/n in every asset, 1 = everything in one asset
        n = self._stock_dim
        ew_deviation = float(np.abs(w - 1.0 / n).sum() / (2.0 * (1.0 - 1.0 / n))) if n > 1 else 0.0

        reward = (
            np.log(rate_of_return)
            - self._vol_penalty * port_vol
            - self._ew_penalty * (1.0 - ew_deviation)
        )
        self._reward = float(reward) * self._reward_scaling

        self._last_trades = executed
        self._last_step = {
            "requested_trades": requested,
            "turnover": turnover,
            "portfolio_vol": port_vol,
            "ew_deviation": ew_deviation,
        }
        self._asset_memory["final"].append(self._portfolio_value)
        self._date_memory.append(self._sorted_times[self._time_index])
        self._portfolio_return_memory.append(rate_of_return - 1.0)
        self._portfolio_reward_memory.append(self._reward)
        self._actions_memory.append(executed)
        self._weights_memory.append(self._weights.copy())
        self._turnover_memory.append(turnover)
        self._vol_memory.append(port_vol)
        self._ew_deviation_memory.append(ew_deviation)
        return self._observation(), self._reward, False, False, self._info()

    def render(self):
        return None

    # ------------------------------------------------------------------ helpers
    def _execute_trades(self, requested, prices):
        """Sell first, then fill buys pro rata to the cash left. Return the shares traded."""
        executed = np.zeros_like(requested)

        # sells first, capped at the shares held
        sell = np.flatnonzero(requested < 0)
        if sell.size:
            shares = np.minimum(-requested[sell], self._holdings[sell])
            value = shares * prices[sell]
            self._cash += float(value.sum() * (1.0 - self._sell_cost_pct))
            self._cost += float(value.sum() * self._sell_cost_pct)
            self._holdings[sell] -= shares
            executed[sell] = -shares
            self._trades += int(np.count_nonzero(shares))

        # buys next. If they cost more than the cash left, scale every buy by the same
        # fraction and round down, then spend the leftover one share at a time
        buy = np.flatnonzero(requested > 0)
        if buy.size:
            wanted = requested[buy]
            unit_cost = prices[buy] * (1.0 + self._buy_cost_pct)
            total_cost = float(wanted @ unit_cost)
            if total_cost <= self._cash:
                shares = wanted.copy()
            else:
                target = wanted * (self._cash / total_cost)     # proportional amount, in shares
                shares = np.floor(target).astype(np.int64)
                left = self._cash - float(shares @ unit_cost)
                # each pass gives at most one more share per asset, furthest below target first
                while True:
                    fits = np.flatnonzero((shares < wanted) & (unit_cost <= left))
                    if fits.size == 0:
                        break
                    for j in fits[np.argsort(-(target - shares)[fits], kind="stable")]:
                        if unit_cost[j] <= left:
                            shares[j] += 1
                            left -= unit_cost[j]
            self._cash -= float(shares @ unit_cost)
            self._cost += float((shares * prices[buy]).sum() * self._buy_cost_pct)
            self._holdings[buy] += shares
            executed[buy] = shares
            self._trades += int(np.count_nonzero(shares))
        self._cash = max(self._cash, 0.0)   # clear float dust from the cost arithmetic
        return executed

    def _observation(self):
        t = self._time_index
        state = self._state_array[t - self._time_window + 1 : t + 1].transpose(1, 2, 0)
        return {"state": state, "weights": self._weights.astype(np.float32)}

    def _info(self):
        return {
            "date": self._sorted_times[self._time_index],
            "portfolio_value": self._portfolio_value,
            "cash": self._cash,
            "trades": self._last_trades,
            "total_cost": self._cost,
            **self._last_step,
        }

    def _print_summary(self):
        values = np.asarray(self._asset_memory["final"], dtype=float)
        returns = np.asarray(self._portfolio_return_memory[1:], dtype=float)
        sharpe = np.sqrt(252) * returns.mean() / returns.std(ddof=1) if returns.std(ddof=1) > 0 else float("nan")
        max_dd = float((values / np.maximum.accumulate(values) - 1).min())
        print("=================================")
        print(f"Initial portfolio value: {values[0]:.2f}")
        print(f"Final portfolio value: {values[-1]:.2f}")
        print(f"Maximum DrawDown: {max_dd:.4f}")
        print(f"Sharpe ratio: {sharpe:.3f}")
        print(f"Total trading cost: {self._cost:.2f} over {self._trades} trades")
        print(f"Mean daily turnover: {np.mean(self._turnover_memory or [0.0]):.4f}")
        print(f"Mean deviation from equal weight: {np.mean(self._ew_deviation_memory[1:] or [0.0]):.4f}")
        print("=================================")

    def get_sb_env(self, env_number=1):
        """Vectorized version for Stable Baselines 3, plus the first observation."""
        e = DummyVecEnv([lambda: self] * env_number)
        obs = e.reset()
        return e, obs
