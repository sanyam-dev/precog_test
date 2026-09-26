# Precog

Portfolio research on 100 anonymized assets. Prices live in `data/anonymized_data/`. Indicators, signals, and scaled features are built once, then used for a next-close forecast, a pairwise similarity score, and reinforcement-learning allocation.

## Algorithm

1. **Features.** `Utils` loads the CSVs and builds stockstats indicators per asset (RSI, MACD, ADX, Bollinger, and so on), plus rule signals. Trailing 10-session z-scores and min-max scales replace raw price and volume so later models do not see the level of each asset. The panel is `data/features_panel.csv`. `RLUtils` adds the day of week and the T-bill return and writes `data/rl_features.csv`.

2. **Next-close forecast.** In `task2.ipynb` the target is each asset's next close. Near-duplicate columns are dropped, PCA is fit on the first 80% of dates, and a pooled LightGBM model is scored with expanding date folds. The scaler and PCA are refit inside each training fold.

3. **Pair score.** For the last 63 sessions (about three months), each asset is a matrix of sessions by features. A pair's correlation vector has one Pearson correlation per feature. The score is the magnitude of that vector. Raw prices are left out. A feature that is missing inside the window is left out of the vector.

4. **Allocation.** `task3.ipynb` trains PPO and DDPG in `PortfolioOptimizationEnv` (`poe.py`). The action is a portfolio weight vector. Optuna tunes on 25 assets, then the best settings are retrained on all assets. The baseline is equal weight. Trial logs go under `logs/`.

`trading_env.py` is a separate share-trading environment: whole shares, sells first, buys scaled to the cash left. Its reward is log return minus the risk-free log return.

## Files

| Path | Role |
| --- | --- |
| `utils.py` | Indicators, signals, PCA, LightGBM walk-forward, pair scores |
| `RLUtils.py` | RL feature list, env setup, Optuna tuning, episode metrics |
| `poe.py` | Weight-allocation environment |
| `trading_env.py` | Whole-share trading environment |
| `models.py` | Stable-Baselines agents used with the share env |
| `task2.ipynb` | Feature panel, forecast, pair scores |
| `task3.ipynb` | PPO and DDPG versus equal weight |
| `task 1/` | Early signal and simulation notebook |
| `task4/` | Rolling DCC-GARCH correlation. See `task4/README.md` |
| `task5/` | Match each anonymized asset to an S&P 500 ticker. See `task5/README.md` |

Task 5 correlates daily log returns and assigns each asset to one ticker (Hungarian algorithm), then checks that anonymized price over real price is a constant. Task 4 fits DCC-GARCH on a backward window and records the one-step-ahead correlation.
