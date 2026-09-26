import pandas as pd
import os
from stockstats import wrap
from stockstats import StockDataFrame as sdf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Chart colors: one fixed color per series, blue <-> gray <-> red for signed values.
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
MUTED = "#52514e"
DIVERGING = LinearSegmentedColormap.from_list("blue_gray_red", ["#104281", "#f0efec", "#b3261e"])

class Utils:

    FEATURE_LIST = [
        "log-ret", # daily log retur
        "return", # daily return
        "boll_ub", # Bollinger upper band (20 SMA + 2 std)
        "boll_lb", # Bollinger lower band (20 SMA - 2 std)
        "rsi_6",  # 6 day RSI
        "rsi",
        "cci_6",
        "cci",
        "adx",
        "close_5_sma",
        "close_20_ema",
        "close_5_ema",
        "cci_5",
        "adx",
        "pdi",
        "ndi",
        "stochrsi_3",
        "macds",
        "macd",
        "macdh",
        "tema",
        "pvo"
        ]

    SIGNAL_COLS = ["macd_signal", "rsi_signal", "adx_signal", "cci_signal", "boll_signal"]


    @staticmethod
    def create_df(path: str) -> pd.DataFrame:
        df = pd.DataFrame()
        df['tic'] = []
        df['date'] = []
        parts = []
        for files in os.listdir(path):
            adf = pd.read_csv(os.path.join(path, files))
            adf['tic'] = [files[:-4] for _ in range(adf.shape[0])]
            adf.columns = adf.columns.str.lower()
            adf['date'] = pd.to_datetime(adf['date'])
            # adf = adf.set_index('date')
            parts.append(adf)
        
        df = pd.concat(parts)
        df = df.sort_values(by=['date', 'tic'])
        df = df.reset_index()
        return wrap(df)

    @staticmethod
    def train_test_split(
        df: pd.DataFrame,
        train_window: int,
        test_window: int,
        date_col: str = "date",
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split a price dataframe into train and test by trading sessions.

        A trading session is one unique date present in the data.
        Train is the first `train_window` sessions. Test is the next
        `test_window` sessions. `date` may be a column or the index
        (as after stockstats.wrap).
        """
        if date_col in df.columns:
            dates = pd.to_datetime(df[date_col])
        elif df.index.name == date_col or isinstance(df.index, pd.DatetimeIndex):
            dates = pd.Series(pd.to_datetime(df.index), index=df.index)
        else:
            raise KeyError(f"No '{date_col}' column or datetime index on dataframe")

        sessions = pd.Index(dates.unique()).sort_values()
        needed = train_window + test_window
        if len(sessions) < needed:
            raise ValueError(
                f"Need {needed} trading sessions, dataframe has {len(sessions)}"
            )

        train_dates = sessions[:train_window]
        test_dates = sessions[train_window:needed]
        train = df.loc[dates.isin(train_dates)].copy()
        test = df.loc[dates.isin(test_dates)].copy()
        return train, test

    @staticmethod
    def get_first_n_assets(df:pd.DataFrame, n: int):
        assets = df['tic'].unique()[:n]
        return wrap(df[df['tic'].isin(assets)])

    @staticmethod
    def rule_signals(df: pd.DataFrame) -> pd.DataFrame:
        """Signal columns for the rules. Each ticker is calculated on its own past.

        Windows are the stockstats defaults, except rsi_6 and the 20-session
        volume average, which are the windows named in the rules.
        A signal is published only when volume is above that average.
        """
        frame = pd.DataFrame(df.reset_index())
        if "Date" in frame.columns and "date" not in frame.columns:
            frame = frame.rename(columns={"Date": "date"})
        if "tic" not in frame.columns:
            return Utils._rule_signals_one(frame)

        parts = []
        for tic, group in frame.groupby("tic", sort=False):
            part = Utils._rule_signals_one(group)
            part["tic"] = tic
            parts.append(part)
        out = pd.concat(parts)
        return out.sort_values("tic", kind="mergesort").sort_index(kind="mergesort")

    @staticmethod
    def _hold(up: pd.Series, down: pd.Series) -> pd.Series:
        state = 0
        values = []
        for crossed_up, crossed_down in zip(up.fillna(False), down.fillna(False)):
            if crossed_up:
                state = 1
            elif crossed_down:
                state = -1
            values.append(state)
        return pd.Series(values, index=up.index, dtype="int8")

    @staticmethod
    def _side(long: pd.Series, short: pd.Series) -> pd.Series:
        signal = pd.Series(0, index=long.index, dtype="int8")
        long = long.fillna(False)
        short = short.fillna(False)
        signal = signal.mask(long & ~short, 1)
        signal = signal.mask(short & ~long, -1)
        return signal

    @staticmethod
    def _rule_signals_one(frame: pd.DataFrame) -> pd.DataFrame:
        work = frame.sort_values("date") if "date" in frame.columns else frame
        sdf = wrap(work)
        for column in ("adx", "pdi", "ndi", "rsi", "rsi_6", "cci", "macd", "macds", "stochrsi"):
            _ = sdf[column]
        px = pd.DataFrame(sdf)
        volume_20_sma = px["volume"].rolling(20, min_periods=20).mean()
        confirmed = px["volume"] > volume_20_sma

        adx, pdi, ndi = px["adx"], px["pdi"], px["ndi"]
        rsi, rsi_6, cci = px["rsi"], px["rsi_6"], px["cci"]
        macd, macds, stoch = px["macd"], px["macds"], px["stochrsi"]
        prev_pdi, prev_ndi = pdi.shift(1), ndi.shift(1)
        prev_macd, prev_macds = macd.shift(1), macds.shift(1)
        prev_cci, prev_stoch = cci.shift(1), stoch.shift(1)

        di_up = (pdi > ndi) & (prev_pdi <= prev_ndi) & (adx > 25) & confirmed
        di_down = (pdi < ndi) & (prev_pdi >= prev_ndi) & (adx > 25) & confirmed
        macd_up = (macd > macds) & (prev_macd <= prev_macds) & (rsi > 50) & (rsi <= 70) & confirmed
        macd_down = (macd < macds) & (prev_macd >= prev_macds) & (rsi < 50) & (rsi >= 30) & confirmed

        ranging, trending = adx < 20, adx > 25
        regime_long = (ranging & ((rsi < 30) | (cci < -100))) | (trending & (pdi > ndi))
        regime_short = (ranging & ((rsi > 70) | (cci > 100))) | (trending & (pdi < ndi))

        cci_up = (cci > 100) & (prev_cci <= 100) & (adx > 25)
        cci_down = (cci < -100) & (prev_cci >= -100) & (adx > 25)
        uptrend, downtrend = macd > 0, macd < 0
        stoch_uptrend = uptrend | (pdi > ndi)
        stoch_downtrend = downtrend | (pdi < ndi)
        stoch_up = (stoch > 20) & (prev_stoch <= 20) & stoch_uptrend & ~stoch_downtrend
        stoch_down = (stoch < 80) & (prev_stoch >= 80) & stoch_downtrend & ~stoch_uptrend

        signals = pd.DataFrame(
            {
                "volume_20_sma": volume_20_sma,
                "adx_di_cross": Utils._hold(di_up, di_down).where(confirmed, 0),
                "adx_regime": Utils._side(regime_long & confirmed, regime_short & confirmed),
                "macd_rsi": Utils._hold(macd_up, macd_down).where(confirmed, 0),
                "trend_pullback": Utils._side((uptrend & (rsi_6 < 30) & confirmed), (downtrend & (rsi_6 > 70) & confirmed)),
                "cci_adx": Utils._side(cci_up & confirmed, cci_down & confirmed),
                "stochrsi_trend": Utils._side(stoch_up & confirmed, stoch_down & confirmed),
            },
            index=px.index, 
        )
        return signals



    @staticmethod
    def get_features(df: pd.DataFrame):
        derived = ['residual_return', 'return']
        df = df.reset_index()
        parts = []
        stock_cols = [col for col in Utils.FEATURE_LIST if col not in derived]
        features : sdf = wrap(df)
        for tic, group in features.groupby("tic", sort=False):
            a = wrap(group.sort_values(by='date'))
            _ = a[stock_cols]
            # a['residual_return'] = Utils.get_residual_return(a)
            part=pd.DataFrame(a).reset_index()
            part['tic'] = tic
            parts.append(part)
        
        out = pd.concat(parts)
        out = out.sort_values(['date', 'tic'])
        out.drop(columns=['index'], inplace=True)
        return out

    @staticmethod
    def rolling_standardize(x, window=10):
        """Z-score against the trailing `window`-day mean and std (no look-ahead)."""
        roll = x.rolling(window)
        return (x - roll.mean()) / roll.std()

    @staticmethod
    def rolling_minmax(x, window=10):
        """Scale to [0, 1] with the trailing `window`-day min and max (no look-ahead)."""
        roll = x.rolling(window)
        lo, hi = roll.min(), roll.max()
        return (x - lo) / (hi - lo).replace(0, np.nan)

    # A trading year is ~252 sessions, so three months is 63 sessions.
    CORR_WINDOW = 63
    # Price levels repeat the same path (close, bands, moving averages). They are
    # not part of the feature matrix unless the caller passes them in `columns`.
    _PRICE_LEVELS = {
        "open", "high", "low", "close", "volume",
        "boll", "boll_ub", "boll_lb",
        "close_5_sma", "close_20_ema", "close_5_ema", "tema",
    }

    @staticmethod
    def _feature_columns(features, columns):
        if columns is not None:
            return list(columns)
        numeric = [
            c for c in features.columns
            if c not in {"date", "tic"} and pd.api.types.is_numeric_dtype(features[c])
        ]
        chosen = [c for c in numeric if c not in Utils._PRICE_LEVELS]
        return chosen or numeric

    @staticmethod
    def _feature_cube(features, columns):
        """Date x asset x feature array, dates and assets sorted.

        Each feature is placed on the same dates and assets. A feature with no
        value on a date stays missing there, instead of dropping that date and
        changing the matrix shape.
        """
        work = features[["date", "tic", *columns]].replace([np.inf, -np.inf], np.nan).copy()
        work["date"] = pd.to_datetime(work["date"])
        dates = pd.Index(pd.unique(work["date"])).sort_values()
        assets = pd.Index(sorted(pd.unique(work["tic"])))
        wide = []
        for column in columns:
            panel = (
                work.pivot_table(index="date", columns="tic", values=column, aggfunc="last")
                .reindex(index=dates, columns=assets)
            )
            wide.append(panel.to_numpy(dtype=float))
        cube = np.stack(wide, axis=-1)  # dates x assets x features
        return dates, list(assets), cube

    @staticmethod
    def _feature_corr_vectors(block):
        """Per-feature asset correlation for one window.

        `block` is (sessions, assets, features). Returns (vectors, scores)
        with vectors shaped (assets, assets, features). A feature that is
        constant or missing for an asset is left out of that asset's vector.
        The score is the Euclidean magnitude of the finite part of the vector.
        """
        sessions, n_assets, n_features = block.shape
        vectors = np.full((n_assets, n_assets, n_features), np.nan)
        for f in range(n_features):
            column = block[:, :, f]
            if np.isnan(column).any():
                continue
            centered = column - column.mean(axis=0)
            std = np.sqrt((centered ** 2).sum(axis=0) / (sessions - 1))
            valid = std > 0
            if valid.sum() < 2:
                continue
            standardized = np.zeros_like(centered)
            standardized[:, valid] = centered[:, valid] / std[valid]
            corr = (standardized.T @ standardized) / (sessions - 1)
            corr[~valid, :] = np.nan
            corr[:, ~valid] = np.nan
            np.fill_diagonal(corr, np.where(valid, 1.0, np.nan))
            vectors[:, :, f] = corr
        scores = np.sqrt(np.nansum(vectors ** 2, axis=2))
        return vectors, scores

    @staticmethod
    def feature_pair_correlation(features, columns=None, window=None):
        """Correlation of asset pairs from each asset's trailing feature matrix.

        For a window of `window` sessions (default 63, about three months),
        each asset is a matrix of shape (sessions, features). For every pair,
        the correlation vector has one Pearson correlation per feature, using
        only dates inside that window. The pair score is the magnitude of
        that vector. A feature with a missing value in the window is left out of the vector.

        Returns (scores, vectors, history). `scores` is the latest asset x
        asset matrix of magnitudes. `vectors` is the latest correlation
        vector for each unordered pair, one column per feature. `history`
        is the mean off-diagonal score each time the window is full.
        """
        window = Utils.CORR_WINDOW if window is None else window
        columns = Utils._feature_columns(features, columns)
        dates, assets, cube = Utils._feature_cube(features, columns)
        n_assets = len(assets)
        upper = np.triu_indices(n_assets, k=1)
        means, mean_dates, latest_vectors, latest_scores = [], [], None, None
        for end in range(window, len(dates) + 1):
            vectors, scores = Utils._feature_corr_vectors(cube[end - window:end])
            if np.all(np.isnan(vectors)):
                continue
            means.append(float(scores[upper].mean()))
            mean_dates.append(dates[end - 1])
            latest_vectors, latest_scores = vectors, scores
        if latest_scores is None:
            raise ValueError(f"No complete {window}-session feature window")

        scores = pd.DataFrame(latest_scores, index=assets, columns=assets)
        end = pd.Timestamp(mean_dates[-1])
        start = pd.Timestamp(dates[dates.get_loc(mean_dates[-1]) - window + 1])
        scores.attrs.update(end=end, start=start, window=window, features=list(columns))

        left, right = upper
        vectors = pd.DataFrame(
            latest_vectors[left, right, :],
            index=pd.MultiIndex.from_arrays([np.array(assets)[left], np.array(assets)[right]], names=["asset_a", "asset_b"]),
            columns=list(columns),
        )
        history = pd.Series(means, index=pd.Index(mean_dates, name="date"), name="mean_score")
        return scores, vectors, history

    @staticmethod
    def asset_corr_pairs(corr, column="score"):
        """Unique asset pairs from a score matrix, highest score first."""
        labels = corr.columns.to_numpy()
        left, right = np.triu_indices(len(labels), k=1)
        pairs = pd.DataFrame({
            "asset_a": labels[left],
            "asset_b": labels[right],
            column: corr.to_numpy()[left, right],
        })
        return pairs.sort_values(column, ascending=False).reset_index(drop=True)

    @staticmethod
    def plot_asset_corr(corr):
        """Heatmap of pair scores. The diagonal is an asset with itself and is left blank."""
        shown = corr.to_numpy(dtype=float).copy()
        np.fill_diagonal(shown, np.nan)
        end = pd.Timestamp(corr.attrs.get("end"))
        start = pd.Timestamp(corr.attrs.get("start"))
        window = corr.attrs.get("window", Utils.CORR_WINDOW)
        n_features = len(corr.attrs.get("features", []))
        fig, ax = plt.subplots(figsize=(8, 6.5))
        image = ax.imshow(shown, cmap="Blues", vmin=0, vmax=np.nanmax(shown), aspect="equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(
            f"Pair score from {n_features} features\n"
            f"trailing {window} sessions, {start:%Y-%m-%d} to {end:%Y-%m-%d}"
        )
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Magnitude of correlation vector")
        fig.tight_layout()
        plt.show()
        return ax


    @staticmethod
    def get_macd_signal(macd, macds, macdh):
        rising  = macdh > macdh.shift(1) 
        falling = macdh < macdh.shift(1)

        buy  = (macd > macds) & rising      # above signal, momentum growing
        sell = (macd < macds) & falling     # below signal, momentum falling

        signal = pd.Series(0, index=macd.index, dtype="int8")
        signal = signal.mask(buy, 1)
        signal = signal.mask(sell, -1)
        return signal

    @staticmethod
    def get_rsi_signal(rsi, rsi_6):
        oversold = 30
        overbought = 70

        prev_fast, prev_slow = rsi_6.shift(1), rsi.shift(1)        
        rsi_crossover_up = ((rsi < rsi_6) & (prev_fast > prev_slow))
        rsi_crossover_down = ((rsi > rsi_6) & (prev_fast < prev_slow))
        
        bull_zone = (rsi_6 > 50) & (rsi > 50)
        bear_zone = (rsi_6 < 50) & (rsi < 50)

        buy = rsi_crossover_up & bull_zone
        sell = rsi_crossover_down & bear_zone

        oversold_exit = (rsi_6 > oversold) & (prev_fast <= oversold)
        overbought_exit = (rsi_6 < overbought) & (prev_fast >= overbought)
        long = buy | oversold_exit
        short = sell | overbought_exit
        signal = pd.Series(0, index=rsi.index, dtype="int8")
        signal = signal.mask(long & ~short, 1)
        signal = signal.mask(short & ~long, -1)
        return signal

    @staticmethod
    def get_adx_signal(adx, pdi=None, ndi=None, slope_win=3):
        if isinstance(adx, pd.DataFrame):
            df = adx
            if "tic" in df.columns:
                return pd.concat(
                    Utils.get_adx_signal(g["adx"], g["pdi"], g["ndi"], slope_win)
                    for _, g in df.groupby("tic", sort=False)
                )
            adx, pdi, ndi = df["adx"], df["pdi"], df["ndi"]
        slope_up = adx.diff(slope_win) > 0      # 3-day slope: fewer 1-day flips

        buy  = (pdi > ndi) & slope_up           # uptrend strengthening
        sell = (ndi > pdi) & slope_up           # downtrend strengthening

        signal = pd.Series(0, index=adx.index, dtype="int8")
        signal = signal.mask(buy, 1)
        signal = signal.mask(sell, -1)
        return signal

    @staticmethod
    def get_cci_signal(cci, pdi, ndi):
        """Buy when CCI > 100 and +DI > −DI. Sell when CCI < −100 and −DI > +DI."""
        signal = pd.Series(0, index=cci.index, dtype="int8")
        signal = signal.mask((cci > 100) & (pdi > ndi), 1)
        signal = signal.mask((cci < -100) & (ndi > pdi), -1)
        return signal

    @staticmethod
    def get_boll_signal(close, boll_ub, boll_lb):
        """Investopedia reading of Bollinger Bands.

        Close touches or falls below the lower band -> oversold -> buy (+1).
        Close touches or rises above the upper band -> overbought -> sell (-1).
        """
        signal = pd.Series(0, index=close.index, dtype="int8")
        signal = signal.mask(close <= boll_lb, 1)
        signal = signal.mask(close >= boll_ub, -1)
        return signal

    @staticmethod
    def add_signals(features: pd.DataFrame) -> pd.DataFrame:
        """Write macd/rsi/adx/cci/boll signal columns onto features, one ticker at a time.

        Each `<name>_signal_held` column holds +1 from a buy until the next sell,
        and -1 from a sell until the next buy (0 before the first signal).
        """
        groups = features.groupby("tic", sort=False) if "tic" in features.columns else [(None, features)]
        parts = []
        for _, group in groups:
            g = group.sort_values("date").copy() if "date" in group.columns else group.copy()
            g["macd_signal"] = Utils.get_macd_signal(g["macd"], g["macds"], g["macdh"]).to_numpy()
            g["rsi_signal"] = Utils.get_rsi_signal(g["rsi"], g["rsi_6"]).to_numpy()
            g["adx_signal"] = Utils.get_adx_signal(g["adx"], g["pdi"], g["ndi"]).to_numpy()
            g["cci_signal"] = Utils.get_cci_signal(g["cci"], g["pdi"], g["ndi"]).to_numpy()
            g["boll_signal"] = Utils.get_boll_signal(g["close"], g["boll_ub"], g["boll_lb"]).to_numpy()
            for col in Utils.SIGNAL_COLS:
                g[col + "_held"] = Utils.hold_until_opposite(g[col]).to_numpy()
            parts.append(g)
        out = pd.concat(parts)
        if {"date", "tic"}.issubset(out.columns):
            out = out.sort_values(["date", "tic"])
        return out.reset_index(drop=True)

    @staticmethod
    def eval_signal(features, col, cost=0.001):
        """Equal-weight daily PnL of a ±1/0 signal column.

        Signal on day t is held on day t+1. Cost is charged on |Δheld|.
        """
        f = features.sort_values(["tic", "date"]).copy()
        g = f.groupby("tic", sort=False)
        f["r"] = g["close"].pct_change()
        f["held"] = g[col].shift(1).fillna(0)
        f["trade"] = f.groupby("tic", sort=False)["held"].diff().abs().fillna(f["held"].abs())
        f["pnl"] = f["held"] * f["r"] - cost * f["trade"]

        daily = f.groupby("date")["pnl"].mean()
        active = f["held"] != 0
        return {
            "sharpe": daily.mean() / daily.std() * np.sqrt(252),
            "ann_return": daily.mean() * 252,
            "turnover/yr": f.groupby("date")["trade"].mean().mean() * 252,
            "hit_rate": (np.sign(f.loc[active, "held"]) == np.sign(f.loc[active, "r"])).mean(),
            "time_in_mkt": active.mean(),
        }

    @staticmethod
    def hold_until_opposite(signal):
        """1 1 1 ... until a sell, then -1 -1 -1 ... until a buy."""
        return signal.replace(0, np.nan).ffill().fillna(0).astype("int8")
    
    @staticmethod
    def get_ic(features, horizons=(1, 5, 10), block=63):
        """Time-series Spearman IC and ICIR vs next-h-day log return.

        Signal on day t is paired with log(close[t+h] / close[t]),
        which is the sum of `log-ret` from t+1 to t+h.
        Each ticker is shifted on its own dates.
        ICIR is mean / std of non-overlapping `block`-day Spearman ICs.
        """
        skip = {"date", "tic"}
        cols = [
            c for c in features.columns
            if c not in skip and pd.api.types.is_numeric_dtype(features[c])
        ]

        groups = (
            features.groupby("tic", sort=False)
            if "tic" in features.columns
            else [(None, features)]
        )
        
        rows = []
        for col in cols:
            for h in horizons:
                ics, hits, blocks, n = [], [], [], 0
                for _, g in groups:
                    g = g.sort_values("date") if "date" in g.columns else g
                    fwd = g["log-ret"].shift(-h).rolling(h).sum()
                    pair = pd.concat([g[col], fwd], axis=1).dropna()
                    if pair.empty:
                        continue
                    s, r = pair.iloc[:, 0], pair.iloc[:, 1]
                    ics.append(s.corr(r, method="spearman"))
                    hits.append((np.sign(s) == np.sign(r)).mean())
                    n += len(pair)
                    for i in range(0, len(pair) - block + 1, block):
                        sb, rb = s.iloc[i:i + block], r.iloc[i:i + block]
                        if sb.nunique() < 2 or rb.nunique() < 2:
                            continue
                        ic_b = sb.corr(rb, method="spearman")
                        if pd.notna(ic_b):
                            blocks.append(ic_b)
                ic = np.nanmean(ics) if ics else np.nan
                tstat = (
                    ic * np.sqrt(max(n - 2, 0)) / np.sqrt(max(1 - ic ** 2, 1e-12)) / np.sqrt(h)
                    if n > 2 and pd.notna(ic)
                    else np.nan
                )
                icir = (
                    np.mean(blocks) / np.std(blocks, ddof=1)
                    if len(blocks) > 1 and np.std(blocks, ddof=1) > 0
                    else np.nan
                )
                rows.append({
                    "signal": col,
                    "h": h,
                    "IC": ic,
                    "t": tstat,
                    "ICIR": icir,
                    "hit": np.nanmean(hits) if hits else np.nan,
                    "n": n,
                    "blocks": len(blocks),
                })
        return pd.DataFrame(rows).round(4)

    # ------------------------------------------------------------------ PCA

    @staticmethod
    def fit_pca(X, n_components=None):
        """Standardize then PCA. Returns the fitted (scaler, pca) pipeline.

        `n_components` may be an int, a variance fraction in (0, 1) such as 0.9,
        or None to keep every component (use this for the variance analysis).
        """
        pipe = make_pipeline(StandardScaler(), PCA(n_components=n_components, svd_solver="full"))
        pipe.fit(X)
        return pipe

    @staticmethod
    def pca_variance_table(pipe, thresholds=(0.8, 0.9, 0.95)):
        """Explained variance per component, plus components needed per threshold."""
        ratio = pipe[-1].explained_variance_ratio_
        table = pd.DataFrame({
            "component": [f"PC{i}" for i in range(1, len(ratio) + 1)],
            "explained": ratio,
            "cumulative": np.cumsum(ratio),
        })
        needed = {t: int(np.searchsorted(table["cumulative"], t) + 1) for t in thresholds}
        return table, needed

    @staticmethod
    def pca_loadings(pipe, feature_names, n_components=None):
        """Feature x component loadings (weights on standardized features)."""
        comps = pipe[-1].components_[:n_components]
        cols = [f"PC{i}" for i in range(1, len(comps) + 1)]
        return pd.DataFrame(comps.T, index=list(feature_names), columns=cols)

    @staticmethod
    def top_loadings(loadings, k=5):
        """For each component, the `k` features with the largest |loading|, signed."""
        rows = {}
        for pc in loadings.columns:
            top = loadings[pc].reindex(loadings[pc].abs().sort_values(ascending=False).index)[:k]
            rows[pc] = [f"{name} ({w:+.2f})" for name, w in top.items()]
        return pd.DataFrame(rows, index=[f"#{i}" for i in range(1, k + 1)]).T

    @staticmethod
    def plot_pca_variance(pipe, thresholds=(0.8, 0.9, 0.95), max_components=None):
        """Scree plot: variance explained by each component."""
        table, needed = Utils.pca_variance_table(pipe, thresholds)
        table = table.iloc[:max_components]
        x = np.arange(1, len(table) + 1)

        fig, ax = plt.subplots(figsize=(10, 3.8))
        ax.bar(x, table["explained"], color=SERIES_COLORS[0], width=0.8)
        ax.set_title("Variance explained by each component")
        ax.set_xlabel("Component")
        ax.set_ylabel("Share of variance")
        ax.set_xticks(x)
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        plt.show()
        return table, needed

    @staticmethod
    def plot_cumulative_variance(pipe, target=0.9):
        """Number of components vs cumulative variance, marking the smallest N that reaches `target`.

        Kept components (1..N) are blue, dropped ones gray. Returns N.
        """
        table, needed = Utils.pca_variance_table(pipe, (target,))
        n = needed[target]
        x = np.arange(1, len(table) + 1)
        cum = table["cumulative"].to_numpy()
        kept = x <= n

        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.plot(x, cum, color=MUTED, lw=1, zorder=1)
        ax.plot(x[kept], cum[kept], color=SERIES_COLORS[0], lw=2, zorder=2)
        ax.scatter(x[kept], cum[kept], s=48, color=SERIES_COLORS[0], edgecolor="white", linewidth=1.5, zorder=3,
                   label=f"Kept (top {n})")
        ax.scatter(x[~kept], cum[~kept], s=40, color="#b4b2ab", edgecolor="white", linewidth=1.5, zorder=3,
                   label="Dropped")
        ax.axhline(target, color=MUTED, ls="--", lw=0.8)
        ax.axvline(n, color=MUTED, ls=":", lw=0.8)
        ax.annotate(f"N = {n} components explain {cum[n - 1]:.1%}\n(first N to reach the {target:.0%} target)",
                    (n, cum[n - 1]), xytext=(14, -42), textcoords="offset points", color="#0b0b0b", fontsize=10,
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
        ax.text(x[-1], target, f"{target:.0%} target", color=MUTED, fontsize=9, ha="right", va="bottom")
        ax.set_xticks(x)
        ax.set_ylim(0, 1.02)
        ax.set_title("How many components? Cumulative variance explained")
        ax.set_xlabel("Number of components")
        ax.set_ylabel("Cumulative share of variance")
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(loc="lower right", frameon=False)
        fig.tight_layout()
        plt.show()
        return n

    @staticmethod
    def component_summary(pipe, feature_names, n_components, k=3):
        """Top `n_components` PCs: variance, cumulative variance and the `k` heaviest features."""
        table, _ = Utils.pca_variance_table(pipe)
        top = Utils.top_loadings(Utils.pca_loadings(pipe, feature_names, n_components), k)
        summary = table.iloc[:n_components].set_index("component")
        summary["top_features"] = top.apply(", ".join, axis=1)
        return summary

    @staticmethod
    def plot_pca_loadings(loadings):
        """Heatmap of feature loadings on each component (blue negative, red positive)."""
        lim = np.abs(loadings.to_numpy()).max()
        fig, ax = plt.subplots(figsize=(1.0 + 0.6 * loadings.shape[1], 0.32 * loadings.shape[0] + 1.2))
        im = ax.imshow(loadings, cmap=DIVERGING, vmin=-lim, vmax=lim, aspect="auto")
        ax.set_xticks(range(loadings.shape[1]), loadings.columns)
        ax.set_yticks(range(loadings.shape[0]), loadings.index)
        ax.set_title("PCA loadings (feature weight on each component)")
        fig.colorbar(im, ax=ax, label="Loading", shrink=0.8)
        fig.tight_layout()
        plt.show()

    # ------------------------------------------------------------- LightGBM

    @staticmethod
    def lgb_walk_forward(data, feature_cols, target="y", n_components=None, n_splits=5, lgb_params=None):
        """Pooled LightGBM with expanding-window CV split on dates.

        Every asset shares the same train/validation cut-off. If `n_components`
        is set, a scaler + PCA is fit on each training fold only (no look-ahead)
        and the model sees components instead of raw features.
        Returns (predictions, per-fold metrics).
        """
        import lightgbm as lgb

        params = dict(n_estimators=100, learning_rate=0.05, random_state=42, verbose=-1)
        params.update(lgb_params or {})
        dates = np.sort(data["date"].unique())
        preds, metrics = [], []
        for fold, (tr_d, va_d) in enumerate(TimeSeriesSplit(n_splits=n_splits).split(dates), start=1):
            tr = data[data["date"].isin(dates[tr_d])]
            va = data[data["date"].isin(dates[va_d])]
            X_tr, X_va = tr[feature_cols], va[feature_cols]
            n_used = len(feature_cols)
            if n_components is not None:
                pipe = Utils.fit_pca(X_tr, n_components)
                X_tr, X_va = pipe.transform(X_tr), pipe.transform(X_va)
                n_used = pipe[-1].n_components

            model = lgb.LGBMRegressor(**params)
            model.fit(X_tr, tr[target])

            fold_pred = va[["date", "tic", target]].copy()
            fold_pred["pred"] = model.predict(X_va)
            fold_pred["fold"] = fold
            preds.append(fold_pred)

            y, p = fold_pred[target], fold_pred["pred"]
            metrics.append({
                "fold": fold,
                "train_rows": len(tr),
                "val_rows": len(va),
                "n_inputs": n_used,
                "rmse": np.sqrt(mean_squared_error(y, p)),
                "rmse_zero": np.sqrt(mean_squared_error(y, np.zeros(len(y)))),
                "ic": fold_pred.groupby("date").apply(
                    lambda g: g[target].corr(g["pred"], method="spearman")
                ).mean(),
                "hit": (np.sign(y) == np.sign(p)).mean(),
            })
        return pd.concat(preds, ignore_index=True), pd.DataFrame(metrics)

if __name__ == "__main__":
    df = Utils.create_df("./data/anonymized_data/")
    n = 1
    df = Utils.get_first_n_assets(df, n)
    features = Utils.get_features(df)
    # df = Utils.get_feat(df, n).dropna() 
    a = Utils.get_macd_signal(features['macd'], features['macds'], features['macdh'])