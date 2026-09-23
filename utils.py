import pandas as pd
import os
from stockstats import wrap
from stockstats import StockDataFrame as sdf
import numpy as np

class Utils:

    FEATURE_LIST = [
        "log-ret", # daily log retur
        "return", # daily return  
        "volatility", # RiskMetric EWMA volatility
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
        assets = df['tic'].unique()
        assets = assets[:n]
        return wrap(df.apply(lambda x: x if x['tic'] in assets else None, axis=1).dropna())

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
        derived = ['volatility', 'residual_return', 'return']
        df = df.reset_index()
        parts = []
        stock_cols = [col for col in Utils.FEATURE_LIST if col not in derived]
        features : sdf = wrap(df)
        for tic, group in features.groupby("tic", sort=False):
            a = wrap(group.sort_values(by='date'))
            _ = a[stock_cols]
            a['volatility'] = Utils.get_volatility(a)
            # a['residual_return'] = Utils.get_residual_return(a)
            part=pd.DataFrame(a).reset_index()
            part['tic'] = tic
            parts.append(part)
        
        out = pd.concat(parts)
        out = out.sort_values(['date', 'tic'])
        out.drop(columns=['index'], inplace=True)
        return out

    @staticmethod
    def get_volatility(df, lam=0.94):
        """
        RiskMetrics EWMA volatility from close, one series per tic.
        sigma^2_t = lam * sigma^2_{t-1} + (1 - lam) * r_{t-1}^2
        Default lam=0.94 (daily RiskMetrics).
        """
        def _one(close):
            r2 = close.pct_change().pow(2).shift(1)
            return r2.ewm(alpha=1.0 - lam, adjust=False).mean().pow(0.5)
        return df.groupby("tic", sort=False)["close"].transform(_one)


    @staticmethod
    def get_macd_signal(macd, macds, macdh):
        """
        Takes in a dataframe of features for single asset and returns macd.
        Apply this function with groupby to get the signal for all assets.


        """
        macd_signal_crossover_up = ((macd > macds) & (macd.shift(1) <= macds.shift(1))).apply(lambda x: 1 if x else 0)
        macd_zero_crossover_up = ((macd > 0) & (macd.shift(1) <= 0)).apply(lambda x: 1 if x else 0)
        macd_signal_crossover_down = ((macd < macds) & (macd.shift(1) >= macds.shift(1))).apply(lambda x: -1 if x else 0)
        macd_zero_crossover_down = ((macd < 0) & (macd.shift(1) >= 0)).apply(lambda x: -1 if x else 0)
        macd_signal_crossover = macd_signal_crossover_up | macd_signal_crossover_down
        macd_zero_crossover = macd_zero_crossover_up | macd_zero_crossover_down
        return macd_signal_crossover, macd_zero_crossover

    @staticmethod
    def get
if __name__ == "__main__":
    df = Utils.create_df("./data/anonymized_data/")
    n = 1
    df = Utils.get_first_n_assets(df, n)
    features = Utils.get_features(df)
    # df = Utils.get_feat(df, n).dropna() 
    a, b = Utils.get_macd_signal(features['macd'], features['macds'], features['macds'])