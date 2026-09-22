import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_ta as ta

class Utils:
    @staticmethod
    def load_data():
        DF = dict()
        index = 0
        for i in range(1, 101):
            index = '00' + str(i)
            index = index[-3:]
            file_path = "../data/anonymized_data/Asset_" + index+'.csv'
            df = pd.DataFrame(pd.read_csv(file_path, parse_dates=["Date"]))
            df.set_index('Date', inplace=True)
            DF[i] = df
        DF = pd.concat(DF.values(), axis=1, keys=['Asset_' + str(i) for i in DF.keys()])
        DF.columns.names = ['Asset', 'Metric']
        return DF
    
    @staticmethod
    def get_daily_return(DF):
        daily_return = DF.xs("Close", axis=1, level="Metric").pct_change()
        daily_return.columns = pd.MultiIndex.from_product(
            [DF.columns.get_level_values("Asset").unique(), ["Daily Return"]], names=["Asset", "Metric"]
        )
        return daily_return

    @staticmethod
    def get_log_return(DF):
        log_return = np.log(DF.xs("Close", axis=1, level="Metric") / DF.xs("Close", axis=1, level="Metric").shift(1))
        log_return.columns = pd.MultiIndex.from_product(
            [DF.columns.get_level_values("Asset").unique(), ["Log Return"]], names=["Asset", "Metric"]
        )
        return log_return
    
    @staticmethod
    def with_log_return(DF):
        return pd.concat([DF, Utils.get_log_return(DF)], axis=1).sort_index(axis=1)

    @staticmethod
    def with_daily_return(DF):
        return pd.concat([DF, Utils.get_daily_return(DF)], axis=1).sort_index(axis=1)
    
    @staticmethod
    def with_cumulative_return(DF):
        return pd.concat([DF, Utils.get_cumulative_return(DF)], axis=1).sort_index(axis=1)

    @staticmethod
    def get_cumulative_return(DF):
        cumulative_return = DF.xs("Close", axis=1, level="Metric").div(DF.xs("Close", axis=1, level="Metric").iloc[0])
        cumulative_return.columns = pd.MultiIndex.from_product(
            [DF.columns.get_level_values("Asset").unique(), ["Cumulative Return"]], names=["Asset", "Metric"]
        )
        return cumulative_return

    @staticmethod
    def get_volatility(DF, lam=0.94):
        """
        RiskMetrics EWMA volatility from Close.
        sigma^2_t = lam * sigma^2_{t-1} + (1 - lam) * r_{t-1}^2
        Default lam=0.94 (daily RiskMetrics).
        """
        close = DF.xs("Close", axis=1, level="Metric")
        r = close.pct_change()
        r2 = (r ** 2).shift(1)
        var = r2.ewm(alpha=1.0 - lam, adjust=False).mean()
        ewma_vol = np.sqrt(var)
        ewma_vol.columns = pd.MultiIndex.from_product(
            [DF.columns.get_level_values("Asset").unique(), ["Volatility"]],
            names=["Asset", "Metric"],
        )
        return ewma_vol

    @staticmethod
    def with_volatility(DF, lam=0.94):
        return pd.concat(
            [DF, Utils.get_volatility(DF, lam=lam)], axis=1
        ).sort_index(axis=1)

    @staticmethod
    def get_residual_return(DF):
        """
        Equal-weight market residual of log returns.
        r̃_{i,t} = r_{i,t} - (1/N) Σ_k r_{k,t}
        """
        close = DF.xs("Close", axis=1, level="Metric")
        log_return = np.log(close / close.shift(1))
        avg_return = log_return.mean(axis=1)
        residual = log_return.sub(avg_return, axis=0)
        residual.columns = pd.MultiIndex.from_product(
            [DF.columns.get_level_values("Asset").unique(), ["Residual Return"]],
            names=["Asset", "Metric"],
        )
        return residual

    @staticmethod
    def with_residual_return(DF):
        return pd.concat(
            [DF, Utils.get_residual_return(DF)], axis=1
        ).sort_index(axis=1)

    @staticmethod
    def split_train_test(DF, test_size=0.2):
        df = DF.copy()
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        n_test = int(len(df) * test_size)
        df_test = df.iloc[-n_test:]
        df_train = df.iloc[:-n_test]
        return df_train, df_test

    @staticmethod
    def validate_data(DF, tolerance=1e-10):
        assets = DF.columns.get_level_values(0).unique()

        assert len(assets) == 100, "Expected 100 assets"
        assert DF.index.is_unique, "Duplicate dates found"
        assert DF.index.is_monotonic_increasing, "Dates are not sorted"
        assert not DF.isna().any().any(), "Null values found"
        assert set(DF.columns.get_level_values(1)) == {
            "Open", "High", "Low", "Close", "Volume"
        }, "Incorrect OHLCV columns"

        failures = []
        for asset in assets:
            df = DF[asset]
            row_max = df[["Open", "Close", "Low"]].max(axis=1)
            row_min = df[["Open", "Close", "High"]].min(axis=1)
            invalid_high = df["High"] + tolerance < row_max
            invalid_low = df["Low"] - tolerance > row_min
            invalid_volume = df["Volume"] < 0
            for check, mask in [
                ("High", invalid_high),
                ("Low", invalid_low),
                ("Volume", invalid_volume),
            ]:
                if mask.any():
                    print(f"\n{asset}: invalid {check}")
                    print(df.loc[mask])
                    failures.append(f"{asset}: {check} ({mask.sum()} rows)")
        assert not failures, "Validation failures: " + ", ".join(failures)


    @staticmethod
    def validate_split(tr_, te_, DF):
        assert len(tr_) + len(te_) == len(DF), "Rows lost during split"
        assert tr_.index.intersection(te_.index).empty, "Train/test overlap"
        assert tr_.index.max() < te_.index.min(), "Split is not chronological"
        assert pd.concat([tr_, te_]).equals(DF), "Split changed the data"

    @staticmethod
    def corporate_action_candidates(DF, jump_threshold=0.30, volume_threshold=3):
        candidates = []

        for asset in DF.columns.get_level_values(0).unique():
            df = DF[asset].copy()

            # Overnight price change
            df["price_ratio"] = df["Open"] / df["Close"].shift(1)
            df["overnight_return"] = df["price_ratio"] - 1

            # Volume relative to previous 20-day median
            baseline = df["Volume"].shift(1).rolling(20).median()
            df["volume_multiple"] = df["Volume"] / baseline

            mask = df["overnight_return"].abs() >= jump_threshold

            flagged = df.loc[
                mask,
                ["Open", "Close", "Volume", "price_ratio",
                "overnight_return", "volume_multiple"]
            ].copy()

            if not flagged.empty:
                flagged["asset"] = asset
                flagged["volume_confirmation"] = (
                    flagged["volume_multiple"] >= volume_threshold
                )
                candidates.append(flagged)
        return pd.concat(candidates) if candidates else pd.DataFrame()

    @staticmethod
    def get_features(df):
        features = dict()
        FEATURE_LIST = ["log_return", "return", "volatility", "residual_return"]
        for feature in FEATURE_LIST:
            features[feature] = df[feature].values
        return features

    @staticmethod
    def plot_metric(DF, metric, title=None, alpha=0.85, figsize=(11, 6)):
        panel = DF.xs(metric, axis=1, level="Metric")

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(panel.index, panel.values, lw=1.2, alpha=alpha)
        ax.set_title(title or f"{metric} across assets")
        ax.set_xlabel("Date")
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        plt.show()
        return fig, ax
    
    @staticmethod
    def add_market_indicators(DF):
        parts = []
        for asset in DF.columns.get_level_values(0).unique():
            h = DF[(asset, "High")]
            l = DF[(asset, "Low")]
            c = DF[(asset, "Close")]

            rsi = ta.rsi(c, length=14).rename("RSI")

            tp = (h + l + c) / 3
            mad = (tp - tp.rolling(14).mean()).abs().rolling(14).mean()
            cci = ((tp - tp.rolling(14).mean()) / (0.015 * mad)).rename("CCI")

            adx = ta.adx(h, l, c, length=14)          # ADX_14, DMP_14, DMN_14
            macd = ta.macd(c, fast=12, slow=26, signal=9)  # MACD_*, MACDh_*, MACDs_*

            ind = pd.concat([rsi, cci, adx, macd], axis=1)
            ind.columns = pd.MultiIndex.from_product(
                [[asset], ind.columns], names=["Asset", "Metric"]
            )
            parts.append(ind)

        return pd.concat([DF] + parts, axis=1).sort_index(axis=1)

    # For LASSO Test
    @staticmethod
    def get_y_true(DF):
        return 
if __name__ == "__main__":
    DF = Utils.load_data()
    # Utils.validate_data(DF)
    tr_, te_ = Utils.split_train_test(DF, test_size=0.2)
    Utils.validate_split(tr_, te_, DF)
    candidates = Utils.corporate_action_candidates(DF)
    print(candidates)