"""From FinRL https://github.com/AI4Finance-Foundation/FinRL/blob/master/finrl/meta/preprocessor/yahoodownloader.py (MIT)

Vendored because importing `finrl` pulls in its whole training stack.
Changes for yfinance >= 0.2.5x:
- `proxy` is no longer a yf.download argument
- ask for unadjusted prices + "Adj Close" (auto_adjust now defaults to True)
- flat columns (multi_level_index=False), renamed by name instead of position
- keeps the raw close as `close_raw` next to the adjusted `close`
"""

from __future__ import annotations

import pandas as pd
import yfinance as yf


class YahooDownloader:
    """Provides methods for retrieving daily stock data from
    Yahoo Finance API

    Attributes
    ----------
        start_date : str
            start date of the data
        end_date : str
            end date of the data (exclusive, as in yf.download)
        ticker_list : list
            a list of stock tickers

    Methods
    -------
    fetch_data()
        Fetches data from yahoo API

    """

    COLUMNS = {
        "Date": "date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close_raw",
        "Adj Close": "close",
        "Volume": "volume",
    }

    def __init__(self, start_date: str, end_date: str, ticker_list: list):
        self.start_date = start_date
        self.end_date = end_date
        self.ticker_list = ticker_list
        self.failed: list[str] = []

    def fetch_data(self) -> pd.DataFrame:
        """Fetches data from Yahoo API

        Returns
        -------
        `pd.DataFrame`
            date, open, high, low, close (adjusted), close_raw, volume, tic, day
        """
        parts = []
        self.failed = []
        for tic in self.ticker_list:
            temp_df = yf.download(
                tic,
                start=self.start_date,
                end=self.end_date,
                auto_adjust=False,
                multi_level_index=False,
                progress=False,
            )
            if len(temp_df) == 0:
                self.failed.append(tic)
                continue
            temp_df = temp_df.reset_index().rename(columns=self.COLUMNS)
            temp_df["tic"] = tic
            parts.append(temp_df[list(self.COLUMNS.values()) + ["tic"]])
        if not parts:
            raise ValueError("no data is fetched.")

        data_df = pd.concat(parts, ignore_index=True)
        # create day of the week column (monday = 0)
        data_df["day"] = data_df["date"].dt.dayofweek
        # convert date to standard string format, easy to filter
        data_df["date"] = data_df["date"].dt.strftime("%Y-%m-%d")
        data_df = data_df.dropna()
        print("Shape of DataFrame: ", data_df.shape, "| failed tickers:", len(self.failed))
        return data_df.sort_values(by=["date", "tic"]).reset_index(drop=True)

    def select_equal_rows_stock(self, df):
        df_check = df.tic.value_counts()
        df_check = pd.DataFrame(df_check).reset_index()
        df_check.columns = ["tic", "counts"]
        mean_df = df_check.counts.mean()
        equal_list = list(df.tic.value_counts() >= mean_df)
        names = df.tic.value_counts().index
        select_stocks_list = list(names[equal_list])
        df = df[df.tic.isin(select_stocks_list)]
        return df
