"""Map the anonymized assets back to S&P 500 tickers.

The anonymized closes are real prices times a per-asset constant, so daily
log returns are unchanged. Each asset is matched to the S&P 500 stock whose
Yahoo returns correlate best, with a one-to-one assignment across all assets.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.optimize import linear_sum_assignment

from yahoodownloader import YahooDownloader

ROOT = Path(__file__).resolve().parents[1]
ANON_DIR = ROOT / "data" / "anonymized_data"
ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def load_anonymized(path: Path = ANON_DIR) -> pd.DataFrame:
    """Long frame: date, tic, open, high, low, close, volume."""
    parts = []
    for f in sorted(path.glob("*.csv")):
        df = pd.read_csv(f)
        df.columns = df.columns.str.lower()
        df["tic"] = f.stem
        parts.append(df)
    out = pd.concat(parts, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"])
    return out


def sp500_constituents() -> pd.DataFrame:
    """Current S&P 500 list from Wikipedia, with Yahoo-style tickers (BRK.B -> BRK-B)."""
    html = requests.get(SP500_URL, headers={"User-Agent": "precog-test research script"}, timeout=30).text
    table = pd.read_html(io.StringIO(html))[0]
    table = table.rename(columns={"Symbol": "ticker", "Security": "name", "GICS Sector": "sector",
                                  "GICS Sub-Industry": "sub_industry"})
    table["ticker"] = table["ticker"].str.replace(".", "-", regex=False)
    return table[["ticker", "name", "sector", "sub_industry"]]


def download_prices(tickers, start, end) -> pd.DataFrame:
    """YahooDownloader output for `tickers`, with `date` as datetime."""
    df = YahooDownloader(start, end, list(tickers)).fetch_data()
    df["date"] = pd.to_datetime(df["date"])
    return df


def log_returns(df: pd.DataFrame, price: str = "close") -> pd.DataFrame:
    """date x tic matrix of daily log returns."""
    wide = df.pivot(index="date", columns="tic", values=price).sort_index()
    return np.log(wide).diff().iloc[1:]


def return_correlations(anon: pd.DataFrame, real: pd.DataFrame, min_overlap: int = 250) -> pd.DataFrame:
    """anon asset x real ticker correlation of daily log returns on shared dates."""
    real = real.reindex(anon.index)
    enough = real.notna().sum() >= min_overlap
    real = real.loc[:, enough]
    return pd.DataFrame({a: real.corrwith(anon[a]) for a in anon.columns}).T


def match(corr: pd.DataFrame) -> pd.DataFrame:
    """One-to-one assignment that maximizes total correlation, plus how clear each match is."""
    filled = corr.fillna(-1.0)
    rows, cols = linear_sum_assignment(-filled.to_numpy())
    out = pd.DataFrame({"asset": corr.index[rows], "ticker": corr.columns[cols],
                        "corr": filled.to_numpy()[rows, cols]})
    ranked = np.sort(filled.to_numpy(), axis=1)[:, ::-1]
    out["runner_up_corr"] = ranked[rows, 1]
    out["runner_up"] = [filled.loc[a].drop(t).idxmax() for a, t in zip(out["asset"], out["ticker"])]
    return out


def price_scale(anon_df: pd.DataFrame, real_df: pd.DataFrame, mapping: pd.DataFrame, price: str = "close") -> pd.DataFrame:
    """Anonymized / real price ratio per matched pair. A constant ratio means only rescaling."""
    anon = anon_df.pivot(index="date", columns="tic", values="close")
    real = real_df.pivot(index="date", columns="tic", values=price).reindex(anon.index)
    rows = []
    for a, t in zip(mapping["asset"], mapping["ticker"]):
        ratio = (anon[a] / real[t]).dropna()
        rows.append({"asset": a, "scale": ratio.median(), "scale_cv": ratio.std() / ratio.mean(),
                     "shared_days": len(ratio)})
    return pd.DataFrame(rows)
