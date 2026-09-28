import numpy as np
import pandas as pd
import datetime
from datetime import timedelta
from sklearn.metrics.pairwise import cosine_similarity
from sympy import N
# Date tic f1 f2 f3 f4 
# 1     1  0.1 0.2 0.1 0.3
# 2     



class StockVectorMachine:

    def __init__(self, df: pd.DataFrame, time_steps: int = 63):
        self.df = df
        self.df.dropna(inplace = True)
        self._features = df.columns.tolist()
        self.assets = df['tic'].unique().tolist()
        self.n_assets = len(self.assets)
        self.n_features = len(self._features)
        self.time_steps = time_steps
        self.dates = pd.to_datetime(self.df['date'].unique())
        self.total_time_steps = len(df['date'].unique())//time_steps + (1 if len(df['date'].unique())%time_steps != 0 else 0)
        self._assets_feature_vector_frame: pd.DataFrame = None # type: ignore
        self._asset_correlation: np.ndarray = None # type: ignore
        self._asset_correlation_start_date = None
        self._asset_correlation_end_date = None

    def _get_asset_correlation_dates(self):
        return self._asset_correlation_start_date, self._asset_correlation_end_date

    def _preprocess_data(self):
        """
        creasts a dataframe of feature vectors for all assets daily
        by standardizing the feature vectors over a rolling time window
        (default : 63 days) to get a standardized feature vector based on 3 months data
        for each asset.
        """
        time_steps = self.time_steps
        features = self._features
        _start_date = self.dates.min().date() # type: ignore
        _current_date = _start_date
        _end_date = self.dates.max().date() # type: ignore
        
        self.standardized: dict[str, pd.DataFrame] = {}
        feature_cols = [c for c in self._features if c not in ("date", "tic")]
        
        for i, a in enumerate(self.assets):
            adf = self.df.loc[self.df["tic"] == a].sort_values("date")
            values = adf[feature_cols]
            roll = values.rolling(window=time_steps, min_periods=time_steps)
            z = (values - roll.mean()) / roll.std()
            z.insert(0, "tic", a)
            z.insert(0, "date", adf["date"].to_numpy())
            self.standardized[a] = z.dropna().reset_index(drop=True)
        self._assets_feature_vector_frame: pd.DataFrame = pd.concat(self.standardized.values()).reset_index(drop=True)
        self._asset_correlation_start_date = self._assets_feature_vector_frame["date"].min()
        self._asset_correlation_end_date = self._assets_feature_vector_frame["date"].max()

    def _get_feature_vector_correlation(self, tic1: str, tic2: str):
        """
        Get the cosine similarity for the feature vectors of two assets daily, 
        from ('2016-05-06', '2026-01-16')
        """
        cols = [f for f in self._features if f not in ("date", "tic")]
        _afv1 = self._assets_feature_vector_frame.loc[self._assets_feature_vector_frame["tic"] == tic1].set_index("date")
        _afv2 = self._assets_feature_vector_frame.loc[self._assets_feature_vector_frame["tic"] == tic2].set_index("date")
        _afv1_mat = _afv1[cols].to_numpy()
        _afv2_mat = _afv2[cols].to_numpy()
        # print(_afv1_mat.shape, _afv2_mat.shape)
        denom = np.linalg.norm(_afv1_mat, axis=1) * np.linalg.norm(_afv2_mat, axis=1)
        sim = np.divide((_afv1_mat * _afv2_mat).sum(axis=1), denom, out=np.full(len(_afv1_mat), np.nan), where=denom != 0)
        return sim

    def _get_asset_correlation(self): 
        """
        Get the correlation for the feature vectors across all assets daily, 
        every pair of assets provide a record of cosine similarity from ('2016-05-06', '2026-01-16')
        """
        n_dates = len(self.standardized[self.assets[0]]["date"].unique())
        cols = [f for f in self._features if f not in ("date", "tic")]
        _fv_matr =  np.stack([ self.standardized[a].set_index("date")[cols].to_numpy(dtype=float) 
            for a in self.assets])  # (assets, dates, features))

        _fv_matr_norm = np.linalg.norm(_fv_matr, axis=2)
        u = np.divide(_fv_matr, _fv_matr_norm[:, :, None], out=np.full(_fv_matr.shape, np.nan), where=_fv_matr_norm[:, :, None] != 0)
        sim = np.einsum('adf,bdf->abd', u, u)
        self._asset_correlation = sim
        return sim

       
if __name__ == "__main__":
    df = pd.read_csv("data/rl_features.csv")
    stc = StockVectorMachine(df)
    stc._preprocess_data()




