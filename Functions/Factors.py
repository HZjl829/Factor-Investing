import numpy as np
import pandas as pd
from scipy.stats import linregress

class MomentumFactor:
    """
    Compute different momentum-based signals on a price panel.

    Parameters
    ----------
    prices : pd.DataFrame
        Monthly price data with DateTimeIndex and tickers as columns.
    """

    def __init__(self, prices: pd.DataFrame):
        # expect monthly data, no gaps in calendar index
        self.prices = prices.copy().sort_index()

    def simple_return(self, lookback: int = 12, skip: int = 1) -> pd.DataFrame:
        """
        Simple momentum = (P[t-skip] / P[t-skip-lookback]) - 1
        """

        p1 = self.prices.shift(skip)                 # P[t-skip]
        p0 = self.prices.shift(lookback)      # P[t-skip-lookback]
        return p1.div(p0).subtract(1)

    def _slope(self, x: np.ndarray) -> float:
        """
        Internal: slope of log-price on time [0..n-1].
        """
        y = np.log(x)
        t = np.arange(len(y))
        slope, _, _, _, _ = linregress(t, y)
        return slope

    def _tstat(self, x: np.ndarray) -> float:
        """
        Internal: t-stat of slope = slope / stderr
        """
        y = np.log(x)
        t = np.arange(len(y))
        slope, _, _, _, stderr = linregress(t, y) # assume residual are normalli distributed
        return slope / stderr

    def slope_regression(self, lookback: int = 12, skip: int = 0) -> pd.DataFrame:
        """
        Rolling regression slope of log-price vs. time over `lookback` months.
        `skip` shifts the window forward to avoid overlapping with the most recent data.
        """
        shifted = self.prices.shift(skip)
        
        return shifted.rolling(window=lookback).apply(self._slope, raw=True)

    def tstat_regression(self, lookback: int = 12, skip: int = 0) -> pd.DataFrame:
        """
        Rolling t-statistic of the slope from regressing log-price on time.
        """
        shifted = self.prices.shift(skip)
        return shifted.rolling(window=lookback).apply(self._tstat, raw=True)

    def consensus_slope_signal(
        self,
        min_lookback: int = 3,
        max_lookback: int = 12,
        skip: int = 1
    ) -> pd.DataFrame:
        """
        Consensus regression-based momentum signal:
        - Compute rolling slope for each lookback in [min_lookback..max_lookback],
          skipping `skip` months.
        - If all slopes > 0, signal = +1 (long);
        - If all slopes < 0, signal = -1 (short);
        - Otherwise, signal = 0 (no position).
        Only dates with sufficient lookback and skip are included.
        """
        # 1) Generate slope panels for each lookback
        slopes = [
            self.slope_regression(lookback=lb, skip=skip)
            for lb in range(min_lookback, max_lookback + 1)
        ]
        # 2) Determine the first valid index after all lookbacks and skip
        valid_start = max_lookback
        # 3) Trim each slope DataFrame to have no NaNs from insufficient history
        slopes = [s.iloc[valid_start:] for s in slopes]
        # 4) Stack into a 3D array: (L, T_valid, N)
        arr = np.stack([s.values for s in slopes], axis=0)
        # 5) Consensus masks
        
        pos = np.all(arr > 0, axis=0)
        neg = np.all(arr < 0, axis=0)
        # 6) Build signal DataFrame on trimmed index
        idx = slopes[0].index
        sig = pd.DataFrame(0, index=idx, columns=self.prices.columns, dtype=int)
        sig.values[pos] = 1
        sig.values[neg] = -1
        return sig
    
    def average_slope_signal(
        self,
        min_lookback: int = 3,
        max_lookback: int = 12,
        skip: int = 1
    ) -> pd.DataFrame:
        """
        Average regression-based momentum signal:
        - Compute rolling slope for each lookback in [min_lookback..max_lookback],
          skipping `skip` months.
        - Take the average of those slopes at each date for each asset.
        Only dates with sufficient lookback are included, starting at index = max_lookback.
        """
        # 1) Generate slope panels for each lookback
        slopes = [
            self.slope_regression(lookback=lb, skip=skip)
            for lb in range(min_lookback, max_lookback + 1)
        ]
        # 2) Determine the first valid index after maximum lookback
        valid_start = max_lookback
        # 3) Trim each slope DataFrame to have no NaNs from insufficient history
        slopes = [s.iloc[valid_start:] for s in slopes]
        # 4) Stack into a 3D array: (L, T_valid, N)
        arr = np.stack([s.values for s in slopes], axis=0)
        # 5) Compute average slope across lookbacks
        avg_arr = np.nanmean(arr, axis=0)
        # 6) Build DataFrame of average slopes
        idx = slopes[0].index
        avg_slope = pd.DataFrame(avg_arr, index=idx, columns=self.prices.columns)
        return avg_slope

if __name__ == "__main__":
     # assume `monthly_prices` is your 252×34 DataFrame
    monthly_prices = pd.read_csv('./Data/asset_monthly_prices.csv').set_index('Date',)



    mf = MomentumFactor(monthly_prices)

    # 1) 12×1 simple returns

    mom_ret = mf.simple_return(lookback=3, skip=1)

 

    # 3) 6-month slope, skipping most recent 1 month
    mom_slope6 = mf.slope_regression(lookback=3, skip=1)

    # 4) 12-month t-stat, skipping 1 month
    mom_t = mf.tstat_regression(lookback=3, skip=1)

    # 5) Consensus slope signal
    mom_consensus = mf.consensus_slope_signal(min_lookback=3, max_lookback=12, skip=1)
    print("Momentum Returns:\n", mom_ret.head(10))
    
    print("Momentum Slope (6 months, skip 1):\n", mom_slope6.head(10))
    print("Momentum T-stat:\n", mom_t.head(10))
    print("Consensus Momentum Signal:\n", mom_consensus.head(20))
    