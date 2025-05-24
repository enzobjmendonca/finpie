from typing import Optional, List, Union
import pandas as pd
import numpy as np

from finpie.data.timeseries import TimeSeries

class Statistical:
    """
    Class for performing statistical analysis on time series data.
    
    This class provides various statistical measures and indicators commonly used
    in quantitative finance, such as z-scores, half-life, and mean reversion metrics.
    """
    
    def __init__(self, timeseries: TimeSeries, column: str = None):
        """
        Initialize StatisticalAnalytics object.
        
        Args:
            timeseries: TimeSeries object to analyze
            column: Name of the column to analyze (default: 'close')
        """
        if not isinstance(timeseries, TimeSeries):
            raise TypeError("timeseries must be a TimeSeries object")
            
        if column not in timeseries.data.columns:
            raise ValueError(f"Column '{column}' not found in time series data")
            
        self.timeseries = timeseries
        self.column = column if column is not None else timeseries.data.columns[0]
        self.data = timeseries.data[self.column]
    
    def zscore(self, window: int = 20, min_periods: Optional[int] = None) -> pd.Series:
        """
        Calculate the z-score of the time series.
        
        Args:
            window: Size of the rolling window for mean and std calculation
            min_periods: Minimum number of observations required
            
        Returns:
            Series containing z-scores
        """
        if min_periods is None:
            min_periods = window
            
        rolling_mean = self.data.rolling(window, min_periods=min_periods).mean()
        rolling_std = self.data.rolling(window, min_periods=min_periods).std()
        
        return (self.data - rolling_mean) / rolling_std
    
    def half_life(self) -> float:
        """
        Calculate the half-life of mean reversion.
        
        Returns:
            Half-life in number of periods
        """
        series = self.data
        series_lag = series.shift(1)
        series_ret = series - series_lag
        
        # Drop NaN values
        valid_data = pd.concat([series_lag, series_ret], axis=1).dropna()
        
        # Calculate half-life
        series_lag = valid_data.iloc[:, 0]
        series_ret = valid_data.iloc[:, 1]
        
        # OLS regression
        X = pd.concat([pd.Series(1, index=series_lag.index), series_lag], axis=1)
        beta = np.linalg.inv(X.T @ X) @ X.T @ series_ret
        
        # Calculate half-life
        half_life = -np.log(2) / beta[1]
        return half_life
    
    def hurst_exponent(self, lags: Optional[List[int]] = None) -> float:
        """
        Calculate the Hurst exponent to determine if the series is mean-reverting.
        
        Args:
            lags: List of lags to use in calculation. If None, uses default lags.
            
        Returns:
            Hurst exponent (H < 0.5 indicates mean reversion)
        """
        if lags is None:
            lags = [2, 3, 4, 5, 6, 7, 8, 9, 10]
            
        series = self.data
        tau = []
        lagged_var = []
        
        for lag in lags:
            # Calculate variance of lagged differences
            tau.append(lag)
            lagged_var.append(np.log(series.diff(lag).var()))
            
        # Linear regression
        m = np.polyfit(np.log(tau), lagged_var, 1)
        hurst = m[0] / 2.0
        
        return hurst
    
    def spread_to_ma(self, window: int = 20, min_periods: Optional[int] = None) -> pd.Series:
        """
        Calculate the spread between the series and its moving average.
        
        Args:
            window: Size of the rolling window for mean calculation
            min_periods: Minimum number of observations required
            
        Returns:
            Series containing spreads
        """
        if min_periods is None:
            min_periods = window
            
        rolling_mean = self.data.rolling(window, min_periods=min_periods).mean()
        return self.data - rolling_mean
    
    def trading_signals(self, zscore_threshold: float = 2.0, window: int = 20) -> pd.DataFrame:
        """
        Generate trading signals based on z-score thresholds.
        
        Args:
            zscore_threshold: Threshold for generating signals
            window: Size of the rolling window for z-score calculation
            
        Returns:
            DataFrame containing trading signals
        """
        zscores = self.zscore(window=window)
        
        signals = pd.DataFrame(index=self.data.index)
        signals['zscore'] = zscores
        signals['signal'] = 0
        
        # Generate signals
        signals.loc[zscores > zscore_threshold, 'signal'] = -1  # Short signal
        signals.loc[zscores < -zscore_threshold, 'signal'] = 1   # Long signal
        
        return signals 