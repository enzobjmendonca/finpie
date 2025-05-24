"""
Statistical analysis tools for financial time series data.

This module provides various statistical measures and indicators commonly used in quantitative finance,
including z-scores, half-life calculations, and mean reversion metrics.
"""

from typing import Optional, List, Union
import pandas as pd
import numpy as np

from finpie.data.timeseries import TimeSeries

class Statistical:
    """
    Class for performing statistical analysis on time series data.
    
    This class provides various statistical measures and indicators commonly used
    in quantitative finance, such as z-scores, half-life, and mean reversion metrics.
    
    Attributes:
        timeseries (TimeSeries): The input time series data
        column (str): The column name to analyze
        data (pd.Series): The actual data series being analyzed
    
    Example:
        >>> from finpie import TimeSeries, Statistical
        >>> ts = TimeSeries(data)
        >>> stats = Statistical(ts)
        >>> zscore = stats.zscore(window=20)
        >>> half_life = stats.half_life()
    """
    
    def __init__(self, timeseries: TimeSeries, column: str = None):
        """
        Initialize StatisticalAnalytics object.
        
        Args:
            timeseries: TimeSeries object to analyze
            column: Name of the column to analyze (default: 'close')
            
        Raises:
            TypeError: If timeseries is not a TimeSeries object
            ValueError: If column is not found in the time series data
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
        
        The z-score measures how many standard deviations an observation is from the mean.
        Values above 2 or below -2 are typically considered significant deviations.
        
        Args:
            window: Size of the rolling window for mean and std calculation
            min_periods: Minimum number of observations required
            
        Returns:
            Series containing z-scores
            
        Example:
            >>> zscore = stats.zscore(window=20)
            >>> # Values above 2 indicate overbought conditions
            >>> # Values below -2 indicate oversold conditions
        """
        if min_periods is None:
            min_periods = window
            
        rolling_mean = self.data.rolling(window, min_periods=min_periods).mean()
        rolling_std = self.data.rolling(window, min_periods=min_periods).std()
        
        return (self.data - rolling_mean) / rolling_std
    
    def half_life(self) -> float:
        """
        Calculate the half-life of mean reversion.
        
        The half-life is the time it takes for a deviation from the mean to decay by half.
        A shorter half-life indicates stronger mean reversion.
        
        Returns:
            Half-life in number of periods
            
        Example:
            >>> half_life = stats.half_life()
            >>> # A half-life of 5 means it takes 5 periods for a deviation
            >>> # to decay by half
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
        
        The Hurst exponent (H) characterizes the long-term memory of a time series:
        - H < 0.5: Mean-reverting series
        - H = 0.5: Random walk
        - H > 0.5: Trending series
        
        Args:
            lags: List of lags to use in calculation. If None, uses default lags.
            
        Returns:
            Hurst exponent (H < 0.5 indicates mean reversion)
            
        Example:
            >>> hurst = stats.hurst_exponent()
            >>> if hurst < 0.5:
            >>>     print("Series is mean-reverting")
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
        
        This is useful for identifying deviations from the trend and potential
        mean reversion opportunities.
        
        Args:
            window: Size of the rolling window for mean calculation
            min_periods: Minimum number of observations required
            
        Returns:
            Series containing spreads
            
        Example:
            >>> spread = stats.spread_to_ma(window=20)
            >>> # Large positive/negative spreads may indicate
            >>> # potential mean reversion opportunities
        """
        if min_periods is None:
            min_periods = window
            
        rolling_mean = self.data.rolling(window, min_periods=min_periods).mean()
        return self.data - rolling_mean
    
    def trading_signals(self, zscore_threshold: float = 2.0, window: int = 20) -> pd.DataFrame:
        """
        Generate trading signals based on z-score thresholds.
        
        This method generates trading signals based on z-score deviations:
        - Long signal when z-score < -threshold
        - Short signal when z-score > threshold
        
        Args:
            zscore_threshold: Threshold for generating signals
            window: Size of the rolling window for z-score calculation
            
        Returns:
            DataFrame containing trading signals with columns:
            - zscore: The calculated z-scores
            - signal: 1 for long, -1 for short, 0 for no signal
            
        Example:
            >>> signals = stats.trading_signals(zscore_threshold=2.0)
            >>> # signals['signal'] contains 1 (long), -1 (short), or 0 (no signal)
        """
        zscores = self.zscore(window=window)
        
        signals = pd.DataFrame(index=self.data.index)
        signals['zscore'] = zscores
        signals['signal'] = 0
        
        # Generate signals
        signals.loc[zscores > zscore_threshold, 'signal'] = -1  # Short signal
        signals.loc[zscores < -zscore_threshold, 'signal'] = 1   # Long signal
        
        return signals 