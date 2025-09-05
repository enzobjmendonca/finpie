from typing import Dict, Any, Optional, Union
import pandas as pd
import numpy as np
import logging

from .timeseries import TimeSeries, TimeSeriesMetadata

# Configure logger
logger = logging.getLogger(__name__)

class SpreadTimeSeries(TimeSeries):
    """A class for handling spread-based time series.
    
    This class extends TimeSeries to provide functionality for analyzing the spread between two time series.
    It's commonly used in spread trading and statistical arbitrage strategies, where the relationship
    between two assets is analyzed through their price spread, often with a hedge ratio.
    
    Since it extends TimeSeries (which extends pd.DataFrame), it inherits all pandas DataFrame
    functionality while adding specialized spread-specific operations.
    
    Attributes:
        series1 (TimeSeries): The first time series
        series2 (TimeSeries): The second time series
        hedge_ratio (float): The hedge ratio used to calculate the spread
    """
    
    def __init__(self, series1: Union[TimeSeries, pd.DataFrame, pd.Series], 
                 series2: Union[TimeSeries, pd.DataFrame, pd.Series],
                 hedge_ratio: Optional[float] = None,
                 metadata: TimeSeriesMetadata = None):
        """Initialize a SpreadTimeSeries object.
        
        Args:
            series1: First TimeSeries object, DataFrame, or Series
            series2: Second TimeSeries object, DataFrame, or Series
            hedge_ratio: Optional hedge ratio for series2. If None, will be calculated using OLS regression
            metadata: Optional metadata for the spread time series
                
        Note:
            If series1 or series2 are not TimeSeries objects, they will be converted automatically.
            The hedge ratio is calculated using OLS regression if not provided.
        """
        # Convert inputs to TimeSeries if needed
        if not isinstance(series1, TimeSeries):
            if isinstance(series1, pd.Series):
                series1 = series1.to_frame()
            series1_metadata = TimeSeriesMetadata(
                name=series1.columns[0] if hasattr(series1, 'columns') and len(series1.columns) > 0 else 'series1',
                symbol=series1.columns[0] if hasattr(series1, 'columns') and len(series1.columns) > 0 else 'series1',
                source='unknown',
                start_date=series1.index[0] if len(series1.index) > 0 else None,
                end_date=series1.index[-1] if len(series1.index) > 0 else None,
                frequency='',
                currency='',
                additional_info={}
            )
            series1 = TimeSeries(series1, series1_metadata)
            
        if not isinstance(series2, TimeSeries):
            if isinstance(series2, pd.Series):
                series2 = series2.to_frame()
            series2_metadata = TimeSeriesMetadata(
                name=series2.columns[0] if hasattr(series2, 'columns') and len(series2.columns) > 0 else 'series2',
                symbol=series2.columns[0] if hasattr(series2, 'columns') and len(series2.columns) > 0 else 'series2',
                source='unknown',
                start_date=series2.index[0] if len(series2.index) > 0 else None,
                end_date=series2.index[-1] if len(series2.index) > 0 else None,
                frequency='',
                currency='',
                additional_info={}
            )
            series2 = TimeSeries(series2, series2_metadata)
        
        # Store references to original series
        self.series1 = series1
        self.series2 = series2
        
        # Align the two time series and calculate spread
        aligned_data, calculated_hedge_ratio = self._align_and_calculate_spread(hedge_ratio)
        self.hedge_ratio = calculated_hedge_ratio
        
        # Create metadata if not provided
        if metadata is None:
            metadata = self._create_spread_metadata()
        
        # Initialize parent TimeSeries with spread data
        super().__init__(aligned_data, metadata)
    
    def _align_and_calculate_spread(self, hedge_ratio: Optional[float] = None) -> tuple[pd.DataFrame, float]:
        """Align the two series and calculate the spread."""
        # Get the data columns (assuming single column for each)
        s1_col = self.series1.columns[0]
        s2_col = self.series2.columns[0]
        
        # Align data using inner join
        aligned_data = self.series1.join(
            self.series2, 
            how='inner', 
            lsuffix='_series1', 
            rsuffix='_series2'
        )
        
        # If column names are the same, pandas will add suffixes
        if len(aligned_data.columns) == 2:
            series1_col = aligned_data.columns[0]
            series2_col = aligned_data.columns[1]
        else:
            # Find the correct columns
            series1_col = f"{s1_col}_series1" if f"{s1_col}_series1" in aligned_data.columns else s1_col
            series2_col = f"{s2_col}_series2" if f"{s2_col}_series2" in aligned_data.columns else s2_col
        
        # Calculate hedge ratio if not provided
        if hedge_ratio is None:
            hedge_ratio = self._calculate_hedge_ratio(aligned_data[series1_col], aligned_data[series2_col])
            logger.debug(f"Calculated hedge ratio: {hedge_ratio}")
        
        # Calculate spread: series1 - hedge_ratio * series2
        spread_values = aligned_data[series1_col] - hedge_ratio * aligned_data[series2_col]
        
        # Create DataFrame with spread
        spread_data = pd.DataFrame({'spread': spread_values})
        
        return spread_data, hedge_ratio
    
    def _calculate_hedge_ratio(self, x: pd.Series, y: pd.Series) -> float:
        """Calculate the hedge ratio using OLS regression.
        
        Args:
            x: First time series (independent variable)
            y: Second time series (dependent variable)
            
        Returns:
            Calculated hedge ratio using OLS regression
            
        Note:
            The hedge ratio is calculated as the coefficient of x in the regression y = α + β*x + ε
        """
        # Remove NaN values
        valid_data = pd.DataFrame({'x': x, 'y': y}).dropna()
        
        if len(valid_data) < 2:
            logger.warning("Insufficient data for hedge ratio calculation, using 1.0")
            return 1.0
        
        x_clean = valid_data['x']
        y_clean = valid_data['y']
        
        # Add constant for regression (intercept)
        X = pd.concat([pd.Series(1, index=x_clean.index, name='const'), x_clean], axis=1)
        
        try:
            # Calculate hedge ratio using OLS: (X'X)^(-1)X'y
            XTX_inv = np.linalg.inv(X.T @ X)
            beta = XTX_inv @ X.T @ y_clean
            return beta.iloc[1]  # Return the coefficient for x (not the intercept)
        except np.linalg.LinAlgError:
            logger.warning("Singular matrix in hedge ratio calculation, using correlation-based estimate")
            # Fallback to correlation-based estimate
            return (x_clean.cov(y_clean) / x_clean.var()) if x_clean.var() != 0 else 1.0
    
    def _create_spread_metadata(self) -> TimeSeriesMetadata:
        """Create metadata for the spread time series."""
        s1_symbol = self.series1.metadata.symbol if self.series1.metadata else 'unknown'
        s2_symbol = self.series2.metadata.symbol if self.series2.metadata else 'unknown'
        s1_name = self.series1.metadata.name if self.series1.metadata else 'unknown'
        s2_name = self.series2.metadata.name if self.series2.metadata else 'unknown'
        
        return TimeSeriesMetadata(
            name=f"{s1_name}-{s2_name}",
            symbol=f"{s1_symbol}-{s2_symbol}",
            source="spread",
            start_date=self.index[0] if len(self.index) > 0 else None,
            end_date=self.index[-1] if len(self.index) > 0 else None,
            frequency=self.series1.metadata.frequency if self.series1.metadata else '',
            currency=self.series1.metadata.currency if self.series1.metadata else '',
            additional_info={
                'series1_symbol': s1_symbol,
                'series2_symbol': s2_symbol,
                'hedge_ratio': self.hedge_ratio,
                'series1_info': self.series1.metadata.additional_info if self.series1.metadata else {},
                'series2_info': self.series2.metadata.additional_info if self.series2.metadata else {},
                'spread_type': 'simple'
            }
        )
    
    def get_series1(self) -> TimeSeries:
        """
        Get the first time series.
        
        Returns:
            TimeSeries: The first time series
        """
        return self.series1
    
    def get_series2(self) -> TimeSeries:
        """
        Get the second time series.
        
        Returns:
            TimeSeries: The second time series
        """
        return self.series2
    
    def get_hedge_ratio(self) -> float:
        """
        Get the current hedge ratio.
        
        Returns:
            The hedge ratio used to calculate the spread
        """
        return self.hedge_ratio
    
    def recalculate_hedge_ratio(self, window: Optional[int] = None) -> float:
        """
        Recalculate the hedge ratio using the most recent data.
        
        Args:
            window: Optional window size for calculation. If None, uses all data.
            
        Returns:
            New hedge ratio
        """
        # Get aligned data
        s1_col = self.series1.columns[0]
        s2_col = self.series2.columns[0]
        
        aligned_data = self.series1.join(self.series2, how='inner', lsuffix='_s1', rsuffix='_s2')
        
        if len(aligned_data.columns) == 2:
            series1_col = aligned_data.columns[0]
            series2_col = aligned_data.columns[1]
        else:
            series1_col = f"{s1_col}_s1" if f"{s1_col}_s1" in aligned_data.columns else s1_col
            series2_col = f"{s2_col}_s2" if f"{s2_col}_s2" in aligned_data.columns else s2_col
        
        # Use window if specified
        if window is not None and len(aligned_data) > window:
            aligned_data = aligned_data.tail(window)
        
        new_hedge_ratio = self._calculate_hedge_ratio(aligned_data[series1_col], aligned_data[series2_col])
        
        logger.info(f"Hedge ratio updated from {self.hedge_ratio:.4f} to {new_hedge_ratio:.4f}")
        return new_hedge_ratio
    
    def update_hedge_ratio(self, new_hedge_ratio: float) -> 'SpreadTimeSeries':
        """
        Create a new SpreadTimeSeries with an updated hedge ratio.
        
        Args:
            new_hedge_ratio: New hedge ratio to use
            
        Returns:
            New SpreadTimeSeries object with updated hedge ratio
        """
        return SpreadTimeSeries(self.series1, self.series2, new_hedge_ratio)
    
    def get_spread_stats(self) -> Dict[str, float]:
        """
        Get statistical summary of the spread.
        
        Returns:
            Dict containing spread statistics
        """
        spread_series = self['spread']
        
        return {
            'mean': spread_series.mean(),
            'median': spread_series.median(),
            'std': spread_series.std(),
            'min': spread_series.min(),
            'max': spread_series.max(),
            'skew': spread_series.skew(),
            'kurt': spread_series.kurt(),
            'count': spread_series.count(),
            'hedge_ratio': self.hedge_ratio
        }
    
    def rolling_spread_stats(self, window: int, min_periods: Optional[int] = None) -> pd.DataFrame:
        """
        Calculate rolling statistics for the spread.
        
        Args:
            window: Size of the rolling window
            min_periods: Minimum number of observations required
            
        Returns:
            DataFrame with rolling spread statistics
        """
        return self.rolling_stats(window, ['mean', 'std', 'min', 'max', 'skew', 'kurt'], min_periods)
    
    def spread_z_score(self, window: int = 252, min_periods: Optional[int] = None) -> pd.Series:
        """
        Calculate z-score of the spread using rolling mean and standard deviation.
        
        Args:
            window: Rolling window size for mean and std calculation
            min_periods: Minimum periods required
            
        Returns:
            Series with z-scores
        """
        if min_periods is None:
            min_periods = window
        
        spread_series = self['spread']
        rolling_mean = spread_series.rolling(window, min_periods=min_periods).mean()
        rolling_std = spread_series.rolling(window, min_periods=min_periods).std()
        
        z_score = (spread_series - rolling_mean) / rolling_std
        return z_score
    
    def spread_percentile(self, window: int = 252, min_periods: Optional[int] = None) -> pd.Series:
        """
        Calculate rolling percentile of current spread value within the window.
        
        Args:
            window: Rolling window size
            min_periods: Minimum periods required
            
        Returns:
            Series with percentile values (0-1)
        """
        if min_periods is None:
            min_periods = window
        
        spread_series = self['spread']
        
        def calc_percentile(x):
            if len(x) < min_periods:
                return np.nan
            current_value = x.iloc[-1]
            return (x < current_value).sum() / len(x)
        
        return spread_series.rolling(window, min_periods=min_periods).apply(calc_percentile)
    
    def mean_reversion_signals(self, z_threshold: float = 2.0, window: int = 252) -> pd.DataFrame:
        """
        Generate mean reversion trading signals based on z-score.
        
        Args:
            z_threshold: Z-score threshold for signal generation
            window: Window for z-score calculation
            
        Returns:
            DataFrame with signals (-1: short spread, 0: neutral, 1: long spread)
        """
        z_score = self.spread_z_score(window)
        
        signals = pd.DataFrame(index=self.index)
        signals['z_score'] = z_score
        signals['signal'] = 0
        
        # Long spread when z-score is below negative threshold (spread is cheap)
        signals.loc[z_score < -z_threshold, 'signal'] = 1
        
        # Short spread when z-score is above positive threshold (spread is expensive)
        signals.loc[z_score > z_threshold, 'signal'] = -1
        
        # Add signal strength (absolute z-score)
        signals['signal_strength'] = np.abs(z_score)
        
        # Add entry/exit flags
        signals['entry'] = signals['signal'] != signals['signal'].shift(1)
        signals['exit'] = (signals['signal'] == 0) & (signals['signal'].shift(1) != 0)
        
        return signals
    
    def rolling_hedge_ratio(self, window: int, min_periods: Optional[int] = None) -> pd.Series:
        """
        Calculate rolling hedge ratio over time.
        
        Args:
            window: Rolling window size
            min_periods: Minimum periods required
            
        Returns:
            Series with rolling hedge ratios
        """
        if min_periods is None:
            min_periods = window
        
        # Get aligned data
        s1_col = self.series1.columns[0]
        s2_col = self.series2.columns[0]
        
        aligned_data = self.series1.join(self.series2, how='inner', lsuffix='_s1', rsuffix='_s2')
        
        if len(aligned_data.columns) == 2:
            series1_col = aligned_data.columns[0]
            series2_col = aligned_data.columns[1]
        else:
            series1_col = f"{s1_col}_s1" if f"{s1_col}_s1" in aligned_data.columns else s1_col
            series2_col = f"{s2_col}_s2" if f"{s2_col}_s2" in aligned_data.columns else s2_col
        
        def calc_hedge_ratio(data):
            if len(data) < min_periods:
                return np.nan
            try:
                x = data[series1_col]
                y = data[series2_col]
                valid_data = pd.DataFrame({'x': x, 'y': y}).dropna()
                if len(valid_data) < 2:
                    return np.nan
                
                x_clean = valid_data['x']
                y_clean = valid_data['y']
                
                # Simple correlation-based hedge ratio
                return (x_clean.cov(y_clean) / x_clean.var()) if x_clean.var() != 0 else np.nan
            except:
                return np.nan
        
        return aligned_data.rolling(window, min_periods=min_periods).apply(calc_hedge_ratio)
    
    def cointegration_test(self) -> Dict[str, Any]:
        """
        Perform basic cointegration test on the two series.
        
        Returns:
            Dictionary with cointegration test results
        """
        try:
            from statsmodels.tsa.stattools import coint
            
            # Get aligned data
            s1_col = self.series1.columns[0]
            s2_col = self.series2.columns[0]
            
            aligned_data = self.series1.join(self.series2, how='inner', lsuffix='_s1', rsuffix='_s2')
            
            if len(aligned_data.columns) == 2:
                series1_data = aligned_data.iloc[:, 0].dropna()
                series2_data = aligned_data.iloc[:, 1].dropna()
            else:
                series1_col = f"{s1_col}_s1" if f"{s1_col}_s1" in aligned_data.columns else s1_col
                series2_col = f"{s2_col}_s2" if f"{s2_col}_s2" in aligned_data.columns else s2_col
                series1_data = aligned_data[series1_col].dropna()
                series2_data = aligned_data[series2_col].dropna()
            
            # Align the series
            common_index = series1_data.index.intersection(series2_data.index)
            s1_aligned = series1_data.loc[common_index]
            s2_aligned = series2_data.loc[common_index]
            
            # Perform cointegration test
            coint_stat, p_value, critical_values = coint(s1_aligned, s2_aligned)
            
            return {
                'cointegration_statistic': coint_stat,
                'p_value': p_value,
                'critical_values': {
                    '1%': critical_values[0],
                    '5%': critical_values[1],
                    '10%': critical_values[2]
                },
                'is_cointegrated_5pct': p_value < 0.05,
                'is_cointegrated_1pct': p_value < 0.01
            }
            
        except ImportError:
            logger.warning("statsmodels not available for cointegration test")
            return {'error': 'statsmodels package required for cointegration test'}
        except Exception as e:
            logger.error(f"Error in cointegration test: {e}")
            return {'error': str(e)}
    
    def to_dict_ts(self) -> Dict[str, Any]:
        """
        Convert the SpreadTimeSeries to a dictionary representation.
        
        Returns:
            Dictionary containing the spread time series data and metadata
        """
        base_dict = super().to_dict_ts()
        base_dict.update({
            'series1': self.series1.to_dict_ts(),
            'series2': self.series2.to_dict_ts(),
            'hedge_ratio': self.hedge_ratio,
            'spread_type': 'SpreadTimeSeries'
        })
        return base_dict
    
    @classmethod
    def from_dict_ts(cls, data_dict: Dict[str, Any]) -> 'SpreadTimeSeries':
        """
        Create a SpreadTimeSeries object from a dictionary representation.
        
        Args:
            data_dict: Dictionary containing spread time series data and metadata
            
        Returns:
            New SpreadTimeSeries object
        """
        series1 = TimeSeries.from_dict_ts(data_dict['series1'])
        series2 = TimeSeries.from_dict_ts(data_dict['series2'])
        hedge_ratio = data_dict['hedge_ratio']
        return cls(series1, series2, hedge_ratio)
    
    def __repr__(self) -> str:
        """String representation of the SpreadTimeSeries object."""
        df_repr = pd.DataFrame.__repr__(self)
        if self.metadata:
            metadata_info = (f"\nSpreadTimeSeries Metadata:\n"
                           f"Spread: {self.metadata.symbol}\n"
                           f"Series1: {self.metadata.additional_info.get('series1_symbol', 'Unknown')}\n"
                           f"Series2: {self.metadata.additional_info.get('series2_symbol', 'Unknown')}\n"
                           f"Hedge Ratio: {self.hedge_ratio:.4f}\n"
                           f"Frequency: {self.metadata.frequency}\n"
                           f"Source: {self.metadata.source}")
            return df_repr + metadata_info
        return df_repr
