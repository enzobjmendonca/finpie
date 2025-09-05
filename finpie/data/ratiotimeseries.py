from typing import Dict, Any, Optional, Union
import pandas as pd
import numpy as np
import logging

from .timeseries import TimeSeries, TimeSeriesMetadata

# Configure logger
logger = logging.getLogger(__name__)

class RatioTimeSeries(TimeSeries):
    """A class for handling ratio-based time series.
    
    This class extends TimeSeries to provide functionality for analyzing the ratio between two time series.
    It's commonly used in pair trading and relative value strategies, where the relationship between
    two assets is analyzed through their price ratio.
    
    Since it extends TimeSeries (which extends pd.DataFrame), it inherits all pandas DataFrame
    functionality while adding specialized ratio-specific operations.
    
    Attributes:
        numerator (TimeSeries): The numerator time series
        denominator (TimeSeries): The denominator time series
    """
    
    def __init__(self, numerator: Union[TimeSeries, pd.DataFrame, pd.Series], 
                 denominator: Union[TimeSeries, pd.DataFrame, pd.Series],
                 metadata: TimeSeriesMetadata = None):
        """Initialize a RatioTimeSeries object.
        
        Args:
            numerator: TimeSeries object, DataFrame, or Series for the numerator
            denominator: TimeSeries object, DataFrame, or Series for the denominator
            metadata: Optional metadata for the ratio time series
            
        Note:
            If numerator or denominator are not TimeSeries objects, they will be converted automatically.
        """
        # Convert inputs to TimeSeries if needed
        if not isinstance(numerator, TimeSeries):
            if isinstance(numerator, pd.Series):
                numerator = numerator.to_frame()
            numerator_metadata = TimeSeriesMetadata(
                name=numerator.columns[0] if hasattr(numerator, 'columns') and len(numerator.columns) > 0 else 'numerator',
                symbol=numerator.columns[0] if hasattr(numerator, 'columns') and len(numerator.columns) > 0 else 'numerator',
                source='unknown',
                start_date=numerator.index[0] if len(numerator.index) > 0 else None,
                end_date=numerator.index[-1] if len(numerator.index) > 0 else None,
                frequency='',
                currency='',
                additional_info={}
            )
            numerator = TimeSeries(numerator, numerator_metadata)
            
        if not isinstance(denominator, TimeSeries):
            if isinstance(denominator, pd.Series):
                denominator = denominator.to_frame()
            denominator_metadata = TimeSeriesMetadata(
                name=denominator.columns[0] if hasattr(denominator, 'columns') and len(denominator.columns) > 0 else 'denominator',
                symbol=denominator.columns[0] if hasattr(denominator, 'columns') and len(denominator.columns) > 0 else 'denominator',
                source='unknown',
                start_date=denominator.index[0] if len(denominator.index) > 0 else None,
                end_date=denominator.index[-1] if len(denominator.index) > 0 else None,
                frequency='',
                currency='',
                additional_info={}
            )
            denominator = TimeSeries(denominator, denominator_metadata)
        
        # Store references to original series
        self.numerator = numerator
        self.denominator = denominator
        
        # Align the two time series
        aligned_data = self._align_and_calculate_ratio()
        
        # Create metadata if not provided
        if metadata is None:
            metadata = self._create_ratio_metadata()
        
        # Initialize parent TimeSeries with ratio data
        super().__init__(aligned_data, metadata)
    
    def _align_and_calculate_ratio(self) -> pd.DataFrame:
        """Align the numerator and denominator series and calculate the ratio."""
        # Get the data columns (assuming single column for each)
        num_col = self.numerator.columns[0]
        den_col = self.denominator.columns[0]
        
        # Align data using inner join to avoid division by NaN
        aligned_data = self.numerator.join(
            self.denominator, 
            how='inner', 
            lsuffix='_numerator', 
            rsuffix='_denominator'
        )
        
        # If column names are the same, pandas will add suffixes
        if len(aligned_data.columns) == 2:
            numerator_col = aligned_data.columns[0]
            denominator_col = aligned_data.columns[1]
        else:
            # Find the correct columns
            numerator_col = f"{num_col}_numerator" if f"{num_col}_numerator" in aligned_data.columns else num_col
            denominator_col = f"{den_col}_denominator" if f"{den_col}_denominator" in aligned_data.columns else den_col
        
        # Calculate ratio, avoiding division by zero
        ratio_values = aligned_data[numerator_col] / aligned_data[denominator_col]
        
        # Replace inf values with NaN
        ratio_values = ratio_values.replace([np.inf, -np.inf], np.nan)
        
        # Create DataFrame with ratio
        ratio_data = pd.DataFrame({'ratio': ratio_values})
        
        return ratio_data
    
    def _create_ratio_metadata(self) -> TimeSeriesMetadata:
        """Create metadata for the ratio time series."""
        num_symbol = self.numerator.metadata.symbol if self.numerator.metadata else 'unknown'
        den_symbol = self.denominator.metadata.symbol if self.denominator.metadata else 'unknown'
        num_name = self.numerator.metadata.name if self.numerator.metadata else 'unknown'
        den_name = self.denominator.metadata.name if self.denominator.metadata else 'unknown'
        
        return TimeSeriesMetadata(
            name=f"{num_name}/{den_name}",
            symbol=f"{num_symbol}/{den_symbol}",
            source="ratio",
            start_date=self.index[0] if len(self.index) > 0 else None,
            end_date=self.index[-1] if len(self.index) > 0 else None,
            frequency=self.numerator.metadata.frequency if self.numerator.metadata else '',
            currency=self.numerator.metadata.currency if self.numerator.metadata else '',
            additional_info={
                'numerator_symbol': num_symbol,
                'denominator_symbol': den_symbol,
                'numerator_info': self.numerator.metadata.additional_info if self.numerator.metadata else {},
                'denominator_info': self.denominator.metadata.additional_info if self.denominator.metadata else {},
                'ratio_type': 'simple'
            }
        )
    
    def get_numerator(self) -> TimeSeries:
        """
        Get the numerator time series.
        
        Returns:
            TimeSeries: The numerator time series
        """
        return self.numerator
    
    def get_denominator(self) -> TimeSeries:
        """
        Get the denominator time series.
        
        Returns:
            TimeSeries: The denominator time series
        """
        return self.denominator
    
    def get_ratio_stats(self) -> Dict[str, float]:
        """
        Get statistical summary of the ratio.
        
        Returns:
            Dict containing ratio statistics
        """
        ratio_series = self['ratio']
        
        return {
            'mean': ratio_series.mean(),
            'median': ratio_series.median(),
            'std': ratio_series.std(),
            'min': ratio_series.min(),
            'max': ratio_series.max(),
            'skew': ratio_series.skew(),
            'kurt': ratio_series.kurt(),
            'count': ratio_series.count()
        }
    
    def rebalance_ratio(self, new_numerator: Union[TimeSeries, pd.DataFrame, pd.Series] = None,
                       new_denominator: Union[TimeSeries, pd.DataFrame, pd.Series] = None) -> 'RatioTimeSeries':
        """
        Create a new RatioTimeSeries with updated numerator or denominator.
        
        Args:
            new_numerator: New numerator series (if None, uses existing)
            new_denominator: New denominator series (if None, uses existing)
            
        Returns:
            New RatioTimeSeries object
        """
        updated_numerator = new_numerator if new_numerator is not None else self.numerator
        updated_denominator = new_denominator if new_denominator is not None else self.denominator
        
        return RatioTimeSeries(updated_numerator, updated_denominator)
    
    def rolling_ratio_stats(self, window: int, min_periods: Optional[int] = None) -> pd.DataFrame:
        """
        Calculate rolling statistics for the ratio.
        
        Args:
            window: Size of the rolling window
            min_periods: Minimum number of observations required
            
        Returns:
            DataFrame with rolling ratio statistics
        """
        return self.rolling_stats(window, ['mean', 'std', 'min', 'max', 'skew', 'kurt'], min_periods)
    
    def ratio_z_score(self, window: int = 252, min_periods: Optional[int] = None) -> pd.Series:
        """
        Calculate z-score of the ratio using rolling mean and standard deviation.
        
        Args:
            window: Rolling window size for mean and std calculation
            min_periods: Minimum periods required
            
        Returns:
            Series with z-scores
        """
        if min_periods is None:
            min_periods = window
        
        ratio_series = self['ratio']
        rolling_mean = ratio_series.rolling(window, min_periods=min_periods).mean()
        rolling_std = ratio_series.rolling(window, min_periods=min_periods).std()
        
        z_score = (ratio_series - rolling_mean) / rolling_std
        return z_score
    
    def ratio_percentile(self, window: int = 252, min_periods: Optional[int] = None) -> pd.Series:
        """
        Calculate rolling percentile of current ratio value within the window.
        
        Args:
            window: Rolling window size
            min_periods: Minimum periods required
            
        Returns:
            Series with percentile values (0-1)
        """
        if min_periods is None:
            min_periods = window
        
        ratio_series = self['ratio']
        
        def calc_percentile(x):
            if len(x) < min_periods:
                return np.nan
            current_value = x.iloc[-1]
            return (x < current_value).sum() / len(x)
        
        return ratio_series.rolling(window, min_periods=min_periods).apply(calc_percentile)
    
    def mean_reversion_signals(self, z_threshold: float = 2.0, window: int = 252) -> pd.DataFrame:
        """
        Generate mean reversion trading signals based on z-score.
        
        Args:
            z_threshold: Z-score threshold for signal generation
            window: Window for z-score calculation
            
        Returns:
            DataFrame with signals (-1: short ratio, 0: neutral, 1: long ratio)
        """
        z_score = self.ratio_z_score(window)
        
        signals = pd.DataFrame(index=self.index)
        signals['z_score'] = z_score
        signals['signal'] = 0
        
        # Long ratio when z-score is below negative threshold (ratio is cheap)
        signals.loc[z_score < -z_threshold, 'signal'] = 1
        
        # Short ratio when z-score is above positive threshold (ratio is expensive)
        signals.loc[z_score > z_threshold, 'signal'] = -1
        
        # Add signal strength (absolute z-score)
        signals['signal_strength'] = np.abs(z_score)
        
        return signals
    
    def to_dict_ts(self) -> Dict[str, Any]:
        """
        Convert the RatioTimeSeries to a dictionary representation.
        
        Returns:
            Dictionary containing the ratio time series data and metadata
        """
        base_dict = super().to_dict_ts()
        base_dict.update({
            'numerator': self.numerator.to_dict_ts(),
            'denominator': self.denominator.to_dict_ts(),
            'ratio_type': 'RatioTimeSeries'
        })
        return base_dict
    
    @classmethod
    def from_dict_ts(cls, data_dict: Dict[str, Any]) -> 'RatioTimeSeries':
        """
        Create a RatioTimeSeries object from a dictionary representation.
        
        Args:
            data_dict: Dictionary containing ratio time series data and metadata
            
        Returns:
            New RatioTimeSeries object
        """
        numerator = TimeSeries.from_dict_ts(data_dict['numerator'])
        denominator = TimeSeries.from_dict_ts(data_dict['denominator'])
        return cls(numerator, denominator)
    
    def __repr__(self) -> str:
        """String representation of the RatioTimeSeries object."""
        df_repr = pd.DataFrame.__repr__(self)
        if self.metadata:
            metadata_info = (f"\nRatioTimeSeries Metadata:\n"
                           f"Ratio: {self.metadata.symbol}\n"
                           f"Numerator: {self.metadata.additional_info.get('numerator_symbol', 'Unknown')}\n"
                           f"Denominator: {self.metadata.additional_info.get('denominator_symbol', 'Unknown')}\n"
                           f"Frequency: {self.metadata.frequency}\n"
                           f"Source: {self.metadata.source}")
            return df_repr + metadata_info
        return df_repr
