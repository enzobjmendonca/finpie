from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
import pandas as pd
import numpy as np

@dataclass
class TimeSeriesMetadata:
    """Metadata for a time series."""
    symbol: str
    source: str
    start_date: datetime
    end_date: datetime
    frequency: str
    currency: str
    additional_info: Dict[str, Any]

class TimeSeries:
    """
    Base class for time series data.
    
    This class provides core functionality for handling time series data,
    including basic operations like resampling, returns calculation,
    and statistical measures.
    """
    
    def __init__(self, data: pd.DataFrame, metadata: TimeSeriesMetadata):
        """
        Initialize a TimeSeries object.
        
        Args:
            data: DataFrame with datetime index and price columns
            metadata: TimeSeriesMetadata object containing series information
        """
        self.data = data
        self.metadata = metadata
        
        # Validate data
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data must have a DatetimeIndex")
            
        # Sort index if not already sorted
        if not data.index.is_monotonic_increasing:
            self.data = data.sort_index()
    
    @property
    def start_date(self) -> datetime:
        """Get the start date of the time series."""
        return self.data.index[0]
    
    @property
    def end_date(self) -> datetime:
        """Get the end date of the time series."""
        return self.data.index[-1]
    
    @property
    def frequency(self) -> str:
        """Get the frequency of the time series."""
        return self.metadata.frequency
    
    def resample(self, freq: str) -> 'TimeSeries':
        """
        Resample the time series to a different frequency.
        
        Args:
            freq: Target frequency (e.g., '1D' for daily, '1H' for hourly)
            
        Returns:
            New TimeSeries object with resampled data
        """
        resampled_data = self.data.resample(freq).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum' if 'volume' in self.data.columns else None
        }).dropna()
        
        # Create new metadata with updated frequency
        new_metadata = TimeSeriesMetadata(
            symbol=self.metadata.symbol,
            source=self.metadata.source,
            start_date=resampled_data.index[0],
            end_date=resampled_data.index[-1],
            frequency=freq,
            currency=self.metadata.currency,
            additional_info=self.metadata.additional_info
        )
        
        return TimeSeries(resampled_data, new_metadata)
    
    def returns(self, method: str = 'simple') -> 'TimeSeries':
        """
        Calculate returns for the time series.
        
        Args:
            method: Return calculation method ('log' or 'simple')
            
        Returns:
            New TimeSeries object with returns data
        """
        if method not in ['log', 'simple']:
            raise ValueError("Method must be either 'log' or 'simple'")
            
        if method == 'log':
            returns_df = np.log(self.data / self.data.shift(1))
        else:
            returns_df = self.data.pct_change()
            
        # Create new metadata
        if (self.metadata != None):
            new_metadata = TimeSeriesMetadata(
                symbol=f"{self.metadata.symbol}_returns",
                source=self.metadata.source,
                start_date=returns_df.index[0],
                end_date=returns_df.index[-1],
                frequency=self.metadata.frequency,
                currency=self.metadata.currency,
                additional_info={
                **self.metadata.additional_info,
                    'return_type': method
                }
            )
        else:
            new_metadata = None
        
        return TimeSeries(returns_df, new_metadata)
    
    def rolling(self, window: int, min_periods: Optional[int] = None) -> 'TimeSeries':
        """
        Calculate rolling statistics for the time series.
        
        Args:
            window: Size of the rolling window
            min_periods: Minimum number of observations required
            
        Returns:
            New TimeSeries object with rolling statistics
        """
        if min_periods is None:
            min_periods = window
            
        rolling_data = pd.DataFrame({
            'mean': self.data.rolling(window, min_periods=min_periods).mean(),
            'std': self.data.rolling(window, min_periods=min_periods).std(),
            'min': self.data.rolling(window, min_periods=min_periods).min(),
            'max': self.data.rolling(window, min_periods=min_periods).max()
        })
        
        # Create new metadata
        new_metadata = TimeSeriesMetadata(
            symbol=f"{self.metadata.symbol}_rolling",
            source=self.metadata.source,
            start_date=rolling_data.index[0],
            end_date=rolling_data.index[-1],
            frequency=self.metadata.frequency,
            currency=self.metadata.currency,
            additional_info={
                **self.metadata.additional_info,
                'window': window,
                'min_periods': min_periods
            }
        )
        
        return TimeSeries(rolling_data, new_metadata)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the time series to a dictionary representation.
        
        Returns:
            Dictionary containing the time series data and metadata
        """
        return {
            'data': self.data.to_dict(),
            'metadata': {
                'symbol': self.metadata.symbol,
                'source': self.metadata.source,
                'start_date': self.metadata.start_date.isoformat(),
                'end_date': self.metadata.end_date.isoformat(),
                'frequency': self.metadata.frequency,
                'currency': self.metadata.currency,
                'additional_info': self.metadata.additional_info
            }
        }
    
    @classmethod
    def from_dict(cls, data_dict: Dict[str, Any]) -> 'TimeSeries':
        """
        Create a TimeSeries object from a dictionary representation.
        
        Args:
            data_dict: Dictionary containing time series data and metadata
            
        Returns:
            New TimeSeries object
        """
        # Convert data dictionary to DataFrame
        data = pd.DataFrame.from_dict(data_dict['data'])
        data.index = pd.to_datetime(data.index)
        
        # Create metadata
        metadata = TimeSeriesMetadata(
            symbol=data_dict['metadata']['symbol'],
            source=data_dict['metadata']['source'],
            start_date=datetime.fromisoformat(data_dict['metadata']['start_date']),
            end_date=datetime.fromisoformat(data_dict['metadata']['end_date']),
            frequency=data_dict['metadata']['frequency'],
            currency=data_dict['metadata']['currency'],
            additional_info=data_dict['metadata']['additional_info']
        )
        
        return cls(data, metadata)
    
    def __repr__(self) -> str:
        """String representation of the TimeSeries object."""
        if (self.metadata != None):
            return (f"TimeSeries(symbol='{self.metadata.symbol}', "
                    f"source='{self.metadata.source}', "
                    f"start_date='{self.start_date}', "
                    f"end_date='{self.end_date}', "
                    f"frequency='{self.frequency}')")
        else:
            return (f"TimeSeries(data={self.data}, "
                    f"metadata={self.metadata})")

    def cum_returns(self, intraday_only: bool = False) -> pd.Series:
        """
        Calculate the cumulative returns of the time series.
        """
        returns = self.returns(intraday_only)
        return returns.data.add(1).cumprod() - 1
    
    def volatility(self, intraday_only: bool = False) -> pd.Series:
        """
        Calculate the volatility of the time series.
        """
        returns = self.returns(intraday_only)
        return returns.data.std() 
    
    def mean_return(self, intraday_only: bool = False) -> pd.Series:
        """
        Calculate the mean return of the time series.
        """
        returns = self.returns(intraday_only)
        return returns.data.mean()
    
    def sharpe_ratio(self, intraday_only: bool = False) -> pd.Series:
        """
        Calculate the Sharpe ratio of the time series.
        """
        returns = self.returns(intraday_only)
        return (returns.data.mean() / returns.data.std()) * np.sqrt(252)
    
    def max_drawdown(self, intraday_only: bool = False) -> pd.Series:
        """
        Calculate the maximum drawdown of the time series.
        """
        returns = self.returns(intraday_only)
        # Calculate cumulative returns
        cum_rets = self.cum_returns(intraday_only)
        # Calculate running maximum
        running_max = cum_rets.expanding().max()
        # Calculate drawdown
        drawdown = cum_rets - running_max
        # Get the maximum drawdown
        return drawdown.min()

    def value_at_risk(self, confidence_level: float = 0.05, intraday_only: bool = False) -> pd.Series:
        """
        Calculate the Value at Risk (VaR) of the time series.
        """
        returns = self.returns(intraday_only)
        return returns.quantile(confidence_level)    
    
    def expected_shortfall(self, confidence_level: float = 0.05, intraday_only: bool = False) -> pd.Series:
        """
        Calculate the Expected Shortfall (ES) of the time series.
        """
        returns = self.returns(intraday_only)
        return returns.quantile(confidence_level)    
    
    def skewness(self, intraday_only: bool = False) -> pd.Series:
        """
        Calculate the skewness of the time series.
        """
        returns = self.returns(intraday_only)
        return returns.skew()    
    
    def kurtosis(self, intraday_only: bool = False) -> pd.Series:
        """
        Calculate the kurtosis of the time series.
        """
        returns = self.returns(intraday_only)
        return returns.kurt()    
    
    def autocorrelation(self, lag: int = 1, intraday_only: bool = False) -> pd.Series:
        """
        Calculate the autocorrelation of the time series.
        """
        returns = self.returns(intraday_only)
        return returns.autocorr(lag)
        
    
    
    
    
    
    
    
    
