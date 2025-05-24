from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
import pandas as pd
import numpy as np
import logging

# Configure logger
logger = logging.getLogger(__name__)

@dataclass
class TimeSeriesMetadata:
    """Metadata for a time series."""
    name: str
    symbol: str
    source: str
    start_date: datetime
    end_date: datetime
    frequency: str
    currency: str
    additional_info: Dict[str, Any]
    is_returns: bool = False

class TimeSeries:
    """
    Base class for time series data.
    
    This class provides core functionality for handling time series data,
    including basic operations like resampling, returns calculation,
    and statistical measures.
    """
    
    def __init__(self, data: pd.DataFrame, metadata: TimeSeriesMetadata = None):
        """
        Initialize a TimeSeries object.
        
        Args:
            data: DataFrame with datetime index and price columns
            metadata: TimeSeriesMetadata object containing series information
        """
        logger.info(f"Initializing TimeSeries for symbol: {metadata.symbol if metadata else 'Unknown'}")
        self.data = data
        self.metadata = metadata if metadata != None else TimeSeriesMetadata(name='', symbol='', source='', 
                                                                             start_date=None, end_date=None, frequency='', 
                                                                             currency='', additional_info={})
        # Validate data
        if not isinstance(data.index, pd.DatetimeIndex):
            logger.error("Data must have a DatetimeIndex")
            raise ValueError("Data must have a DatetimeIndex")
        
        if isinstance(data, pd.DataFrame) and len(data.columns) > 1:
            logger.error("Data must have only one column, please use MultiTimeSeries for multiple columns")
            raise ValueError("Data must have only one column, please use MultiTimeSeries for multiple columns")

        if isinstance(data, pd.Series):
            logger.debug("Converting Series to DataFrame")
            self.metadata.name = self.data.name
            self.data = data.to_frame()
        else:
            self.metadata.name = self.data.columns[0]

        # Sort index if not already sorted
        if not data.index.is_monotonic_increasing and not data.index.is_monotonic_decreasing:
            logger.debug("Sorting index as it's not monotonic increasing or decreasing")
            self.data = data.sort_index()
            
        logger.info(f"TimeSeries initialized with {len(self.data)} data points from {self.start_date} to {self.end_date}")
    
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
        logger.info(f"Resampling data to frequency: {freq}")
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
        
        logger.info(f"Resampling complete. New data points: {len(resampled_data)}")
        return TimeSeries(resampled_data, new_metadata)
    
    def returns(self, intraday_only: bool = False, method: str = 'simple') -> 'TimeSeries':
        """
        Calculate returns for the time series.
        
        Args:
            intraday_only: Whether to drop the first record of each day
            method: Return calculation method ('log' or 'simple')
            
        Returns:
            TimeSeries object with returns data
        """
        logger.info(f"Calculating {method} returns with intraday_only={intraday_only}")
        
        if method not in ['log', 'simple']:
            logger.error(f"Invalid method: {method}. Must be either 'log' or 'simple'")
            raise ValueError("Method must be either 'log' or 'simple'")
            
        if method == 'log':
            returns_df = np.log(self.data / self.data.shift(1))
            logger.debug("Calculated log returns")
        else:
            returns_df = self.data.pct_change()
            logger.debug("Calculated simple returns")
        
        if intraday_only:
            logger.debug("Dropping first record of each day")
            # Group by date and drop first record of each day
            returns_df = returns_df.groupby(returns_df.index.date).apply(lambda x: x.iloc[1:]).reset_index(level=0, drop=True)
            
        returns_metadata = TimeSeriesMetadata(
            name=self.metadata.name + '_returns',
            symbol=self.metadata.symbol + '_returns' if self.metadata != None else None,
            source=self.metadata.source if self.metadata != None else None,
            start_date=returns_df.index[0],
            end_date=returns_df.index[-1],
            is_returns=True,
            frequency=self.metadata.frequency if self.metadata != None else None,
            currency=self.metadata.currency if self.metadata != None else None,
            additional_info=self.metadata.additional_info if self.metadata != None else {}
        )
        
        logger.info(f"Returns calculation complete. Data points: {len(returns_df)}")
        return TimeSeries(returns_df, returns_metadata)
    
    def rolling(self, window: int, stats: List[str] = ['mean', 'std', 'min', 'max'], min_periods: Optional[int] = None) -> pd.DataFrame:
        """
        Calculate rolling statistics for the time series.
        
        Args:
            window: Size of the rolling window
            stats: List of statistics to calculate available stats: ['mean', 'std', 'min', 'max', 'skew', 'kurt', 'sum', 'count', 'median', 'var', 'quantiles']
            min_periods: Minimum number of observations required
            
        Returns:
            New TimeSeries object with rolling statistics
        """
        if min_periods is None:
            min_periods = window
            
        rolling_data = pd.DataFrame()

        for col in self.data.columns:
            for stat in stats:
                if stat == 'quantiles':
                    for q in [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]:
                        rolling_data[f'{col}_quantile_{q}'] = self.data.rolling(window, min_periods=min_periods)[col].quantile(q)
                elif stat in ['mean', 'std', 'min', 'max', 'skew', 'kurt', 'sum', 'count', 'median', 'var']:
                    rolling_data[f'{col}_{stat}'] = self.data.rolling(window, min_periods=min_periods)[col].agg(stat)
                else:
                    raise ValueError(f"Invalid statistic: {stat}")
        return rolling_data
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the time series to a dictionary representation.
        
        Returns:
            Dictionary containing the time series data and metadata
        """
        return {
            'data': self.data.to_dict(),
            'metadata': {
                'name': self.metadata.name,
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
                    f"name='{self.metadata.name}', "
                    f"source='{self.metadata.source}', "
                    f"start_date='{self.start_date}', "
                    f"end_date='{self.end_date}', "
                    f"frequency='{self.frequency}')")
        else:
            return (f"TimeSeries(data={self.data}, "
                    f"metadata={self.metadata})")

    def cum_returns(self, intraday_only: bool = False, method: str = 'simple') -> pd.Series:
        """
        Calculate the cumulative returns of the time series.
        """
        returns = self.returns(intraday_only, method)
        return returns.data.add(1).cumprod() - 1
    
    def volatility(self, intraday_only: bool = False, method: str = 'simple') -> pd.Series:
        """
        Calculate the volatility of the time series.
        """
        returns = self.returns(intraday_only, method)
        return returns.data.std() 
    
    def mean_return(self, intraday_only: bool = False, method: str = 'simple') -> pd.Series:
        """
        Calculate the mean return of the time series.
        """
        returns = self.returns(intraday_only, method)
        return returns.data.mean()
    
    def sharpe_ratio(self, intraday_only: bool = False, method: str = 'simple') -> pd.Series:
        """
        Calculate the Sharpe ratio of the time series.
        """
        returns = self.returns(intraday_only, method)
        return (returns.data.mean() / returns.data.std()) * np.sqrt(252)
    
    def max_drawdown(self, percentage: bool = True, intraday_only: bool = False, method: str = 'simple') -> pd.Series:
        """
        Calculate the maximum drawdown of the time series.
        """
        # Calculate cumulative returns
        if percentage:
            cum_rets = self.cum_returns(intraday_only, method)
        else:
            cum_rets = self.data
        # Calculate running maximum
        running_max = cum_rets.expanding().max()
        # Calculate drawdown
        drawdown = cum_rets - running_max
        # Get the maximum drawdown
        return drawdown.min()

    def value_at_risk(self, percentage: bool = True, confidence_level: float = 0.05, intraday_only: bool = False, method: str = 'simple') -> pd.Series:
        """
        Calculate the Value at Risk (VaR) of the time series.
        """
        if percentage:
            return self.returns(intraday_only, method).data.quantile(confidence_level)
        else:
            return self.data.diff().quantile(confidence_level)
    
    def skewness(self, intraday_only: bool = False, method: str = 'simple') -> pd.Series:
        """
        Calculate the skewness of the time series.
        """
        return self.returns(intraday_only, method).data.skew()
    
    def kurtosis(self, intraday_only: bool = False, method: str = 'simple') -> pd.Series:
        """
        Calculate the kurtosis of the time series.
        """
        return self.returns(intraday_only, method).data.kurt()
    
    def autocorrelation(self, lag: int = 1, intraday_only: bool = False, method: str = 'simple') -> pd.Series:
        """
        Calculate the autocorrelation of the time series.
        """
        returns_df = self.returns(intraday_only, method).data
        acorr_map = {}
        for col in returns_df.columns:
            acorr_map[col] = returns_df[col].autocorr(lag)
        return pd.Series(acorr_map)
        
    
    
    
    
    
    
    
    
