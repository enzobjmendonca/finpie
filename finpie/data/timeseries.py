from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple, Union
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

class TimeSeries(pd.DataFrame):
    """
    TimeSeries class that extends pandas DataFrame.
    
    This class provides core functionality for handling time series data,
    including basic operations like resampling, returns calculation,
    and statistical measures while maintaining full pandas DataFrame compatibility.
    """
    
    # Define metadata properties that should be preserved during pandas operations
    _metadata = ['_ts_metadata']
    
    def __init__(self, data: Union[pd.DataFrame, pd.Series, np.ndarray, dict, list] = None, 
                 metadata: TimeSeriesMetadata = None, index=None, columns=None, copy=None, **kwargs):
        """
        Initialize a TimeSeries object.
        
        Args:
            data: DataFrame, Series, array, dict, or list with datetime index and price columns
            metadata: TimeSeriesMetadata object containing series information
            index: Index to use for resulting frame
            columns: Column labels to use for resulting frame
            **kwargs: Additional arguments passed to pandas DataFrame constructor
        """
        # Handle different input types
        if isinstance(data, pd.Series):
            # Convert Series to DataFrame
            logger.debug("Converting Series to DataFrame")
            if data.name is None:
                data.name = 'close'
            data = data.to_frame()
        
        if isinstance(data, dict):
            data = pd.DataFrame(data)

        # Initialize DataFrame - handle copy parameter separately since it might conflict
        df_kwargs = kwargs.copy()
        if copy is not None:
            df_kwargs['copy'] = copy

        # Validate and process data
        self._validate_and_process(data, index)

        super().__init__(data=data, index=index, columns=columns, **df_kwargs)
        
        # Set default metadata if none provided
        if metadata is None:
            metadata = TimeSeriesMetadata(
                name=self.columns[0] if len(self.columns) > 0 else '',
                symbol='',
                source='',
                start_date=self.index[0] if len(self.index) > 0 else datetime.now(),
                end_date=self.index[-1] if len(self.index) > 0 else datetime.now(),
                frequency='',
                currency='',
                additional_info={}
            )
        
        # Store metadata
        self._ts_metadata = metadata
    
    def _validate_and_process(self, data, index):
        """Validate and process the time series data."""
        # Ensure datetime index
        if index is None and not isinstance(data.index, pd.DatetimeIndex):
            try:
                data.index = pd.to_datetime(data.index)
                logger.debug("Converted index to DatetimeIndex")
            except Exception as e:
                logger.error(f"Failed to convert index to DatetimeIndex: {e}")
                raise ValueError("Data must have a DatetimeIndex")
        
        # Check for single column constraint (only for base TimeSeries)
        if self.__class__.__name__ == 'TimeSeries' and len(data.columns) > 1:
            logger.error("Base TimeSeries must have only one column, use MultiTimeSeries for multiple columns")
            raise ValueError("Base TimeSeries must have only one column, use MultiTimeSeries for multiple columns")
        
        if index is None:
            index = data.index
        # Sort index if not already sorted
        if not index.is_monotonic_increasing and not index.is_monotonic_decreasing:
            logger.debug("Sorting index as it's not monotonic")
            data.sort_index(inplace=True)
    
    @property
    def _constructor(self):
        """Return the constructor for this class (used by pandas operations)."""
        return type(self)
    
    @property  
    def _constructor_sliced(self):
        """Return the constructor for sliced objects (typically Series)."""
        return pd.Series
    
    def __finalize__(self, other, method=None, **kwargs):
        """
        Propagate metadata from other to self.
        This method is called at the end of most pandas operations.
        """
        result = super().__finalize__(other, method, **kwargs)
        if isinstance(other, TimeSeries) and hasattr(other, '_ts_metadata'):
            object.__setattr__(result, '_ts_metadata', other._ts_metadata)
        return result
    
    @property
    def metadata(self) -> TimeSeriesMetadata:
        """Get the metadata of the time series."""
        if hasattr(self, '_ts_metadata'):
            return self._ts_metadata
        else:
            return TimeSeriesMetadata(
                name=self.columns[0] if len(self.columns) > 0 else '',
                symbol='',
                source='',
                start_date=self.index[0] if len(self.index) > 0 else datetime.now(),
                end_date=self.index[-1] if len(self.index) > 0 else datetime.now(),
                frequency='',
                currency='',
                additional_info={}
            )
    
    @metadata.setter
    def metadata(self, value: TimeSeriesMetadata):
        """Set the metadata of the time series."""
        self._ts_metadata = value
    
    @property
    def start_date(self) -> datetime:
        """Get the start date of the time series."""
        return self.index[0] if len(self.index) > 0 else None
    
    @property
    def end_date(self) -> datetime:
        """Get the end date of the time series."""
        return self.index[-1] if len(self.index) > 0 else None
    
    @property
    def frequency(self) -> str:
        """Get the frequency of the time series."""
        return self._ts_metadata.frequency if self._ts_metadata else None
    
    def resample_ts(self, freq: str) -> 'TimeSeries':
        """
        Resample the time series to a different frequency.
        
        Args:
            freq: Target frequency (e.g., '1D' for daily, '1H' for hourly)
            
        Returns:
            New TimeSeries object with resampled data
        """
        # Define aggregation rules for common OHLCV columns
        agg_rules = {}
        for col in self.columns:
            col_lower = col.lower()
            if 'open' in col_lower:
                agg_rules[col] = 'first'
            elif 'high' in col_lower:
                agg_rules[col] = 'max'
            elif 'low' in col_lower:
                agg_rules[col] = 'min'
            elif 'close' in col_lower or 'price' in col_lower:
                agg_rules[col] = 'last'
            elif 'volume' in col_lower:
                agg_rules[col] = 'sum'
            else:
                agg_rules[col] = 'last'  # Default to last value
        
        resampled_data = self.resample(freq).agg(agg_rules).dropna()
        
        # Create new metadata with updated frequency
        new_metadata = TimeSeriesMetadata(
            name=self._ts_metadata.name,
            symbol=self._ts_metadata.symbol,
            source=self._ts_metadata.source,
            start_date=resampled_data.index[0],
            end_date=resampled_data.index[-1],
            frequency=freq,
            currency=self._ts_metadata.currency,
            additional_info=self._ts_metadata.additional_info,
            is_returns=self._ts_metadata.is_returns
        )
        
        return TimeSeries(resampled_data, new_metadata)
    
    def returns(self, intraday_only: bool = False, method: str = 'absolute') -> 'TimeSeries':
        """
        Calculate returns for the time series.
        
        Args:
            intraday_only: Whether to drop the first record of each day
            method: Return calculation method ('log', 'simple', 'absolute')
            
        Returns:
            TimeSeries object with returns data
        """
        if method not in ['log', 'simple', 'absolute']:
            logger.error(f"Invalid method: {method}. Must be 'log', 'simple', or 'absolute'")
            raise ValueError("Method must be 'log', 'simple', or 'absolute'")
        
        if self._ts_metadata.is_returns:
            logger.warning("Time series is already returns, calculating returns on returns")
        
        if method == 'log':
            returns_df = np.log(self / self.shift(1))
            logger.debug("Calculated log returns")
        elif method == 'simple':
            returns_df = self.pct_change()
            logger.debug("Calculated simple returns")
        elif method == 'absolute':
            returns_df = self.diff()
            logger.debug("Calculated absolute returns")
        
        if intraday_only:
            logger.debug("Dropping first record of each day")
            # Group by date and drop first record of each day
            returns_df = returns_df.groupby(returns_df.index.date).apply(lambda x: x.iloc[1:])
            if isinstance(returns_df.index, pd.MultiIndex):
                returns_df.index = returns_df.index.droplevel(0)
        
        # Create new metadata for returns
        returns_metadata = TimeSeriesMetadata(
            name=self._ts_metadata.name + '_returns',
            symbol=self._ts_metadata.symbol + '_returns' if self._ts_metadata.symbol else '',
            source=self._ts_metadata.source,
            start_date=returns_df.index[0] if len(returns_df.index) > 0 else datetime.now(),
            end_date=returns_df.index[-1] if len(returns_df.index) > 0 else datetime.now(),
            is_returns=True,
            frequency=self._ts_metadata.frequency,
            currency=self._ts_metadata.currency,
            additional_info=self._ts_metadata.additional_info
        )
        
        return TimeSeries(returns_df, returns_metadata)
    
    def value(self, index: int) -> Union[float, pd.Series]:
        """
        Get the value of the time series at a specific index.
        
        Args:
            index: Integer index position
            
        Returns:
            Value(s) at the specified index
        """
        return self.iloc[index]
    
    def rolling_stats(self, window: int, stats: List[str] = ['mean', 'std', 'min', 'max'], 
                     min_periods: Optional[int] = None) -> pd.DataFrame:
        """
        Calculate rolling statistics for the time series.
        
        Args:
            window: Size of the rolling window
            stats: List of statistics to calculate
            min_periods: Minimum number of observations required
            
        Returns:
            DataFrame containing rolling statistics
        """
        if min_periods is None:
            min_periods = window
        
        rolling_data = pd.DataFrame(index=self.index)
        
        for col in self.columns:
            for stat in stats:
                if stat == 'quantiles':
                    for q in [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]:
                        rolling_data[f'{col}_quantile_{q}'] = self[col].rolling(
                            window, min_periods=min_periods).quantile(q)
                elif stat in ['mean', 'std', 'min', 'max', 'skew', 'kurt', 'sum', 'count', 'median', 'var']:
                    rolling_data[f'{col}_{stat}'] = getattr(
                        self[col].rolling(window, min_periods=min_periods), stat)()
                else:
                    raise ValueError(f"Invalid statistic: {stat}")
        
        return rolling_data
    
    def to_dict_ts(self) -> Dict[str, Any]:
        """
        Convert the time series to a dictionary representation.
        
        Returns:
            Dictionary containing the time series data and metadata
        """
        return {
            'data': self.to_dict(),
            'metadata': {
                'name': self._ts_metadata.name,
                'symbol': self._ts_metadata.symbol,
                'source': self._ts_metadata.source,
                'start_date': self._ts_metadata.start_date.isoformat() if self._ts_metadata.start_date else None,
                'end_date': self._ts_metadata.end_date.isoformat() if self._ts_metadata.end_date else None,
                'frequency': self._ts_metadata.frequency,
                'currency': self._ts_metadata.currency,
                'additional_info': self._ts_metadata.additional_info,
                'is_returns': self._ts_metadata.is_returns
            }
        }
    
    @classmethod
    def from_dict_ts(cls, data_dict: Dict[str, Any]) -> 'TimeSeries':
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
        metadata_dict = data_dict['metadata']
        metadata = TimeSeriesMetadata(
            name=metadata_dict['name'],
            symbol=metadata_dict['symbol'],
            source=metadata_dict['source'],
            start_date=datetime.fromisoformat(metadata_dict['start_date']) if metadata_dict['start_date'] else None,
            end_date=datetime.fromisoformat(metadata_dict['end_date']) if metadata_dict['end_date'] else None,
            frequency=metadata_dict['frequency'],
            currency=metadata_dict['currency'],
            additional_info=metadata_dict['additional_info'],
            is_returns=metadata_dict.get('is_returns', False)
        )
        
        return cls(data, metadata)
    
    def cum_returns(self, intraday_only: bool = False, method: str = 'absolute') -> pd.DataFrame:
        """
        Calculate the cumulative returns of the time series.
        
        Args:
            intraday_only: Whether to use intraday only returns
            method: Return calculation method
            
        Returns:
            DataFrame with cumulative returns
        """
        returns = self.returns(intraday_only, method)
        return (returns + 1).cumprod() - 1
    
    def volatility(self, intraday_only: bool = False, method: str = 'absolute', 
                  annualized: bool = True) -> pd.Series:
        """
        Calculate the volatility of the time series.
        
        Args:
            intraday_only: Whether to use intraday only returns
            method: Return calculation method
            annualized: Whether to annualize the volatility
            
        Returns:
            Series with volatility values
        """
        returns = self.returns(intraday_only, method)
        volatility = returns.std()
        if annualized:
            return volatility * np.sqrt(252)
        else:
            return volatility
    
    def mean_return(self, intraday_only: bool = False, method: str = 'absolute') -> pd.Series:
        """
        Calculate the mean return of the time series.
        
        Args:
            intraday_only: Whether to use intraday only returns
            method: Return calculation method
            
        Returns:
            Series with mean return values
        """
        returns = self.returns(intraday_only, method)
        return returns.mean()
    
    def sharpe_ratio(self, intraday_only: bool = False, method: str = 'absolute') -> pd.Series:
        """
        Calculate the Sharpe ratio of the time series.
        
        Args:
            intraday_only: Whether to use intraday only returns
            method: Return calculation method
            
        Returns:
            Series with Sharpe ratio values
        """
        returns = self.returns(intraday_only, method)
        return (returns.mean() / returns.std()) * np.sqrt(252)
    
    def max_drawdown(self, intraday_only: bool = False, 
                    method: str = 'absolute') -> pd.Series:
        """
        Calculate the maximum drawdown of the time series.
        
        Args:
            percentage: Whether to use percentage returns or absolute values
            intraday_only: Whether to use intraday only returns
            method: Return calculation method
            
        Returns:
            Series with maximum drawdown values
        """
        if method != 'absolute':
            cum_rets = self.cum_returns(intraday_only, method)
        else:
            cum_rets = self.copy()
        
        # Calculate running maximum
        running_max = cum_rets.expanding().max()
        # Calculate drawdown
        drawdown = cum_rets - running_max
        # Get the maximum drawdown
        return drawdown.min()
    
    def generate_bootstrapped_timeseries(self, simulations: int = 1000, intraday_only: bool = False, 
                                       method: str = 'absolute') -> List['TimeSeries']:
        """
        Generate bootstrapped time series.
        
        Args:
            simulations: Number of simulations to run
            intraday_only: Whether to use intraday only returns
            method: Return calculation method
            
        Returns:
            List of bootstrapped TimeSeries objects
        """
        if self._ts_metadata.is_returns:
            returns = self.copy()
        else:
            returns = self.returns(intraday_only, method)
        
        returns = returns.fillna(0)
        shuffled_timeseries = []
        
        for i in range(simulations):
            shuffled_data = pd.DataFrame(index=returns.index)
            for col in returns.columns:
                shuffled = np.random.permutation(returns[col].values)
                shuffled_data[col] = np.cumsum(shuffled)
            
            ts = TimeSeries(shuffled_data, metadata=None)
            shuffled_timeseries.append(ts)
        
        return shuffled_timeseries
    
    def value_at_risk(self, confidence_level: float = 0.05, 
                     intraday_only: bool = False, method: str = 'absolute') -> pd.Series:
        """
        Calculate the Value at Risk (VaR) of the time series.
        
        Args:
            confidence_level: Confidence level for VaR calculation
            intraday_only: Whether to use intraday only returns
            method: Return calculation method
            
        Returns:
            Series with VaR values
        """
        if method != 'absolute':
            return self.returns(intraday_only, method).quantile(confidence_level)
        else:
            return self.diff().quantile(confidence_level)
    
    def skewness(self, intraday_only: bool = False, method: str = 'absolute') -> pd.Series:
        """
        Calculate the skewness of the time series.
        
        Args:
            intraday_only: Whether to use intraday only returns
            method: Return calculation method
            
        Returns:
            Series with skewness values
        """
        return self.returns(intraday_only, method).skew()
    
    def kurtosis(self, intraday_only: bool = False, method: str = 'absolute') -> pd.Series:
        """
        Calculate the kurtosis of the time series.
        
        Args:
            intraday_only: Whether to use intraday only returns
            method: Return calculation method
            
        Returns:
            Series with kurtosis values
        """
        return self.returns(intraday_only, method).kurt()
    
    def autocorrelation(self, lag: int = 1, intraday_only: bool = False, 
                       method: str = 'absolute') -> pd.Series:
        """
        Calculate the autocorrelation of the time series.
        
        Args:
            lag: Number of periods to lag
            intraday_only: Whether to use intraday only returns
            method: Return calculation method
            
        Returns:
            Series with autocorrelation values
        """
        returns_df = self.returns(intraday_only, method)
        acorr_result = {}
        for col in returns_df.columns:
            acorr_result[col] = returns_df[col].autocorr(lag)
        return pd.Series(acorr_result)
    
    def slice(self, n: Union[int, float], is_percentage: bool = True) -> Tuple['TimeSeries', 'TimeSeries']:
        """
        Slice the time series into two parts.
        
        Args:
            n: Split point (percentage if is_percentage=True, otherwise row count)
            is_percentage: Whether n represents a percentage
            
        Returns:
            Tuple of two TimeSeries objects
        """
        if is_percentage:
            split_idx = int(n * len(self))
        else:
            split_idx = int(n)
        
        #cmon, this type(self) stuff is nice
        if is_percentage:
            return (type(self)(self.iloc[:split_idx]), type(self)(self.iloc[split_idx:]))
        else:
            return (type(self)(self.iloc[:split_idx]), type(self)(self.iloc[split_idx:]))
        
    
    @classmethod
    def read_parquet(cls, path: str) -> 'TimeSeries':
        """
        Read a TimeSeries object from a parquet file.
        """
        return cls(pd.read_parquet(path))
    
    @classmethod
    def read_csv(cls, path: str) -> 'TimeSeries':
        """
        Read a TimeSeries object from a csv file.
        """
        return cls(pd.read_csv(path))
    
    @classmethod
    def read_excel(cls, path: str) -> 'TimeSeries':
        """
        Read a TimeSeries object from a excel file.
        """
        return cls(pd.read_excel(path))
    
    @classmethod
    def read_json(cls, path: str) -> 'TimeSeries':
        """
        Read a TimeSeries object from a json file.
        """
        return cls(pd.read_json(path))
    
    @classmethod
    def read_html(cls, path: str) -> 'TimeSeries':
        """
        Read a TimeSeries object from a html file.
        """
        return cls(pd.read_html(path))