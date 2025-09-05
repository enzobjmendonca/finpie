from typing import List, Dict, Any, Optional, Union, Tuple
import pandas as pd
import numpy as np
import logging

try:
    from .timeseries import TimeSeries, TimeSeriesMetadata
except ImportError:
    from timeseries import TimeSeries, TimeSeriesMetadata

# Configure logger
logger = logging.getLogger(__name__)

class MultiTimeSeries(TimeSeries):
    """A class for handling multiple time series together.
    
    This class extends TimeSeries to provide functionality for analyzing multiple time series
    simultaneously. It supports operations like correlation analysis, portfolio construction,
    and risk metrics across multiple assets.
    
    Since it extends TimeSeries (which extends pd.DataFrame), it inherits all pandas DataFrame
    functionality while adding specialized multi-series operations.
    """
    _metadata = ['_individual_timeseries']
    
    def __init__(self, timeseries: Union[List[TimeSeries], List[pd.DataFrame], List[pd.Series], pd.DataFrame], 
                 is_returns: bool = False, metadata: TimeSeriesMetadata = None, **kwargs):
        """Initialize a MultiTimeSeries object.
        
        Args:
            timeseries: Can be one of:
                - List of TimeSeries objects
                - List of pandas DataFrames
                - List of pandas Series
                - Single DataFrame with multiple columns
            is_returns: Whether the input data represents returns
            metadata: Optional metadata for the combined series
                
        Raises:
            ValueError: If input is invalid or empty
        """
        # Store individual timeseries list temporarily
        _individual_timeseries = None
        
        # Handle different input types
        if isinstance(timeseries, pd.DataFrame): 
            # Single DataFrame - treat as multi-column time series
            combined_data = timeseries
        
        elif isinstance(timeseries, list):
            _individual_timeseries = []
            if not timeseries:
                raise ValueError("At least one TimeSeries must be provided")
            
            if all(isinstance(ts, TimeSeries) for ts in timeseries):
                # List of TimeSeries objects
                _individual_timeseries = timeseries
            elif all(isinstance(ts, (pd.DataFrame, pd.Series)) for ts in timeseries):
                # List of DataFrames or Series
                for i, ts in enumerate(timeseries):
                    if isinstance(ts, pd.Series):
                        ts_name = ts.name if ts.name else f'series_{i}'
                        ts = ts.to_frame()
                    else:
                        ts_name = ts.columns[0] if len(ts.columns) > 0 else f'series_{i}'
                    
                    ts_metadata = TimeSeriesMetadata(
                        name=ts_name,
                        symbol=ts_name,
                        source='unknown',
                        start_date=ts.index[0] if len(ts.index) > 0 else None,
                        end_date=ts.index[-1] if len(ts.index) > 0 else None,
                        frequency='',
                        currency='',
                        additional_info={},
                        is_returns=is_returns
                    )
                    _individual_timeseries.append(TimeSeries(ts, ts_metadata))
            else:
                raise ValueError("All elements in list must be either TimeSeries objects or pandas DataFrames/Series")
            
            # Align and combine all series
            combined_data = self._align_series_static(_individual_timeseries, is_returns)
        else:
            raise ValueError("Input must be either a pandas DataFrame or a list of TimeSeries/DataFrame objects")
        
        # Create combined metadata
        if metadata is None:
            metadata = self._create_metadata_static(combined_data)
        
        # Initialize parent TimeSeries with combined data
        super().__init__(combined_data, metadata, **kwargs)
        
        self._individual_timeseries = _individual_timeseries
        
    def _align_series(self, is_returns: bool = False) -> pd.DataFrame:
        """Align all time series to a common index."""
        if not self._individual_timeseries:
            return pd.DataFrame()
        
        # Start with the first time series
        aligned_df = self._individual_timeseries[0].copy()
        
        # Rename column to avoid conflicts
        if len(aligned_df.columns) == 1:
            old_name = aligned_df.columns[0]
            new_name = self._individual_timeseries[0].metadata.symbol or f'series_0'
            aligned_df = aligned_df.rename(columns={old_name: new_name})
        
        # Merge with subsequent time series
        for i, ts in enumerate(self._individual_timeseries[1:], 1):
            ts_data = ts.copy()
            
            # Rename column to match symbol or create unique name
            if len(ts_data.columns) == 1:
                old_name = ts_data.columns[0]
                new_name = ts.metadata.symbol or f'series_{i}'
                ts_data = ts_data.rename(columns={old_name: new_name})
            
            aligned_df = aligned_df.merge(
                ts_data, 
                left_index=True, 
                right_index=True, 
                how='outer',
                suffixes=('', f'_{i}')
            )
        
        # Handle missing values
        if not is_returns:
            aligned_df = aligned_df.ffill()
        aligned_df = aligned_df.fillna(0)
        
        return aligned_df
    
    @staticmethod
    def _align_series_static(individual_timeseries: List[TimeSeries], is_returns: bool = False) -> pd.DataFrame:
        """Align all time series to a common index."""
        if not individual_timeseries:
            return pd.DataFrame()
        
        # Start with the first time series
        aligned_df = pd.DataFrame(individual_timeseries[0].copy())
        
        # Rename column to avoid conflicts
        if len(aligned_df.columns) == 1:
            old_name = aligned_df.columns[0]
            new_name = individual_timeseries[0].metadata.symbol or f'series_0'
            aligned_df = aligned_df.rename(columns={old_name: new_name})
        
        # Merge with subsequent time series
        for i, ts in enumerate(individual_timeseries[1:], 1):
            ts_data = ts.copy()
            
            # Rename column to match symbol or create unique name
            if len(ts_data.columns) == 1:
                old_name = ts_data.columns[0]
                new_name = ts.metadata.symbol or f'series_{i}'
                ts_data = ts_data.rename(columns={old_name: new_name})
            
            aligned_df = aligned_df.merge(
                ts_data, 
                left_index=True, 
                right_index=True, 
                how='outer',
                suffixes=('', f'_{i}')
            )
        
        # Handle missing values
        if not is_returns:
            aligned_df = aligned_df.ffill()
        aligned_df = aligned_df.fillna(0)
        
        return aligned_df
    
    @property
    def individual_timeseries(self) -> List[TimeSeries]:
        if self._individual_timeseries is None:
            self._individual_timeseries = [TimeSeries(self[col]) for col in self.columns]
        return self._individual_timeseries
    
    def _create_metadata(self) -> TimeSeriesMetadata:
        """Create metadata for the combined time series."""
        symbols = []
        names = []
        sources = []
        
        for ts in self._individual_timeseries:
            if ts.metadata:
                if ts.metadata.symbol:
                    symbols.append(ts.metadata.symbol)
                if ts.metadata.name:
                    names.append(ts.metadata.name)
                if ts.metadata.source:
                    sources.append(ts.metadata.source)
        
        return TimeSeriesMetadata(
            symbol=",".join(symbols) if symbols else "",
            name=",".join(names) if names else "",
            source="combined",
            start_date=self.index[0] if len(self.index) > 0 else None,
            end_date=self.index[-1] if len(self.index) > 0 else None,
            frequency=self._individual_timeseries[0].metadata.frequency if self._individual_timeseries and self._individual_timeseries[0].metadata else None,
            currency=self._individual_timeseries[0].metadata.currency if self._individual_timeseries and self._individual_timeseries[0].metadata else None,
            additional_info={
                'num_series': len(self._individual_timeseries),
                'symbols': symbols,
                'sources': list(set(sources))
            }
        )
    
    @staticmethod
    def _create_metadata_static(combined_data: pd.DataFrame) -> TimeSeriesMetadata:
        """Create metadata for the combined time series."""        
        return TimeSeriesMetadata(
            symbol=",".join(combined_data.columns),
            name=",".join(combined_data.columns),
            source="combined",
            start_date=combined_data.index[0] if len(combined_data.index) > 0 else None,
            end_date=combined_data.index[-1] if len(combined_data.index) > 0 else None,
            frequency='',
            currency='',
            additional_info={}
        )
    
    def correlation(self, returns: bool = True, method: str = 'pearson', 
                   min_periods: Optional[int] = None) -> pd.DataFrame:
        """Calculate correlation matrix between time series.
        
        Args:
            returns (bool): Whether to use returns or original series. Defaults to True.
            method (str): Correlation method ('pearson', 'kendall', or 'spearman'). Defaults to 'pearson'.
            min_periods (Optional[int]): Minimum number of observations required. Defaults to None.
            
        Returns:
            pd.DataFrame: Correlation matrix between all time series.
            
        Note:
            If returns=True and the series is already returns, a warning will be logged.
        """
        if returns:
            if self.metadata.is_returns:
                logger.warning("Time series is already returns, calculating correlation on returns data. "
                             "If you want correlation on original series, set returns=False.")
            # Calculate returns using pct_change directly on the DataFrame
            data_to_use = self.pct_change()
        else:
            data_to_use = self
        
        return data_to_use.corr(method=method, min_periods=min_periods)
    
    def covariance(self, returns: bool = True, min_periods: Optional[int] = None) -> pd.DataFrame:
        """Calculate covariance matrix between time series.
        
        Args:
            returns (bool): Whether to use returns or original series. Defaults to True.
            min_periods (Optional[int]): Minimum number of observations required. Defaults to None.
            
        Returns:
            pd.DataFrame: Covariance matrix between all time series.
            
        Note:
            If returns=True and the series is already returns, a warning will be logged.
        """
        if returns:
            if self.metadata.is_returns:
                logger.warning("Time series is already returns, calculating covariance on returns data. "
                             "If you want covariance on original series, set returns=False.")
            # Calculate returns using pct_change directly on the DataFrame
            data_to_use = self.pct_change()
        else:
            data_to_use = self
        
        return data_to_use.cov(min_periods=min_periods)
    
    def returns(self, intraday_only: bool = False, method: str = 'absolute') -> 'MultiTimeSeries':
        """
        Calculate returns for all time series.
        
        Args:
            intraday_only: Whether to drop the first record of each day
            method: Return calculation method ('log' or 'simple')
            
        Returns:
            New MultiTimeSeries object with returns data
        """
        if self.metadata.is_returns:
            logger.warning("Time series is already returns, calculating returns on returns data. "
                         "If you want to use the original data, create a new MultiTimeSeries.")
        
        # Calculate returns for each individual TimeSeries
        returns_df = self.pct_change() if method == 'simple' else self.diff() if method == 'absolute' else self.log_returns() if method == 'log' else None
        returns_df.fillna(0, inplace=True)
        # Create new MultiTimeSeries with returns data
        return MultiTimeSeries(returns_df, is_returns=True)
    
    def portfolio(self, weights: Dict[str, float], 
                 intraday_only: bool = False, method: str = 'absolute', 
                 shares: bool = False, portfolio_name: str = 'portfolio') -> TimeSeries:
        """Calculate portfolio returns using given weights.
        
        Args:
            weights (Dict[str, float]): Dictionary mapping symbols to weights
            percentage (bool): Whether to use percentage returns. Defaults to False.
            intraday_only (bool): Whether to use intraday only returns. Defaults to False.
            method (str): Return calculation method ('log' or 'simple'). Defaults to 'simple'.
            shares (bool): Whether to use weights as number of shares instead of percentage. Defaults to False.
            portfolio_name (str): Name for the portfolio. Defaults to 'portfolio'.
            
        Returns:
            TimeSeries: Portfolio returns time series.
            
        Raises:
            ValueError: If weights don't sum to 1.0 (when shares=False) or if symbols are not found.
        """
        # Validate weights
        missing_symbols = [symbol for symbol in weights.keys() if symbol not in self.columns]
        if missing_symbols:
            raise ValueError(f"Symbols not found in time series: {missing_symbols}")
            
        if not shares and not np.isclose(sum(weights.values()), 1.0):
            raise ValueError("Weights must sum to 1.0 when shares=False")
        
        # Use returns or original data
        if method != 'absolute':
            data_to_use = self.returns(intraday_only, method)
        else:
            data_to_use = self
        
        # Calculate portfolio values
        portfolio_values = pd.Series(0.0, index=data_to_use.index)
        
        for symbol, weight in weights.items():
            if symbol in data_to_use.columns:
                portfolio_values += weight * data_to_use[symbol]
        
        # Create portfolio DataFrame
        portfolio_data = pd.DataFrame({portfolio_name: portfolio_values})
        
        # Create portfolio metadata
        portfolio_metadata = TimeSeriesMetadata(
            name=portfolio_name,
            symbol=portfolio_name,
            source="portfolio",
            start_date=portfolio_data.index[0],
            end_date=portfolio_data.index[-1],
            frequency=self.metadata.frequency,
            currency=self.metadata.currency,
            additional_info={
                'is_shares': shares,
                'weights': weights,
                'constituents': list(weights.keys()),
                'calculation_method': method if percentage else 'absolute'
            }
        )
        
        return TimeSeries(portfolio_data, portfolio_metadata)
    
    def rolling_correlation(self, window: int, min_periods: Optional[int] = None) -> pd.DataFrame:
        """
        Calculate rolling correlation matrices.
        
        Args:
            window: Size of the rolling window
            min_periods: Minimum number of observations required
            
        Returns:
            DataFrame with rolling correlation data
        """
        if min_periods is None:
            min_periods = window
        
        return self.rolling(window, min_periods=min_periods).corr()
    
    def get_individual_series(self, symbol: str) -> TimeSeries:
        """
        Get an individual time series by symbol.
        
        Args:
            symbol: Symbol of the time series to retrieve
            
        Returns:
            TimeSeries object for the specified symbol
            
        Raises:
            ValueError: If symbol is not found
        """
        for ts in self._individual_timeseries:
            if ts.metadata and ts.metadata.symbol == symbol:
                return ts
        
        # If not found in individual series, try to extract from combined data
        if symbol in self.columns:
            ts_data = self[[symbol]]
            ts_metadata = TimeSeriesMetadata(
                name=symbol,
                symbol=symbol,
                source=self.metadata.source,
                start_date=self.start_date,
                end_date=self.end_date,
                frequency=self.metadata.frequency,
                currency=self.metadata.currency,
                additional_info={'extracted_from_multi': True}
            )
            return TimeSeries(ts_data, ts_metadata)
        
        raise ValueError(f"Symbol '{symbol}' not found in MultiTimeSeries")
    
    def add_series(self, new_series: Union[TimeSeries, pd.DataFrame, pd.Series], symbol: str = None) -> 'MultiTimeSeries':
        """
        Add a new time series to the MultiTimeSeries.
        
        Args:
            new_series: TimeSeries, DataFrame, or Series to add
            symbol: Symbol name for the new series (if not TimeSeries)
            
        Returns:
            New MultiTimeSeries object with the added series
        """
        if not isinstance(new_series, TimeSeries):
            if isinstance(new_series, pd.Series):
                new_series = new_series.to_frame()
            
            ts_name = symbol or (new_series.columns[0] if len(new_series.columns) > 0 else 'new_series')
            ts_metadata = TimeSeriesMetadata(
                name=ts_name,
                symbol=ts_name,
                source='added',
                start_date=new_series.index[0] if len(new_series.index) > 0 else None,
                end_date=new_series.index[-1] if len(new_series.index) > 0 else None,
                frequency='',
                currency='',
                additional_info={}
            )
            new_series = TimeSeries(new_series, ts_metadata)
        
        # Create new MultiTimeSeries with added series
        new_timeseries_list = self._individual_timeseries + [new_series]
        return MultiTimeSeries(new_timeseries_list)
    
    def remove_series(self, symbol: str) -> 'MultiTimeSeries':
        """
        Remove a time series by symbol.
        
        Args:
            symbol: Symbol of the time series to remove
            
        Returns:
            New MultiTimeSeries object without the specified series
            
        Raises:
            ValueError: If symbol is not found
        """
        # Remove from individual timeseries
        original_length = len(self._individual_timeseries)
        new_timeseries_list = [
            ts for ts in self._individual_timeseries 
            if not (ts.metadata and ts.metadata.symbol == symbol)
        ]
        
        if len(new_timeseries_list) == original_length:
            raise ValueError(f"Symbol '{symbol}' not found in MultiTimeSeries")
        
        # Create new MultiTimeSeries without the removed series
        if new_timeseries_list:
            return MultiTimeSeries(new_timeseries_list)
        else:
            # Return empty MultiTimeSeries
            return MultiTimeSeries(pd.DataFrame())
    
    def to_dict_ts(self) -> Dict[str, Any]:
        """
        Convert the MultiTimeSeries to a dictionary representation.
        
        Returns:
            Dictionary containing the time series data and metadata
        """
        return {
            'individual_timeseries': [ts.to_dict_ts() for ts in self._individual_timeseries],
            'combined_data': self.to_dict(),
            'metadata': {
                'symbol': self.metadata.symbol,
                'name': self.metadata.name,
                'source': self.metadata.source,
                'start_date': self.metadata.start_date.isoformat() if self.metadata.start_date else None,
                'end_date': self.metadata.end_date.isoformat() if self.metadata.end_date else None,
                'frequency': self.metadata.frequency,
                'currency': self.metadata.currency,
                'additional_info': self.metadata.additional_info,
                'is_returns': self.metadata.is_returns
            }
        }
    
    @classmethod
    def from_dict_ts(cls, data_dict: Dict[str, Any]) -> 'MultiTimeSeries':
        """
        Create a MultiTimeSeries object from a dictionary representation.
        
        Args:
            data_dict: Dictionary containing time series data and metadata
            
        Returns:
            New MultiTimeSeries object
        """
        individual_timeseries = [
            TimeSeries.from_dict_ts(ts_dict) 
            for ts_dict in data_dict['individual_timeseries']
        ]
        return cls(individual_timeseries)
    
    def __repr__(self) -> str:
        """String representation of the MultiTimeSeries object."""
        df_repr = pd.DataFrame.__repr__(self)
        if self.metadata:
            metadata_info = (f"\nMultiTimeSeries Metadata:\n"
                           f"Symbols: {self.metadata.symbol}\n"
                           f"Source: {self.metadata.source}\n"
                           f"Frequency: {self.metadata.frequency}\n"
                           f"Number of series: {self.metadata.additional_info.get('num_series', 'Unknown')}\n"
                           f"Is Returns: {self.metadata.is_returns}")
            return df_repr + metadata_info
        return df_repr
