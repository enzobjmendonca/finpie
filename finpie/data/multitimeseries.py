from typing import List, Dict, Any, Optional, Union
import pandas as pd
import numpy as np

from finpie.data.timeseries import TimeSeries, TimeSeriesMetadata

class MultiTimeSeries(TimeSeries):
    """
    Class for handling multiple time series together.
    
    This class provides functionality for analyzing multiple time series
    simultaneously, including correlation analysis, portfolio construction,
    and risk metrics.
    """
    
    def __init__(self, timeseries: Union[List[TimeSeries], List[pd.DataFrame], List[pd.Series], pd.DataFrame]):
        """
        Initialize a MultiTimeSeries object.
        
        Args:
            timeseries: List of TimeSeries objects to combine
        """
        # Handle different input types
        if isinstance(timeseries, pd.DataFrame):
            # Single DataFrame - split into list of TimeSeries by columns
            self.data = timeseries
            self.timeseries = [TimeSeries(timeseries[[col]], None) for col in timeseries.columns]
        elif isinstance(timeseries, list):
            if not timeseries:
                raise ValueError("At least one TimeSeries must be provided")
            if all(isinstance(ts, TimeSeries) for ts in timeseries):
                # List of TimeSeries objects
                self.timeseries = timeseries
            elif all(isinstance(ts, pd.DataFrame) for ts in timeseries) or all(isinstance(ts, pd.Series) for ts in timeseries):
                # List of DataFrames
                self.timeseries = [TimeSeries(ts, None) for ts in timeseries]
            else:
                raise ValueError("All elements in list must be either TimeSeries objects or pandas DataFrames")
            self._align_series()
        else:
            raise ValueError("Input must be either a pandas DataFrame or a list of TimeSeries/DataFrame objects")
        
        # Create combined metadata
        self.metadata = self._create_metadata()

    def _align_series(self) -> None:
        """Align all time series to a common index."""

        aligned_df = self.timeseries[0].data
        col_index = 0   
        for ts in self.timeseries[1:]:
            aligned_df = aligned_df.merge(ts.data, left_index=True, right_index=True, suffixes=('', '_' + str(col_index)))
            col_index += 1
        
        # Combine all aligned DataFrames
        self.data = aligned_df
        
    
    def _create_metadata(self) -> TimeSeriesMetadata:
        """Create metadata for the combined time series."""
        return TimeSeriesMetadata(
            symbol=",".join(ts.metadata.symbol for ts in self.timeseries if ts.metadata != None and ts.metadata.symbol != None),
            name=",".join(ts.metadata.name for ts in self.timeseries if ts.metadata != None and ts.metadata.name != None),
            source="combined",
            start_date=self.data.index[0],
            end_date=self.data.index[-1],
            frequency=self.timeseries[0].metadata.frequency if self.timeseries[0].metadata != None else None,
            currency=self.timeseries[0].metadata.currency if self.timeseries[0].metadata != None else None,
            additional_info={
                'num_series': len(self.timeseries),
                'symbols': [ts.metadata.symbol for ts in self.timeseries if ts.metadata != None and ts.metadata.symbol != None],
                'sources': [ts.metadata.source for ts in self.timeseries if ts.metadata != None and ts.metadata.source != None]
            }
        )
    
    def correlation(self, returns: bool = True, method: str = 'pearson', min_periods: Optional[int] = None) -> pd.DataFrame:
        """
        Calculate correlation matrix between time series.
        
        Args:
            returns: Whether to use returns or original series
            method: Correlation method ('pearson', 'kendall', or 'spearman')
            min_periods: Minimum number of observations required
            
        Returns:
            DataFrame containing correlation matrix
        """
        if returns:
            if self.metadata.is_returns:
                logger.warning("Time series is already returns, be aware that the correlation will be calculated on the series returns.\
                                If you want to calculate the correlation on the original series, set returns to False.")
            return self.returns().data.corr(method=method, min_periods=min_periods)
        else:
            return self.data.corr(method=method, min_periods=min_periods)
    
    def rolling(self, window: int, stats: List[str] = ['mean', 'std', 'min', 'max', 'skew', 'kurt', 'sum', 'count', 'median', 'var', 'quantiles'], min_periods: Optional[int] = None) -> pd.DataFrame:
        """
        Calculate rolling statistics for the time series.
        
        Args:
            window: Size of the rolling window
            stats: List of statistics to calculate available stats: ['mean', 'std', 'min', 'max', 'skew', 'kurt', 'sum', 'count', 'median', 'var', 'quantiles']
            min_periods: Minimum number of observations required
            
        Returns:
            DataFrame containing rolling statistics
        """
        rolling_data = []
        for ts in self.timeseries:
            rolling_data.append(ts.rolling(window, stats, min_periods))
        
        aligned_df = rolling_data[0]
        col_index = 0   
        for rolling_df in rolling_data[1:]:
            aligned_df = aligned_df.merge(rolling_df, left_index=True, right_index=True, suffixes=('', '_' + str(col_index)))
            col_index += 1
        
        # Combine all aligned DataFrames
        return aligned_df

    def covariance(self, returns: bool = True, min_periods: Optional[int] = None) -> pd.DataFrame:
        """
        Calculate covariance matrix between time series.
        
        Args:
            returns: Whether to use returns or original series
            min_periods: Minimum number of observations required
            
        Returns:
            DataFrame containing covariance matrix
        """
        if returns:
            if self.metadata.is_returns:
                logger.warning("Time series is already returns, be aware that the covariance will be calculated on the series returns.\
                                If you want to calculate the covariance on the original series, set returns to False.")
            return self.returns().data.cov(min_periods=min_periods)
        else:
            return self.data.cov(min_periods=min_periods)
    
    def returns(self, intraday_only: bool = False, method: str = 'simple') -> 'MultiTimeSeries':
        """
        Calculate returns for all time series.
        
        Args:
            method: Return calculation method ('log' or 'simple')
            
        Returns:
            New MultiTimeSeries object with returns data
        """
        if self.metadata.is_returns:
            logger.warning("Time series is already returns, be aware that the returns will be calculated on the series returns.\
                            If you want to calculate the returns on the original series, use this object directly.")
        returns_series = []
        for ts in self.timeseries:
            returns_series.append(ts.returns(method=method))
        return MultiTimeSeries(returns_series)
    
    def portfolio(self, weights: Dict[str, float], percentage: bool = False, 
                 intraday_only: bool = False, method: str = 'simple', shares: bool = False) -> pd.DataFrame:
        """
        Calculate portfolio returns using given weights.
        
        Args:
            weights: Dictionary mapping symbols to weights
            percentage: Whether to use percentage returns
            intraday_only: Whether to use intraday only returns
            method: Return calculation method ('log' or 'simple')
            shares: Whether to weights as number of shares instead of percentage
        Returns:
            pd.DataFrame object containing portfolio returns
        """
        # Validate weights
        if not all(symbol in self.data.columns for symbol in weights.keys()):
            raise ValueError("All symbols in weights must be present in the time series")
            
        if not shares and not np.isclose(sum(weights.values()), 1.0):
            raise ValueError("Weights must sum to 1.0")
            
        # Calculate portfolio returns
        if percentage:
            data = self.returns(intraday_only, method).data
        else:
            data = self.data
        portfolio_returns = pd.Series(0.0, index=data.index)
        
        total_weight = 0
        for symbol, weight in weights.items():
            total_weight += weight
            portfolio_returns += weight * data[symbol]
        if shares:
            portfolio_returns = portfolio_returns / total_weight

        # Create portfolio time series
        portfolio_data = pd.DataFrame({'returns': portfolio_returns})
        portfolio_metadata = TimeSeriesMetadata(
            name="portfolio",
            symbol="portfolio",
            source="combined",
            start_date=portfolio_data.index[0],
            end_date=portfolio_data.index[-1],
            frequency=self.metadata.frequency,
            currency=self.metadata.currency,
            additional_info={
                'weights': weights,
                'constituents': list(weights.keys())
            }
        )
        
        return TimeSeries(portfolio_data, portfolio_metadata)
    
    def rolling_correlation(self, window: int, min_periods: Optional[int] = None) -> Dict[str, pd.DataFrame]:
        """
        Calculate rolling correlation matrices.
        
        Args:
            window: Size of the rolling window
            min_periods: Minimum number of observations required
            
        Returns:
            Dictionary mapping dates to correlation matrices
        """
        if min_periods is None:
            min_periods = window
            
        return self.data.rolling(window, min_periods=min_periods).corr()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the MultiTimeSeries to a dictionary representation.
        
        Returns:
            Dictionary containing the time series data and metadata
        """
        return {
            'timeseries': [ts.to_dict() for ts in self.timeseries],
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
    def from_dict(cls, data_dict: Dict[str, Any]) -> 'MultiTimeSeries':
        """
        Create a MultiTimeSeries object from a dictionary representation.
        
        Args:
            data_dict: Dictionary containing time series data and metadata
            
        Returns:
            New MultiTimeSeries object
        """
        timeseries = [TimeSeries.from_dict(ts_dict) for ts_dict in data_dict['timeseries']]
        return cls(timeseries)
    
    def __repr__(self) -> str:
        """String representation of the MultiTimeSeries object."""
        return (f"MultiTimeSeries(symbols='{self.metadata.symbol}', "
                f"start_date='{self.metadata.start_date}', "
                f"end_date='{self.metadata.end_date}', "
                f"frequency='{self.metadata.frequency}')") 