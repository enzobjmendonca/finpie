from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np

from .timeseries import TimeSeries, TimeSeriesMetadata

class MultiTimeSeries:
    """
    Class for handling multiple time series together.
    
    This class provides functionality for analyzing multiple time series
    simultaneously, including correlation analysis, portfolio construction,
    and risk metrics.
    """
    
    def __init__(self, timeseries: List[TimeSeries]):
        """
        Initialize a MultiTimeSeries object.
        
        Args:
            timeseries: List of TimeSeries objects to combine
        """
        if not timeseries:
            raise ValueError("At least one TimeSeries must be provided")
            
        # Store original time series
        self.timeseries = timeseries
        
        # Align all time series to common index
        self._align_series()
        
        # Create combined metadata
        self.metadata = self._create_metadata()
    
    def _align_series(self) -> None:
        """Align all time series to a common index."""
        # Get common index
        common_index = self.timeseries[0].data.index
        for ts in self.timeseries[1:]:
            common_index = common_index.intersection(ts.data.index)
            
        if len(common_index) == 0:
            raise ValueError("No common dates found between time series")
            
        # Align all series to common index
        self.aligned_data = pd.DataFrame(index=common_index)
        for ts in self.timeseries:
            self.aligned_data[ts.metadata.symbol] = ts.data['close']
    
    def _create_metadata(self) -> TimeSeriesMetadata:
        """Create metadata for the combined time series."""
        return TimeSeriesMetadata(
            symbol=",".join(ts.metadata.symbol for ts in self.timeseries),
            source="combined",
            start_date=self.aligned_data.index[0],
            end_date=self.aligned_data.index[-1],
            frequency=self.timeseries[0].metadata.frequency,
            currency=self.timeseries[0].metadata.currency,
            additional_info={
                'num_series': len(self.timeseries),
                'symbols': [ts.metadata.symbol for ts in self.timeseries],
                'sources': [ts.metadata.source for ts in self.timeseries]
            }
        )
    
    def correlation(self, method: str = 'pearson', min_periods: Optional[int] = None) -> pd.DataFrame:
        """
        Calculate correlation matrix between time series.
        
        Args:
            method: Correlation method ('pearson', 'kendall', or 'spearman')
            min_periods: Minimum number of observations required
            
        Returns:
            DataFrame containing correlation matrix
        """
        return self.aligned_data.corr(method=method, min_periods=min_periods)
    
    def covariance(self, min_periods: Optional[int] = None) -> pd.DataFrame:
        """
        Calculate covariance matrix between time series.
        
        Args:
            min_periods: Minimum number of observations required
            
        Returns:
            DataFrame containing covariance matrix
        """
        return self.aligned_data.cov(min_periods=min_periods)
    
    def returns(self, method: str = 'log') -> 'MultiTimeSeries':
        """
        Calculate returns for all time series.
        
        Args:
            method: Return calculation method ('log' or 'simple')
            
        Returns:
            New MultiTimeSeries object with returns data
        """
        returns_series = []
        for ts in self.timeseries:
            returns_series.append(ts.returns(method=method))
        return MultiTimeSeries(returns_series)
    
    def portfolio_returns(self, weights: Dict[str, float]) -> TimeSeries:
        """
        Calculate portfolio returns using given weights.
        
        Args:
            weights: Dictionary mapping symbols to weights
            
        Returns:
            TimeSeries object containing portfolio returns
        """
        # Validate weights
        if not all(symbol in self.aligned_data.columns for symbol in weights.keys()):
            raise ValueError("All symbols in weights must be present in the time series")
            
        if not np.isclose(sum(weights.values()), 1.0):
            raise ValueError("Weights must sum to 1.0")
            
        # Calculate portfolio returns
        returns = self.returns()
        portfolio_returns = pd.Series(0.0, index=returns.aligned_data.index)
        
        for symbol, weight in weights.items():
            portfolio_returns += weight * returns.aligned_data[symbol]
            
        # Create portfolio time series
        portfolio_data = pd.DataFrame({'returns': portfolio_returns})
        portfolio_metadata = TimeSeriesMetadata(
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
            
        correlations = {}
        for i in range(len(self.aligned_data) - window + 1):
            window_data = self.aligned_data.iloc[i:i+window]
            if len(window_data) >= min_periods:
                date = window_data.index[-1]
                correlations[date] = window_data.corr()
                
        return correlations
    
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