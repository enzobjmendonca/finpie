from typing import Dict, Any, Optional, Union, List
import pandas as pd
import numpy as np

from finpie.data.timeseries import TimeSeries, TimeSeriesMetadata

class SpreadTimeSeries(TimeSeries):
    """
    Class for handling spread-based time series.
    
    This class provides functionality for analyzing the spread between two time series,
    commonly used in spread trading and statistical arbitrage strategies.
    """
    
    def __init__(self, series1: TimeSeries, series2: TimeSeries, hedge_ratio: Optional[float] = None):
        """
        Initialize a SpreadTimeSeries object.
        
        Args:
            series1: First TimeSeries object
            series2: Second TimeSeries object
            hedge_ratio: Optional hedge ratio for series2. If None, will be calculated using OLS regression
        """
        # Validate inputs
        if not isinstance(series1, TimeSeries):
            series1 = TimeSeries(series1)
        if not isinstance(series2, TimeSeries):
            series2 = TimeSeries(series2)
            
        # Align data
        spread_data = series1.data.join(series2.data, how='inner', lsuffix='_series1', rsuffix='_series2')

        # Calculate hedge ratio if not provided
        #TODO: use dynamic hedge ratio to avoid look-ahead bias
        self.hedge_ratio = hedge_ratio
        if hedge_ratio is None:
            self.hedge_ratio = self._calculate_hedge_ratio(spread_data[spread_data.columns[0]], spread_data[spread_data.columns[1]])
        # Calculate spread
        spread_data['spread'] = spread_data['close_series1'] - self.hedge_ratio * spread_data['close_series2']
        
        # Create metadata
        metadata = TimeSeriesMetadata(
            name=f"{series1.metadata.name}-{series2.metadata.name}",
            symbol=f"{series1.metadata.symbol}-{series2.metadata.symbol}",
            source="spread",
            start_date=spread_data.index[0],
            end_date=spread_data.index[-1],
            frequency=series1.metadata.frequency,
            currency=series1.metadata.currency,
            additional_info={
                'series1': series1.metadata.symbol,
                'series2': series2.metadata.symbol,
                'hedge_ratio': self.hedge_ratio,
                'series1_info': series1.metadata.additional_info,
                'series2_info': series2.metadata.additional_info
            }
        )
        
        super().__init__(spread_data['spread'], metadata)
        self.series1 = series1
        self.series2 = series2
    
    def _calculate_hedge_ratio(self, x: pd.Series, y: pd.Series) -> float:
        """
        Calculate the hedge ratio using OLS regression. Both series must have the same index.
        
        Args:
            series1: First pd.Series object
            series2: Second pd.Series object
            
        Returns:
            Calculated hedge ratio
        """        
        # Add constant for regression
        X = pd.concat([pd.Series(1, index=x.index), x], axis=1)
        
        # Calculate hedge ratio using OLS
        beta = np.linalg.inv(X.T @ X) @ X.T @ y
        return beta[1]  # Return the coefficient for series2
    
    def get_hedge_ratio(self) -> float:
        """
        Get the hedge ratio.
        """
        return self.hedge_ratio
    
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the SpreadTimeSeries to a dictionary representation.
        
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
            },
            'series1': self.series1.to_dict(),
            'series2': self.series2.to_dict(),
            'hedge_ratio': self.hedge_ratio
        }
    
    @classmethod
    def from_dict(cls, data_dict: Dict[str, Any]) -> 'SpreadTimeSeries':
        """
        Create a SpreadTimeSeries object from a dictionary representation.
        
        Args:
            data_dict: Dictionary containing time series data and metadata
            
        Returns:
            New SpreadTimeSeries object
        """
        series1 = TimeSeries.from_dict(data_dict['series1'])
        series2 = TimeSeries.from_dict(data_dict['series2'])
        hedge_ratio = data_dict['hedge_ratio']
        return cls(series1, series2, hedge_ratio)
    
    def __repr__(self) -> str:
        """String representation of the SpreadTimeSeries object."""
        return (f"SpreadTimeSeries(spread='{self.metadata.symbol}', "
                f"start_date='{self.start_date}', "
                f"end_date='{self.end_date}', "
                f"frequency='{self.frequency}', "
                f"hedge_ratio={self.hedge_ratio:.4f})") 