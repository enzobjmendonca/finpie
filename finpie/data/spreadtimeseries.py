from typing import Dict, Any, Optional, Union, List
import pandas as pd
import numpy as np

from finpie.data.timeseries import TimeSeries, TimeSeriesMetadata

class SpreadTimeSeries(TimeSeries):
    """A class for handling spread-based time series.
    
    This class extends TimeSeries to provide functionality for analyzing the spread between two time series.
    It's commonly used in spread trading and statistical arbitrage strategies, where the relationship
    between two assets is analyzed through their price spread, often with a hedge ratio.
    
    Attributes:
        series1 (TimeSeries): The first time series
        series2 (TimeSeries): The second time series
        hedge_ratio (float): The hedge ratio used to calculate the spread
        data (pd.DataFrame): The spread time series data
        metadata (TimeSeriesMetadata): Metadata for the spread time series
    """
    
    def __init__(self, series1: TimeSeries, series2: TimeSeries, hedge_ratio: Optional[float] = None):
        """Initialize a SpreadTimeSeries object.
        
        Args:
            series1 (TimeSeries): First TimeSeries object. Can also be a DataFrame or Series.
            series2 (TimeSeries): Second TimeSeries object. Can also be a DataFrame or Series.
            hedge_ratio (Optional[float]): Optional hedge ratio for series2. If None, will be calculated
                using OLS regression. Defaults to None.
                
        Note:
            If series1 or series2 are not TimeSeries objects, they will be converted automatically.
            The hedge ratio is calculated using OLS regression if not provided.
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
        """Calculate the hedge ratio using OLS regression.
        
        Args:
            x (pd.Series): First time series
            y (pd.Series): Second time series
            
        Returns:
            float: Calculated hedge ratio using OLS regression
            
        Note:
            Both series must have the same index. The hedge ratio is calculated as the coefficient
            of the second series in the OLS regression of y on x.
        """        
        # Add constant for regression
        X = pd.concat([pd.Series(1, index=x.index), x], axis=1)
        
        # Calculate hedge ratio using OLS
        beta = np.linalg.inv(X.T @ X) @ X.T @ y
        return beta[1]  # Return the coefficient for series2
    
    def get_hedge_ratio(self) -> float:
        """Get the current hedge ratio.
        
        Returns:
            float: The hedge ratio used to calculate the spread
        """
        return self.hedge_ratio
    
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the SpreadTimeSeries to a dictionary representation.
        
        Returns:
            Dict[str, Any]: Dictionary containing:
                - data: The spread time series data
                - metadata: Metadata for the spread time series
                - series1: Dictionary representation of the first time series
                - series2: Dictionary representation of the second time series
                - hedge_ratio: The hedge ratio used to calculate the spread
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
        """Create a SpreadTimeSeries object from a dictionary representation.
        
        Args:
            data_dict (Dict[str, Any]): Dictionary containing:
                - series1: Dictionary representation of the first time series
                - series2: Dictionary representation of the second time series
                - hedge_ratio: The hedge ratio used to calculate the spread
                
        Returns:
            SpreadTimeSeries: New SpreadTimeSeries object reconstructed from the dictionary
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