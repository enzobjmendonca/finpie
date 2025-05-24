from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

from finpie.data.timeseries import TimeSeries, TimeSeriesMetadata

class RatioTimeSeries(TimeSeries):
    """
    Class for handling ratio-based time series.
    
    This class provides functionality for analyzing the ratio between two time series,
    commonly used in pair trading and relative value strategies.
    """
    
    def __init__(self, numerator: TimeSeries, denominator: TimeSeries):
        """
        Initialize a RatioTimeSeries object.
        
        Args:
            numerator: TimeSeries object for the numerator
            denominator: TimeSeries object for the denominator
        """
        # Validate inputs
        if not isinstance(numerator, TimeSeries):
            numerator = TimeSeries(numerator)
        if not isinstance(denominator, TimeSeries):
            denominator = TimeSeries(denominator)
            
        # Calculate ratio
        ratio_data = numerator.data.join(denominator.data, how='inner', lsuffix='_numerator', rsuffix='_denominator')
        ratio_data['ratio'] = ratio_data[ratio_data.columns[0]] / ratio_data[ratio_data.columns[1]]
        
        # Create metadata
        metadata = TimeSeriesMetadata(
            name=f"{numerator.metadata.symbol} ({numerator.metadata.name})/{denominator.metadata.symbol} ({denominator.metadata.name})",
            symbol=f"{numerator.metadata.symbol}/{denominator.metadata.symbol}",
            source="ratio",
            start_date=ratio_data.index[0],
            end_date=ratio_data.index[-1],
            frequency=numerator.metadata.frequency,
            currency=numerator.metadata.currency,
            additional_info={
                'numerator': numerator.metadata.symbol,
                'denominator': denominator.metadata.symbol,
                'numerator_info': numerator.metadata.additional_info,
                'denominator_info': denominator.metadata.additional_info
            }
        )
        
        super().__init__(ratio_data['ratio'], metadata)
        self.numerator = numerator
        self.denominator = denominator
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the RatioTimeSeries to a dictionary representation.
        
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
            },
            'numerator': self.numerator.to_dict(),
            'denominator': self.denominator.to_dict()
        }
    
    @classmethod
    def from_dict(cls, data_dict: Dict[str, Any]) -> 'RatioTimeSeries':
        """
        Create a RatioTimeSeries object from a dictionary representation.
        
        Args:
            data_dict: Dictionary containing time series data and metadata
            
        Returns:
            New RatioTimeSeries object
        """
        numerator = TimeSeries.from_dict(data_dict['numerator'])
        denominator = TimeSeries.from_dict(data_dict['denominator'])
        return cls(numerator, denominator)
    
    def __repr__(self) -> str:
        """String representation of the RatioTimeSeries object."""
        return (f"RatioTimeSeries(ratio='{self.metadata.symbol}', "
                f"start_date='{self.start_date}', "
                f"end_date='{self.end_date}', "
                f"frequency='{self.frequency}')") 