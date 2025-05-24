from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

from finpie.data.timeseries import TimeSeries, TimeSeriesMetadata

class RatioTimeSeries(TimeSeries):
    """A class for handling ratio-based time series.
    
    This class extends TimeSeries to provide functionality for analyzing the ratio between two time series.
    It's commonly used in pair trading and relative value strategies, where the relationship between
    two assets is analyzed through their price ratio.
    
    Attributes:
        numerator (TimeSeries): The numerator time series
        denominator (TimeSeries): The denominator time series
        data (pd.DataFrame): The ratio time series data
        metadata (TimeSeriesMetadata): Metadata for the ratio time series
    """
    
    def __init__(self, numerator: TimeSeries, denominator: TimeSeries):
        """Initialize a RatioTimeSeries object.
        
        Args:
            numerator (TimeSeries): TimeSeries object for the numerator. Can also be a DataFrame or Series.
            denominator (TimeSeries): TimeSeries object for the denominator. Can also be a DataFrame or Series.
            
        Note:
            If numerator or denominator are not TimeSeries objects, they will be converted automatically.
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
        """Convert the RatioTimeSeries to a dictionary representation.
        
        Returns:
            Dict[str, Any]: Dictionary containing:
                - data: The ratio time series data
                - metadata: Metadata for the ratio time series
                - numerator: Dictionary representation of the numerator time series
                - denominator: Dictionary representation of the denominator time series
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
        """Create a RatioTimeSeries object from a dictionary representation.
        
        Args:
            data_dict (Dict[str, Any]): Dictionary containing:
                - numerator: Dictionary representation of the numerator time series
                - denominator: Dictionary representation of the denominator time series
                
        Returns:
            RatioTimeSeries: New RatioTimeSeries object reconstructed from the dictionary
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