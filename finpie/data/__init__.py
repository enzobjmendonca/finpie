"""
Enhanced data module for handling financial data acquisition, processing, and management.

This module provides redesigned time series classes that extend pandas DataFrame
for better integration with the pandas ecosystem while maintaining all
specialized financial time series functionality.

Classes:
    TimeSeries: Base time series class extending pd.DataFrame
    MultiTimeSeries: Multiple time series handling extending TimeSeries  
    RatioTimeSeries: Ratio-based time series extending TimeSeries
    SpreadTimeSeries: Spread-based time series extending TimeSeries
    TimeSeriesMetadata: Metadata container for time series information
"""

from .timeseries import TimeSeries, TimeSeriesMetadata
from .multitimeseries import MultiTimeSeries
from .ratiotimeseries import RatioTimeSeries
from .spreadtimeseries import SpreadTimeSeries

__all__ = [
    'TimeSeries',
    'TimeSeriesMetadata', 
    'MultiTimeSeries',
    'RatioTimeSeries',
    'SpreadTimeSeries'
]

__version__ = '2.0.0'
__author__ = 'FinPie Development Team'
__description__ = 'Enhanced financial time series data structures extending pandas DataFrame'
