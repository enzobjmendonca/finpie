from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional, List, Dict, Any
import pandas as pd

from finpie.data.timeseries import TimeSeries, TimeSeriesMetadata

class DataSource(ABC):
    """
    Abstract base class for all data sources.
    """
    @abstractmethod
    def get_prices(self, symbol: str, start_date: Optional[str] = None, 
                       end_date: Optional[str] = None, interval: str = '1d', 
                       columns: List[str] = ['close']) -> TimeSeries:
        """
        Get OHLC (Open, High, Low, Close) price data for a symbol.
        
        Args:
            symbol: The symbol to fetch data for
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            interval: Data interval (e.g., '1d' for daily, '1h' for hourly)
            
        Returns:
            TimeSeries: The historical data with metadata
        """
        pass
    
    @abstractmethod
    def get_metadata(self, symbol: str) -> Dict[str, Any]:
        """
        Get metadata for a symbol.
        
        Args:
            symbol: The symbol to fetch metadata for
            
        Returns:
            Dictionary containing symbol metadata
        """
        pass

    @abstractmethod
    def get_available_symbols(self) -> List[str]:
        """
        Get list of available symbols from the data source.

        Returns:
            List[str]: List of available symbols
        """
        pass

    @abstractmethod
    def get_symbol_info(self, symbol: str) -> dict:
        """
        Get information about a specific symbol.

        Args:
            symbol (str): The symbol to get information for

        Returns:
            dict: Information about the symbol
        """
        pass 