from typing import Dict, Optional, List, Any
import pandas as pd
from datetime import datetime
import platform
import warnings

from finpie.datasource.sources.base import DataSource
from finpie.datasource.sources.yahoo import YahooFinanceSource
from finpie.datasource.sources.alpha_vantage import AlphaVantageSource
from finpie.datasource.sources.mt5 import MT5Source
from finpie.data.timeseries import TimeSeries

class DataService:
    """
    Service class for managing multiple data sources and providing a unified interface
    for accessing financial data.
    """
    def __init__(self):
        """Initialize the data service with an empty source registry."""
        self._sources: Dict[str, DataSource] = {}

    def register_source(self, source: DataSource) -> None:
        """
        Register a new data source.

        Args:
            source: The data source to register
        """
        self._sources[source.name] = source

    def get_source(self, name: str) -> DataSource:
        """
        Get a registered data source by name.

        Args:
            name: The name of the data source

        Returns:
            The requested data source

        Raises:
            KeyError: If the source is not found
        """
        if name not in self._sources:
            raise KeyError(f"Data source '{name}' not found")
        return self._sources[name]

    def list_sources(self) -> List[str]:
        """
        Get a list of all registered data source names.

        Returns:
            List of data source names
        """
        return list(self._sources.keys())

    def get_ohlc_prices(self, symbol: str, source: str = 'yahoo_finance', 
                       start_date: Optional[str] = None, end_date: Optional[str] = None,
                       interval: str = '1d') -> TimeSeries:
        """
        Get OHLC price data from a specific source.

        Args:
            symbol: The symbol to fetch data for
            source: The name of the data source to use
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            interval: Data interval (e.g., '1d' for daily, '1h' for hourly)

        Returns:s
            TimeSeries: The historical data with metadata

        Raises:
            KeyError: If the source is not found
        """
        data_source = self.get_source(source)
        return data_source.get_prices(symbol, start_date, end_date, interval, ['open', 'high', 'low', 'close'])

    def get_close_prices(self, symbol: str, source: str = 'yahoo_finance', 
                       start_date: Optional[str] = None, end_date: Optional[str] = None,
                       interval: str = '1d') -> TimeSeries:
        """
        Get close price data from a specific source.

        Args:
            symbol: The symbol to fetch data for
            source: The name of the data source to use
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            interval: Data interval (e.g., '1d' for daily, '1h' for hourly)

        Returns:
            TimeSeries: The historical data with metadata

        Raises:
            KeyError: If the source is not found
        """
        data_source = self.get_source(source)
        return data_source.get_prices(symbol, start_date, end_date, interval, ['close'])

    def get_metadata(self, symbol: str, source: str = 'yahoo_finance') -> Dict[str, Any]:
        """
        Get metadata for a symbol from a specific source.

        Args:
            symbol: The symbol to fetch metadata for
            source: The name of the data source to use

        Returns:
            Dictionary containing symbol metadata

        Raises:
            KeyError: If the source is not found
        """
        data_source = self.get_source(source)
        return data_source.get_metadata(symbol)

    @classmethod
    def create_default_service(cls, alpha_vantage_key: Optional[str] = None) -> 'DataService':
        """
        Create a DataService instance with default data sources.

        Args:
            alpha_vantage_key: Alpha Vantage API key (optional)
            mt5_login: MT5 account login (optional)
            mt5_password: MT5 account password (optional)
            mt5_server: MT5 server name (optional)

        Returns:
            DataService instance with default sources registered
        """
        service = cls()

        # Register Yahoo Finance source (always available)
        service.register_source(YahooFinanceSource())

        # Register Alpha Vantage source if API key is provided
        if alpha_vantage_key:
            service.register_source(AlphaVantageSource(alpha_vantage_key))

        # Register MT5 source if available and credentials are provided
        try:
            mt5_source = MT5Source()
            service.register_source(mt5_source)
        except Exception as e:
            print(f"Warning: Failed to initialize MT5 source: {e}")

        return service 