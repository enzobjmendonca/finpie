from typing import Dict, Optional, List, Any, Union
import pandas as pd
from datetime import datetime
import platform
import warnings
import json
from finpie.datasource.sources.base import DataSource
from finpie.datasource.sources.yahoo import YahooFinanceSource
from finpie.datasource.sources.alpha_vantage import AlphaVantageSource
from finpie.datasource.sources.status_invest import StatusInvestSource
from finpie.data import TimeSeries, MultiTimeSeries
from datetime import timedelta

# Optional imports
try:
    from finpie.datasource.sources.mt5 import MT5Source
except ImportError:
    MT5Source = None

class DataService:
    """
    Service class for managing multiple data sources and providing a unified interface
    for accessing financial data.
    """
    def __init__(self):
        """Initialize the data service with an empty source registry."""
        self._sources: Dict[str, DataSource] = {}
        self.subscribed = {}

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
                       interval: str = '1d', subscribed: bool = False) -> TimeSeries:
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
        prices = data_source.get_prices(symbol, start_date, end_date, interval, ['open', 'high', 'low', 'close'])
        if subscribed:
            self.subscribed[(symbol, source, interval, json.dumps(['open', 'high', 'low', 'close']))] = prices
        return prices

    def get_close_prices(self, symbol: str, source: str = 'yahoo_finance', 
                       start_date: Optional[str] = None, end_date: Optional[str] = None,
                       interval: str = '1d', subscribed: bool = False) -> TimeSeries:
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
        if (symbol, source, interval, json.dumps(['close'])) in self.subscribed:
            return self.subscribed[(symbol, source, interval, json.dumps(['close']))]
        
        data_source = self.get_source(source)
        prices = data_source.get_prices(symbol, start_date, end_date, interval, ['close'])
        if subscribed:
            self.subscribed[(symbol, source, interval, json.dumps(['close']))] = prices
        return prices

    def update_subscribed(self, time: datetime):
        for params in self.subscribed.keys():
            start = self.subscribed[params].index[-1]
            end = time
            update_data = self.get_source(params[1]).get_prices(params[0], start, end, params[2], json.loads(params[3]))
            self.subscribed[params].update(update_data)

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
    
    def get_price(self, symbol: str, asof_time: datetime, source: str = 'yahoo_finance'):
        if (symbol, source, '1m', json.dumps(['close'])) in self.subscribed:
            timeseries = self.subscribed[(symbol, source, '1m', json.dumps(['close']))]
            timeseries = timeseries[timeseries.index <= asof_time]
            return timeseries.iloc[-1]['close']
        else:
            timeseries = self.get_close_prices(symbol, source=source, start_date=asof_time - timedelta(days=1), end_date=asof_time, interval='1m', subscribed=True)
            return timeseries.iloc[-1]['close']

    def get_spread(self, symbol: str):
        if 'WIN' in symbol:
            return 5
        else:
            return 0.01

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
        if MT5Source is not None:
            service.register_source(MT5Source())

        return service 