from datetime import datetime
from typing import List, Optional, Dict, Any
import yfinance as yf
import pandas as pd

from finpie.datasource.sources.base import DataSource
from finpie.data.timeseries import TimeSeries, TimeSeriesMetadata

class YahooFinanceSource(DataSource):
    """
    Yahoo Finance data source implementation.
    """
    def __init__(self):
        """Initialize the Yahoo Finance data source."""
        self.name = "yahoo_finance"
        self._ticker_cache = {}

    def _get_ticker(self, symbol: str) -> yf.Ticker:
        """
        Get or create a Ticker object for the given symbol.

        Args:
            symbol (str): The symbol to get a Ticker for

        Returns:
            yf.Ticker: The Ticker object
        """
        if symbol not in self._ticker_cache:
            self._ticker_cache[symbol] = yf.Ticker(symbol)
        return self._ticker_cache[symbol]

    def get_prices(self, symbol: str, start_date: Optional[str] = None, 
                       end_date: Optional[str] = None, interval: str = '1d',
                       columns: List[str] = ['close']) -> TimeSeries:
        """
        Get OHLC price data from Yahoo Finance.
        
        Args:
            symbol: The symbol to fetch data for
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            interval: Data interval (e.g., '1d' for daily, '1h' for hourly)
            
        Returns:
            TimeSeries: The historical data with metadata
        """
        ticker = self._get_ticker(symbol)
        
        # Convert string dates to datetime if provided
        start = datetime.strptime(start_date, '%Y-%m-%d') if start_date else None
        end = datetime.strptime(end_date, '%Y-%m-%d') if end_date else None
        
        # Map interval to Yahoo Finance format
        interval_map = {
            '1d': '1d',
            '1h': '1h',
            '1m': '1m',
            '5m': '5m',
            '15m': '15m',
            '30m': '30m',
            '60m': '60m'
        }
        yf_interval = interval_map.get(interval, '1d')
        
        # Fetch data
        df = ticker.history(start=start, end=end, interval=yf_interval)
        
        # Ensure we have the required columns
        required_columns = ['Open', 'High', 'Low', 'Close']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"Missing required columns. Available columns: {df.columns.tolist()}")
        
        # Rename columns to lowercase
        df = df.rename(columns={
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close'
        })
        df.index.names = ['timestamp']
        # Select only desired columns
        df = df[columns]
        
        # Get metadata
        info = ticker.info
        metadata = TimeSeriesMetadata(
            name=symbol,
            symbol=symbol,
            source='Yahoo Finance',
            start_date=df.index[0],
            end_date=df.index[-1],
            frequency=interval,
            currency=info.get('currency', 'USD'),
            additional_info={
                'name': info.get('longName', symbol),
                'sector': info.get('sector'),
                'industry': info.get('industry'),
                'exchange': info.get('exchange'),
                'market_cap': info.get('marketCap'),
                'pe_ratio': info.get('trailingPE'),
                'dividend_yield': info.get('dividendYield'),
                'beta': info.get('beta'),
                '52_week_high': info.get('fiftyTwoWeekHigh'),
                '52_week_low': info.get('fiftyTwoWeekLow')
            }
        )
        
        return TimeSeries(df, metadata)

    
    def get_metadata(self, symbol: str) -> Dict[str, Any]:
        """
        Get metadata for a symbol from Yahoo Finance.
        
        Args:
            symbol: The symbol to fetch metadata for
            
        Returns:
            Dictionary containing symbol metadata
        """
        ticker = self._get_ticker(symbol)
        info = ticker.info
        
        # Extract relevant metadata
        metadata = {
            'name': info.get('longName', symbol),
            'sector': info.get('sector'),
            'industry': info.get('industry'),
            'currency': info.get('currency'),
            'exchange': info.get('exchange'),
            'market_cap': info.get('marketCap'),
            'pe_ratio': info.get('trailingPE'),
            'dividend_yield': info.get('dividendYield'),
            'beta': info.get('beta'),
            '52_week_high': info.get('fiftyTwoWeekHigh'),
            '52_week_low': info.get('fiftyTwoWeekLow')
        }
        
        return metadata

    def get_available_symbols(self) -> List[str]:
        """
        Get list of available symbols from Yahoo Finance.
        Note: This is a limited implementation as Yahoo Finance doesn't provide
        a direct way to get all available symbols.

        Returns:
            List[str]: List of available symbols
        """
        # This is a placeholder - in practice, you might want to maintain
        # a list of known symbols or use a different approach
        return []

    def get_symbol_info(self, symbol: str) -> dict:
        """
        Get information about a specific symbol from Yahoo Finance.

        Args:
            symbol (str): The symbol to get information for

        Returns:
            dict: Information about the symbol
        """
        ticker = self._get_ticker(symbol)
        return ticker.info 