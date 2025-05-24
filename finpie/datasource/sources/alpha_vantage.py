from alpha_vantage.timeseries import TimeSeries as AVTimeSeries
from alpha_vantage.fundamentaldata import FundamentalData
from datetime import datetime
from typing import List, Dict, Optional, Any
import requests
import pandas as pd

from finpie.datasource.sources.base import DataSource
from finpie.data.timeseries import TimeSeries, TimeSeriesMetadata

class AlphaVantageSource(DataSource):
    """
    Alpha Vantage data source implementation.
    Provides access to real-time and historical stock data, forex, and cryptocurrencies.
    """
    BASE_URL = "https://www.alphavantage.co/query"

    def __init__(self, api_key: str):
        """
        Initialize the Alpha Vantage data source.

        Args:
            api_key (str): Alpha Vantage API key
        """
        self.name = "alpha_vantage"
        self.api_key = api_key
        self._ts = AVTimeSeries(key=api_key, output_format='pandas')
        self._fd = FundamentalData(key=api_key, output_format='pandas')
        self._cache = {}

    def _make_request(self, function: str, params: Dict) -> Dict:
        """
        Make a request to the Alpha Vantage API.

        Args:
            function (str): API function to call
            params (Dict): Additional parameters for the request

        Returns:
            Dict: API response

        Raises:
            ValueError: If the API request fails
        """
        params.update({
            'function': function,
            'apikey': self.api_key
        })

        response = requests.get(self.BASE_URL, params=params)
        data = response.json()

        if 'Error Message' in data:
            raise ValueError(f"Alpha Vantage API error: {data['Error Message']}")

        return data

    def _convert_frequency(self, frequency: str) -> str:
        """
        Convert frequency string to Alpha Vantage interval.

        Args:
            frequency (str): Frequency string (e.g., '1d', '1h', '1m')

        Returns:
            str: Alpha Vantage interval
        """
        frequency_map = {
            '1m': '1min',
            '5m': '5min',
            '15m': '15min',
            '30m': '30min',
            '1h': '60min',
            '1d': 'daily',
            '1w': 'weekly',
            '1M': 'monthly'
        }
        return frequency_map.get(frequency, 'daily')

    def get_available_symbols(self) -> List[str]:
        """
        Get list of available symbols from Alpha Vantage.
        Note: Alpha Vantage doesn't provide a direct way to get all symbols,
        so this is a limited implementation.

        Returns:
            List[str]: List of available symbols
        """
        # This is a placeholder - in practice, you might want to maintain
        # a list of known symbols or use a different approach
        return []

    def get_symbol_info(self, symbol: str) -> dict:
        """
        Get information about a specific symbol from Alpha Vantage.

        Args:
            symbol (str): The symbol to get information for

        Returns:
            dict: Information about the symbol
        """
        if symbol in self._cache:
            return self._cache[symbol]

        data = self._make_request('OVERVIEW', {'symbol': symbol})
        
        info = {
            'name': data.get('Name', ''),
            'type': data.get('AssetType', ''),
            'currency': data.get('Currency', 'USD'),
            'market_cap': data.get('MarketCapitalization', ''),
            'sector': data.get('Sector', ''),
            'industry': data.get('Industry', ''),
            'exchange': data.get('Exchange', ''),
            'country': data.get('Country', ''),
            'description': data.get('Description', '')
        }

        self._cache[symbol] = info
        return info

    def get_prices(self, symbol: str, start_date: Optional[str] = None, 
                       end_date: Optional[str] = None, interval: str = '1d',
                       columns: List[str] = ['close']) -> TimeSeries:
        """
        Get OHLC price data from Alpha Vantage.
        
        Args:
            symbol: The symbol to fetch data for
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            interval: Data interval (e.g., '1d' for daily, '1h' for hourly)
            
        Returns:
            TimeSeries: The historical data with metadata
        """
        # Map interval to Alpha Vantage format
        interval_map = {
            '1d': 'daily',
            '1h': 'hourly',
            '5m': '5min',
            '15m': '15min',
            '30m': '30min',
            '60m': '60min'
        }
        av_interval = interval_map.get(interval, 'daily')
        
        # Fetch data
        if av_interval == 'daily':
            df, _ = self._ts.get_daily(symbol=symbol, outputsize='full')
        else:
            df, _ = self._ts.get_intraday(symbol=symbol, interval=av_interval, outputsize='full')
        
        # Ensure we have the required columns
        required_columns = ['1. open', '2. high', '3. low', '4. close']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"Missing required columns. Available columns: {df.columns.tolist()}")
        
        # Rename columns to lowercase
        df = df.rename(columns={
            '1. open': 'open',
            '2. high': 'high',
            '3. low': 'low',
            '4. close': 'close'
        })
        
        # Select only desired columns
        df = df[columns ]
        
        # Filter by date range if provided
        if start_date:
            start = datetime.strptime(start_date, '%Y-%m-%d')
            df = df[df.index >= start]
        if end_date:
            end = datetime.strptime(end_date, '%Y-%m-%d')
            df = df[df.index <= end]
        
        # Get metadata
        info = self.get_symbol_info(symbol)
        metadata = TimeSeriesMetadata(
            name=symbol,
            symbol=symbol,
            source='Alpha Vantage',
            start_date=df.index[0],
            end_date=df.index[-1],
            frequency=interval,
            currency=info.get('currency', 'USD'),
            additional_info={
                'name': info.get('name', symbol),
                'type': info.get('type'),
                'sector': info.get('sector'),
                'industry': info.get('industry'),
                'exchange': info.get('exchange'),
                'market_cap': info.get('market_cap'),
                'pe_ratio': info.get('pe_ratio'),
                'dividend_yield': info.get('dividend_yield'),
                'beta': info.get('beta'),
                '52_week_high': info.get('52_week_high'),
                '52_week_low': info.get('52_week_low')
            }
        )
        
        return TimeSeries(df, metadata)
    
    def get_metadata(self, symbol: str) -> Dict[str, Any]:
        """
        Get metadata for a symbol from Alpha Vantage.
        
        Args:
            symbol: The symbol to fetch metadata for
            
        Returns:
            Dictionary containing symbol metadata
        """
        if symbol in self._cache:
            return self._cache[symbol]
            
        # Get company overview
        overview, _ = self._fd.get_company_overview(symbol=symbol)
        
        # Extract relevant metadata
        metadata = {
            'name': overview.get('Name', symbol),
            'sector': overview.get('Sector'),
            'industry': overview.get('Industry'),
            'currency': overview.get('Currency'),
            'exchange': overview.get('Exchange'),
            'market_cap': float(overview.get('MarketCapitalization', 0)),
            'pe_ratio': float(overview.get('PERatio', 0)),
            'dividend_yield': float(overview.get('DividendYield', 0)),
            'beta': float(overview.get('Beta', 0)),
            '52_week_high': float(overview.get('52WeekHigh', 0)),
            '52_week_low': float(overview.get('52WeekLow', 0))
        }
        
        self._cache[symbol] = metadata
        return metadata 