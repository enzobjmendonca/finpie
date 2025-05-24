from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
import MetaTrader5 as mt5
import pandas as pd
import platform
import warnings

from finpie.datasource.sources.base import DataSource
from finpie.data.timeseries import TimeSeries, TimeSeriesMetadata

# Only import MT5 on Windows
if platform.system() == 'Windows':
    try:
        import MetaTrader5 as mt5
    except ImportError:
        warnings.warn("MetaTrader5 module not available. MT5 functionality will be disabled.")
        mt5 = None
else:
    warnings.warn("MetaTrader5 is only available on Windows. MT5 functionality will be disabled.")
    mt5 = None

MAX_PAYLOAD_SIZE = 1_000_000

class MT5Source(DataSource):
    """
    MetaTrader 5 data source implementation.
    Provides access to real-time and historical forex, stocks, and futures data.
    """
    def __init__(self):
        """
        Initialize the MT5 data source.
        """
        self.name = "mt5"
        self._initialized = False
        
        if mt5 is None:
            warnings.warn("MetaTrader5 module not available. MT5 functionality will be disabled.")
            return
            
        # Initialize MT5
        if not mt5.initialize():
            warnings.warn(f"Failed to initialize MT5: {mt5.last_error()}")
            return
                
        self._initialized = True

    def __del__(self):
        """Cleanup MT5 connection."""
        if self._initialized:
            mt5.shutdown()

    def _convert_timeframe(self, frequency: str, to_minutes: bool = False) -> int:
        """
        Convert frequency string to MT5 timeframe constant.

        Args:
            frequency (str): Frequency string (e.g., '1m', '1h', '1d')

        Returns:
            int: MT5 timeframe constant
        """
        timeframe_map = {
            '1m': mt5.TIMEFRAME_M1,
            '5m': mt5.TIMEFRAME_M5,
            '15m': mt5.TIMEFRAME_M15,
            '30m': mt5.TIMEFRAME_M30,
            '1h': mt5.TIMEFRAME_H1,
            '4h': mt5.TIMEFRAME_H4,
            '1d': mt5.TIMEFRAME_D1,
            '1w': mt5.TIMEFRAME_W1,
            '1M': mt5.TIMEFRAME_MN1
        }
        if to_minutes:
            timeframe_map = {
                '1m': 1,
                '5m': 5,
                '15m': 15,
                '30m': 30,
                '1h': 60,
                '4h': 240,
                '1d': 1440,
                '1w': 10080,
                '1M': 44640
            }
        return timeframe_map.get(frequency, mt5.TIMEFRAME_D1)

    def get_rate_from_range(self, symbol: str, start_date: datetime, end_date: datetime, timeframe: int) -> pd.DataFrame:
        rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
            
        if rates is None or len(rates) == 0:
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        return df

    def get_historical_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        frequency: str = '1d'
    ) -> TimeSeries:
        """
        Fetch historical data from MT5.

        Args:
            symbol (str): The symbol to fetch data for
            start_date (datetime): Start date for the data
            end_date (datetime): End date for the data
            frequency (str): Data frequency (e.g., '1d' for daily, '1h' for hourly)

        Returns:
            TimeSeries: The historical data with metadata
        """
        timeframe = self._convert_timeframe(frequency)

        payload_size = self.calc_estimated_payload_size(symbol, start_date, end_date, interval)
        if payload_size > MAX_PAYLOAD_SIZE:
            # Split the date range into daily chunks to avoid payload size limit
            all_data = []
            current_date = start_date
            while current_date < end_date:
                next_date = min(current_date + timedelta(days=1), end_date)
                df = self.get_rate_from_range(symbol, current_date, next_date, timeframe)
                all_data.append(df)
                current_date = next_date
                
            # Concatenate all the daily data
            df = pd.concat(all_data)
        else:    
            df = self.get_rate_from_range(symbol, start_date, end_date, timeframe)
        
        # Rename columns to match standard format
        df.rename(columns={
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'tick_volume': 'Volume'
        }, inplace=True)

        # Get symbol info
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            raise ValueError(f"Symbol {symbol} not found in MT5")

        metadata = TimeSeriesMetadata(
            name=symbol,
            symbol=symbol,
            source='MetaTrader 5',
            start_date=df.index[0],
            end_date=df.index[-1],
            frequency=frequency,
            currency=symbol_info.currency_base,
            additional_info={
                'description': symbol_info.description,
                'digits': symbol_info.digits,
                'spread': symbol_info.spread,
                'trade_contract_size': symbol_info.trade_contract_size,
                'volume_min': symbol_info.volume_min,
                'volume_max': symbol_info.volume_max
            }
        )

        return TimeSeries(df, metadata)

    def get_prices(self, symbol: str, start_date: Optional[str] = None, 
                       end_date: Optional[str] = None, interval: str = '1d',
                       columns: List[str] = ['close']) -> TimeSeries:
        """
        Get OHLC price data from MT5.
        
        Args:
            symbol: The symbol to fetch data for
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            interval: Data interval (e.g., '1d' for daily, '1h' for hourly)
            
        Returns:
            TimeSeries: The historical data with metadata
        """
        if not self._initialized:
            raise RuntimeError("MT5 is not initialized")
            
        # Map interval to MT5 timeframe
        timeframe_map = {
            '1m': mt5.TIMEFRAME_M1,
            '5m': mt5.TIMEFRAME_M5,
            '15m': mt5.TIMEFRAME_M15,
            '30m': mt5.TIMEFRAME_M30,
            '1h': mt5.TIMEFRAME_H1,
            '4h': mt5.TIMEFRAME_H4,
            '1d': mt5.TIMEFRAME_D1,
            '1w': mt5.TIMEFRAME_W1,
            '1M': mt5.TIMEFRAME_MN1
        }
        timeframe = timeframe_map.get(interval, mt5.TIMEFRAME_D1)
        
        # Convert dates to datetime
        start_date = datetime.strptime(start_date, '%Y-%m-%d') if start_date else datetime(2000, 1, 1)
        end_date = datetime.strptime(end_date, '%Y-%m-%d') if end_date else datetime.now()
        
        payload_size = self.calc_estimated_payload_size(symbol, start_date, end_date, interval)
        if payload_size > MAX_PAYLOAD_SIZE:
            # Split the date range into daily chunks to avoid payload size limit
            all_data = []
            current_date = start_date
            while current_date < end_date:
                next_date = min(current_date + timedelta(days=1), end_date)
                df = self.get_rate_from_range(symbol, current_date, next_date, timeframe)
                all_data.append(df)
                current_date = next_date
                
            # Concatenate all the daily data
            df = pd.concat(all_data)
        else:    
            df = self.get_rate_from_range(symbol, start_date, end_date, timeframe)
        
        # Rename columns to lowercase
        df = df.rename(columns={
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close'
        })
        
        # Select only desired columns
        df = df[columns]
        
        # Get metadata
        info = self.get_symbol_info(symbol)

        metadata = TimeSeriesMetadata(
            name=symbol,
            symbol=symbol,
            source='MetaTrader 5',
            start_date=df.index[0],
            end_date=df.index[-1],
            frequency=interval,
            currency=info.get('currency', 'USD'),
            additional_info={
                'name': info.get('name', symbol),
                'type': info.get('type'),
                'exchange': info.get('exchange'),
                'point': info.get('point'),
                'digits': info.get('digits'),
                'spread': info.get('spread'),
                'trade_contract_size': info.get('trade_contract_size'),
                'volume_min': info.get('volume_min'),
                'volume_max': info.get('volume_max')
            }
        )
        
        return TimeSeries(df, metadata)
    
    def get_metadata(self, symbol: str) -> Dict[str, Any]:
        """
        Get metadata for a symbol from MT5.
        
        Args:
            symbol: The symbol to fetch metadata for
            
        Returns:
            Dictionary containing symbol metadata
        """
        if not self._initialized:
            raise RuntimeError("MT5 is not initialized")
            
        # Get symbol info
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            raise ValueError(f"Symbol {symbol} not found")
            
        return {
            'name': symbol_info.name,
            'type': symbol_info.type,
            'currency': symbol_info.currency_base,
            'exchange': symbol_info.exchange,
            'point': symbol_info.point,
            'digits': symbol_info.digits,
            'spread': symbol_info.spread,
            'trade_contract_size': symbol_info.trade_contract_size,
            'volume_min': symbol_info.volume_min,
            'volume_max': symbol_info.volume_max
        }

    def get_available_symbols(self) -> List[str]:
        """
        Get list of available symbols from MT5.

        Returns:
            List[str]: List of available symbols
        """
        if not self._initialized:
            raise RuntimeError("MT5 is not initialized")
            
        symbols = mt5.symbols_get()
        return [symbol.name for symbol in symbols]

    def get_symbol_info(self, symbol: str) -> dict:
        """
        Get information about a specific symbol from MT5.

        Args:
            symbol (str): The symbol to get information for

        Returns:
            dict: Information about the symbol
        """
        if not self._initialized:
            raise RuntimeError("MT5 is not initialized")
            
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            raise ValueError(f"Symbol {symbol} not found")
        return {
            'name': symbol_info.name,
            'type': symbol_info.category,
            'currency': symbol_info.currency_base,
            'exchange': symbol_info.exchange,
            'point': symbol_info.point,
            'digits': symbol_info.digits,
            'spread': symbol_info.spread,
            'trade_contract_size': symbol_info.trade_contract_size,
            'volume_min': symbol_info.volume_min,
            'volume_max': symbol_info.volume_max
        }

    def calc_estimated_payload_size(self, symbol: str, start_date: datetime, end_date: datetime, interval: str) -> int:
        """
        Calculate the estimated payload size for a given symbol and date range.

        Args:
            symbol: The symbol to calculate the payload size for
            start_date: The start date for the payload
            end_date: The end date for the payload
            interval: The interval for the payload

        Returns:
            int: The estimated payload size in bytes
        """
        interval_minutes = self._convert_timeframe(interval)
        num_days = (end_date - start_date).days
        num_rows = num_days * 24 * 60 / interval_minutes
        return num_rows # approximate size of each row in bytes

