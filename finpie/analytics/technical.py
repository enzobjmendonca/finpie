from typing import Optional, Union, List
import pandas as pd
import numpy as np
from finpie.data.timeseries import TimeSeries

class Technical:
    """
    A collection of technical indicators for quantitative and technical analysis.
    All methods take a TimeSeries as input and return a TimeSeries as output.
    """
    
    def __init__(self, timeseries: TimeSeries, column: str = None):
        """
        Initialize TechnicalIndicators with a TimeSeries object.
        
        Args:
            timeseries: TimeSeries object containing price data
        """
        if not isinstance(timeseries, TimeSeries):
            raise TypeError("timeseries must be a TimeSeries object")
            
        self.timeseries = timeseries
        self.data = timeseries.data
        self.column = column if column is not None else timeseries.data.columns[0]

    def sma(self, window: int = 20) -> TimeSeries:
        """
        Calculate Simple Moving Average.
        
        Args:
            window: Size of the moving window
            column: Column to calculate SMA for
            
        Returns:
            TimeSeries containing SMA values
        """
        sma = self.data[self.column].rolling(window=window).mean()
        return TimeSeries(pd.DataFrame({f'sma_{window}': sma}))
    
    def ema(self, window: int = 20) -> TimeSeries:
        """
        Calculate Exponential Moving Average.
        
        Args:
            window: Size of the moving window
            column: Column to calculate EMA for
            
        Returns:
            TimeSeries containing EMA values
        """
        ema = self.data[self.column].ewm(span=window, adjust=False).mean()
        return TimeSeries(pd.DataFrame({f'ema_{window}': ema}))
    
    def rsi(self, window: int = 14) -> TimeSeries:
        """
        Calculate Relative Strength Index.
        
        Args:
            window: Size of the moving window
            column: Column to calculate RSI for
            
        Returns:
            TimeSeries containing RSI values
        """
        delta = self.data[self.column].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return TimeSeries(pd.DataFrame({f'rsi_{window}': rsi}))
    
    def macd(self, fast: int = 12, slow: int = 26, signal: int = 9) -> TimeSeries:
        """
        Calculate Moving Average Convergence Divergence.
        
        Args:
            fast: Fast period
            slow: Slow period
            signal: Signal period
            column: Column to calculate MACD for
            
        Returns:
            TimeSeries containing MACD momentum (histogram)
        """
        exp1 = self.data[self.column].ewm(span=fast, adjust=False).mean()
        exp2 = self.data[self.column].ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        histogram = macd - signal_line
        
        return TimeSeries(histogram)
    
    def atr(self, window: int = 14) -> TimeSeries:
        """
        Calculate Average True Range.
        
        Args:
            window: Size of the moving window
            
        Returns:
            TimeSeries containing ATR values
        """
        high = self.data['high']
        low = self.data['low']
        close = self.data['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=window).mean()
        
        return TimeSeries(pd.DataFrame({f'atr_{window}': atr}))
    
    def stochastic_oscillator(self, k_window: int = 14, d_window: int = 3) -> TimeSeries:
        """
        Calculate Stochastic Oscillator.
        
        Args:
            k_window: %K period
            d_window: %D period
            
        Returns:
            TimeSeries containing %K and %D values
        """
        low_min = self.data['low'].rolling(window=k_window).min()
        high_max = self.data['high'].rolling(window=k_window).max()
        
        k = 100 * ((self.data['close'] - low_min) / (high_max - low_min))
        d = k.rolling(window=d_window).mean()
        
        return TimeSeries(pd.DataFrame({
            f'stoch_k_{k_window}': k,
            f'stoch_d_{d_window}': d
        }))
    
    def adx(self, window: int = 14) -> TimeSeries:
        """
        Calculate Average Directional Index.
        
        Args:
            window: Size of the moving window
            
        Returns:
            TimeSeries containing ADX values
        """
        high = self.data['high']
        low = self.data['low']
        close = self.data['close']
        
        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate Directional Movement
        up_move = high - high.shift()
        down_move = low.shift() - low
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Calculate smoothed averages
        tr_smoothed = tr.rolling(window=window).sum()
        plus_di = 100 * pd.Series(plus_dm).rolling(window=window).sum() / tr_smoothed
        minus_di = 100 * pd.Series(minus_dm).rolling(window=window).sum() / tr_smoothed
        
        # Calculate ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=window).mean()
        
        return TimeSeries(pd.DataFrame({
            'adx': adx,
            'plus_di': plus_di,
            'minus_di': minus_di
        }))
    
    def obv(self) -> TimeSeries:
        """
        Calculate On-Balance Volume.
        
        Returns:
            TimeSeries containing OBV values
        """
        close = self.data['close']
        volume = self.data['volume']
        
        obv = pd.Series(0, index=close.index)
        for i in range(1, len(close)):
            if close[i] > close[i-1]:
                obv[i] = obv[i-1] + volume[i]
            elif close[i] < close[i-1]:
                obv[i] = obv[i-1] - volume[i]
            else:
                obv[i] = obv[i-1]
                
        return TimeSeries(pd.DataFrame({'obv': obv})) 