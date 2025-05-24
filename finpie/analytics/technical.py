"""
Technical analysis tools for financial time series data.

This module provides a comprehensive set of technical indicators commonly used in
financial analysis, including moving averages, momentum indicators, and oscillators.
"""

from typing import Optional, Union, List
import pandas as pd
import numpy as np
from finpie.data.timeseries import TimeSeries

class Technical:
    """
    A collection of technical indicators for quantitative and technical analysis.
    
    This class provides a comprehensive set of technical indicators commonly used in
    financial analysis. All methods take a TimeSeries as input and return a TimeSeries
    as output.
    
    Attributes:
        timeseries (TimeSeries): The input time series data
        data (pd.DataFrame): The actual data being analyzed
        column (str): The column name to analyze
    
    Example:
        >>> from finpie import TimeSeries, Technical
        >>> ts = TimeSeries(data)
        >>> tech = Technical(ts)
        >>> sma = tech.sma(window=20)
        >>> rsi = tech.rsi(window=14)
    """
    
    def __init__(self, timeseries: TimeSeries, column: str = None):
        """
        Initialize TechnicalIndicators with a TimeSeries object.
        
        Args:
            timeseries: TimeSeries object containing price data
            column: Name of the column to analyze (default: first column)
            
        Raises:
            TypeError: If timeseries is not a TimeSeries object
        """
        if not isinstance(timeseries, TimeSeries):
            raise TypeError("timeseries must be a TimeSeries object")
            
        self.timeseries = timeseries
        self.data = timeseries.data
        self.column = column if column is not None else timeseries.data.columns[0]

    def sma(self, window: int = 20) -> TimeSeries:
        """
        Calculate Simple Moving Average.
        
        The SMA is calculated by taking the arithmetic mean of a set of values
        over a specified time period.
        
        Args:
            window: Size of the moving window
            
        Returns:
            TimeSeries containing SMA values
            
        Example:
            >>> sma = tech.sma(window=20)
            >>> # sma.data contains the simple moving average values
        """
        sma = self.data[self.column].rolling(window=window).mean()
        return TimeSeries(pd.DataFrame({f'sma_{window}': sma}))
    
    def ema(self, window: int = 20) -> TimeSeries:
        """
        Calculate Exponential Moving Average.
        
        The EMA gives more weight to recent prices, making it more responsive
        to price changes than the SMA.
        
        Args:
            window: Size of the moving window
            
        Returns:
            TimeSeries containing EMA values
            
        Example:
            >>> ema = tech.ema(window=20)
            >>> # ema.data contains the exponential moving average values
        """
        ema = self.data[self.column].ewm(span=window, adjust=False).mean()
        return TimeSeries(pd.DataFrame({f'ema_{window}': ema}))
    
    def rsi(self, window: int = 14) -> TimeSeries:
        """
        Calculate Relative Strength Index.
        
        The RSI is a momentum oscillator that measures the speed and change of
        price movements. It ranges from 0 to 100, with values above 70 indicating
        overbought conditions and values below 30 indicating oversold conditions.
        
        Args:
            window: Size of the moving window
            
        Returns:
            TimeSeries containing RSI values
            
        Example:
            >>> rsi = tech.rsi(window=14)
            >>> # rsi.data contains the RSI values
            >>> # Values above 70 indicate overbought conditions
            >>> # Values below 30 indicate oversold conditions
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
        
        The MACD is a trend-following momentum indicator that shows the relationship
        between two moving averages of a security's price.
        
        Args:
            fast: Fast period
            slow: Slow period
            signal: Signal period
            
        Returns:
            TimeSeries containing MACD histogram values
            
        Example:
            >>> macd = tech.macd(fast=12, slow=26, signal=9)
            >>> # macd.data contains the MACD histogram values
            >>> # Positive values indicate bullish momentum
            >>> # Negative values indicate bearish momentum
        """
        exp1 = self.data[self.column].ewm(span=fast, adjust=False).mean()
        exp2 = self.data[self.column].ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        histogram = macd - signal_line
        
        return TimeSeries(pd.DataFrame({f'macd_hist': histogram}))
    
    def atr(self, window: int = 14) -> TimeSeries:
        """
        Calculate Average True Range.
        
        The ATR is a measure of volatility that shows the degree of price volatility
        over a specified time period.
        
        Args:
            window: Size of the moving window
            
        Returns:
            TimeSeries containing ATR values
            
        Example:
            >>> atr = tech.atr(window=14)
            >>> # atr.data contains the ATR values
            >>> # Higher values indicate higher volatility
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
        
        The Stochastic Oscillator is a momentum indicator comparing a particular
        closing price to a range of prices over a certain period of time.
        
        Args:
            k_window: %K period
            d_window: %D period
            
        Returns:
            TimeSeries containing %K and %D values
            
        Example:
            >>> stoch = tech.stochastic_oscillator(k_window=14, d_window=3)
            >>> # stoch.data contains %K and %D values
            >>> # Values above 80 indicate overbought conditions
            >>> # Values below 20 indicate oversold conditions
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
        
        The ADX is a trend strength indicator that measures the strength of a trend,
        regardless of its direction.
        
        Args:
            window: Size of the moving window
            
        Returns:
            TimeSeries containing ADX, +DI, and -DI values
            
        Example:
            >>> adx = tech.adx(window=14)
            >>> # adx.data contains ADX, +DI, and -DI values
            >>> # ADX > 25 indicates a strong trend
            >>> # +DI > -DI indicates bullish trend
            >>> # -DI > +DI indicates bearish trend
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
        
        The OBV is a momentum indicator that uses volume flow to predict changes in price.
        It shows whether volume is flowing into or out of a security.
        
        Returns:
            TimeSeries containing OBV values
            
        Example:
            >>> obv = tech.obv()
            >>> # obv.data contains the OBV values
            >>> # Rising OBV indicates positive volume flow
            >>> # Falling OBV indicates negative volume flow
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