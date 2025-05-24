"""
Analytics module for performing statistical and technical analysis on financial data.

This module provides three main components:

1. Statistical Analysis (finpie.analytics.statistical)
   - Z-scores, half-life, Hurst exponent
   - Mean reversion analysis
   - Trading signals based on statistical measures

2. Technical Analysis (finpie.analytics.technical)
   - Moving averages (SMA, EMA)
   - Momentum indicators (RSI, MACD)
   - Volatility indicators (ATR)
   - Trend indicators (ADX)
   - Volume indicators (OBV)
   - Oscillators (Stochastic)

3. LLM-based Forecasting (finpie.analytics.llm)
   - Transformer-based market forecasting
   - Market data tokenization
   - Sequence generation for predictions

Example:
    >>> from finpie import TimeSeries, Statistical, Technical
    >>> # Create a TimeSeries object
    >>> ts = TimeSeries(data)
    >>> # Perform statistical analysis
    >>> stats = Statistical(ts)
    >>> zscore = stats.zscore()
    >>> # Calculate technical indicators
    >>> tech = Technical(ts)
    >>> rsi = tech.rsi()
"""

from .statistical import Statistical
from .technical import Technical
from .llm import LLMForecaster

__all__ = [
    'Statistical',
    'Technical',
    'LLMForecaster'
] 