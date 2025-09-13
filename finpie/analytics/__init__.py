"""
Analytics module for time series analysis and forecasting.

This module provides tools for analyzing financial time series data, including:
- Statistical analysis and risk metrics
- Forecasting models including LLM-based forecasting
- Performance measurement and portfolio analytics
- Technical indicators and market analysis

The module is designed to work seamlessly with the TimeSeries class from finpie.data.
"""

from .llm import LLMForecaster
from .genetic_portfolio_optimizer import GeneticPortfolioOptimizer
from .indicators.indicator import Indicator
from .indicators.trade_imbalance import TradeImbalance
from .indicators.rsi import Rsi
from .indicators.momentum import Momentum
from .indicators.value_ma_distance import ValueMaDistance
from .indicators.ma_ma_distance import MaMaDistance
from .indicators.value import Value

__all__ = [
    'LLMForecaster',
    'GeneticPortfolioOptimizer',
    'TradeImbalance',
    'Rsi',
    'Momentum',
    'ValueMaDistance',
    'MaMaDistance',
    'Value'
] 