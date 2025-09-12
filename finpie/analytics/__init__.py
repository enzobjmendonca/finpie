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

__all__ = [
    'LLMForecaster'
] 