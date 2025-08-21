"""
LLM-based forecasting tools for financial time series data.

This module serves as the main entry point for the LLM forecasting system.
It imports all components from the modular llm package for backward compatibility.

For new code, consider importing directly from the llm package:
    from finpie.analytics.llm import LLMForecaster, MarketTokenizer

This module provides the following main components:
- MarketTokenizer: Converts market data into discrete tokens
- ReturnTokenDataset: PyTorch dataset for handling tokenized market data
- MarketTransformer: Transformer model architecture for market data
- LLMForecaster: High-level interface for training and using the model
"""

# Import all components from the modular llm package
from .llm import (
    MarketTokenizer,
    ReturnTokenDataset,
    MarketTransformer,
    LLMForecaster,
    get_device
)

# For backward compatibility - export all classes at module level
__all__ = [
    'MarketTokenizer',
    'ReturnTokenDataset',
    'MarketTransformer', 
    'LLMForecaster',
    'get_device'
]

# Convenience alias for the main class
Forecaster = LLMForecaster
