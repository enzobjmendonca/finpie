"""
LLM-based forecasting tools for financial time series data.

This package implements a transformer-based language model for financial market data forecasting.
It provides a modular architecture with separate components for tokenization, dataset handling,
transformer modeling, and high-level forecasting.

The package provides the following main components:
- MarketTokenizer: Converts market data into discrete tokens
- ReturnTokenDataset: PyTorch dataset for handling tokenized market data
- MarketTransformer: Transformer model architecture for market data
- LLMForecaster: High-level interface for training and using the model

Example:
    >>> from finpie.analytics.llm import LLMForecaster
    >>> forecaster = LLMForecaster(seq_len=60, batch_size=64)
    >>> forecaster.build_dataset(returns)
    >>> forecaster.train(epochs=10)
    >>> predictions = forecaster.fit(prompt, steps=10)
"""

from .tokenizer import MarketTokenizer
from .dataset import ReturnTokenDataset
from .transformer import MarketTransformer, get_device
from .forecaster import LLMForecaster

# For backward compatibility
__all__ = [
    'MarketTokenizer',
    'ReturnTokenDataset', 
    'MarketTransformer',
    'LLMForecaster',
    'get_device'
]

# Version info
__version__ = "1.0.0"
__author__ = "finpie"
