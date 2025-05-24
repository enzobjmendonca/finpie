Analytics Module
===============

This module provides various analytics tools for financial data analysis, including statistical analysis, technical indicators, and machine learning-based forecasting.

.. automodule:: finpie.analytics
   :members:
   :undoc-members:
   :show-inheritance:

Statistical Analytics
-------------------

.. autoclass:: finpie.analytics.statistical.Statistical
   :members:
   :undoc-members:
   :show-inheritance:

   The Statistical class provides various statistical measures and indicators commonly used in quantitative finance:

   - Z-scores for identifying overbought/oversold conditions
   - Half-life calculation for mean reversion analysis
   - Hurst exponent for determining if a series is mean-reverting
   - Spread to moving average calculations
   - Trading signals based on statistical measures

Technical Analytics
-----------------

.. autoclass:: finpie.analytics.technical.Technical
   :members:
   :undoc-members:
   :show-inheritance:

   The Technical class provides a comprehensive set of technical indicators:

   - Moving Averages (SMA, EMA)
   - Momentum Indicators (RSI, MACD)
   - Volatility Indicators (ATR)
   - Trend Indicators (ADX)
   - Volume Indicators (OBV)
   - Oscillators (Stochastic)

LLM Analytics
------------

.. autoclass:: finpie.analytics.llm.LLMForecaster
   :members:
   :undoc-members:
   :show-inheritance:

   The LLMForecaster class implements a transformer-based language model for financial market data forecasting:

   - Market data tokenization
   - Transformer model architecture
   - Training and inference capabilities
   - Sequence generation for market predictions

.. autoclass:: finpie.analytics.llm.MarketTokenizer
   :members:
   :undoc-members:
   :show-inheritance:

   The MarketTokenizer class handles the conversion between continuous market data and discrete tokens:

   - Equal-width and equal-frequency binning
   - Token-to-value conversion
   - Customizable binning parameters

.. autoclass:: finpie.analytics.llm.ReturnTokenDataset
   :members:
   :undoc-members:
   :show-inheritance:

   The ReturnTokenDataset class provides a PyTorch dataset for handling tokenized market data:

   - Sequence generation for training
   - Efficient data loading
   - Customizable sequence lengths

.. autoclass:: finpie.analytics.llm.MarketTransformer
   :members:
   :undoc-members:
   :show-inheritance:

   The MarketTransformer class implements the transformer model architecture:

   - Token and positional embeddings
   - Multi-head attention
   - Transformer encoder layers
   - Output projection 