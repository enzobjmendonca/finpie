"""
FinPie - A comprehensive Python library for Brazilian financial data analysis and quantitative research
"""

__version__ = "0.1.31"

from finpie.data import TimeSeries, MultiTimeSeries, RatioTimeSeries, SpreadTimeSeries
from finpie.datasource.service import DataService
from finpie.datasource.sources.yahoo import YahooFinanceSource
from finpie.datasource.sources.alpha_vantage import AlphaVantageSource
from finpie.datasource.sources.status_invest import StatusInvestSource

# Optional imports
try:
    from finpie.datasource.sources.mt5 import MT5Source
except ImportError:
    MT5Source = None  # MT5Source will be None if MetaTrader5 is not installed

from finpie.datasource.sources.schemas.status_invest import FundamentalsParams
from finpie.analytics.statistical import Statistical
from finpie.analytics.technical import Technical
from finpie.analytics.llm import LLMForecaster, MarketTokenizer

__all__ = [
    'TimeSeries',
    'MultiTimeSeries',
    'RatioTimeSeries',
    'SpreadTimeSeries',
    'DataService',
    'YahooFinanceSource',
    'AlphaVantageSource',
    'StatusInvestSource',
    'MT5Source',
    'FundamentalsParams',
    'Statistical',
    'Technical',
    'LLMForecaster',
    'MarketTokenizer'
] 