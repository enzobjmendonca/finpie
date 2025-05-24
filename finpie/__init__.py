"""
FinPie - A comprehensive Python library for Brazilian financial data analysis and quantitative research
"""

__version__ = "0.1.0"

from finpie.datasource.sources.status_invest import StatusInvestSource
from finpie.datasource.sources.mt5 import MT5Source
from finpie.datasource.sources.yahoo import YahooFinanceSource
from finpie.datasource.sources.alpha_vantage import AlphaVantageSource
from finpie.datasource.sources.schemas.status_invest import FundamentalsParams
from finpie.datasource.service import DataService
from finpie.data.timeseries import TimeSeries
from finpie.data.multitimeseries import MultiTimeSeries
from finpie.data.ratiotimeseries import RatioTimeSeries
from finpie.data.spreadtimeseries import SpreadTimeSeries
from finpie.analytics.statistical import Statistical
from finpie.analytics.technical import Technical
from finpie.analytics.llm import LLMForecaster, MarketTokenizer

__all__ = ['StatusInvestSource', 'MT5Source', 'YahooFinanceSource', 'AlphaVantageSource', 'FundamentalsParams', 
            'TimeSeries', 'MultiTimeSeries', 'RatioTimeSeries', 'SpreadTimeSeries',
            'Statistical', 'Technical', 'LLMForecaster', 'MarketTokenizer', 'DataService'] 