from finpie.analytics.indicators.value import Value
from finpie.analytics.indicators.ma_ma_distance import MaMaDistance
from finpie.analytics.indicators.momentum import Momentum
from finpie.analytics.indicators.rsi import Rsi
from finpie.analytics.indicators.value_ma_distance import ValueMaDistance
from finpie.strategy.domain import Domain
from functools import lru_cache
import json
from datetime import datetime, timedelta

class SignalService:
    def __init__(self, domain: Domain):
        self.domain = domain
        self.market_data_service = domain.get_service('market_data_service')
        self.indicator_map = {
            'value': {
                'class': Value,
                'instances': {}
            },
            'value_ma_distance': {
                'class': ValueMaDistance,
                'instances': {}
            },
            'ma_ma_distance': {
                'class': MaMaDistance,
                'instances': {}
            },
            'rsi': {
                'class': Rsi,
                'instances': {}
            },
            'momentum': {
                'class': Momentum,
                'instances': {}
            }
        }
        domain.set_service('signal_service', self)
    
    def get_signal(self, params: dict) -> dict:
        serialized_params = json.dumps(params)
        return self.get_signal_cached(self.domain.get_market_time(), serialized_params)
    
    @lru_cache(maxsize=1000)
    def get_signal_cached(self, asof_time: datetime, serialized_params: str):
        params = json.loads(serialized_params)
        main_indicator = self.get_indicator(asof_time, params['main_indicator'])
        indicator_value = main_indicator.z_score(params['z_score_window'])
        if 'secondary_indicator' in params:
            secondary_indicator = self.get_indicator(asof_time, params['secondary_indicator'])
            secondary_indicator_value = secondary_indicator.z_score(params['z_score_window'])
            indicator_value = indicator_value - secondary_indicator_value
            indicator_value = indicator_value.dropna()
            indicator_value = indicator_value.z_score(params['z_score_window'])
        return indicator_value

    def get_indicator(self, asof_time: datetime, indicator_params: dict):
        serialized_params = json.dumps(indicator_params)
        return self.get_indicator_cached(asof_time, serialized_params)
    
    @lru_cache(maxsize=1000)
    def get_indicator_cached(self, asof_time: datetime, serialized_params: str):
        params = json.loads(serialized_params)
        indicator_instance = self.indicator_map[params['indicator_type']]['instances'].get(serialized_params)
        if indicator_instance is None:
            indicator_class = self.indicator_map[params['indicator_type']]['class']
            start = asof_time - timedelta(days=7)
            timeseries = self.market_data_service.get_close_prices(params['symbol'], source='mt5', start_date=start, end_date=asof_time, interval='1m', subscribed=True)
            indicator_instance = indicator_class(timeseries, **params['params'])
            self.indicator_map[params['indicator_type']]['instances'][serialized_params] = indicator_instance
        return indicator_instance.calculate()
