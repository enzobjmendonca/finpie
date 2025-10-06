from dataclasses import asdict
from datetime import date, datetime
from typing import Any
import pandas as pd

class Domain:
    def __init__(self):
        self.market_time = datetime.now()
        self.fills = []
        self.strategy_data_history = {}
        self.strategy_data = []
        self.services = {}
        self.strategies = {}

    def set_service(self, service_name: str, service: Any):
        self.services[service_name] = service

    def get_service(self, service_name: str) -> Any:
        if service_name in self.services:
            return self.services[service_name]
        return None

    def get_market_time(self) -> datetime:
        return self.market_time

    def set_market_time(self, dt_time: datetime):
        self.market_time = dt_time
    
    def react(self, dt_time: datetime):
        self.market_time = dt_time
        self.services['market_data_service'].update_subscribed(self.market_time)
        for strategy in self.strategies.values():
            strategy.react()
        # Net everything and send to market    
        self.services['crb'].react()
        # Only publish one time per minute
        if not hasattr(self, "_last_publish_minute"):
            self._last_publish_minute = None
        current_minute = self.market_time.replace(second=0, microsecond=0)
        if self._last_publish_minute != current_minute:
            print(f"Publishing to supabase at {current_minute}")
            self.publish_to_supabase()
            self._last_publish_minute = current_minute

    def add_fill(self, fills):
        self.fills.append(fills)

    def add_strategy_data(self, strategy_data):
        self.strategy_data.append(strategy_data)

    def report_eod(self, strategy_data):
        if strategy_data.name not in self.strategy_data_history:
            self.strategy_data_history[strategy_data.name] = []
        self.strategy_data_history[strategy_data.name].append(strategy_data)

    def publish_to_supabase(self):
        if self.get_service('supabase_client'):
            try:
                self.get_service('supabase_client').upsert_fills(self.fills)
                self.fills = []
                self.get_service('supabase_client').upsert_data(self.strategy_data, 'strategy_data')
                self.get_service('supabase_client').upsert_data(self.services['crb'].crb_data_map.values(), 'crb_data')
            except Exception as e:
                print(e)

    def get_fills(self) -> list:
        return self.fills

    def save_strategy_data(self, strategy_name: str, strategy_data):
        if strategy_name not in self.strategy_data_history:
            self.strategy_data_history[strategy_name] = []
        data = asdict(strategy_data)
        data['time'] = self.market_time
        self.strategy_data_history[strategy_name].append(data)
        