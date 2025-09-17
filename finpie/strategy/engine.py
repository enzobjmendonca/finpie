import time
from finpie.strategy.domain import Domain
from finpie.datasource.service import DataService
from finpie.strategy.signal_service import SignalService
from datetime import datetime
from finpie.strategy.strategy import Strategy, StrategyParams

from finpie.strategy.supabase_client import SupabaseClient

class Engine:
    def __init__(self, instance_id: str):
        self.instance_id = instance_id
        self.domain = Domain()
        self.data_service = DataService.create_default_service()
        self.domain.set_service('market_data_service', self.data_service)
        self.signal_service = SignalService(self.domain)
        self.domain.set_service('signal_service', self.signal_service)
        self.supabase_client = SupabaseClient('https://axyjplnsepzaictqknjl.supabase.co', 'sb_secret_0-HxZgwl4I7gPc9QFkZhWQ_Ibj-5Fgw')
        self.domain.set_service('supabase_client', self.supabase_client)

    def create_strategies(self, strategies: list[StrategyParams]):
        for strategy in strategies:
            strategy = Strategy(self.domain, strategy)
            self.domain.strategies[strategy.params.name] = strategy

    def load_strategies(self):
        strategies = self.supabase_client.get_strategy_params(instance=self.instance_id)
        self.create_strategies(strategies)

    def run(self):
        i = 0
        while True:
            start = datetime.now()
            self.domain.react(datetime.now())
            end = datetime.now()
            print("cycle time: ", end - start, "cycle number: ", i, ' from ', start, ' to ', end)
            i += 1
            time.sleep(5)
    
    def backtest(self, start_date: datetime, end_date: datetime, reference_symbol: str):
        timeseries = self.data_service.get_close_prices(reference_symbol, source='mt5', start_date=start_date, end_date=end_date, interval='1m', subscribed=True)
        i = 0
        for time in timeseries.index:
            i += 1
            if i % 1000 == 0:
                print(time)
            self.domain.react(time)


def main(instance_name: str):
    engine = Engine(instance_name)
    engine.load_strategies()
    engine.run()

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python engine.py <instance_name>")
        sys.exit(1)
    main(sys.argv[1])

