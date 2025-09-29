import time
from finpie.strategy.domain import Domain
from finpie.datasource.service import DataService
from finpie.strategy.signal_service import SignalService
from datetime import datetime
from finpie.strategy.strategy import Strategy, StrategyParams
from finpie.strategy.crb import CRB, CRBParams
from finpie.strategy.supabase_client import SupabaseClient

class Engine:
    def __init__(self, instance_id: str, supabase_url: str, supabase_key: str):
        self.instance_id = instance_id
        self.domain = Domain()
        self.crb = CRB(domain=self.domain, params=CRBParams(magic=2, symbol_map={'WIN$N': 'WINV25'}, multiplier=0.5))
        self.domain.set_service('crb', self.crb)
        self.data_service = DataService.create_default_service()
        self.domain.set_service('market_data_service', self.data_service)
        self.signal_service = SignalService(self.domain)
        self.domain.set_service('signal_service', self.signal_service)
        if supabase_url and supabase_key:
            self.supabase_client = SupabaseClient(supabase_url, supabase_key)
            self.domain.set_service('supabase_client', self.supabase_client)
        else:
            self.supabase_client = None

    def create_strategies(self, strategies: list[StrategyParams]):
        for strategy in strategies:
            strategy = Strategy(self.domain, strategy)
            self.domain.strategies[strategy.params.name] = strategy

    def load_strategies(self):
        if self.supabase_client:
            strategies = self.supabase_client.get_strategy_params(instance=self.instance_id)
        else:
            #Add your own strategies here
            strategies = [
                StrategyParams(
                    instance=self.instance_id,
                    name='test',
                    symbol='IBOV',
                    order_size=100,
                    max_position=1000,
                    max_volume=10000,
                    take_profit=100,
                    stop_loss=100,
                    start_time='10:00',
                    end_time='17:00',
                    signal_params={},
                    signal_threshold=1.0,
                    signal_direction=1
                )
            ]
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


def main(instance_name: str, supabase_url: str, supabase_key: str):
    engine = Engine(instance_name, supabase_url, supabase_key)
    engine.load_strategies()
    engine.run()

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python engine.py <instance_name>")
        sys.exit(1)
    if len(sys.argv) < 4:
        print("Starting engine without supabase")
        supabase_url = None
        supabase_key = None
    else:
        print("Starting engine with supabase")
        supabase_url = sys.argv[2]
        supabase_key = sys.argv[3]
    main(sys.argv[1], supabase_url, supabase_key)
