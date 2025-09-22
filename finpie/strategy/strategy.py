from abc import ABC, abstractmethod
from dataclasses import dataclass

from finpie.strategy.domain import Domain
from datetime import datetime
import traceback
from uuid import uuid4

@dataclass
class StrategyParams:
    instance: str
    name: str
    symbol: str
    order_size: float
    max_position: float
    max_volume: float
    take_profit: float
    stop_loss: float
    start_time: str
    end_time: str
    signal_params: dict
    signal_threshold: float
    signal_direction: int # 1 for long, -1 for short, 0 for neutral

@dataclass
class StrategyData:
    name: str
    is_trading: bool = False
    position: float = 0
    delta: float = 0
    volume: float = 0
    buy_volume: float = 0
    sell_volume: float = 0
    buy_price: float = 0
    sell_price: float = 0
    buy_notional: float = 0
    sell_notional: float = 0
    pnl: float = 0
    signal: float = 0
    time: datetime = datetime.now()
    has_changed: bool = False

    def reset(self):
        self.position = 0
        self.delta = 0
        self.volume = 0
        self.pnl = 0
        self.signal = 0
        self.buy_volume = 0
        self.sell_volume = 0
        self.buy_notional = 0
        self.sell_notional = 0
        self.buy_price = 0
        self.sell_price = 0
    
    def update(self, data: dict):
        for key, value in data.items():
            # Only update attribute if value has changed
            if getattr(self, key, None) != value:
                setattr(self, key, value)
                if key != 'time':
                    self.has_changed = True
@dataclass
class Fill:
    id: str
    name: str
    time: datetime
    symbol: str
    size: float
    price: float
    side: str
    comment: str

class Strategy:
    def __init__(self, domain: Domain, params: StrategyParams):
        self.params = params
        self.data = StrategyData(name=params.name)
        self.domain = domain
        self.domain.add_strategy_data(self.data)
        self.start_hour, self.start_minute = map(int, self.params.start_time.split(':'))
        self.end_hour, self.end_minute = map(int, self.params.end_time.split(':'))
        self.last_tick_time = None
        self.last_day = None
        self.signal_service = domain.get_service('signal_service')
        self.market_data_service = domain.get_service('market_data_service')
        self.signal_error_count = 0

    def react(self):
        try:
            if self.in_trading_window():
                if not self.data.is_trading:
                    self.check_new_day()
                if self.data.is_trading:
                    self.update_data()
                    if self.data.pnl > self.params.take_profit:
                        self.stop("Take Profit")
                    elif self.data.pnl < self.params.stop_loss:
                        self.stop("Stop Loss")
                    elif self.is_first_tick() and self.params.order_size + self.data.volume <= self.params.max_volume:
                        self.data.update({'signal': self.calc_signal()})
                        if self.data.signal * self.params.signal_direction >= self.params.signal_threshold and \
                            self.data.position + self.params.order_size <= self.params.max_position:
                            self.buy()
                        elif self.data.signal * self.params.signal_direction <= -self.params.signal_threshold and \
                            self.data.position - self.params.order_size >= -self.params.max_position:
                            self.sell()
            else:
                self.stop("Out of Trading Window")
        except Exception as e:
            print(self.params.name)
            traceback.print_exc()
            self.stop("Error in react")
    
    def buy(self):
        fill_price = self.get_price() + self.market_data_service.get_spread(self.params.symbol)
        fill = Fill(str(uuid4()), self.params.name, self.domain.get_market_time(), self.params.symbol, self.params.order_size, fill_price, 'buy', str(self.data.signal))
        self.process_fill(fill)

    def sell(self):
        fill_price = self.get_price() - self.market_data_service.get_spread(self.params.symbol)
        fill = Fill(str(uuid4()), self.params.name, self.domain.get_market_time(), self.params.symbol, self.params.order_size, fill_price, 'sell', str(self.data.signal))
        self.process_fill(fill)
    
    def process_fill(self, fill: Fill):
        self.data.update(
            {
                'position': self.data.position + fill.size * (1 if fill.side == 'buy' else -1),
                'volume': self.data.volume + fill.size,
                'buy_volume': self.data.buy_volume + fill.size if fill.side == 'buy' else self.data.buy_volume,
                'sell_volume': self.data.sell_volume + fill.size if fill.side == 'sell' else self.data.sell_volume,
                'buy_notional': self.data.buy_notional + fill.size * fill.price if fill.side == 'buy' else self.data.buy_notional,
                'sell_notional': self.data.sell_notional + fill.size * fill.price if fill.side == 'sell' else self.data.sell_notional,
                'buy_price': self.data.buy_notional / self.data.buy_volume if self.data.buy_volume > 0 else 0,
                'sell_price': self.data.sell_notional / self.data.sell_volume if self.data.sell_volume > 0 else 0,
            }
        )
        self.domain.add_fill(fill)
    
    def calc_signal(self):
        try:
            signal_series = self.signal_service.get_signal(self.params.signal_params)
            return signal_series.fillna(0).iloc[-1][signal_series.columns[0]]
        except Exception as e:
            self.signal_error_count += 1
            print(f"Error calculating signal for strategy {self.params.name}, error count: {self.signal_error_count}")
            if self.signal_error_count > 10:
                self.stop("Error calculating signal")
            return 0
    
    def check_new_day(self):
        now = self.domain.get_market_time()
        new_day = now.date() != self.last_day
        self.last_day = now.date()
        if new_day:
            self.start()
        
    def is_first_tick(self):
        now = self.domain.get_market_time()
        now_tick = now.timestamp() // 60
        is_first_tick = now_tick != self.last_tick_time
        self.last_tick_time = now_tick
        return is_first_tick

    def in_trading_window(self):
        now = self.domain.get_market_time()

        current_time = now.hour * 60 + now.minute
        start_time = self.start_hour * 60 + self.start_minute
        end_time = self.end_hour * 60 + self.end_minute
        
        return start_time <= current_time <= end_time

    def stop(self, reason: str):
        print(f"Stopping strategy {self.params.name} because {reason}")
        if self.data.is_trading:
            fill = Fill(str(uuid4()), self.params.name, self.domain.get_market_time(), 
                                self.params.symbol, abs(self.data.position), self.get_price(), 
                                'buy' if self.data.position < 0 else 'sell', reason)
            self.process_fill(fill)
            self.domain.report_eod(self.data)
            self.data.update({'is_trading': False})

    def start(self):
        print(f"Starting strategy {self.params.name}")
        self.signal_error_count = 0
        self.data.reset()
        self.data.update({'is_trading': True})

    def update_data(self):
        self.data.update(
            {
                'pnl': self.calc_pnl(),
                'delta': self.data.position * self.get_price(),
                'time': self.domain.get_market_time()
            }
        )
        

    def calc_pnl(self):
        return (self.data.sell_notional - self.data.buy_notional) + (self.data.position * self.get_price())

    def get_price(self):
        return self.market_data_service.get_price(self.params.symbol, self.domain.get_market_time(), source='mt5')

    