from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Dict


@dataclass
class Data(ABC):

    @abstractmethod
    def reset(self): pass

    def update(self, data: dict):
        for key, value in data.items():
            # Only update attribute if value has changed
            if getattr(self, key, None) != value:
                setattr(self, key, value)
                if key != 'time':
                    self.has_changed = True

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
class StrategyData(Data):
    name: str
    symbol: str
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
    has_changed: bool = False
    time: datetime = datetime.now()

    def reset(self):
        self.is_trading = False
        self.position = 0
        self.delta = 0
        self.volume = 0
        self.buy_volume = 0
        self.sell_volume = 0
        self.buy_price = 0
        self.sell_price = 0
        self.buy_notional = 0
        self.sell_notional = 0
        self.pnl = 0
        self.signal = 0
        self.time = datetime.now()
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


@dataclass
class CRBParams:
    magic: int
    symbol_map: Dict[str, Dict[str, str]]
    multiplier: float
    start_time: str
    end_time: str

@dataclass
class CRBData(Data):
    symbol: str
    position: float = 0
    volume: float = 0
    target_position: float = 0
    buy_volume: float = 0
    sell_volume: float = 0
    buy_notional: float = 0
    sell_notional: float = 0
    buy_price: float = 0
    sell_price: float = 0
    pnl: float = 0
    has_changed: bool = False
    time: datetime = datetime.now()
    is_trading: bool = False

    def reset(self):
        self.position = 0
        self.volume = 0
        self.target_position = 0
        self.delta = 0
        self.buy_volume = 0
        self.sell_volume = 0
        self.buy_notional = 0
        self.sell_notional = 0
        self.buy_price = 0
        self.sell_price = 0
        self.pnl = 0
        self.has_changed = True
        self.time = datetime.now()
