from dataclasses import dataclass
import datetime
import math
from typing import Dict
from finpie.strategy.domain import Domain
import MetaTrader5 as mt5
import pandas as pd
import numpy as np

@dataclass
class CRBParams:
    magic: int
    symbol_map: Dict[str, str]
    multiplier: float

class CRB:
    def __init__(self, domain: Domain, params: CRBParams):
        self.domain = domain
        self.params = params
        if not mt5.initialize():
            raise RuntimeError("MT5 is not initialized")

    def get_position(self, symbol: str, magic: int) -> float:
        """
        Track the current net position for a given symbol and magic number.

        Args:
            symbol (str): The trading symbol to check.
            magic (int): The magic number identifying the strategy.

        Returns:
            float: The net position (sum of volumes for all open positions).
        """
        positions = mt5.positions_get(symbol=symbol)
        if positions is None:
            return 0.0
        net_position = 0.0
        for pos in positions:
            if pos.magic == magic:
                if pos.type == mt5.POSITION_TYPE_BUY:
                    net_position += pos.volume
                elif pos.type == mt5.POSITION_TYPE_SELL:
                    net_position -= pos.volume
        return net_position
        
    def react(self):
        target_position = pd.DataFrame(self.domain.strategy_data).groupby('symbol').agg({'position': 'sum'})['position']
        target_position = target_position * self.params.multiplier
        for symbol, position in target_position.items():
            if symbol in self.params.symbol_map:
                symbol = self.params.symbol_map[symbol]
            current_position = self.get_position(symbol, self.params.magic)
            if current_position != position:
                delta = position - current_position
                delta = float(abs(round(delta)))
                print(f"Delta: {delta}, Target Position: {position}, Current Position: {current_position}")
                if delta >= 1 and position > current_position:
                    self.buy(symbol, delta)
                elif delta >= 1 and position < current_position:
                    self.sell(symbol, delta)
        
    def buy(self, symbol: str, size: float):
        tick_info = mt5.symbol_info_tick(symbol)
        spread = tick_info.ask - tick_info.bid
        price = tick_info.ask + (2 * spread)
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": size,
            "type": mt5.ORDER_TYPE_BUY,
            "price": price,
            "magic": self.params.magic,
            "comment": "CRB",
            "type_filling": mt5.ORDER_FILLING_FOK,
        }  
        mt5.order_send(request)
        
    def sell(self, symbol: str, size: float):
        tick_info = mt5.symbol_info_tick(symbol)
        spread = tick_info.ask - tick_info.bid
        price = tick_info.bid - (2 * spread)
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": size,
            "type": mt5.ORDER_TYPE_SELL,
            "price": price,
            "magic": self.params.magic,
            "comment": "CRB",
            "type_filling": mt5.ORDER_FILLING_FOK,
        }   
        mt5.order_send(request)

