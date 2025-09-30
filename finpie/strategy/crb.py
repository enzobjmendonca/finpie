from datetime import datetime, timedelta
import math
from typing import Dict
from uuid import uuid4
from finpie.strategy.data import CRBData, CRBParams, Fill
from finpie.strategy.domain import Domain
import MetaTrader5 as mt5
import pandas as pd
import numpy as np

class CRB:
    def __init__(self, domain: Domain, params: CRBParams):
        self.domain = domain
        self.params = params
        self.start_hour, self.start_minute = map(int, self.params.start_time.split(':'))
        self.end_hour, self.end_minute = map(int, self.params.end_time.split(':'))
        self.crb_data_map = {}
        self.processed_fills = set()
        self.start_time = self.domain.get_market_time()
        if not mt5.initialize():
            raise RuntimeError("MT5 is not initialized")

    def in_trading_window(self):
        now = self.domain.get_market_time()

        current_time = now.hour * 60 + now.minute
        start_time = self.start_hour * 60 + self.start_minute
        end_time = self.end_hour * 60 + self.end_minute
        
        return start_time <= current_time <= end_time

    def update_data(self, symbol: str, magic: int):
        to_date=self.domain.get_market_time()
        from_date= to_date - timedelta(days=1)
        deals=mt5.history_deals_get(from_date, to_date, group=f"*{symbol}*")
        for deal in deals:
            if deal.ticket in self.processed_fills or pd.to_datetime(deal.time, unit='s') < self.start_time:
                continue
            if deal.magic == magic:
                fill = Fill(str(uuid4()), "CRB", self.domain.get_market_time(), symbol, deal.volume, deal.price, "buy" if deal.type == mt5.DEAL_TYPE_BUY else "sell", "CRB")
                self.process_fill(fill)
                self.processed_fills.add(deal.ticket)
        if symbol not in self.crb_data_map:
            self.crb_data_map[symbol] = CRBData(symbol=symbol)
        data = self.crb_data_map[symbol]
        data.update({'pnl': self.calc_pnl(data), 'time': self.domain.get_market_time()})
        return data
    
    def calc_pnl(self, data: CRBData):
        tick_info = mt5.symbol_info_tick(data.symbol)
        reference_price = (tick_info.ask + tick_info.bid) / 2
        return (data.sell_notional - data.buy_notional) + (data.position * reference_price)

    def process_fill(self, fill: Fill):
        if fill.symbol not in self.crb_data_map:
            self.crb_data_map[fill.symbol] = CRBData(symbol=fill.symbol)
        data = self.crb_data_map[fill.symbol]
        contract_size = 1
        if fill.symbol in self.params.symbol_map:
            contract_size = self.params.symbol_map[fill.symbol]['contract_size']
        data.update(
            {
                'position': data.position + fill.size * (1 if fill.side == 'buy' else -1),
                'volume': data.volume + fill.size,
                'buy_volume': data.buy_volume + fill.size if fill.side == 'buy' else data.buy_volume,
                'sell_volume': data.sell_volume + fill.size if fill.side == 'sell' else data.sell_volume,
                'buy_notional': data.buy_notional + fill.size * fill.price * contract_size if fill.side == 'buy' else data.buy_notional,
                'sell_notional': data.sell_notional + fill.size * fill.price * contract_size if fill.side == 'sell' else data.sell_notional,
                'buy_price': data.buy_notional / data.buy_volume if data.buy_volume > 0 else 0,
                'sell_price': data.sell_notional / data.sell_volume if data.sell_volume > 0 else 0,
            }
        )
        self.domain.add_fill(fill)

    def react(self):
        multiplier = 1
        if symbol in self.params.symbol_map:
            multiplier = self.params.symbol_map[symbol]['multiplier']
        target_positions = pd.DataFrame(self.domain.strategy_data).groupby('symbol').agg({'position': 'sum'})['position'] * multiplier
        for symbol, target_position in target_positions.items():
            if symbol in self.params.symbol_map:
                symbol = self.params.symbol_map[symbol]['symbol']
            data = self.update_data(symbol, self.params.magic)
            data.update({'target_position': target_position})
            if data.position != data.target_position:
                delta = data.target_position - data.position
                delta = round(delta)
                print(f"Delta: {delta}, Target Position: {data.target_position}, Current Position: {data.position}")
                if delta >= 1:
                    print(f"Buying {delta} of {symbol}")
                    self.buy(symbol, delta)
                elif delta <= -1:
                    print(f"Selling {delta} of {symbol}")
                    self.sell(symbol, abs(delta))
        
    def buy(self, symbol: str, size: float):
        tick_info = mt5.symbol_info_tick(symbol)
        spread = tick_info.ask - tick_info.bid
        price = tick_info.ask + (2 * spread)
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(size),
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
            "volume": float(size),
            "type": mt5.ORDER_TYPE_SELL,
            "price": price,
            "magic": self.params.magic,
            "comment": "CRB",
            "type_filling": mt5.ORDER_FILLING_FOK,
        }   
        mt5.order_send(request)

