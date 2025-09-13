from finpie.data.multitimeseries import MultiTimeSeries
from finpie.data.timeseries import TimeSeries
from .indicator import Indicator
import pandas as pd

class TradeImbalance(Indicator):
    def __init__(self, timeseries: MultiTimeSeries, window: int = 5, bid_col: str = 'bid', ask_col: str = 'ask', volume_col: str = 'volume', last_col: str = 'last') -> None:
        super().__init__(timeseries)
        self.window = window
        self.bid_col = bid_col
        self.ask_col = ask_col
        self.volume_col = volume_col
        self.last_col = last_col
    
    def calculate(self) -> TimeSeries:
        on_bid = self.timeseries[self.volume_col] * (self.timeseries[self.last_col] <= self.timeseries[self.bid_col])
        on_ask = self.timeseries[self.volume_col] * (self.timeseries[self.last_col] >= self.timeseries[self.ask_col])

        bid_ask_volume = pd.DataFrame({'on_bid': on_bid, 'on_ask': on_ask})
        bid_ask_volume['minute'] = bid_ask_volume.index.floor('min')
        bid_ask_volume = bid_ask_volume.groupby('minute').sum()
        
        bid_ask_volume = bid_ask_volume.groupby('minute').sum()
        bid_ask_volume = bid_ask_volume.rolling(window=self.window).sum()
        bid_ask_volume['imbalance'] = bid_ask_volume['on_ask'] / (bid_ask_volume['on_ask'] + bid_ask_volume['on_bid'])
        return TimeSeries(bid_ask_volume['imbalance'])

