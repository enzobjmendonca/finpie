from finpie.data.timeseries import TimeSeries
from .indicator import Indicator

class MaMaDistance(Indicator):
    def __init__(self, timeseries: TimeSeries, window_1: int = 14, window_2: int = 20) -> None:
        super().__init__(timeseries)
        self.window_1 = window_1
        self.window_2 = window_2
    
    def calculate(self) -> TimeSeries:
        ma_1 = self.timeseries.rolling(window=self.window_1).mean()
        ma_2 = self.timeseries.rolling(window=self.window_2).mean()
        distance = ma_1 - ma_2
        return self.timeseries.__class__(distance)