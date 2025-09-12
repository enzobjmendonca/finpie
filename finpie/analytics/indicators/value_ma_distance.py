from finpie.data.timeseries import TimeSeries
from .indicator import Indicator

class ValueMaDistance(Indicator):
    def __init__(self, timeseries: TimeSeries, window: int = 14) -> None:
        super().__init__(timeseries)
        self.window = window
    
    def calculate(self) -> TimeSeries:
        ma = self.timeseries.rolling(window=self.window).mean()
        distance = self.timeseries - ma
        return self.timeseries.__class__(distance)