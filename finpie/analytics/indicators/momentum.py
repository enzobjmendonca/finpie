from finpie.data.timeseries import TimeSeries
from .indicator import Indicator

class Momentum(Indicator):
    def __init__(self, timeseries: TimeSeries, window: int = 14) -> None:
        super().__init__(timeseries)
        self.window = window
    
    def calculate(self) -> TimeSeries:
        momentum = (self.timeseries / self.timeseries.shift(self.window)) * 100
        return self.timeseries.__class__(momentum)