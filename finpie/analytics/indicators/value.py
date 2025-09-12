from finpie.data.timeseries import TimeSeries
from .indicator import Indicator

class ValueMaDistance(Indicator):
    def __init__(self, timeseries: TimeSeries) -> None:
        super().__init__(timeseries)
    
    def calculate(self) -> TimeSeries:
        return self.timeseries