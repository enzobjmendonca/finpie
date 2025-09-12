from finpie.data.timeseries import TimeSeries
import talib
from .indicator import Indicator

class Rsi(Indicator):
    def __init__(self, timeseries: TimeSeries, window: int = 14) -> None:
        super().__init__(timeseries)
        self.window = window
    
    def calculate(self) -> TimeSeries:
        rsi = []
        for col in self.timeseries.columns:
            col_rsi = talib.RSI(self.timeseries[col], timeperiod=self.window)
            col_rsi.rename(col, inplace=True)
            rsi.append(col_rsi)
        return self.timeseries.__class__(rsi)