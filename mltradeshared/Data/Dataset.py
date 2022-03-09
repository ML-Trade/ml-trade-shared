from dataclasses import dataclass
from datetime import datetime
from mimetypes import init
import numpy as np
from datetime import datetime

@dataclass
class TimeMeasurement():
    measurement: str
    multiplier: int

@dataclass
class Dataset:
    train_x: np.ndarray
    train_y: np.ndarray
    val_x: np.ndarray
    val_y: np.ndarray

class DatasetMetadata:
    def __init__(self, *,
        symbol: str,
        start: datetime,
        end: datetime,
        candle_time: TimeMeasurement,
        forecast_period: TimeMeasurement,
        sequence_length: int,
        train_split: float

    ) -> None:
        self.symbol = symbol
        self.start = start
        self.end = end
        self.candle_time = candle_time
        self.forecast_period = forecast_period
        self.sequence_length = sequence_length = sequence_length
        self.train_split = train_split

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "start": self.start.timestamp(),
            "end": self.end.timestamp(),
            "candle_time": {
                "multiplier": self.candle_time.multiplier,
                "measurement": self.candle_time.measurement
            },
            "forecast_period": {
                "multiplier": self.forecast_period.multiplier,
                "measurement": self.forecast_period.measurement
            },
            "sequence_length": self.sequence_length,
            "train_split": self.train_split
        }
    
    @staticmethod
    def from_dict(meta: dict):
        return DatasetMetadata(
            symbol = meta["symbol"],
            start = datetime.fromtimestamp(meta["start"]),
            end = datetime.fromtimestamp(meta["end"]),
            candle_time = TimeMeasurement(meta["candle_time"]["measurement"], meta["candle_time"]["multiplier"]),
            forecast_period = TimeMeasurement(meta["forecast_period"]["measurement"], meta["forecast_period"]["multiplier"]),
            sequence_length = meta["sequence_length"],
            train_split = meta["train_split"]
        )