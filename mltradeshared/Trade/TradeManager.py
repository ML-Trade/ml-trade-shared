from collections import deque
from datetime import datetime
from typing import Callable, Deque, List, Optional, Union, Any
import numpy as np
import math
from dataclasses import dataclass
from mltradeshared.utils import get_time_delta


@dataclass
class Trade():
    is_buy: bool
    lot_size: float
    open_time: datetime
    open_price: float
    ticket_id: Optional[int] = None
    close_time: Optional[datetime] = None
    close_price: Optional[float] = None

@dataclass
class TimeMeasurement():
    measurement: str
    multiplier: int

class TradeManager():
    def __init__(self, balance: float, forecast_period: TimeMeasurement, trade_cooldown: TimeMeasurement, risk_per_trade: float, max_open_risk: float, std_dif: float, min_stds_to_trade: float, max_trade_history: Union[int, None] = 500) -> None:
        """
        std_dif: The std deviation for the difference between the predictions
        min_stds_to_trade: Minimum number of standard deviations the difference between class predictions should be before a trade is executed
        trade_cooldown: How long we should wait before trading again immediately after a trade
        """
        if risk_per_trade > max_open_risk:
            raise Exception("risk_per_trade MUST be lower than max_open_risk... Otherwise you can't open a single trade!")
        self.balance = balance
        self.forecast_period = forecast_period
        self.risk_per_trade = risk_per_trade
        self.max_open_risk = max_open_risk
        self.min_stds_to_trade = min_stds_to_trade
        self.std_dif = std_dif
        self.open_trades: List[Trade] = []
        self.closed_trades: Deque[Trade] = deque(maxlen=max_trade_history)
        self.balance_history: Deque[float] = deque(maxlen=max_trade_history)
        self.balance_history.append(balance)
        self.trade_cooldown = get_time_delta(trade_cooldown.multiplier, trade_cooldown.measurement)
        self.last_trade_time = datetime.min

    def make_trade(self, prediction: np.ndarray, current_candle: dict, callback: Callable[[Trade], Any]):
        """
        The callback should call the api, and assign the ticket_id for the trade
        """
        is_buy = prediction[0] > prediction[1]
        open_time = datetime.fromtimestamp(current_candle["t"] / 1000)
        # TODO: Get ACTUAL open price from api (we wont otherwise account for slippage or spreads etc...)
        open_price = current_candle["c"]
        # TODO: Get ACTUAL pip size from api. Have ATR as a parameter and use 1 ATR for stop, 2 for target (or something like that)
        # lot_size = self.calculate_lot_size(open_price, stop_loss, pip_size)
        lot_size = 0.1
        trade = Trade(is_buy, lot_size, open_time, open_price)

        self.last_trade_time = open_time
        self.open_trades.append(trade)
        callback(self.open_trades[-1])

    def check_open_trades(self, current_candle: dict, closed_trades_callback: Callable[[List[Trade]], Any]):
        """
        This should be called on every single candle. This is so that we can check whether we should
        close existing trades

        Currently very simple, just closes after the forecast period

        The callback is used to handle the api calls to actually close the closed trades and is required
        """
        timestamp = current_candle["t"]
        current_time = datetime.fromtimestamp(timestamp / 1000)
        
        forecast_time_delta = get_time_delta(self.forecast_period.multiplier, self.forecast_period.measurement)

        recently_closed_trades = []
        def handle_trades(trade: Trade):
            target_time = trade.open_time + forecast_time_delta
            if current_time >= target_time:
                trade.close_time = current_time
                trade.close_price = float(current_candle["c"])
                self.closed_trades.append(trade)
                recently_closed_trades.append(trade)

                dif_price = abs(trade.close_price - trade.open_price)
                is_win = (trade.is_buy and trade.close_price > trade.open_price) or (not trade.is_buy and trade.close_price < trade.open_price)
                won_per_lot = dif_price * 100000
                if not is_win: won_per_lot *= -1
                self.set_balance(self.balance + (won_per_lot * trade.lot_size))

                return False
            return True
        
        self.open_trades = list(filter(handle_trades, self.open_trades))
        closed_trades_callback(recently_closed_trades)        
        

    def should_make_trade(self, prediction: np.ndarray, current_candle: dict) -> bool:
        max_open_trades = math.floor(self.max_open_risk / self.risk_per_trade)
        if len(self.open_trades) >= max_open_trades:
            return False

        dif = abs(prediction[0] - prediction[1])
        min_dif_to_trade = self.min_stds_to_trade * self.std_dif
        if dif < min_dif_to_trade:
            return False

        current_time = datetime.fromtimestamp(current_candle["t"] / 1000)
        next_allowed_trade_time = self.last_trade_time + self.trade_cooldown
        if current_time < next_allowed_trade_time:
            return False
            
        return True
        
    def set_balance(self, balance: float):
        self.balance = balance
        self.balance_history.append(balance)
        

    def calculate_lot_size(self, price: float, stop_loss: float, pip_size = 0.0001):
        # TODO: This is actually slightly innacurate, since as price fluctuates, pip_value changes
        pip_value = pip_size / price # For 1 lot
        dif = abs(stop_loss - price)
        dif_pips = dif / pip_size
        dif_cost_per_lot = dif_pips * pip_value
        target_cost = self.balance * self.risk_per_trade

        num_lots = target_cost / dif_cost_per_lot
        return num_lots