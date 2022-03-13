from collections import deque
from datetime import datetime
from typing import Callable, Deque, List, Optional, Union, Any
import numpy as np
import math
from dataclasses import dataclass
from mltradeshared.utils import get_time_delta
from mltradeshared.Data import TimeMeasurement


@dataclass
class Trade():
    is_buy: bool
    lot_size: float
    open_time: datetime
    open_price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    ticket_id: Optional[int] = None
    close_time: Optional[datetime] = None
    close_price: Optional[float] = None


class TradeManager():
    def __init__(self, *,
        balance: float,
        max_trade_time: TimeMeasurement,
        trade_cooldown: TimeMeasurement,
        risk_per_trade: float,
        max_open_risk: float,
        dif_percentiles: List[float],
        fraction_to_trade: float,
        stop_loss_ATR: float,
        take_profit_ATR: float,
        max_trade_history: Union[int, None] = 500
    ) -> None:
        """
        std_dif: The std deviation for the difference between the predictions
        min_stds_to_trade: Minimum number of standard deviations the difference between class predictions should be before a trade is executed
        trade_cooldown: How long we should wait before trading again immediately after a trade
        fraction_to_trade: What fraction of best trades to take. For example, if this is 0.05, take only the top 5% of trades]
            (with highest dif between buy and sell prediction therefore highest classification confidence) 

        """
        if risk_per_trade > max_open_risk:
            raise Exception("risk_per_trade MUST be lower than max_open_risk... Otherwise you can't open a single trade!")
        self.balance = balance
        self.max_trade_time = max_trade_time
        self.risk_per_trade = risk_per_trade
        self.max_open_risk = max_open_risk
        self.fraction_to_trade = fraction_to_trade
        self.dif_percentiles = dif_percentiles
        self.open_trades: List[Trade] = []
        self.closed_trades: Deque[Trade] = deque(maxlen=max_trade_history)
        self.balance_history: Deque[float] = deque(maxlen=max_trade_history)
        self.balance_history.append(balance)
        self.trade_cooldown = get_time_delta(trade_cooldown.multiplier, trade_cooldown.measurement)
        self.last_trade_time = datetime.min
        self.stop_loss_ATR = stop_loss_ATR
        self.take_profit_ATR = take_profit_ATR

    def _calc_targets(self, open_price: float, is_buy: bool, ATR: float):
        stop_dif = ATR * self.stop_loss_ATR
        tp_dif = ATR * self.take_profit_ATR
        stop_loss, take_profit = 0.0, 0.0
        if is_buy:
            stop_loss = open_price - stop_dif
            take_profit = open_price + tp_dif
        else:
            stop_loss = open_price + stop_dif
            take_profit = open_price - tp_dif
        return stop_loss, take_profit

    def make_trade(self, prediction: np.ndarray, current_candle: dict, callback: Callable[[Trade], Any], *, ATR: float, pip_size = 0.0001):
        """
        The callback should call the api, and assign the ticket_id for the trade
        """
        is_buy = prediction[0] > prediction[1]
        open_time = datetime.fromtimestamp(current_candle["t"])
        # TODO: Get ACTUAL open price from api (we wont otherwise account for slippage or spreads etc...)
        open_price = current_candle["c"]
        # TODO: Get ACTUAL pip size from api. Have ATR as a parameter and use 1 ATR for stop, 2 for target (or something like that)
        stop_loss, take_profit = self._calc_targets(open_price, is_buy, ATR)
        lot_size = self.calculate_lot_size(open_price, stop_loss, pip_size)
        trade = Trade(is_buy, lot_size, open_time, open_price, stop_loss=stop_loss, take_profit=take_profit)

        self.last_trade_time = open_time
        self.open_trades.append(trade)
        callback(self.open_trades[-1])


    def _check_targets(self, current_candle: dict, trade: Trade):
        hit_stop, hit_tp = False, False
        if trade.stop_loss is not None:
            hit_stop = current_candle["h"] > trade.stop_loss and current_candle["l"] < trade.stop_loss
        if trade.take_profit is not None:
            hit_tp = current_candle["h"] > trade.take_profit and current_candle["l"] < trade.take_profit
        if hit_stop and hit_tp: # We must guess which was hit first
            if current_candle["o"] < current_candle["c"]:
                if trade.is_buy: hit_tp = False
                else: hit_stop = False
            else:
                if trade.is_buy: hit_stop = False
                else: hit_tp = False
        return hit_stop, hit_tp


    def check_open_trades(self, current_candle: dict, closed_trades_callback: Callable[[List[Trade]], Any], pip_size = 0.0001):
        """
        This should be called on every single candle. This is so that we can check whether we should
        close existing trades

        Currently very simple, just closes after the forecast period

        The callback is used to handle the api calls to actually close the closed trades and is required
        """
        timestamp = current_candle["t"]
        current_time = datetime.fromtimestamp(timestamp)
        
        max_trade_time_delta = get_time_delta(self.max_trade_time.multiplier, self.max_trade_time.measurement)

        recently_closed_trades = []
        def handle_trades(trade: Trade):
            target_time = trade.open_time + max_trade_time_delta
            hit_stop, hit_tp = self._check_targets(current_candle, trade)
            if hit_stop:
                hit_stop, hit_tp = self._check_targets(current_candle, trade)
            if current_time >= target_time or hit_tp or hit_stop:
                trade.close_time = current_time
                trade.close_price = float(current_candle["c"])
                if hit_tp:
                    trade.close_price = trade.take_profit or trade.close_price
                if hit_stop:
                    trade.close_price = trade.stop_loss or trade.close_price
                self.closed_trades.append(trade)
                recently_closed_trades.append(trade)

                dif_price = abs(trade.close_price - trade.open_price)
                is_win = (trade.is_buy and trade.close_price > trade.open_price) or (not trade.is_buy and trade.close_price < trade.open_price)
                dif_pips = dif_price / pip_size
                pip_value = pip_size / trade.close_price * 100000
                won_per_lot = dif_pips * pip_value
                print(f"{'WINNER' if is_win else 'LOSER'} and we {'HIT STOP' if hit_stop else ''} + {'HIT TP' if hit_tp else ''}")
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

        index = (1 - self.fraction_to_trade) * len(self.dif_percentiles)
        min_dif_to_trade = 0.0
        if int(index) == index:
            min_dif_to_trade = self.dif_percentiles[int(index)]
        else:
            lower_index, upper_index = math.floor(index), math.ceil(index)
            frac = index - lower_index
            lower, upper = self.dif_percentiles[lower_index], self.dif_percentiles[upper_index]
            min_dif_to_trade = lower + (frac * (upper - lower)) # Interpolation bitch
        
        if dif < min_dif_to_trade:
            return False

        current_time = datetime.fromtimestamp(current_candle["t"])
        next_allowed_trade_time = self.last_trade_time + self.trade_cooldown
        if current_time < next_allowed_trade_time:
            return False
            
        return True
        
    def set_balance(self, balance: float):
        self.balance = balance
        self.balance_history.append(balance)
        

    def calculate_lot_size(self, price: float, stop_loss: float, pip_size = 0.0001):
        # TODO: This is actually slightly innacurate, since as price fluctuates, pip_value changes
        pip_value = pip_size / price * 100000 # For 1 lot
        dif = abs(stop_loss - price)
        dif_pips = dif / pip_size
        dif_cost_per_lot = dif_pips * pip_value
        target_cost = self.balance * self.risk_per_trade

        num_lots = target_cost / dif_cost_per_lot
        return num_lots