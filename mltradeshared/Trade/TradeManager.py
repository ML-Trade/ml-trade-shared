from collections import deque
from typing import Deque, List, Union
import numpy as np
import math

class Trade():
    pass

class TradeManager():
    def __init__(self, balance: float, risk_per_trade: float, max_open_risk: float, std_dif: float, min_stds_to_trade: float, max_trade_history: Union[int, None] = 500) -> None:
        """
        std_dif: The std deviation for the difference between the predictions
        min_stds_to_trade: Minimum number of standard deviations the difference between class predictions should be before a trade is executed
        """
        if risk_per_trade < max_open_risk:
            raise Exception("risk_per_trade MUST be lower than max_open_risk... Otherwise you can't open a single trade!")
        self.balance = balance
        self.risk_per_trade = risk_per_trade
        self.max_open_risk = max_open_risk
        self.min_stds_to_trade = min_stds_to_trade
        self.std_dif = std_dif
        self.open_trades: List[Trade] = []
        self.closed_trades: Deque[Trade] = deque(maxlen=max_trade_history)

    def make_trade(self, prediction: np.ndarray):
        trade = Trade()
        self.open_trades.append(trade)

    def should_make_trade(self, prediction: np.ndarray) -> bool:
        max_open_trades = math.floor(self.max_open_risk / self.risk_per_trade)
        if len(self.open_trades) >= max_open_trades:
            return False
        dif = abs(prediction[0] - prediction[1])
        if dif < self.min_stds_to_trade:
            return False
        return True
        
    def set_balance(self, balance: float):
        self.balance = balance
        

    def calculate_lot_size(self, price: float, stop_loss: float, pip_size = 0.0001):
        pip_value = pip_size / price # For 1 lot
        dif = abs(stop_loss - price)
        dif_pips = dif / pip_size
        dif_cost_per_lot = dif_pips * pip_value
        target_cost = self.balance * self.risk_per_trade

        num_lots = target_cost / dif_cost_per_lot
        return num_lots