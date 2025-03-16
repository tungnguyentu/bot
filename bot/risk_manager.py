import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class RiskManager:
    """Manages risk parameters for the trading bot."""
    
    def __init__(self, sl_atr_multiplier=1.5, tp_atr_multiplier=2.0):
        """Initialize RiskManager with ATR multipliers for SL and TP."""
        self.sl_atr_multiplier = sl_atr_multiplier
        self.tp_atr_multiplier = tp_atr_multiplier
    
    def calculate_risk_params(self, df, current_position=None):
        """
        Calculate risk parameters based on ATR for a potential trade.
        
        Args:
            df: DataFrame with indicators including ATR
            current_position: Current position details if any
            
        Returns:
            Dictionary with risk parameters including SL and TP prices
        """
        if df.empty:
            logger.warning("Empty dataframe, cannot calculate risk parameters")
            return {'sl_price': None, 'tp_price': None, 'risk_reward_ratio': None}
        
        # Get the latest data point
        last_row = df.iloc[-1]
        current_price = last_row['close']
        atr = last_row['atr']
        
        # Default values
        sl_price = None
        tp_price = None
        
        # If we have a prediction or a current position, calculate risk parameters
        if current_position is not None:
            # For existing position, use the original values
            side = current_position['side']
            sl_price = current_position['sl_price']
            tp_price = current_position['tp_price']
        else:
            # For potential new positions, use predicted direction based on indicators
            side = self._determine_side(last_row)
            
            # Calculate SL and TP based on ATR
            if side == 'BUY':
                sl_price = current_price - (atr * self.sl_atr_multiplier)
                tp_price = current_price + (atr * self.tp_atr_multiplier)
            elif side == 'SELL':
                sl_price = current_price + (atr * self.sl_atr_multiplier)
                tp_price = current_price - (atr * self.tp_atr_multiplier)
        
        # Calculate risk-reward ratio
        risk_reward_ratio = None
        if sl_price is not None and tp_price is not None and current_price:
            if side == 'BUY':
                risk = current_price - sl_price
                reward = tp_price - current_price
            else:  # SELL
                risk = sl_price - current_price
                reward = current_price - tp_price
            
            risk_reward_ratio = reward / risk if risk > 0 else 0
        
        return {
            'side': side,
            'sl_price': sl_price,
            'tp_price': tp_price,
            'risk_reward_ratio': risk_reward_ratio,
            'atr': atr
        }
    
    def _determine_side(self, row):
        """
        Determine trade direction based on technical indicators.
        
        Args:
            row: Series with the latest indicator values
            
        Returns:
            'BUY', 'SELL', or None
        """
        buy_signals = 0
        sell_signals = 0
        
        # Bollinger Bands
        if row['close'] < row['bb_lower']:
            buy_signals += 1
        elif row['close'] > row['bb_upper']:
            sell_signals += 1
        
        # RSI
        if row['rsi'] < 30:
            buy_signals += 1
        elif row['rsi'] > 70:
            sell_signals += 1
        
        # MACD
        if row['macd'] > row['macd_signal'] and row['macd'] > 0:
            buy_signals += 1
        elif row['macd'] < row['macd_signal'] and row['macd'] < 0:
            sell_signals += 1
        
        # VWAP
        if row['close'] < row['vwap']:
            buy_signals += 1
        elif row['close'] > row['vwap']:
            sell_signals += 1
        
        # Determine the side with the most signals
        if buy_signals > sell_signals:
            return 'BUY'
        elif sell_signals > buy_signals:
            return 'SELL'
        else:
            return None
    
    def adjust_position_size(self, account_balance, entry_price, sl_price, max_risk_pct=0.02):
        """
        Calculate position size based on account balance and risk tolerance.
        
        Args:
            account_balance: Current account balance
            entry_price: Entry price for the trade
            sl_price: Stop-loss price
            max_risk_pct: Maximum percentage of account balance to risk (default 2%)
            
        Returns:
            Recommended position size
        """
        if entry_price == sl_price:
            logger.warning("Entry price and stop-loss price are the same, cannot calculate position size")
            return 0
        
        # Calculate the risk per unit
        risk_per_unit = abs(entry_price - sl_price)
        
        # Calculate maximum acceptable loss in account currency
        max_loss = account_balance * max_risk_pct
        
        # Calculate position size
        position_size = max_loss / risk_per_unit
        
        return position_size
