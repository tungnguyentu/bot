import pandas as pd
import numpy as np
import os
import sys
import logging

# Add the parent directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from indicators.technical_indicators import TechnicalIndicators

logger = logging.getLogger(__name__)

class ScalpingStrategy:
    def __init__(self):
        """Initialize the Scalping Strategy."""
        self.name = "Scalping"
        self.timeframe = config.TRADING_TIMEFRAMES["scalping"]
        self.description = "Short-term strategy using RSI, Bollinger Bands, and Moving Averages"
        
    def analyze(self, df):
        """
        Analyze market data using scalping strategy indicators.
        
        Args:
            df: DataFrame with OHLCV data and indicators
            
        Returns:
            dict: Analysis results with signals and reasoning
        """
        if len(df) < 30:  # Need enough data for indicators
            return {"signal": "neutral", "strength": 0, "reasoning": "Insufficient data for analysis"}
        
        # Ensure all required indicators are calculated
        df = TechnicalIndicators.add_scalping_indicators(df)
        
        # Get the latest candle for analysis
        latest = df.iloc[-1]
        previous = df.iloc[-2]
        
        # Initialize signal parameters
        signal = "neutral"
        strength = 0
        reasons = []
        
        # ========== LONG SIGNALS ==========
        long_signals = 0
        
        # RSI oversold and rising
        if latest['rsi'] < config.SCALPING_RSI_OVERSOLD and latest['rsi'] > previous['rsi']:
            long_signals += 2
            reasons.append(f"RSI oversold and rising ({latest['rsi']:.2f})")
        
        # Price near or below lower Bollinger Band
        if latest['close'] <= latest['bb_lower'] * 1.005:
            long_signals += 2
            reasons.append(f"Price at/below lower Bollinger Band (Price: {latest['close']:.2f}, BB: {latest['bb_lower']:.2f})")
        
        # EMA crossover (short crossing above long)
        if previous['ema_short'] < previous['ema_long'] and latest['ema_short'] > latest['ema_long']:
            long_signals += 3
            reasons.append(f"EMA crossover: short crossed above long")
        
        # MACD histogram turning positive
        if previous['macd_hist'] < 0 and latest['macd_hist'] > 0:
            long_signals += 1
            reasons.append(f"MACD histogram turned positive")
            
        # Bullish candlestick patterns
        if latest['engulfing_bullish'] > 0 or latest['hammer'] > 0 or latest['morning_star'] > 0:
            long_signals += 1
            reasons.append(f"Bullish candlestick pattern detected")
        
        # ========== SHORT SIGNALS ==========
        short_signals = 0
        
        # RSI overbought and falling
        if latest['rsi'] > config.SCALPING_RSI_OVERBOUGHT and latest['rsi'] < previous['rsi']:
            short_signals += 2
            reasons.append(f"RSI overbought and falling ({latest['rsi']:.2f})")
        
        # Price near or above upper Bollinger Band
        if latest['close'] >= latest['bb_upper'] * 0.995:
            short_signals += 2
            reasons.append(f"Price at/above upper Bollinger Band (Price: {latest['close']:.2f}, BB: {latest['bb_upper']:.2f})")
        
        # EMA crossover (short crossing below long)
        if previous['ema_short'] > previous['ema_long'] and latest['ema_short'] < latest['ema_long']:
            short_signals += 3
            reasons.append(f"EMA crossover: short crossed below long")
        
        # MACD histogram turning negative
        if previous['macd_hist'] > 0 and latest['macd_hist'] < 0:
            short_signals += 1
            reasons.append(f"MACD histogram turned negative")
            
        # Bearish candlestick patterns
        if latest['engulfing_bearish'] < 0 or latest['hanging_man'] < 0 or latest['evening_star'] < 0:
            short_signals += 1
            reasons.append(f"Bearish candlestick pattern detected")
        
        # ========== SIGNAL DETERMINATION ==========
        # Minimum threshold for a valid signal is 4 points
        threshold = 4
        
        if long_signals >= threshold and long_signals > short_signals:
            signal = "buy"
            strength = min(long_signals / 10, 1.0)  # Scale between 0 and 1
        elif short_signals >= threshold and short_signals > long_signals:
            signal = "sell"
            strength = min(short_signals / 10, 1.0)  # Scale between 0 and 1
        
        # Return analysis results
        return {
            "signal": signal,
            "strength": strength,
            "reasoning": "; ".join(reasons) if reasons else "No significant signals detected"
        }
    
    def get_take_profit_price(self, entry_price, side, atr=None):
        """Calculate take profit price based on ATR or fixed percentage."""
        if side == "buy":
            if atr:
                return entry_price + (atr * 1.5)  # 1.5 times ATR for take profit
            return entry_price * 1.01  # 1% default take profit
        elif side == "sell":
            if atr:
                return entry_price - (atr * 1.5)
            return entry_price * 0.99  # 1% default take profit
        return entry_price
        
    def get_stop_loss_price(self, entry_price, side, atr=None):
        """Calculate stop loss price based on ATR or fixed percentage."""
        if side == "buy":
            if atr:
                return entry_price - (atr * 1)  # 1 times ATR for stop loss
            return entry_price * 0.995  # 0.5% default stop loss
        elif side == "sell":
            if atr:
                return entry_price + (atr * 1)
            return entry_price * 1.005  # 0.5% default stop loss
        return entry_price
