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

class SwingStrategy:
    def __init__(self):
        """Initialize the Swing Trading Strategy."""
        self.name = "Swing Trading"
        self.timeframe = config.TRADING_TIMEFRAMES["swing"]
        self.description = "Medium-term strategy using Ichimoku Cloud, MACD, and ADX"
        
    def analyze(self, df):
        """
        Analyze market data using swing trading strategy indicators.
        
        Args:
            df: DataFrame with OHLCV data and indicators
            
        Returns:
            dict: Analysis results with signals and reasoning
        """
        if len(df) < 80:  # Need enough data for Ichimoku calculations
            return {"signal": "neutral", "strength": 0, "reasoning": "Insufficient data for analysis"}
        
        # Ensure all required indicators are calculated
        df = TechnicalIndicators.add_swing_indicators(df)
        
        # Get the latest candles for analysis
        latest = df.iloc[-1]
        previous = df.iloc[-2]
        
        # Initialize signal parameters
        signal = "neutral"
        strength = 0
        reasons = []
        
        # ========== LONG SIGNALS ==========
        long_signals = 0
        
        # Ichimoku signals
        if latest['close'] > latest['ichimoku_span_a'] and latest['close'] > latest['ichimoku_span_b']:
            # Price above cloud (bullish)
            long_signals += 2
            reasons.append(f"Price above Ichimoku cloud")
            
            if latest['ichimoku_conversion'] > latest['ichimoku_base']:
                # Bullish TK cross
                long_signals += 1
                reasons.append(f"Bullish Ichimoku TK cross")
        
        # MACD signals
        if previous['macd'] < previous['macd_signal'] and latest['macd'] > latest['macd_signal']:
            # MACD crossed above signal line
            long_signals += 2
            reasons.append(f"MACD crossed above signal line")
        
        # ADX signal (strong trend with positive DI)
        if latest['adx'] > 25 and latest['di_plus'] > latest['di_minus']:
            long_signals += 2
            reasons.append(f"Strong uptrend: ADX={latest['adx']:.2f} with +DI>{'-DI'}")
            
        # Volume confirmation
        if latest['volume'] > latest['volume_ma'] * 1.2:
            long_signals += 1
            reasons.append(f"Volume spike: {latest['volume']:.2f} > {latest['volume_ma']:.2f}")
        
        # RSI not overbought
        if latest['rsi'] > 40 and latest['rsi'] < 70:
            long_signals += 1
            reasons.append(f"RSI in positive zone but not overbought: {latest['rsi']:.2f}")
        
        # ========== SHORT SIGNALS ==========
        short_signals = 0
        
        # Ichimoku signals
        if latest['close'] < latest['ichimoku_span_a'] and latest['close'] < latest['ichimoku_span_b']:
            # Price below cloud (bearish)
            short_signals += 2
            reasons.append(f"Price below Ichimoku cloud")
            
            if latest['ichimoku_conversion'] < latest['ichimoku_base']:
                # Bearish TK cross
                short_signals += 1
                reasons.append(f"Bearish Ichimoku TK cross")
        
        # MACD signals
        if previous['macd'] > previous['macd_signal'] and latest['macd'] < latest['macd_signal']:
            # MACD crossed below signal line
            short_signals += 2
            reasons.append(f"MACD crossed below signal line")
        
        # ADX signal (strong trend with negative DI)
        if latest['adx'] > 25 and latest['di_minus'] > latest['di_plus']:
            short_signals += 2
            reasons.append(f"Strong downtrend: ADX={latest['adx']:.2f} with -DI>+DI")
            
        # Volume confirmation
        if latest['volume'] > latest['volume_ma'] * 1.2:
            short_signals += 1
            reasons.append(f"Volume spike: {latest['volume']:.2f} > {latest['volume_ma']:.2f}")
        
        # RSI not oversold
        if latest['rsi'] < 60 and latest['rsi'] > 30:
            short_signals += 1
            reasons.append(f"RSI in negative zone but not oversold: {latest['rsi']:.2f}")
        
        # ========== SIGNAL DETERMINATION ==========
        # Minimum threshold for a valid swing signal is 5 points
        threshold = 5
        
        if long_signals >= threshold and long_signals > short_signals:
            signal = "buy"
            strength = min(long_signals / 10, 1.0)
        elif short_signals >= threshold and short_signals > long_signals:
            signal = "sell"
            strength = min(short_signals / 10, 1.0)
        
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
                return entry_price + (atr * 3)  # 3 times ATR for take profit (swing trades aim for larger moves)
            return entry_price * 1.03  # 3% default take profit
        elif side == "sell":
            if atr:
                return entry_price - (atr * 3)
            return entry_price * 0.97  # 3% default take profit
        return entry_price
        
    def get_stop_loss_price(self, entry_price, side, atr=None):
        """Calculate stop loss price based on ATR or fixed percentage."""
        if side == "buy":
            if atr:
                return entry_price - (atr * 2)  # 2 times ATR for stop loss
            return entry_price * 0.985  # 1.5% default stop loss
        elif side == "sell":
            if atr:
                return entry_price + (atr * 2)
            return entry_price * 1.015  # 1.5% default stop loss
        return entry_price
