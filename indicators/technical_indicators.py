import pandas as pd
import numpy as np
import os
import sys
# Replace talib with ta library
import ta
from ta.trend import MACD, EMAIndicator, SMAIndicator, ADXIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import MFIIndicator, OnBalanceVolumeIndicator, VolumeWeightedAveragePrice

# Add the parent directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

class TechnicalIndicators:
    @staticmethod
    def add_all_indicators(df, timeframe="scalping"):
        """Add all necessary technical indicators based on trading timeframe."""
        df = df.copy()
        
        # Make sure we have high/low/close/volume for the indicators
        if 'high' not in df or 'low' not in df or 'close' not in df or 'volume' not in df:
            return df
            
        if timeframe == "scalping":
            df = TechnicalIndicators.add_scalping_indicators(df)
        else:
            df = TechnicalIndicators.add_swing_indicators(df)
        
        # Common indicators
        df = TechnicalIndicators.add_volatility_indicators(df)
        df = TechnicalIndicators.add_volume_indicators(df)
        df = TechnicalIndicators.add_candlestick_patterns(df)
        
        return df
    
    @staticmethod
    def add_scalping_indicators(df):
        """Add indicators for scalping strategy."""
        # RSI
        rsi = RSIIndicator(close=df['close'], window=config.SCALPING_RSI_PERIOD)
        df['rsi'] = rsi.rsi()
        
        # Bollinger Bands
        bb = BollingerBands(
            close=df['close'], 
            window=config.SCALPING_BB_PERIOD, 
            window_dev=config.SCALPING_BB_STD
        )
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_middle'] = bb.bollinger_mavg()
        df['bb_lower'] = bb.bollinger_lband()
        
        # Moving Averages
        ema_short = EMAIndicator(close=df['close'], window=config.SCALPING_MA_PERIOD_SHORT)
        ema_long = EMAIndicator(close=df['close'], window=config.SCALPING_MA_PERIOD_LONG)
        df['ema_short'] = ema_short.ema_indicator()
        df['ema_long'] = ema_long.ema_indicator()
        
        # MACD for confirmation
        macd = MACD(
            close=df['close'], 
            window_slow=26, 
            window_fast=12, 
            window_sign=9
        )
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_hist'] = macd.macd_diff()
        
        return df
    
    @staticmethod
    def add_swing_indicators(df):
        """Add indicators for swing trading strategy."""
        # MACD
        macd = MACD(
            close=df['close'], 
            window_slow=config.SWING_MACD_SLOW, 
            window_fast=config.SWING_MACD_FAST, 
            window_sign=config.SWING_MACD_SIGNAL
        )
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_hist'] = macd.macd_diff()
        
        # Ichimoku Cloud
        df['ichimoku_conversion'] = TechnicalIndicators._ichimoku_conversion_line(df)
        df['ichimoku_base'] = TechnicalIndicators._ichimoku_base_line(df)
        df['ichimoku_span_a'] = TechnicalIndicators._ichimoku_leading_span_a(df)
        df['ichimoku_span_b'] = TechnicalIndicators._ichimoku_leading_span_b(df)
        
        # RSI for confirmation
        rsi = RSIIndicator(close=df['close'], window=14)
        df['rsi'] = rsi.rsi()
        
        # Directional Movement Index
        adx_indicator = ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=14)
        df['adx'] = adx_indicator.adx()
        df['di_plus'] = adx_indicator.adx_pos()
        df['di_minus'] = adx_indicator.adx_neg()
        
        return df
    
    @staticmethod
    def add_volatility_indicators(df):
        """Add volatility indicators."""
        # ATR - Average True Range
        atr_indicator = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14)
        df['atr'] = atr_indicator.average_true_range()
        
        # Bollinger Bands Width (volatility measure)
        if 'bb_upper' in df and 'bb_lower' in df and 'bb_middle' in df:
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        elif 'close' in df:  # Calculate if not already there
            bb = BollingerBands(close=df['close'], window=20, window_dev=2)
            upper = bb.bollinger_hband()
            middle = bb.bollinger_mavg()
            lower = bb.bollinger_lband()
            df['bb_width'] = (upper - lower) / middle
        
        return df
    
    @staticmethod
    def add_volume_indicators(df):
        """Add volume-based indicators."""
        # On-Balance Volume
        obv = OnBalanceVolumeIndicator(close=df['close'], volume=df['volume'])
        df['obv'] = obv.on_balance_volume()
        
        # Volume moving average
        volume_sma = SMAIndicator(close=df['volume'], window=20)
        df['volume_ma'] = volume_sma.sma_indicator()
        
        # Money Flow Index
        mfi = MFIIndicator(high=df['high'], low=df['low'], close=df['close'], volume=df['volume'], window=14)
        df['mfi'] = mfi.money_flow_index()
        
        # Volume Weighted Average Price (VWAP)
        vwap = VolumeWeightedAveragePrice(
            high=df['high'], low=df['low'], close=df['close'], volume=df['volume'], window=14
        )
        df['vwap'] = vwap.volume_weighted_average_price()
        
        return df
    
    @staticmethod
    def add_candlestick_patterns(df):
        """Add candlestick pattern recognition."""
        # Since 'ta' library doesn't have built-in candlestick patterns,
        # we'll implement the most common ones manually
        
        # Bullish Engulfing
        df['engulfing_bullish'] = TechnicalIndicators._bullish_engulfing(df)
        
        # Bearish Engulfing
        df['engulfing_bearish'] = TechnicalIndicators._bearish_engulfing(df)
        
        # Hammer
        df['hammer'] = TechnicalIndicators._hammer(df)
        
        # Hanging Man
        df['hanging_man'] = TechnicalIndicators._hanging_man(df)
        
        # Morning Star (approximation)
        df['morning_star'] = TechnicalIndicators._morning_star(df)
        
        # Evening Star (approximation)
        df['evening_star'] = TechnicalIndicators._evening_star(df)
        
        return df
        
    # Ichimoku helper methods (unchanged)
    @staticmethod
    def _ichimoku_conversion_line(df):
        period = config.ICHIMOKU_CONVERSION_LINE_PERIOD
        high_values = df['high'].rolling(window=period).max()
        low_values = df['low'].rolling(window=period).min()
        return (high_values + low_values) / 2

    @staticmethod
    def _ichimoku_base_line(df):
        period = config.ICHIMOKU_BASE_LINE_PERIOD
        high_values = df['high'].rolling(window=period).max()
        low_values = df['low'].rolling(window=period).min()
        return (high_values + low_values) / 2

    @staticmethod
    def _ichimoku_leading_span_a(df):
        conversion = TechnicalIndicators._ichimoku_conversion_line(df)
        base = TechnicalIndicators._ichimoku_base_line(df)
        return ((conversion + base) / 2).shift(config.ICHIMOKU_DISPLACEMENT)

    @staticmethod
    def _ichimoku_leading_span_b(df):
        period = config.ICHIMOKU_LEADING_SPAN_B_PERIOD
        high_values = df['high'].rolling(window=period).max()
        low_values = df['low'].rolling(window=period).min()
        return ((high_values + low_values) / 2).shift(config.ICHIMOKU_DISPLACEMENT)
    
    # Custom candlestick pattern implementations
    @staticmethod
    def _bullish_engulfing(df):
        """Detect bullish engulfing patterns."""
        bullish_engulfing = np.zeros(len(df))
        
        # Previous candle is bearish (close < open) and current candle is bullish (close > open)
        # Current candle completely engulfs the previous one
        for i in range(1, len(df)):
            prev_bearish = df['close'].iloc[i-1] < df['open'].iloc[i-1]
            curr_bullish = df['close'].iloc[i] > df['open'].iloc[i]
            curr_engulfs_prev = (df['open'].iloc[i] < df['close'].iloc[i-1] and 
                                df['close'].iloc[i] > df['open'].iloc[i-1])
            
            if prev_bearish and curr_bullish and curr_engulfs_prev:
                bullish_engulfing[i] = 100  # Signal strength, similar to talib
        
        return bullish_engulfing
    
    @staticmethod
    def _bearish_engulfing(df):
        """Detect bearish engulfing patterns."""
        bearish_engulfing = np.zeros(len(df))
        
        # Previous candle is bullish (close > open) and current candle is bearish (close < open)
        # Current candle completely engulfs the previous one
        for i in range(1, len(df)):
            prev_bullish = df['close'].iloc[i-1] > df['open'].iloc[i-1]
            curr_bearish = df['close'].iloc[i] < df['open'].iloc[i]
            curr_engulfs_prev = (df['open'].iloc[i] > df['close'].iloc[i-1] and 
                                df['close'].iloc[i] < df['open'].iloc[i-1])
            
            if prev_bullish and curr_bearish and curr_engulfs_prev:
                bearish_engulfing[i] = -100  # Negative value for bearish signal
        
        return bearish_engulfing
    
    @staticmethod
    def _hammer(df):
        """Detect hammer candlestick patterns."""
        hammer = np.zeros(len(df))
        
        for i in range(1, len(df)):
            body_size = abs(df['close'].iloc[i] - df['open'].iloc[i])
            total_range = df['high'].iloc[i] - df['low'].iloc[i]
            
            if total_range == 0:  # Avoid division by zero
                continue
                
            body_to_range_ratio = body_size / total_range
            
            # Hammer has a small body at the top and a long lower shadow
            is_small_body = body_to_range_ratio < 0.3
            has_upper_wick = (df['high'].iloc[i] - max(df['open'].iloc[i], df['close'].iloc[i])) / total_range < 0.1
            has_lower_wick = (min(df['open'].iloc[i], df['close'].iloc[i]) - df['low'].iloc[i]) / total_range > 0.6
            
            # In a downtrend
            is_downtrend = df['close'].iloc[i-1] < df['open'].iloc[i-1]
            
            if is_small_body and has_upper_wick and has_lower_wick and is_downtrend:
                hammer[i] = 100
        
        return hammer
    
    @staticmethod
    def _hanging_man(df):
        """Detect hanging man candlestick patterns."""
        hanging_man = np.zeros(len(df))
        
        for i in range(1, len(df)):
            body_size = abs(df['close'].iloc[i] - df['open'].iloc[i])
            total_range = df['high'].iloc[i] - df['low'].iloc[i]
            
            if total_range == 0:  # Avoid division by zero
                continue
                
            body_to_range_ratio = body_size / total_range
            
            # Hanging man has a small body at the top and a long lower shadow
            is_small_body = body_to_range_ratio < 0.3
            has_upper_wick = (df['high'].iloc[i] - max(df['open'].iloc[i], df['close'].iloc[i])) / total_range < 0.1
            has_lower_wick = (min(df['open'].iloc[i], df['close'].iloc[i]) - df['low'].iloc[i]) / total_range > 0.6
            
            # In an uptrend
            is_uptrend = df['close'].iloc[i-1] > df['open'].iloc[i-1]
            
            if is_small_body and has_upper_wick and has_lower_wick and is_uptrend:
                hanging_man[i] = -100  # Negative for bearish signal
        
        return hanging_man
    
    @staticmethod
    def _morning_star(df):
        """Detect morning star patterns (simplified)."""
        morning_star = np.zeros(len(df))
        
        # Need at least 3 candles
        for i in range(2, len(df)):
            # First candle is bearish with long body
            first_bearish = df['close'].iloc[i-2] < df['open'].iloc[i-2]
            first_body_size = abs(df['close'].iloc[i-2] - df['open'].iloc[i-2])
            
            # Second candle is small (indecision)
            second_body_size = abs(df['close'].iloc[i-1] - df['open'].iloc[i-1])
            is_small_body = second_body_size < first_body_size * 0.3
            
            # Gap down between first and second candles
            gap_down = max(df['open'].iloc[i-1], df['close'].iloc[i-1]) < df['close'].iloc[i-2]
            
            # Third candle is bullish with significant body
            third_bullish = df['close'].iloc[i] > df['open'].iloc[i]
            third_body_size = abs(df['close'].iloc[i] - df['open'].iloc[i])
            is_significant_body = third_body_size > second_body_size * 2
            
            # Third candle closes into the first candle's body
            closes_into_first = df['close'].iloc[i] > (df['open'].iloc[i-2] + 
                                                     (df['close'].iloc[i-2] - df['open'].iloc[i-2]) / 2)
            
            if first_bearish and is_small_body and third_bullish and is_significant_body and closes_into_first:
                morning_star[i] = 100
        
        return morning_star
    
    @staticmethod
    def _evening_star(df):
        """Detect evening star patterns (simplified)."""
        evening_star = np.zeros(len(df))
        
        # Need at least 3 candles
        for i in range(2, len(df)):
            # First candle is bullish with long body
            first_bullish = df['close'].iloc[i-2] > df['open'].iloc[i-2]
            first_body_size = abs(df['close'].iloc[i-2] - df['open'].iloc[i-2])
            
            # Second candle is small (indecision)
            second_body_size = abs(df['close'].iloc[i-1] - df['open'].iloc[i-1])
            is_small_body = second_body_size < first_body_size * 0.3
            
            # Gap up between first and second candles
            gap_up = min(df['open'].iloc[i-1], df['close'].iloc[i-1]) > df['close'].iloc[i-2]
            
            # Third candle is bearish with significant body
            third_bearish = df['close'].iloc[i] < df['open'].iloc[i]
            third_body_size = abs(df['close'].iloc[i] - df['open'].iloc[i])
            is_significant_body = third_body_size > second_body_size * 2
            
            # Third candle closes into the first candle's body
            closes_into_first = df['close'].iloc[i] < (df['close'].iloc[i-2] - 
                                                     (df['close'].iloc[i-2] - df['open'].iloc[i-2]) / 2)
            
            if first_bullish and is_small_body and third_bearish and is_significant_body and closes_into_first:
                evening_star[i] = -100  # Negative for bearish signal
        
        return evening_star
