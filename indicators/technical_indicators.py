import pandas as pd
import numpy as np
import sys
import talib
from talib import abstract

sys.path.append('/Users/tungnt/Downloads/game')
import config

class TechnicalIndicators:
    @staticmethod
    def add_all_indicators(df, timeframe="scalping"):
        """Add all necessary technical indicators based on trading timeframe."""
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
        df['rsi'] = talib.RSI(df['close'], timeperiod=config.SCALPING_RSI_PERIOD)
        
        # Bollinger Bands
        upper, middle, lower = talib.BBANDS(
            df['close'],
            timeperiod=config.SCALPING_BB_PERIOD,
            nbdevup=config.SCALPING_BB_STD,
            nbdevdn=config.SCALPING_BB_STD
        )
        df['bb_upper'] = upper
        df['bb_middle'] = middle
        df['bb_lower'] = lower
        
        # Moving Averages
        df['ema_short'] = talib.EMA(df['close'], timeperiod=config.SCALPING_MA_PERIOD_SHORT)
        df['ema_long'] = talib.EMA(df['close'], timeperiod=config.SCALPING_MA_PERIOD_LONG)
        
        # MACD for confirmation
        macd, signal, hist = talib.MACD(
            df['close'], 
            fastperiod=12, 
            slowperiod=26, 
            signalperiod=9
        )
        df['macd'] = macd
        df['macd_signal'] = signal
        df['macd_hist'] = hist
        
        return df
    
    @staticmethod
    def add_swing_indicators(df):
        """Add indicators for swing trading strategy."""
        # MACD
        macd, signal, hist = talib.MACD(
            df['close'], 
            fastperiod=config.SWING_MACD_FAST, 
            slowperiod=config.SWING_MACD_SLOW, 
            signalperiod=config.SWING_MACD_SIGNAL
        )
        df['macd'] = macd
        df['macd_signal'] = signal
        df['macd_hist'] = hist
        
        # Ichimoku Cloud
        conversion_line = TechnicalIndicators._ichimoku_conversion_line(df)
        base_line = TechnicalIndicators._ichimoku_base_line(df)
        leading_span_a = TechnicalIndicators._ichimoku_leading_span_a(df)
        leading_span_b = TechnicalIndicators._ichimoku_leading_span_b(df)
        
        df['ichimoku_conversion'] = conversion_line
        df['ichimoku_base'] = base_line
        df['ichimoku_span_a'] = leading_span_a
        df['ichimoku_span_b'] = leading_span_b
        
        # RSI for confirmation
        df['rsi'] = talib.RSI(df['close'], timeperiod=14)
        
        # Directional Movement Index
        df['adx'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
        df['di_plus'] = talib.PLUS_DI(df['high'], df['low'], df['close'], timeperiod=14)
        df['di_minus'] = talib.MINUS_DI(df['high'], df['low'], df['close'], timeperiod=14)
        
        return df
    
    @staticmethod
    def add_volatility_indicators(df):
        """Add volatility indicators."""
        # ATR - Average True Range
        df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        
        # Bollinger Bands Width (volatility measure)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle'] if 'bb_upper' in df else np.nan
        
        return df
    
    @staticmethod
    def add_volume_indicators(df):
        """Add volume-based indicators."""
        # On-Balance Volume
        df['obv'] = talib.OBV(df['close'], df['volume'])
        
        # Volume moving average
        df['volume_ma'] = talib.SMA(df['volume'], timeperiod=20)
        
        # Money Flow Index
        df['mfi'] = talib.MFI(df['high'], df['low'], df['close'], df['volume'], timeperiod=14)
        
        return df
    
    @staticmethod
    def add_candlestick_patterns(df):
        """Add candlestick pattern recognition."""
        # Bullish patterns
        df['engulfing_bullish'] = talib.CDLENGULFING(df['open'], df['high'], df['low'], df['close'])
        df['hammer'] = talib.CDLHAMMER(df['open'], df['high'], df['low'], df['close'])
        df['morning_star'] = talib.CDLMORNINGSTAR(df['open'], df['high'], df['low'], df['close'])
        
        # Bearish patterns
        df['engulfing_bearish'] = talib.CDLENGULFING(df['open'], df['high'], df['low'], df['close'])
        df['hanging_man'] = talib.CDLHANGINGMAN(df['open'], df['high'], df['low'], df['close'])
        df['evening_star'] = talib.CDLEVENINGSTAR(df['open'], df['high'], df['low'], df['close'])
        
        return df
        
    # Ichimoku helper methods
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
