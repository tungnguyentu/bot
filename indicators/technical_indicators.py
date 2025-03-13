import pandas as pd
import numpy as np
import talib


class TechnicalIndicators:
    def add_bollinger_bands(self, df, period=14, std_dev=2):
        """
        Add Bollinger Bands to DataFrame
        """
        if len(df) >= period:
            upper, middle, lower = talib.BBANDS(
                df["close"].values,
                timeperiod=period,
                nbdevup=std_dev,
                nbdevdn=std_dev,
                matype=0,
            )

            df["bb_upper"] = upper
            df["bb_middle"] = middle
            df["bb_lower"] = lower

        return df

    def add_rsi(self, df, period=6):
        """
        Add Relative Strength Index to DataFrame
        """
        if len(df) >= period:
            df["rsi"] = talib.RSI(df["close"].values, timeperiod=period)

        return df

    def add_macd(self, df, fast_period=5, slow_period=13, signal_period=1):
        """
        Add MACD to DataFrame
        """
        if len(df) >= slow_period:
            macd, signal, hist = talib.MACD(
                df["close"].values,
                fastperiod=fast_period,
                slowperiod=slow_period,
                signalperiod=signal_period,
            )

            df["macd"] = macd
            df["macd_signal"] = signal
            df["macd_hist"] = hist

        return df

    def add_vwap(self, df):
        """
        Add Volume Weighted Average Price to DataFrame
        """
        # Create a copy to avoid modifying the original DataFrame
        df = df.copy()

        # Calculate VWAP
        df["vwap"] = (
            df["volume"] * ((df["high"] + df["low"] + df["close"]) / 3)
        ).cumsum() / df["volume"].cumsum()

        return df

    def add_atr(self, df, period=14):
        """
        Add Average True Range to DataFrame
        """
        if len(df) >= period:
            df["atr"] = talib.ATR(
                df["high"].values,
                df["low"].values,
                df["close"].values,
                timeperiod=period,
            )

        return df

    def add_sma(self, df, period=20):
        """
        Add Simple Moving Average to DataFrame
        """
        if len(df) >= period:
            df[f"sma_{period}"] = talib.SMA(df["close"].values, timeperiod=period)

        return df

    def add_ema(self, df, period=20):
        """
        Add Exponential Moving Average to DataFrame
        """
        if len(df) >= period:
            df[f"ema_{period}"] = talib.EMA(df["close"].values, timeperiod=period)

        return df
