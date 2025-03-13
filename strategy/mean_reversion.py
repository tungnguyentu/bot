import logging
import pandas as pd
import numpy as np
from indicators.technical_indicators import TechnicalIndicators

logger = logging.getLogger("BinanceBot.MeanReversionStrategy")


class MeanReversionStrategy:
    def __init__(self, market_data, config):
        self.market_data = market_data
        self.config = config
        self.indicators = TechnicalIndicators()

    def generate_signal(self, symbol, historical_data):
        """
        Generate trading signals based on mean reversion strategy
        Returns: (direction, reasoning) or None if no signal
        """
        if not historical_data or len(historical_data) < self.config.LOOKBACK_PERIOD:
            logger.info(f"Not enough data for {symbol} to generate signals")
            return None

        # Convert to pandas DataFrame for easier calculation
        df = pd.DataFrame(historical_data)
        
        # Calculate indicators
        df = self.calculate_indicators(df)
        
        # Get the last row for analysis
        last = df.iloc[-1]
        prev = df.iloc[-2]

        # Check if price is too close to VWAP
        price_to_vwap_pct = abs((last["close"] - last["vwap"]) / last["close"]) * 100
        if price_to_vwap_pct < 0.2:  # If price is within 0.2% of VWAP
            return None

        # Look for long signals
        if (
            last["close"] <= last["bb_lower"]  # Price at or below lower BB
            and last["rsi"] < self.config.RSI_OVERSOLD  # RSI oversold
            and last["macd_hist"] > prev["macd_hist"]
        ):  # MACD histogram turning positive

            reasoning = (
                f"LONG signal: Price below BB lower band ({round(last['bb_lower'], 2)}), "
                f"RSI oversold ({round(last['rsi'], 2)}), "
                f"MACD histogram turning positive"
            )

            return ("LONG", reasoning)

        # Look for short signals
        elif (
            last["close"] >= last["bb_upper"]  # Price at or above upper BB
            and last["rsi"] > self.config.RSI_OVERBOUGHT  # RSI overbought
            and last["macd_hist"] < prev["macd_hist"]
        ):  # MACD histogram turning negative

            reasoning = (
                f"SHORT signal: Price above BB upper band ({round(last['bb_upper'], 2)}), "
                f"RSI overbought ({round(last['rsi'], 2)}), "
                f"MACD histogram turning negative"
            )

            return ("SHORT", reasoning)

        # No signal
        return None
        
    def calculate_indicators(self, df):
        """Calculate all technical indicators needed for the strategy"""
        # Calculate Bollinger Bands
        df = self.indicators.add_bollinger_bands(
            df, period=self.config.BB_PERIOD, std_dev=self.config.BB_STD_DEV
        )

        # Calculate RSI
        df = self.indicators.add_rsi(df, period=self.config.RSI_PERIOD)

        # Calculate MACD
        df = self.indicators.add_macd(
            df,
            fast_period=self.config.MACD_FAST,
            slow_period=self.config.MACD_SLOW,
            signal_period=self.config.MACD_SIGNAL,
        )

        # Calculate VWAP
        df = self.indicators.add_vwap(df)

        # Calculate ATR for stop loss and take profit
        df = self.indicators.add_atr(df, period=self.config.ATR_PERIOD)
        
        return df
    
    def prepare_state_for_rl(self, row):
        """
        Prepare a state vector for reinforcement learning
        """
        # BB position: Where is price relative to BB bands? (-1 to 1)
        if 'bb_upper' in row and 'bb_lower' in row and 'close' in row:
            bb_range = row['bb_upper'] - row['bb_lower']
            if bb_range > 0:
                bb_position = 2 * (row['close'] - row['bb_lower']) / bb_range - 1
            else:
                bb_position = 0
        else:
            bb_position = 0
        
        # RSI normalized (0-1)
        rsi = row['rsi'] / 100 if 'rsi' in row else 0.5
        
        # MACD histogram
        macd_hist = row['macd_hist'] if 'macd_hist' in row else 0
        
        # VWAP distance (normalized)
        if 'vwap' in row and 'close' in row:
            vwap_dist = (row['close'] - row['vwap']) / row['close']
        else:
            vwap_dist = 0
        
        # ATR (volatility)
        atr = row['atr'] if 'atr' in row else 0
        
        return [bb_position, rsi, macd_hist, vwap_dist, atr]
