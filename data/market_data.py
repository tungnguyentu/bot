import logging
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta

logger = logging.getLogger("BinanceBot.MarketData")


class MarketData:
    def __init__(self, exchange):
        self.exchange = exchange
        self.data_cache = {}  # Cache historical data to reduce API calls
        self.last_update = {}  # Track when data was last updated
        self.last_check_news = datetime.now()

    def get_historical_data(self, symbol, timeframe, limit=100):
        """
        Get historical candlestick data for a symbol, using cache when possible
        """
        current_time = time.time()
        cache_key = f"{symbol}_{timeframe}"

        # Define cache validity period based on timeframe
        cache_validity = {
            "1m": 30,  # 30 seconds
            "5m": 60,  # 1 minute
            "15m": 180,  # 3 minutes
            "30m": 300,  # 5 minutes
            "1h": 600,  # 10 minutes
            "4h": 1800,  # 30 minutes
            "1d": 3600,  # 1 hour
        }.get(
            timeframe, 60
        )  # Default to 60 seconds

        # Check if we have cached data and if it's still valid
        if (
            cache_key in self.data_cache
            and cache_key in self.last_update
            and current_time - self.last_update[cache_key] < cache_validity
        ):
            return self.data_cache[cache_key]

        # Fetch new data from the exchange
        data = self.exchange.get_historical_data(symbol, timeframe, limit)

        if data and len(data) > 0:
            # Update cache
            self.data_cache[cache_key] = data
            self.last_update[cache_key] = current_time
            return data
        else:
            # If API call failed but we have cached data, use it
            if cache_key in self.data_cache:
                logger.warning(
                    f"Failed to get new data for {symbol}, using cached data"
                )
                return self.data_cache[cache_key]

            logger.error(
                f"Failed to get historical data for {symbol} and no cache available"
            )
            return []

    def check_high_impact_news(self):
        """
        Check if there are any high-impact news events scheduled in the next 24 hours
        Returns True if trading should be avoided
        """
        # This is a stub method - would connect to a news/economic calendar API
        # For now, always return False (no high impact news)

        # Only check for news every hour to avoid unnecessary API calls
        if datetime.now() - self.last_check_news < timedelta(hours=1):
            return False

        self.last_check_news = datetime.now()

        # Here you would implement your news API call
        # Example: query ForexFactory, Investing.com, or a dedicated news API

        # Return False for now (no high impact news detected)
        return False

    def calculate_volatility(self, symbol, timeframe="1h", lookback=24):
        """
        Calculate the market volatility for a symbol
        Returns the ATR as a percentage of price
        """
        data = self.get_historical_data(symbol, timeframe, lookback + 14)
        if not data or len(data) < 14:
            return None

        df = pd.DataFrame(data)

        # Calculate ATR
        high = df["high"].values
        low = df["low"].values
        close = pd.Series(df["close"].values)

        tr1 = pd.Series(high) - pd.Series(low)
        tr2 = abs(pd.Series(high) - close.shift())
        tr3 = abs(pd.Series(low) - close.shift())

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().iloc[-1]

        # Get current price
        current_price = df["close"].iloc[-1]

        # Return ATR as a percentage of current price
        return (atr / current_price) * 100

    def is_market_volatile(self, symbol, threshold=1.5):
        """
        Check if market is too volatile for mean reversion
        Returns True if market is considered too volatile
        """
        volatility = self.calculate_volatility(symbol)
        if volatility is None:
            return False  # If we can't calculate, assume it's not too volatile

        logger.info(f"Market volatility for {symbol}: {volatility:.2f}%")

        # If volatility is above threshold percentage, consider market too volatile
        return volatility > threshold
