import numpy as np
import pandas as pd
import ta

import config
from utils.logger import setup_logger

class ScalpingStrategy:
    def __init__(self, client):
        self.client = client
        self.logger = setup_logger('scalping_strategy', 'logs/strategies.log')
        self.params = config.STRATEGY_PARAMS['scalping']
        
    def generate_signal(self, symbol):
        """
        Generate trading signals based on scalping strategy.
        
        Returns:
            dict: Signal details including action, price levels, and reasoning,
                  or None if no valid signal
        """
        try:
            # Get recent price data
            df = self.client.get_historical_klines(
                symbol,
                config.TIMEFRAMES['scalping'],
                limit=100
            )
            
            if df.empty:
                self.logger.warning(f"No data available for {symbol}")
                return None
                
            # Calculate indicators
            df = self.calculate_indicators(df)
            
            # Check for signals
            last_row = df.iloc[-1]
            prev_row = df.iloc[-2]
            
            # Default signal is no action
            signal = None
            
            # Buy signal conditions
            if (last_row['rsi'] < self.params['rsi_oversold'] and 
                last_row['close'] < last_row['lower_band'] and
                last_row['close'] > last_row['ma_fast'] and
                last_row['ma_fast'] > prev_row['ma_fast']):
                
                # Calculate price targets
                entry_price = last_row['close']
                stop_loss = entry_price - (2 * (last_row['atr']))
                take_profit = entry_price + (3 * (last_row['atr']))
                
                reasoning = (
                    f"RSI oversold ({last_row['rsi']:.2f}) + "
                    f"Price below lower Bollinger Band + "
                    f"Price above fast MA ({last_row['ma_fast']:.2f}) + "
                    f"Fast MA rising"
                )
                
                signal = {
                    'action': 'BUY',
                    'price': entry_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'reasoning': reasoning
                }
                
            # Sell signal conditions
            elif (last_row['rsi'] > self.params['rsi_overbought'] and 
                  last_row['close'] > last_row['upper_band'] and
                  last_row['close'] < last_row['ma_fast'] and
                  last_row['ma_fast'] < prev_row['ma_fast']):
                
                # Calculate price targets
                entry_price = last_row['close']
                stop_loss = entry_price + (2 * (last_row['atr']))
                take_profit = entry_price - (3 * (last_row['atr']))
                
                reasoning = (
                    f"RSI overbought ({last_row['rsi']:.2f}) + "
                    f"Price above upper Bollinger Band + "
                    f"Price below fast MA ({last_row['ma_fast']:.2f}) + "
                    f"Fast MA declining"
                )
                
                signal = {
                    'action': 'SELL',
                    'price': entry_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'reasoning': reasoning
                }
                
            return signal
            
        except Exception as e:
            self.logger.error(f"Error generating scalping signal for {symbol}: {str(e)}")
            return None
            
    def calculate_indicators(self, df):
        """Calculate technical indicators for the scalping strategy."""
        try:
            # RSI
            rsi_indicator = ta.momentum.RSIIndicator(
                close=df['close'], 
                window=self.params['rsi_period']
            )
            df['rsi'] = rsi_indicator.rsi()
            
            # Bollinger Bands
            bb_indicator = ta.volatility.BollingerBands(
                close=df['close'],
                window=self.params['bb_period'],
                window_dev=self.params['bb_std']
            )
            df['upper_band'] = bb_indicator.bollinger_hband()
            df['middle_band'] = bb_indicator.bollinger_mavg()
            df['lower_band'] = bb_indicator.bollinger_lband()
            
            # Moving Averages
            ema_fast = ta.trend.EMAIndicator(
                close=df['close'], 
                window=self.params['ma_fast']
            )
            ema_slow = ta.trend.EMAIndicator(
                close=df['close'], 
                window=self.params['ma_slow']
            )
            df['ma_fast'] = ema_fast.ema_indicator()
            df['ma_slow'] = ema_slow.ema_indicator()
            
            # ATR for stop loss calculation
            atr_indicator = ta.volatility.AverageTrueRange(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                window=14
            )
            df['atr'] = atr_indicator.average_true_range()
            
            # Calculate candlestick patterns manually
            # Bullish engulfing
            df['engulfing_bullish'] = (
                (df['close'] > df['open'].shift(1)) &
                (df['open'] < df['close'].shift(1)) &
                (df['close'] > df['open']) &
                (df['close'].shift(1) < df['open'].shift(1))
            )
            
            # Hammer pattern (simplified)
            df['hammer'] = (
                (df['close'] > df['open']) &  # Bullish candle
                ((df['high'] - df['close']) < 0.3 * (df['close'] - df['low'])) &  # Small upper shadow
                ((df['close'] - df['open']) < 0.5 * (df['open'] - df['low']))  # Body is in upper half
            )
            
            # Bearish engulfing
            df['engulfing_bearish'] = (
                (df['close'] < df['open'].shift(1)) &
                (df['open'] > df['close'].shift(1)) &
                (df['close'] < df['open']) &
                (df['close'].shift(1) > df['open'].shift(1))
            )
            
            # Shooting star (simplified)
            df['shooting_star'] = (
                (df['close'] < df['open']) &  # Bearish candle
                ((df['high'] - df['open']) > 2 * (df['open'] - df['close'])) &  # Long upper shadow
                ((df['close'] - df['low']) < 0.3 * (df['high'] - df['close']))  # Small lower shadow
            )
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error calculating indicators: {str(e)}")
            return df
