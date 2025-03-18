import numpy as np
import pandas as pd
import talib

import config
from utils.logger import setup_logger

class SwingStrategy:
    def __init__(self, client):
        self.client = client
        self.logger = setup_logger('swing_strategy', 'logs/strategies.log')
        self.params = config.STRATEGY_PARAMS['swing']
        
    def generate_signal(self, symbol):
        """
        Generate trading signals based on swing trading strategy.
        
        Returns:
            dict: Signal details including action, price levels, and reasoning,
                  or None if no valid signal
        """
        try:
            # Get recent price data
            df = self.client.get_historical_klines(
                symbol,
                config.TIMEFRAMES['swing'],
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
            
            # Buy signal conditions based on Ichimoku Cloud and MACD
            if (last_row['close'] > last_row['senkou_span_a'] and
                last_row['close'] > last_row['senkou_span_b'] and
                last_row['macd'] > last_row['macd_signal'] and
                prev_row['macd'] <= prev_row['macd_signal'] and
                last_row['volume'] > last_row['volume_ma']):
                
                # Calculate price targets
                entry_price = last_row['close']
                stop_loss = min(last_row['kijun_sen'], last_row['senkou_span_b']) * 0.99  # Just below support
                take_profit = entry_price + (1.5 * (entry_price - stop_loss))  # 1.5 risk-reward ratio
                
                reasoning = (
                    f"Price above Ichimoku Cloud + "
                    f"MACD bullish crossover + "
                    f"Volume above average ({last_row['volume']:.2f} > {last_row['volume_ma']:.2f})"
                )
                
                signal = {
                    'action': 'BUY',
                    'price': entry_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'reasoning': reasoning
                }
                
            # Sell signal conditions
            elif (last_row['close'] < last_row['senkou_span_a'] and
                  last_row['close'] < last_row['senkou_span_b'] and
                  last_row['macd'] < last_row['macd_signal'] and
                  prev_row['macd'] >= prev_row['macd_signal'] and
                  last_row['volume'] > last_row['volume_ma']):
                
                # Calculate price targets
                entry_price = last_row['close']
                stop_loss = max(last_row['kijun_sen'], last_row['senkou_span_a']) * 1.01  # Just above resistance
                take_profit = entry_price - (1.5 * (stop_loss - entry_price))  # 1.5 risk-reward ratio
                
                reasoning = (
                    f"Price below Ichimoku Cloud + "
                    f"MACD bearish crossover + "
                    f"Volume above average ({last_row['volume']:.2f} > {last_row['volume_ma']:.2f})"
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
            self.logger.error(f"Error generating swing signal for {symbol}: {str(e)}")
            return None
            
    def calculate_indicators(self, df):
        """Calculate technical indicators for the swing trading strategy."""
        try:
            # MACD
            macd, macd_signal, macd_hist = talib.MACD(
                df['close'].values,
                fastperiod=self.params['macd_fast'],
                slowperiod=self.params['macd_slow'],
                signalperiod=self.params['macd_signal']
            )
            df['macd'] = macd
            df['macd_signal'] = macd_signal
            df['macd_hist'] = macd_hist
            
            # Ichimoku Cloud components
            high_values = df['high'].values
            low_values = df['low'].values
            close_values = df['close'].values
            
            # Tenkan-sen (Conversion Line)
            period9_high = pd.Series(df['high']).rolling(window=self.params['ichimoku_tenkan']).max()
            period9_low = pd.Series(df['low']).rolling(window=self.params['ichimoku_tenkan']).min()
            df['tenkan_sen'] = (period9_high + period9_low) / 2
            
            # Kijun-sen (Base Line)
            period26_high = pd.Series(df['high']).rolling(window=self.params['ichimoku_kijun']).max()
            period26_low = pd.Series(df['low']).rolling(window=self.params['ichimoku_kijun']).min()
            df['kijun_sen'] = (period26_high + period26_low) / 2
            
            # Senkou Span A (Leading Span A)
            df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(self.params['ichimoku_kijun'])
            
            # Senkou Span B (Leading Span B)
            period52_high = pd.Series(df['high']).rolling(window=self.params['ichimoku_senkou_span_b']).max()
            period52_low = pd.Series(df['low']).rolling(window=self.params['ichimoku_senkou_span_b']).min()
            df['senkou_span_b'] = ((period52_high + period52_low) / 2).shift(self.params['ichimoku_kijun'])
            
            # Chikou Span (Lagging Span)
            df['chikou_span'] = df['close'].shift(-self.params['ichimoku_kijun'])
            
            # Volume moving average
            df['volume_ma'] = df['volume'].rolling(window=self.params['volume_ma']).mean()
            
            # ATR for volatility assessment
            df['atr'] = talib.ATR(
                df['high'].values, 
                df['low'].values, 
                df['close'].values, 
                timeperiod=14
            )
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error calculating indicators: {str(e)}")
            return df
