"""
Trading strategy for the AI Trading Bot.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime
import config
from indicators import (
    calculate_rsi, 
    calculate_vwap, 
    calculate_atr, 
    detect_volume_spike,
    calculate_order_book_imbalance,
    calculate_bollinger_bands,
    calculate_macd
)
from utils import (
    calculate_take_profit_price,
    calculate_stop_loss_price,
    calculate_atr_stop_loss,
    save_trade_history
)

# Initialize logger
logger = logging.getLogger('trading_bot')


class Strategy:
    """
    Base strategy class for the trading bot.
    """
    
    def __init__(self, binance_client, telegram_notifier=None, symbol=None, timeframe=None, leverage=None):
        """
        Initialize the base strategy.
        
        Args:
            binance_client (BinanceClient): Binance client
            telegram_notifier (TelegramNotifier, optional): Telegram notifier
            symbol (str): Trading symbol
            timeframe (str): Trading timeframe
            leverage (int): Trading leverage
        """
        self.binance_client = binance_client
        self.telegram_notifier = telegram_notifier
        self.symbol = symbol
        self.timeframe = timeframe
        self.leverage = leverage
        self.active_positions = {}
        
        # Risk management
        self.take_profit_percent = config.TAKE_PROFIT_PERCENT / 100
        self.stop_loss_percent = config.STOP_LOSS_PERCENT / 100
        self.use_atr_for_sl = config.USE_ATR_FOR_SL
        self.atr_multiplier = config.ATR_MULTIPLIER
        self.use_trailing_stop = config.USE_TRAILING_STOP
        self.trailing_stop_activation = config.TRAILING_STOP_ACTIVATION / 100
        self.trailing_stop_callback = config.TRAILING_STOP_CALLBACK / 100
        
        # Position management
        self.max_active_positions = config.MAX_ACTIVE_POSITIONS
        self.position_size = config.POSITION_SIZE
        
        logger.info(f"{self.__class__.__name__} strategy initialized.")

    def analyze_market(self):
        """
        Base market analysis method. Should be implemented by specific strategies.
        """
        raise NotImplementedError("Subclasses must implement analyze_market method")

    def execute_signals(self, analysis):
        """
        Execute trading signals based on market analysis.
        
        Args:
            analysis (dict): Market analysis results
            
        Returns:
            dict: Execution results
        """
        try:
            # Check if we can open new positions
            if len(self.active_positions) >= self.max_active_positions:
                logger.info(f"Maximum number of active positions reached ({self.max_active_positions}).")
                return {'action': 'none', 'reason': 'max_positions_reached'}
            
            # Check for entry signals
            if analysis['long_signal']:
                return self.open_long_position(analysis)
            
            elif analysis['short_signal']:
                return self.open_short_position(analysis)
            
            return {'action': 'none', 'reason': 'no_signal'}
        
        except Exception as e:
            logger.error(f"Error executing signals: {e}")
            if self.telegram_notifier:
                self.telegram_notifier.notify_error(f"Error executing signals: {e}")
            raise

    def open_long_position(self, analysis):
        """
        Open a long position.
        """
        # ... existing code ...

    def open_short_position(self, analysis):
        """
        Open a short position.
        """
        # ... existing code ...

    def manage_positions(self):
        """
        Manage open positions.
        """
        # ... existing code ...

    def close_position(self, position_id, reason, exit_price):
        """
        Close a position.
        """
        # ... existing code ...


class ScalpingStrategy(Strategy):
    """
    Scalping strategy for short-term trades based on RSI and VWAP.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Strategy parameters
        self.rsi_period = config.RSI_PERIOD
        self.rsi_overbought = config.RSI_OVERBOUGHT
        self.rsi_oversold = config.RSI_OVERSOLD
        self.vwap_period = config.VWAP_PERIOD
        self.atr_period = config.ATR_PERIOD
        self.volume_threshold = config.VOLUME_THRESHOLD
    
    def analyze_market(self):
        """
        Analyze market data for scalping opportunities.
        """
        try:
            # Get market data
            klines = self.binance_client.get_klines(
                symbol=self.symbol,
                timeframe=self.timeframe,
                limit=100
            )
            
            # Calculate indicators
            klines['rsi'] = calculate_rsi(klines, period=self.rsi_period)
            klines['vwap'] = calculate_vwap(klines, period=self.vwap_period)
            klines['atr'] = calculate_atr(klines, period=self.atr_period)
            
            # Get latest data
            latest = klines.iloc[-1]
            prev = klines.iloc[-2]
            
            # Determine trend based on VWAP and price action
            trend = 'bullish' if (latest['close'] > latest['vwap'] and latest['close'] > prev['close']) else 'bearish'
            
            # Generate signals with momentum confirmation
            long_signal = (
                trend == 'bullish' and
                latest['rsi'] < self.rsi_oversold and
                latest['close'] > latest['open']  # Green candle confirmation
            )
            
            short_signal = (
                trend == 'bearish' and
                latest['rsi'] > self.rsi_overbought and
                latest['close'] < latest['open']  # Red candle confirmation
            )
            
            # Debug logging
            logger.info(f"Scalping Analysis - Price: {latest['close']:.2f}, RSI: {latest['rsi']:.2f}, VWAP: {latest['vwap']:.2f}")
            logger.info(f"Scalping Conditions - Trend: {trend}, RSI Oversold: {latest['rsi'] < self.rsi_oversold}, RSI Overbought: {latest['rsi'] > self.rsi_overbought}")
            
            result = {
                'timestamp': datetime.now(),
                'symbol': self.symbol,
                'price': latest['close'],
                'rsi': latest['rsi'],
                'vwap': latest['vwap'],
                'atr': latest['atr'],
                'trend': trend,
                'long_signal': long_signal,
                'short_signal': short_signal
            }
            
            if long_signal or short_signal:
                logger.info(f"Scalping Signal Generated - Long: {long_signal}, Short: {short_signal}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing market: {e}")
            if self.telegram_notifier:
                self.telegram_notifier.notify_error(f"Error analyzing market: {e}")
            raise


class SwingStrategy(Strategy):
    """
    Swing trading strategy based on MACD and Bollinger Bands.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Strategy parameters
        self.macd_fast = config.MACD_FAST_PERIOD
        self.macd_slow = config.MACD_SLOW_PERIOD
        self.macd_signal = config.MACD_SIGNAL_PERIOD
        self.bb_period = config.BB_PERIOD
        self.bb_std = config.BB_STD
    
    def analyze_market(self):
        """
        Analyze market data for swing trading opportunities.
        """
        try:
            # Get market data
            klines = self.binance_client.get_klines(
                symbol=self.symbol,
                timeframe=self.timeframe,
                limit=100
            )
            
            # Calculate indicators
            macd_line, signal_line, histogram = calculate_macd(
                klines,
                fast_period=self.macd_fast,
                slow_period=self.macd_slow,
                signal_period=self.macd_signal
            )
            middle_band, upper_band, lower_band = calculate_bollinger_bands(
                klines,
                period=self.bb_period,
                std_dev=self.bb_std
            )
            
            # Get latest data
            latest = klines.iloc[-1]
            prev = klines.iloc[-2]
            latest_macd = macd_line.iloc[-1]
            latest_signal = signal_line.iloc[-1]
            prev_macd = macd_line.iloc[-2]
            prev_signal = signal_line.iloc[-2]
            
            # Generate signals with trend confirmation
            long_signal = (
                latest_macd > latest_signal and  # Current MACD crossover
                prev_macd <= prev_signal and  # Confirm crossover just happened
                latest['close'] < lower_band.iloc[-1] and  # Price below lower BB
                latest['close'] > latest['open']  # Green candle confirmation
            )
            
            short_signal = (
                latest_macd < latest_signal and  # Current MACD crossunder
                prev_macd >= prev_signal and  # Confirm crossunder just happened
                latest['close'] > upper_band.iloc[-1] and  # Price above upper BB
                latest['close'] < latest['open']  # Red candle confirmation
            )
            
            # Debug logging
            logger.info(f"Swing Analysis - Price: {latest['close']:.2f}, MACD: {latest_macd:.2f}, Signal: {latest_signal:.2f}")
            logger.info(f"Swing BB Levels - Lower: {lower_band.iloc[-1]:.2f}, Middle: {middle_band.iloc[-1]:.2f}, Upper: {upper_band.iloc[-1]:.2f}")
            logger.info(f"Swing Conditions - MACD Cross: {latest_macd > latest_signal}, Price vs BB: {latest['close'] < lower_band.iloc[-1] or latest['close'] > upper_band.iloc[-1]}")
            
            result = {
                'timestamp': datetime.now(),
                'symbol': self.symbol,
                'price': latest['close'],
                'macd': latest_macd,
                'signal': latest_signal,
                'middle_band': middle_band.iloc[-1],
                'upper_band': upper_band.iloc[-1],
                'lower_band': lower_band.iloc[-1],
                'long_signal': long_signal,
                'short_signal': short_signal
            }
            
            if long_signal or short_signal:
                logger.info(f"Swing Signal Generated - Long: {long_signal}, Short: {short_signal}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing market: {e}")
            if self.telegram_notifier:
                self.telegram_notifier.notify_error(f"Error analyzing market: {e}")
            raise


class BreakoutStrategy(Strategy):
    """
    Breakout trading strategy based on ATR and Volume.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Strategy parameters
        self.atr_period = config.ATR_PERIOD
        self.volume_threshold = config.VOLUME_THRESHOLD
        self.breakout_period = config.BREAKOUT_PERIOD
    
    def analyze_market(self):
        """
        Analyze market data for breakout opportunities.
        """
        try:
            # Get market data
            klines = self.binance_client.get_klines(
                symbol=self.symbol,
                timeframe=self.timeframe,
                limit=self.breakout_period + 20  # Extra candles for indicators
            )
            
            # Calculate indicators
            klines['atr'] = calculate_atr(klines, period=self.atr_period)
            klines['volume_spike'] = detect_volume_spike(klines, threshold=self.volume_threshold)
            
            # Calculate recent high/low with shorter period for more opportunities
            recent_high = klines['high'].rolling(window=self.breakout_period).max()
            recent_low = klines['low'].rolling(window=self.breakout_period).min()
            
            # Get latest data
            latest = klines.iloc[-1]
            prev = klines.iloc[-2]
            
            # Calculate average volume for volume confirmation
            avg_volume = klines['volume'].rolling(window=self.breakout_period).mean().iloc[-1]
            volume_confirmed = latest['volume'] > avg_volume * self.volume_threshold
            
            # Generate signals with multiple confirmations
            long_signal = (
                latest['close'] > recent_high.iloc[-2] and  # Break above recent high
                latest['close'] > latest['open'] and  # Green candle
                volume_confirmed and  # Volume confirmation
                latest['close'] > latest['close'] * (1 + 0.001)  # Minimum price movement
            )
            
            short_signal = (
                latest['close'] < recent_low.iloc[-2] and  # Break below recent low
                latest['close'] < latest['open'] and  # Red candle
                volume_confirmed and  # Volume confirmation
                latest['close'] < latest['close'] * (1 - 0.001)  # Minimum price movement
            )
            
            # Debug logging
            logger.info(f"Breakout Analysis - Price: {latest['close']:.2f}, Recent High: {recent_high.iloc[-2]:.2f}, Recent Low: {recent_low.iloc[-2]:.2f}")
            logger.info(f"Breakout Conditions - Volume Confirmed: {volume_confirmed}, Candle: {'Green' if latest['close'] > latest['open'] else 'Red'}")
            logger.info(f"Breakout Levels - Break High: {latest['close'] > recent_high.iloc[-2]}, Break Low: {latest['close'] < recent_low.iloc[-2]}")
            
            result = {
                'timestamp': datetime.now(),
                'symbol': self.symbol,
                'price': latest['close'],
                'atr': latest['atr'],
                'recent_high': recent_high.iloc[-1],
                'recent_low': recent_low.iloc[-1],
                'volume_spike': latest['volume_spike'],
                'long_signal': long_signal,
                'short_signal': short_signal
            }
            
            if long_signal or short_signal:
                logger.info(f"Breakout Signal Generated - Long: {long_signal}, Short: {short_signal}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing market: {e}")
            if self.telegram_notifier:
                self.telegram_notifier.notify_error(f"Error analyzing market: {e}")
            raise 