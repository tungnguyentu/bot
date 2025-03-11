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
            
            # Determine trend based on VWAP
            trend = 'bullish' if latest['close'] > latest['vwap'] else 'bearish'
            
            # Generate signals
            long_signal = (
                trend == 'bullish' and
                latest['rsi'] < self.rsi_oversold
            )
            
            short_signal = (
                trend == 'bearish' and
                latest['rsi'] > self.rsi_overbought
            )
            
            return {
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
            macd_data = calculate_macd(
                klines,
                fast_period=self.macd_fast,
                slow_period=self.macd_slow,
                signal_period=self.macd_signal
            )
            bb_data = calculate_bollinger_bands(
                klines,
                period=self.bb_period,
                std=self.bb_std
            )
            
            # Get latest data
            latest = klines.iloc[-1]
            latest_macd = macd_data.iloc[-1]
            latest_bb = bb_data.iloc[-1]
            
            # Generate signals
            long_signal = (
                latest_macd['macd'] > latest_macd['signal'] and  # MACD crossover
                latest['close'] < latest_bb['lower_band']  # Price below lower BB
            )
            
            short_signal = (
                latest_macd['macd'] < latest_macd['signal'] and  # MACD crossunder
                latest['close'] > latest_bb['upper_band']  # Price above upper BB
            )
            
            return {
                'timestamp': datetime.now(),
                'symbol': self.symbol,
                'price': latest['close'],
                'macd': latest_macd['macd'],
                'macd_signal': latest_macd['signal'],
                'bb_upper': latest_bb['upper_band'],
                'bb_lower': latest_bb['lower_band'],
                'long_signal': long_signal,
                'short_signal': short_signal
            }
            
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
            
            # Calculate recent high/low
            recent_high = klines['high'].rolling(window=self.breakout_period).max()
            recent_low = klines['low'].rolling(window=self.breakout_period).min()
            
            # Get latest data
            latest = klines.iloc[-1]
            
            # Generate signals
            long_signal = (
                latest['close'] > recent_high.iloc[-2] and  # Break above recent high
                latest['volume_spike'] and  # Confirmed by volume
                latest['close'] > latest['open']  # Green candle
            )
            
            short_signal = (
                latest['close'] < recent_low.iloc[-2] and  # Break below recent low
                latest['volume_spike'] and  # Confirmed by volume
                latest['close'] < latest['open']  # Red candle
            )
            
            return {
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
            
        except Exception as e:
            logger.error(f"Error analyzing market: {e}")
            if self.telegram_notifier:
                self.telegram_notifier.notify_error(f"Error analyzing market: {e}")
            raise 