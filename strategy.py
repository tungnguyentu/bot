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
    calculate_order_book_imbalance
)
from utils import (
    calculate_take_profit_price,
    calculate_stop_loss_price,
    calculate_atr_stop_loss,
    save_trade_history
)

# Initialize logger
logger = logging.getLogger('trading_bot')


class ScalpingStrategy:
    """
    Scalping strategy for the trading bot.
    """
    
    def __init__(self, binance_client, telegram_notifier=None):
        """
        Initialize the scalping strategy.
        
        Args:
            binance_client (BinanceClient): Binance client
            telegram_notifier (TelegramNotifier, optional): Telegram notifier
        """
        self.binance_client = binance_client
        self.telegram_notifier = telegram_notifier
        self.symbol = config.SYMBOL
        self.timeframe = config.TIMEFRAME
        self.active_positions = {}
        
        # Strategy parameters
        self.rsi_period = config.RSI_PERIOD
        self.rsi_overbought = config.RSI_OVERBOUGHT
        self.rsi_oversold = config.RSI_OVERSOLD
        self.vwap_period = config.VWAP_PERIOD
        self.atr_period = config.ATR_PERIOD
        self.volume_threshold = config.VOLUME_THRESHOLD
        
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
        
        logger.info("Scalping strategy initialized.")
    
    def analyze_market(self):
        """
        Analyze market data and generate trading signals.
        
        Returns:
            dict: Analysis results
        """
        try:
            # Get market data
            klines = self.binance_client.get_klines(
                symbol=self.symbol,
                timeframe=self.timeframe,
                limit=100
            )
            
            # Get order book
            order_book = self.binance_client.get_order_book(
                symbol=self.symbol,
                limit=20
            )
            
            # Calculate indicators
            klines['rsi'] = calculate_rsi(klines, period=self.rsi_period)
            klines['vwap'] = calculate_vwap(klines, period=self.vwap_period)
            klines['atr'] = calculate_atr(klines, period=self.atr_period)
            klines['volume_spike'] = detect_volume_spike(klines, threshold=self.volume_threshold)
            
            # Calculate order book imbalance
            order_book_imbalance = calculate_order_book_imbalance(order_book)
            
            # Get latest data
            latest = klines.iloc[-1]
            
            # Determine trend based on VWAP
            trend = 'bullish' if latest['close'] > latest['vwap'] else 'bearish'
            
            # Check for entry conditions - removed volume spike requirement
            long_signal = (
                trend == 'bullish' and
                latest['rsi'] < self.rsi_oversold
                # Volume spike requirement removed
            )
            
            short_signal = (
                trend == 'bearish' and
                latest['rsi'] > self.rsi_overbought
                # Volume spike requirement removed
            )
            
            # Prepare analysis results
            analysis = {
                'timestamp': datetime.now(),
                'symbol': self.symbol,
                'price': latest['close'],
                'rsi': latest['rsi'],
                'vwap': latest['vwap'],
                'atr': latest['atr'],
                'volume': latest['volume'],
                'volume_spike': latest['volume_spike'],
                'order_book_imbalance': order_book_imbalance,
                'trend': trend,
                'long_signal': long_signal,
                'short_signal': short_signal
            }
            
            logger.info(f"Market analysis completed for {self.symbol}.")
            logger.debug(f"Analysis results: {analysis}")
            
            return analysis
        
        except Exception as e:
            logger.error(f"Error analyzing market: {e}")
            if self.telegram_notifier:
                self.telegram_notifier.notify_error(f"Error analyzing market: {e}")
            raise
    
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
        
        Args:
            analysis (dict): Market analysis results
            
        Returns:
            dict: Position details
        """
        try:
            # Get account balance
            balance = self.binance_client.get_account_balance()
            usdt_balance = next((item for item in balance['info']['assets'] if item['asset'] == 'USDT'), None)
            
            if not usdt_balance:
                logger.error("Could not find USDT balance.")
                return {'action': 'none', 'reason': 'no_balance'}
            
            available_balance = float(usdt_balance['availableBalance'])
            
            # Calculate position size
            position_amount = available_balance * self.position_size
            
            # Set leverage
            self.binance_client.set_leverage(self.symbol, config.LEVERAGE)
            
            # Calculate entry price
            entry_price = analysis['price']
            
            # Calculate stop loss price
            if self.use_atr_for_sl:
                stop_loss_price = calculate_atr_stop_loss(
                    entry_price=entry_price,
                    atr_value=analysis['atr'],
                    atr_multiplier=self.atr_multiplier,
                    position_type='long'
                )
            else:
                stop_loss_price = calculate_stop_loss_price(
                    entry_price=entry_price,
                    stop_loss_percent=self.stop_loss_percent,
                    position_type='long'
                )
            
            # Calculate take profit price
            take_profit_price = calculate_take_profit_price(
                entry_price=entry_price,
                take_profit_percent=self.take_profit_percent,
                position_type='long'
            )
            
            # Calculate position size in contracts
            price_precision = 8  # Default precision
            quantity_precision = 3  # Default precision
            
            # Get market info for precision
            markets = self.binance_client.exchange.load_markets()
            if self.symbol in markets:
                market = markets[self.symbol]
                price_precision = market['precision']['price']
                quantity_precision = market['precision']['amount']
            
            # Calculate quantity
            quantity = position_amount / entry_price
            quantity = round(quantity, quantity_precision)
            
            # Open position with market order
            order = self.binance_client.create_market_order(
                symbol=self.symbol,
                side='buy',
                amount=quantity
            )
            
            # Store position details
            position_id = f"long_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            position = {
                'id': position_id,
                'symbol': self.symbol,
                'type': 'long',
                'entry_price': entry_price,
                'quantity': quantity,
                'stop_loss': stop_loss_price,
                'take_profit': take_profit_price,
                'entry_time': datetime.now(),
                'entry_order': order,
                'status': 'open',
                'trailing_stop_activated': False,
                'trailing_stop_price': None
            }
            
            self.active_positions[position_id] = position
            
            # Send notification
            if self.telegram_notifier and config.NOTIFY_ON_TRADE_OPEN:
                self.telegram_notifier.notify_trade_open(
                    symbol=self.symbol,
                    position_type='long',
                    entry_price=entry_price,
                    position_size=quantity,
                    stop_loss=stop_loss_price,
                    take_profit=take_profit_price
                )
            
            logger.info(f"Opened long position for {self.symbol} at {entry_price}.")
            
            # Save trade history
            save_trade_history({
                'id': position_id,
                'symbol': self.symbol,
                'type': 'long',
                'entry_price': float(entry_price),
                'quantity': float(quantity),
                'stop_loss': float(stop_loss_price),
                'take_profit': float(take_profit_price),
                'entry_time': datetime.now().isoformat(),
                'status': 'open'
            })
            
            return {
                'action': 'open_long',
                'position': position
            }
        
        except Exception as e:
            logger.error(f"Error opening long position: {e}")
            if self.telegram_notifier:
                self.telegram_notifier.notify_error(f"Error opening long position: {e}")
            raise
    
    def open_short_position(self, analysis):
        """
        Open a short position.
        
        Args:
            analysis (dict): Market analysis results
            
        Returns:
            dict: Position details
        """
        try:
            # Get account balance
            balance = self.binance_client.get_account_balance()
            usdt_balance = next((item for item in balance['info']['assets'] if item['asset'] == 'USDT'), None)
            
            if not usdt_balance:
                logger.error("Could not find USDT balance.")
                return {'action': 'none', 'reason': 'no_balance'}
            
            available_balance = float(usdt_balance['availableBalance'])
            
            # Calculate position size
            position_amount = available_balance * self.position_size
            
            # Set leverage
            self.binance_client.set_leverage(self.symbol, config.LEVERAGE)
            
            # Calculate entry price
            entry_price = analysis['price']
            
            # Calculate stop loss price
            if self.use_atr_for_sl:
                stop_loss_price = calculate_atr_stop_loss(
                    entry_price=entry_price,
                    atr_value=analysis['atr'],
                    atr_multiplier=self.atr_multiplier,
                    position_type='short'
                )
            else:
                stop_loss_price = calculate_stop_loss_price(
                    entry_price=entry_price,
                    stop_loss_percent=self.stop_loss_percent,
                    position_type='short'
                )
            
            # Calculate take profit price
            take_profit_price = calculate_take_profit_price(
                entry_price=entry_price,
                take_profit_percent=self.take_profit_percent,
                position_type='short'
            )
            
            # Calculate position size in contracts
            price_precision = 8  # Default precision
            quantity_precision = 3  # Default precision
            
            # Get market info for precision
            markets = self.binance_client.exchange.load_markets()
            if self.symbol in markets:
                market = markets[self.symbol]
                price_precision = market['precision']['price']
                quantity_precision = market['precision']['amount']
            
            # Calculate quantity
            quantity = position_amount / entry_price
            quantity = round(quantity, quantity_precision)
            
            # Open position with market order
            order = self.binance_client.create_market_order(
                symbol=self.symbol,
                side='sell',
                amount=quantity
            )
            
            # Store position details
            position_id = f"short_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            position = {
                'id': position_id,
                'symbol': self.symbol,
                'type': 'short',
                'entry_price': entry_price,
                'quantity': quantity,
                'stop_loss': stop_loss_price,
                'take_profit': take_profit_price,
                'entry_time': datetime.now(),
                'entry_order': order,
                'status': 'open',
                'trailing_stop_activated': False,
                'trailing_stop_price': None
            }
            
            self.active_positions[position_id] = position
            
            # Send notification
            if self.telegram_notifier and config.NOTIFY_ON_TRADE_OPEN:
                self.telegram_notifier.notify_trade_open(
                    symbol=self.symbol,
                    position_type='short',
                    entry_price=entry_price,
                    position_size=quantity,
                    stop_loss=stop_loss_price,
                    take_profit=take_profit_price
                )
            
            logger.info(f"Opened short position for {self.symbol} at {entry_price}.")
            
            # Save trade history
            save_trade_history({
                'id': position_id,
                'symbol': self.symbol,
                'type': 'short',
                'entry_price': float(entry_price),
                'quantity': float(quantity),
                'stop_loss': float(stop_loss_price),
                'take_profit': float(take_profit_price),
                'entry_time': datetime.now().isoformat(),
                'status': 'open'
            })
            
            return {
                'action': 'open_short',
                'position': position
            }
        
        except Exception as e:
            logger.error(f"Error opening short position: {e}")
            if self.telegram_notifier:
                self.telegram_notifier.notify_error(f"Error opening short position: {e}")
            raise
    
    def manage_positions(self):
        """
        Manage open positions (check for take profit, stop loss, trailing stop).
        
        Returns:
            list: List of actions taken
        """
        if not self.active_positions:
            return []
        
        actions = []
        
        try:
            # Get current price
            klines = self.binance_client.get_klines(
                symbol=self.symbol,
                timeframe='1m',
                limit=1
            )
            current_price = klines.iloc[-1]['close']
            
            # Check each position
            positions_to_remove = []
            
            for position_id, position in self.active_positions.items():
                if position['status'] != 'open':
                    continue
                
                position_type = position['type']
                entry_price = position['entry_price']
                stop_loss = position['stop_loss']
                take_profit = position['take_profit']
                
                # Check if position should be closed
                if position_type == 'long':
                    # Check for stop loss
                    if current_price <= stop_loss:
                        self.close_position(position_id, 'stop_loss', current_price)
                        positions_to_remove.append(position_id)
                        actions.append({
                            'action': 'close_long',
                            'reason': 'stop_loss',
                            'position_id': position_id,
                            'price': current_price
                        })
                        continue
                    
                    # Check for take profit
                    if current_price >= take_profit:
                        self.close_position(position_id, 'take_profit', current_price)
                        positions_to_remove.append(position_id)
                        actions.append({
                            'action': 'close_long',
                            'reason': 'take_profit',
                            'position_id': position_id,
                            'price': current_price
                        })
                        continue
                    
                    # Check for trailing stop
                    if self.use_trailing_stop:
                        # Calculate profit percentage
                        profit_percent = (current_price - entry_price) / entry_price
                        
                        # Check if trailing stop should be activated
                        if profit_percent >= self.trailing_stop_activation:
                            if not position['trailing_stop_activated']:
                                # Activate trailing stop
                                position['trailing_stop_activated'] = True
                                position['trailing_stop_price'] = current_price * (1 - self.trailing_stop_callback)
                                logger.info(f"Trailing stop activated for position {position_id} at {position['trailing_stop_price']}.")
                            else:
                                # Update trailing stop if price moves up
                                new_trailing_stop = current_price * (1 - self.trailing_stop_callback)
                                if new_trailing_stop > position['trailing_stop_price']:
                                    position['trailing_stop_price'] = new_trailing_stop
                                    logger.info(f"Trailing stop updated for position {position_id} to {position['trailing_stop_price']}.")
                            
                            # Check if price hits trailing stop
                            if position['trailing_stop_activated'] and current_price <= position['trailing_stop_price']:
                                self.close_position(position_id, 'trailing_stop', current_price)
                                positions_to_remove.append(position_id)
                                actions.append({
                                    'action': 'close_long',
                                    'reason': 'trailing_stop',
                                    'position_id': position_id,
                                    'price': current_price
                                })
                
                elif position_type == 'short':
                    # Check for stop loss
                    if current_price >= stop_loss:
                        self.close_position(position_id, 'stop_loss', current_price)
                        positions_to_remove.append(position_id)
                        actions.append({
                            'action': 'close_short',
                            'reason': 'stop_loss',
                            'position_id': position_id,
                            'price': current_price
                        })
                        continue
                    
                    # Check for take profit
                    if current_price <= take_profit:
                        self.close_position(position_id, 'take_profit', current_price)
                        positions_to_remove.append(position_id)
                        actions.append({
                            'action': 'close_short',
                            'reason': 'take_profit',
                            'position_id': position_id,
                            'price': current_price
                        })
                        continue
                    
                    # Check for trailing stop
                    if self.use_trailing_stop:
                        # Calculate profit percentage
                        profit_percent = (entry_price - current_price) / entry_price
                        
                        # Check if trailing stop should be activated
                        if profit_percent >= self.trailing_stop_activation:
                            if not position['trailing_stop_activated']:
                                # Activate trailing stop
                                position['trailing_stop_activated'] = True
                                position['trailing_stop_price'] = current_price * (1 + self.trailing_stop_callback)
                                logger.info(f"Trailing stop activated for position {position_id} at {position['trailing_stop_price']}.")
                            else:
                                # Update trailing stop if price moves down
                                new_trailing_stop = current_price * (1 + self.trailing_stop_callback)
                                if new_trailing_stop < position['trailing_stop_price']:
                                    position['trailing_stop_price'] = new_trailing_stop
                                    logger.info(f"Trailing stop updated for position {position_id} to {position['trailing_stop_price']}.")
                            
                            # Check if price hits trailing stop
                            if position['trailing_stop_activated'] and current_price >= position['trailing_stop_price']:
                                self.close_position(position_id, 'trailing_stop', current_price)
                                positions_to_remove.append(position_id)
                                actions.append({
                                    'action': 'close_short',
                                    'reason': 'trailing_stop',
                                    'position_id': position_id,
                                    'price': current_price
                                })
            
            # Remove closed positions
            for position_id in positions_to_remove:
                del self.active_positions[position_id]
            
            return actions
        
        except Exception as e:
            logger.error(f"Error managing positions: {e}")
            if self.telegram_notifier:
                self.telegram_notifier.notify_error(f"Error managing positions: {e}")
            return []
    
    def close_position(self, position_id, reason, exit_price):
        """
        Close a position.
        
        Args:
            position_id (str): Position ID
            reason (str): Reason for closing position
            exit_price (float): Exit price
            
        Returns:
            bool: True if position was closed successfully, False otherwise
        """
        try:
            position = self.active_positions.get(position_id)
            
            if not position:
                logger.warning(f"Position {position_id} not found.")
                return False
            
            # Close position with market order
            side = 'sell' if position['type'] == 'long' else 'buy'
            
            order = self.binance_client.create_market_order(
                symbol=self.symbol,
                side=side,
                amount=position['quantity']
            )
            
            # Calculate profit/loss
            if position['type'] == 'long':
                profit_loss = (exit_price - position['entry_price']) * position['quantity']
                profit_loss_percent = (exit_price - position['entry_price']) / position['entry_price'] * 100
            else:  # short
                profit_loss = (position['entry_price'] - exit_price) * position['quantity']
                profit_loss_percent = (position['entry_price'] - exit_price) / position['entry_price'] * 100
            
            # Update position status
            position['status'] = 'closed'
            position['exit_price'] = exit_price
            position['exit_time'] = datetime.now()
            position['exit_order'] = order
            position['profit_loss'] = profit_loss
            position['profit_loss_percent'] = profit_loss_percent
            position['close_reason'] = reason
            
            # Send notification
            if self.telegram_notifier and config.NOTIFY_ON_TRADE_CLOSE:
                self.telegram_notifier.notify_trade_close(
                    symbol=self.symbol,
                    position_type=position['type'],
                    entry_price=position['entry_price'],
                    exit_price=exit_price,
                    profit_loss=profit_loss,
                    profit_loss_percent=profit_loss_percent
                )
            
            logger.info(f"Closed {position['type']} position for {self.symbol} at {exit_price} ({reason}).")
            logger.info(f"Profit/Loss: {profit_loss:.4f} ({profit_loss_percent:.2f}%).")
            
            # Save trade history
            save_trade_history({
                'id': position_id,
                'symbol': self.symbol,
                'type': position['type'],
                'entry_price': float(position['entry_price']),
                'exit_price': float(exit_price),
                'quantity': float(position['quantity']),
                'entry_time': position['entry_time'].isoformat(),
                'exit_time': datetime.now().isoformat(),
                'profit_loss': float(profit_loss),
                'profit_loss_percent': float(profit_loss_percent),
                'close_reason': reason,
                'status': 'closed'
            })
            
            return True
        
        except Exception as e:
            logger.error(f"Error closing position: {e}")
            if self.telegram_notifier:
                self.telegram_notifier.notify_error(f"Error closing position: {e}")
            return False 