import logging
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from binance.client import Client
from binance.exceptions import BinanceAPIException
import traceback

logger = logging.getLogger(__name__)

class TradingBot:
    def __init__(self, config, data_collector, model_manager, telegram, is_test_mode=True):
        self.config = config
        self.data_collector = data_collector
        self.model_manager = model_manager
        self.telegram = telegram
        self.is_test_mode = is_test_mode
        
        # Create Binance client with API keys
        self.client = Client(config.binance_api_key, config.binance_api_secret)
        
        # Bot state
        self.active_positions = {}  # symbol -> position details
        self.daily_pnl = 0
        self.start_balance = self._get_account_balance() if not is_test_mode else config.initial_balance
        self.current_balance = self.start_balance
        self.daily_trades = 0
        self.total_trades = 0
        self.daily_start_time = datetime.now()
        
        # Initialize bot
        self._initialize()
        
    def _initialize(self):
        """Initialize the bot and send startup notification"""
        # Check if we're in test mode
        mode_str = "TEST MODE" if self.is_test_mode else "LIVE TRADING"
        
        # Send welcome message
        welcome_msg = (
            f"🤖 *AI Trading Bot Started*\n"
            f"Mode: {mode_str}\n"
            f"Symbol: {self.data_collector.symbol}\n"
            f"Timeframe: {self.data_collector.interval}\n"
            f"Starting Balance: {self.start_balance:.2f} USDT\n"
            f"Max Open Positions: {self.config.max_open_positions}\n"
            f"Leverage: {self.config.initial_leverage}x\n"
            f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        self.telegram.send_message(welcome_msg)
        
        # If not in test mode, check and set leverage on Binance
        if not self.is_test_mode:
            try:
                # Set leverage for the symbol
                self.client.futures_change_leverage(
                    symbol=self.data_collector.symbol,
                    leverage=self.config.initial_leverage
                )
                logger.info(f"Leverage set to {self.config.initial_leverage}x for {self.data_collector.symbol}")
            except BinanceAPIException as e:
                logger.error(f"Failed to set leverage: {e}")
                self.telegram.send_message(f"⚠️ Failed to set leverage: {str(e)}")
    
    def _get_account_balance(self):
        """Get account balance from Binance"""
        if self.is_test_mode:
            return self.config.initial_balance
            
        try:
            account_info = self.client.futures_account()
            for asset in account_info['assets']:
                if asset['asset'] == 'USDT':
                    return float(asset['walletBalance'])
            return self.config.initial_balance  # Default if not found
        except BinanceAPIException as e:
            logger.error(f"Failed to get account balance: {e}")
            return self.config.initial_balance
    
    def _get_active_positions(self):
        """Get currently open positions from Binance"""
        if self.is_test_mode:
            return self.active_positions
            
        try:
            positions = self.client.futures_position_information()
            active_positions = {}
            
            for position in positions:
                symbol = position['symbol']
                position_amt = float(position['positionAmt'])
                
                if position_amt != 0:  # Position exists
                    active_positions[symbol] = {
                        'side': 'LONG' if position_amt > 0 else 'SHORT',
                        'size': abs(position_amt),
                        'entry_price': float(position['entryPrice']),
                        'mark_price': float(position['markPrice']),
                        'pnl': float(position['unRealizedProfit']),
                        'leverage': int(position['leverage'])
                    }
            
            return active_positions
        except BinanceAPIException as e:
            logger.error(f"Failed to get active positions: {e}")
            return {}
    
    def _execute_trade(self, symbol, side, quantity):
        """Execute a trade on Binance"""
        if self.is_test_mode:
            logger.info(f"TEST MODE - Would execute {side} order for {quantity} {symbol}")
            
            # Simulate adding position
            current_price = float(self.client.get_symbol_ticker(symbol=symbol)['price'])
            
            position = {
                'side': side,
                'size': quantity,
                'entry_price': current_price,
                'mark_price': current_price,
                'pnl': 0.0,
                'leverage': self.config.initial_leverage,
                'entry_time': datetime.now()
            }
            
            self.active_positions[symbol] = position
            
            self.telegram.send_message(
                f"🔄 *TEST TRADE EXECUTED*\n"
                f"Symbol: {symbol}\n"
                f"Side: {side}\n"
                f"Quantity: {quantity}\n"
                f"Price: {current_price}\n"
                f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            return True
        
        try:
            # Prepare order parameters
            order_side = 'BUY' if side == 'LONG' else 'SELL'
            
            # Execute market order
            order = self.client.futures_create_order(
                symbol=symbol,
                side=order_side,
                type='MARKET',
                quantity=quantity
            )
            
            logger.info(f"Order executed: {order}")
            
            # Send notification
            self.telegram.send_message(
                f"✅ *TRADE EXECUTED*\n"
                f"Symbol: {symbol}\n"
                f"Side: {side}\n"
                f"Quantity: {quantity}\n"
                f"Order ID: {order['orderId']}\n"
                f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            
            return True
        except BinanceAPIException as e:
            logger.error(f"Order execution failed: {e}")
            self.telegram.send_message(f"⚠️ Order execution failed: {str(e)}")
            return False
    
    def _close_position(self, symbol):
        """Close an existing position"""
        if symbol not in self.active_positions:
            logger.warning(f"No active position for {symbol} to close")
            return False
        
        position = self.active_positions[symbol]
        
        if self.is_test_mode:
            logger.info(f"TEST MODE - Would close {position['side']} position for {symbol}")
            
            # Calculate PnL for test mode
            current_price = float(self.client.get_symbol_ticker(symbol=symbol)['price'])
            entry_price = position['entry_price']
            size = position['size']
            leverage = position['leverage']
            
            # Calculate PnL
            if position['side'] == 'LONG':
                pnl = (current_price - entry_price) / entry_price * size * leverage
            else:
                pnl = (entry_price - current_price) / entry_price * size * leverage
            
            self.daily_pnl += pnl
            self.current_balance += pnl
            self.daily_trades += 1
            self.total_trades += 1
            
            # Remove position
            del self.active_positions[symbol]
            
            # Send notification
            self.telegram.send_message(
                f"🔄 *TEST POSITION CLOSED*\n"
                f"Symbol: {symbol}\n"
                f"Side: {position['side']}\n"
                f"PnL: {pnl:.2f} USDT\n"
                f"Entry: {entry_price}\n"
                f"Exit: {current_price}\n"
                f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            
            return True
        
        try:
            # Determine order side (opposite of position side)
            order_side = 'SELL' if position['side'] == 'LONG' else 'BUY'
            
            # Execute market order to close
            order = self.client.futures_create_order(
                symbol=symbol,
                side=order_side,
                type='MARKET',
                quantity=position['size'],
                reduceOnly=True
            )
            
            logger.info(f"Position closed: {order}")
            
            # Get PnL
            pnl = position['pnl']
            self.daily_pnl += pnl
            self.daily_trades += 1
            self.total_trades += 1
            
            # Send notification
            self.telegram.send_message(
                f"✅ *POSITION CLOSED*\n"
                f"Symbol: {symbol}\n"
                f"Side: {position['side']}\n"
                f"PnL: {pnl:.2f} USDT\n"
                f"Order ID: {order['orderId']}\n"
                f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            
            # Remove from active positions
            del self.active_positions[symbol]
            
            return True
        except BinanceAPIException as e:
            logger.error(f"Failed to close position: {e}")
            self.telegram.send_message(f"⚠️ Failed to close position: {str(e)}")
            return False
    
    def _check_daily_reset(self):
        """Check if we need to reset daily metrics"""
        now = datetime.now()
        if (now - self.daily_start_time).days > 0:
            # It's a new day, send summary and reset
            daily_summary = (
                f"📊 *Daily Trading Summary*\n"
                f"Date: {self.daily_start_time.strftime('%Y-%m-%d')}\n"
                f"PnL: {self.daily_pnl:.2f} USDT\n"
                f"Trades: {self.daily_trades}\n"
                f"Win Rate: {self._calculate_win_rate():.1f}%\n"
                f"Current Balance: {self.current_balance:.2f} USDT\n"
                f"Daily Return: {(self.daily_pnl / self.start_balance) * 100:.2f}%"
            )
            
            self.telegram.send_message(daily_summary)
            
            # Reset daily metrics
            self.daily_pnl = 0
            self.daily_trades = 0
            self.daily_start_time = now
    
    def _check_drawdown(self):
        """Check if we've exceeded the max daily drawdown"""
        # Calculate drawdown as percentage of starting balance
        drawdown_pct = abs(self.daily_pnl) / self.start_balance if self.daily_pnl < 0 else 0
        
        if drawdown_pct > self.config.max_daily_drawdown:
            logger.warning(f"Max daily drawdown exceeded: {drawdown_pct:.2%}")
            self.telegram.send_message(
                f"⚠️ *RISK ALERT: Max Daily Drawdown Exceeded*\n"
                f"Current Drawdown: {drawdown_pct:.2%}\n"
                f"Max Allowed: {self.config.max_daily_drawdown:.2%}\n"
                f"Trading has been paused for today."
            )
            return True
        
        return False
    
    def _calculate_position_size(self, symbol):
        """Calculate position size based on risk parameters"""
        try:
            # Get current price
            current_price = float(self.client.get_symbol_ticker(symbol=symbol)['price'])
            
            # Get latest data for ATR calculation
            latest_data = self.data_collector.get_latest_data(limit=20)
            atr = latest_data['atr'].iloc[-1]
            
            # Calculate risk amount (% of balance)
            risk_amount = self.current_balance * self.config.position_size_percent
            
            # Calculate stop loss distance based on ATR
            sl_distance = atr * self.config.stop_loss_atr_multiplier
            
            # Calculate position size
            position_size_usd = risk_amount * self.config.initial_leverage
            position_size_qty = position_size_usd / current_price
            
            # Round to appropriate precision
            symbol_info = self.client.get_symbol_info(symbol)
            precision = 0
            
            for filter in symbol_info['filters']:
                if filter['filterType'] == 'LOT_SIZE':
                    step_size = float(filter['stepSize'])
                    precision = len(str(step_size).split('.')[1]) if '.' in str(step_size) else 0
                    break
            
            position_size_qty = round(position_size_qty, precision)
            
            return position_size_qty, current_price
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            traceback.print_exc()
            return 0, 0
    
    def _calculate_win_rate(self):
        """Calculate win rate from today's trades"""
        if self.daily_trades == 0:
            return 0
        
        # For this simple implementation, we'll just estimate
        # In a real system, you would track each trade result
        winning_trades = max(0, round(self.daily_pnl / abs(self.daily_pnl) * self.daily_trades * 0.5 + self.daily_trades * 0.5)) if self.daily_pnl != 0 else 0
        
        return (winning_trades / self.daily_trades) * 100 if self.daily_trades > 0 else 0
    
    def _can_open_new_position(self, symbol):
        """Check if we can open a new position based on current constraints"""
        # Check if we're already at max positions
        if len(self.active_positions) >= self.config.max_open_positions:
            return False
        
        # Check if we already have a position for this symbol
        if symbol in self.active_positions:
            return False
        
        # Check if we've exceeded daily drawdown
        if self._check_drawdown():
            return False
        
        # Check VWAP condition to avoid trading in consolidation
        latest_data = self.data_collector.get_latest_data(limit=5)
        current_price = float(self.client.get_symbol_ticker(symbol=symbol)['price'])
        vwap = latest_data['vwap'].iloc[-1]
        
        # If price is too close to VWAP (within 0.5%), avoid trading
        if abs(current_price - vwap) / vwap < 0.005:
            logger.info(f"Price too close to VWAP, avoiding trade")
            return False
            
        return True
        
    def run(self):
        """Main bot loop"""
        logger.info("Starting trading bot loop")
        
        try:
            while True:
                try:
                    # Check if we need to reset daily metrics
                    self._check_daily_reset()
                    
                    # Update active positions
                    if not self.is_test_mode:
                        self.active_positions = self._get_active_positions()
                    
                    # Get current market data
                    market_data = self.data_collector.get_latest_data()
                    symbol = self.data_collector.symbol
                    
                    # Get current account state for the RL model
                    account_state = {
                        'balance': self.current_balance,
                        'position': 1 if symbol in self.active_positions and self.active_positions[symbol]['side'] == 'LONG' else 
                                   -1 if symbol in self.active_positions and self.active_positions[symbol]['side'] == 'SHORT' else 0,
                        'position_price': self.active_positions.get(symbol, {}).get('entry_price', 0),
                        'position_size_usd': self.active_positions.get(symbol, {}).get('size', 0) * 
                                            self.active_positions.get(symbol, {}).get('entry_price', 0)
                    }
                    
                    # Get trading action from model
                    action = self.model_manager.get_trading_action(market_data, account_state)
                    
                    # Map action to trading decision
                    # action: 0 (do nothing), 1 (close position), 2 (open long), 3 (open short)
                    if action == 1 and symbol in self.active_positions:
                        # Close position
                        logger.info(f"Model suggests closing position for {symbol}")
                        self._close_position(symbol)
                        
                    elif action == 2 and self._can_open_new_position(symbol):
                        # Open long position
                        logger.info(f"Model suggests opening LONG position for {symbol}")
                        quantity, price = self._calculate_position_size(symbol)
                        
                        if quantity > 0:
                            self._execute_trade(symbol, 'LONG', quantity)
                            
                    elif action == 3 and self._can_open_new_position(symbol):
                        # Open short position
                        logger.info(f"Model suggests opening SHORT position for {symbol}")
                        quantity, price = self._calculate_position_size(symbol)
                        
                        if quantity > 0:
                            self._execute_trade(symbol, 'SHORT', quantity)
                    
                    # Wait before next iteration
                    # Adjust based on timeframe
                    sleep_time = 60  # 1 minute default
                    if self.data_collector.interval == '1m':
                        sleep_time = 15
                    elif self.data_collector.interval == '5m':
                        sleep_time = 60
                    elif self.data_collector.interval == '15m':
                        sleep_time = 180
                    elif self.data_collector.interval == '1h':
                        sleep_time = 600
                    
                    logger.info(f"Sleeping for {sleep_time} seconds...")
                    time.sleep(sleep_time)
                    
                except Exception as e:
                    logger.error(f"Error in trading loop: {e}")
                    traceback.print_exc()
                    self.telegram.send_message(f"⚠️ Error in trading loop: {str(e)}")
                    time.sleep(60)  # Sleep before retrying
        
        except KeyboardInterrupt:
            logger.info("Bot stopped by user")
            self.telegram.send_message("🛑 Bot stopped manually")
