import logging
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

# Add the parent directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logger = logging.getLogger(__name__)

class RiskManager:
    def __init__(self, client):
        """
        Initialize the Risk Manager.
        
        Args:
            client: Binance API client instance
        """
        self.client = client
        self.trades = []  # List of all trades for tracking performance
        self.open_positions = {}  # Dictionary of open positions
        self.initial_balance = None
        self.max_balance = None
        self.max_drawdown = 0
        self.load_trading_history()
        
    def load_trading_history(self):
        """Load trading history or initialize if first run."""
        # In a production system, this would load from database
        try:
            # Get account balance - using different methods for testnet vs production
            if config.TRADING_MODE == "paper_trading":
                # For testnet
                account_info = self.client.futures_account()
                usdt_balance = next((item['balance'] for item in account_info['assets'] if item['asset'] == 'USDT'), 0)
            else:
                # For production
                account_info = self.client.futures_account_balance()
                usdt_balance = next((item['balance'] for item in account_info if item['asset'] == 'USDT'), 0)
            
            self.initial_balance = float(usdt_balance)
            self.max_balance = self.initial_balance
            
            logger.info(f"Initial account balance: {self.initial_balance} USDT")
            
        except Exception as e:
            logger.error(f"Error loading account balance: {e}")
            self.initial_balance = 1000  # Default for backtesting
            self.max_balance = self.initial_balance
    
    def calculate_position_size(self, symbol, entry_price, stop_loss_price, max_risk_usd=None):
        """
        Calculate position size based on risk parameters.
        
        Args:
            symbol: Trading symbol
            entry_price: Entry price
            stop_loss_price: Stop loss price
            max_risk_usd: Maximum risk in USD (overrides percentage-based risk)
            
        Returns:
            float: Position size in base currency
        """
        try:
            # Get account balance - using different methods for testnet vs production
            if config.TRADING_MODE == "paper_trading":
                # For testnet
                account_info = self.client.futures_account()
                usdt_balance = next((item['balance'] for item in account_info['assets'] if item['asset'] == 'USDT'), 0)
            else:
                # For production
                account_info = self.client.futures_account_balance()
                usdt_balance = next((item['balance'] for item in account_info if item['asset'] == 'USDT'), 0)
                
            account_balance = float(usdt_balance)
            
            # Calculate risk amount
            risk_percentage = config.RISK_PER_TRADE
            risk_amount = account_balance * risk_percentage
            
            if max_risk_usd and max_risk_usd < risk_amount:
                risk_amount = max_risk_usd
                
            # Calculate price distance to stop loss
            price_risk = abs(entry_price - stop_loss_price)
            risk_per_unit = price_risk / entry_price
            
            # Calculate position size with leverage
            leverage = config.TRADING_LEVERAGE
            position_size = (risk_amount * leverage) / (entry_price * risk_per_unit)
            
            # Round down to appropriate precision
            symbol_info = self.client.get_symbol_info(symbol)
            lot_size_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'), None)
            
            if lot_size_filter:
                step_size = float(lot_size_filter['stepSize'])
                precision = len(str(step_size).split('.')[-1].rstrip('0'))
                position_size = np.floor(position_size * (10 ** precision)) / (10 ** precision)
            
            logger.info(f"Calculated position size for {symbol}: {position_size} units")
            return position_size
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0
    
    def check_max_drawdown(self):
        """
        Check if the current drawdown exceeds the maximum allowed drawdown.
        
        Returns:
            bool: True if maximum drawdown is exceeded
        """
        try:
            # Get current account balance - using different methods for testnet vs production
            if config.TRADING_MODE == "paper_trading":
                # For testnet
                account_info = self.client.futures_account()
                usdt_balance = next((item['balance'] for item in account_info['assets'] if item['asset'] == 'USDT'), 0)
            else:
                # For production
                account_info = self.client.futures_account_balance()
                usdt_balance = next((item['balance'] for item in account_info if item['asset'] == 'USDT'), 0)
                
            current_balance = float(usdt_balance)
            
            # Update max balance if current balance is higher
            if current_balance > self.max_balance:
                self.max_balance = current_balance
            
            # Calculate current drawdown
            if self.max_balance > 0:
                current_drawdown = (self.max_balance - current_balance) / self.max_balance
                if current_drawdown > self.max_drawdown:
                    self.max_drawdown = current_drawdown
                
                logger.info(f"Current drawdown: {current_drawdown:.2%}, Max drawdown: {self.max_drawdown:.2%}")
                
                # Check if drawdown exceeds maximum allowed
                if current_drawdown > config.MAX_DRAWDOWN_PERCENTAGE:
                    logger.warning(f"Maximum drawdown exceeded: {current_drawdown:.2%}. Trading halted.")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking drawdown: {e}")
            return False
    
    def can_open_new_trade(self, symbol):
        """
        Check if a new trade can be opened based on risk rules.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            bool: True if a new trade can be opened
        """
        # Check if max drawdown is exceeded
        if self.check_max_drawdown():
            return False
        
        # Check if we have too many open positions
        if len(self.open_positions) >= config.MAX_OPEN_TRADES:
            logger.info(f"Maximum number of open trades reached ({len(self.open_positions)}/{config.MAX_OPEN_TRADES})")
            return False
        
        # Check if the symbol is already being traded
        if symbol in self.open_positions:
            logger.info(f"Position already open for {symbol}")
            return False
            
        return True
        
    def record_trade(self, trade_data):
        """
        Record a completed trade for performance tracking.
        
        Args:
            trade_data: Dictionary with trade details
        """
        trade_data['timestamp'] = datetime.now()
        self.trades.append(trade_data)
        
        # In production, this would save to database
        logger.info(f"Trade recorded: {trade_data}")
        
    def register_open_position(self, symbol, position_data):
        """Register a new open position."""
        self.open_positions[symbol] = position_data
        logger.info(f"Open position registered for {symbol}: {position_data}")
        
    def unregister_position(self, symbol):
        """Remove a closed position from tracking."""
        if symbol in self.open_positions:
            del self.open_positions[symbol]
            logger.info(f"Position for {symbol} has been closed")
    
    def get_performance_metrics(self):
        """
        Calculate trading performance metrics.
        
        Returns:
            dict: Dictionary of performance metrics
        """
        if not self.trades:
            return {
                "total_trades": 0,
                "win_rate": 0,
                "profit_factor": 0,
                "max_drawdown": 0,
                "avg_profit_per_trade": 0
            }
        
        # Calculate metrics
        total_trades = len(self.trades)
        winning_trades = [t for t in self.trades if t.get('pnl', 0) > 0]
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        
        total_profit = sum([t.get('pnl', 0) for t in self.trades if t.get('pnl', 0) > 0])
        total_loss = abs(sum([t.get('pnl', 0) for t in self.trades if t.get('pnl', 0) < 0]))
        profit_factor = total_profit / total_loss if total_loss > 0 else 0
        
        avg_profit = sum([t.get('pnl', 0) for t in self.trades]) / total_trades if total_trades > 0 else 0
        
        return {
            "total_trades": total_trades,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "max_drawdown": self.max_drawdown,
            "avg_profit_per_trade": avg_profit
        }
