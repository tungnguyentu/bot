import logging
import pandas as pd
import numpy as np
from datetime import datetime
from binance.client import Client
from binance.exceptions import BinanceAPIException
import os
import uuid
import math
import time
import requests
from requests.exceptions import RequestException
from urllib3.exceptions import HTTPError
import http.client
from functools import wraps
from .config import BotConfig
from .notifier import TelegramNotifier

logger = logging.getLogger(__name__)

def retry_on_connection_error(max_retries=3, initial_delay=1, backoff_factor=2):
    """
    Decorator to retry functions on connection errors with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay between retries in seconds
        backoff_factor: Multiplier for delay on each retry
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None
            
            for retry in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except (BinanceAPIException, requests.exceptions.RequestException, 
                        http.client.RemoteDisconnected, ConnectionError, HTTPError) as e:
                    last_exception = e
                    
                    # Check if we should retry
                    retry_codes = [-1000, -1001, -1002, -1007, -1013, -1015, -1018, -1021, -1022, -2019]
                    
                    if isinstance(e, BinanceAPIException) and e.code not in retry_codes:
                        # Don't retry for specific API errors that won't benefit from retrying
                        raise e
                    
                    if retry < max_retries:
                        wait_time = delay * (backoff_factor ** retry)
                        logger.warning(f"Connection error: {e}. Retrying in {wait_time:.2f}s ({retry+1}/{max_retries})")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"Failed after {max_retries} retries: {e}")
                        raise last_exception
            
            return None  # This line should not be reached
        return wrapper
    return decorator

class Trader:
    """Handles trading execution on Binance."""
    
    def __init__(self, config: BotConfig, notifier: TelegramNotifier):
        """Initialize trader with configuration."""
        self.config = config
        self.notifier = notifier
        self.client = self._initialize_client()
        self.positions = {}  # track active positions
        self.trade_history = []  # track historical trades
        # Cache for symbol information
        self.symbol_info = {}
        # Initial account balance to use for trading (this is our margin amount)
        self.initial_balance = config.invest
        self.current_balance = config.invest
    
    @retry_on_connection_error(max_retries=3, initial_delay=1)
    def _initialize_client(self):
        """Initialize Binance client based on the trading mode."""
        api_key = os.getenv('BINANCE_API_KEY')
        api_secret = os.getenv('BINANCE_API_SECRET')
        
        # Use testnet credentials for test mode
        if self.config.mode == 'test':
            api_key = os.getenv('BINANCE_TESTNET_API_KEY', api_key)
            api_secret = os.getenv('BINANCE_TESTNET_API_SECRET', api_secret)
            return Client(api_key, api_secret, testnet=True)
        
        # Use regular API for live or backtest
        return Client(api_key, api_secret)
    
    @retry_on_connection_error()
    def get_current_position(self):
        """Get the current position for the configured symbol."""
        # In backtest mode, use tracked positions
        if self.config.mode == 'backtest':
            return self.positions.get(self.config.symbol)
        
        # In test or live mode, get position from Binance
        try:
            position_info = self.client.futures_position_information(symbol=self.config.symbol)
            
            for position in position_info:
                if float(position['positionAmt']) != 0:  # We have an open position
                    side = 'BUY' if float(position['positionAmt']) > 0 else 'SELL'
                    entry_price = float(position['entryPrice'])
                    
                    # We need to retrieve SL/TP from open orders since position doesn't have it
                    sl_price, tp_price = self._get_sl_tp_from_orders()
                    
                    return {
                        'id': self.config.symbol,
                        'symbol': self.config.symbol,
                        'side': side,
                        'entry_price': entry_price,
                        'quantity': abs(float(position['positionAmt'])),
                        'sl_price': sl_price,
                        'tp_price': tp_price,
                        'entry_time': datetime.now().timestamp()
                    }
            
            return None
        except BinanceAPIException as e:
            logger.error(f"Error getting current position: {e}")
            return None
    
    @retry_on_connection_error()
    def _get_sl_tp_from_orders(self):
        """Get SL and TP prices from open orders."""
        sl_price = None
        tp_price = None
        
        try:
            open_orders = self.client.futures_get_open_orders(symbol=self.config.symbol)
            
            for order in open_orders:
                if order['type'] == 'STOP_MARKET' or order['type'] == 'STOP':
                    sl_price = float(order['stopPrice'])
                elif order['type'] == 'TAKE_PROFIT_MARKET' or order['type'] == 'TAKE_PROFIT':
                    tp_price = float(order['stopPrice'])
            
            return sl_price, tp_price
        except BinanceAPIException as e:
            logger.error(f"Error getting SL/TP from orders: {e}")
            return None, None
    
    @retry_on_connection_error()
    def _get_symbol_info(self, symbol):
        """
        Get symbol information and cache it.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dictionary with symbol information
        """
        if symbol in self.symbol_info:
            return self.symbol_info[symbol]
        
        try:
            # Get exchange information
            if self.config.mode == 'test' or self.config.mode == 'live':
                # For futures trading
                exchange_info = self.client.futures_exchange_info()
                
                # Find the symbol information
                for sym_info in exchange_info['symbols']:
                    if sym_info['symbol'] == symbol:
                        self.symbol_info[symbol] = sym_info
                        logger.info(f"Found symbol info for {symbol}")
                        return sym_info
            else:
                # For spot trading (used in backtest)
                exchange_info = self.client.get_exchange_info()
                
                # Find the symbol information
                for sym_info in exchange_info['symbols']:
                    if sym_info['symbol'] == symbol:
                        self.symbol_info[symbol] = sym_info
                        logger.info(f"Found symbol info for {symbol}")
                        return sym_info
            
            logger.error(f"Symbol {symbol} not found in exchange info")
            return None
        except BinanceAPIException as e:
            logger.error(f"Error getting symbol info: {e}")
            return None
    
    def _adjust_quantity_precision(self, symbol, quantity):
        """
        Adjust the quantity to the correct precision for the symbol.
        
        Args:
            symbol: Trading symbol
            quantity: Original quantity
            
        Returns:
            Adjusted quantity with correct precision
        """
        # Get symbol info
        symbol_info = self._get_symbol_info(symbol)
        
        if not symbol_info:
            logger.warning(f"No symbol info found for {symbol}, using original quantity")
            return quantity
        
        # Check if we're using futures
        if self.config.mode == 'test' or self.config.mode == 'live':
            # For futures trading
            # Find the LOT_SIZE filter which defines the precision
            lot_size_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'), None)
            
            if lot_size_filter:
                step_size = float(lot_size_filter['stepSize'])
                
                # Calculate precision
                precision = 0
                if step_size < 1:
                    precision = len(str(step_size).split('.')[-1].rstrip('0'))
                
                # Calculate the correct quantity
                quantity = float(quantity)
                adjusted_quantity = math.floor(quantity / step_size) * step_size
                adjusted_quantity = round(adjusted_quantity, precision)
                
                logger.info(f"Futures: Adjusted quantity from {quantity} to {adjusted_quantity} for {symbol} (precision: {precision}, step size: {step_size})")
                return "{:.{}f}".format(adjusted_quantity, precision)  # Return as string with correct precision
        else:
            # For spot trading (used in backtest)
            lot_size_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'), None)
            
            if lot_size_filter:
                min_qty = float(lot_size_filter['minQty'])
                step_size = float(lot_size_filter['stepSize'])
                
                # Calculate precision based on step size
                precision = 0
                if step_size < 1:
                    precision = len(str(step_size).split('.')[-1].rstrip('0'))
                
                # Calculate the correct quantity
                quantity = float(quantity)
                adjusted_quantity = math.floor(quantity / step_size) * step_size
                if adjusted_quantity < min_qty:
                    adjusted_quantity = min_qty
                
                # Format to the correct precision
                adjusted_quantity = round(adjusted_quantity, precision)
                
                logger.info(f"Spot: Adjusted quantity from {quantity} to {adjusted_quantity} for {symbol} (precision: {precision})")
                return adjusted_quantity
        
        # If we reach here, no appropriate filter was found
        logger.warning(f"No appropriate filter found for {symbol}, using original quantity")
        return quantity
    
    @retry_on_connection_error()
    def open_position(self, side, quantity, entry_price, sl_price, tp_price):
        """
        Open a new position.
        
        Args:
            side: 'BUY' or 'SELL'
            quantity: Position size
            entry_price: Entry price
            sl_price: Stop-loss price
            tp_price: Take-profit price
            
        Returns:
            Dictionary with position details if successful, None otherwise
        """
        if self.config.mode == 'backtest':
            # In backtest mode, just track the position
            position_id = f"{self.config.symbol}_{datetime.now().timestamp()}"
            position = {
                'id': position_id,
                'symbol': self.config.symbol,
                'side': side,
                'entry_price': entry_price,
                'quantity': quantity,
                'sl_price': sl_price,
                'tp_price': tp_price,
                'entry_time': datetime.now().timestamp()
            }
            self.positions[self.config.symbol] = position
            logger.info(f"Backtest: Opened {side} position of {quantity} {self.config.symbol} at {entry_price}")
            return position
        
        # In test or live mode, execute on Binance
        try:
            # Set leverage
            self.client.futures_change_leverage(
                symbol=self.config.symbol, 
                leverage=self.config.leverage
            )
            
            # Set margin type (ISOLATED)
            try:
                self.client.futures_change_margin_type(
                    symbol=self.config.symbol,
                    marginType='ISOLATED'
                )
            except BinanceAPIException as e:
                # Margin type might already be set
                if e.code != -4046:  # Code for "No need to change margin type"
                    raise e
            
            # Calculate order parameters
            order_side = 'BUY' if side == 'BUY' else 'SELL'
            opposite_side = 'SELL' if side == 'BUY' else 'BUY'
            
            # Ensure sufficient distance between entry and SL/TP for ADAUSDT or other low-priced assets
            if self.config.symbol == 'ADAUSDT' or entry_price < 5:  # Handle low-priced assets
                logger.info(f"Adjusting SL/TP distances for low-priced asset: {self.config.symbol}")
                # Get symbol price info
                symbol_info = self._get_symbol_info(self.config.symbol)
                tick_size = 0.0001  # Default for ADAUSDT
                
                if symbol_info:
                    price_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'PRICE_FILTER'), None)
                    if price_filter:
                        tick_size = float(price_filter['tickSize'])
                
                # Ensure minimum distance is respected (typically 1% for low-priced assets)
                min_distance = max(entry_price * 0.01, tick_size * 10)
                
                if side == 'BUY':
                    # For BUY: ensure SL is low enough and TP is high enough
                    if entry_price - sl_price < min_distance:
                        sl_price = entry_price - min_distance
                        logger.info(f"Adjusted SL price to ensure minimum distance: {sl_price}")
                    if tp_price - entry_price < min_distance:
                        tp_price = entry_price + min_distance
                        logger.info(f"Adjusted TP price to ensure minimum distance: {tp_price}")
                else:  # SELL
                    # For SELL: ensure SL is high enough and TP is low enough
                    if sl_price - entry_price < min_distance:
                        sl_price = entry_price + min_distance
                        logger.info(f"Adjusted SL price to ensure minimum distance: {sl_price}")
                    if entry_price - tp_price < min_distance:
                        tp_price = entry_price - min_distance
                        logger.info(f"Adjusted TP price to ensure minimum distance: {tp_price}")
            
            # For test mode, use a fixed small quantity
            if self.config.mode == 'test':
                # This quantity is already precision-adjusted in calculate_position_size
                adjusted_quantity = quantity
            else:
                # Normal adjustment for live trading
                adjusted_quantity = self._adjust_quantity_precision(self.config.symbol, quantity)
            
            # Open position with market order
            order = self.client.futures_create_order(
                symbol=self.config.symbol,
                side=order_side,
                type='MARKET',
                quantity=adjusted_quantity
            )
            
            logger.info(f"Opened {side} position of {adjusted_quantity} {self.config.symbol}")
            
            # Get the executed price from the order fills
            executed_price = float(order['avgPrice']) if 'avgPrice' in order else entry_price
            
            # Round SL/TP prices to the correct precision
            sl_price = self._adjust_price_precision(self.config.symbol, sl_price)
            tp_price = self._adjust_price_precision(self.config.symbol, tp_price)
            
            # Re-check distances after execution and precision adjustment
            try:
                # Get the latest market data to ensure SL/TP are valid
                latest_market_data = self.client.futures_mark_price(symbol=self.config.symbol)
                current_market_price = float(latest_market_data['markPrice'])
                logger.info(f"Current market price: {current_market_price}, Entry price: {executed_price}")
                
                # For low-priced assets, ensure SL/TP are far enough from current price
                if self.config.symbol == 'ADAUSDT' or current_market_price < 5:
                    min_distance = current_market_price * 0.01
                    
                    if side == 'BUY':
                        if current_market_price - float(sl_price) < min_distance:
                            sl_price = self._adjust_price_precision(self.config.symbol, current_market_price - min_distance)
                            logger.info(f"Re-adjusted SL price to avoid immediate trigger: {sl_price}")
                        if float(tp_price) - current_market_price < min_distance:
                            tp_price = self._adjust_price_precision(self.config.symbol, current_market_price + min_distance)
                            logger.info(f"Re-adjusted TP price to avoid immediate trigger: {tp_price}")
                    else:  # SELL
                        if float(sl_price) - current_market_price < min_distance:
                            sl_price = self._adjust_price_precision(self.config.symbol, current_market_price + min_distance)
                            logger.info(f"Re-adjusted SL price to avoid immediate trigger: {sl_price}")
                        if current_market_price - float(tp_price) < min_distance:
                            tp_price = self._adjust_price_precision(self.config.symbol, current_market_price - min_distance)
                            logger.info(f"Re-adjusted TP price to avoid immediate trigger: {tp_price}")
            except Exception as e:
                logger.warning(f"Could not re-check price distances: {e}")
            
            # Place stop loss using STOP_MARKET with reduceOnly=True
            try:
                sl_order = self.client.futures_create_order(
                    symbol=self.config.symbol,
                    side=opposite_side,
                    type='STOP_MARKET',
                    stopPrice=sl_price,
                    reduceOnly=True,
                    quantity=adjusted_quantity
                )
                logger.info(f"Set SL at {sl_price}")
                sl_order_id = sl_order['orderId']
            except BinanceAPIException as e:
                logger.error(f"Failed to set SL order: {e}")
                sl_order_id = None
            
            # Place take profit using TAKE_PROFIT_MARKET with reduceOnly=True
            try:
                tp_order = self.client.futures_create_order(
                    symbol=self.config.symbol,
                    side=opposite_side,
                    type='TAKE_PROFIT_MARKET',
                    stopPrice=tp_price,
                    reduceOnly=True,
                    quantity=adjusted_quantity
                )
                logger.info(f"Set TP at {tp_price}")
                tp_order_id = tp_order['orderId']
            except BinanceAPIException as e:
                logger.error(f"Failed to set TP order: {e}")
                tp_order_id = None
            
            # Track position
            position = {
                'id': order['orderId'],
                'symbol': self.config.symbol,
                'side': side,
                'entry_price': executed_price,
                'quantity': float(adjusted_quantity) if isinstance(adjusted_quantity, str) else adjusted_quantity,
                'sl_price': float(sl_price) if isinstance(sl_price, str) else sl_price,
                'tp_price': float(tp_price) if isinstance(tp_price, str) else tp_price,
                'entry_time': datetime.now().timestamp(),
                'sl_order_id': sl_order_id,
                'tp_order_id': tp_order_id
            }
            
            return position
            
        except BinanceAPIException as e:
            logger.error(f"Error opening position: {e}")
            return None
    
    def _adjust_price_precision(self, symbol, price):
        """
        Adjust the price to the correct precision for the symbol.
        
        Args:
            symbol: Trading symbol
            price: Original price
            
        Returns:
            Adjusted price with correct precision
        """
        # Get symbol info
        symbol_info = self._get_symbol_info(symbol)
        
        if not symbol_info:
            logger.warning(f"No symbol info found for {symbol}, using original price")
            return price
        
        # Find the PRICE_FILTER which defines the price precision
        price_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'PRICE_FILTER'), None)
        
        if not price_filter:
            logger.warning(f"No PRICE_FILTER found for {symbol}, using original price")
            return price
        
        # Get the tick size
        tick_size = float(price_filter['tickSize'])
        
        # Calculate precision
        precision = 0
        if tick_size < 1:
            precision = len(str(tick_size).split('.')[-1].rstrip('0'))
        
        # Round to the correct precision
        price = float(price)
        adjusted_price = round(price / tick_size) * tick_size
        adjusted_price = round(adjusted_price, precision)
        
        logger.info(f"Adjusted price from {price} to {adjusted_price} for {symbol} (precision: {precision})")
        return "{:.{}f}".format(adjusted_price, precision)  # Return as string with correct precision
    
    @retry_on_connection_error()
    def close_position(self, position_id, exit_price, reason):
        """
        Close an existing position.
        
        Args:
            position_id: ID of the position to close
            exit_price: Exit price
            reason: Reason for closing the position
            
        Returns:
            Dictionary with position details if successful, None otherwise
        """
        # Get position details
        position = None
        if self.config.mode == 'backtest':
            position = self.positions.get(self.config.symbol)
        else:
            position = self.get_current_position()
        
        if position is None:
            logger.warning(f"No position found with id {position_id}")
            return None
        
        # Calculate PnL
        entry_price = position['entry_price']
        quantity = position['quantity']
        side = position['side']
        
        if side == 'BUY':
            pnl = (exit_price - entry_price) * quantity
            pnl_pct = (exit_price / entry_price - 1) * 100
        else:  # SELL
            pnl = (entry_price - exit_price) * quantity
            pnl_pct = (entry_price / exit_price - 1) * 100
        
        if self.config.mode == 'backtest':
            # In backtest mode, just track the closed position
            self.positions.pop(self.config.symbol, None)
            logger.info(f"Backtest: Closed {side} position of {quantity} {self.config.symbol} at {exit_price}, PnL: {pnl:.2f} ({pnl_pct:.2f}%)")
        else:
            # In test or live mode, execute on Binance
            try:
                # Close position with market order
                order_side = 'SELL' if side == 'BUY' else 'BUY'
                
                # Cancel existing SL/TP orders
                if 'sl_order_id' in position:
                    try:
                        self.client.futures_cancel_order(
                            symbol=self.config.symbol,
                            orderId=position['sl_order_id']
                        )
                    except BinanceAPIException as e:
                        logger.warning(f"Error cancelling SL order: {e}")
                
                if 'tp_order_id' in position:
                    try:
                        self.client.futures_cancel_order(
                            symbol=self.config.symbol,
                            orderId=position['tp_order_id']
                        )
                    except BinanceAPIException as e:
                        logger.warning(f"Error cancelling TP order: {e}")
                
                # Adjust quantity to match symbol precision
                adjusted_quantity = self._adjust_quantity_precision(self.config.symbol, quantity)
                
                # Close the position
                order = self.client.futures_create_order(
                    symbol=self.config.symbol,
                    side=order_side,
                    type='MARKET',  # Use 'MARKET' instead of Client.ORDER_TYPE_MARKET
                    reduceOnly=True,
                    quantity=adjusted_quantity
                )
                
                # If we have actual execution price, use it
                if 'avgPrice' in order:
                    exit_price = float(order['avgPrice'])
                    
                    # Recalculate PnL with actual exit price
                    if side == 'BUY':
                        pnl = (exit_price - entry_price) * quantity
                        pnl_pct = (exit_price / entry_price - 1) * 100
                    else:  # SELL
                        pnl = (entry_price - exit_price) * quantity
                        pnl_pct = (entry_price / exit_price - 1) * 100
                
                logger.info(f"Closed {side} position of {adjusted_quantity} {self.config.symbol} at {exit_price}, PnL: {pnl:.2f} ({pnl_pct:.2f}%)")
                
            except BinanceAPIException as e:
                logger.error(f"Error closing position: {e}")
                return None
        
        # Record trade in history
        trade_record = {
            'symbol': self.config.symbol,
            'side': side,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'quantity': quantity,
            'entry_time': position.get('entry_time', 0),
            'exit_time': datetime.now().timestamp(),
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'reason': reason
        }
        
        self.trade_history.append(trade_record)
        
        return trade_record
    
    def run_backtest(self, df, predictions, risk_manager):
        """
        Run a backtest simulation.
        
        Args:
            df: DataFrame with price data and indicators
            predictions: Series of model predictions
            risk_manager: RiskManager instance
            
        Returns:
            Dictionary with backtest results
        """
        logger.info("Starting backtest simulation")
        
        # Initialize backtest state
        self.positions = {}
        self.trade_history = []
        current_position = None
        
        # Track account balance - use the investment amount
        starting_balance = self.initial_balance
        current_balance = starting_balance
        balance_history = []
        
        # Iterate through each bar
        for i in range(1, len(df)):
            # Current and previous data
            current_row = df.iloc[i]
            prev_row = df.iloc[i-1]
            current_time = df.index[i]
            current_price = current_row['close']
            
            # Get prediction for this bar
            prediction = predictions[i]
            
            # Check if we have a position
            current_position = self.positions.get(self.config.symbol)
            
            # Record balance
            balance_history.append({
                'timestamp': current_time,
                'balance': current_balance
            })
            
            # If no position, check for entry signal
            if current_position is None:
                if prediction != 0:  # We have a prediction
                    # Determine side
                    side = "BUY" if prediction > 0 else "SELL"
                    
                    # Calculate risk parameters
                    risk_params = risk_manager.calculate_risk_params(df.iloc[:i+1])
                    sl_price = risk_params['sl_price']
                    tp_price = risk_params['tp_price']
                    
                    if sl_price is not None and tp_price is not None:
                        # Determine quantity based on risk
                        risk_per_unit = abs(current_price - sl_price)
                        max_risk = current_balance * 0.02  # 2% max risk
                        quantity = max_risk / risk_per_unit
                        
                        # Open position
                        position = self.open_position(
                            side=side,
                            quantity=quantity,
                            entry_price=current_price,
                            sl_price=sl_price,
                            tp_price=tp_price
                        )
            
            # If we have a position, check for exit conditions
            else:
                # Check if SL or TP hit
                side = current_position['side']
                entry_price = current_position['entry_price']
                sl_price = current_position['sl_price']
                tp_price = current_position['tp_price']
                
                exit_signal = None
                
                # Check SL hit
                if side == 'BUY' and current_price <= sl_price:
                    exit_signal = "Stop loss hit"
                elif side == 'SELL' and current_price >= sl_price:
                    exit_signal = "Stop loss hit"
                
                # Check TP hit
                if side == 'BUY' and current_price >= tp_price:
                    exit_signal = "Take profit hit"
                elif side == 'SELL' and current_price <= tp_price:
                    exit_signal = "Take profit hit"
                
                # Check for reversal signal
                if (side == 'BUY' and prediction < 0) or (side == 'SELL' and prediction > 0):
                    exit_signal = "Trend reversal signal"
                
                # Exit position if we have a signal
                if exit_signal:
                    result = self.close_position(
                        position_id=current_position['id'],
                        exit_price=current_price,
                        reason=exit_signal
                    )
                    
                    if result:
                        # Update balance
                        current_balance += result['pnl']
        
        # Calculate performance metrics
        total_trades = len(self.trade_history)
        profitable_trades = sum(1 for trade in self.trade_history if trade['pnl'] > 0)
        win_rate = profitable_trades / total_trades if total_trades > 0 else 0
        
        total_profit = sum(trade['pnl'] for trade in self.trade_history)
        profit_factor = 0
        if total_trades > 0:
            gross_profit = sum(trade['pnl'] for trade in self.trade_history if trade['pnl'] > 0)
            gross_loss = abs(sum(trade['pnl'] for trade in self.trade_history if trade['pnl'] < 0))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Calculate max drawdown
        peak = 0
        max_drawdown = 0
        for record in balance_history:
            balance = record['balance']
            if balance > peak:
                peak = balance
            drawdown = (peak - balance) / peak if peak > 0 else 0
            max_drawdown = max(max_drawdown, drawdown)
        
        # Return backtest results
        results = {
            'starting_balance': starting_balance,
            'ending_balance': current_balance,
            'total_return': (current_balance / starting_balance - 1) * 100,
            'total_trades': total_trades,
            'profitable_trades': profitable_trades,
            'win_rate': win_rate * 100,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown * 100,
            'trade_history': self.trade_history,
            'balance_history': balance_history
        }
        
        logger.info(f"Backtest completed with {total_trades} trades, win rate: {win_rate:.2%}, total return: {results['total_return']:.2f}%")
        return results
    
    @retry_on_connection_error()
    def get_account_balance(self):
        """Get the current account balance to use as margin."""
        if self.config.mode == 'backtest':
            return self.current_balance
        
        try:
            # For futures trading, check actual available balance
            account_info = self.client.futures_account()
            available_balance = float(account_info['availableBalance'])
            
            # For testnet, we need to be extra cautious - use a very small amount
            if self.config.mode == 'test':
                # Use a very small fixed amount for testnet trading to ensure success
                test_margin = 5.0  # Just use 5 USDT for test trades
                logger.info(f"Testnet mode: Using fixed margin of {test_margin} USDT (available: {available_balance} USDT)")
                return test_margin
            
            # For live trading, use the configured investment amount or what's available, whichever is smaller
            usable_margin = min(self.initial_balance, available_balance)
            logger.info(f"Available margin: {available_balance} USDT, Using: {usable_margin} USDT")
            return usable_margin
            
        except BinanceAPIException as e:
            logger.error(f"Error getting account balance: {e}")
            # If we can't get the balance, be conservative and assume minimum amount
            return 5.0 if self.config.mode == 'test' else min(self.initial_balance, 10)
    
    def calculate_position_size(self, entry_price, sl_price):
        """
        Calculate position size based on risk parameters and investment amount.
        
        Args:
            entry_price: Entry price for the trade
            sl_price: Stop-loss price
            
        Returns:
            Appropriate position size (as string for futures, float for backtest)
        """
        # For backtest or live modes, proceed with normal calculation
        # Get current margin balance
        margin = self.get_account_balance()
        
        # Maximum amount to risk per trade (2% of margin)
        max_risk_amount = margin * 0.02
        
        # Calculate risk per unit
        risk_per_unit = abs(entry_price - sl_price)
        
        # Calculate position size in base currency
        if risk_per_unit > 0:
            position_size = max_risk_amount / risk_per_unit
        else:
            position_size = 0
            logger.warning("Risk per unit is zero or negative, setting position size to 0")
        
        # Adjust for leverage
        position_size = position_size * self.config.leverage
        
        # Calculate maximum position size based on margin and leverage
        # The formula is (available_balance * leverage) / entry_price
        # But we need to account for maintenance margin requirements
        # Use 50% of the theoretical max to be safe
        max_position_size = (margin * self.config.leverage * 0.5) / entry_price
        
        # Use the smaller of our calculated values
        position_size = min(position_size, max_position_size)
        
        # Adjust for precision
        position_size = self._adjust_quantity_precision(self.config.symbol, position_size)
        
        logger.info(f"Calculated position size: {position_size} based on margin: {margin}, risk: {max_risk_amount}, entry: {entry_price}, SL: {sl_price}, leverage: {self.config.leverage}x")
        
        return position_size
    
    def _get_min_notional(self, symbol):
        """Get the minimum notional value for a symbol."""
        symbol_info = self._get_symbol_info(symbol)
        
        if not symbol_info:
            return 5  # Default minimum notional value
        
        # Find the MIN_NOTIONAL filter
        min_notional_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'MIN_NOTIONAL'), None)
        
        if min_notional_filter:
            return float(min_notional_filter['notional'])
        
        # No MIN_NOTIONAL filter found, return a safe default
        return 5  # Default minimum value
