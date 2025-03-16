import logging
import pandas as pd
import numpy as np
from datetime import datetime
from binance.client import Client
from binance.exceptions import BinanceAPIException
import os
import uuid
import math
from .config import BotConfig
from .notifier import TelegramNotifier

logger = logging.getLogger(__name__)

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
        # Initial account balance to use for trading
        self.initial_balance = config.invest
        self.current_balance = config.invest
    
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
            order_side = Client.SIDE_BUY if side == 'BUY' else Client.SIDE_SELL
            
            # Adjust quantity to match symbol precision
            adjusted_quantity = self._adjust_quantity_precision(self.config.symbol, quantity)
            
            # Open position with market order
            order = self.client.futures_create_order(
                symbol=self.config.symbol,
                side=order_side,
                type=Client.ORDER_TYPE_MARKET,
                quantity=adjusted_quantity
            )
            
            logger.info(f"Opened {side} position of {adjusted_quantity} {self.config.symbol}")
            
            # Get the executed price from the order fills
            executed_price = float(order['avgPrice']) if 'avgPrice' in order else entry_price
            
            # Round SL/TP prices to the correct precision
            sl_price = self._adjust_price_precision(self.config.symbol, sl_price)
            tp_price = self._adjust_price_precision(self.config.symbol, tp_price)
            
            # Place stop loss
            sl_order = self.client.futures_create_order(
                symbol=self.config.symbol,
                side=Client.SIDE_SELL if side == 'BUY' else Client.SIDE_BUY,
                type=Client.ORDER_TYPE_STOP_MARKET,
                stopPrice=sl_price,
                closePosition=True
            )
            
            # Place take profit
            tp_order = self.client.futures_create_order(
                symbol=self.config.symbol,
                side=Client.SIDE_SELL if side == 'BUY' else Client.SIDE_BUY,
                type=Client.ORDER_TYPE_TAKE_PROFIT_MARKET,
                stopPrice=tp_price,
                closePosition=True
            )
            
            # Track position
            position = {
                'id': order['orderId'],
                'symbol': self.config.symbol,
                'side': side,
                'entry_price': executed_price,
                'quantity': float(adjusted_quantity),
                'sl_price': float(sl_price),
                'tp_price': float(tp_price),
                'entry_time': datetime.now().timestamp(),
                'sl_order_id': sl_order['orderId'],
                'tp_order_id': tp_order['orderId']
            }
            
            logger.info(f"Set SL at {sl_price} and TP at {tp_price}")
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
                order_side = Client.SIDE_SELL if side == 'BUY' else Client.SIDE_BUY
                
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
                    type=Client.ORDER_TYPE_MARKET,
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
    
    def get_account_balance(self):
        """Get the current account balance for trading."""
        if self.config.mode == 'backtest':
            return self.current_balance
        
        try:
            # For futures trading
            account_info = self.client.futures_account_balance()
            for balance in account_info:
                if balance['asset'] == 'USDT':
                    # Use either the actual balance or the configured investment amount, whichever is lower
                    actual_balance = float(balance['balance'])
                    return min(actual_balance, self.initial_balance)
            
            return self.initial_balance  # Default to initial investment if no USDT balance found
        except BinanceAPIException as e:
            logger.error(f"Error getting account balance: {e}")
            return self.initial_balance
    
    def calculate_position_size(self, entry_price, sl_price):
        """
        Calculate position size based on risk parameters and investment amount.
        
        Args:
            entry_price: Entry price for the trade
            sl_price: Stop-loss price
            
        Returns:
            Appropriate position size (as string for futures, float for backtest)
        """
        # Get current balance
        balance = self.get_account_balance()
        
        # Maximum amount to risk per trade (2% of balance)
        max_risk_amount = balance * 0.02
        
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
        
        # Limit position size to ensure it doesn't exceed investment amount
        max_position_size = (balance * self.config.leverage) / entry_price
        position_size = min(position_size, max_position_size)
        
        # If position size is too small, return zero
        if position_size < 0.000001:
            logger.warning(f"Position size {position_size} is too small, setting to 0")
            return "0" if self.config.mode in ['test', 'live'] else 0.0
        
        # Adjust for precision - this will return a string or float depending on mode
        position_size = self._adjust_quantity_precision(self.config.symbol, position_size)
        
        logger.info(f"Calculated position size: {position_size} based on balance: {balance}, risk: {max_risk_amount}, entry: {entry_price}, SL: {sl_price}")
        
        return position_size
