"""
Binance client for the AI Trading Bot.
"""

import os
import time
import logging
import ccxt
from dotenv import load_dotenv
from utils import rate_limit_handler, convert_to_dataframe

# Load environment variables
load_dotenv()

# Get Binance API credentials from environment variables
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')
BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET')

# Initialize logger
logger = logging.getLogger('trading_bot')


class BinanceClient:
    """
    Binance client for the trading bot.
    """
    
    def __init__(self, api_key=None, api_secret=None, testnet=False):
        """
        Initialize Binance client.
        
        Args:
            api_key (str): Binance API key
            api_secret (str): Binance API secret
            testnet (bool): Use testnet
        """
        self.api_key = api_key or BINANCE_API_KEY
        self.api_secret = api_secret or BINANCE_API_SECRET
        self.testnet = testnet
        
        if not self.api_key or not self.api_secret:
            logger.error("Binance API credentials not found.")
            raise ValueError("Binance API credentials not found.")
        
        # Initialize ccxt client
        self.exchange = ccxt.binance({
            'apiKey': self.api_key,
            'secret': self.api_secret,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future',
                'adjustForTimeDifference': True,
                'testnet': self.testnet
            }
        })
        
        logger.info(f"Binance client initialized (testnet: {self.testnet}).")
    
    @rate_limit_handler(max_requests=50, time_window=60, buffer=0.8)
    def get_klines(self, symbol, timeframe, limit=100):
        """
        Get klines (candlestick data) from Binance.
        
        Args:
            symbol (str): Trading symbol
            timeframe (str): Timeframe (e.g., '5m', '1h')
            limit (int): Number of klines to get
            
        Returns:
            pd.DataFrame: DataFrame with OHLCV data
        """
        try:
            # Get klines from Binance
            klines = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            # Convert to DataFrame
            df = convert_to_dataframe(klines)
            
            logger.info(f"Got {len(df)} klines for {symbol} ({timeframe}).")
            
            return df
        
        except Exception as e:
            logger.error(f"Error getting klines: {e}")
            raise
    
    @rate_limit_handler(max_requests=50, time_window=60, buffer=0.8)
    def get_order_book(self, symbol, limit=20):
        """
        Get order book from Binance.
        
        Args:
            symbol (str): Trading symbol
            limit (int): Depth of order book
            
        Returns:
            dict: Order book data
        """
        try:
            # Get order book from Binance
            order_book = self.exchange.fetch_order_book(symbol, limit=limit)
            
            logger.info(f"Got order book for {symbol} (depth: {limit}).")
            
            return order_book
        
        except Exception as e:
            logger.error(f"Error getting order book: {e}")
            raise
    
    @rate_limit_handler(max_requests=50, time_window=60, buffer=0.8)
    def get_account_balance(self):
        """
        Get account balance from Binance.
        
        Returns:
            dict: Account balance data
        """
        try:
            # Get account balance from Binance
            balance = self.exchange.fetch_balance()
            
            logger.info("Got account balance.")
            
            return balance
        
        except Exception as e:
            logger.error(f"Error getting account balance: {e}")
            raise
    
    @rate_limit_handler(max_requests=50, time_window=60, buffer=0.8)
    def set_leverage(self, symbol, leverage):
        """
        Set leverage for a symbol.
        
        Args:
            symbol (str): Trading symbol
            leverage (int): Leverage
            
        Returns:
            dict: Response from Binance
        """
        try:
            # Set leverage using the correct futures API method
            response = self.exchange.set_leverage(leverage=leverage, symbol=symbol)
            
            logger.info(f"Set leverage for {symbol} to {leverage}x.")
            
            return response
        
        except Exception as e:
            logger.error(f"Error setting leverage: {e}")
            raise
    
    @rate_limit_handler(max_requests=50, time_window=60, buffer=0.8)
    def create_market_order(self, symbol, side, amount):
        """
        Create a market order.
        
        Args:
            symbol (str): Trading symbol
            side (str): Order side ('buy' or 'sell')
            amount (float): Order amount
            
        Returns:
            dict: Order data
        """
        try:
            # Create market order
            order = self.exchange.create_order(
                symbol=symbol,
                type='market',
                side=side,
                amount=amount
            )
            
            logger.info(f"Created market {side} order for {amount} {symbol}.")
            
            return order
        
        except Exception as e:
            logger.error(f"Error creating market order: {e}")
            raise
    
    @rate_limit_handler(max_requests=50, time_window=60, buffer=0.8)
    def create_limit_order(self, symbol, side, amount, price):
        """
        Create a limit order.
        
        Args:
            symbol (str): Trading symbol
            side (str): Order side ('buy' or 'sell')
            amount (float): Order amount
            price (float): Order price
            
        Returns:
            dict: Order data
        """
        try:
            # Create limit order
            order = self.exchange.create_order(
                symbol=symbol,
                type='limit',
                side=side,
                amount=amount,
                price=price
            )
            
            logger.info(f"Created limit {side} order for {amount} {symbol} at {price}.")
            
            return order
        
        except Exception as e:
            logger.error(f"Error creating limit order: {e}")
            raise
    
    @rate_limit_handler(max_requests=50, time_window=60, buffer=0.8)
    def create_stop_loss_order(self, symbol, side, amount, stop_price, price=None):
        """
        Create a stop loss order.
        
        Args:
            symbol (str): Trading symbol
            side (str): Order side ('buy' or 'sell')
            amount (float): Order amount
            stop_price (float): Stop price
            price (float): Limit price (optional, for stop-limit orders)
            
        Returns:
            dict: Order data
        """
        try:
            # Determine order type
            order_type = 'stop_market' if price is None else 'stop_limit'
            
            # Create stop loss order
            params = {'stopPrice': stop_price}
            if price is not None:
                params['price'] = price
            
            order = self.exchange.create_order(
                symbol=symbol,
                type=order_type,
                side=side,
                amount=amount,
                params=params
            )
            
            logger.info(f"Created {order_type} {side} order for {amount} {symbol} at {stop_price}.")
            
            return order
        
        except Exception as e:
            logger.error(f"Error creating stop loss order: {e}")
            raise
    
    @rate_limit_handler(max_requests=50, time_window=60, buffer=0.8)
    def create_take_profit_order(self, symbol, side, amount, stop_price, price=None):
        """
        Create a take profit order.
        
        Args:
            symbol (str): Trading symbol
            side (str): Order side ('buy' or 'sell')
            amount (float): Order amount
            stop_price (float): Stop price
            price (float): Limit price (optional, for take-profit-limit orders)
            
        Returns:
            dict: Order data
        """
        try:
            # Determine order type
            order_type = 'take_profit_market' if price is None else 'take_profit_limit'
            
            # Create take profit order
            params = {'stopPrice': stop_price}
            if price is not None:
                params['price'] = price
            
            order = self.exchange.create_order(
                symbol=symbol,
                type=order_type,
                side=side,
                amount=amount,
                params=params
            )
            
            logger.info(f"Created {order_type} {side} order for {amount} {symbol} at {stop_price}.")
            
            return order
        
        except Exception as e:
            logger.error(f"Error creating take profit order: {e}")
            raise
    
    @rate_limit_handler(max_requests=50, time_window=60, buffer=0.8)
    def cancel_order(self, order_id, symbol):
        """
        Cancel an order.
        
        Args:
            order_id (str): Order ID
            symbol (str): Trading symbol
            
        Returns:
            dict: Response from Binance
        """
        try:
            # Cancel order
            response = self.exchange.cancel_order(order_id, symbol)
            
            logger.info(f"Cancelled order {order_id} for {symbol}.")
            
            return response
        
        except Exception as e:
            logger.error(f"Error cancelling order: {e}")
            raise
    
    @rate_limit_handler(max_requests=50, time_window=60, buffer=0.8)
    def get_open_orders(self, symbol=None):
        """
        Get open orders.
        
        Args:
            symbol (str): Trading symbol (optional)
            
        Returns:
            list: Open orders
        """
        try:
            # Get open orders
            open_orders = self.exchange.fetch_open_orders(symbol=symbol)
            
            logger.info(f"Got {len(open_orders)} open orders{' for ' + symbol if symbol else ''}.")
            
            return open_orders
        
        except Exception as e:
            logger.error(f"Error getting open orders: {e}")
            raise
    
    @rate_limit_handler(max_requests=50, time_window=60, buffer=0.8)
    def get_positions(self, symbol=None):
        """
        Get positions.
        
        Args:
            symbol (str): Trading symbol (optional)
            
        Returns:
            list: Positions
        """
        try:
            # Get positions
            positions = self.exchange.fetch_positions(symbol=symbol)
            
            # Filter out positions with zero amount
            positions = [p for p in positions if float(p['contracts']) > 0]
            
            logger.info(f"Got {len(positions)} positions{' for ' + symbol if symbol else ''}.")
            
            return positions
        
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            raise
    
    @rate_limit_handler(max_requests=50, time_window=60, buffer=0.8)
    def close_position(self, symbol, position_type):
        """
        Close a position.
        
        Args:
            symbol (str): Trading symbol
            position_type (str): Position type ('long' or 'short')
            
        Returns:
            dict: Order data
        """
        try:
            # Get positions
            positions = self.get_positions(symbol=symbol)
            
            # Find position to close
            position = None
            for p in positions:
                if p['side'] == 'long' and position_type == 'long':
                    position = p
                    break
                elif p['side'] == 'short' and position_type == 'short':
                    position = p
                    break
            
            if not position:
                logger.warning(f"No {position_type} position found for {symbol}.")
                return None
            
            # Determine side for closing position
            side = 'sell' if position_type == 'long' else 'buy'
            
            # Close position
            order = self.create_market_order(
                symbol=symbol,
                side=side,
                amount=float(position['contracts'])
            )
            
            logger.info(f"Closed {position_type} position for {symbol}.")
            
            return order
        
        except Exception as e:
            logger.error(f"Error closing position: {e}")
            raise 