"""
Binance client for the AI Trading Bot.
"""

import os
import time
import logging
from datetime import datetime
import pandas as pd
import numpy as np
import requests
from binance.client import Client
from binance.exceptions import BinanceAPIException
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
        
        # Initialize Binance client with timeout and retry
        try:
            self.client = Client(
                api_key=self.api_key,
                api_secret=self.api_secret,
                testnet=self.testnet,
                requests_params={'timeout': 30}
            )
            
            # Test connection
            self._test_connection()
            
            logger.info(f"Binance client initialized (testnet: {self.testnet}).")
        except Exception as e:
            logger.error(f"Error initializing Binance client: {e}")
            if "Invalid API-key" in str(e):
                logger.error("Invalid API key. Please check your credentials.")
            elif "Connection" in str(e):
                logger.error("Connection error. Please check your internet connection.")
            raise
    
    def _test_connection(self):
        """
        Test connection to Binance API.
        
        Raises:
            Exception: If connection fails
        """
        try:
            # Try to ping the server
            self.client.ping()
            logger.info("Connection to Binance API successful.")
        except requests.exceptions.RequestException as e:
            logger.error(f"Connection error: {e}")
            raise Exception(f"Failed to connect to Binance API: {e}")
        except Exception as e:
            logger.error(f"API error: {e}")
            raise
    
    @rate_limit_handler(max_requests=50, time_window=60, buffer=0.8)
    def get_klines(self, symbol, timeframe, limit=100):
        """
        Get klines (candlestick data).
        
        Args:
            symbol (str): Trading symbol
            timeframe (str): Timeframe
            limit (int): Number of candles to get
            
        Returns:
            pandas.DataFrame: Klines data
        """
        try:
            # Convert timeframe to Binance format
            interval = self._convert_timeframe(timeframe)
            
            # Get klines
            klines = self.client.get_klines(
                symbol=symbol,
                interval=interval,
                limit=limit
            )
            
            # Convert to dataframe
            df = convert_to_dataframe(klines)
            
            logger.info(f"Got {len(df)} klines for {symbol} ({timeframe}).")
            
            return df
        
        except Exception as e:
            logger.error(f"Error getting klines: {e}")
            raise
    
    @rate_limit_handler(max_requests=50, time_window=60, buffer=0.8)
    def get_order_book(self, symbol, limit=20):
        """
        Get order book.
        
        Args:
            symbol (str): Trading symbol
            limit (int): Depth of the order book
            
        Returns:
            dict: Order book data
        """
        try:
            # Get order book
            order_book = self.client.get_order_book(symbol=symbol, limit=limit)
            
            logger.info(f"Got order book for {symbol}.")
            
            return order_book
        
        except Exception as e:
            logger.error(f"Error getting order book: {e}")
            raise
    
    @rate_limit_handler(max_requests=50, time_window=60, buffer=0.8)
    def get_account_balance(self):
        """
        Get account balance.
        
        Returns:
            dict: Account balance data
        """
        try:
            # Get account information
            account = self.client.get_account()
            
            logger.info("Got account balance.")
            
            return account
        
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
            # Set leverage using the Binance Futures API
            response = self.client.futures_change_leverage(
                symbol=symbol,
                leverage=leverage
            )
            
            logger.info(f"Set leverage for {symbol} to {leverage}x.")
            
            return response
        
        except BinanceAPIException as e:
            logger.error(f"Binance API error setting leverage: {e}")
            if e.code == -4046:  # Leverage is too high
                logger.warning(f"Leverage {leverage} is too high for {symbol}. Using maximum allowed leverage.")
                # Try to get the maximum allowed leverage and set it
                try:
                    # Get symbol info
                    symbol_info = self.client.futures_exchange_info()
                    max_leverage = 20  # Default max leverage
                    
                    # Find the symbol in the exchange info
                    for s in symbol_info['symbols']:
                        if s['symbol'] == symbol:
                            max_leverage = int(s.get('leverageBracket', [{'bracket': 1, 'initialLeverage': 20}])[0]['initialLeverage'])
                            break
                    
                    # Set the maximum allowed leverage
                    response = self.client.futures_change_leverage(
                        symbol=symbol,
                        leverage=max_leverage
                    )
                    
                    logger.info(f"Set leverage for {symbol} to {max_leverage}x (maximum allowed).")
                    
                    return response
                except Exception as e2:
                    logger.error(f"Error setting maximum leverage: {e2}")
                    logger.warning(f"Continuing without setting leverage. Default leverage will be used.")
                    return {"leverage": 1, "symbol": symbol, "status": "default"}
            else:
                logger.warning(f"Continuing without setting leverage. Default leverage will be used.")
                return {"leverage": 1, "symbol": symbol, "status": "default"}
        except Exception as e:
            logger.error(f"Error setting leverage: {e}")
            logger.warning(f"Continuing without setting leverage. Default leverage will be used.")
            return {"leverage": 1, "symbol": symbol, "status": "default"}
    
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
            order = self.client.futures_create_order(
                symbol=symbol,
                side=side.upper(),
                type='MARKET',
                quantity=amount
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
            order = self.client.futures_create_order(
                symbol=symbol,
                side=side.upper(),
                type='LIMIT',
                timeInForce='GTC',
                quantity=amount,
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
            price (float, optional): Limit price (for stop-limit orders)
            
        Returns:
            dict: Order data
        """
        try:
            # Determine order type
            order_type = 'STOP_MARKET' if price is None else 'STOP'
            
            # Create stop loss order
            order_params = {
                'symbol': symbol,
                'side': side.upper(),
                'type': order_type,
                'quantity': amount,
                'stopPrice': stop_price
            }
            
            # Add price for stop-limit orders
            if price is not None:
                order_params['price'] = price
                order_params['timeInForce'] = 'GTC'
            
            order = self.client.futures_create_order(**order_params)
            
            logger.info(f"Created stop loss {side} order for {amount} {symbol} at {stop_price}.")
            
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
            price (float, optional): Limit price (for take-profit-limit orders)
            
        Returns:
            dict: Order data
        """
        try:
            # Determine order type
            order_type = 'TAKE_PROFIT_MARKET' if price is None else 'TAKE_PROFIT'
            
            # Create take profit order
            order_params = {
                'symbol': symbol,
                'side': side.upper(),
                'type': order_type,
                'quantity': amount,
                'stopPrice': stop_price
            }
            
            # Add price for take-profit-limit orders
            if price is not None:
                order_params['price'] = price
                order_params['timeInForce'] = 'GTC'
            
            order = self.client.futures_create_order(**order_params)
            
            logger.info(f"Created take profit {side} order for {amount} {symbol} at {stop_price}.")
            
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
            response = self.client.futures_cancel_order(
                symbol=symbol,
                orderId=order_id
            )
            
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
            symbol (str, optional): Trading symbol
            
        Returns:
            list: Open orders
        """
        try:
            # Get open orders
            if symbol:
                open_orders = self.client.futures_get_open_orders(symbol=symbol)
            else:
                open_orders = self.client.futures_get_open_orders()
            
            logger.info(f"Got {len(open_orders)} open orders.")
            
            return open_orders
        
        except Exception as e:
            logger.error(f"Error getting open orders: {e}")
            raise
    
    @rate_limit_handler(max_requests=50, time_window=60, buffer=0.8)
    def get_positions(self, symbol=None):
        """
        Get positions.
        
        Args:
            symbol (str, optional): Trading symbol
            
        Returns:
            list: Positions
        """
        try:
            # Get positions
            positions = self.client.futures_position_information()
            
            # Filter by symbol if provided
            if symbol:
                positions = [p for p in positions if p['symbol'] == symbol]
            
            # Filter out positions with zero amount
            positions = [p for p in positions if float(p['positionAmt']) != 0]
            
            logger.info(f"Got {len(positions)} positions.")
            
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
            # Get position
            positions = self.get_positions(symbol)
            
            # Find the position to close
            position = None
            for p in positions:
                if p['symbol'] == symbol:
                    position_amt = float(p['positionAmt'])
                    if (position_type == 'long' and position_amt > 0) or (position_type == 'short' and position_amt < 0):
                        position = p
                        break
            
            if not position:
                logger.warning(f"No {position_type} position found for {symbol}.")
                return None
            
            # Determine side and amount
            side = 'SELL' if position_type == 'long' else 'BUY'
            amount = abs(float(position['positionAmt']))
            
            # Create market order to close position
            order = self.client.futures_create_order(
                symbol=symbol,
                side=side,
                type='MARKET',
                quantity=amount,
                reduceOnly=True
            )
            
            logger.info(f"Closed {position_type} position for {symbol}.")
            
            return order
        
        except Exception as e:
            logger.error(f"Error closing position: {e}")
            raise
    
    def _convert_timeframe(self, timeframe):
        """
        Convert timeframe to Binance format.
        
        Args:
            timeframe (str): Timeframe
            
        Returns:
            str: Binance timeframe format
        """
        # Binance uses the same format as our timeframe (e.g., '5m', '1h', '1d')
        return timeframe 