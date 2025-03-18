import time
import numpy as np
import pandas as pd
from binance.client import Client
from binance.exceptions import BinanceAPIException
from binance.enums import *

import config
from utils.logger import setup_logger

class BinanceClient:
    def __init__(self):
        self.logger = setup_logger('binance_client', 'logs/binance_api.log')
        self.client = Client(config.API_KEY, config.API_SECRET, testnet=config.TESTNET)
        
        # Test API connection
        try:
            self.client.ping()
            self.logger.info("Successfully connected to Binance API")
        except BinanceAPIException as e:
            self.logger.error(f"Failed to connect to Binance API: {e}")
            raise
            
        # Set default leverage for all symbols
        if not config.TESTNET:  # Only set leverage in live mode
            for symbol in config.SYMBOLS:
                try:
                    self.client.futures_change_leverage(
                        symbol=symbol, 
                        leverage=config.DEFAULT_LEVERAGE
                    )
                    self.logger.info(f"Set leverage for {symbol} to {config.DEFAULT_LEVERAGE}x")
                except BinanceAPIException as e:
                    self.logger.error(f"Failed to set leverage for {symbol}: {e}")

    def get_historical_klines(self, symbol, interval, limit=100):
        """Get historical klines/candlestick data for a symbol."""
        retry_count = 0
        while retry_count < config.EXECUTION_RETRY_ATTEMPTS:
            try:
                klines = self.client.futures_klines(symbol=symbol, interval=interval, limit=limit)
                
                # Convert to DataFrame
                df = pd.DataFrame(klines, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_asset_volume', 'number_of_trades',
                    'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
                ])
                
                # Convert types
                numeric_columns = ['open', 'high', 'low', 'close', 'volume']
                df[numeric_columns] = df[numeric_columns].astype(float)
                
                # Convert timestamp to datetime
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                
                return df
                
            except BinanceAPIException as e:
                retry_count += 1
                self.logger.warning(f"API Error in get_historical_klines (attempt {retry_count}): {e}")
                if retry_count >= config.EXECUTION_RETRY_ATTEMPTS:
                    self.logger.error(f"Failed to get historical klines after {config.EXECUTION_RETRY_ATTEMPTS} attempts")
                    raise
                time.sleep(config.EXECUTION_RETRY_DELAY)
            
            except Exception as e:
                self.logger.error(f"Unexpected error in get_historical_klines: {e}")
                raise

    def get_balance(self):
        """Get account balance in USDT."""
        try:
            if config.TESTNET:
                account = self.client.futures_account()
            else:
                account = self.client.futures_account()
                
            # Find USDT balance
            for asset in account['assets']:
                if asset['asset'] == 'USDT':
                    return float(asset['walletBalance'])
            
            return 0.0
        except BinanceAPIException as e:
            self.logger.error(f"Failed to get account balance: {e}")
            return 0.0

    def get_symbol_price(self, symbol):
        """Get current price for a symbol."""
        try:
            ticker = self.client.futures_symbol_ticker(symbol=symbol)
            return float(ticker['price'])
        except BinanceAPIException as e:
            self.logger.error(f"Failed to get price for {symbol}: {e}")
            return None

    def place_order(self, symbol, side, quantity, price=None, stop_loss=None, take_profit=None):
        """Place an order with optional stop loss and take profit."""
        try:
            # Determine if we're doing a limit or market order
            order_type = ORDER_TYPE_LIMIT if price else ORDER_TYPE_MARKET
            
            # Main order parameters
            order_params = {
                'symbol': symbol,
                'side': side,
                'quantity': self._format_quantity(symbol, quantity),
                'type': order_type
            }
            
            # Add price for limit orders
            if price:
                order_params['price'] = self._format_price(symbol, price)
                order_params['timeInForce'] = TIME_IN_FORCE_GTC
            
            # Place the main order
            order = self.client.futures_create_order(**order_params)
            self.logger.info(f"Placed {side} order for {symbol}: {quantity} @ {price if price else 'MARKET'}")
            
            # Get filled price if it was a market order
            execution_price = price
            if not execution_price:
                time.sleep(1)  # Give time for the order to process
                order_details = self.client.futures_get_order(symbol=symbol, orderId=order['orderId'])
                execution_price = float(order_details['avgPrice'])
            
            # Set stop loss if provided
            if stop_loss:
                sl_price = self._format_price(symbol, stop_loss)
                sl_side = 'BUY' if side == 'SELL' else 'SELL'
                
                sl_order = self.client.futures_create_order(
                    symbol=symbol,
                    side=sl_side,
                    type=ORDER_TYPE_STOP_MARKET,
                    quantity=self._format_quantity(symbol, quantity),
                    stopPrice=sl_price,
                    closePosition=True
                )
                self.logger.info(f"Placed stop-loss for {symbol} at {sl_price}")
            
            # Set take profit if provided
            if take_profit:
                tp_price = self._format_price(symbol, take_profit)
                tp_side = 'BUY' if side == 'SELL' else 'SELL'
                
                tp_order = self.client.futures_create_order(
                    symbol=symbol,
                    side=tp_side,
                    type=ORDER_TYPE_TAKE_PROFIT_MARKET,
                    quantity=self._format_quantity(symbol, quantity),
                    stopPrice=tp_price,
                    closePosition=True
                )
                self.logger.info(f"Placed take-profit for {symbol} at {tp_price}")
            
            return {
                'order_id': order['orderId'],
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'price': execution_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit
            }
            
        except BinanceAPIException as e:
            self.logger.error(f"Failed to place order for {symbol}: {e}")
            return None

    def get_open_positions(self):
        """Get all currently open positions."""
        try:
            positions = []
            account = self.client.futures_account()
            
            for position in account['positions']:
                pos_amt = float(position['positionAmt'])
                if pos_amt != 0:  # Position is open
                    positions.append({
                        'symbol': position['symbol'],
                        'size': pos_amt,
                        'entry_price': float(position['entryPrice']),
                        'mark_price': float(position['markPrice']),
                        'pnl': float(position['unrealizedProfit']),
                        'leverage': int(position['leverage'])
                    })
            
            return positions
        except BinanceAPIException as e:
            self.logger.error(f"Failed to get open positions: {e}")
            return []

    def close_position(self, symbol):
        """Close an open position for a symbol."""
        try:
            positions = self.get_open_positions()
            position = next((p for p in positions if p['symbol'] == symbol), None)
            
            if not position:
                self.logger.warning(f"No open position found for {symbol}")
                return False
                
            # Determine the side for closing the position
            side = 'SELL' if position['size'] > 0 else 'BUY'
            quantity = abs(position['size'])
            
            # Place market order to close
            order = self.client.futures_create_order(
                symbol=symbol,
                side=side,
                type=ORDER_TYPE_MARKET,
                quantity=self._format_quantity(symbol, quantity),
                reduceOnly=True
            )
            
            self.logger.info(f"Closed position for {symbol}: {quantity} @ MARKET")
            return True
            
        except BinanceAPIException as e:
            self.logger.error(f"Failed to close position for {symbol}: {e}")
            return False

    def _format_quantity(self, symbol, quantity):
        """Format the quantity according to the symbol's quantity precision."""
        # In a full implementation, this would fetch the precision from the exchange info
        # For simplicity, we'll just round to 3 decimal places
        return round(quantity, 3)

    def _format_price(self, symbol, price):
        """Format the price according to the symbol's price precision."""
        # In a full implementation, this would fetch the precision from the exchange info
        # For simplicity, we'll just round to 2 decimal places
        return round(price, 2)
