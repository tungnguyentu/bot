import logging
import time
import math
from binance.client import Client
from binance.exceptions import BinanceAPIException

logger = logging.getLogger("BinanceBot.BinanceFutures")

class BinanceFutures:
    def __init__(self, api_key, api_secret):
        self.client = Client(api_key, api_secret)
        self.logger = logging.getLogger(__name__)
    
    def get_historical_data(self, symbol, timeframe, limit=100):
        """
        Get historical kline/candlestick data for a symbol
        """
        interval_map = {
            "1m": Client.KLINE_INTERVAL_1MINUTE,
            "5m": Client.KLINE_INTERVAL_5MINUTE,
            "15m": Client.KLINE_INTERVAL_15MINUTE,
            "30m": Client.KLINE_INTERVAL_30MINUTE,
            "1h": Client.KLINE_INTERVAL_1HOUR,
            "4h": Client.KLINE_INTERVAL_4HOUR,
            "1d": Client.KLINE_INTERVAL_1DAY,
        }
        
        try:
            klines = self.client.futures_klines(
                symbol=symbol,
                interval=interval_map[timeframe],
                limit=limit
            )
            
            # Convert klines to a more usable format
            data = []
            for k in klines:
                data.append({
                    'timestamp': k[0],
                    'open': float(k[1]),
                    'high': float(k[2]),
                    'low': float(k[3]),
                    'close': float(k[4]),
                    'volume': float(k[5]),
                })
            
            return data
            
        except BinanceAPIException as e:
            self.logger.error(f"Error getting historical data: {e}")
            return []
    
    def get_account_balance(self):
        """
        Get futures account balance
        """
        try:
            account = self.client.futures_account_balance()
            # Find USDT balance
            for balance in account:
                if balance['asset'] == 'USDT':
                    return float(balance['balance'])
            return 0
        except BinanceAPIException as e:
            self.logger.error(f"Error getting account balance: {e}")
            return 0
    
    def set_leverage(self, symbol, leverage):
        """
        Set leverage for a symbol
        """
        try:
            response = self.client.futures_change_leverage(
                symbol=symbol,
                leverage=leverage
            )
            self.logger.info(f"Set leverage for {symbol} to {leverage}x")
            return response
        except BinanceAPIException as e:
            self.logger.error(f"Error setting leverage: {e}")
            return None
    
    def get_current_price(self, symbol):
        """
        Get current price of a symbol
        """
        try:
            ticker = self.client.futures_symbol_ticker(symbol=symbol)
            return float(ticker['price'])
        except BinanceAPIException as e:
            self.logger.error(f"Error getting current price: {e}")
            return None
    
    def get_position(self, symbol):
        """
        Get current open position for a symbol
        """
        try:
            positions = self.client.futures_position_information()
            for position in positions:
                if position['symbol'] == symbol:
                    return {
                        'symbol': position['symbol'],
                        'size': float(position['positionAmt']),
                        'entry_price': float(position['entryPrice']),
                        'mark_price': float(position['markPrice']),
                        'unrealized_pnl': float(position['unRealizedProfit']),
                        'liquidation_price': float(position['liquidationPrice']) if position['liquidationPrice'] != '0' else None
                    }
            return {'symbol': symbol, 'size': 0}
        except BinanceAPIException as e:
            self.logger.error(f"Error getting position info: {e}")
            return {'symbol': symbol, 'size': 0}
    
    def create_market_buy_order(self, symbol, quantity):
        """
        Create a market buy order
        """
        try:
            order = self.client.futures_create_order(
                symbol=symbol,
                side='BUY',
                type='MARKET',
                quantity=self._format_quantity(symbol, quantity)
            )
            self.logger.info(f"Created market buy order for {quantity} {symbol}")
            return order
        except BinanceAPIException as e:
            self.logger.error(f"Error creating market buy order: {e}")
            return None
    
    def create_market_sell_order(self, symbol, quantity):
        """
        Create a market sell order
        """
        try:
            order = self.client.futures_create_order(
                symbol=symbol,
                side='SELL',
                type='MARKET',
                quantity=self._format_quantity(symbol, quantity)
            )
            self.logger.info(f"Created market sell order for {quantity} {symbol}")
            return order
        except BinanceAPIException as e:
            self.logger.error(f"Error creating market sell order: {e}")
            return None
    
    def create_stop_loss_order(self, symbol, side, quantity, stop_price):
        """
        Create a stop loss order
        """
        try:
            order_side = 'BUY' if side == 'SHORT' else 'SELL'
            order = self.client.futures_create_order(
                symbol=symbol,
                side=order_side,
                type='STOP_MARKET',
                stopPrice=self._format_price(symbol, stop_price),
                closePosition='true'
            )
            self.logger.info(f"Set stop loss for {symbol} at {stop_price}")
            return order
        except BinanceAPIException as e:
            self.logger.error(f"Error creating stop loss order: {e}")
            return None
    
    def create_take_profit_order(self, symbol, side, quantity, take_profit_price):
        """
        Create a take profit order
        """
        try:
            order_side = 'BUY' if side == 'SHORT' else 'SELL'
            order = self.client.futures_create_order(
                symbol=symbol,
                side=order_side,
                type='TAKE_PROFIT_MARKET',
                stopPrice=self._format_price(symbol, take_profit_price),
                quantity=self._format_quantity(symbol, quantity)
            )
            self.logger.info(f"Set take profit for {symbol} at {take_profit_price}")
            return order
        except BinanceAPIException as e:
            self.logger.error(f"Error creating take profit order: {e}")
            return None
    
    def cancel_all_orders(self, symbol):
        """
        Cancel all open orders for a symbol
        """
        try:
            response = self.client.futures_cancel_all_open_orders(symbol=symbol)
            self.logger.info(f"Cancelled all open orders for {symbol}")
            return response
        except BinanceAPIException as e:
            self.logger.error(f"Error cancelling orders: {e}")
            return None
    
    def get_symbol_info(self, symbol):
        """
        Get symbol information
        """
        try:
            exchange_info = self.client.futures_exchange_info()
            for sym in exchange_info['symbols']:
                if sym['symbol'] == symbol:
                    return sym
            return None
        except BinanceAPIException as e:
            self.logger.error(f"Error getting symbol info: {e}")
            return None
    
    def _format_quantity(self, symbol, quantity):
        """
        Format quantity according to symbol's quantity precision
        """
        info = self.get_symbol_info(symbol)
        if info:
            step_size = None
            for filter in info['filters']:
                if filter['filterType'] == 'LOT_SIZE':
                    step_size = float(filter['stepSize'])
                    break
            
            if step_size:
                precision = int(round(-math.log10(step_size)))
                return round(quantity, precision)
        
        # Default to 3 decimal places if we can't get symbol info
        return round(quantity, 3)
    
    def _format_price(self, symbol, price):
        """
        Format price according to symbol's price precision
        """
        info = self.get_symbol_info(symbol)
        if info:
            tick_size = None
            for filter in info['filters']:
                if filter['filterType'] == 'PRICE_FILTER':
                    tick_size = float(filter['tickSize'])
                    break
            
            if tick_size:
                precision = int(round(-math.log10(tick_size)))
                return round(price, precision)
        
        # Default to 2 decimal places if we can't get symbol info
        return round(price, 2)
