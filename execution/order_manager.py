import logging
import time
import sys
from datetime import datetime
from binance.exceptions import BinanceAPIException

sys.path.append('/Users/tungnt/Downloads/game')
import config

logger = logging.getLogger(__name__)

class OrderManager:
    def __init__(self, client, risk_manager):
        """
        Initialize the Order Manager.
        
        Args:
            client: Binance API client instance
            risk_manager: RiskManager instance
        """
        self.client = client
        self.risk_manager = risk_manager
        
    def place_order(self, symbol, side, order_type, quantity=None, price=None,
                   stop_price=None, take_profit=None, stop_loss=None, 
                   time_in_force="GTC", reduce_only=False, close_position=False):
        """
        Place an order on Binance Futures.
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            side: 'BUY' or 'SELL'
            order_type: 'LIMIT', 'MARKET', 'STOP', 'TAKE_PROFIT'
            quantity: Order quantity
            price: Order price (required for limit orders)
            stop_price: Stop price for stop orders
            take_profit: Take profit price
            stop_loss: Stop loss price
            time_in_force: Time in force, default 'GTC' (Good Till Cancelled)
            reduce_only: If True, the order will only reduce position
            close_position: If True, the order will close position
            
        Returns:
            dict: Order information or None on failure
        """
        try:
            params = {
                "symbol": symbol,
                "side": side,
                "type": order_type,
                "timeInForce": time_in_force if order_type != "MARKET" else None,
                "reduceOnly": reduce_only,
            }
            
            # Add quantity or closePosition based on parameters
            if close_position:
                params["closePosition"] = True
            elif quantity:
                params["quantity"] = quantity
                
            # Add price for non-market orders
            if order_type != "MARKET" and price:
                params["price"] = price
                
            # Add stop price for stop orders
            if order_type in ["STOP", "STOP_MARKET", "TAKE_PROFIT", "TAKE_PROFIT_MARKET"] and stop_price:
                params["stopPrice"] = stop_price
                
            # Clean None values
            params = {k: v for k, v in params.items() if v is not None}
                
            # Place the order
            if config.TRADING_MODE == "live_trading":
                order = self.client.futures_create_order(**params)
                logger.info(f"Order placed: {order}")
                
                # If this is an entry order, place take profit and stop loss orders
                if not reduce_only and not close_position and order["status"] == "FILLED":
                    if take_profit:
                        tp_side = "SELL" if side == "BUY" else "BUY"
                        self.place_order(
                            symbol=symbol,
                            side=tp_side,
                            order_type="TAKE_PROFIT",
                            quantity=quantity,
                            price=take_profit,
                            stop_price=take_profit,
                            reduce_only=True
                        )
                        
                    if stop_loss:
                        sl_side = "SELL" if side == "BUY" else "BUY"
                        self.place_order(
                            symbol=symbol,
                            side=sl_side,
                            order_type="STOP",
                            quantity=quantity,
                            price=stop_loss,
                            stop_price=stop_loss,
                            reduce_only=True
                        )
                
                return order
            else:
                # Simulate order for backtesting/paper trading
                simulated_order = {
                    "symbol": symbol,
                    "side": side,
                    "type": order_type,
                    "quantity": quantity,
                    "price": price if price else self._get_market_price(symbol),
                    "status": "FILLED",  # Assume immediate fill for simulation
                    "time": int(time.time() * 1000),
                    "orderId": int(time.time() * 1000)  # Use timestamp as dummy order ID
                }
                logger.info(f"Simulated order: {simulated_order}")
                return simulated_order
                
        except BinanceAPIException as e:
            logger.error(f"Binance API error placing order: {e}")
            return None
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return None
            
    def execute_entry(self, symbol, side, price=None, risk_per_trade=None, stop_loss=None, take_profit=None):
        """
        Execute an entry order with calculated risk management.
        
        Args:
            symbol: Trading symbol
            side: 'BUY' or 'SELL'
            price: Entry price (None for market price)
            risk_per_trade: Maximum risk per trade in USD
            stop_loss: Stop loss price
            take_profit: Take profit price
            
        Returns:
            dict: Order information or None on failure
        """
        # Check if we can open a new trade based on risk rules
        if not self.risk_manager.can_open_new_trade(symbol):
            logger.warning(f"Cannot open new trade for {symbol} due to risk constraints")
            return None
        
        # Set leverage
        try:
            if config.TRADING_MODE == "live_trading":
                self.client.futures_change_leverage(symbol=symbol, leverage=config.TRADING_LEVERAGE)
        except Exception as e:
            logger.error(f"Error setting leverage: {e}")
        
        # Get current market price if not provided
        if not price:
            ticker = self.client.futures_ticker(symbol=symbol)
            price = float(ticker['lastPrice'])
        
        # Determine order type
        order_type = "LIMIT" if price else "MARKET"
        
        # Calculate stop loss if not provided
        if not stop_loss:
            # Default stop loss calculation
            stop_loss = price * 0.99 if side == "BUY" else price * 1.01
            
        # Calculate take profit if not provided
        if not take_profit:
            # Default take profit calculation
            take_profit = price * 1.02 if side == "BUY" else price * 0.98
        
        # Calculate position size based on risk
        quantity = self.risk_manager.calculate_position_size(
            symbol=symbol,
            entry_price=price,
            stop_loss_price=stop_loss,
            max_risk_usd=risk_per_trade
        )
        
        if quantity <= 0:
            logger.warning(f"Calculated position size too small for {symbol}")
            return None
        
        # Place the order
        order = self.place_order(
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price if order_type == "LIMIT" else None,
            take_profit=take_profit,
            stop_loss=stop_loss
        )
        
        if order:
            # Register the position with the risk manager
            position_data = {
                "entry_price": price,
                "quantity": quantity,
                "side": side,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "entry_time": datetime.now(),
                "order_id": order.get("orderId", 0)
            }
            self.risk_manager.register_open_position(symbol, position_data)
            
        return order
        
    def execute_exit(self, symbol, side, quantity=None, price=None, close_position=False):
        """
        Execute an exit order.
        
        Args:
            symbol: Trading symbol
            side: 'BUY' or 'SELL' (opposite of entry side)
            quantity: Quantity to exit (None to close entire position)
            price: Exit price (None for market price)
            close_position: If True, close entire position regardless of quantity
            
        Returns:
            dict: Order information or None on failure
        """
        # Determine order type
        order_type = "LIMIT" if price else "MARKET"
        
        # Place the exit order
        order = self.place_order(
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            close_position=close_position,
            reduce_only=True if not close_position else None
        )
        
        if order and (close_position or (quantity and quantity >= self.risk_manager.open_positions.get(symbol, {}).get('quantity', 0))):
            # Unregister the position
            self.risk_manager.unregister_position(symbol)
            
        return order
        
    def update_trailing_stop(self, symbol, current_price):
        """
        Update trailing stop for an open position if price has moved in favorable direction.
        
        Args:
            symbol: Trading symbol
            current_price: Current market price
        """
        if symbol not in self.risk_manager.open_positions:
            return
        
        position = self.risk_manager.open_positions[symbol]
        side = position.get('side')
        entry_price = position.get('entry_price')
        stop_loss = position.get('stop_loss')
        
        # Check if price has moved enough to update trailing stop
        if side == "BUY":
            # For long positions
            price_movement = (current_price - entry_price) / entry_price
            if price_movement > config.TRAILING_STOP_ACTIVATION:
                # Calculate new stop loss that locks in some profit
                new_stop = current_price * (1 - config.TRAILING_STOP_ACTIVATION/2)
                if new_stop > stop_loss:
                    # Update stop loss if new level is higher
                    position['stop_loss'] = new_stop
                    logger.info(f"Updated trailing stop for {symbol} to {new_stop}")
                    
                    # In live trading, modify the stop loss order
                    if config.TRADING_MODE == "live_trading":
                        try:
                            # Cancel existing stop loss order and create new one
                            # Implementation depends on how we're tracking order IDs
                            pass
                        except Exception as e:
                            logger.error(f"Error updating trailing stop: {e}")
        else:
            # For short positions
            price_movement = (entry_price - current_price) / entry_price
            if price_movement > config.TRAILING_STOP_ACTIVATION:
                # Calculate new stop loss that locks in some profit
                new_stop = current_price * (1 + config.TRAILING_STOP_ACTIVATION/2)
                if new_stop < stop_loss:
                    # Update stop loss if new level is lower
                    position['stop_loss'] = new_stop
                    logger.info(f"Updated trailing stop for {symbol} to {new_stop}")
                    
                    # In live trading, modify the stop loss order
                    if config.TRADING_MODE == "live_trading":
                        try:
                            # Cancel existing stop loss order and create new one
                            # Implementation depends on how we're tracking order IDs
                            pass
                        except Exception as e:
                            logger.error(f"Error updating trailing stop: {e}")
                    
    def _get_market_price(self, symbol):
        """Get current market price for a symbol."""
        try:
            ticker = self.client.futures_ticker(symbol=symbol)
            return float(ticker['lastPrice'])
        except Exception as e:
            logger.error(f"Error getting market price: {e}")
            return None
            
    def cancel_all_orders(self, symbol):
        """Cancel all open orders for a symbol."""
        try:
            if config.TRADING_MODE == "live_trading":
                self.client.futures_cancel_all_open_orders(symbol=symbol)
                logger.info(f"Cancelled all open orders for {symbol}")
            return True
        except Exception as e:
            logger.error(f"Error cancelling orders: {e}")
            return False
