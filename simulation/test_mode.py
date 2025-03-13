import logging
import time
import random
import math
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from binance.client import Client

logger = logging.getLogger("BinanceBot.TestModeExchange")


class TestModeExchange:
    """
    A simulated exchange for testing trading strategies without risking real money.
    This class mimics the behavior of the BinanceFutures class but executes trades in a simulation.
    """

    def __init__(self, starting_balance=1000.0, price_data_source=None):
        """
        Initialize the test exchange with a starting balance

        Args:
            starting_balance: The initial balance in USDT
            price_data_source: Optional real client to fetch initial price data
        """
        self.balance = starting_balance
        self.positions = {}  # Symbol -> position details
        self.orders = {}  # Symbol -> list of orders
        self.prices = {}  # Symbol -> current price
        self.price_data_source = price_data_source
        self.historical_data_cache = {}  # Cache for historical data

        # For simulation
        self.price_volatility = 0.002  # 0.2% price movement per update

        logger.info(f"Test mode exchange initialized with {starting_balance} USDT")

    def _generate_order_id(self):
        """Generate a random order ID"""
        return f"test_order_{int(time.time())}_{random.randint(1000, 9999)}"

    def get_historical_data(self, symbol, timeframe, limit=100):
        """
        Get historical kline/candlestick data for a symbol
        In test mode, we either use cached data or generate synthetic data
        """
        cache_key = f"{symbol}_{timeframe}_{limit}"

        # If we have it in cache and it's recent enough, return it
        if cache_key in self.historical_data_cache:
            # Add a small random price change to the last candle
            data = self.historical_data_cache[cache_key].copy()
            last_candle = data[-1].copy()
            price_change = last_candle["close"] * (1 + random.uniform(-0.003, 0.003))
            last_candle["close"] = price_change
            data[-1] = last_candle

            # Update current price
            self.prices[symbol] = price_change

            return data

        # If we have a real client, try to get real data first
        if self.price_data_source:
            try:
                interval_map = {
                    "1m": Client.KLINE_INTERVAL_1MINUTE,
                    "5m": Client.KLINE_INTERVAL_5MINUTE,
                    "15m": Client.KLINE_INTERVAL_15MINUTE,
                    "30m": Client.KLINE_INTERVAL_30MINUTE,
                    "1h": Client.KLINE_INTERVAL_1HOUR,
                    "4h": Client.KLINE_INTERVAL_4HOUR,
                    "1d": Client.KLINE_INTERVAL_1DAY,
                }

                klines = self.price_data_source.futures_klines(
                    symbol=symbol, interval=interval_map[timeframe], limit=limit
                )

                # Convert klines to a more usable format
                data = []
                for k in klines:
                    data.append(
                        {
                            "timestamp": k[0],
                            "open": float(k[1]),
                            "high": float(k[2]),
                            "low": float(k[3]),
                            "close": float(k[4]),
                            "volume": float(k[5]),
                        }
                    )

                # Cache it
                self.historical_data_cache[cache_key] = data

                # Set current price
                self.prices[symbol] = float(data[-1]["close"])

                return data

            except Exception as e:
                logger.warning(f"Error getting real historical data: {e}")

        # Generate synthetic data if we don't have real data
        data = self._generate_synthetic_data(symbol, timeframe, limit)
        self.historical_data_cache[cache_key] = data

        return data

    def _generate_synthetic_data(self, symbol, timeframe, limit):
        """Generate synthetic candlestick data for testing"""
        # Base price depends on the symbol
        if symbol == "BTCUSDT":
            base_price = 40000
        elif symbol == "ETHUSDT":
            base_price = 2500
        elif symbol == "BNBUSDT":
            base_price = 350
        elif symbol == "SOLUSDT":
            base_price = 120
        else:
            base_price = 100

        # Amount of time per candle
        timeframe_minutes = {
            "1m": 1,
            "5m": 5,
            "15m": 15,
            "30m": 30,
            "1h": 60,
            "4h": 240,
            "1d": 1440,
        }
        minutes = timeframe_minutes.get(timeframe, 15)

        # Generate timestamps
        end_time = int(time.time() * 1000)
        timestamps = [end_time - (i * minutes * 60 * 1000) for i in range(limit)]
        timestamps.reverse()

        # Generate price data with a random walk
        data = []
        current_price = base_price

        for i in range(limit):
            # Add some randomness to price movements
            change_percent = random.uniform(-0.01, 0.01)
            price_change = current_price * change_percent
            current_price += price_change

            # Generate candle high/low with some volatility around open/close
            if i == 0:
                open_price = current_price * (1 - random.uniform(0, 0.005))
            else:
                open_price = data[i - 1]["close"]

            close_price = current_price
            high_price = max(open_price, close_price) * (
                1 + random.uniform(0.001, 0.008)
            )
            low_price = min(open_price, close_price) * (
                1 - random.uniform(0.001, 0.008)
            )

            # Generate random volume
            volume = base_price * random.uniform(10, 100)

            data.append(
                {
                    "timestamp": timestamps[i],
                    "open": open_price,
                    "high": high_price,
                    "low": low_price,
                    "close": close_price,
                    "volume": volume,
                }
            )

        # Cache the current price
        self.prices[symbol] = current_price

        return data

    def update_simulated_prices(self):
        """Update simulated prices and check if any orders are triggered"""
        for symbol in self.prices:
            # Update price with small random movement
            price_change = self.prices[symbol] * random.uniform(
                -self.price_volatility, self.price_volatility
            )
            new_price = self.prices[symbol] + price_change
            self.prices[symbol] = new_price

            # Check if any orders are triggered
            if symbol in self.orders:
                triggered_orders = []

                for order in self.orders[symbol]:
                    # Check stop loss and take profit orders
                    if order["type"] in ["STOP_MARKET", "TAKE_PROFIT_MARKET"]:
                        # For long positions
                        if order["position_side"] == "LONG":
                            if (
                                order["type"] == "STOP_MARKET"
                                and new_price <= order["stop_price"]
                            ) or (
                                order["type"] == "TAKE_PROFIT_MARKET"
                                and new_price >= order["stop_price"]
                            ):
                                triggered_orders.append(order)
                        # For short positions
                        elif order["position_side"] == "SHORT":
                            if (
                                order["type"] == "STOP_MARKET"
                                and new_price >= order["stop_price"]
                            ) or (
                                order["type"] == "TAKE_PROFIT_MARKET"
                                and new_price <= order["stop_price"]
                            ):
                                triggered_orders.append(order)

                # Execute triggered orders
                for order in triggered_orders:
                    self._execute_order(symbol, order)
                    self.orders[symbol].remove(order)

    def _execute_order(self, symbol, order):
        """Execute a simulated order"""
        # Calculate PnL
        if symbol in self.positions and self.positions[symbol]["size"] != 0:
            position = self.positions[symbol]
            current_price = self.prices[symbol]

            if position["size"] > 0:  # Long position
                pnl = (current_price - position["entry_price"]) * position["size"]
                pnl_percentage = ((current_price / position["entry_price"]) - 1) * 100
            else:  # Short position
                pnl = (position["entry_price"] - current_price) * abs(position["size"])
                pnl_percentage = ((position["entry_price"] / current_price) - 1) * 100

            # Close position partially or fully
            quantity = (
                min(abs(position["size"]), order["quantity"])
                if "quantity" in order
                else abs(position["size"])
            )
            direction = "LONG" if position["size"] > 0 else "SHORT"

            # Log the execution
            order_type = (
                "Stop Loss" if order["type"] == "STOP_MARKET" else "Take Profit"
            )
            
            # Format price values for display
            price_str = f"{current_price:.2f}" if current_price >= 10 else f"{current_price:.4f}"
            entry_str = f"{position['entry_price']:.2f}" if position['entry_price'] >= 10 else f"{position['entry_price']:.4f}"
            
            logger.info(
                f"TEST MODE: {order_type} triggered for {symbol} {direction}\n"
                f"Entry: {entry_str} USDT\n"
                f"Exit: {price_str} USDT\n"
                f"P&L: {pnl_percentage:.2f}%"
            )

            # Update balance
            self.balance += pnl

            # Update position
            if quantity == abs(position["size"]):
                self.positions[symbol]["size"] = 0
            else:
                if position["size"] > 0:
                    self.positions[symbol]["size"] -= quantity
                else:
                    self.positions[symbol]["size"] += quantity

    def get_account_balance(self):
        """Get test account balance"""
        return self.balance

    def set_leverage(self, symbol, leverage):
        """Set leverage for a symbol (simulated)"""
        logger.info(f"TEST MODE: Set leverage for {symbol} to {leverage}x")
        return {"leverage": leverage, "symbol": symbol}

    def get_current_price(self, symbol):
        """Get current price of a symbol"""
        # If we don't have a price yet, generate one
        if symbol not in self.prices:
            # Generate a reasonable initial price based on symbol
            if symbol == "BTCUSDT":
                self.prices[symbol] = random.uniform(38000, 42000)
            elif symbol == "ETHUSDT":
                self.prices[symbol] = random.uniform(2300, 2700)
            elif symbol == "BNBUSDT":
                self.prices[symbol] = random.uniform(330, 370)
            elif symbol == "SOLUSDT":
                self.prices[symbol] = random.uniform(100, 140)
            else:
                self.prices[symbol] = 100.0

        return self.prices[symbol]

    def get_position(self, symbol):
        """Get current open position for a symbol"""
        if symbol not in self.positions:
            self.positions[symbol] = {
                "symbol": symbol,
                "size": 0,
                "entry_price": 0,
                "mark_price": self.get_current_price(symbol),
                "unrealized_pnl": 0,
                "liquidation_price": None,
            }
        else:
            # Update mark price and unrealized PnL
            current_price = self.get_current_price(symbol)
            position = self.positions[symbol]

            if position["size"] != 0:
                if position["size"] > 0:  # Long position
                    position["unrealized_pnl"] = (
                        current_price - position["entry_price"]
                    ) * position["size"]
                else:  # Short position
                    position["unrealized_pnl"] = (
                        position["entry_price"] - current_price
                    ) * abs(position["size"])

            position["mark_price"] = current_price

        return self.positions[symbol]

    def create_market_buy_order(self, symbol, quantity):
        """Create a simulated market buy order"""
        current_price = self.get_current_price(symbol)

        # Initialize orders list if not exists
        if symbol not in self.orders:
            self.orders[symbol] = []

        # Initialize position if not exists
        if symbol not in self.positions:
            self.positions[symbol] = {
                "symbol": symbol,
                "size": 0,
                "entry_price": 0,
                "mark_price": current_price,
                "unrealized_pnl": 0,
                "liquidation_price": None,
            }

        # Calculate total cost with slippage
        slippage = current_price * 0.001  # 0.1% slippage
        execution_price = current_price + slippage
        cost = quantity * execution_price

        # Update position
        if self.positions[symbol]["size"] != 0:
            # If we already have a position, calculate the new average entry price
            total_size = self.positions[symbol]["size"] + quantity
            avg_price = (
                self.positions[symbol]["entry_price"] * self.positions[symbol]["size"]
                + execution_price * quantity
            ) / total_size
            self.positions[symbol]["entry_price"] = avg_price
            self.positions[symbol]["size"] = total_size
        else:
            # New position
            self.positions[symbol]["entry_price"] = execution_price
            self.positions[symbol]["size"] = quantity

        self.positions[symbol]["mark_price"] = current_price

        # Deduct from balance
        self.balance -= cost

        order_id = self._generate_order_id()
        logger.info(
            f"TEST MODE: Created market buy order for {quantity} {symbol} at {execution_price} USDT"
        )

        return {
            "orderId": order_id,
            "symbol": symbol,
            "status": "FILLED",
            "price": str(execution_price),
            "origQty": str(quantity),
            "executedQty": str(quantity),
            "side": "BUY",
        }

    def create_market_sell_order(self, symbol, quantity):
        """Create a simulated market sell order"""
        current_price = self.get_current_price(symbol)

        # Initialize orders list if not exists
        if symbol not in self.orders:
            self.orders[symbol] = []

        # Initialize position if not exists
        if symbol not in self.positions:
            self.positions[symbol] = {
                "symbol": symbol,
                "size": 0,
                "entry_price": 0,
                "mark_price": current_price,
                "unrealized_pnl": 0,
                "liquidation_price": None,
            }

        # Calculate total proceeds with slippage
        slippage = current_price * 0.001  # 0.1% slippage
        execution_price = current_price - slippage
        proceeds = quantity * execution_price

        # Update position
        if self.positions[symbol]["size"] != 0:
            # If we already have a position, calculate the new average entry price
            total_size = self.positions[symbol]["size"] - quantity

            if abs(total_size) < 0.0001:  # Close to zero, consider fully closed
                self.positions[symbol]["size"] = 0
                self.positions[symbol]["entry_price"] = 0
            else:
                # If position direction changes, reset entry price
                if self.positions[symbol]["size"] > 0 and total_size < 0:
                    self.positions[symbol]["entry_price"] = execution_price
                    self.positions[symbol]["size"] = total_size
                else:
                    # Calculate new average if adding to existing short or reducing long
                    if self.positions[symbol]["size"] < 0:  # Existing short
                        avg_price = (
                            self.positions[symbol]["entry_price"]
                            * abs(self.positions[symbol]["size"])
                            + execution_price * quantity
                        ) / abs(total_size)
                        self.positions[symbol]["entry_price"] = avg_price

                    self.positions[symbol]["size"] = total_size
        else:
            # New short position
            self.positions[symbol]["entry_price"] = execution_price
            self.positions[symbol]["size"] = -quantity

        self.positions[symbol]["mark_price"] = current_price

        # Add to balance
        self.balance += proceeds

        order_id = self._generate_order_id()
        logger.info(
            f"TEST MODE: Created market sell order for {quantity} {symbol} at {execution_price} USDT"
        )

        return {
            "orderId": order_id,
            "symbol": symbol,
            "status": "FILLED",
            "price": str(execution_price),
            "origQty": str(quantity),
            "executedQty": str(quantity),
            "side": "SELL",
        }

    def create_stop_loss_order(self, symbol, side, quantity, stop_price):
        """Create a simulated stop loss order"""
        if symbol not in self.orders:
            self.orders[symbol] = []

        order_id = self._generate_order_id()

        order = {
            "orderId": order_id,
            "symbol": symbol,
            "status": "NEW",
            "type": "STOP_MARKET",
            "side": "BUY" if side == "SHORT" else "SELL",
            "position_side": side,
            "stop_price": stop_price,
            "quantity": quantity,
        }

        self.orders[symbol].append(order)
        logger.info(f"TEST MODE: Set stop loss for {symbol} at {stop_price}")

        return {"orderId": order_id, "symbol": symbol, "status": "NEW"}

    def create_take_profit_order(self, symbol, side, quantity, take_profit_price):
        """Create a simulated take profit order"""
        if symbol not in self.orders:
            self.orders[symbol] = []

        order_id = self._generate_order_id()

        order = {
            "orderId": order_id,
            "symbol": symbol,
            "status": "NEW",
            "type": "TAKE_PROFIT_MARKET",
            "side": "BUY" if side == "SHORT" else "SELL",
            "position_side": side,
            "stop_price": take_profit_price,
            "quantity": quantity,
        }

        self.orders[symbol].append(order)
        logger.info(f"TEST MODE: Set take profit for {symbol} at {take_profit_price}")

        return {"orderId": order_id, "symbol": symbol, "status": "NEW"}

    def cancel_all_orders(self, symbol):
        """Cancel all simulated open orders for a symbol"""
        if symbol in self.orders:
            order_count = len(self.orders[symbol])
            self.orders[symbol] = []
            logger.info(f"TEST MODE: Cancelled {order_count} orders for {symbol}")
            return {"symbol": symbol, "count": order_count}

        return {"symbol": symbol, "count": 0}

    def get_symbol_info(self, symbol):
        """Get simulated symbol information"""
        # Return a simplified version of symbol info with standard filters
        return {
            "symbol": symbol,
            "filters": [
                {"filterType": "PRICE_FILTER", "tickSize": "0.01"},
                {"filterType": "LOT_SIZE", "stepSize": "0.001"},
            ],
        }
