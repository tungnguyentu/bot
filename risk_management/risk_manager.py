import logging
import pandas as pd
import numpy as np
from indicators.technical_indicators import TechnicalIndicators

logger = logging.getLogger("BinanceBot.RiskManager")


class RiskManager:
    def __init__(self, exchange, config):
        self.exchange = exchange
        self.config = config
        self.indicators = TechnicalIndicators()
        self.open_orders = {}  # Track open orders by symbol

    def can_open_trade(self, symbol):
        """
        Check if we can open a new trade based on risk parameters
        """
        # Check if we already have max number of open positions
        open_positions = 0
        for sym in self.config.TRADING_SYMBOLS:
            position = self.exchange.get_position(sym)
            if position["size"] != 0:
                open_positions += 1

        max_positions = len(self.config.TRADING_SYMBOLS)
        if open_positions >= max_positions:
            logger.info(
                f"Max positions reached ({open_positions}/{max_positions}). Cannot open new trade for {symbol}."
            )
            return False

        # Check account balance and ensure we have sufficient margin
        balance = self.exchange.get_account_balance()

        # Calculate max risk based on account balance
        max_risk_amount = balance * self.config.ACCOUNT_RISK_PER_TRADE

        if max_risk_amount < 10:  # Minimum viable trade size
            logger.info(
                f"Insufficient balance for new trade. Risk amount: ${max_risk_amount}"
            )
            return False

        # Set leverage for the symbol
        self.exchange.set_leverage(symbol, self.config.LEVERAGE)

        return True

    def calculate_position_size(self, symbol):
        """
        Calculate position size based on risk parameters
        """
        # Get account balance
        balance = self.exchange.get_account_balance()

        # Get current price
        price = self.exchange.get_current_price(symbol)

        # Get ATR to determine stop loss distance
        historical_data = self.exchange.get_historical_data(
            symbol, self.config.TIMEFRAME, 100
        )
        df = pd.DataFrame(historical_data)
        df = self.indicators.add_atr(df, self.config.ATR_PERIOD)
        atr = df["atr"].iloc[-1]

        # Calculate stop loss distance in price
        sl_distance = atr * self.config.SL_ATR_MULTIPLIER

        # Calculate risk per trade in USD
        risk_amount = balance * self.config.ACCOUNT_RISK_PER_TRADE

        # Calculate position size in the base currency
        # risk_amount = position_size * (sl_distance / price)
        position_size = (risk_amount * price) / sl_distance

        # Apply leverage
        position_size = position_size * self.config.LEVERAGE

        logger.info(
            f"Calculated position size for {symbol}: {position_size} "
            + f"(Risk: ${risk_amount}, Balance: ${balance}, SL distance: {sl_distance})"
        )

        return position_size

    def set_stop_loss_take_profit(self, symbol, direction, order):
        """
        Set stop loss and take profit orders for a position
        """
        try:
            # Cancel any existing orders first
            self.exchange.cancel_all_orders(symbol)

            # Get position information
            position = self.exchange.get_position(symbol)
            if position["size"] == 0:
                logger.warning(f"No position found for {symbol} when setting SL/TP")
                return

            # Get ATR value
            historical_data = self.exchange.get_historical_data(
                symbol, self.config.TIMEFRAME, 100
            )
            df = pd.DataFrame(historical_data)
            df = self.indicators.add_atr(df, self.config.ATR_PERIOD)
            atr = df["atr"].iloc[-1]

            entry_price = position["entry_price"]
            position_size = abs(position["size"])

            # Calculate stop loss price
            if direction == "LONG":
                stop_price = entry_price - (atr * self.config.SL_ATR_MULTIPLIER)
                partial_tp_price = entry_price + (
                    atr * self.config.PARTIAL_TP_ATR_MULTIPLIER
                )
                full_tp_price = entry_price + (atr * self.config.FULL_TP_ATR_MULTIPLIER)
            else:  # SHORT
                stop_price = entry_price + (atr * self.config.SL_ATR_MULTIPLIER)
                partial_tp_price = entry_price - (
                    atr * self.config.PARTIAL_TP_ATR_MULTIPLIER
                )
                full_tp_price = entry_price - (atr * self.config.FULL_TP_ATR_MULTIPLIER)

            # Set stop loss order
            sl_order = self.exchange.create_stop_loss_order(
                symbol, direction, position_size, stop_price
            )

            # Set take profit orders - partial and full
            partial_size = position_size * self.config.PARTIAL_TP_SIZE
            remaining_size = position_size - partial_size

            # Partial TP
            partial_tp_order = self.exchange.create_take_profit_order(
                symbol, direction, partial_size, partial_tp_price
            )

            # Full TP
            full_tp_order = self.exchange.create_take_profit_order(
                symbol, direction, remaining_size, full_tp_price
            )

            # Store order IDs for future reference
            self.open_orders[symbol] = {
                "stop_loss": sl_order,
                "partial_tp": partial_tp_order,
                "full_tp": full_tp_order,
            }

        except Exception as e:
            logger.error(f"Error setting SL/TP for {symbol}: {str(e)}")

    def update_risk_levels(self, symbol, position, historical_data):
        """
        Update stop-loss and take-profit levels based on price movement
        """
        if position["size"] == 0:
            return

        direction = "LONG" if position["size"] > 0 else "SHORT"

        # Calculate ATR
        df = pd.DataFrame(historical_data)
        df = self.indicators.add_atr(df, self.config.ATR_PERIOD)
        atr = df["atr"].iloc[-1]

        # Get the current price
        current_price = position["mark_price"]
        entry_price = position["entry_price"]

        # Calculate potential profit in percentage
        if direction == "LONG":
            profit_pct = (current_price - entry_price) / entry_price * 100
        else:  # SHORT
            profit_pct = (entry_price - current_price) / entry_price * 100

        # If in profit, consider moving stop loss to breakeven or trailing
        if profit_pct > 0.5:  # If more than 0.5% in profit
            # Cancel existing orders
            self.exchange.cancel_all_orders(symbol)

            position_size = abs(position["size"])

            # Calculate new stop loss (breakeven or trailing)
            if direction == "LONG":
                # Trailing stop loss at a distance of 1x ATR below current price
                new_stop = max(entry_price, current_price - (atr * 1.0))
                # Take profit remains the same
                partial_tp_price = entry_price + (
                    atr * self.config.PARTIAL_TP_ATR_MULTIPLIER
                )
                full_tp_price = entry_price + (atr * self.config.FULL_TP_ATR_MULTIPLIER)
            else:  # SHORT
                # Trailing stop loss at a distance of 1x ATR above current price
                new_stop = min(entry_price, current_price + (atr * 1.0))
                # Take profit remains the same
                partial_tp_price = entry_price - (
                    atr * self.config.PARTIAL_TP_ATR_MULTIPLIER
                )
                full_tp_price = entry_price - (atr * self.config.FULL_TP_ATR_MULTIPLIER)

            # Format price values for display
            new_stop_str = f"{new_stop:.2f}" if new_stop >= 10 else f"{new_stop:.4f}"
            old_stops = self.open_orders[symbol]['stop_loss'] if symbol in self.open_orders else None
            old_stop_price = old_stops['stopPrice'] if old_stops and 'stopPrice' in old_stops else None

            # Set updated stop loss order
            sl_order = self.exchange.create_stop_loss_order(
                symbol, direction, position_size, new_stop
            )

            # Set take profit orders - partial and full
            partial_size = position_size * self.config.PARTIAL_TP_SIZE
            remaining_size = position_size - partial_size

            # Partial TP
            partial_tp_order = self.exchange.create_take_profit_order(
                symbol, direction, partial_size, partial_tp_price
            )

            # Full TP
            full_tp_order = self.exchange.create_take_profit_order(
                symbol, direction, remaining_size, full_tp_price
            )

            # Update stored order IDs
            self.open_orders[symbol] = {
                "stop_loss": sl_order,
                "partial_tp": partial_tp_order,
                "full_tp": full_tp_order,
            }

            # Only send notification if stop loss was actually changed
            if old_stop_price is None or abs(new_stop - float(old_stop_price)) > 0.0001:
                # Send notification about updated stop loss
                logger.info(f"Updated stop loss for {symbol} {direction} position to {new_stop}")
                
                # Format the update message
                update_msg = f"🔒 Stop loss updated for {symbol} {direction}\n" \
                            f"New stop loss: {new_stop_str} USDT\n" \
                            f"Current price: {current_price:.2f} USDT\n" \
                            f"Current profit: {profit_pct:.2f}%"
                
                # Use the TelegramNotifier to send the update
                from notifications.telegram_notifier import TelegramNotifier
                notifier = TelegramNotifier(
                    token=self.config.TELEGRAM_BOT_TOKEN, 
                    chat_id=self.config.TELEGRAM_CHAT_ID
                )
                notifier.send_message(update_msg)
