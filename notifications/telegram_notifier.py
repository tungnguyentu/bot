import logging
import requests
import time

logger = logging.getLogger("BinanceBot.TelegramNotifier")


class TelegramNotifier:
    def __init__(self, token, chat_id):
        self.token = token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{token}/"

    def send_message(self, message, retry=3):
        """
        Send a message to the Telegram chat
        """
        if not self.token or not self.chat_id:
            logger.warning(
                "Telegram token or chat_id not configured. Message not sent."
            )
            return False

        url = self.base_url + "sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": message,
            "parse_mode": "Markdown",
        }

        for attempt in range(retry):
            try:
                response = requests.post(url, json=payload, timeout=10)

                if response.status_code == 200:
                    logger.debug(f"Message sent to Telegram: {message[:50]}...")
                    return True
                else:
                    logger.warning(
                        f"Failed to send message to Telegram: {response.text}"
                    )

            except Exception as e:
                logger.error(f"Error sending message to Telegram: {str(e)}")

            # Only sleep if we're going to retry
            if attempt < retry - 1:
                time.sleep(2)

        return False

    def send_trade_alert(
        self, symbol, direction, entry_price, stop_loss, take_profit_1, take_profit_2, reasoning
    ):
        """
        Send a formatted trade alert to Telegram
        """
        emoji = "🟢" if direction == "LONG" else "🔴"
        
        # Format price values for better readability
        entry_str = f"{entry_price:.2f}" if entry_price >= 10 else f"{entry_price:.4f}"
        sl_str = f"{stop_loss:.2f}" if stop_loss >= 10 else f"{stop_loss:.4f}"
        tp1_str = f"{take_profit_1:.2f}" if take_profit_1 >= 10 else f"{take_profit_1:.4f}"
        tp2_str = f"{take_profit_2:.2f}" if take_profit_2 >= 10 else f"{take_profit_2:.4f}"
        
        message = (
            f"{emoji} *{direction} Trade on {symbol}*\n\n"
            f"*Entry:* {entry_str} USDT\n"
            f"*Stop Loss:* {sl_str} USDT\n"
            f"*Take Profit 1:* {tp1_str} USDT\n"
            f"*Take Profit 2:* {tp2_str} USDT\n\n"
            f"*Reasoning:*\n{reasoning}"
        )

        return self.send_message(message)

    def send_trade_update(self, symbol, direction, reason, pnl=None):
        """
        Send a trade update (SL adjustment, partial TP, etc.)
        """
        emoji = "🔄"
        message = f"{emoji} *Update on {direction} {symbol}*\n\n" f"{reason}"

        if pnl is not None:
            pnl_emoji = "✅" if pnl >= 0 else "❌"
            message += f"\n\n*P&L:* {pnl_emoji} {pnl:.2f}%"

        return self.send_message(message)

    def send_error(self, error_message):
        """
        Send an error notification
        """
        message = f"⚠️ *ERROR:*\n\n{error_message}"
        return self.send_message(message)
