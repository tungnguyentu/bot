import time
import requests
from datetime import datetime

from utils.logger import setup_logger

class TelegramNotifier:
    def __init__(self, token, chat_id):
        self.token = token
        self.chat_id = chat_id
        self.logger = setup_logger('telegram_notifier', 'logs/telegram.log')
        self.base_url = f"https://api.telegram.org/bot{token}/"
        
    def send_message(self, message):
        """
        Send a text message to the configured Telegram chat.
        
        Args:
            message (str): Message text to send
        
        Returns:
            bool: True if message was sent successfully, False otherwise
        """
        try:
            if not self.token or not self.chat_id:
                self.logger.warning("Telegram token or chat ID not configured")
                return False
                
            url = f"{self.base_url}sendMessage"
            data = {
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': 'Markdown'
            }
            
            response = requests.post(url, data=data, timeout=10)
            
            if response.status_code == 200:
                self.logger.info(f"Message sent successfully")
                return True
            else:
                self.logger.error(f"Failed to send message: {response.text}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error sending Telegram message: {str(e)}")
            return False
            
    def send_trade_notification(self, strategy_type, symbol, action, entry, stop_loss, take_profit, reasoning):
        """
        Send a formatted trade notification.
        
        Args:
            strategy_type (str): Type of trading strategy ('Scalping' or 'Swing Trading')
            symbol (str): Trading pair symbol
            action (str): Trade action ('BUY' or 'SELL')
            entry (float): Entry price
            stop_loss (float): Stop loss price
            take_profit (float): Take profit price
            reasoning (str): Trade reasoning in natural language
        """
        try:
            # Calculate risk-reward ratio
            risk = abs(entry - stop_loss)
            reward = abs(take_profit - entry)
            rr_ratio = reward / risk if risk > 0 else 0
            
            # Format action emoji
            action_emoji = "🟢 BUY" if action == "BUY" else "🔴 SELL"
            
            # Create the message
            message = (
                f"*{strategy_type} Signal: {action_emoji} {symbol}*\n\n"
                f"*Entry:* {entry:.2f}\n"
                f"*Stop Loss:* {stop_loss:.2f}\n"
                f"*Take Profit:* {take_profit:.2f}\n"
                f"*Risk:Reward:* 1:{rr_ratio:.2f}\n\n"
                f"*Signal Reasoning:*\n{reasoning}\n\n"
                f"*Time:* {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            
            self.send_message(message)
            
        except Exception as e:
            self.logger.error(f"Error sending trade notification: {str(e)}")
            
    def send_error_notification(self, error_message):
        """Send an error notification."""
        try:
            message = f"🚨 *ERROR*\n\n{error_message}\n\nTime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            self.send_message(message)
        except Exception as e:
            self.logger.error(f"Error sending error notification: {str(e)}")
