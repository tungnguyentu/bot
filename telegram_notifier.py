"""
Telegram notification service for the AI Trading Bot.
"""

import os
import logging
import telegram
from telegram.ext import Updater
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()

# Get Telegram credentials from environment variables
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

# Initialize logger
logger = logging.getLogger('trading_bot')


class TelegramNotifier:
    """
    Telegram notification service for the trading bot.
    """
    
    def __init__(self, token=None, chat_id=None):
        """
        Initialize Telegram notifier.
        
        Args:
            token (str): Telegram bot token
            chat_id (str): Telegram chat ID
        """
        self.token = token or TELEGRAM_BOT_TOKEN
        self.chat_id = chat_id or TELEGRAM_CHAT_ID
        self.bot = None
        
        if not self.token or not self.chat_id:
            logger.warning("Telegram credentials not found. Notifications will not be sent.")
            return
        
        try:
            self.bot = telegram.Bot(token=self.token)
            logger.info("Telegram bot initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize Telegram bot: {e}")
            self.bot = None
    
    def send_message(self, message):
        """
        Send message to Telegram chat.
        
        Args:
            message (str): Message to send
            
        Returns:
            bool: True if message was sent successfully, False otherwise
        """
        if not self.bot:
            logger.warning("Telegram bot not initialized. Message not sent.")
            return False
        
        try:
            self.bot.send_message(chat_id=self.chat_id, text=message, parse_mode='Markdown')
            logger.info("Telegram message sent successfully.")
            return True
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")
            return False
    
    def notify_trade_open(self, symbol, position_type, entry_price, position_size, stop_loss, take_profit):
        """
        Send notification about trade opening.
        
        Args:
            symbol (str): Trading symbol
            position_type (str): Position type ('long' or 'short')
            entry_price (float): Entry price
            position_size (float): Position size
            stop_loss (float): Stop loss price
            take_profit (float): Take profit price
        """
        message = f"🚀 *TRADE OPENED*\n\n" \
                 f"*Symbol:* {symbol}\n" \
                 f"*Position:* {'LONG 📈' if position_type == 'long' else 'SHORT 📉'}\n" \
                 f"*Entry Price:* {entry_price:.2f}\n" \
                 f"*Position Size:* {position_size:.4f}\n" \
                 f"*Stop Loss:* {stop_loss:.2f}\n" \
                 f"*Take Profit:* {take_profit:.2f}\n" \
                 f"*Time:* {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        self.send_message(message)
    
    def notify_trade_close(self, symbol, position_type, entry_price, exit_price, profit_loss, profit_loss_percent):
        """
        Send notification about trade closing.
        
        Args:
            symbol (str): Trading symbol
            position_type (str): Position type ('long' or 'short')
            entry_price (float): Entry price
            exit_price (float): Exit price
            profit_loss (float): Profit/loss amount
            profit_loss_percent (float): Profit/loss percentage
        """
        # Determine emoji based on profit/loss
        if profit_loss > 0:
            emoji = "✅"
            result = "PROFIT"
        else:
            emoji = "❌"
            result = "LOSS"
        
        message = f"{emoji} *TRADE CLOSED - {result}*\n\n" \
                 f"*Symbol:* {symbol}\n" \
                 f"*Position:* {'LONG 📈' if position_type == 'long' else 'SHORT 📉'}\n" \
                 f"*Entry Price:* {entry_price:.2f}\n" \
                 f"*Exit Price:* {exit_price:.2f}\n" \
                 f"*P/L:* {profit_loss:.4f} ({profit_loss_percent:.2f}%)\n" \
                 f"*Time:* {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        self.send_message(message)
    
    def notify_error(self, error_message):
        """
        Send notification about error.
        
        Args:
            error_message (str): Error message
        """
        message = f"⚠️ *ERROR*\n\n" \
                 f"{error_message}\n" \
                 f"*Time:* {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        self.send_message(message)
    
    def notify_system_status(self, status_message):
        """
        Send notification about system status.
        
        Args:
            status_message (str): Status message
        """
        message = f"ℹ️ *SYSTEM STATUS*\n\n" \
                 f"{status_message}\n" \
                 f"*Time:* {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        self.send_message(message) 