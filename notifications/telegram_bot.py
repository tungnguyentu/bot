import logging
import os
import sys
import asyncio
from telegram import Bot
from telegram.error import TelegramError
import threading
import queue
from datetime import datetime
import time
import re

# Add the parent directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logger = logging.getLogger(__name__)

class TelegramNotifier:
    def __init__(self):
        """Initialize the Telegram notification system."""
        self.bot_token = config.TELEGRAM_BOT_TOKEN
        self.chat_id = config.TELEGRAM_CHAT_ID
        self.bot = Bot(token=self.bot_token) if self.bot_token else None
        self.message_queue = queue.Queue()
        self.is_running = False
        
        # Start message processing thread if credentials are available
        if self.bot_token and self.chat_id:
            self.start_message_thread()
        else:
            logger.warning("Telegram notifications disabled. Bot token or chat ID not provided.")
            
    def start_message_thread(self):
        """Start a thread for processing and sending messages."""
        if not self.is_running:
            self.is_running = True
            self.message_thread = threading.Thread(target=self._process_message_queue, daemon=True)
            self.message_thread.start()
            logger.info("Telegram message processing thread started")
            
    def _process_message_queue(self):
        """Process messages in the queue and send them."""
        while self.is_running:
            try:
                if not self.message_queue.empty():
                    message = self.message_queue.get()
                    self._send_message_sync(message)  # Use sync version instead of async
                    self.message_queue.task_done()
                else:
                    # Sleep briefly to avoid high CPU usage
                    time.sleep(0.5)
            except Exception as e:
                logger.error(f"Error processing message queue: {e}")
                time.sleep(5)  # Longer sleep on error
    
    def _send_message_sync(self, message):
        """Send a message synchronously to the configured Telegram chat."""
        if not self.bot:
            return
            
        try:
            # Sanitize markdown to prevent parsing errors
            message = self._sanitize_markdown(message)
            
            # Use send_message synchronously
            self.bot.send_message(chat_id=self.chat_id, text=message, parse_mode="Markdown")
            logger.debug(f"Telegram message sent: {message[:50]}...")
        except TelegramError as e:
            logger.error(f"Error sending Telegram message: {e}")
        except Exception as e:
            logger.error(f"Unexpected error sending Telegram message: {e}")
    
    async def _send_message(self, message):
        """This async version is kept for compatibility but not used."""
        return  # Do nothing - we're using the sync version instead
    
    def _sanitize_markdown(self, text):
        """
        Sanitize markdown text to prevent parsing errors.
        Escapes characters that might cause issues with Telegram's Markdown parser.
        """
        # Replace problematic characters in code blocks and other markdown
        # Ensure closing characters for * and _ and `
        
        # First fix any unclosed formatting
        open_stars = text.count('*') 
        if open_stars % 2 != 0:
            text += '*'  # Add closing star
            
        open_underscores = text.count('_')
        if open_underscores % 2 != 0:
            text += '_'  # Add closing underscore
            
        # If there are backticks, make sure they're properly closed
        open_backticks = text.count('`')
        if open_backticks % 2 != 0:
            text += '`'  # Add closing backtick
            
        return text
            
    def send_trade_notification(self, action, symbol, side, quantity, price, reasoning=None):
        """
        Send a notification about a trade.
        
        Args:
            action: The action taken (e.g., 'Entry', 'Exit', 'Stop Loss')
            symbol: Trading symbol
            side: 'BUY' or 'SELL'
            quantity: Trade quantity
            price: Trade price
            reasoning: Optional explanation of the trade decision
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        message = f"🤖 *TRADING BOT ALERT* 🤖\n\n"
        message += f"*{action.upper()}*: {symbol}\n"
        
        # Add emoji based on side
        if side.upper() == "BUY":
            message += "🟢 *LONG Position*\n"
        else:
            message += "🔴 *SHORT Position*\n"
            
        message += f"*Price*: ${price:.2f}\n"
        message += f"*Quantity*: {quantity}\n"
        message += f"*Time*: {timestamp}\n"
        
        if reasoning:
            # Sanitize reasoning text
            reasoning = reasoning.replace('*', '').replace('_', '').replace('`', '')
            message += f"\n*Analysis*:\n{reasoning}\n"
            
        # Add a footer
        message += f"\n_Trading Mode: {config.TRADING_MODE}_"
        
        self.message_queue.put(message)
        
    def send_performance_update(self, metrics):
        """
        Send a performance metrics update.
        
        Args:
            metrics: Dictionary of performance metrics
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        message = f"📊 *PERFORMANCE UPDATE* 📊\n\n"
        message += f"*Total Trades*: {metrics.get('total_trades', 0)}\n"
        message += f"*Win Rate*: {metrics.get('win_rate', 0)*100:.1f}%\n"
        message += f"*Profit Factor*: {metrics.get('profit_factor', 0):.2f}\n"
        message += f"*Max Drawdown*: {metrics.get('max_drawdown', 0)*100:.1f}%\n"
        message += f"*Avg. Profit/Trade*: ${metrics.get('avg_profit_per_trade', 0):.2f}\n"
        message += f"*Time*: {timestamp}\n"
        
        self.message_queue.put(message)
        
    def send_error_alert(self, error_message):
        """Send an alert about a system error."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        message = f"⚠️ *ERROR ALERT* ⚠️\n\n"
        message += f"{error_message}\n"
        message += f"*Time*: {timestamp}\n"
        
        self.message_queue.put(message)
        
    def send_system_status(self, status):
        """Send system status information."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        message = f"ℹ️ *SYSTEM STATUS* ℹ️\n\n"
        message += f"{status}\n"
        message += f"*Time*: {timestamp}\n"
        
        self.message_queue.put(message)
