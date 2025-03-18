import logging
import sys
import asyncio
from telegram import Bot
from telegram.error import TelegramError
import threading
import queue
from datetime import datetime

sys.path.append('/Users/tungnt/Downloads/game')
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
                    asyncio.run(self._send_message(message))
                    self.message_queue.task_done()
                else:
                    # Sleep briefly to avoid high CPU usage
                    time.sleep(0.5)
            except Exception as e:
                logger.error(f"Error processing message queue: {e}")
                time.sleep(5)  # Longer sleep on error
                
    async def _send_message(self, message):
        """Send a message to the configured Telegram chat."""
        if not self.bot:
            return
            
        try:
            await self.bot.send_message(chat_id=self.chat_id, text=message, parse_mode="Markdown")
            logger.debug(f"Telegram message sent: {message[:50]}...")
        except TelegramError as e:
            logger.error(f"Error sending Telegram message: {e}")
        except Exception as e:
            logger.error(f"Unexpected error sending Telegram message: {e}")
            
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
