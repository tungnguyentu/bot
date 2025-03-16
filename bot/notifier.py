import logging
import os
import requests
from datetime import datetime

logger = logging.getLogger(__name__)

class TelegramNotifier:
    """Sends notifications to Telegram."""
    
    def __init__(self):
        """Initialize Telegram notifier using environment variables."""
        self.token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.chat_id = os.getenv('TELEGRAM_CHAT_ID')
        self.enabled = self.token is not None and self.chat_id is not None
        
        if not self.enabled:
            logger.warning("Telegram notifications disabled: missing BOT_TOKEN or CHAT_ID")
        else:
            logger.info("Telegram notifier initialized")
    
    def send_message(self, message):
        """Send a text message to Telegram."""
        if not self.enabled:
            logger.info(f"Telegram message (disabled): {message}")
            return
        
        try:
            url = f"https://api.telegram.org/bot{self.token}/sendMessage"
            data = {
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": "HTML"
            }
            response = requests.post(url, data=data)
            
            if response.status_code != 200:
                logger.error(f"Failed to send Telegram message: {response.text}")
            
        except Exception as e:
            logger.error(f"Error sending Telegram message: {e}")
    
    def send_trade_entry(self, symbol, side, entry_price, sl_price, tp_price, confidence, reason):
        """Send a notification for trade entry."""
        emoji = "🟢" if side == "BUY" else "🔴"
        message = (
            f"{emoji} <b>New {side} Signal</b>\n\n"
            f"<b>Symbol:</b> {symbol}\n"
            f"<b>Entry Price:</b> {entry_price:.4f}\n"
            f"<b>Stop Loss:</b> {sl_price:.4f}\n"
            f"<b>Take Profit:</b> {tp_price:.4f}\n"
            f"<b>Risk/Reward:</b> {abs((tp_price - entry_price) / (entry_price - sl_price)):.2f}\n"
            f"<b>Confidence:</b> {confidence:.2%}\n"
            f"<b>Signal Reason:</b> {reason}\n"
            f"<b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        self.send_message(message)
    
    def send_trade_exit(self, symbol, exit_price, pnl, reason):
        """Send a notification for trade exit."""
        emoji = "✅" if pnl > 0 else "❌"
        message = (
            f"{emoji} <b>Position Closed</b>\n\n"
            f"<b>Symbol:</b> {symbol}\n"
            f"<b>Exit Price:</b> {exit_price:.4f}\n"
            f"<b>PnL:</b> {pnl:.2f} ({pnl:.2%})\n"
            f"<b>Exit Reason:</b> {reason}\n"
            f"<b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        self.send_message(message)
    
    def send_performance_report(self, results):
        """Send a performance report."""
        message = (
            f"📊 <b>Performance Report</b>\n\n"
            f"<b>Total Trades:</b> {results['total_trades']}\n"
            f"<b>Win Rate:</b> {results['win_rate']:.2f}%\n"
            f"<b>Profit Factor:</b> {results['profit_factor']:.2f}\n"
            f"<b>Total Return:</b> {results['total_return']:.2f}%\n"
            f"<b>Starting Balance:</b> ${results['starting_balance']:.2f}\n"
            f"<b>Ending Balance:</b> ${results['ending_balance']:.2f}\n"
            f"<b>Max Drawdown:</b> {results['max_drawdown']:.2f}%\n"
        )
        self.send_message(message)
    
    def send_error(self, error_message):
        """Send an error notification."""
        message = f"🚨 <b>Error</b>\n\n{error_message}"
        self.send_message(message)
