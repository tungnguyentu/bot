import requests
import logging
import time
import re

logger = logging.getLogger(__name__)

class TelegramNotifier:
    def __init__(self, token, chat_id):
        self.token = token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{token}/"
        self.enabled = token and chat_id
        
        if self.enabled:
            logger.info("Telegram notifications enabled")
        else:
            logger.warning("Telegram notifications disabled: missing token or chat_id")
    
    def send_message(self, message, parse_mode='Markdown', retry=3):
        """Send a message to the configured Telegram chat"""
        if not self.enabled:
            logger.info(f"Telegram notification would send: {message}")
            return False
        
        # Clean up the message to avoid Markdown parsing errors
        if parse_mode == 'Markdown':
            # Escape Markdown special characters
            message = self._escape_markdown(message)
        
        url = self.base_url + "sendMessage"
        data = {
            "chat_id": self.chat_id,
            "text": message,
            "parse_mode": parse_mode
        }
        
        for attempt in range(retry):
            try:
                response = requests.post(url, data=data)
                
                if response.status_code == 200:
                    logger.debug(f"Telegram message sent: {message[:50]}...")
                    return True
                else:
                    logger.warning(f"Failed to send Telegram message: {response.status_code} {response.text}")
                    # If it's a parsing error, try without parse_mode
                    if 'parse entities' in response.text and attempt == 0:
                        logger.info("Retrying without parse_mode due to parsing error")
                        data["parse_mode"] = ""
                    else:
                        time.sleep(1)
            except Exception as e:
                logger.error(f"Error sending Telegram message: {e}")
                time.sleep(1)
        
        return False
    
    def _escape_markdown(self, text):
        """Escape Markdown special characters to prevent parsing errors"""
        # Characters to escape: _*[]()~`>#+-=|{}.!
        special_chars = ['_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!']
        for char in special_chars:
            text = text.replace(char, '\\' + char)
        return text

    def send_photo(self, photo_path, caption=None, parse_mode='Markdown', retry=3):
        """Send a photo with optional caption to the configured Telegram chat"""
        if not self.enabled:
            logger.info(f"Telegram photo would send: {photo_path} with caption {caption}")
            return False
            
        url = self.base_url + "sendPhoto"
        
        for attempt in range(retry):
            try:
                with open(photo_path, 'rb') as photo:
                    data = {
                        "chat_id": self.chat_id,
                        "parse_mode": parse_mode
                    }
                    
                    if caption:
                        data["caption"] = caption
                    
                    files = {"photo": photo}
                    response = requests.post(url, data=data, files=files)
                    
                    if response.status_code == 200:
                        logger.debug(f"Telegram photo sent: {photo_path}")
                        return True
                    else:
                        logger.warning(f"Failed to send Telegram photo: {response.status_code} {response.text}")
                        time.sleep(1)
            except Exception as e:
                logger.error(f"Error sending Telegram photo: {e}")
                time.sleep(1)
                
        return False
    
    def send_trade_notification(self, trade_type, symbol, side, quantity, price, pnl=None):
        """Send a formatted trade notification"""
        if trade_type == "open":
            emoji = "🔴" if side == "SHORT" else "🟢"
            message = (
                f"{emoji} *NEW {side} POSITION*\n"
                f"Symbol: {symbol}\n"
                f"Quantity: {quantity}\n"
                f"Price: {price}\n"
                f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}"
            )
        else:  # close
            emoji = "✅" if pnl and pnl > 0 else "❌"
            pnl_text = f"PnL: {pnl:.2f} USDT" if pnl is not None else ""
            message = (
                f"{emoji} *POSITION CLOSED*\n"
                f"Symbol: {symbol}\n"
                f"Side: {side}\n"
                f"Quantity: {quantity}\n"
                f"Price: {price}\n"
                f"{pnl_text}\n"
                f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}"
            )
            
        return self.send_message(message)
    
    def send_error_notification(self, error_message):
        """Send error notification"""
        message = (
            f"⚠️ *ERROR*\n"
            f"{error_message}\n"
            f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}"
        )
        return self.send_message(message)
        
    def send_daily_summary(self, date, pnl, trades, win_rate, balance):
        """Send daily trading summary"""
        return_pct = (pnl / (balance - pnl) * 100) if balance != pnl else 0
        message = (
            f"📊 *Daily Trading Summary*\n"
            f"Date: {date}\n"
            f"PnL: {pnl:.2f} USDT\n"
            f"Trades: {trades}\n"
            f"Win Rate: {win_rate:.1f}%\n"
            f"Current Balance: {balance:.2f} USDT\n"
            f"Daily Return: {return_pct:.2f}%"
        )
        return self.send_message(message)
