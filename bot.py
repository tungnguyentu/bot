import time
import schedule
import traceback
from datetime import datetime
import logging

import config
from binance_client import BinanceClient
from strategies.strategy_selector import StrategySelector
from telegram_notifications import TelegramNotifier
from risk_management import RiskManager
from utils.logger import setup_logger

class TradingBot:
    def __init__(self):
        self.logger = setup_logger('trading_bot', 'logs/trading_bot.log')
        self.logger.info("Initializing Trading Bot...")
        
        self.client = BinanceClient()
        self.telegram = TelegramNotifier(config.TELEGRAM_TOKEN, config.TELEGRAM_CHAT_ID)
        self.risk_manager = RiskManager(self.client)
        self.strategy_selector = StrategySelector(self.client)
        
        self.running = False
        self.current_drawdown = 0.0
        self.initial_balance = 0.0
        
        self.logger.info("Trading Bot initialized successfully")
        self.telegram.send_message("🤖 Binance Futures Trading Bot initialized and ready to trade!")

    def start(self):
        """Start the trading bot."""
        self.running = True
        self.initial_balance = self.client.get_balance()
        
        self.logger.info(f"Starting bot with initial balance: {self.initial_balance} USDT")
        self.telegram.send_message(f"🚀 Bot started with initial balance: {self.initial_balance} USDT")
        
        # Schedule regular tasks
        schedule.every(1).minutes.do(self.check_market_conditions)
        schedule.every(1).hours.do(self.report_status)
        schedule.every(6).hours.do(self.update_drawdown)
        
        try:
            while self.running:
                schedule.run_pending()
                time.sleep(1)
        except KeyboardInterrupt:
            self.logger.info("Bot stopped by user")
            self.telegram.send_message("⚠️ Bot stopped manually")
        except Exception as e:
            error_msg = f"Critical error: {str(e)}\n{traceback.format_exc()}"
            self.logger.error(error_msg)
            self.telegram.send_message(f"🚨 Critical error: {str(e)}")
            self.restart()

    def check_market_conditions(self):
        """Check market conditions and execute trading strategies."""
        if not self.running:
            return
            
        try:
            # Check if max drawdown reached
            if self.current_drawdown >= config.MAX_DRAWDOWN:
                self.logger.warning(f"Max drawdown reached: {self.current_drawdown:.2%}")
                self.telegram.send_message(f"⚠️ Max drawdown limit reached ({self.current_drawdown:.2%}). Trading halted.")
                self.running = False
                return
                
            # Check if we can open new positions based on risk management
            if not self.risk_manager.can_open_new_position():
                self.logger.info("Risk limits reached, skipping strategy execution")
                return
                
            # Process each trading symbol
            for symbol in config.SYMBOLS:
                # Determine best strategy for current market conditions
                best_strategy = self.strategy_selector.select_strategy(symbol)
                
                # Execute the selected strategy
                if best_strategy == 'scalping':
                    self.execute_scalping(symbol)
                elif best_strategy == 'swing':
                    self.execute_swing(symbol)
                    
        except Exception as e:
            self.logger.error(f"Error in market check: {str(e)}")
            self.telegram.send_message(f"⚠️ Error during market check: {str(e)}")

    def execute_scalping(self, symbol):
        """Execute the scalping strategy for a symbol."""
        from strategies.scalping import ScalpingStrategy
        
        strategy = ScalpingStrategy(self.client)
        signal = strategy.generate_signal(symbol)
        
        if signal:
            reasoning = signal['reasoning']
            self.logger.info(f"Scalping signal for {symbol}: {signal['action']} - {reasoning}")
            
            # Execute the trade if signal is valid
            if signal['action'] in ['BUY', 'SELL']:
                risk_amount = self.risk_manager.calculate_position_size(symbol, signal['stop_loss'])
                
                order_result = self.client.place_order(
                    symbol=symbol,
                    side=signal['action'],
                    quantity=risk_amount,
                    price=signal.get('price', None),  # Use limit price if provided, otherwise market
                    stop_loss=signal['stop_loss'],
                    take_profit=signal['take_profit']
                )
                
                # Notify about the trade
                if order_result:
                    self.telegram.send_trade_notification(
                        strategy_type="Scalping",
                        symbol=symbol,
                        action=signal['action'],
                        entry=order_result['price'],
                        stop_loss=signal['stop_loss'],
                        take_profit=signal['take_profit'],
                        reasoning=reasoning
                    )

    def execute_swing(self, symbol):
        """Execute the swing trading strategy for a symbol."""
        from strategies.swing import SwingStrategy
        
        strategy = SwingStrategy(self.client)
        signal = strategy.generate_signal(symbol)
        
        if signal:
            reasoning = signal['reasoning']
            self.logger.info(f"Swing signal for {symbol}: {signal['action']} - {reasoning}")
            
            # Execute the trade if signal is valid
            if signal['action'] in ['BUY', 'SELL']:
                risk_amount = self.risk_manager.calculate_position_size(symbol, signal['stop_loss'])
                
                order_result = self.client.place_order(
                    symbol=symbol,
                    side=signal['action'],
                    quantity=risk_amount,
                    price=signal.get('price', None),
                    stop_loss=signal['stop_loss'],
                    take_profit=signal['take_profit']
                )
                
                # Notify about the trade
                if order_result:
                    self.telegram.send_trade_notification(
                        strategy_type="Swing Trading",
                        symbol=symbol,
                        action=signal['action'],
                        entry=order_result['price'],
                        stop_loss=signal['stop_loss'],
                        take_profit=signal['take_profit'],
                        reasoning=reasoning
                    )

    def update_drawdown(self):
        """Update the current drawdown value."""
        current_balance = self.client.get_balance()
        self.current_drawdown = max(0, (self.initial_balance - current_balance) / self.initial_balance)
        self.logger.info(f"Current drawdown: {self.current_drawdown:.2%}")

    def report_status(self):
        """Send a status report."""
        current_balance = self.client.get_balance()
        pnl = current_balance - self.initial_balance
        pnl_percent = (pnl / self.initial_balance) * 100 if self.initial_balance > 0 else 0
        
        open_positions = self.client.get_open_positions()
        
        status_msg = (
            f"📊 *Status Report*\n"
            f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Balance: {current_balance:.2f} USDT\n"
            f"PnL: {pnl:.2f} USDT ({pnl_percent:.2f}%)\n"
            f"Drawdown: {self.current_drawdown:.2%}\n"
            f"Open Positions: {len(open_positions)}\n"
        )
        
        for pos in open_positions:
            status_msg += f"- {pos['symbol']}: {pos['size']} ({pos['pnl']:.2f} USDT)\n"
        
        self.telegram.send_message(status_msg)

    def restart(self):
        """Attempt to restart the bot after a critical error."""
        self.logger.info("Attempting to restart the bot...")
        self.telegram.send_message("🔄 Attempting to restart the bot...")
        
        # Wait a bit before restarting
        time.sleep(60)
        
        try:
            self.running = False
            time.sleep(5)
            self.start()
        except Exception as e:
            self.logger.error(f"Failed to restart: {str(e)}")
            self.telegram.send_message("❌ Failed to restart the bot. Manual intervention required.")

if __name__ == "__main__":
    bot = TradingBot()
    bot.start()
