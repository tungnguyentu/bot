import logging
import time
import random
from .config import BotConfig
from .data_collector import BinanceDataCollector
from .indicators import TechnicalIndicators
from .model import XGBoostModel
from .trader import Trader
from .risk_manager import RiskManager
from .notifier import TelegramNotifier

logger = logging.getLogger(__name__)

class TradingBot:
    """Main trading bot class that orchestrates all components."""
    
    def __init__(self, config: BotConfig):
        """Initialize the trading bot with the given configuration."""
        self.config = config
        self.data_collector = BinanceDataCollector(config)
        self.indicators = TechnicalIndicators()
        self.model = XGBoostModel()
        self.risk_manager = RiskManager(
            sl_atr_multiplier=config.sl_atr_multiplier,
            tp_atr_multiplier=config.tp_atr_multiplier
        )
        self.notifier = TelegramNotifier()
        self.trader = Trader(config, self.notifier)
        
        # Initialize state
        self.is_running = False
        self.current_position = None
    
    def run(self):
        """Run the trading bot based on the configured mode."""
        self.is_running = True
        
        if self.config.train:
            self._train_model()
        
        if self.config.mode == 'backtest':
            self._run_backtest()
        elif self.config.mode == 'test' or self.config.mode == 'live':
            self._run_trading_loop()
    
    def _train_model(self):
        """Train the ML model with historical data."""
        logger.info("Training ML model...")
        # Get historical data
        historical_data = self.data_collector.get_historical_data()
        
        # Calculate indicators
        data_with_indicators = self.indicators.calculate_all(historical_data)
        
        # Train the model
        self.model.train(data_with_indicators)
        logger.info("Model training completed")
    
    def _run_backtest(self):
        """Run backtest with historical data."""
        logger.info("Starting backtest...")
        
        # Get historical data
        historical_data = self.data_collector.get_historical_data()
        
        # Calculate indicators
        data_with_indicators = self.indicators.calculate_all(historical_data)
        
        # Get predictions
        predictions = self.model.predict(data_with_indicators)
        
        # Run backtest simulation
        backtest_results = self.trader.run_backtest(
            data_with_indicators, 
            predictions, 
            self.risk_manager
        )
        
        # Log and save backtest results
        logger.info(f"Backtest completed with results: {backtest_results}")
        self.notifier.send_performance_report(backtest_results)
    
    def _run_trading_loop(self):
        """Run the main trading loop for test or live trading."""
        logger.info(f"Starting trading loop in {self.config.mode} mode...")
        self.notifier.send_message(f"🚀 Trading bot started in {self.config.mode} mode for {self.config.symbol} with investment of ${self.config.invest}")
        
        try:
            # If quick mode is enabled and we're in test mode, execute a trade immediately
            if self.config.quick and self.config.mode == 'test':
                self._execute_quick_test_trade()
                if not self.config.train:  # If not in training mode, exit after quick trade
                    logger.info("Quick test trade executed. Exiting...")
                    return
            
            while self.is_running:
                # Get latest market data
                market_data = self.data_collector.get_latest_data()
                
                # Calculate indicators
                data_with_indicators = self.indicators.calculate_all(market_data)
                
                # Get prediction and confidence
                prediction, confidence = self.model.predict_with_confidence(data_with_indicators)
                
                # Get current position (if any)
                self.current_position = self.trader.get_current_position()
                
                # Calculate risk parameters
                risk_params = self.risk_manager.calculate_risk_params(
                    data_with_indicators, 
                    self.current_position
                )
                
                # Execute trading logic
                if self.current_position is None:
                    # No position, check for entry
                    if prediction != 0 and confidence > 0.7:  # Threshold for entry
                        entry_price = market_data['close'].iloc[-1]
                        
                        # Determine position side
                        side = "BUY" if prediction > 0 else "SELL"
                        
                        # Calculate SL and TP levels
                        sl_price = risk_params['sl_price']
                        tp_price = risk_params['tp_price']
                        
                        # Calculate appropriate position size based on investment amount
                        position_size = self.trader.calculate_position_size(entry_price, sl_price)
                        
                        if position_size > 0:
                            # Execute trade
                            trade_result = self.trader.open_position(
                                side=side,
                                quantity=position_size,
                                entry_price=entry_price,
                                sl_price=sl_price,
                                tp_price=tp_price
                            )
                            
                            if trade_result:
                                trade_reason = self._generate_trade_reason(data_with_indicators)
                                self.notifier.send_trade_entry(
                                    symbol=self.config.symbol,
                                    side=side,
                                    entry_price=entry_price,
                                    sl_price=sl_price,
                                    tp_price=tp_price,
                                    confidence=confidence,
                                    reason=trade_reason
                                )
                else:
                    # Have position, check for exit conditions
                    exit_signal = self._check_exit_signal(
                        data_with_indicators, 
                        prediction, 
                        confidence,
                        self.current_position
                    )
                    
                    if exit_signal:
                        exit_price = market_data['close'].iloc[-1]
                        exit_result = self.trader.close_position(
                            position_id=self.current_position['id'],
                            exit_price=exit_price,
                            reason=exit_signal
                        )
                        
                        if exit_result:
                            self.notifier.send_trade_exit(
                                symbol=self.config.symbol,
                                exit_price=exit_price,
                                pnl=exit_result['pnl'],
                                reason=exit_signal
                            )
                
                # Sleep to avoid excessive API calls
                time.sleep(60)  # Check every minute
                
        except KeyboardInterrupt:
            logger.info("Trading bot stopped by user")
            self.notifier.send_message("⚠️ Trading bot stopped manually")
        except Exception as e:
            logger.exception("Error in trading loop")
            self.notifier.send_message(f"🚨 Error: {str(e)}")
        finally:
            self.is_running = False
    
    def _execute_quick_test_trade(self):
        """Execute a quick test trade immediately for testing purposes."""
        logger.info("Quick test mode enabled. Executing immediate test trade...")
        
        # Get latest market data
        market_data = self.data_collector.get_latest_data()
        
        # Calculate indicators
        data_with_indicators = self.indicators.calculate_all(market_data)
        
        # Get current position (if any)
        self.current_position = self.trader.get_current_position()
        
        # Only proceed if we don't have a position already
        if self.current_position is None:
            # Calculate risk parameters
            risk_params = self.risk_manager.calculate_risk_params(data_with_indicators)
            
            # Get the current price
            entry_price = market_data['close'].iloc[-1]
            
            # Randomly choose a side for the test trade
            side = random.choice(["BUY", "SELL"])
            
            # Calculate SL and TP levels
            sl_price = risk_params['sl_price']
            tp_price = risk_params['tp_price']
            
            if sl_price is None or tp_price is None:
                # Fallback if risk_manager couldn't calculate SL/TP
                atr = data_with_indicators['atr'].iloc[-1]
                if side == "BUY":
                    sl_price = entry_price - (atr * self.config.sl_atr_multiplier)
                    tp_price = entry_price + (atr * self.config.tp_atr_multiplier)
                else:  # SELL
                    sl_price = entry_price + (atr * self.config.sl_atr_multiplier)
                    tp_price = entry_price - (atr * self.config.tp_atr_multiplier)
            
            # Calculate appropriate position size based on investment amount
            position_size = self.trader.calculate_position_size(entry_price, sl_price)
            
            if position_size > 0:
                logger.info(f"Opening quick test {side} position for {self.config.symbol} at {entry_price}")
                
                # Execute trade
                trade_result = self.trader.open_position(
                    side=side,
                    quantity=position_size,
                    entry_price=entry_price,
                    sl_price=sl_price,
                    tp_price=tp_price
                )
                
                if trade_result:
                    self.notifier.send_trade_entry(
                        symbol=self.config.symbol,
                        side=side,
                        entry_price=entry_price,
                        sl_price=sl_price,
                        tp_price=tp_price,
                        confidence=0.9,  # High confidence for test trade
                        reason="Quick test trade (forced execution)"
                    )
                    
                    logger.info(f"Quick test trade executed successfully: {side} position at {entry_price}")
                    self.notifier.send_message(f"✅ Quick test trade executed successfully. Check positions in Binance testnet.")
                else:
                    logger.error("Failed to execute quick test trade")
                    self.notifier.send_message("❌ Failed to execute quick test trade. Check logs for details.")
            else:
                logger.error("Could not calculate valid position size for quick test trade")
                self.notifier.send_message("❌ Failed to execute quick test trade: Invalid position size")
        else:
            logger.info(f"Quick test mode: Position already exists for {self.config.symbol}, skipping test trade")
            self.notifier.send_message(f"ℹ️ Quick test mode: Position already exists for {self.config.symbol}")
    
    def _generate_trade_reason(self, data):
        """Generate reasoning for trade entry based on indicators."""
        last_row = data.iloc[-1]
        
        reasons = []
        
        # Check Bollinger Bands
        if last_row['close'] < last_row['bb_lower']:
            reasons.append("Price below lower Bollinger Band (oversold)")
        elif last_row['close'] > last_row['bb_upper']:
            reasons.append("Price above upper Bollinger Band (overbought)")
        
        # Check RSI
        if last_row['rsi'] < 30:
            reasons.append("RSI below 30 (oversold)")
        elif last_row['rsi'] > 70:
            reasons.append("RSI above 70 (overbought)")
        
        # Check MACD
        if last_row['macd'] > last_row['macd_signal'] and last_row['macd'] > 0:
            reasons.append("MACD above signal line (bullish)")
        elif last_row['macd'] < last_row['macd_signal'] and last_row['macd'] < 0:
            reasons.append("MACD below signal line (bearish)")
        
        # Check VWAP
        if last_row['close'] < last_row['vwap']:
            reasons.append("Price below VWAP (potential buy)")
        elif last_row['close'] > last_row['vwap']:
            reasons.append("Price above VWAP (potential sell)")
        
        # If no specific reasons, use a generic one
        if not reasons:
            reasons.append("AI model prediction")
        
        return ", ".join(reasons)
    
    def _check_exit_signal(self, data, prediction, confidence, position):
        """Check if we should exit the current position."""
        last_row = data.iloc[-1]
        
        # 1. Check if SL or TP hit (this would be handled by exchange in live trading)
        if position['side'] == 'BUY':
            if last_row['close'] <= position['sl_price']:
                return "Stop loss hit"
            if last_row['close'] >= position['tp_price']:
                return "Take profit hit"
        else:  # SELL position
            if last_row['close'] >= position['sl_price']:
                return "Stop loss hit"
            if last_row['close'] <= position['tp_price']:
                return "Take profit hit"
        
        # 2. Check for trend reversal
        if position['side'] == 'BUY' and prediction < 0 and confidence > 0.7:
            return "Trend reversal signal"
        if position['side'] == 'SELL' and prediction > 0 and confidence > 0.7:
            return "Trend reversal signal"
        
        # 3. No exit signal
        return None
