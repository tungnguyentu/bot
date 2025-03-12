"""
AI-powered trading strategies for the AI Trading Bot.
"""

import os
import logging
import numpy as np
import pandas as pd
from datetime import datetime
import time
import joblib

import config
from strategy import Strategy
from indicators import (
    calculate_rsi,
    calculate_vwap,
    calculate_atr,
    detect_volume_spike,
    calculate_bollinger_bands,
    calculate_macd,
)
from data_processor import DataProcessor
from ai_models import (
    LSTMModel,
    XGBoostModel,
    RandomForestModel,
    RLModel,
    EnsembleModel,
    TradingEnvironment,
)
from utils import (
    calculate_take_profit_price,
    calculate_stop_loss_price,
    calculate_atr_stop_loss,
    save_trade_history,
)

# Initialize logger
logger = logging.getLogger("trading_bot")


class AIStrategy(Strategy):
    """
    Base AI strategy class for the trading bot.
    """
    
    def __init__(self, binance_client, telegram_notifier=None, symbol=None, timeframe=None, leverage=None):
        """
        Initialize the AI strategy.
        
        Args:
            binance_client (BinanceClient): Binance client
            telegram_notifier (TelegramNotifier, optional): Telegram notifier
            symbol (str): Trading symbol
            timeframe (str): Trading timeframe
            leverage (int): Trading leverage
        """
        super().__init__(binance_client, telegram_notifier, symbol, timeframe, leverage)
        
        # Initialize data processor
        self.data_processor = DataProcessor(symbol=self.symbol, timeframe=self.timeframe)
        
        # Initialize models
        self.lstm_model = LSTMModel(symbol=self.symbol, timeframe=self.timeframe)
        self.xgboost_model = XGBoostModel(symbol=self.symbol, timeframe=self.timeframe)
        self.random_forest_model = RandomForestModel(symbol=self.symbol, timeframe=self.timeframe)
        self.rl_model = RLModel(symbol=self.symbol, timeframe=self.timeframe)
        
        # Initialize ensemble model
        self.ensemble_model = EnsembleModel(symbol=self.symbol, timeframe=self.timeframe)
        
        # Load models if available
        self._load_models()
        
        # Strategy parameters
        self.sequence_length = 60  # Number of candles to use for prediction
        self.confidence_threshold = 0.6  # Minimum confidence for trade signals
        self.use_ensemble = True  # Whether to use ensemble model
        
        # Model weights for ensemble
        self.model_weights = {
            'lstm': 0.3,
            'xgboost': 0.3,
            'random_forest': 0.2,
            'rl': 0.2,
        }
        
        # Add models to ensemble
        self._setup_ensemble()
        
        logger.info(f"AI strategy initialized for {self.symbol} ({self.timeframe}).")

    def _load_models(self):
        """
        Load AI models if available.
        """
        try:
            # Load LSTM model
            self.lstm_model.load_model()
            
            # Load XGBoost model
            self.xgboost_model.load_model()
            
            # Load Random Forest model
            self.random_forest_model.load_model()
            
            # Load RL model
            self.rl_model.load_model()
            
            logger.info(f"AI models loaded for {self.symbol} ({self.timeframe}).")
        
        except Exception as e:
            logger.warning(f"Error loading AI models: {e}")
            logger.warning("AI models will need to be trained before use.")

    def _setup_ensemble(self):
        """
        Set up ensemble model with individual models and weights.
        """
        try:
            # Add models to ensemble
            if self.lstm_model.model is not None:
                self.ensemble_model.add_model('lstm', self.lstm_model, weight=self.model_weights['lstm'])
            
            if self.xgboost_model.model is not None:
                self.ensemble_model.add_model('xgboost', self.xgboost_model, weight=self.model_weights['xgboost'])
            
            if self.random_forest_model.model is not None:
                self.ensemble_model.add_model('random_forest', self.random_forest_model, weight=self.model_weights['random_forest'])
            
            if self.rl_model.model is not None:
                self.ensemble_model.add_model('rl', self.rl_model, weight=self.model_weights['rl'])
            
            logger.info(f"Ensemble model set up with {len(self.ensemble_model.models)} models.")
        
        except Exception as e:
            logger.error(f"Error setting up ensemble model: {e}")
            self.use_ensemble = False
            logger.warning("Ensemble model will not be used.")

    def train_models(self, start_date=None, end_date=None):
        """
        Train AI models on historical data.
        
        Args:
            start_date (str): Start date (YYYY-MM-DD)
            end_date (str): End date (YYYY-MM-DD)
            
        Returns:
            bool: True if training was successful, False otherwise
        """
        try:
            # Fetch historical data
            df = self.data_processor.fetch_historical_data(start_date=start_date, end_date=end_date)
            
            # Add technical indicators
            df = self.data_processor.add_technical_indicators(df)
            
            # Add sentiment data
            df = self.data_processor.add_sentiment_data(df)
            
            # Prepare data for training
            X_train, y_train, X_val, y_val, X_test, y_test, feature_names = self.data_processor.prepare_data_for_training(
                df, target_column='return_1', sequence_length=self.sequence_length
            )
            
            # Train LSTM model
            self.lstm_model.build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
            self.lstm_model.train(X_train, y_train, X_val, y_val, epochs=50, batch_size=32)
            
            # Prepare data for classification models (convert to binary classification)
            y_train_class = (y_train > 0).astype(int)
            y_val_class = (y_val > 0).astype(int)
            
            # Reshape data for classification models
            X_train_flat = X_train.reshape(X_train.shape[0], -1)
            X_val_flat = X_val.reshape(X_val.shape[0], -1)
            
            # Train XGBoost model
            self.xgboost_model.build_model()
            self.xgboost_model.train(X_train_flat, y_train_class, X_val_flat, y_val_class)
            
            # Train Random Forest model
            self.random_forest_model.build_model()
            self.random_forest_model.train(X_train_flat, y_train_class)
            
            # Train RL model
            env = TradingEnvironment(df, initial_balance=10000, transaction_fee=0.001, window_size=self.sequence_length)
            self.rl_model.build_model(env)
            self.rl_model.train(env, total_timesteps=10000)
            
            # Set up ensemble model
            self._setup_ensemble()
            
            logger.info(f"AI models trained successfully for {self.symbol} ({self.timeframe}).")
            
            return True
        
        except Exception as e:
            logger.error(f"Error training AI models: {e}")
            return False

    def analyze_market(self):
        """
        Analyze market data using AI models.
        
        Returns:
            dict: Market analysis results
        """
        try:
            # Get market data
            klines = self.binance_client.get_klines(
                symbol=self.symbol,
                timeframe=self.timeframe,
                limit=self.sequence_length + 50  # Extra candles for indicators
            )
            
            # Add technical indicators
            klines = self.data_processor.add_technical_indicators(klines)
            
            # Remove columns that weren't present during training
            columns_to_remove = [
                'number_of_trades', 
                'quote_asset_volume', 
                'taker_buy_base_asset_volume', 
                'taker_buy_quote_asset_volume',
                'close_time',
                'ignore'
            ]
            
            for col in columns_to_remove:
                if col in klines.columns:
                    klines = klines.drop(columns=[col])
            
            # Add sentiment features that were present during training but might be missing now
            if 'news_sentiment' not in klines.columns:
                klines['news_sentiment'] = 0.0
            
            if 'social_sentiment' not in klines.columns:
                klines['social_sentiment'] = 0.0
                
            if 'google_trends' not in klines.columns:
                klines['google_trends'] = 0.0
            
            # Prepare data for prediction
            X = self.data_processor.prepare_data_for_prediction(klines, sequence_length=self.sequence_length)
            
            # Get latest data
            latest = klines.iloc[-1]
            
            # Make predictions
            if self.use_ensemble and len(self.ensemble_model.models) > 0:
                # Use ensemble model
                signal_proba = self.ensemble_model.predict_proba(X)[0]
                
                # Generate signals based on probability and confidence threshold
                long_signal = signal_proba > self.confidence_threshold
                short_signal = signal_proba < (1 - self.confidence_threshold)
                
                # Calculate confidence scores
                long_confidence = signal_proba
                short_confidence = 1 - signal_proba
            else:
                # Use individual models
                predictions = {}
                
                # LSTM prediction
                if self.lstm_model.model is not None:
                    lstm_pred = self.lstm_model.predict(X)[0][0]
                    predictions['lstm'] = 1 if lstm_pred > 0 else 0
                    lstm_confidence = abs(lstm_pred)
                else:
                    predictions['lstm'] = 0
                    lstm_confidence = 0
                
                # XGBoost prediction
                if self.xgboost_model.model is not None:
                    X_flat = X.reshape(X.shape[0], -1)
                    xgb_proba = self.xgboost_model.predict_proba(X_flat)[0][1]
                    predictions['xgboost'] = 1 if xgb_proba > 0.5 else 0
                    xgb_confidence = xgb_proba if xgb_proba > 0.5 else 1 - xgb_proba
                else:
                    predictions['xgboost'] = 0
                    xgb_confidence = 0
                
                # Random Forest prediction
                if self.random_forest_model.model is not None:
                    X_flat = X.reshape(X.shape[0], -1)
                    rf_proba = self.random_forest_model.predict_proba(X_flat)[0][1]
                    predictions['random_forest'] = 1 if rf_proba > 0.5 else 0
                    rf_confidence = rf_proba if rf_proba > 0.5 else 1 - rf_proba
                else:
                    predictions['random_forest'] = 0
                    rf_confidence = 0
                
                # RL prediction
                if self.rl_model.model is not None:
                    rl_action = self.rl_model.predict(X[0])
                    predictions['rl'] = 1 if rl_action == 1 else 0
                    rl_confidence = 0.8  # Fixed confidence for RL
                else:
                    predictions['rl'] = 0
                    rl_confidence = 0
                
                # Calculate weighted average prediction
                weighted_pred = 0
                total_weight = 0
                
                for model_name, pred in predictions.items():
                    if model_name in self.model_weights:
                        weighted_pred += pred * self.model_weights[model_name]
                        total_weight += self.model_weights[model_name]
                
                if total_weight > 0:
                    weighted_pred /= total_weight
                
                # Generate signals based on weighted prediction and confidence threshold
                long_signal = weighted_pred > self.confidence_threshold
                short_signal = weighted_pred < (1 - self.confidence_threshold)
                
                # Calculate confidence scores
                long_confidence = weighted_pred
                short_confidence = 1 - weighted_pred
            
            # Adjust position size based on confidence
            position_size_multiplier = max(long_confidence, short_confidence)
            adjusted_position_size = self.position_size * position_size_multiplier
            
            # Adjust take profit and stop loss based on volatility
            atr_value = latest['atr']
            volatility_multiplier = min(1.5, max(0.5, atr_value / klines['atr'].mean()))
            
            adjusted_take_profit = self.take_profit_percent * volatility_multiplier
            adjusted_stop_loss = self.stop_loss_percent * volatility_multiplier
            
            # Log analysis results
            logger.info(f"AI Analysis - Price: {latest['close']:.2f}, Long Signal: {long_signal}, Short Signal: {short_signal}")
            logger.info(f"AI Confidence - Long: {long_confidence:.2f}, Short: {short_confidence:.2f}")
            logger.info(f"AI Adjustments - Position Size: {adjusted_position_size:.2f}, Take Profit: {adjusted_take_profit:.2f}%, Stop Loss: {adjusted_stop_loss:.2f}%")
            
            # Return analysis results
            result = {
                'timestamp': datetime.now(),
                'symbol': self.symbol,
                'price': latest['close'],
                'long_signal': long_signal,
                'short_signal': short_signal,
                'long_confidence': long_confidence,
                'short_confidence': short_confidence,
                'adjusted_position_size': adjusted_position_size,
                'adjusted_take_profit': adjusted_take_profit,
                'adjusted_stop_loss': adjusted_stop_loss,
                'atr': latest['atr'],
                'volatility_multiplier': volatility_multiplier
            }
            
            return result
        
        except Exception as e:
            logger.error(f"Error analyzing market with AI: {e}")
            if self.telegram_notifier:
                self.telegram_notifier.notify_error(f"Error analyzing market with AI: {str(e).replace('*', '')}")
            raise

    def open_long_position(self, analysis):
        """
        Open a long position with AI-optimized parameters.
        
        Args:
            analysis (dict): Market analysis results
            
        Returns:
            dict: Position data
        """
        try:
            # Get current price
            current_price = analysis['price']
            
            # Use AI-adjusted position size
            position_size = analysis.get('adjusted_position_size', self.position_size)
            
            # Use AI-adjusted take profit and stop loss
            take_profit_percent = analysis.get('adjusted_take_profit', self.take_profit_percent)
            stop_loss_percent = analysis.get('adjusted_stop_loss', self.stop_loss_percent)
            
            # Calculate take profit and stop loss prices
            take_profit_price = calculate_take_profit_price(current_price, take_profit_percent, 'long')
            
            if self.use_atr_for_sl:
                stop_loss_price = calculate_atr_stop_loss(
                    current_price, analysis['atr'], self.atr_multiplier, 'long'
                )
            else:
                stop_loss_price = calculate_stop_loss_price(current_price, stop_loss_percent, 'long')
            
            # Create market order
            order = self.binance_client.create_market_order(
                symbol=self.symbol,
                side='buy',
                amount=position_size
            )
            
            # Generate position ID
            position_id = f"long_{self.symbol}_{int(time.time())}"
            
            # Store position data
            position_data = {
                'id': position_id,
                'symbol': self.symbol,
                'type': 'long',
                'entry_price': current_price,
                'take_profit': take_profit_price,
                'stop_loss': stop_loss_price,
                'quantity': position_size,
                'entry_time': datetime.now(),
                'confidence': analysis.get('long_confidence', 0.0),
                'order_id': order['orderId'] if 'orderId' in order else None
            }
            
            self.active_positions[position_id] = position_data
            
            # Send notification
            if self.telegram_notifier:
                self.telegram_notifier.notify_trade_open(
                    self.symbol,
                    'long',
                    current_price,
                    position_size,
                    stop_loss_price,
                    take_profit_price
                )
            
            logger.info(f"Opened long position for {self.symbol} at {current_price} with confidence {analysis.get('long_confidence', 0.0):.2f}")
            
            return {
                'action': 'open_long',
                'position_id': position_id,
                'entry_price': current_price,
                'take_profit': take_profit_price,
                'stop_loss': stop_loss_price,
                'confidence': analysis.get('long_confidence', 0.0)
            }
        
        except Exception as e:
            logger.error(f"Error opening long position: {e}")
            if self.telegram_notifier:
                self.telegram_notifier.notify_error(f"Error opening long position: {e}")
            return {'action': 'error', 'reason': str(e)}

    def open_short_position(self, analysis):
        """
        Open a short position with AI-optimized parameters.
        
        Args:
            analysis (dict): Market analysis results
            
        Returns:
            dict: Position data
        """
        try:
            # Get current price
            current_price = analysis['price']
            
            # Use AI-adjusted position size
            position_size = analysis.get('adjusted_position_size', self.position_size)
            
            # Use AI-adjusted take profit and stop loss
            take_profit_percent = analysis.get('adjusted_take_profit', self.take_profit_percent)
            stop_loss_percent = analysis.get('adjusted_stop_loss', self.stop_loss_percent)
            
            # Calculate take profit and stop loss prices
            take_profit_price = calculate_take_profit_price(current_price, take_profit_percent, 'short')
            
            if self.use_atr_for_sl:
                stop_loss_price = calculate_atr_stop_loss(
                    current_price, analysis['atr'], self.atr_multiplier, 'short'
                )
            else:
                stop_loss_price = calculate_stop_loss_price(current_price, stop_loss_percent, 'short')
            
            # Create market order
            order = self.binance_client.create_market_order(
                symbol=self.symbol,
                side='sell',
                amount=position_size
            )
            
            # Generate position ID
            position_id = f"short_{self.symbol}_{int(time.time())}"
            
            # Store position data
            position_data = {
                'id': position_id,
                'symbol': self.symbol,
                'type': 'short',
                'entry_price': current_price,
                'take_profit': take_profit_price,
                'stop_loss': stop_loss_price,
                'quantity': position_size,
                'entry_time': datetime.now(),
                'confidence': analysis.get('short_confidence', 0.0),
                'order_id': order['orderId'] if 'orderId' in order else None
            }
            
            self.active_positions[position_id] = position_data
            
            # Send notification
            if self.telegram_notifier:
                self.telegram_notifier.notify_trade_open(
                    self.symbol,
                    'short',
                    current_price,
                    position_size,
                    stop_loss_price,
                    take_profit_price
                )
            
            logger.info(f"Opened short position for {self.symbol} at {current_price} with confidence {analysis.get('short_confidence', 0.0):.2f}")
            
            return {
                'action': 'open_short',
                'position_id': position_id,
                'entry_price': current_price,
                'take_profit': take_profit_price,
                'stop_loss': stop_loss_price,
                'confidence': analysis.get('short_confidence', 0.0)
            }
        
        except Exception as e:
            logger.error(f"Error opening short position: {e}")
            if self.telegram_notifier:
                self.telegram_notifier.notify_error(f"Error opening short position: {e}")
            return {'action': 'error', 'reason': str(e)}

    def manage_positions(self):
        """
        Manage open positions with AI-optimized trailing stops.
        
        Returns:
            dict: Position management results
        """
        try:
            # Get current market data
            klines = self.binance_client.get_klines(
                symbol=self.symbol,
                timeframe=self.timeframe,
                limit=2
            )
            
            # Get current price
            current_price = klines.iloc[-1]['close']
            
            # Initialize results
            results = {
                'positions_updated': 0,
                'positions_closed': 0,
                'trailing_stops_adjusted': 0
            }
            
            # Manage each position
            for position_id in list(self.active_positions.keys()):
                position = self.active_positions[position_id]
                
                # Check if position should be closed
                if position['type'] == 'long':
                    # Check take profit
                    if current_price >= position['take_profit']:
                        self.close_position(position_id, 'take_profit', current_price)
                        results['positions_closed'] += 1
                        continue
                    
                    # Check stop loss
                    if current_price <= position['stop_loss']:
                        self.close_position(position_id, 'stop_loss', current_price)
                        results['positions_closed'] += 1
                        continue
                    
                    # Check trailing stop
                    if self.use_trailing_stop:
                        # Calculate price movement
                        price_movement = (current_price - position['entry_price']) / position['entry_price']
                        
                        # Check if trailing stop should be activated
                        if price_movement >= self.trailing_stop_activation:
                            # Calculate new stop loss
                            new_stop_loss = max(
                                position['stop_loss'],
                                current_price * (1 - self.trailing_stop_callback)
                            )
                            
                            # Update stop loss if it has changed
                            if new_stop_loss > position['stop_loss']:
                                position['stop_loss'] = new_stop_loss
                                results['trailing_stops_adjusted'] += 1
                                logger.info(f"Adjusted trailing stop for {position_id} to {new_stop_loss:.2f}")
                
                elif position['type'] == 'short':
                    # Check take profit
                    if current_price <= position['take_profit']:
                        self.close_position(position_id, 'take_profit', current_price)
                        results['positions_closed'] += 1
                        continue
                    
                    # Check stop loss
                    if current_price >= position['stop_loss']:
                        self.close_position(position_id, 'stop_loss', current_price)
                        results['positions_closed'] += 1
                        continue
                    
                    # Check trailing stop
                    if self.use_trailing_stop:
                        # Calculate price movement
                        price_movement = (position['entry_price'] - current_price) / position['entry_price']
                        
                        # Check if trailing stop should be activated
                        if price_movement >= self.trailing_stop_activation:
                            # Calculate new stop loss
                            new_stop_loss = min(
                                position['stop_loss'],
                                current_price * (1 + self.trailing_stop_callback)
                            )
                            
                            # Update stop loss if it has changed
                            if new_stop_loss < position['stop_loss']:
                                position['stop_loss'] = new_stop_loss
                                results['trailing_stops_adjusted'] += 1
                                logger.info(f"Adjusted trailing stop for {position_id} to {new_stop_loss:.2f}")
                
                results['positions_updated'] += 1
            
            return results
        
        except Exception as e:
            logger.error(f"Error managing positions: {e}")
            if self.telegram_notifier:
                self.telegram_notifier.notify_error(f"Error managing positions: {e}")
            return {'action': 'error', 'reason': str(e)}


class AIScalpingStrategy(AIStrategy):
    """
    AI-powered scalping strategy for short-term trades.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Strategy-specific parameters
        self.timeframe = config.SCALPING_TIMEFRAME
        self.take_profit_percent = config.SCALPING_TAKE_PROFIT / 100
        self.stop_loss_percent = config.SCALPING_STOP_LOSS / 100
        
        # AI parameters
        self.sequence_length = 30  # Shorter sequence for scalping
        self.confidence_threshold = 0.65  # Higher confidence for scalping
        
        # Model weights for ensemble
        self.model_weights = {
            'lstm': 0.2,
            'xgboost': 0.4,
            'random_forest': 0.3,
            'rl': 0.1,
        }
        
        # Set up ensemble
        self._setup_ensemble()
        
        logger.info(f"AI Scalping strategy initialized for {self.symbol} ({self.timeframe}).")


class AISwingStrategy(AIStrategy):
    """
    AI-powered swing strategy for medium-term trades.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Strategy-specific parameters
        self.timeframe = config.SWING_TIMEFRAME
        self.take_profit_percent = config.SWING_TAKE_PROFIT / 100
        self.stop_loss_percent = config.SWING_STOP_LOSS / 100
        
        # AI parameters
        self.sequence_length = 60  # Longer sequence for swing trading
        self.confidence_threshold = 0.6  # Moderate confidence for swing trading
        
        # Model weights for ensemble
        self.model_weights = {
            'lstm': 0.3,
            'xgboost': 0.3,
            'random_forest': 0.2,
            'rl': 0.2,
        }
        
        # Set up ensemble
        self._setup_ensemble()
        
        logger.info(f"AI Swing strategy initialized for {self.symbol} ({self.timeframe}).")


class AIBreakoutStrategy(AIStrategy):
    """
    AI-powered breakout strategy for trend-following trades.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Strategy-specific parameters
        self.timeframe = config.BREAKOUT_TIMEFRAME
        self.take_profit_percent = config.BREAKOUT_TAKE_PROFIT / 100
        self.stop_loss_percent = config.BREAKOUT_STOP_LOSS / 100
        
        # AI parameters
        self.sequence_length = 45  # Medium sequence for breakout detection
        self.confidence_threshold = 0.7  # Higher confidence for breakout trading
        
        # Model weights for ensemble
        self.model_weights = {
            'lstm': 0.25,
            'xgboost': 0.25,
            'random_forest': 0.25,
            'rl': 0.25,
        }
        
        # Set up ensemble
        self._setup_ensemble()
        
        logger.info(f"AI Breakout strategy initialized for {self.symbol} ({self.timeframe}).") 