"""
AI models for the AI Trading Bot.
"""

import os
import logging
import numpy as np
import pandas as pd
import joblib
import time
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import xgboost as xgb
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import gym
from gym import spaces

import config
from data_processor import DataProcessor

# Initialize logger
logger = logging.getLogger("trading_bot")


class LSTMModel:
    """
    LSTM model for time series prediction.
    """

    def __init__(self, symbol=None, timeframe=None):
        """
        Initialize the LSTM model.
        
        Args:
            symbol (str): Trading symbol
            timeframe (str): Trading timeframe
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.model = None
        
        # Model parameters
        self.units = 64
        self.dropout = 0.2
        self.learning_rate = 0.001
        self.loss = 'mse'
        self.metrics = ['mae']
        
        logger.info(f"LSTM model initialized for {self.symbol} ({self.timeframe}).")

    def build_model(self, input_shape, output_shape=1):
        """
        Build the LSTM model.

        Args:
            input_shape (tuple): Input shape (sequence_length, n_features)
            output_shape (int): Output shape

        Returns:
            tensorflow.keras.models.Sequential: LSTM model
        """
        try:
            model = Sequential()
            
            # LSTM layers
            model.add(LSTM(units=self.units, return_sequences=True, input_shape=input_shape))
            model.add(Dropout(self.dropout))
            
            model.add(LSTM(units=self.units, return_sequences=True))
            model.add(Dropout(self.dropout))
            
            model.add(LSTM(units=self.units, return_sequences=False))
            model.add(Dropout(self.dropout))
            
            # Output layer
            model.add(Dense(units=output_shape))
            
            # Compile model
            model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss=self.loss, metrics=self.metrics)
            
            self.model = model
            
            logger.info(f"LSTM model built with input shape {input_shape} and output shape {output_shape}.")
            
            return model
        
        except Exception as e:
            logger.error(f"Error building LSTM model: {e}")
            raise

    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        """
        Train the LSTM model.

        Args:
            X_train (np.array): Training features
            y_train (np.array): Training target
            X_val (np.array): Validation features
            y_val (np.array): Validation target
            epochs (int): Number of epochs
            batch_size (int): Batch size

        Returns:
            tensorflow.keras.models.Sequential: Trained LSTM model
        """
        try:
            # Build model if not already built
            if self.model is None:
                self.build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
            
            # Callbacks
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            model_checkpoint = ModelCheckpoint(f"models/lstm_model_{self.symbol}_{self.timeframe}.h5", save_best_only=True)
            
            # Train model
            history = self.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_val, y_val),
                callbacks=[early_stopping, model_checkpoint],
                verbose=1
            )
            
            logger.info(f"LSTM model trained for {len(history.history['loss'])} epochs.")
            
            return self.model
        
        except Exception as e:
            logger.error(f"Error training LSTM model: {e}")
            raise

    def predict(self, X):
        """
        Make predictions with the LSTM model.

        Args:
            X (np.array): Features

        Returns:
            np.array: Predictions
        """
        try:
            # Load model if not already loaded
            if self.model is None:
                self.load_model()
            
            # Make predictions
            predictions = self.model.predict(X)
            
            return predictions
        
        except Exception as e:
            logger.error(f"Error making predictions with LSTM model: {e}")
            raise

    def save_model(self):
        """
        Save the model to disk.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if self.model is None:
                logger.warning("No model to save.")
                return False
            
            # Create models directory if it doesn't exist
            os.makedirs('models', exist_ok=True)
            
            # Save model in the newer Keras format
            model_path = f'models/lstm_model_{self.symbol}_{self.timeframe}.keras'
            self.model.save(model_path, save_format='keras')
            
            logger.info(f"LSTM model saved to {model_path}")
            return True
        
        except Exception as e:
            logger.error(f"Error saving LSTM model: {e}")
            return False

    def load_model(self):
        """
        Load the model from disk.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Check for model in the newer Keras format
            model_path_keras = f'models/lstm_model_{self.symbol}_{self.timeframe}.keras'
            
            # Check for model in the legacy H5 format
            model_path_h5 = f'models/lstm_model_{self.symbol}_{self.timeframe}.h5'
            
            # Try to load the model from either format
            if os.path.exists(model_path_keras):
                self.model = tf.keras.models.load_model(model_path_keras)
                logger.info(f"LSTM model loaded from {model_path_keras}")
                return True
            elif os.path.exists(model_path_h5):
                self.model = tf.keras.models.load_model(model_path_h5)
                logger.info(f"LSTM model loaded from {model_path_h5}")
                # Save in the newer format for future use
                self.save_model()
                return True
            else:
                logger.warning(f"No LSTM model found for {self.symbol} ({self.timeframe}).")
                return False
        
        except Exception as e:
            logger.error(f"Error loading LSTM model: {e}")
            return False


class XGBoostModel:
    """
    XGBoost model for classification.
    """

    def __init__(self, symbol=None, timeframe=None):
        """
        Initialize the XGBoost model.

        Args:
            symbol (str): Trading symbol
            timeframe (str): Timeframe
        """
        self.symbol = symbol or config.SYMBOL
        self.timeframe = timeframe or config.TIMEFRAME
        self.model = None
        self.model_path = f"models/xgboost_{self.symbol}_{self.timeframe}.pkl"
        
        logger.info(f"XGBoost model initialized for {self.symbol} ({self.timeframe}).")

    def build_model(self, params=None):
        """
        Build the XGBoost model.

        Args:
            params (dict): Model parameters

        Returns:
            xgboost.XGBClassifier: XGBoost model
        """
        try:
            # Default parameters
            if params is None:
                params = {
                    'max_depth': 6,
                    'learning_rate': 0.05,
                    'n_estimators': 100,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'objective': 'binary:logistic',
                    'eval_metric': 'logloss',
                    'random_state': 42
                }
            
            # Build model
            self.model = xgb.XGBClassifier(**params)
            
            logger.info(f"XGBoost model built with parameters: {params}")
            
            return self.model
        
        except Exception as e:
            logger.error(f"Error building XGBoost model: {e}")
            raise

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the XGBoost model.

        Args:
            X_train (np.array): Training features
            y_train (np.array): Training target
            X_val (np.array): Validation features
            y_val (np.array): Validation target

        Returns:
            xgboost.XGBClassifier: Trained XGBoost model
        """
        try:
            # Build model if not already built
            if self.model is None:
                self.build_model()
            
            # Prepare evaluation set
            eval_set = [(X_train, y_train)]
            if X_val is not None and y_val is not None:
                eval_set.append((X_val, y_val))
            
            # Train model
            self.model.fit(
                X_train, y_train,
                eval_set=eval_set,
                early_stopping_rounds=10,
                verbose=True
            )
            
            logger.info(f"XGBoost model trained.")
            
            # Save model
            self.save_model()
            
            return self.model
        
        except Exception as e:
            logger.error(f"Error training XGBoost model: {e}")
            raise

    def predict(self, X):
        """
        Make predictions with the XGBoost model.

        Args:
            X (np.array): Features

        Returns:
            np.array: Predictions
        """
        try:
            # Load model if not already loaded
            if self.model is None:
                self.load_model()
            
            # Make predictions
            predictions = self.model.predict(X)
            
            return predictions
        
        except Exception as e:
            logger.error(f"Error making predictions with XGBoost model: {e}")
            raise

    def predict_proba(self, X):
        """
        Make probability predictions with the XGBoost model.

        Args:
            X (np.array): Features

        Returns:
            np.array: Probability predictions
        """
        try:
            # Load model if not already loaded
            if self.model is None:
                self.load_model()
            
            # Make probability predictions
            probabilities = self.model.predict_proba(X)
            
            return probabilities
        
        except Exception as e:
            logger.error(f"Error making probability predictions with XGBoost model: {e}")
            raise

    def save_model(self):
        """
        Save the XGBoost model.
        """
        try:
            # Create models directory if it doesn't exist
            os.makedirs('models', exist_ok=True)
            
            # Save model
            joblib.dump(self.model, self.model_path)
            
            logger.info(f"XGBoost model saved to {self.model_path}.")
        
        except Exception as e:
            logger.error(f"Error saving XGBoost model: {e}")
            raise

    def load_model(self):
        """
        Load the XGBoost model.

        Returns:
            xgboost.XGBClassifier: Loaded XGBoost model
        """
        try:
            # Check if model file exists
            if not os.path.exists(self.model_path):
                logger.warning(f"XGBoost model file not found at {self.model_path}.")
                return None
            
            # Load model
            self.model = joblib.load(self.model_path)
            
            logger.info(f"XGBoost model loaded from {self.model_path}.")
            
            return self.model
        
        except Exception as e:
            logger.error(f"Error loading XGBoost model: {e}")
            raise


class RandomForestModel:
    """
    Random Forest model for classification.
    """

    def __init__(self, symbol=None, timeframe=None):
        """
        Initialize the Random Forest model.

        Args:
            symbol (str): Trading symbol
            timeframe (str): Timeframe
        """
        self.symbol = symbol or config.SYMBOL
        self.timeframe = timeframe or config.TIMEFRAME
        self.model = None
        self.model_path = f"models/random_forest_{self.symbol}_{self.timeframe}.pkl"
        
        logger.info(f"Random Forest model initialized for {self.symbol} ({self.timeframe}).")

    def build_model(self, params=None):
        """
        Build the Random Forest model.

        Args:
            params (dict): Model parameters

        Returns:
            sklearn.ensemble.RandomForestClassifier: Random Forest model
        """
        try:
            # Default parameters
            if params is None:
                params = {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'min_samples_split': 2,
                    'min_samples_leaf': 1,
                    'random_state': 42
                }
            
            # Build model
            self.model = RandomForestClassifier(**params)
            
            logger.info(f"Random Forest model built with parameters: {params}")
            
            return self.model
        
        except Exception as e:
            logger.error(f"Error building Random Forest model: {e}")
            raise

    def train(self, X_train, y_train):
        """
        Train the Random Forest model.

        Args:
            X_train (np.array): Training features
            y_train (np.array): Training target

        Returns:
            sklearn.ensemble.RandomForestClassifier: Trained Random Forest model
        """
        try:
            # Build model if not already built
            if self.model is None:
                self.build_model()
            
            # Train model
            self.model.fit(X_train, y_train)
            
            logger.info(f"Random Forest model trained.")
            
            # Save model
            self.save_model()
            
            return self.model
        
        except Exception as e:
            logger.error(f"Error training Random Forest model: {e}")
            raise

    def predict(self, X):
        """
        Make predictions with the Random Forest model.

        Args:
            X (np.array): Features

        Returns:
            np.array: Predictions
        """
        try:
            # Load model if not already loaded
            if self.model is None:
                self.load_model()
            
            # Make predictions
            predictions = self.model.predict(X)
            
            return predictions
        
        except Exception as e:
            logger.error(f"Error making predictions with Random Forest model: {e}")
            raise

    def predict_proba(self, X):
        """
        Make probability predictions with the Random Forest model.

        Args:
            X (np.array): Features

        Returns:
            np.array: Probability predictions
        """
        try:
            # Load model if not already loaded
            if self.model is None:
                self.load_model()
            
            # Make probability predictions
            probabilities = self.model.predict_proba(X)
            
            return probabilities
        
        except Exception as e:
            logger.error(f"Error making probability predictions with Random Forest model: {e}")
            raise

    def save_model(self):
        """
        Save the Random Forest model.
        """
        try:
            # Create models directory if it doesn't exist
            os.makedirs('models', exist_ok=True)
            
            # Save model
            joblib.dump(self.model, self.model_path)
            
            logger.info(f"Random Forest model saved to {self.model_path}.")
        
        except Exception as e:
            logger.error(f"Error saving Random Forest model: {e}")
            raise

    def load_model(self):
        """
        Load the Random Forest model.

        Returns:
            sklearn.ensemble.RandomForestClassifier: Loaded Random Forest model
        """
        try:
            # Check if model file exists
            if not os.path.exists(self.model_path):
                logger.warning(f"Random Forest model file not found at {self.model_path}.")
                return None
            
            # Load model
            self.model = joblib.load(self.model_path)
            
            logger.info(f"Random Forest model loaded from {self.model_path}.")
            
            return self.model
        
        except Exception as e:
            logger.error(f"Error loading Random Forest model: {e}")
            raise


class TradingEnvironment(gym.Env):
    """
    Trading environment for reinforcement learning.
    """

    def __init__(self, df, initial_balance=10000, transaction_fee=0.001, window_size=60):
        """
        Initialize the trading environment.

        Args:
            df (pd.DataFrame): DataFrame with OHLCV data
            initial_balance (float): Initial balance
            transaction_fee (float): Transaction fee
            window_size (int): Window size for observations
        """
        super(TradingEnvironment, self).__init__()
        
        self.df = df
        self.initial_balance = initial_balance
        self.transaction_fee = transaction_fee
        self.window_size = window_size
        
        # Action space: 0 = hold, 1 = buy, 2 = sell
        self.action_space = spaces.Discrete(3)
        
        # Observation space: price data + account info
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(window_size + 3,), dtype=np.float32
        )
        
        # Reset environment
        self.reset()
        
        logger.info(f"Trading environment initialized with {len(df)} data points.")

    def reset(self):
        """
        Reset the environment.

        Returns:
            np.array: Initial observation
        """
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        self.cost_basis = 0
        self.total_shares_bought = 0
        self.total_shares_sold = 0
        self.total_fees_paid = 0
        
        return self._get_observation()

    def step(self, action):
        """
        Take a step in the environment.

        Args:
            action (int): Action to take (0 = hold, 1 = buy, 2 = sell)

        Returns:
            tuple: (observation, reward, done, info)
        """
        # Get current price
        current_price = self.df.iloc[self.current_step]['close']
        
        # Take action
        if action == 1:  # Buy
            # Calculate maximum shares that can be bought
            max_shares = self.balance / (current_price * (1 + self.transaction_fee))
            shares_bought = max_shares
            
            # Update balance and shares held
            self.balance -= shares_bought * current_price * (1 + self.transaction_fee)
            self.shares_held += shares_bought
            self.total_shares_bought += shares_bought
            self.total_fees_paid += shares_bought * current_price * self.transaction_fee
            
            # Update cost basis
            self.cost_basis = current_price
        
        elif action == 2:  # Sell
            # Calculate shares sold
            shares_sold = self.shares_held
            
            # Update balance and shares held
            self.balance += shares_sold * current_price * (1 - self.transaction_fee)
            self.shares_held = 0
            self.total_shares_sold += shares_sold
            self.total_fees_paid += shares_sold * current_price * self.transaction_fee
            
            # Reset cost basis
            self.cost_basis = 0
        
        # Update net worth
        self.net_worth = self.balance + self.shares_held * current_price
        
        # Calculate reward
        reward = self.net_worth - self.initial_balance
        
        # Move to next step
        self.current_step += 1
        
        # Check if done
        done = self.current_step >= len(self.df) - 1
        
        # Get observation
        observation = self._get_observation()
        
        # Get info
        info = {
            'balance': self.balance,
            'shares_held': self.shares_held,
            'net_worth': self.net_worth,
            'cost_basis': self.cost_basis,
            'total_shares_bought': self.total_shares_bought,
            'total_shares_sold': self.total_shares_sold,
            'total_fees_paid': self.total_fees_paid
        }
        
        return observation, reward, done, info

    def _get_observation(self):
        """
        Get the current observation.

        Returns:
            np.array: Observation
        """
        # Get window of price data
        price_window = self.df.iloc[self.current_step - self.window_size:self.current_step]['close'].values
        
        # Normalize price window
        price_window = (price_window - np.mean(price_window)) / np.std(price_window)
        
        # Add account info to observation
        observation = np.append(
            price_window,
            [
                self.balance / self.initial_balance,
                self.shares_held * self.df.iloc[self.current_step]['close'] / self.initial_balance,
                self.cost_basis / self.df.iloc[self.current_step]['close'] if self.cost_basis > 0 else 0
            ]
        )
        
        return observation


class RLModel:
    """
    Reinforcement Learning model for trading.
    """

    def __init__(self, symbol=None, timeframe=None):
        """
        Initialize the RL model.

        Args:
            symbol (str): Trading symbol
            timeframe (str): Timeframe
        """
        self.symbol = symbol or config.SYMBOL
        self.timeframe = timeframe or config.TIMEFRAME
        self.model = None
        self.model_path = f"models/rl_{self.symbol}_{self.timeframe}"
        
        logger.info(f"RL model initialized for {self.symbol} ({self.timeframe}).")

    def build_model(self, env):
        """
        Build the RL model.

        Args:
            env (gym.Env): Trading environment

        Returns:
            stable_baselines3.PPO: RL model
        """
        try:
            # Build model
            self.model = PPO(
                'MlpPolicy',
                env,
                verbose=1,
                tensorboard_log=f"./logs/{self.symbol}_{self.timeframe}/"
            )
            
            logger.info(f"RL model built.")
            
            return self.model
        
        except Exception as e:
            logger.error(f"Error building RL model: {e}")
            raise

    def train(self, env, total_timesteps=10000):
        """
        Train the RL model.

        Args:
            env (gym.Env): Trading environment
            total_timesteps (int): Total timesteps for training

        Returns:
            stable_baselines3.PPO: Trained RL model
        """
        try:
            # Build model if not already built
            if self.model is None:
                self.build_model(env)
            
            # Train model
            self.model.learn(total_timesteps=total_timesteps)
            
            logger.info(f"RL model trained for {total_timesteps} timesteps.")
            
            # Save model
            self.save_model()
            
            return self.model
        
        except Exception as e:
            logger.error(f"Error training RL model: {e}")
            raise

    def predict(self, observation):
        """
        Make predictions with the RL model.

        Args:
            observation (np.array): Observation

        Returns:
            tuple: (action, _)
        """
        try:
            # Load model if not already loaded
            if self.model is None:
                self.load_model()
            
            # Make predictions
            action, _ = self.model.predict(observation)
            
            return action
        
        except Exception as e:
            logger.error(f"Error making predictions with RL model: {e}")
            raise

    def save_model(self):
        """
        Save the RL model.
        """
        try:
            # Create models directory if it doesn't exist
            os.makedirs('models', exist_ok=True)
            
            # Save model
            self.model.save(self.model_path)
            
            logger.info(f"RL model saved to {self.model_path}.")
        
        except Exception as e:
            logger.error(f"Error saving RL model: {e}")
            raise

    def load_model(self):
        """
        Load the RL model.

        Returns:
            stable_baselines3.PPO: Loaded RL model
        """
        try:
            # Check if model file exists
            if not os.path.exists(f"{self.model_path}.zip"):
                logger.warning(f"RL model file not found at {self.model_path}.zip.")
                return None
            
            # Load model
            self.model = PPO.load(self.model_path)
            
            logger.info(f"RL model loaded from {self.model_path}.")
            
            return self.model
        
        except Exception as e:
            logger.error(f"Error loading RL model: {e}")
            raise


class EnsembleModel:
    """
    Ensemble model combining multiple models.
    """

    def __init__(self, symbol=None, timeframe=None):
        """
        Initialize the ensemble model.

        Args:
            symbol (str): Trading symbol
            timeframe (str): Timeframe
        """
        self.symbol = symbol or config.SYMBOL
        self.timeframe = timeframe or config.TIMEFRAME
        self.models = {}
        self.weights = {}
        
        logger.info(f"Ensemble model initialized for {self.symbol} ({self.timeframe}).")

    def add_model(self, model_name, model, weight=1.0):
        """
        Add a model to the ensemble.

        Args:
            model_name (str): Model name
            model: Model instance
            weight (float): Model weight
        """
        self.models[model_name] = model
        self.weights[model_name] = weight
        
        logger.info(f"Added {model_name} to ensemble with weight {weight}.")

    def predict(self, X, threshold=0.5):
        """
        Make predictions with the ensemble model.

        Args:
            X (np.array): Features
            threshold (float): Threshold for classification

        Returns:
            np.array: Predictions
        """
        try:
            # Get predictions from each model
            predictions = {}
            for model_name, model in self.models.items():
                if model_name == 'lstm':
                    # Regression prediction
                    pred = model.predict(X)
                    # Convert to classification: 1 if positive, 0 if negative
                    predictions[model_name] = (pred > 0).astype(int)
                elif model_name == 'xgboost' or model_name == 'random_forest':
                    # Classification prediction
                    predictions[model_name] = model.predict(X)
                elif model_name == 'rl':
                    # RL prediction
                    predictions[model_name] = np.array([model.predict(x) for x in X])
            
            # Combine predictions with weights
            weighted_predictions = np.zeros(predictions[list(predictions.keys())[0]].shape)
            total_weight = 0
            
            for model_name, pred in predictions.items():
                weighted_predictions += pred * self.weights[model_name]
                total_weight += self.weights[model_name]
            
            # Normalize by total weight
            weighted_predictions /= total_weight
            
            # Apply threshold
            final_predictions = (weighted_predictions > threshold).astype(int)
            
            return final_predictions
        
        except Exception as e:
            logger.error(f"Error making predictions with ensemble model: {e}")
            raise

    def predict_proba(self, X):
        """
        Make probability predictions with the ensemble model.

        Args:
            X (np.array): Features

        Returns:
            np.array: Probability predictions
        """
        try:
            # Get probability predictions from each model
            probabilities = {}
            for model_name, model in self.models.items():
                if model_name == 'lstm':
                    # Regression prediction
                    pred = model.predict(X)
                    # Convert to probability: sigmoid(pred)
                    probabilities[model_name] = 1 / (1 + np.exp(-pred))
                elif model_name == 'xgboost' or model_name == 'random_forest':
                    # Classification probability
                    probabilities[model_name] = model.predict_proba(X)[:, 1]
                elif model_name == 'rl':
                    # RL doesn't provide probabilities, so use action as probability
                    probabilities[model_name] = np.array([model.predict(x)[0] / 2 for x in X])
            
            # Combine probabilities with weights
            weighted_probabilities = np.zeros(probabilities[list(probabilities.keys())[0]].shape)
            total_weight = 0
            
            for model_name, prob in probabilities.items():
                weighted_probabilities += prob * self.weights[model_name]
                total_weight += self.weights[model_name]
            
            # Normalize by total weight
            weighted_probabilities /= total_weight
            
            return weighted_probabilities
        
        except Exception as e:
            logger.error(f"Error making probability predictions with ensemble model: {e}")
            raise 