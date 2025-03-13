import os
import logging
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import tensorflow as tf
from tensorflow import keras
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from gym import spaces
import gym
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

from models.trading_env import TradingEnvironment

logger = logging.getLogger(__name__)

class ModelManager:
    def __init__(self, config):
        self.config = config
        self.scaler = StandardScaler()
        self.xgb_model = None
        self.rl_model = None
        self.features = config.features
        
        # Create models directory if it doesn't exist
        os.makedirs(self.config.model_dir, exist_ok=True)
        
        # Initialize models
        self._load_or_create_models()
    
    def _load_or_create_models(self):
        """Load existing models or create new ones if they don't exist"""
        xgb_path = os.path.join(self.config.model_dir, f"{self.config.prediction_model_name}.json")
        scaler_path = os.path.join(self.config.model_dir, "feature_scaler.pkl")
        
        # Try to load XGBoost model
        if os.path.exists(xgb_path):
            logger.info("Loading existing XGBoost model...")
            self.xgb_model = xgb.Booster()
            self.xgb_model.load_model(xgb_path)
        else:
            logger.info("Creating new XGBoost model...")
            self.xgb_model = None
        
        # Try to load feature scaler
        if os.path.exists(scaler_path):
            logger.info("Loading feature scaler...")
            self.scaler = joblib.load(scaler_path)
        
        # For the RL model, we'll check if it exists when we need it
    
    def prepare_features(self, df):
        """Prepare features for model training or prediction"""
        # Drop rows with NaN values
        df = df.dropna()
        
        # Extract features
        X = df[self.features].copy()
        
        # Create target variable (1 if price goes up in next candle, 0 otherwise)
        df['next_close'] = df['close'].shift(-1)
        df['price_direction'] = (df['next_close'] > df['close']).astype(int)
        y = df['price_direction'].copy()
        
        # Add some additional engineered features
        X['close_to_vwap'] = df['close'] / df['vwap'] - 1
        X['bb_position'] = (df['close'] - df['bollinger_lower']) / (df['bollinger_upper'] - df['bollinger_lower'])
        
        return X, y
    
    def train(self, historical_data):
        """Train both XGBoost and RL models"""
        self._train_xgb_model(historical_data)
        self._train_rl_model(historical_data)
    
    def _train_xgb_model(self, historical_data):
        """Train XGBoost model for predicting price direction"""
        logger.info("Training XGBoost model for price direction prediction...")
        
        # Prepare features and target
        X, y = self.prepare_features(historical_data)
        
        # Remove rows with NaN values
        valid_idx = ~(X.isna().any(axis=1) | y.isna())
        X = X[valid_idx]
        y = y[valid_idx]
        
        if len(X) == 0:
            logger.error("No valid data for training after NaN removal")
            return
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        # Scale features
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Save scaler for later use
        joblib.dump(self.scaler, os.path.join(self.config.model_dir, "feature_scaler.pkl"))
        
        # Create and train model
        dtrain = xgb.DMatrix(X_train_scaled, label=y_train)
        dtest = xgb.DMatrix(X_test_scaled, label=y_test)
        
        params = {
            'max_depth': 6,
            'eta': 0.1,
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'nthread': -1
        }
        
        evallist = [(dtest, 'eval'), (dtrain, 'train')]
        num_round = 100
        
        self.xgb_model = xgb.train(params, dtrain, num_round, evallist, early_stopping_rounds=10)
        
        # Save model
        self.xgb_model.save_model(os.path.join(self.config.model_dir, f"{self.config.prediction_model_name}.json"))
        
        # Evaluate model
        y_pred = self.xgb_model.predict(dtest)
        y_pred_binary = (y_pred > 0.5).astype(int)
        accuracy = (y_pred_binary == y_test.values).mean()
        logger.info(f"XGBoost model accuracy: {accuracy:.4f}")
    
    def _train_rl_model(self, historical_data):
        """Train Reinforcement Learning model using PPO"""
        logger.info("Training PPO-based RL model for trading strategy...")
        
        # Create environment
        env = TradingEnvironment(historical_data, self.config)
        env = DummyVecEnv([lambda: env])
        
        # Callbacks for saving and evaluating
        save_path = os.path.join(self.config.model_dir, "checkpoints")
        os.makedirs(save_path, exist_ok=True)
        
        checkpoint_callback = CheckpointCallback(
            save_freq=10000,
            save_path=save_path,
            name_prefix=self.config.rl_model_name
        )
        
        # Create or load model
        rl_path = os.path.join(self.config.model_dir, f"{self.config.rl_model_name}")
        
        if os.path.exists(rl_path + ".zip"):
            logger.info("Loading existing PPO model...")
            self.rl_model = PPO.load(rl_path, env=env)
        else:
            logger.info("Creating new PPO model...")
            self.rl_model = PPO(
                "MlpPolicy",
                env,
                verbose=1,
                tensorboard_log=os.path.join(self.config.model_dir, "tb_logs"),
                learning_rate=0.0003
            )
        
        # Train model
        self.rl_model.learn(
            total_timesteps=100000,  # Adjust based on your requirements
            callback=checkpoint_callback
        )
        
        # Save final model
        self.rl_model.save(rl_path)
        
        logger.info("RL model training completed")
    
    def predict_direction(self, market_data):
        """Predict price direction using XGBoost model"""
        if self.xgb_model is None:
            logger.warning("XGBoost model not trained yet. Using random prediction.")
            return np.random.choice([0, 1])
        
        features = market_data[self.features].iloc[-1:].copy()
        features['close_to_vwap'] = market_data['close'].iloc[-1] / market_data['vwap'].iloc[-1] - 1
        features['bb_position'] = (market_data['close'].iloc[-1] - market_data['bollinger_lower'].iloc[-1]) / (
                market_data['bollinger_upper'].iloc[-1] - market_data['bollinger_lower'].iloc[-1])
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Create DMatrix
        dmatrix = xgb.DMatrix(features_scaled)
        
        # Get prediction
        prediction = self.xgb_model.predict(dmatrix)[0]
        
        return prediction > 0.5  # Return True if probability > 0.5
    
    def get_trading_action(self, market_data, account_state):
        """Get trading action from the RL model"""
        if self.rl_model is None:
            logger.warning("RL model not trained yet. Using default action (do nothing).")
            return 0  # No action
        
        # Create environment with current state
        env = TradingEnvironment(market_data, self.config, initial_state=account_state)
        
        # Get observation
        obs = env.get_observation()
        
        # Get action from model
        action, _ = self.rl_model.predict(obs, deterministic=True)
        
        return action
