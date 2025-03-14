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
        rl_path = os.path.join(self.config.model_dir, f"{self.config.rl_model_name}.zip")
        
        # Try to load XGBoost model
        if os.path.exists(xgb_path):
            logger.info(f"Loading existing XGBoost model from {xgb_path}...")
            try:
                self.xgb_model = xgb.Booster()
                self.xgb_model.load_model(xgb_path)
                logger.info("XGBoost model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load XGBoost model: {e}")
                self.xgb_model = None
        else:
            logger.info("No existing XGBoost model found, will train a new one")
            self.xgb_model = None
        
        # Try to load feature scaler
        if os.path.exists(scaler_path):
            logger.info(f"Loading feature scaler from {scaler_path}...")
            try:
                self.scaler = joblib.load(scaler_path)
                logger.info("Feature scaler loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load feature scaler: {e}")
                self.scaler = StandardScaler()
        
        # Try to load RL model
        if os.path.exists(rl_path):
            logger.info(f"Loading existing RL model from {rl_path}...")
            try:
                # Create a dummy environment for loading the model
                dummy_env = DummyVecEnv([lambda: TradingEnvironment(pd.DataFrame({col: [0] for col in self.features}), self.config)])
                self.rl_model = PPO.load(rl_path, env=dummy_env)
                logger.info("RL model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load RL model: {e}")
                self.rl_model = None
        else:
            logger.info("No existing RL model found, will train a new one")
            self.rl_model = None
    
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
        X['close_to_vwap'] = df['close'] / (df['vwap'] + 1e-8) - 1  # Avoid division by zero
        X['bb_position'] = (df['close'] - df['bollinger_lower']) / (df['bollinger_upper'] - df['bollinger_lower'] + 1e-8)
        
        return X, y
    
    def train(self, historical_data):
        """Train both XGBoost and RL models"""
        # Clean data before training
        historical_data = self._preprocess_data(historical_data)
        
        self._train_xgb_model(historical_data)
        self._train_rl_model(historical_data)
    
    def _preprocess_data(self, data):
        """Preprocess data to ensure it's suitable for training"""
        # Make a copy to avoid modifying original
        df = data.copy()
        
        # Drop rows with NaN values
        df.dropna(inplace=True)
        
        # Replace infinite values with large numbers
        df.replace([np.inf, -np.inf], [1e6, -1e6], inplace=True)
        
        # Remove extreme outliers (more than 5 std from mean) for each feature
        for col in self.features:
            if col in df.columns:
                mean = df[col].mean()
                std = df[col].std()
                if std > 0:  # Avoid division by zero
                    df = df[(df[col] > mean - 5 * std) & (df[col] < mean + 5 * std)]
        
        logger.info(f"Data preprocessed: {len(data)} rows -> {len(df)} rows")
        return df
    
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
        
        try:
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
            rl_zip_path = f"{rl_path}.zip"
            
            if os.path.exists(rl_zip_path):
                logger.info(f"Loading existing PPO model from {rl_zip_path}...")
                try:
                    self.rl_model = PPO.load(rl_path, env=env)
                    logger.info("Existing PPO model loaded for further training")
                except Exception as e:
                    logger.error(f"Error loading existing model, creating new one: {e}")
                    self.rl_model = None
            
            if self.rl_model is None:
                logger.info("Creating new PPO model...")
                # Configure PPO with smaller network and parameters to prevent NaN issues
                policy_kwargs = dict(
                    net_arch=[64, 64]  # Smaller network architecture
                )
                
                self.rl_model = PPO(
                    "MlpPolicy",
                    env,
                    policy_kwargs=policy_kwargs,
                    verbose=1,
                    tensorboard_log=os.path.join(self.config.model_dir, "tb_logs"),
                    learning_rate=0.0001,  # Lower learning rate
                    gamma=0.99,
                    n_steps=2048,
                    ent_coef=0.01,  # Add some entropy for exploration
                    clip_range=0.2,
                    batch_size=64
                )
            
            # Train model
            logger.info("Starting RL model training...")
            self.rl_model.learn(
                total_timesteps=100000,  # Adjust based on your requirements
                callback=checkpoint_callback
            )
            
            # Save final model
            try:
                save_path = os.path.join(self.config.model_dir, f"{self.config.rl_model_name}")
                self.rl_model.save(save_path)
                logger.info(f"RL model saved to {save_path}.zip")
                
                # Verify the file was created
                if os.path.exists(f"{save_path}.zip"):
                    logger.info(f"Verified: RL model file exists at {save_path}.zip")
                    # Try to reload it as a test
                    test_model = PPO.load(save_path)
                    logger.info("RL model reload test successful")
                else:
                    logger.error(f"RL model file not found at {save_path}.zip after saving")
            except Exception as e:
                logger.error(f"Error saving RL model: {e}", exc_info=True)
            
            logger.info("RL model training completed")
            
        except Exception as e:
            logger.error(f"Error training RL model: {e}", exc_info=True)
            raise
    
    def predict_direction(self, market_data):
        """Predict price direction using XGBoost model"""
        if self.xgb_model is None:
            logger.warning("XGBoost model not trained yet. Using random prediction.")
            return np.random.choice([0, 1])
        
        # Clean input data
        market_data = self._preprocess_data(market_data)
        if len(market_data) == 0:
            logger.warning("No valid data for prediction after preprocessing")
            return np.random.choice([0, 1])
        
        features = market_data[self.features].iloc[-1:].copy()
        features['close_to_vwap'] = market_data['close'].iloc[-1] / (market_data['vwap'].iloc[-1] + 1e-8) - 1
        features['bb_position'] = (market_data['close'].iloc[-1] - market_data['bollinger_lower'].iloc[-1]) / (
                market_data['bollinger_upper'].iloc[-1] - market_data['bollinger_lower'].iloc[-1] + 1e-8)
        
        # Check for NaN values
        if features.isna().any().any():
            logger.warning("NaN values in prediction features, using default prediction")
            return np.random.choice([0, 1])
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Create DMatrix
        dmatrix = xgb.DMatrix(features_scaled)
        
        # Get prediction
        prediction = self.xgb_model.predict(dmatrix)[0]
        
        return prediction > 0.5  # Return True if probability > 0.5
    
    def get_trading_action(self, market_data, account_state):
        """Get trading action from the RL model"""
        try:
            if self.rl_model is None:
                logger.warning("RL model not trained yet. Using default action (do nothing).")
                # Check if the model file exists but wasn't loaded properly
                rl_path = os.path.join(self.config.model_dir, f"{self.config.rl_model_name}.zip")
                if os.path.exists(rl_path):
                    logger.error(f"RL model file exists at {rl_path} but couldn't be loaded. Try deleting it and retraining.")
                    
                return 0  # No action
            
            # Clean input data
            market_data = self._preprocess_data(market_data)
            if len(market_data) == 0:
                logger.warning("No valid data for prediction after preprocessing")
                return 0
            
            # Create environment with current state
            env = TradingEnvironment(market_data, self.config, initial_state=account_state)
            
            # Get observation
            obs = env.get_observation()
            
            # Log info about the observation and model
            logger.debug(f"Observation shape: {obs.shape}, dtype: {obs.dtype}")
            
            # Check for NaN or infinite values in a safe way
            for i in range(len(obs)):
                if not np.isfinite(float(obs[i])):
                    logger.warning(f"Non-finite value detected in observation at index {i}, using default action")
                    return 0
            
            # Get action from model
            action, _ = self.rl_model.predict(obs, deterministic=True)
            logger.debug(f"Model predicted action: {action}")
            
            return action
            
        except Exception as e:
            logger.error(f"Error getting trading action: {e}", exc_info=True)
            return 0  # Default to no action on error

