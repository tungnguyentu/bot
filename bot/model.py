import logging
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

class XGBoostModel:
    """XGBoost model for predicting trading signals."""
    
    def __init__(self, model_path='models/xgboost_model.pkl'):
        """Initialize XGBoost model."""
        self.model_path = model_path
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        
        # Try to load existing model
        try:
            self.load()
        except (FileNotFoundError, Exception) as e:
            logger.warning(f"No existing model found or error loading model: {e}")
    
    def _prepare_features(self, df):
        """Prepare features for model training or prediction."""
        # Define features to use
        feature_columns = [
            'open', 'high', 'low', 'close', 'volume',
            'bb_middle', 'bb_upper', 'bb_lower', 'bb_width',
            'rsi', 'macd', 'macd_signal', 'macd_hist',
            'vwap', 'atr'
        ]
        
        self.feature_columns = feature_columns
        
        # Create features DataFrame
        X = df[feature_columns].copy()
        
        return X
    
    def _prepare_labels(self, df, prediction_horizon=5):
        """Prepare labels for model training."""
        # Future price change as a percentage
        df['future_return'] = df['close'].shift(-prediction_horizon) / df['close'] - 1
        
        # Create classification labels:
        # 1 for significant positive return (buy signal)
        # -1 for significant negative return (sell signal)
        # 0 for neutral (no trade)
        threshold = df['atr'].rolling(window=14).mean() * 0.5 / df['close']
        df['signal'] = 0
        df.loc[df['future_return'] > threshold, 'signal'] = 1
        df.loc[df['future_return'] < -threshold, 'signal'] = -1
        
        y = df['signal'].copy()
        
        return y
    
    def train(self, df):
        """Train the XGBoost model with the given data."""
        logger.info("Preparing data for model training")
        
        # Prepare features and labels
        X = self._prepare_features(df)
        y = self._prepare_labels(df)
        
        # Remove NaN values
        valid_idx = ~(X.isna().any(axis=1) | y.isna())
        X = X[valid_idx]
        y = y[valid_idx]
        
        if len(X) == 0:
            logger.error("No valid data for training after NaN removal")
            return
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        logger.info(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")
        
        # Define XGBoost parameters
        params = {
            'objective': 'multi:softprob',
            'num_class': 3,  # 3 classes: -1, 0, 1
            'max_depth': 5,
            'eta': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 1,
            'gamma': 0.1,
            'n_jobs': -1  # Use all available cores
        }
        
        # Train model
        logger.info("Training XGBoost model")
        dtrain = xgb.DMatrix(X_train, label=y_train + 1)  # +1 because XGBoost requires labels 0, 1, 2
        dtest = xgb.DMatrix(X_test, label=y_test + 1)
        
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=100,
            evals=[(dtrain, 'train'), (dtest, 'test')],
            early_stopping_rounds=10,
            verbose_eval=False
        )
        
        # Save model
        self.save()
        
        logger.info("Model training completed")
    
    def predict(self, df):
        """Predict signals using the trained model."""
        if self.model is None:
            logger.error("Model not trained or loaded. Cannot make predictions.")
            return np.zeros(len(df))
        
        # Prepare features
        X = self._prepare_features(df)
        
        # Handle NaN values - updated to use non-deprecated methods
        X = X.ffill().bfill()  # Replace deprecated fillna(method='ffill/bfill')
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        dmatrix = xgb.DMatrix(X_scaled)
        raw_preds = self.model.predict(dmatrix)
        
        # Convert to class labels (-1, 0, 1)
        # raw_preds is a matrix of probabilities for each class
        # We take the class with highest probability
        pred_classes = np.argmax(raw_preds, axis=1) - 1  # -1 to convert back from 0,1,2 to -1,0,1
        
        return pred_classes
    
    def predict_with_confidence(self, df):
        """Predict signals with confidence scores."""
        if self.model is None:
            logger.error("Model not trained or loaded. Cannot make predictions.")
            return 0, 0.0
        
        # Prepare features
        X = self._prepare_features(df)
        
        # Handle NaN values - updated to use non-deprecated methods
        X = X.ffill().bfill()  # Replace deprecated fillna(method='ffill/bfill')
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        dmatrix = xgb.DMatrix(X_scaled)
        raw_preds = self.model.predict(dmatrix)
        
        # Get the last prediction (most recent data point)
        last_pred_probs = raw_preds[-1]
        pred_class = np.argmax(last_pred_probs) - 1  # -1 to convert back from 0,1,2 to -1,0,1
        confidence = last_pred_probs[np.argmax(last_pred_probs)]
        
        return pred_class, float(confidence)
    
    def save(self):
        """Save the trained model to disk."""
        if self.model is not None:
            joblib.dump({
                'model': self.model,
                'scaler': self.scaler,
                'feature_columns': self.feature_columns
            }, self.model_path)
            logger.info(f"Model saved to {self.model_path}")
    
    def load(self):
        """Load the trained model from disk."""
        saved_model = joblib.load(self.model_path)
        self.model = saved_model['model']
        self.scaler = saved_model['scaler']
        self.feature_columns = saved_model['feature_columns']
        logger.info(f"Model loaded from {self.model_path}")
