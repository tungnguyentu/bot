"""
Data collection and preprocessing module for the AI Trading Bot.
"""

import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import ccxt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from ta.trend import SMAIndicator, EMAIndicator, MACD, VortexIndicator
from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator, VolumePriceTrendIndicator
import joblib
import requests
from bs4 import BeautifulSoup
from textblob import TextBlob
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from pytrends.request import TrendReq

import config
from indicators import (
    calculate_rsi,
    calculate_vwap,
    calculate_atr,
    detect_volume_spike,
    calculate_bollinger_bands,
    calculate_macd,
)
from utils import setup_logger, convert_to_dataframe

# Initialize logger
logger = logging.getLogger("trading_bot")

# Download NLTK resources if not already downloaded
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')


class DataProcessor:
    """
    Data collection and preprocessing for the AI Trading Bot.
    """

    def __init__(self, symbol=None, timeframe=None, lookback_period=365):
        """
        Initialize the data processor.

        Args:
            symbol (str): Trading symbol
            timeframe (str): Timeframe
            lookback_period (int): Number of days to look back for historical data
        """
        self.symbol = symbol or config.SYMBOL
        self.timeframe = timeframe or config.TIMEFRAME
        self.lookback_period = lookback_period
        
        # Initialize exchange for historical data
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future',
                'adjustForTimeDifference': True
            }
        })
        
        # Initialize scalers
        self.price_scaler = MinMaxScaler(feature_range=(0, 1))
        self.volume_scaler = MinMaxScaler(feature_range=(0, 1))
        self.feature_scaler = StandardScaler()
        
        # Initialize sentiment analyzer
        self.sia = SentimentIntensityAnalyzer()
        
        # Initialize Google Trends
        self.pytrends = TrendReq(hl='en-US', tz=360)
        
        logger.info(f"Data processor initialized for {self.symbol} ({self.timeframe}).")

    def fetch_historical_data(self, start_date=None, end_date=None):
        """
        Fetch historical data for the specified symbol and timeframe.

        Args:
            start_date (str): Start date (YYYY-MM-DD)
            end_date (str): End date (YYYY-MM-DD)

        Returns:
            pd.DataFrame: Historical data
        """
        try:
            # Set default dates if not provided
            if not end_date:
                end_date = datetime.now().strftime('%Y-%m-%d')
            
            if not start_date:
                start_date = (datetime.now() - timedelta(days=self.lookback_period)).strftime('%Y-%m-%d')
            
            # Convert dates to timestamps
            start_timestamp = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
            end_timestamp = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)
            
            # Fetch historical data
            logger.info(f"Fetching historical data for {self.symbol} ({self.timeframe}) from {start_date} to {end_date}...")
            
            # Initialize empty list for all klines
            all_klines = []
            
            # Fetch data in chunks to avoid rate limits
            current_timestamp = start_timestamp
            while current_timestamp < end_timestamp:
                # Fetch klines
                klines = self.exchange.fetch_ohlcv(
                    symbol=self.symbol,
                    timeframe=self.timeframe,
                    since=current_timestamp,
                    limit=1000
                )
                
                if not klines:
                    break
                
                # Add klines to list
                all_klines.extend(klines)
                
                # Update current timestamp
                current_timestamp = klines[-1][0] + 1
                
                # Sleep to avoid rate limits
                self.exchange.sleep(self.exchange.rateLimit / 1000)
            
            # Convert to DataFrame
            df = convert_to_dataframe(all_klines)
            
            # Filter by date range
            df = df[(df.index >= start_date) & (df.index <= end_date)]
            
            logger.info(f"Fetched {len(df)} candles for {self.symbol} ({self.timeframe}).")
            
            return df
        
        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            raise

    def add_technical_indicators(self, df):
        """
        Add technical indicators to the DataFrame.

        Args:
            df (pd.DataFrame): DataFrame with OHLCV data

        Returns:
            pd.DataFrame: DataFrame with technical indicators
        """
        try:
            # Make a copy to avoid modifying the original DataFrame
            df_with_indicators = df.copy()
            
            # Basic indicators
            df_with_indicators['rsi'] = RSIIndicator(close=df_with_indicators['close']).rsi()
            df_with_indicators['atr'] = AverageTrueRange(high=df_with_indicators['high'], low=df_with_indicators['low'], close=df_with_indicators['close']).average_true_range()
            df_with_indicators['vwap'] = calculate_vwap(df_with_indicators)
            
            # Trend indicators
            df_with_indicators['sma_20'] = SMAIndicator(close=df_with_indicators['close'], window=20).sma_indicator()
            df_with_indicators['sma_50'] = SMAIndicator(close=df_with_indicators['close'], window=50).sma_indicator()
            df_with_indicators['sma_200'] = SMAIndicator(close=df_with_indicators['close'], window=200).sma_indicator()
            df_with_indicators['ema_12'] = EMAIndicator(close=df_with_indicators['close'], window=12).ema_indicator()
            df_with_indicators['ema_26'] = EMAIndicator(close=df_with_indicators['close'], window=26).ema_indicator()
            
            # MACD
            macd = MACD(close=df_with_indicators['close'])
            df_with_indicators['macd'] = macd.macd()
            df_with_indicators['macd_signal'] = macd.macd_signal()
            df_with_indicators['macd_diff'] = macd.macd_diff()
            
            # Bollinger Bands
            bollinger = BollingerBands(close=df_with_indicators['close'])
            df_with_indicators['bb_upper'] = bollinger.bollinger_hband()
            df_with_indicators['bb_middle'] = bollinger.bollinger_mavg()
            df_with_indicators['bb_lower'] = bollinger.bollinger_lband()
            df_with_indicators['bb_width'] = (df_with_indicators['bb_upper'] - df_with_indicators['bb_lower']) / df_with_indicators['bb_middle']
            
            # Momentum indicators
            df_with_indicators['stoch_k'] = StochasticOscillator(high=df_with_indicators['high'], low=df_with_indicators['low'], close=df_with_indicators['close']).stoch()
            df_with_indicators['stoch_d'] = StochasticOscillator(high=df_with_indicators['high'], low=df_with_indicators['low'], close=df_with_indicators['close']).stoch_signal()
            df_with_indicators['williams_r'] = WilliamsRIndicator(high=df_with_indicators['high'], low=df_with_indicators['low'], close=df_with_indicators['close']).williams_r()
            
            # Volume indicators
            df_with_indicators['obv'] = OnBalanceVolumeIndicator(close=df_with_indicators['close'], volume=df_with_indicators['volume']).on_balance_volume()
            df_with_indicators['vpt'] = VolumePriceTrendIndicator(close=df_with_indicators['close'], volume=df_with_indicators['volume']).volume_price_trend()
            df_with_indicators['volume_ma'] = df_with_indicators['volume'].rolling(window=20).mean()
            df_with_indicators['volume_ratio'] = df_with_indicators['volume'] / df_with_indicators['volume_ma']
            
            # Vortex Indicator
            vortex = VortexIndicator(high=df_with_indicators['high'], low=df_with_indicators['low'], close=df_with_indicators['close'])
            df_with_indicators['vortex_pos'] = vortex.vortex_indicator_pos()
            df_with_indicators['vortex_neg'] = vortex.vortex_indicator_neg()
            df_with_indicators['vortex_diff'] = df_with_indicators['vortex_pos'] - df_with_indicators['vortex_neg']
            
            # Price action features
            df_with_indicators['body_size'] = abs(df_with_indicators['close'] - df_with_indicators['open']) / df_with_indicators['open']
            df_with_indicators['upper_shadow'] = (df_with_indicators['high'] - df_with_indicators[['open', 'close']].max(axis=1)) / df_with_indicators['open']
            df_with_indicators['lower_shadow'] = (df_with_indicators[['open', 'close']].min(axis=1) - df_with_indicators['low']) / df_with_indicators['open']
            df_with_indicators['candle_range'] = (df_with_indicators['high'] - df_with_indicators['low']) / df_with_indicators['open']
            
            # Trend direction and strength
            df_with_indicators['price_sma20_ratio'] = df_with_indicators['close'] / df_with_indicators['sma_20']
            df_with_indicators['price_sma50_ratio'] = df_with_indicators['close'] / df_with_indicators['sma_50']
            df_with_indicators['price_sma200_ratio'] = df_with_indicators['close'] / df_with_indicators['sma_200']
            df_with_indicators['sma20_sma50_ratio'] = df_with_indicators['sma_20'] / df_with_indicators['sma_50']
            
            # Volatility
            df_with_indicators['volatility'] = df_with_indicators['close'].pct_change().rolling(window=20).std()
            
            # Returns
            df_with_indicators['return_1'] = df_with_indicators['close'].pct_change(1)
            df_with_indicators['return_5'] = df_with_indicators['close'].pct_change(5)
            df_with_indicators['return_10'] = df_with_indicators['close'].pct_change(10)
            df_with_indicators['return_20'] = df_with_indicators['close'].pct_change(20)
            
            # Fill NaN values
            df_with_indicators = df_with_indicators.fillna(method='bfill')
            
            logger.info(f"Added {len(df_with_indicators.columns) - len(df.columns)} technical indicators.")
            
            return df_with_indicators
        
        except Exception as e:
            logger.error(f"Error adding technical indicators: {e}")
            raise

    def add_sentiment_data(self, df):
        """
        Add sentiment data to the DataFrame.

        Args:
            df (pd.DataFrame): DataFrame with OHLCV data

        Returns:
            pd.DataFrame: DataFrame with sentiment data
        """
        try:
            # Make a copy to avoid modifying the original DataFrame
            df_with_sentiment = df.copy()
            
            # Initialize sentiment columns
            df_with_sentiment['news_sentiment'] = 0.0
            df_with_sentiment['social_sentiment'] = 0.0
            df_with_sentiment['google_trends'] = 0.0
            
            # Get sentiment data for each day
            for date in df_with_sentiment.index.unique():
                date_str = date.strftime('%Y-%m-%d')
                
                # Get news sentiment (simplified example)
                try:
                    news_sentiment = self._get_news_sentiment(self.symbol.replace('USDT', ''), date_str)
                    df_with_sentiment.loc[date, 'news_sentiment'] = news_sentiment
                except Exception as e:
                    logger.warning(f"Error getting news sentiment for {date_str}: {e}")
                
                # Get social media sentiment (simplified example)
                try:
                    social_sentiment = self._get_social_sentiment(self.symbol.replace('USDT', ''), date_str)
                    df_with_sentiment.loc[date, 'social_sentiment'] = social_sentiment
                except Exception as e:
                    logger.warning(f"Error getting social sentiment for {date_str}: {e}")
                
                # Get Google Trends data (simplified example)
                try:
                    google_trend = self._get_google_trends(self.symbol.replace('USDT', ''), date_str)
                    df_with_sentiment.loc[date, 'google_trends'] = google_trend
                except Exception as e:
                    logger.warning(f"Error getting Google Trends for {date_str}: {e}")
            
            # Fill NaN values
            df_with_sentiment = df_with_sentiment.fillna(method='bfill')
            
            logger.info(f"Added sentiment data.")
            
            return df_with_sentiment
        
        except Exception as e:
            logger.error(f"Error adding sentiment data: {e}")
            # Return original DataFrame if error occurs
            return df

    def _get_news_sentiment(self, coin, date):
        """
        Get news sentiment for a specific coin and date.
        This is a simplified example. In a real implementation, you would use a news API.

        Args:
            coin (str): Coin name (e.g., 'BTC')
            date (str): Date string (YYYY-MM-DD)

        Returns:
            float: Sentiment score (-1 to 1)
        """
        # Simplified implementation - in a real bot, you would use a news API
        # For now, return a random sentiment score
        return np.random.uniform(-1, 1)

    def _get_social_sentiment(self, coin, date):
        """
        Get social media sentiment for a specific coin and date.
        This is a simplified example. In a real implementation, you would use a social media API.

        Args:
            coin (str): Coin name (e.g., 'BTC')
            date (str): Date string (YYYY-MM-DD)

        Returns:
            float: Sentiment score (-1 to 1)
        """
        # Simplified implementation - in a real bot, you would use a social media API
        # For now, return a random sentiment score
        return np.random.uniform(-1, 1)

    def _get_google_trends(self, coin, date):
        """
        Get Google Trends data for a specific coin and date.
        This is a simplified example. In a real implementation, you would use the Google Trends API.

        Args:
            coin (str): Coin name (e.g., 'BTC')
            date (str): Date string (YYYY-MM-DD)

        Returns:
            float: Google Trends score (0 to 100)
        """
        # Simplified implementation - in a real bot, you would use the Google Trends API
        # For now, return a random trends score
        return np.random.uniform(0, 100)

    def prepare_data_for_training(self, df, target_column='return_1', sequence_length=60):
        """
        Prepare data for training ML models.

        Args:
            df (pd.DataFrame): DataFrame with features
            target_column (str): Target column for prediction
            sequence_length (int): Sequence length for time series models

        Returns:
            tuple: (X_train, y_train, X_val, y_val, X_test, y_test, feature_names)
        """
        try:
            # Make a copy to avoid modifying the original DataFrame
            df_ml = df.copy()
            
            # Drop unnecessary columns
            drop_columns = ['open_time', 'close_time', 'ignore']
            df_ml = df_ml.drop([col for col in drop_columns if col in df_ml.columns], axis=1)
            
            # Create target variable (future returns)
            df_ml['target'] = df_ml[target_column].shift(-1)
            
            # Drop rows with NaN values
            df_ml = df_ml.dropna()
            
            # Split data into features and target
            X = df_ml.drop(['target'], axis=1)
            y = df_ml['target']
            
            # Get feature names
            feature_names = X.columns.tolist()
            
            # Scale features
            X_scaled = self.feature_scaler.fit_transform(X)
            X_scaled_df = pd.DataFrame(X_scaled, columns=feature_names, index=X.index)
            
            # Create sequences for time series models
            X_sequences, y_sequences = self._create_sequences(X_scaled_df, y, sequence_length)
            
            # Split data into train, validation, and test sets
            train_size = int(len(X_sequences) * 0.7)
            val_size = int(len(X_sequences) * 0.15)
            
            X_train = X_sequences[:train_size]
            y_train = y_sequences[:train_size]
            
            X_val = X_sequences[train_size:train_size+val_size]
            y_val = y_sequences[train_size:train_size+val_size]
            
            X_test = X_sequences[train_size+val_size:]
            y_test = y_sequences[train_size+val_size:]
            
            logger.info(f"Prepared data for training: {len(X_train)} train, {len(X_val)} validation, {len(X_test)} test samples.")
            
            # Save scalers
            os.makedirs('models', exist_ok=True)
            joblib.dump(self.feature_scaler, f'models/feature_scaler_{self.symbol}_{self.timeframe}.pkl')
            
            return X_train, y_train, X_val, y_val, X_test, y_test, feature_names
        
        except Exception as e:
            logger.error(f"Error preparing data for training: {e}")
            raise

    def _create_sequences(self, X, y, sequence_length):
        """
        Create sequences for time series models.

        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target
            sequence_length (int): Sequence length

        Returns:
            tuple: (X_sequences, y_sequences)
        """
        X_sequences = []
        y_sequences = []
        
        for i in range(len(X) - sequence_length):
            X_sequences.append(X.iloc[i:i+sequence_length].values)
            y_sequences.append(y.iloc[i+sequence_length])
        
        return np.array(X_sequences), np.array(y_sequences)

    def prepare_data_for_prediction(self, df, sequence_length=60):
        """
        Prepare data for making predictions.

        Args:
            df (pd.DataFrame): DataFrame with features
            sequence_length (int): Sequence length for time series models

        Returns:
            np.array: Prepared data for prediction
        """
        try:
            # Make a copy to avoid modifying the original DataFrame
            df_pred = df.copy()
            
            # Drop unnecessary columns
            drop_columns = ['open_time', 'close_time', 'ignore']
            df_pred = df_pred.drop([col for col in drop_columns if col in df_pred.columns], axis=1)
            
            # Load scaler
            scaler_path = f'models/feature_scaler_{self.symbol}_{self.timeframe}.pkl'
            if os.path.exists(scaler_path):
                self.feature_scaler = joblib.load(scaler_path)
            else:
                logger.warning(f"Scaler not found at {scaler_path}. Using default scaler.")
            
            # Scale features
            X_scaled = self.feature_scaler.transform(df_pred)
            
            # Create sequence
            X_sequence = X_scaled[-sequence_length:].reshape(1, sequence_length, X_scaled.shape[1])
            
            return X_sequence
        
        except Exception as e:
            logger.error(f"Error preparing data for prediction: {e}")
            raise

    def save_data(self, df, filename):
        """
        Save data to CSV file.

        Args:
            df (pd.DataFrame): DataFrame to save
            filename (str): Filename
        """
        try:
            # Create data directory if it doesn't exist
            os.makedirs('data', exist_ok=True)
            
            # Save data
            filepath = os.path.join('data', filename)
            df.to_csv(filepath)
            
            logger.info(f"Saved data to {filepath}.")
        
        except Exception as e:
            logger.error(f"Error saving data: {e}")
            raise

    def load_data(self, filename):
        """
        Load data from CSV file.

        Args:
            filename (str): Filename

        Returns:
            pd.DataFrame: Loaded data
        """
        try:
            # Load data
            filepath = os.path.join('data', filename)
            df = pd.read_csv(filepath, index_col=0, parse_dates=True)
            
            logger.info(f"Loaded data from {filepath}.")
            
            return df
        
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise 