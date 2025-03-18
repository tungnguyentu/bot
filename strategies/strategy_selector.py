import numpy as np
import pandas as pd
import ta
import traceback

import config
from utils.logger import setup_logger

class StrategySelector:
    def __init__(self, client):
        self.client = client
        self.logger = setup_logger('strategy_selector', 'logs/strategies.log')
        
    def select_strategy(self, symbol):
        """
        Determine whether scalping or swing trading is more appropriate
        for current market conditions.
        
        Returns:
            str: 'scalping' or 'swing'
        """
        try:
            # Get data for both timeframes
            short_tf_data = self.client.get_historical_klines(
                symbol, config.TIMEFRAMES['scalping'], limit=100)
            long_tf_data = self.client.get_historical_klines(
                symbol, config.TIMEFRAMES['swing'], limit=100)
            
            # Calculate volatility metrics
            short_atr = self._calculate_atr(short_tf_data)
            long_atr = self._calculate_atr(long_tf_data)
            
            # Calculate volume metrics - store full series, not just averages
            short_vol = short_tf_data['volume']  # Full series for calculations
            long_vol = long_tf_data['volume']    # Full series for calculations
            
            # Simple average for comparison
            short_vol_avg = short_vol.mean() if not short_vol.empty else 0
            long_vol_avg = long_vol.mean() if not long_vol.empty else 0
            
            # Calculate trend strength
            short_trend_strength = self._calculate_adx(short_tf_data)
            long_trend_strength = self._calculate_adx(long_tf_data)
            
            # Calculate Bollinger Band width (volatility indicator)
            short_bb_width = self._calculate_bb_width(short_tf_data)
            long_bb_width = self._calculate_bb_width(long_tf_data)
            
            # Decision logic - score based approach
            scalping_score = 0
            swing_score = 0
            
            # Volatility conditions
            if short_atr > long_atr * 0.05:  # High short-term volatility
                scalping_score += 1
            else:
                swing_score += 1
                
            # Volume conditions - using safe rolling calculations
            try:
                if len(short_vol) >= 20:
                    short_vol_ma = short_vol.rolling(window=20).mean()
                    if short_vol.iloc[-1] > short_vol_ma.iloc[-1] * 1.2:  # Volume spike
                        scalping_score += 1
            except Exception as e:
                self.logger.warning(f"Error in short volume calculation: {e}")
                
            try:
                if len(long_vol) >= 5:
                    long_vol_ma = long_vol.rolling(window=5).mean()
                    if long_vol.iloc[-1] > long_vol_ma.iloc[-1] * 1.1:  # Steady volume increase
                        swing_score += 1
            except Exception as e:
                self.logger.warning(f"Error in long volume calculation: {e}")
                
            # Trend strength conditions
            if short_trend_strength > 25:  # Strong short-term trend
                scalping_score += 1
            if long_trend_strength > 20:  # Strong long-term trend
                swing_score += 1
                
            # Bollinger Band conditions - check if we have valid Series
            try:
                if isinstance(short_bb_width, pd.Series) and len(short_bb_width) > 20:
                    short_bb_ma = short_bb_width.rolling(window=20).mean()
                    if short_bb_width.iloc[-1] > short_bb_ma.iloc[-1] * 1.2:  # Expanding bands
                        scalping_score += 1
            except Exception as e:
                self.logger.warning(f"Error in short BB width calculation: {e}")
                
            try:
                if isinstance(long_bb_width, pd.Series) and len(long_bb_width) > 20:
                    long_bb_ma = long_bb_width.rolling(window=20).mean()
                    if long_bb_width.iloc[-1] < long_bb_ma.iloc[-1] * 0.8:  # Contracting bands
                        swing_score += 1
            except Exception as e:
                self.logger.warning(f"Error in long BB width calculation: {e}")
                
            # Make decision based on scores
            if scalping_score > swing_score:
                self.logger.info(f"Selected SCALPING for {symbol} (Score: {scalping_score} vs {swing_score})")
                return "scalping"
            else:
                self.logger.info(f"Selected SWING for {symbol} (Score: {swing_score} vs {scalping_score})")
                return "swing"
                
        except Exception as e:
            self.logger.error(f"Error in strategy selection for {symbol}: {str(e)}")
            self.logger.error(traceback.format_exc())
            # Default to swing trading in case of errors (more conservative)
            return "swing"
    
    def _calculate_atr(self, data, period=14):
        """Calculate Average True Range."""
        try:
            if data.empty:
                return 0.0
                
            atr_indicator = ta.volatility.AverageTrueRange(
                high=data['high'],
                low=data['low'],
                close=data['close'],
                window=period,
                fillna=True  # Fill NaN values
            )
            atr = atr_indicator.average_true_range()
            
            # Check for NaN or invalid values
            if atr.empty or pd.isna(atr.iloc[-1]):
                return 0.0
                
            return atr.iloc[-1]
        except Exception as e:
            self.logger.warning(f"Error calculating ATR: {e}")
            return 0.0
    
    def _calculate_adx(self, data, period=14):
        """Calculate Average Directional Index as trend strength indicator."""
        try:
            if data.empty:
                return 0.0
                
            # Make sure we have enough data points to calculate ADX
            if len(data) < period * 2:
                return 0.0
                
            # Handle potential division by zero in ADX calculation
            adx_indicator = ta.trend.ADXIndicator(
                high=data['high'],
                low=data['low'],
                close=data['close'],
                window=period,
                fillna=True  # Fill NaN values
            )
            
            with np.errstate(divide='ignore', invalid='ignore'):
                adx = adx_indicator.adx()
            
            # Check for NaN or invalid values
            if adx.empty or pd.isna(adx.iloc[-1]):
                return 0.0
                
            return adx.iloc[-1]
        except Exception as e:
            self.logger.warning(f"Error calculating ADX: {e}")
            return 0.0
            
    def _calculate_bb_width(self, data, period=20, std=2.0):
        """Calculate Bollinger Band width."""
        try:
            if data.empty or len(data) < period:
                return pd.Series([0.0] * len(data), index=data.index)
                
            bb_indicator = ta.volatility.BollingerBands(
                close=data['close'],
                window=period,
                window_dev=std,
                fillna=True  # Fill NaN values
            )
            upper = bb_indicator.bollinger_hband()
            middle = bb_indicator.bollinger_mavg()
            lower = bb_indicator.bollinger_lband()
            
            # Calculate width relative to middle band with handling for division by zero
            with np.errstate(divide='ignore', invalid='ignore'):
                width = (upper - lower) / middle
                
            # Replace inf/NaN values with 0
            width = width.replace([np.inf, -np.inf, np.nan], 0.0)
            
            return width
        except Exception as e:
            self.logger.warning(f"Error calculating Bollinger Band width: {e}")
            # Return a series of zeros with the same index as data
            if not data.empty:
                return pd.Series([0.0] * len(data), index=data.index)
            return pd.Series([0.0])
