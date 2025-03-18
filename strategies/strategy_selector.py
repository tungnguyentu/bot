import numpy as np
import pandas as pd
import ta

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
            
            # Calculate volume metrics
            short_vol_avg = short_tf_data['volume'].mean()
            long_vol_avg = long_tf_data['volume'].mean()
            
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
                
            # Volume conditions
            if short_vol_avg > short_vol_avg.shift(20).mean() * 1.2:  # Volume spike
                scalping_score += 1
            if long_vol_avg > long_vol_avg.shift(5).mean() * 1.1:  # Steady volume increase
                swing_score += 1
                
            # Trend strength conditions
            if short_trend_strength > 25:  # Strong short-term trend
                scalping_score += 1
            if long_trend_strength > 20:  # Strong long-term trend
                swing_score += 1
                
            # Bollinger Band conditions
            if short_bb_width > short_bb_width.rolling(20).mean() * 1.2:  # Expanding bands
                scalping_score += 1
            if long_bb_width < long_bb_width.rolling(20).mean() * 0.8:  # Contracting bands
                swing_score += 1
                
            # Make decision based on scores
            if scalping_score > swing_score:
                self.logger.info(f"Selected SCALPING for {symbol} (Score: {scalping_score} vs {swing_score})")
                return "scalping"
            else:
                self.logger.info(f"Selected SWING for {symbol} (Score: {swing_score} vs {scalping_score})")
                return "swing"
                
        except Exception as e:
            self.logger.error(f"Error in strategy selection for {symbol}: {str(e)}")
            # Default to swing trading in case of errors (more conservative)
            return "swing"
    
    def _calculate_atr(self, data, period=14):
        """Calculate Average True Range."""
        try:
            atr_indicator = ta.volatility.AverageTrueRange(
                high=data['high'],
                low=data['low'],
                close=data['close'],
                window=period
            )
            atr = atr_indicator.average_true_range()
            return atr.iloc[-1]
        except:
            return 0
    
    def _calculate_adx(self, data, period=14):
        """Calculate Average Directional Index as trend strength indicator."""
        try:
            adx_indicator = ta.trend.ADXIndicator(
                high=data['high'],
                low=data['low'],
                close=data['close'],
                window=period
            )
            adx = adx_indicator.adx()
            return adx.iloc[-1]
        except:
            return 0
            
    def _calculate_bb_width(self, data, period=20, std=2.0):
        """Calculate Bollinger Band width."""
        try:
            bb_indicator = ta.volatility.BollingerBands(
                close=data['close'],
                window=period,
                window_dev=std
            )
            upper = bb_indicator.bollinger_hband()
            middle = bb_indicator.bollinger_mavg()
            lower = bb_indicator.bollinger_lband()
            
            # Calculate width relative to middle band
            width = (upper - lower) / middle
            return width
        except:
            return pd.Series([0] * len(data))
