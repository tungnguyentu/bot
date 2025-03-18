import logging
import sys

sys.path.append('/Users/tungnt/Downloads/game')
import config
from strategies.scalping import ScalpingStrategy
from strategies.swing import SwingStrategy
from data.market_data import MarketData

logger = logging.getLogger(__name__)

class StrategySwitcher:
    def __init__(self, market_data):
        """
        Initialize the strategy switcher.
        
        Args:
            market_data: MarketData instance for accessing market information
        """
        self.market_data = market_data
        self.scalping_strategy = ScalpingStrategy()
        self.swing_strategy = SwingStrategy()
        self.current_strategy = self.scalping_strategy  # Default strategy
        
    def determine_best_strategy(self, symbol):
        """
        Determine the most appropriate trading strategy based on current market conditions.
        
        Args:
            symbol: The trading symbol (e.g., 'BTCUSDT')
            
        Returns:
            The most suitable strategy instance
        """
        # Calculate market volatility to determine strategy
        volatility = self.market_data.calculate_volatility(symbol, interval='1h', window=24)
        
        # Log the detected volatility
        logger.info(f"Current volatility for {symbol}: {volatility:.4f}")
        
        if volatility > config.SWITCH_STRATEGY_VOLATILITY:
            # Market is volatile - use scalping for quick entries and exits
            if not isinstance(self.current_strategy, ScalpingStrategy):
                logger.info(f"Switching to Scalping strategy due to high volatility ({volatility:.4f})")
                self.current_strategy = self.scalping_strategy
        else:
            # Market is less volatile - use swing trading for longer positions
            if not isinstance(self.current_strategy, SwingStrategy):
                logger.info(f"Switching to Swing Trading strategy due to lower volatility ({volatility:.4f})")
                self.current_strategy = self.swing_strategy
                
        return self.current_strategy
        
    def get_strategy_params(self):
        """
        Get parameters for the current strategy.
        
        Returns:
            dict: Strategy parameters including timeframe
        """
        return {
            "name": self.current_strategy.name,
            "timeframe": self.current_strategy.timeframe,
            "description": self.current_strategy.description
        }
