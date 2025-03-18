import math
import config
from utils.logger import setup_logger

class RiskManager:
    def __init__(self, client):
        self.client = client
        self.logger = setup_logger('risk_manager', 'logs/risk.log')
        
    def can_open_new_position(self):
        """
        Check if a new position can be opened based on risk parameters.
        
        Returns:
            bool: True if a new position can be opened, False otherwise
        """
        try:
            # Check max open trades
            open_positions = self.client.get_open_positions()
            if len(open_positions) >= config.MAX_OPEN_TRADES:
                self.logger.info(f"Max open trades reached: {len(open_positions)}/{config.MAX_OPEN_TRADES}")
                return False
                
            # Get account balance
            account_balance = self.client.get_balance()
            
            # Calculate current exposure
            total_exposure = sum(abs(pos['size'] * pos['entry_price']) for pos in open_positions)
            exposure_ratio = total_exposure / account_balance if account_balance > 0 else float('inf')
            
            # Check if exposure is too high (arbitrary threshold of 80% of account)
            if exposure_ratio > 0.8:
                self.logger.info(f"Total exposure too high: {exposure_ratio:.2%}")
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking if new position can be opened: {str(e)}")
            return False  # Default to not allowing new positions if there's an error
            
    def calculate_position_size(self, symbol, stop_loss_price):
        """
        Calculate position size based on risk per trade.
        
        Args:
            symbol (str): Trading symbol
            stop_loss_price (float): Stop loss price level
            
        Returns:
            float: Position size in base currency units
        """
        try:
            # Get current price and account balance
            current_price = self.client.get_symbol_price(symbol)
            account_balance = self.client.get_balance()
            
            if not current_price or account_balance <= 0:
                self.logger.error(f"Invalid price ({current_price}) or balance ({account_balance})")
                return 0
                
            # Calculate risk amount in USDT
            risk_amount = account_balance * config.RISK_PER_TRADE
            
            # Calculate position size based on stop loss distance
            stop_loss_distance = abs(current_price - stop_loss_price)
            
            if stop_loss_distance <= 0:
                self.logger.error(f"Invalid stop loss distance: {stop_loss_distance}")
                return 0
                
            # Position size = (Risk Amount in USDT / Stop Loss Distance in USDT)
            position_size = risk_amount / stop_loss_distance
            
            # Apply leverage
            position_size = position_size * config.DEFAULT_LEVERAGE
            
            # Convert to quantity (base currency)
            quantity = position_size / current_price
            
            self.logger.info(
                f"Position sizing for {symbol}: "
                f"Balance={account_balance:.2f}, Risk={risk_amount:.2f}, "
                f"SL Distance={stop_loss_distance:.2f}, Quantity={quantity:.4f}"
            )
            
            return quantity
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {str(e)}")
            return 0
            
    def adjust_position_for_correlation(self, symbol, base_position_size):
        """
        Adjust position size based on correlation with existing positions.
        
        Args:
            symbol (str): Trading symbol
            base_position_size (float): Initial calculated position size
            
        Returns:
            float: Adjusted position size
        """
        try:
            # For a proper implementation, we would:
            # 1. Get price data for all open positions and current symbol
            # 2. Calculate correlation matrix
            # 3. If correlation is high, reduce position size
            
            # Simplified implementation:
            # Check if we have other positions in the same base currency
            symbol_base = symbol[:-4]  # Assumes XXXUSDT format
            
            open_positions = self.client.get_open_positions()
            similar_positions = [p for p in open_positions if p['symbol'][:len(symbol_base)] == symbol_base]
            
            if similar_positions:
                # Reduce position size if we already have positions in the same asset
                adjustment_factor = 0.7  # 30% reduction
                adjusted_size = base_position_size * adjustment_factor
                self.logger.info(
                    f"Adjusted position size for {symbol} due to correlation: "
                    f"{base_position_size:.4f} -> {adjusted_size:.4f}"
                )
                return adjusted_size
                
            return base_position_size
            
        except Exception as e:
            self.logger.error(f"Error adjusting position for correlation: {str(e)}")
            return base_position_size  # Return original size in case of error
