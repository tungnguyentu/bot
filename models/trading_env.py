import gym
import numpy as np
import pandas as pd
from gym import spaces
import logging

logger = logging.getLogger(__name__)

class TradingEnvironment(gym.Env):
    """Custom Trading Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, data, config, initial_state=None):
        super(TradingEnvironment, self).__init__()
        self.data = data.copy()
        self.config = config
        self.initial_balance = config.initial_balance
        self.position_size = config.position_size_percent
        self.leverage = config.initial_leverage
        self.commission = 0.0004  # 0.04% per trade
        
        # Drop NaN values from data to avoid issues
        self.data = self.data.dropna()
        
        # Ensure data isn't empty after NaN removal
        if len(self.data) == 0:
            raise ValueError("No valid data after NaN removal")
        
        # Current step in the data
        self.current_step = 0
        self.max_steps = len(self.data) - 1
        
        # Portfolio state
        self.balance = self.initial_balance
        self.position = 0  # 1: long, -1: short, 0: no position
        self.position_price = 0
        self.position_size_usd = 0
        self.realized_pnl = 0
        self.unrealized_pnl = 0
        self.position_duration = 0
        
        # Track performance metrics
        self.trades = []
        self.equity_curve = []
        
        # Override with initial_state if provided
        if initial_state:
            self.balance = initial_state.get('balance', self.balance)
            self.position = initial_state.get('position', self.position)
            self.position_price = initial_state.get('position_price', self.position_price)
            self.position_size_usd = initial_state.get('position_size_usd', self.position_size_usd)
        
        # Define action and observation space
        # Actions: 0 (do nothing), 1 (close position), 2 (open long), 3 (open short)
        self.action_space = spaces.Discrete(4)
        
        # Features + account state
        feature_count = len(config.features) + 4  # features + balance, position, unrealized_pnl, position_duration
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(feature_count,), dtype=np.float32)  # Use bounded values to prevent NaNs

    def reset(self):
        # Reset environment to the start
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0
        self.position_price = 0
        self.position_size_usd = 0
        self.realized_pnl = 0
        self.unrealized_pnl = 0
        self.position_duration = 0
        self.trades = []
        self.equity_curve = []
        
        return self._get_observation()

    def step(self, action):
        # Move to the next step
        self.current_step += 1
        if self.current_step >= len(self.data):
            self.current_step = len(self.data) - 1
        
        # Check if we're done
        done = self.current_step >= self.max_steps
        
        # Get current price data
        current_price = self.data['close'].iloc[self.current_step]
        
        # Calculate profit/loss if we have a position
        if self.position != 0:
            if self.position == 1:  # Long position
                self.unrealized_pnl = (current_price - self.position_price) / self.position_price * self.position_size_usd * self.leverage
            else:  # Short position
                self.unrealized_pnl = (self.position_price - current_price) / self.position_price * self.position_size_usd * self.leverage
        else:
            self.unrealized_pnl = 0
        
        # Process the action
        reward = 0
        
        # Action 0: Do nothing
        if action == 0:
            pass
        
        # Action 1: Close position
        elif action == 1 and self.position != 0:
            # Calculate PnL
            if self.position == 1:  # Close long
                pnl = (current_price - self.position_price) / self.position_price * self.position_size_usd * self.leverage
            else:  # Close short
                pnl = (self.position_price - current_price) / self.position_price * self.position_size_usd * self.leverage
            
            # Subtract commission
            commission = self.position_size_usd * self.commission
            pnl -= commission
            
            # Update balance and metrics
            self.balance += pnl
            self.realized_pnl += pnl
            reward += pnl / self.initial_balance  # Normalize reward to avoid too large values
            
            # Record trade
            self.trades.append({
                'entry_step': self.current_step - self.position_duration,
                'exit_step': self.current_step,
                'entry_price': self.position_price,
                'exit_price': current_price,
                'position': self.position,
                'pnl': pnl,
                'balance_after': self.balance
            })
            
            # Reset position
            self.position = 0
            self.position_price = 0
            self.position_size_usd = 0
            self.position_duration = 0
        
        # Action 2: Open long position
        elif action == 2 and self.position == 0:
            # Calculate position size
            self.position_size_usd = self.balance * self.position_size
            
            # Set position data
            self.position = 1
            self.position_price = current_price
            self.position_duration = 0
            
            # Subtract commission
            commission = self.position_size_usd * self.commission
            self.balance -= commission
            reward -= commission / self.initial_balance  # Normalize reward
        
        # Action 3: Open short position
        elif action == 3 and self.position == 0:
            # Calculate position size
            self.position_size_usd = self.balance * self.position_size
            
            # Set position data
            self.position = -1
            self.position_price = current_price
            self.position_duration = 0
            
            # Subtract commission
            commission = self.position_size_usd * self.commission
            self.balance -= commission
            reward -= commission / self.initial_balance  # Normalize reward
        
        # Update position duration
        if self.position != 0:
            self.position_duration += 1
        
        # Update equity
        equity = self.balance + self.unrealized_pnl
        self.equity_curve.append(equity)
        
        # Get observation
        obs = self._get_observation()
        
        # Include extra info
        info = {
            'balance': self.balance,
            'equity': equity,
            'position': self.position,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl
        }
        
        return obs, reward, done, info

    def _get_observation(self):
        """Get current observation from market data and account with enhanced features"""
        try:
            # Get current market data features
            features = self.data.iloc[self.current_step][self.config.features].values.astype(np.float32)
            
            # Normalize price-based features by division with current close price
            close_price = max(1e-8, float(self.data['close'].iloc[self.current_step]))  # Avoid division by zero
            for i, feature in enumerate(self.config.features):
                if feature in ['close', 'open', 'high', 'low', 'bollinger_upper', 'bollinger_middle', 'bollinger_lower', 'vwap']:
                    features[i] = float(features[i]) / close_price
            
            # Calculate additional momentum features to help identify trading opportunities
            if self.current_step >= 5:
                price_momentum_5 = self.data['close'].iloc[self.current_step] / self.data['close'].iloc[self.current_step - 5] - 1.0
                volume_momentum_5 = self.data['volume'].iloc[self.current_step] / max(1.0, self.data['volume'].iloc[self.current_step - 5]) - 1.0
            else:
                price_momentum_5 = 0.0
                volume_momentum_5 = 0.0
                
            # Add momentum as a feature - normalize and clip to prevent extreme values
            price_momentum_5 = np.clip(float(price_momentum_5) * 10.0, -10.0, 10.0)  # Scale up for visibility
            volume_momentum_5 = np.clip(float(volume_momentum_5) * 5.0, -10.0, 10.0)
            
            # Ensure all features are float32
            features = features.astype(np.float32)
            
            # Clip feature values to prevent extreme values
            features = np.clip(features, -10.0, 10.0)
            
            # Add account state: normalized balance, position, unrealized_pnl, position_duration
            norm_balance = float(self.balance / self.initial_balance)
            norm_unrealized_pnl = float(self.unrealized_pnl / (self.initial_balance + 1e-8))  # Avoid division by zero
            
            # Normalize position duration
            position_duration = float(min(self.position_duration / 100, 1.0))  # Normalize to [0, 1]
            
            # Create account state array with momentum indicators
            account_state = np.array([
                norm_balance,
                float(self.position),  # Convert to float explicitly
                norm_unrealized_pnl,
                position_duration,
                price_momentum_5,  # Add price momentum to help model be more responsive
                volume_momentum_5,  # Add volume momentum
            ], dtype=np.float32)
            
            # Clip account state values
            account_state = np.clip(account_state, -10.0, 10.0)
            
            # Concatenate market features and account state
            obs = np.concatenate([features, account_state[:-2]])  # Skip the momentum features in observation
            
            # Make sure observation is float32
            obs = obs.astype(np.float32)
            
            # Manually check for NaN or infinite values
            for i in range(len(obs)):
                if not np.isfinite(float(obs[i])):
                    logger.warning(f"Non-finite value at index {i}: {obs[i]}, replacing with 0.0")
                    obs[i] = 0.0
            
            return obs
            
        except Exception as e:
            logger.error(f"Error creating observation: {str(e)}")
            # Return a safe default observation
            feature_count = len(self.config.features) + 4  # Back to original size
            return np.zeros(feature_count, dtype=np.float32)

    def render(self, mode='human'):
        """Render the current state of the environment"""
        # Print current state
        current_price = self.data['close'].iloc[self.current_step]
        print(f"Step: {self.current_step}")
        print(f"Price: {current_price:.2f}")
        print(f"Balance: {self.balance:.2f}")
        print(f"Position: {self.position}")
        print(f"Unrealized PnL: {self.unrealized_pnl:.2f}")
        print(f"Realized PnL: {self.realized_pnl:.2f}")
        print("---------------------")

    def get_observation(self):
        """Get the current observation without taking a step"""
        return self._get_observation()
