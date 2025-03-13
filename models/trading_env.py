import gym
import numpy as np
import pandas as pd
from gym import spaces

class TradingEnvironment(gym.Env):
    """Custom Trading Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, data, config, initial_state=None):
        super(TradingEnvironment, self).__init__()
        self.data = data
        self.config = config
        self.initial_balance = config.initial_balance
        self.position_size = config.position_size_percent
        self.leverage = config.initial_leverage
        self.commission = 0.0004  # 0.04% per trade
        
        # Current step in the data
        self.current_step = 0
        self.max_steps = len(data) - 1
        
        # Portfolio state
        self.balance = self.initial_balance
        self.position = 0  # 1: long, -1: short, 0: no position
        self.position_price = 0
        self.position_size_usd = 0
        self.realized_pnl = 0
        self.unrealized_pnl = 0
        
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
        feature_count = len(config.features) + 4  # features + balance, position, unrealized_pnl, market_position
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(feature_count,), dtype=np.float32)

    def reset(self):
        # Reset environment to the start
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0
        self.position_price = 0
        self.position_size_usd = 0
        self.realized_pnl = 0
        self.unrealized_pnl = 0
        self.trades = []
        self.equity_curve = []
        
        return self._get_observation()

    def step(self, action):
        # Move to the next step
        self.current_step += 1
        
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
            reward += pnl  # Reward is the PnL
            
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
            reward -= commission  # Commission is a negative reward
        
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
            reward -= commission  # Commission is a negative reward
        
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
        """Get current observation from market data and account"""
        # Get current market data features
        features = self.data.iloc[self.current_step][self.config.features].values
        
        # Normalize price-based features by division with current close price
        close_price = self.data['close'].iloc[self.current_step]
        for i, feature in enumerate(self.config.features):
            if feature in ['close', 'open', 'high', 'low', 'bollinger_upper', 'bollinger_middle', 'bollinger_lower', 'vwap']:
                features[i] /= close_price
        
        # Add account state: normalized balance, position, unrealized_pnl, market_position
        norm_balance = self.balance / self.initial_balance
        norm_unrealized_pnl = self.unrealized_pnl / self.initial_balance
        
        # Market position features
        if self.position != 0:
            entry_price_ratio = self.position_price / close_price
            position_duration = min(self.position_duration / 100, 1)  # Normalize
        else:
            entry_price_ratio = 1.0
            position_duration = 0
        
        account_state = np.array([norm_balance, self.position, norm_unrealized_pnl, position_duration])
        
        # Concatenate market features and account state
        obs = np.concatenate([features, account_state])
        
        return obs.astype(np.float32)

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
