import numpy as np
import pandas as pd
import os
import logging
import random
from collections import deque
from tqdm import tqdm

# Import deep learning libraries if available, otherwise warn user
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model, save_model
    from tensorflow.keras.layers import Dense, LSTM, Dropout
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logging.warning("TensorFlow not available. Reinforcement learning features will be limited.")

logger = logging.getLogger("BinanceBot.RLTrader")

class RLTrader:
    """
    Reinforcement Learning based trading agent
    """
    def __init__(self, state_size=5, action_size=3, batch_size=64, episodes=100):
        self.state_size = state_size  # Number of features in state
        self.action_size = action_size  # 0: Hold, 1: Buy/Long, 2: Sell/Short
        self.batch_size = batch_size
        self.episodes = episodes
        
        # RL parameters
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        
        # Replay memory
        self.memory = deque(maxlen=2000)
        
        # Check if TensorFlow is available
        if TENSORFLOW_AVAILABLE:
            # Build model
            self.model = self._build_model()
            self.target_model = self._build_model()
            self.update_target_model()
        else:
            self.model = None
            self.target_model = None
            logger.warning("TensorFlow not available. Using random decision maker instead.")
    
    def _build_model(self):
        """
        Neural Network for Deep Q-learning
        """
        if not TENSORFLOW_AVAILABLE:
            return None
            
        model = Sequential()
        model.add(Dense(64, input_dim=self.state_size, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model
    
    def update_target_model(self):
        """Copy weights from model to target_model"""
        if TENSORFLOW_AVAILABLE:
            self.target_model.set_weights(self.model.get_weights())
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, explore=True):
        """Return action based on current state"""
        if not TENSORFLOW_AVAILABLE:
            # Return random action if TensorFlow is not available
            return random.randrange(self.action_size)
        
        if explore and np.random.rand() <= self.epsilon:
            # Exploration: random action
            return random.randrange(self.action_size)
        
        # Exploitation: use model prediction
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])
    
    def predict_action(self, state):
        """Predict best action for a given state without exploration"""
        if not TENSORFLOW_AVAILABLE:
            # Use simple heuristics if TensorFlow is not available
            # state format: [bb_pos, rsi, macd, vwap_dist, atr]
            # bb_pos is position relative to bollinger bands (-1 to 1)
            # rsi (0-100)
            # macd is histogram value
            # vwap_dist is relative distance from VWAP
            # atr is volatility
            
            # Simple heuristic rules
            bb_pos, rsi, macd, vwap_dist, atr = state
            
            if bb_pos < -0.8 and rsi < 30 and macd > 0:
                return 1  # Buy/Long
            elif bb_pos > 0.8 and rsi > 70 and macd < 0:
                return 2  # Sell/Short
            else:
                return 0  # Hold
        
        # Use model for prediction
        state_array = np.array(state).reshape(1, -1)
        act_values = self.model.predict(state_array, verbose=0)
        return np.argmax(act_values[0])
    
    def replay(self, batch_size):
        """Train model with random batch from replay memory"""
        if not TENSORFLOW_AVAILABLE or len(self.memory) < batch_size:
            return
            
        minibatch = random.sample(self.memory, batch_size)
        
        states = []
        targets = []
        
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state, verbose=0)
            
            if done:
                target[0][action] = reward
            else:
                t = self.target_model.predict(next_state, verbose=0)
                target[0][action] = reward + self.gamma * np.amax(t)
            
            states.append(state[0])
            targets.append(target[0])
        
        history = self.model.fit(np.array(states), np.array(targets), 
                                 epochs=1, verbose=0, batch_size=batch_size)
        
        # Decay epsilon after each training
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return history.history['loss'][0]
    
    def load_model(self, model_path):
        """Load model from file"""
        if TENSORFLOW_AVAILABLE:
            try:
                self.model = load_model(model_path)
                self.target_model = load_model(model_path)
                return True
            except Exception as e:
                logger.error(f"Error loading model: {e}")
                return False
        return False
    
    def save_model(self, model_path):
        """Save model to file"""
        if TENSORFLOW_AVAILABLE and self.model:
            try:
                self.model.save(model_path)
                return True
            except Exception as e:
                logger.error(f"Error saving model: {e}")
                return False
        return False
    
    def train(self, symbol, data, config):
        """Train RL model on historical data"""
        if not TENSORFLOW_AVAILABLE:
            logger.warning("TensorFlow not available. Cannot train RL model.")
            return None
        
        logger.info(f"Training RL model for {symbol} with {len(data)} samples")
        
        # Prepare environment
        env = TradingEnvironment(data, config)
        
        # Training statistics
        total_rewards = []
        losses = []
        
        # Training loop
        for episode in tqdm(range(self.episodes), desc=f"Training RL model for {symbol}"):
            # Reset environment for new episode
            state = env.reset()
            state = np.reshape(state, [1, self.state_size])
            done = False
            episode_reward = 0
            
            # Episode loop
            while not done:
                # Select action
                action = self.act(state, explore=True)
                
                # Take action
                next_state, reward, done = env.step(action)
                next_state = np.reshape(next_state, [1, self.state_size])
                
                # Remember the experience
                self.remember(state, action, reward, next_state, done)
                
                # Update state
                state = next_state
                episode_reward += reward
                
                # Train the model
                if len(self.memory) > self.batch_size:
                    loss = self.replay(self.batch_size)
                    if loss:
                        losses.append(loss)
            
            # Update target model every 10 episodes
            if episode % 10 == 0:
                self.update_target_model()
            
            total_rewards.append(episode_reward)
            
            # Log training progress
            if episode % 10 == 0:
                avg_reward = np.mean(total_rewards[-10:]) if len(total_rewards) >= 10 else np.mean(total_rewards)
                avg_loss = np.mean(losses[-10:]) if len(losses) >= 10 else np.mean(losses) if losses else 0
                logger.info(f"Episode: {episode}/{self.episodes}, Avg Reward: {avg_reward:.2f}, Avg Loss: {avg_loss:.6f}, Epsilon: {self.epsilon:.4f}")
        
        logger.info(f"Training completed for {symbol}")
        return {"avg_reward": np.mean(total_rewards), "total_episodes": self.episodes}
    
    def evaluate(self, symbol, data, config):
        """Evaluate the trained model on test data"""
        if not TENSORFLOW_AVAILABLE or self.model is None:
            logger.warning("TensorFlow or model not available. Cannot evaluate RL model.")
            return None
        
        logger.info(f"Evaluating RL model for {symbol} with {len(data)} samples")
        
        # Prepare environment
        env = TradingEnvironment(data, config)
        
        # Statistics
        trades = []
        returns = []
        positions = []
        state = env.reset()
        state = np.reshape(state, [1, self.state_size])
        done = False
        total_reward = 0
        
        # Step through the test data
        while not done:
            # Predict action
            action = self.predict_action(state[0])
            
            # Take action
            next_state, reward, done = env.step(action)
            next_state = np.reshape(next_state, [1, self.state_size])
            
            # Update statistics
            total_reward += reward
            state = next_state
            
            # Track positions and trades
            if env.position_changed:
                trades.append({
                    'timestamp': env.current_timestamp,
                    'action': ["HOLD", "LONG", "SHORT"][action],
                    'price': env.current_price,
                    'return_pct': env.trade_return_pct
                })
            
            positions.append({
                'timestamp': env.current_timestamp,
                'price': env.current_price,
                'position': env.position,
                'equity': env.equity
            })
        
        # Calculate performance metrics
        total_trades = len(trades)
        winning_trades = len([t for t in trades if t['return_pct'] > 0])
        
        if total_trades > 0:
            win_rate = winning_trades / total_trades * 100
        else:
            win_rate = 0
            
        # Calculate sharpe ratio
        if len(returns) > 1:
            returns_arr = np.array(returns)
            sharpe_ratio = np.mean(returns_arr) / np.std(returns_arr) * np.sqrt(252) if np.std(returns_arr) > 0 else 0
        else:
            sharpe_ratio = 0
        
        # Calculate total return
        if len(positions) > 0:
            total_return_pct = (positions[-1]['equity'] / positions[0]['equity'] - 1) * 100
        else:
            total_return_pct = 0
        
        return {
            'total_reward': total_reward,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_return': total_return_pct,
            'sharpe_ratio': sharpe_ratio
        }


class TradingEnvironment:
    """
    Trading environment for reinforcement learning
    """
    def __init__(self, df, config):
        self.df = df
        self.config = config
        self.position = 0  # -1: short, 0: neutral, 1: long
        self.position_changed = False
        self.current_step = 0
        self.equity = 1000.0  # Starting equity
        self.trade_return_pct = 0
        self.current_price = 0
        self.current_timestamp = None
    
    def reset(self):
        """Reset the environment to the start"""
        self.current_step = 0
        self.position = 0
        self.equity = 1000.0
        self.position_changed = False
        self.trade_return_pct = 0
        return self._get_state()
    
    def step(self, action):
        """Take an action and move to the next step"""
        # Save current price and position for later calculations
        prev_price = self.current_price if self.current_step > 0 else self.df.iloc[0]['close']
        prev_position = self.position
        
        # Action: 0 = hold, 1 = buy/long, 2 = sell/short
        if action == 1:  # Buy/Long
            new_position = 1
        elif action == 2:  # Sell/Short
            new_position = -1
        else:  # Hold
            new_position = self.position
        
        # Update position
        self.position_changed = (new_position != prev_position)
        self.position = new_position
        
        # Move to the next step
        self.current_step += 1
        if self.current_step >= len(self.df):
            return self._get_state(), 0, True  # State, reward, done
        
        # Calculate reward based on price movement and position
        self.current_price = self.df.iloc[self.current_step]['close']
        self.current_timestamp = self.df.iloc[self.current_step]['timestamp'] if 'timestamp' in self.df.columns else self.current_step
        
        # Calculate price change
        price_change = (self.current_price - prev_price) / prev_price
        
        # Calculate reward
        if prev_position == 1:  # Long
            reward = price_change * 100  # Convert to percentage points
            self.trade_return_pct = price_change * 100
        elif prev_position == -1:  # Short
            reward = -price_change * 100  # Short profits when price falls
            self.trade_return_pct = -price_change * 100
        else:  # Neutral
            reward = 0
            self.trade_return_pct = 0
        
        # Update equity
        if prev_position != 0:
            # Apply leverage to returns
            leveraged_return = price_change * self.config.LEVERAGE * prev_position
            self.equity *= (1 + leveraged_return)
        
        # Penalties for excessive trading
        if self.position_changed:
            # Trading fee (simulated)
            trading_fee = self.equity * 0.001  # 0.1% trading fee
            self.equity -= trading_fee
            reward -= 0.1  # Small penalty to discourage excessive trading
        
        # Check if we're done (end of data or bankrupt)
        done = self.current_step >= len(self.df) - 1 or self.equity <= 0
        
        return self._get_state(), reward, done
    
    def _get_state(self):
        """Get the current state representation"""
        if self.current_step >= len(self.df):
            # Return zeros if we've reached the end
            return np.zeros(5)
        
        # Get current row
        current = self.df.iloc[self.current_step]
        
        # Features:
        # 1. BB position: Where is price relative to BB bands? (-1 to 1)
        if 'bb_upper' in current and 'bb_lower' in current and 'close' in current:
            bb_range = current['bb_upper'] - current['bb_lower']
            if bb_range > 0:
                bb_position = 2 * (current['close'] - current['bb_lower']) / bb_range - 1
            else:
                bb_position = 0
        else:
            bb_position = 0
        
        # 2. RSI normalized (0-1)
        rsi = current['rsi'] / 100 if 'rsi' in current else 0.5
        
        # 3. MACD histogram
        macd_hist = current['macd_hist'] if 'macd_hist' in current else 0
        
        # 4. VWAP distance (normalized)
        if 'vwap' in current and 'close' in current:
            vwap_dist = (current['close'] - current['vwap']) / current['close']
        else:
            vwap_dist = 0
        
        # 5. ATR (volatility)
        atr = current['atr'] if 'atr' in current else 0
        
        return np.array([bb_position, rsi, macd_hist, vwap_dist, atr])
