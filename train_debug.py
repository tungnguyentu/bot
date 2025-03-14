import os
import logging
import argparse
from config import Config
from data.collector import BinanceDataCollector
from models.model_manager import ModelManager
from utils.telegram_notifier import TelegramNotifier
from stable_baselines3 import PPO
from models.trading_env import TradingEnvironment
from stable_baselines3.common.vec_env import DummyVecEnv
import pandas as pd

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,  # Use DEBUG level for more detailed info
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("debug.log"), logging.StreamHandler()],
)
logger = logging.getLogger("debug")

def main():
    parser = argparse.ArgumentParser(description="Debug RL model loading and saving")
    parser.add_argument(
        "--action",
        choices=["train", "verify", "test"],
        default="verify",
        help="Debug action to perform",
    )
    args = parser.parse_args()

    config = Config()
    logger.info(f"Running debug action: {args.action}")
    
    try:
        # Check model directory and files
        model_dir = config.model_dir
        logger.info(f"Model directory: {model_dir}")
        os.makedirs(model_dir, exist_ok=True)
        
        rl_model_path = os.path.join(model_dir, f"{config.rl_model_name}")
        rl_zip_path = f"{rl_model_path}.zip"
        
        if os.path.exists(rl_zip_path):
            logger.info(f"Found RL model file: {rl_zip_path}")
            logger.info(f"File size: {os.path.getsize(rl_zip_path)} bytes")
            logger.info(f"File permissions: {oct(os.stat(rl_zip_path).st_mode)[-3:]}")
        else:
            logger.warning(f"RL model file not found at {rl_zip_path}")
        
        if args.action == "train":
            # Train a new model for testing
            logger.info("Training a new test model")
            
            # Get sample data
            data_collector = BinanceDataCollector(
                api_key=config.binance_api_key,
                api_secret=config.binance_api_secret,
                symbol="BTCUSDT",
                interval="1h",
            )
            
            historical_data = data_collector.get_historical_data()
            logger.info(f"Got historical data: {len(historical_data)} rows")
            
            # Create training environment
            env = TradingEnvironment(historical_data, config)
            env = DummyVecEnv([lambda: env])
            
            # Create and train the model
            model = PPO(
                "MlpPolicy",
                env,
                policy_kwargs=dict(net_arch=[64, 64]),
                verbose=1,
                learning_rate=0.0001,
                gamma=0.99,
                n_steps=2048,
                ent_coef=0.01,
                clip_range=0.2,
                batch_size=64
            )
            
            logger.info("Training model for 1000 timesteps...")
            model.learn(total_timesteps=1000)
            
            # Save model
            save_path = os.path.join(model_dir, "test_model")
            model.save(save_path)
            logger.info(f"Model saved to {save_path}.zip")
            
            # Verify it was saved
            if os.path.exists(f"{save_path}.zip"):
                logger.info(f"Verified: Model file created at {save_path}.zip")
                logger.info(f"File size: {os.path.getsize(f'{save_path}.zip')} bytes")
            else:
                logger.error("Failed to create model file")
            
        elif args.action == "verify":
            # Verify that existing models can be loaded
            logger.info("Verifying model loading...")
            
            # Create a minimal dataframe for testing
            test_data = pd.DataFrame({
                col: [1.0] * 100 for col in config.features
            })
            test_data.index = pd.date_range('2022-01-01', periods=100, freq='H')
            
            # Try to load the main RL model
            if os.path.exists(rl_zip_path):
                try:
                    logger.info(f"Trying to load model from {rl_zip_path}")
                    env = DummyVecEnv([lambda: TradingEnvironment(test_data, config)])
                    model = PPO.load(rl_model_path, env=env)
                    logger.info("Successfully loaded main model")
                    
                    # Test that predict works
                    obs = env.reset()
                    action, _ = model.predict(obs)
                    logger.info(f"Model prediction test successful, action: {action}")
                except Exception as e:
                    logger.error(f"Error loading main model: {e}", exc_info=True)
            
        elif args.action == "test":
            # Test the full model manager
            logger.info("Testing ModelManager...")
            model_manager = ModelManager(config)
            
            # Check if models were loaded
            if model_manager.rl_model is not None:
                logger.info("RL model was loaded successfully")
            else:
                logger.warning("RL model was not loaded")
            
            if model_manager.xgb_model is not None:
                logger.info("XGBoost model was loaded successfully")
            else:
                logger.warning("XGBoost model was not loaded")
    
    except Exception as e:
        logger.error(f"Error during debug operation: {e}", exc_info=True)

if __name__ == "__main__":
    main()
