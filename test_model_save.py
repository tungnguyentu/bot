import os
import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import legacy as legacy_optimizers

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_model_save")

def create_model(input_shape=(30, 46), output_shape=1):
    """Create a simple LSTM model."""
    model = Sequential()
    
    # LSTM layers
    model.add(LSTM(units=64, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    
    model.add(LSTM(units=64, return_sequences=True))
    model.add(Dropout(0.2))
    
    model.add(LSTM(units=64, return_sequences=False))
    model.add(Dropout(0.2))
    
    # Output layer
    model.add(Dense(units=output_shape))
    
    # Compile model with legacy Adam optimizer
    model.compile(
        optimizer=legacy_optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return model

def test_save_load():
    """Test saving and loading a model."""
    # Create a model
    model = create_model()
    
    # Create some dummy data
    X = np.random.random((100, 30, 46))
    y = np.random.random((100, 1))
    
    # Train the model
    model.fit(X, y, epochs=1, batch_size=32, verbose=1)
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Try saving in different formats
    try:
        # Try saving in the newer Keras format
        model_path_keras = 'models/test_model.keras'
        model.save(model_path_keras)
        logger.info(f"Model saved to {model_path_keras}")
        
        # Try loading the model
        loaded_model = tf.keras.models.load_model(model_path_keras, compile=False)
        # Recompile with legacy optimizer
        loaded_model.compile(
            optimizer=legacy_optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        logger.info(f"Model loaded from {model_path_keras}")
        
        # Make a prediction
        pred = loaded_model.predict(X[:1])
        logger.info(f"Prediction: {pred}")
        
    except Exception as e:
        logger.error(f"Error with .keras format: {e}")
        
        try:
            # Try saving in the older H5 format
            model_path_h5 = 'models/test_model.h5'
            model.save(model_path_h5)
            logger.info(f"Model saved to {model_path_h5}")
            
            # Try loading the model
            loaded_model = tf.keras.models.load_model(model_path_h5, compile=False)
            # Recompile with legacy optimizer
            loaded_model.compile(
                optimizer=legacy_optimizers.Adam(learning_rate=0.001),
                loss='mse',
                metrics=['mae']
            )
            logger.info(f"Model loaded from {model_path_h5}")
            
            # Make a prediction
            pred = loaded_model.predict(X[:1])
            logger.info(f"Prediction: {pred}")
            
        except Exception as e:
            logger.error(f"Error with .h5 format: {e}")
            
            try:
                # Try saving weights only
                weights_path = 'models/test_model_weights'
                model.save_weights(weights_path)
                logger.info(f"Model weights saved to {weights_path}")
                
                # Create a new model
                new_model = create_model()
                
                # Load weights
                new_model.load_weights(weights_path)
                logger.info(f"Model weights loaded to new model")
                
                # Make a prediction
                pred = new_model.predict(X[:1])
                logger.info(f"Prediction: {pred}")
                
            except Exception as e:
                logger.error(f"Error with weights: {e}")

if __name__ == "__main__":
    test_save_load() 