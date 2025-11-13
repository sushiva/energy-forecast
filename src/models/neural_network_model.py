"""
Neural Network model for energy forecasting using TensorFlow/Keras.

This module implements a deep neural network for regression with:
- Multiple hidden layers with ReLU activation
- Feature normalization (StandardScaler)
- Early stopping to prevent overfitting
- Model persistence (save/load)
- Comprehensive evaluation metrics
"""

import numpy as np
import pandas as pd
import pickle
import os
from typing import Dict, Tuple, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
import matplotlib.pyplot as plt


class NeuralNetworkModel:
    """
    Neural Network model for energy consumption prediction.
    
    Architecture:
        - Input Layer: n_features
        - Hidden Layer 1: 64 neurons (ReLU)
        - Hidden Layer 2: 32 neurons (ReLU)
        - Hidden Layer 3: 16 neurons (ReLU)
        - Output Layer: 1 neuron (Linear)
    
    Features:
        - Feature normalization using StandardScaler
        - Early stopping to prevent overfitting
        - Adam optimizer with learning rate 0.001
        - MSE loss function
    """
    
    def __init__(
        self,
        hidden_layers: Tuple[int, ...] = (64, 32, 16),
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epochs: int = 100,
        validation_split: float = 0.2,
        early_stopping_patience: int = 10,
        random_state: int = 42
    ):
        """
        Initialize Neural Network model.
        
        Args:
            hidden_layers: Tuple of neurons in each hidden layer (default: (64, 32, 16))
            learning_rate: Learning rate for Adam optimizer (default: 0.001)
            batch_size: Batch size for training (default: 32)
            epochs: Maximum number of training epochs (default: 100)
            validation_split: Fraction of training data for validation (default: 0.2)
            early_stopping_patience: Epochs to wait before early stopping (default: 10)
            random_state: Random seed for reproducibility (default: 42)
        """
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.validation_split = validation_split
        self.early_stopping_patience = early_stopping_patience
        self.random_state = random_state
        
        # Model components
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.history = None
        
        # Set random seeds for reproducibility
        np.random.seed(random_state)
        tf.random.set_seed(random_state)
    
    def _build_model(self, n_features: int) -> keras.Model:
        """
        Build the neural network architecture.
        
        Args:
            n_features: Number of input features
            
        Returns:
            Compiled Keras model
        """
        model = keras.Sequential(name="EnergyForecastingNN")
        
        # Input layer
        model.add(layers.Input(shape=(n_features,), name="input"))
        
        # Hidden layers
        for i, neurons in enumerate(self.hidden_layers):
            model.add(layers.Dense(
                neurons,
                activation='relu',
                name=f'hidden_{i+1}'
            ))
        
        # Output layer (linear activation for regression)
        model.add(layers.Dense(1, activation='linear', name='output'))
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'NeuralNetworkModel':
        """
        Train the neural network model.
        
        Args:
            X: Training features
            y: Training target
            
        Returns:
            self: Trained model instance
        """
        print("Training Neural Network Model...")
        print(f"Architecture: Input → {' → '.join(map(str, self.hidden_layers))} → Output")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Batch size: {self.batch_size}")
        print(f"Max epochs: {self.epochs}")
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Normalize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Build model
        self.model = self._build_model(n_features=X.shape[1])
        
        # Print model summary
        print("\nModel Architecture:")
        self.model.summary()
        
        # Setup callbacks
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=self.early_stopping_patience,
            restore_best_weights=True,
            verbose=1
        )
        
        # Train model
        print(f"\nTraining with {len(X)} samples...")
        self.history = self.model.fit(
            X_scaled,
            y.values,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_split=self.validation_split,
            callbacks=[early_stop],
            verbose=1
        )
        
        # Calculate training metrics
        y_pred_train = self.predict(X)
        train_r2 = r2_score(y, y_pred_train)
        train_rmse = np.sqrt(mean_squared_error(y, y_pred_train))
        
        print(f"\n✅ Training Complete!")
        print(f"   Training R²: {train_r2:.4f}")
        print(f"   Training RMSE: {train_rmse:.2f} kWh")
        print(f"   Epochs trained: {len(self.history.history['loss'])}")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X: Features for prediction
            
        Returns:
            Predicted energy consumption values
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        # Normalize features
        X_scaled = self.scaler.transform(X)
        
        # Predict
        predictions = self.model.predict(X_scaled, verbose=0)
        
        return predictions.flatten()
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Evaluate model performance on test data.
        
        Args:
            X: Test features
            y: True target values
            
        Returns:
            Dictionary containing evaluation metrics
        """
        # Make predictions
        y_pred = self.predict(X)
        
        # Calculate metrics
        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)
        
        # Calculate MAPE (avoiding division by zero)
        mape = np.mean(np.abs((y - y_pred) / y)) * 100
        
        metrics = {
            'r2_score': r2,
            'rmse': rmse,
            'mae': mae,
            'mape': mape
        }
        
        print("\n📊 Model Evaluation:")
        print(f"   R² Score: {r2:.4f} ({r2*100:.2f}% variance explained)")
        print(f"   RMSE: {rmse:.2f} kWh")
        print(f"   MAE: {mae:.2f} kWh")
        print(f"   MAPE: {mape:.2f}%")
        
        return metrics
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """
        Plot training and validation loss over epochs.
        
        Args:
            save_path: Optional path to save the plot
        """
        if self.history is None:
            raise ValueError("No training history available. Train the model first.")
        
        plt.figure(figsize=(12, 5))
        
        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot MAE
        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['mae'], label='Training MAE')
        plt.plot(self.history.history['val_mae'], label='Validation MAE')
        plt.xlabel('Epoch')
        plt.ylabel('MAE (kWh)')
        plt.title('Training and Validation MAE')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n📊 Training history plot saved to: {save_path}")
        
        plt.show()
    
    def get_feature_importance(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_repeats: int = 10
    ) -> pd.DataFrame:
        """
        Calculate feature importance using permutation importance.
        
        Note: For neural networks, we use permutation importance since
        there's no built-in feature importance like tree-based models.
        
        Args:
            X: Features
            y: Target values
            n_repeats: Number of times to permute each feature
            
        Returns:
            DataFrame with feature importance scores
        """
        print("\n🔍 Calculating feature importance (this may take a moment)...")
        
        # Get baseline performance
        baseline_predictions = self.predict(X)
        baseline_rmse = np.sqrt(mean_squared_error(y, baseline_predictions))
        
        importance_scores = []
        
        for feature in X.columns:
            # Calculate importance by permuting each feature
            feature_importances = []
            
            for _ in range(n_repeats):
                X_permuted = X.copy()
                X_permuted[feature] = np.random.permutation(X_permuted[feature].values)
                
                permuted_predictions = self.predict(X_permuted)
                permuted_rmse = np.sqrt(mean_squared_error(y, permuted_predictions))
                
                # Importance = increase in error when feature is permuted
                importance = permuted_rmse - baseline_rmse
                feature_importances.append(importance)
            
            # Average importance across repeats
            avg_importance = np.mean(feature_importances)
            importance_scores.append({
                'feature': feature,
                'importance': avg_importance
            })
        
        # Create DataFrame and sort
        importance_df = pd.DataFrame(importance_scores)
        importance_df = importance_df.sort_values('importance', ascending=False)
        importance_df = importance_df.reset_index(drop=True)
        
        print("\n📊 Top 10 Most Important Features:")
        for idx, row in importance_df.head(10).iterrows():
            print(f"   {idx+1}. {row['feature']}: {row['importance']:.4f}")
        
        return importance_df
    
    def save_model(self, filepath: str):
        """
        Save the trained model, scaler, and metadata.
        
        Args:
            filepath: Path to save the model (without extension)
        """
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save Keras model
        model_path = f"{filepath}_model.keras"
        self.model.save(model_path)
        
        # Save scaler and metadata
        metadata = {
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'hidden_layers': self.hidden_layers,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'validation_split': self.validation_split,
            'early_stopping_patience': self.early_stopping_patience,
            'random_state': self.random_state
        }
        
        metadata_path = f"{filepath}_metadata.pkl"
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"\n💾 Model saved successfully!")
        print(f"   Model: {model_path}")
        print(f"   Metadata: {metadata_path}")
    
    @classmethod
    def load_model(cls, filepath: str) -> 'NeuralNetworkModel':
        """
        Load a saved model.
        
        Args:
            filepath: Path to the saved model (without extension)
            
        Returns:
            Loaded NeuralNetworkModel instance
        """
        # Load Keras model
        model_path = f"{filepath}_model.keras"
        loaded_keras_model = keras.models.load_model(model_path)
        
        # Load metadata
        metadata_path = f"{filepath}_metadata.pkl"
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        # Create instance
        instance = cls(
            hidden_layers=metadata['hidden_layers'],
            learning_rate=metadata['learning_rate'],
            batch_size=metadata['batch_size'],
            epochs=metadata['epochs'],
            validation_split=metadata['validation_split'],
            early_stopping_patience=metadata['early_stopping_patience'],
            random_state=metadata['random_state']
        )
        
        # Set loaded components
        instance.model = loaded_keras_model
        instance.scaler = metadata['scaler']
        instance.feature_names = metadata['feature_names']
        
        print(f"\n✅ Model loaded successfully from: {filepath}")
        
        return instance


# Example usage
if __name__ == "__main__":
    print("Neural Network Model Module")
    print("=" * 50)
    print("\nThis module implements a deep neural network for energy forecasting.")
    print("\nExample usage:")
    print("""
    from models.neural_network_model import NeuralNetworkModel
    
    # Initialize model
    nn_model = NeuralNetworkModel(
        hidden_layers=(64, 32, 16),
        learning_rate=0.001,
        batch_size=32,
        epochs=100
    )
    
    # Train
    nn_model.fit(X_train, y_train)
    
    # Evaluate
    metrics = nn_model.evaluate(X_test, y_test)
    
    # Predict
    predictions = nn_model.predict(X_test)
    
    # Save
    nn_model.save_model('models/neural_network_model')
    """)