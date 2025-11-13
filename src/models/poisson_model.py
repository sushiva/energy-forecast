"""
Poisson Model Module
Implements PoissonRegressor for non-negative predictions
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
from pathlib import Path
from typing import Tuple, Optional


class PoissonModel:
    """
    Poisson Regression model for non-negative predictions.
    
    PoissonRegressor uses a log link function internally, which automatically
    ensures predictions are always positive. It's designed for count/rate data
    and assumes variance equals the mean.
    
    Attributes:
        model: Scikit-learn PoissonRegressor model
        is_trained: Whether the model has been trained
    """
    
    def __init__(self, alpha: float = 1.0, max_iter: int = 300):
        """
        Initialize Poisson regression model
        
        Parameters:
        -----------
        alpha : float, default=1.0
            Regularization strength (L2 penalty)
        max_iter : int, default=300
            Maximum number of iterations for optimization
        """
        self.model = PoissonRegressor(alpha=alpha, max_iter=max_iter)
        self.is_trained = False
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> dict:
        """
        Train the Poisson regression model
        
        Parameters:
        -----------
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training targets (must be non-negative)
            
        Returns:
        --------
        dict
            Training metrics and information
        """
        print("Starting Poisson model training...")
        print(f"Training samples: {len(X_train)}")
        
        if isinstance(y_train, pd.Series):
            y_train_array = y_train.values
        else:
            y_train_array = y_train
        
        if np.any(y_train_array < 0):
            raise ValueError("PoissonRegressor requires non-negative target values")
        
        self.model.fit(X_train, y_train)
        
        self.is_trained = True
        
        y_train_pred = self.model.predict(X_train)
        
        metrics = self._calculate_metrics(y_train, y_train_pred)
        
        print("Training completed successfully")
        print(f"Training R2 Score: {metrics['r2']:.4f}")
        print(f"Training RMSE: {metrics['rmse']:.4f}")
        
        return metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions (guaranteed non-negative)
        
        Parameters:
        -----------
        X : pd.DataFrame
            Features for prediction
            
        Returns:
        --------
        np.ndarray
            Predictions (automatically non-negative due to log link)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions. "
                           "Call train() first.")
        
        y_pred = self.model.predict(X)
        
        return y_pred
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """
        Evaluate model performance on test data
        
        Parameters:
        -----------
        X_test : pd.DataFrame
            Test features
        y_test : pd.Series
            Test targets
            
        Returns:
        --------
        dict
            Evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation. "
                           "Call train() first.")
        
        y_pred = self.predict(X_test)
        
        metrics = self._calculate_metrics(y_test, y_pred)
        
        print("\nTest Set Evaluation:")
        print(f"R2 Score: {metrics['r2']:.4f}")
        print(f"RMSE: {metrics['rmse']:.4f}")
        print(f"MAE: {metrics['mae']:.4f}")
        print(f"MAPE: {metrics['mape']:.2f}%")
        
        return metrics
    
    def _calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> dict:
        """
        Calculate regression metrics
        
        Parameters:
        -----------
        y_true : pd.Series
            True values
        y_pred : np.ndarray
            Predicted values
            
        Returns:
        --------
        dict
            Dictionary of metrics
        """
        if isinstance(y_true, pd.Series):
            y_true = y_true.values
        
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        metrics = {
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2),
            'mape': float(mape)
        }
        
        return metrics
    
    def get_coefficients(self) -> dict:
        """
        Get model coefficients and intercept
        
        Returns:
        --------
        dict
            Dictionary with coefficients and intercept
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        return {
            'intercept': float(self.model.intercept_),
            'coefficients': self.model.coef_.tolist()
        }
    
    def save(self, filepath: str):
        """
        Save model to disk
        
        Parameters:
        -----------
        filepath : str
            Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': self.model,
            'is_trained': self.is_trained
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to: {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'PoissonModel':
        """
        Load model from disk
        
        Parameters:
        -----------
        filepath : str
            Path to the saved model
            
        Returns:
        --------
        PoissonModel
            Loaded model instance
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        instance = cls()
        instance.model = model_data['model']
        instance.is_trained = model_data['is_trained']
        
        print(f"Model loaded from: {filepath}")
        
        return instance


def train_poisson_model(X_train: pd.DataFrame, y_train: pd.Series,
                       X_test: pd.DataFrame, y_test: pd.Series,
                       save_path: Optional[str] = None) -> Tuple[PoissonModel, dict]:
    """
    Convenience function to train and evaluate Poisson model
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training targets
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        Test targets
    save_path : str, optional
        Path to save the trained model
        
    Returns:
    --------
    Tuple[PoissonModel, dict]
        Trained model and evaluation metrics
    """
    model = PoissonModel()
    
    train_metrics = model.train(X_train, y_train)
    
    test_metrics = model.evaluate(X_test, y_test)
    
    results = {
        'train_metrics': train_metrics,
        'test_metrics': test_metrics
    }
    
    if save_path:
        model.save(save_path)
    
    return model, results