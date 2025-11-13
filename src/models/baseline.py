"""
Baseline Model Module
Implements baseline Linear Regression with log transformation
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
from pathlib import Path
from typing import Tuple, Optional

import sys
sys.path.insert(0, '/home/claude')
from src.data.processor import LogTargetTransformer


class BaselineModel:
    """
    Baseline Linear Regression model with log transformation.
    
    This model ensures non-negative predictions by training on log-transformed
    targets and automatically converting predictions back to original scale.
    
    Attributes:
        model: Scikit-learn LinearRegression model
        transformer: LogTargetTransformer for target transformation
        is_trained: Whether the model has been trained
    """
    
    def __init__(self):
        """Initialize baseline model with linear regression and log transformer"""
        self.model = LinearRegression()
        self.transformer = LogTargetTransformer()
        self.is_trained = False
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> dict:
        """
        Train the model on log-transformed targets
        
        Parameters:
        -----------
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training targets (in original scale)
            
        Returns:
        --------
        dict
            Training metrics and information
        """
        print("Starting baseline model training...")
        print(f"Training samples: {len(X_train)}")
        
        y_train_log = self.transformer.fit_transform(y_train)
        
        self.model.fit(X_train, y_train_log)
        
        self.is_trained = True
        
        y_train_pred_log = self.model.predict(X_train)
        y_train_pred = self.transformer.inverse_transform(y_train_pred_log)
        
        metrics = self._calculate_metrics(y_train, y_train_pred)
        
        print("Training completed successfully")
        print(f"Training R2 Score: {metrics['r2']:.4f}")
        print(f"Training RMSE: {metrics['rmse']:.4f}")
        
        return metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions in original scale
        
        Parameters:
        -----------
        X : pd.DataFrame
            Features for prediction
            
        Returns:
        --------
        np.ndarray
            Predictions in original scale (guaranteed non-negative)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions. "
                           "Call train() first.")
        
        y_pred_log = self.model.predict(X)
        
        y_pred = self.transformer.inverse_transform(y_pred_log)
        
        return y_pred
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """
        Evaluate model performance on test data
        
        Parameters:
        -----------
        X_test : pd.DataFrame
            Test features
        y_test : pd.Series
            Test targets (in original scale)
            
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
            'transformer': self.transformer,
            'is_trained': self.is_trained
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to: {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'BaselineModel':
        """
        Load model from disk
        
        Parameters:
        -----------
        filepath : str
            Path to the saved model
            
        Returns:
        --------
        BaselineModel
            Loaded model instance
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        instance = cls()
        instance.model = model_data['model']
        instance.transformer = model_data['transformer']
        instance.is_trained = model_data['is_trained']
        
        print(f"Model loaded from: {filepath}")
        
        return instance


def train_baseline_model(X_train: pd.DataFrame, y_train: pd.Series,
                        X_test: pd.DataFrame, y_test: pd.Series,
                        save_path: Optional[str] = None) -> Tuple[BaselineModel, dict]:
    """
    Convenience function to train and evaluate baseline model
    
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
    Tuple[BaselineModel, dict]
        Trained model and evaluation metrics
    """
    model = BaselineModel()
    
    train_metrics = model.train(X_train, y_train)
    
    test_metrics = model.evaluate(X_test, y_test)
    
    results = {
        'train_metrics': train_metrics,
        'test_metrics': test_metrics
    }
    
    if save_path:
        model.save(save_path)
    
    return model, results