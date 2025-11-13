"""
Evaluation Metrics Module
Comprehensive metrics for regression model evaluation
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_squared_error, 
    mean_absolute_error, 
    r2_score,
    mean_absolute_percentage_error
)
from typing import Dict, Union, Optional


def calculate_rmse(y_true: Union[np.ndarray, pd.Series], 
                   y_pred: np.ndarray) -> float:
    """
    Calculate Root Mean Squared Error
    
    Parameters:
    -----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
        
    Returns:
    --------
    float
        RMSE value
    """
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def calculate_mae(y_true: Union[np.ndarray, pd.Series], 
                  y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Error
    
    Parameters:
    -----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
        
    Returns:
    --------
    float
        MAE value
    """
    return float(mean_absolute_error(y_true, y_pred))


def calculate_mape(y_true: Union[np.ndarray, pd.Series], 
                   y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Percentage Error
    
    Parameters:
    -----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
        
    Returns:
    --------
    float
        MAPE value (as percentage)
    """
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return float(mape)


def calculate_r2(y_true: Union[np.ndarray, pd.Series], 
                 y_pred: np.ndarray) -> float:
    """
    Calculate R² Score
    
    Parameters:
    -----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
        
    Returns:
    --------
    float
        R² value
    """
    return float(r2_score(y_true, y_pred))


def calculate_adjusted_r2(y_true: Union[np.ndarray, pd.Series],
                         y_pred: np.ndarray,
                         n_features: int) -> float:
    """
    Calculate Adjusted R² Score
    
    Parameters:
    -----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
    n_features : int
        Number of features used in the model
        
    Returns:
    --------
    float
        Adjusted R² value
    """
    n = len(y_true)
    r2 = calculate_r2(y_true, y_pred)
    
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - n_features - 1)
    
    return float(adj_r2)


def calculate_all_metrics(y_true: Union[np.ndarray, pd.Series],
                         y_pred: np.ndarray,
                         n_features: Optional[int] = None) -> Dict[str, float]:
    """
    Calculate all regression metrics at once
    
    Parameters:
    -----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
    n_features : int, optional
        Number of features (for adjusted R²)
        
    Returns:
    --------
    Dict[str, float]
        Dictionary containing all metrics
    """
    metrics = {
        'rmse': calculate_rmse(y_true, y_pred),
        'mae': calculate_mae(y_true, y_pred),
        'mape': calculate_mape(y_true, y_pred),
        'r2': calculate_r2(y_true, y_pred)
    }
    
    if n_features is not None:
        metrics['adjusted_r2'] = calculate_adjusted_r2(y_true, y_pred, n_features)
    
    return metrics


def print_metrics(metrics: Dict[str, float], dataset_name: str = "Dataset"):
    """
    Pretty print metrics
    
    Parameters:
    -----------
    metrics : dict
        Dictionary of metrics
    dataset_name : str
        Name of the dataset (e.g., "Training", "Test")
    """
    print(f"\n{dataset_name} Metrics:")
    print("-" * 50)
    print(f"  R² Score:      {metrics['r2']:.4f}")
    if 'adjusted_r2' in metrics:
        print(f"  Adjusted R²:   {metrics['adjusted_r2']:.4f}")
    print(f"  RMSE:          {metrics['rmse']:.4f}")
    print(f"  MAE:           {metrics['mae']:.4f}")
    print(f"  MAPE:          {metrics['mape']:.2f}%")


def compare_metrics(metrics1: Dict[str, float], 
                   metrics2: Dict[str, float],
                   name1: str = "Model 1",
                   name2: str = "Model 2"):
    """
    Compare metrics between two models
    
    Parameters:
    -----------
    metrics1 : dict
        Metrics for first model
    metrics2 : dict
        Metrics for second model
    name1 : str
        Name of first model
    name2 : str
        Name of second model
    """
    print(f"\nModel Comparison: {name1} vs {name2}")
    print("=" * 70)
    print(f"{'Metric':<20} {name1:<20} {name2:<20} {'Winner'}")
    print("-" * 70)
    
    for metric in ['r2', 'rmse', 'mae', 'mape']:
        if metric in metrics1 and metric in metrics2:
            val1 = metrics1[metric]
            val2 = metrics2[metric]
            
            if metric == 'r2':
                winner = name1 if val1 > val2 else name2
            else:
                winner = name1 if val1 < val2 else name2
            
            print(f"{metric.upper():<20} {val1:<20.4f} {val2:<20.4f} {winner}")


def calculate_residuals(y_true: Union[np.ndarray, pd.Series],
                       y_pred: np.ndarray) -> np.ndarray:
    """
    Calculate residuals (errors)
    
    Parameters:
    -----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
        
    Returns:
    --------
    np.ndarray
        Residuals (y_true - y_pred)
    """
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    
    return y_true - y_pred


def calculate_percentage_errors(y_true: Union[np.ndarray, pd.Series],
                               y_pred: np.ndarray) -> np.ndarray:
    """
    Calculate percentage errors for each prediction
    
    Parameters:
    -----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
        
    Returns:
    --------
    np.ndarray
        Percentage errors
    """
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    
    return ((y_true - y_pred) / y_true) * 100


def get_metrics_summary(y_true: Union[np.ndarray, pd.Series],
                       y_pred: np.ndarray) -> pd.DataFrame:
    """
    Get a summary DataFrame of all metrics
    
    Parameters:
    -----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
        
    Returns:
    --------
    pd.DataFrame
        Summary of metrics
    """
    metrics = calculate_all_metrics(y_true, y_pred)
    
    df = pd.DataFrame({
        'Metric': list(metrics.keys()),
        'Value': list(metrics.values())
    })
    
    return df