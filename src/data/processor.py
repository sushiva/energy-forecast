"""
Data Processor Module
Handles data transformations including log transformation for targets
"""

import numpy as np
import pandas as pd
from typing import Union


class LogTargetTransformer:
    """
    Transformer for applying logarithmic transformation to target variable.
    
    This ensures predictions are always non-negative by transforming the target
    to log space during training and back-transforming predictions.
    
    Methods:
        fit_transform: Transform target values to log space for training
        inverse_transform: Transform predictions back to original scale
    """
    
    def __init__(self):
        """Initialize the log transformer"""
        self.is_fitted = False
    
    def fit_transform(self, y: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """
        Transform target variable to log space using log1p (log(1 + y))
        
        Parameters:
        -----------
        y : array-like
            Target values to transform
            
        Returns:
        --------
        np.ndarray
            Log-transformed target values
            
        Notes:
        ------
        Uses log1p instead of log to handle zero values gracefully:
        log1p(y) = log(1 + y)
        """
        if isinstance(y, pd.Series):
            y = y.values
        
        if np.any(y < 0):
            raise ValueError("Target variable contains negative values. "
                           "Please clean the data before transformation.")
        
        y_log = np.log1p(y)
        self.is_fitted = True
        
        return y_log
    
    def inverse_transform(self, y_pred: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """
        Transform predictions back to original scale using expm1 (exp(y) - 1)
        
        Parameters:
        -----------
        y_pred : array-like
            Predictions in log space
            
        Returns:
        --------
        np.ndarray
            Predictions in original scale (guaranteed non-negative)
            
        Notes:
        ------
        Uses expm1 which is the inverse of log1p:
        expm1(y) = exp(y) - 1
        This mathematically guarantees positive predictions.
        """
        if not self.is_fitted:
            raise ValueError("Transformer has not been fitted. "
                           "Call fit_transform first.")
        
        if isinstance(y_pred, pd.Series):
            y_pred = y_pred.values
        
        y_original = np.expm1(y_pred)
        
        return y_original


class DataValidator:
    """
    Validate data quality before model training
    """
    
    @staticmethod
    def check_missing_values(df: pd.DataFrame) -> dict:
        """
        Check for missing values in dataframe
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
            
        Returns:
        --------
        dict
            Dictionary with column names and missing value counts
        """
        missing = df.isnull().sum()
        missing_dict = missing[missing > 0].to_dict()
        
        return missing_dict
    
    @staticmethod
    def check_target_values(y: Union[np.ndarray, pd.Series], 
                           target_name: str = 'target') -> dict:
        """
        Validate target variable
        
        Parameters:
        -----------
        y : array-like
            Target values
        target_name : str
            Name of target variable for reporting
            
        Returns:
        --------
        dict
            Validation report with statistics
        """
        if isinstance(y, pd.Series):
            y = y.values
        
        report = {
            'target_name': target_name,
            'n_samples': len(y),
            'n_missing': np.isnan(y).sum(),
            'n_negative': (y < 0).sum(),
            'min_value': float(np.nanmin(y)),
            'max_value': float(np.nanmax(y)),
            'mean_value': float(np.nanmean(y)),
            'std_value': float(np.nanstd(y))
        }
        
        return report
    
    @staticmethod
    def validate_for_training(X: pd.DataFrame, y: Union[np.ndarray, pd.Series]) -> bool:
        """
        Comprehensive validation before training
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix
        y : array-like
            Target variable
            
        Returns:
        --------
        bool
            True if validation passes, raises error otherwise
        """
        missing = DataValidator.check_missing_values(X)
        if missing:
            raise ValueError(f"Features contain missing values: {missing}")
        
        if isinstance(y, pd.Series):
            y_array = y.values
        else:
            y_array = y
            
        if np.any(np.isnan(y_array)):
            raise ValueError("Target variable contains missing values")
        
        if np.any(y_array < 0):
            print(f"Warning: Target contains {(y_array < 0).sum()} negative values")
        
        return True


def preprocess_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Basic feature preprocessing
    
    Parameters:
    -----------
    X : pd.DataFrame
        Feature matrix
        
    Returns:
    --------
    pd.DataFrame
        Preprocessed features
    """
    X_processed = X.copy()
    
    return X_processed