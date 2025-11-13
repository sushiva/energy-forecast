"""
Data Splitter Module
Handles train/test splitting with various strategies
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional


class DataSplitter:
    """
    Handle data splitting for training and evaluation
    """
    
    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        """
        Initialize the data splitter
        
        Parameters:
        -----------
        test_size : float, default=0.2
            Proportion of data to use for testing (0.0 to 1.0)
        random_state : int, default=42
            Random seed for reproducibility
        """
        if not 0.0 < test_size < 1.0:
            raise ValueError("test_size must be between 0.0 and 1.0")
        
        self.test_size = test_size
        self.random_state = random_state
    
    def split(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, 
                                                              pd.Series, pd.Series]:
        """
        Split data into training and test sets
        
        Parameters:
        -----------
        X : pd.DataFrame
            Features
        y : pd.Series
            Target variable
            
        Returns:
        --------
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
            X_train, X_test, y_train, y_test
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.test_size, 
            random_state=self.random_state,
            shuffle=True
        )
        
        print(f"Data split completed:")
        print(f"  Train samples: {len(X_train)} ({(1-self.test_size)*100:.0f}%)")
        print(f"  Test samples: {len(X_test)} ({self.test_size*100:.0f}%)")
        print(f"  Total samples: {len(X)}")
        
        return X_train, X_test, y_train, y_test


class TimeSeriesSplitter:
    """
    Handle time series data splitting (no shuffling)
    """
    
    def __init__(self, test_size: float = 0.2):
        """
        Initialize the time series splitter
        
        Parameters:
        -----------
        test_size : float, default=0.2
            Proportion of data to use for testing (from the end)
        """
        if not 0.0 < test_size < 1.0:
            raise ValueError("test_size must be between 0.0 and 1.0")
        
        self.test_size = test_size
    
    def split(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame,
                                                              pd.Series, pd.Series]:
        """
        Split time series data without shuffling
        
        Parameters:
        -----------
        X : pd.DataFrame
            Features (in chronological order)
        y : pd.Series
            Target variable (in chronological order)
            
        Returns:
        --------
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
            X_train, X_test, y_train, y_test
        """
        split_idx = int(len(X) * (1 - self.test_size))
        
        X_train = X.iloc[:split_idx].copy()
        X_test = X.iloc[split_idx:].copy()
        y_train = y.iloc[:split_idx].copy()
        y_test = y.iloc[split_idx:].copy()
        
        print(f"Time series split completed (no shuffling):")
        print(f"  Train samples: {len(X_train)} ({(1-self.test_size)*100:.0f}%)")
        print(f"  Test samples: {len(X_test)} ({self.test_size*100:.0f}%)")
        print(f"  Total samples: {len(X)}")
        
        return X_train, X_test, y_train, y_test


def create_train_val_test_split(X: pd.DataFrame, y: pd.Series,
                                val_size: float = 0.1,
                                test_size: float = 0.2,
                                random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame,
                                                                  pd.DataFrame, pd.Series,
                                                                  pd.Series, pd.Series]:
    """
    Create train, validation, and test splits
    
    Parameters:
    -----------
    X : pd.DataFrame
        Features
    y : pd.Series
        Target variable
    val_size : float, default=0.1
        Proportion for validation set
    test_size : float, default=0.2
        Proportion for test set
    random_state : int, default=42
        Random seed
        
    Returns:
    --------
    Tuple
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    val_size_adjusted = val_size / (1 - test_size)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state
    )
    
    print(f"Three-way split completed:")
    print(f"  Train samples: {len(X_train)} ({(1-val_size-test_size)*100:.0f}%)")
    print(f"  Validation samples: {len(X_val)} ({val_size*100:.0f}%)")
    print(f"  Test samples: {len(X_test)} ({test_size*100:.0f}%)")
    print(f"  Total samples: {len(X)}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def get_split_info(X_train: pd.DataFrame, X_test: pd.DataFrame,
                   y_train: pd.Series, y_test: pd.Series) -> dict:
    """
    Get information about the data split
    
    Parameters:
    -----------
    X_train, X_test : pd.DataFrame
        Training and test features
    y_train, y_test : pd.Series
        Training and test targets
        
    Returns:
    --------
    dict
        Dictionary with split statistics
    """
    info = {
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'total_samples': len(X_train) + len(X_test),
        'n_features': X_train.shape[1],
        'train_target_mean': float(y_train.mean()),
        'test_target_mean': float(y_test.mean()),
        'train_target_std': float(y_train.std()),
        'test_target_std': float(y_test.std()),
        'train_target_min': float(y_train.min()),
        'train_target_max': float(y_train.max()),
        'test_target_min': float(y_test.min()),
        'test_target_max': float(y_test.max())
    }
    
    return info