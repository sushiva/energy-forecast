"""
Data Loader Module
Handles loading of energy efficiency dataset
"""

import pandas as pd
from pathlib import Path


class EnergyDataLoader:
    """Load energy efficiency dataset from CSV"""
    
    def __init__(self, data_path: str):
        """
        Initialize the data loader
        
        Parameters:
        -----------
        data_path : str
            Path to the CSV data file
        """
        self.data_path = Path(data_path)
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
    
    def load_data(self) -> pd.DataFrame:
        """
        Load the energy efficiency dataset
        
        Returns:
        --------
        pd.DataFrame
            Loaded dataset with all features and targets
        """
        df = pd.read_csv(self.data_path)
        
        print(f"Data loaded successfully: {len(df)} samples, {len(df.columns)} columns")
        
        return df
    
    def get_feature_target_split(self, df: pd.DataFrame, target_col: str = 'Y1'):
        """
        Split dataframe into features and target
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        target_col : str
            Name of target column (default: 'Y1')
            
        Returns:
        --------
        X : pd.DataFrame
            Features
        y : pd.Series
            Target variable
        """
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataset")
        
        feature_cols = [col for col in df.columns if col.startswith('X')]
        
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        
        print(f"Features: {feature_cols}")
        print(f"Target: {target_col}")
        print(f"X shape: {X.shape}, y shape: {y.shape}")
        
        return X, y


def load_energy_data(data_path: str, target: str = 'Y1'):
    """
    Convenience function to load data and split features/target
    
    Parameters:
    -----------
    data_path : str
        Path to CSV file
    target : str
        Target column name (default: 'Y1')
        
    Returns:
    --------
    X : pd.DataFrame
        Features
    y : pd.Series
        Target variable
    """
    loader = EnergyDataLoader(data_path)
    df = loader.load_data()
    X, y = loader.get_feature_target_split(df, target_col=target)
    
    return X, y
