"""
Random Forest Model Module
Ensemble model for energy forecasting
"""

import numpy as np
import pickle
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from src.evaluation.metrics import calculate_all_metrics, print_metrics


class RandomForestModel:
    """
    Random Forest Regressor for energy forecasting
    Handles training, prediction, evaluation, and model persistence
    """
    
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, random_state=42, n_jobs=-1, **kwargs):
        """
        Initialize Random Forest model
        
        Parameters:
        -----------
        n_estimators : int, default=100
            Number of trees in the forest
        max_depth : int, optional
            Maximum depth of trees (None = no limit)
        min_samples_split : int, default=2
            Minimum samples required to split a node
        min_samples_leaf : int, default=1
            Minimum samples required at a leaf node
        random_state : int, default=42
            Random seed for reproducibility
        n_jobs : int, default=-1
            Number of parallel jobs (-1 = use all cores)
        **kwargs : dict
            Additional RandomForest parameters
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.kwargs = kwargs
        
        # Initialize model
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs
        )
        
        self.is_trained = False
        self.train_metrics = None
    
    def train(self, X_train, y_train, verbose=True):
        """
        Train the Random Forest model
        
        Parameters:
        -----------
        X_train : pd.DataFrame or np.ndarray
            Training features
        y_train : pd.Series or np.ndarray
            Training target
        verbose : bool, default=True
            Print training information
            
        Returns:
        --------
        dict
            Training metrics
        """
        if verbose:
            print("Starting Random Forest model training...")
            print(f"Training samples: {len(X_train)}")
            print(f"Features: {X_train.shape[1]}")
            print(f"Trees: {self.n_estimators}")
        
        # Train model
        self.model.fit(X_train, y_train)
        
        self.is_trained = True
        
        # Calculate training metrics
        y_train_pred = self.predict(X_train)
        self.train_metrics = calculate_all_metrics(
            y_train, 
            y_train_pred,
            n_features=X_train.shape[1]
        )
        
        if verbose:
            print("Training completed successfully")
            print(f"Training R2 Score: {self.train_metrics['r2']:.4f}")
            print(f"Training RMSE: {self.train_metrics['rmse']:.4f}")
        
        return self.train_metrics
    
    def fit(self, X, y):
        """
        Alias for train() method to match scikit-learn interface
        Used by comparison scripts
        """
        return self.train(X, y, verbose=True)
    
    def predict(self, X):
        """
        Make predictions
        
        Parameters:
        -----------
        X : pd.DataFrame or np.ndarray
            Features for prediction
            
        Returns:
        --------
        np.ndarray
            Predictions (guaranteed non-negative)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Make predictions
        predictions = self.model.predict(X)
        
        # Ensure non-negative predictions
        predictions = np.maximum(predictions, 0)
        
        return predictions
    
    def evaluate(self, X_test, y_test, verbose=True):
        """
        Evaluate model on test set
        
        Parameters:
        -----------
        X_test : pd.DataFrame or np.ndarray
            Test features
        y_test : pd.Series or np.ndarray
            Test target
        verbose : bool, default=True
            Print evaluation results
            
        Returns:
        --------
        dict
            Test metrics
        """
        if verbose:
            print("\nRandom Forest Model Evaluation:")
        
        # Make predictions
        y_pred = self.predict(X_test)
        
        # Calculate metrics
        metrics = calculate_all_metrics(
            y_test, 
            y_pred,
            n_features=X_test.shape[1]
        )
        
        if verbose:
            print(f"R2 Score: {metrics['r2']:.4f}")
            print(f"RMSE: {metrics['rmse']:.4f}")
            print(f"MAE: {metrics['mae']:.4f}")
            print(f"MAPE: {metrics['mape']:.2f}%")
        
        return metrics
    
    def get_feature_importance(self, feature_names=None, top_n=None):
        """
        Get feature importance scores
        
        Parameters:
        -----------
        feature_names : list, optional
            Names of features
        top_n : int, optional
            Return only top N features
            
        Returns:
        --------
        dict or pd.DataFrame
            Feature importance scores
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        importance = self.model.feature_importances_
        
        if feature_names is not None:
            import pandas as pd
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            if top_n is not None:
                importance_df = importance_df.head(top_n)
            
            return importance_df
        
        return importance
    
    def save(self, filepath):
        """
        Save model to disk
        
        Parameters:
        -----------
        filepath : str
            Path to save the model
        """
        # Create directory if it doesn't exist
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_data = {
            'model': self.model,
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'random_state': self.random_state,
            'n_jobs': self.n_jobs,
            'kwargs': self.kwargs,
            'is_trained': self.is_trained,
            'train_metrics': self.train_metrics
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to: {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """
        Load model from disk
        
        Parameters:
        -----------
        filepath : str
            Path to the saved model
            
        Returns:
        --------
        RandomForestModel
            Loaded model instance
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Create instance
        instance = cls(
            n_estimators=model_data['n_estimators'],
            max_depth=model_data['max_depth'],
            min_samples_split=model_data['min_samples_split'],
            min_samples_leaf=model_data['min_samples_leaf'],
            random_state=model_data['random_state'],
            n_jobs=model_data['n_jobs'],
            **model_data['kwargs']
        )
        
        # Restore model state
        instance.model = model_data['model']
        instance.is_trained = model_data['is_trained']
        instance.train_metrics = model_data['train_metrics']
        
        print(f"Model loaded from: {filepath}")
        
        return instance
    
    def get_params(self):
        """
        Get model parameters
        
        Returns:
        --------
        dict
            Model parameters
        """
        return {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'random_state': self.random_state,
            'n_jobs': self.n_jobs,
            **self.kwargs
        }