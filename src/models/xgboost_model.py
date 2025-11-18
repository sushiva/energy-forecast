"""
XGBoost Model Module
Gradient boosting model for energy forecasting
"""
import joblib
import numpy as np
import pickle
from pathlib import Path
from xgboost import XGBRegressor
from src.evaluation.metrics import calculate_all_metrics



class XGBoostModel:
    """
    XGBoost Regressor for energy forecasting
    Handles training, prediction, evaluation, and model persistence
    """
    
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=5, 
                 random_state=42, **kwargs):
        """
        Initialize XGBoost model
        
        Parameters:
        -----------
        n_estimators : int, default=100
            Number of boosting rounds
        learning_rate : float, default=0.1
            Learning rate (eta)
        max_depth : int, default=5
            Maximum tree depth
        random_state : int, default=42
            Random seed for reproducibility
        **kwargs : dict
            Additional XGBoost parameters
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self.kwargs = kwargs
        
        # Initialize model
        self.model = XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
            **kwargs
        )
        
        self.is_trained = False
        self.train_metrics = None
    
    def train(self, X_train, y_train, X_val=None, y_val=None, verbose=True):
        """
        Train the XGBoost model
        
        Parameters:
        -----------
        X_train : pd.DataFrame or np.ndarray
            Training features
        y_train : pd.Series or np.ndarray
            Training target
        X_val : pd.DataFrame or np.ndarray, optional
            Validation features for early stopping
        y_val : pd.Series or np.ndarray, optional
            Validation target for early stopping
        verbose : bool, default=True
            Print training information
            
        Returns:
        --------
        dict
            Training metrics
        """
        if verbose:
            print("Starting XGBoost model training...")
            print(f"Training samples: {len(X_train)}")
            print(f"Features: {X_train.shape[1]}")
        
        # Prepare eval set for early stopping if validation data provided
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_train, y_train), (X_val, y_val)]
            if verbose:
                print(f"Validation samples: {len(X_val)}")
        
        # Train model
        self.model.fit(
            X_train, 
            y_train,
            eval_set=eval_set,
            verbose=False
        )
        
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
            print("\nXGBoost Model Evaluation:")
        
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
            'learning_rate': self.learning_rate,
            'max_depth': self.max_depth,
            'random_state': self.random_state,
            'kwargs': self.kwargs,
            'is_trained': self.is_trained,
            'train_metrics': self.train_metrics
        }
        
        # with open(filepath, 'wb') as f:
        #    pickle.dump(model_data, f) 

        # --- CHANGED LINE ---
         # We remove the open(filepath, 'wb') block and use joblib directly.
        model_data = joblib.load(filepath)    
        
        print(f"Model saved to: {filepath}")
    
    @classmethod
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
        XGBoostModel
            Loaded model instance
        """
        model_data = joblib.load(filepath)
        
        # Create instance with default parameters
        instance = cls()
        
        # Restore model state
        instance.model = model_data['model']
        instance.feature_names = model_data.get('feature_names', None)
        instance.is_trained = True
        
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
            'learning_rate': self.learning_rate,
            'max_depth': self.max_depth,
            'random_state': self.random_state,
            **self.kwargs
        }