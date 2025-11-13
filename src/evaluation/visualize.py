"""
Visualization Module
Plotting functions for model evaluation and analysis
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Union, Optional, List
import seaborn as sns

sns.set_style("whitegrid")


def plot_actual_vs_predicted(y_true: Union[np.ndarray, pd.Series],
                             y_pred: np.ndarray,
                             title: str = "Actual vs Predicted",
                             save_path: Optional[str] = None):
    """
    Plot actual vs predicted values
    
    Parameters:
    -----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
    title : str
        Plot title
    save_path : str, optional
        Path to save the plot
    """
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.6, edgecolors='black')
    
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 
             'r--', lw=2, label='Perfect Prediction')
    
    plt.xlabel('Actual Values', fontsize=12)
    plt.ylabel('Predicted Values', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()


def plot_residuals(y_true: Union[np.ndarray, pd.Series],
                  y_pred: np.ndarray,
                  title: str = "Residual Plot",
                  save_path: Optional[str] = None):
    """
    Plot residuals vs predicted values
    
    Parameters:
    -----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
    title : str
        Plot title
    save_path : str, optional
        Path to save the plot
    """
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    
    residuals = y_true - y_pred
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.6, edgecolors='black')
    plt.axhline(y=0, color='r', linestyle='--', lw=2)
    
    plt.xlabel('Predicted Values', fontsize=12)
    plt.ylabel('Residuals', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()


def plot_residual_distribution(y_true: Union[np.ndarray, pd.Series],
                               y_pred: np.ndarray,
                               title: str = "Residual Distribution",
                               save_path: Optional[str] = None):
    """
    Plot histogram of residuals
    
    Parameters:
    -----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
    title : str
        Plot title
    save_path : str, optional
        Path to save the plot
    """
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    
    residuals = y_true - y_pred
    
    plt.figure(figsize=(10, 6))
    plt.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    plt.axvline(x=0, color='r', linestyle='--', lw=2, label='Zero Error')
    
    plt.xlabel('Residuals', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()


def plot_error_distribution(y_true: Union[np.ndarray, pd.Series],
                           y_pred: np.ndarray,
                           title: str = "Absolute Error Distribution",
                           save_path: Optional[str] = None):
    """
    Plot distribution of absolute errors
    
    Parameters:
    -----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
    title : str
        Plot title
    save_path : str, optional
        Path to save the plot
    """
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    
    errors = np.abs(y_true - y_pred)
    
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=30, edgecolor='black', alpha=0.7, color='orange')
    
    plt.xlabel('Absolute Error', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()


def plot_prediction_comparison(y_true: Union[np.ndarray, pd.Series],
                               y_pred: np.ndarray,
                               n_samples: int = 50,
                               title: str = "Prediction Comparison",
                               save_path: Optional[str] = None):
    """
    Plot comparison of predictions vs actual for first n samples
    
    Parameters:
    -----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
    n_samples : int
        Number of samples to plot
    title : str
        Plot title
    save_path : str, optional
        Path to save the plot
    """
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    
    n_samples = min(n_samples, len(y_true))
    indices = np.arange(n_samples)
    
    plt.figure(figsize=(14, 6))
    plt.plot(indices, y_true[:n_samples], 'o-', label='Actual', 
             markersize=8, linewidth=2)
    plt.plot(indices, y_pred[:n_samples], 's-', label='Predicted', 
             markersize=6, linewidth=2, alpha=0.7)
    
    plt.xlabel('Sample Index', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()


def plot_comprehensive_evaluation(y_true: Union[np.ndarray, pd.Series],
                                 y_pred: np.ndarray,
                                 model_name: str = "Model",
                                 save_path: Optional[str] = None):
    """
    Create comprehensive evaluation plots in a single figure
    
    Parameters:
    -----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
    model_name : str
        Name of the model
    save_path : str, optional
        Path to save the plot
    """
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{model_name} - Comprehensive Evaluation', 
                 fontsize=16, fontweight='bold')
    
    axes[0, 0].scatter(y_true, y_pred, alpha=0.6, edgecolors='black')
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    axes[0, 0].plot([min_val, max_val], [min_val, max_val], 
                    'r--', lw=2, label='Perfect Prediction')
    axes[0, 0].set_xlabel('Actual Values', fontsize=11)
    axes[0, 0].set_ylabel('Predicted Values', fontsize=11)
    axes[0, 0].set_title('Actual vs Predicted', fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].scatter(y_pred, residuals, alpha=0.6, edgecolors='black')
    axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0, 1].set_xlabel('Predicted Values', fontsize=11)
    axes[0, 1].set_ylabel('Residuals', fontsize=11)
    axes[0, 1].set_title('Residual Plot', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    axes[1, 0].axvline(x=0, color='r', linestyle='--', lw=2)
    axes[1, 0].set_xlabel('Residuals', fontsize=11)
    axes[1, 0].set_ylabel('Frequency', fontsize=11)
    axes[1, 0].set_title('Residual Distribution', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    n_samples = min(50, len(y_true))
    indices = np.arange(n_samples)
    axes[1, 1].plot(indices, y_true[:n_samples], 'o-', label='Actual', 
                    markersize=6, linewidth=1.5)
    axes[1, 1].plot(indices, y_pred[:n_samples], 's-', label='Predicted', 
                    markersize=4, linewidth=1.5, alpha=0.7)
    axes[1, 1].set_xlabel('Sample Index', fontsize=11)
    axes[1, 1].set_ylabel('Value', fontsize=11)
    axes[1, 1].set_title(f'First {n_samples} Predictions', fontsize=12, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()