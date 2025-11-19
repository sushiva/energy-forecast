"""
Main Training Pipeline - Energy Consumption Forecasting
========================================================

This is the main entry point for training and comparing all models
on the real UCI Energy Efficiency dataset.

Models trained:
- Baseline (Linear Regression)
- XGBoost (Gradient Boosting)
- Random Forest
- Neural Network

Usage:
    python scripts/train_pipeline.py
    python scripts/train_pipeline.py --data data/raw/energy-efficiency-data.csv
    python scripts/train_pipeline.py --target Y2  # For cooling load

Author: Sudhir Shivaram Bhargav
Date: November 2025
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split

from src.data.loader import load_energy_data
from src.models.baseline import BaselineModel
from src.models.xgboost_model import XGBoostModel
from src.models.random_forest_model import RandomForestModel
from src.models.neural_network_model import NeuralNetworkModel
from src.evaluation.metrics import calculate_all_metrics, print_metrics

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Train and compare all energy forecasting models'
    )
    
    parser.add_argument('--data', type=str,
                       default='data/raw/energy-efficiency-data.csv',
                       help='Path to energy efficiency data CSV')
    
    parser.add_argument('--target', type=str,
                       default='Y1',
                       choices=['Y1', 'Y2'],
                       help='Target variable (Y1=Heating Load, Y2=Cooling Load)')
    
    parser.add_argument('--test-size', type=float,
                       default=0.2,
                       help='Test set size (default: 0.2)')
    
    parser.add_argument('--random-state', type=int,
                       default=42,
                       help='Random state for reproducibility')
    
    parser.add_argument('--output-dir', type=str,
                       default='models/advanced',
                       help='Directory to save trained models')
    
    return parser.parse_args()


def compare_all_models(X_train, y_train, X_test, y_test, output_dir='models/advanced'):
    """
    Train and compare all models
    
    Parameters:
    -----------
    X_train, y_train : Training data
    X_test, y_test : Test data
    output_dir : Directory to save models and results
    
    Returns:
    --------
    dict : Dictionary of trained models and their metrics
    """
    print("\n" + "=" * 80)
    print("TRAINING AND COMPARING ALL MODELS")
    print("=" * 80)
    
    results = {}
    
    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # ========================================================================
    # 1. BASELINE MODEL (Linear Regression)
    # ========================================================================
    print("\n" + "-" * 80)
    print("1. BASELINE MODEL: Linear Regression")
    print("-" * 80)
    
    baseline = BaselineModel()
    baseline.train(X_train, y_train)
    y_pred_baseline = baseline.predict(X_test)
    
    metrics_baseline = calculate_all_metrics(y_test, y_pred_baseline, n_features=X_test.shape[1])
    print_metrics(metrics_baseline, "Baseline Model")
    
    # Save
    baseline.save(f'{output_dir}/baseline_model.pkl')
    print(f"Saved: {output_dir}/baseline_model.pkl")
    
    results['Baseline'] = {
        'model': baseline,
        'metrics': metrics_baseline,
        'predictions': y_pred_baseline
    }
    
    # ========================================================================
    # 2. XGBOOST MODEL
    # ========================================================================
    print("\n" + "-" * 80)
    print("2. XGBOOST MODEL: Gradient Boosting")
    print("-" * 80)
    
    xgb = XGBoostModel(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    xgb.train(X_train, y_train, verbose=False)
    y_pred_xgb = xgb.predict(X_test)
    
    metrics_xgb = calculate_all_metrics(y_test, y_pred_xgb, n_features=X_test.shape[1])
    print_metrics(metrics_xgb, "XGBoost Model")
    
    # Save
    xgb.save(f'{output_dir}/xgboost_best.pkl')
    print(f"Saved: {output_dir}/xgboost_best.pkl")
    
    results['XGBoost'] = {
        'model': xgb,
        'metrics': metrics_xgb,
        'predictions': y_pred_xgb
    }
    
    # ========================================================================
    # 3. RANDOM FOREST MODEL
    # ========================================================================
    print("\n" + "-" * 80)
    print("3. RANDOM FOREST MODEL")
    print("-" * 80)
    
    rf = RandomForestModel(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    rf.train(X_train, y_train, verbose=False)
    y_pred_rf = rf.predict(X_test)
    
    metrics_rf = calculate_all_metrics(y_test, y_pred_rf, n_features=X_test.shape[1])
    print_metrics(metrics_rf, "Random Forest Model")
    
    # Save
    rf.save(f'{output_dir}/random_forest_model.pkl')
    print(f"Saved: {output_dir}/random_forest_model.pkl")
    
    results['RandomForest'] = {
        'model': rf,
        'metrics': metrics_rf,
        'predictions': y_pred_rf
    }
    
    # ========================================================================
    # 4. NEURAL NETWORK MODEL
    # ========================================================================
    print("\n" + "-" * 80)
    print("4. NEURAL NETWORK MODEL")
    print("-" * 80)
    
    nn = NeuralNetworkModel(
        hidden_layers=[64, 32, 16],
        epochs=100,
        batch_size=32,
        learning_rate=0.001,
        random_state=42
    )
    nn.fit(X_train, y_train)
    y_pred_nn = nn.predict(X_test)
    
    metrics_nn = calculate_all_metrics(y_test, y_pred_nn, n_features=X_test.shape[1])
    print_metrics(metrics_nn, "Neural Network Model")
    
    # Save
    nn.save_model(f'{output_dir}/neural_network_model.h5')
    print(f"Saved: {output_dir}/neural_network_model.h5")
    
    results['NeuralNetwork'] = {
        'model': nn,
        'metrics': metrics_nn,
        'predictions': y_pred_nn
    }
    
    return results


def create_comparison_visualization(results, y_test, output_dir='models/advanced'):
    """Create comprehensive comparison visualization"""
    print("\n" + "=" * 80)
    print("CREATING COMPARISON VISUALIZATIONS")
    print("=" * 80)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Model Comparison: Energy Consumption Forecasting', 
                 fontsize=16, fontweight='bold')
    
    # Prepare data
    model_names = list(results.keys())
    r2_scores = [results[m]['metrics']['r2'] * 100 for m in model_names]
    rmse_values = [results[m]['metrics']['rmse'] for m in model_names]
    mae_values = [results[m]['metrics']['mae'] for m in model_names]
    mape_values = [results[m]['metrics']['mape'] for m in model_names]
    
    colors = ['#FFA500', '#00AA00', '#0066CC', '#AA00AA']
    
    # Plot 1: R² Scores
    ax = axes[0, 0]
    bars = ax.bar(model_names, r2_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    ax.set_ylabel('R² Score (%)', fontsize=12, fontweight='bold')
    ax.set_title('Model Accuracy (R² Score)', fontsize=13, fontweight='bold')
    ax.axhline(y=99, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, score in zip(bars, r2_scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.2f}%', ha='center', va='bottom', fontweight='bold')
    
    # Highlight best
    best_idx = r2_scores.index(max(r2_scores))
    bars[best_idx].set_edgecolor('gold')
    bars[best_idx].set_linewidth(4)
    
    # Plot 2: RMSE Comparison
    ax = axes[0, 1]
    bars = ax.bar(model_names, rmse_values, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    ax.set_ylabel('RMSE (kWh)', fontsize=12, fontweight='bold')
    ax.set_title('Prediction Error (RMSE - Lower is Better)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, rmse in zip(bars, rmse_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{rmse:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 3: MAE Comparison
    ax = axes[1, 0]
    bars = ax.bar(model_names, mae_values, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    ax.set_ylabel('MAE (kWh)', fontsize=12, fontweight='bold')
    ax.set_title('Mean Absolute Error (MAE)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, mae in zip(bars, mae_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{mae:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 4: Predictions vs Actual
    ax = axes[1, 1]
    for i, (name, color) in enumerate(zip(model_names, colors)):
        y_pred = results[name]['predictions']
        ax.scatter(y_test, y_pred, alpha=0.6, s=30, color=color, label=name)
    
    # Perfect prediction line
    min_val, max_val = y_test.min(), y_test.max()
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Perfect Prediction')
    
    ax.set_xlabel('Actual Energy (kWh)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Predicted Energy (kWh)', fontsize=12, fontweight='bold')
    ax.set_title('Predictions vs Actual Values', fontsize=13, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    output_path = f'{output_dir}/model_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved comparison plot: {output_path}")
    
    plt.close()


def print_final_summary(results):
    """Print final summary with best model recommendation"""
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    
    # Create summary table
    summary_data = []
    for model_name, data in results.items():
        metrics = data['metrics']
        summary_data.append({
            'Model': model_name,
            'R² (%)': f"{metrics['r2']*100:.2f}",
            'RMSE': f"{metrics['rmse']:.2f}",
            'MAE': f"{metrics['mae']:.2f}",
            'MAPE (%)': f"{metrics['mape']:.2f}"
        })
    
    summary_df = pd.DataFrame(summary_data)
    print("\nPerformance Comparison:")
    print(summary_df.to_string(index=False))
    
    # Identify best model
    best_r2 = max(results.items(), key=lambda x: x[1]['metrics']['r2'])
    best_model_name = best_r2[0]
    best_metrics = best_r2[1]['metrics']
    
    print("\n" + "=" * 80)
    print(f"BEST MODEL: {best_model_name}")
    print("=" * 80)
    print(f"  R² Score: {best_metrics['r2']*100:.2f}%")
    print(f"  RMSE: {best_metrics['rmse']:.2f} kWh")
    print(f"  MAE: {best_metrics['mae']:.2f} kWh")
    print(f"  MAPE: {best_metrics['mape']:.2f}%")
    print("=" * 80)


def main():
    """Main training pipeline"""
    
    print("=" * 80)
    print("ENERGY CONSUMPTION FORECASTING - MAIN TRAINING PIPELINE")
    print("=" * 80)
    
    # Parse arguments
    args = parse_arguments()
    
    print(f"\nConfiguration:")
    print(f"  Data: {args.data}")
    print(f"  Target: {args.target} ({'Heating Load' if args.target == 'Y1' else 'Cooling Load'})")
    print(f"  Test size: {args.test_size}")
    print(f"  Random state: {args.random_state}")
    print(f"  Output directory: {args.output_dir}")
    
    # ========================================================================
    # STEP 1: LOAD DATA
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 1: LOAD REAL DATA")
    print("=" * 80)
    
    X, y = load_energy_data(args.data, target=args.target)
    
    print(f"Loaded real UCI Energy Efficiency dataset")
    print(f"  Samples: {X.shape[0]}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Target ({args.target}) range: {y.min():.2f} - {y.max():.2f} kWh")
    print(f"\nFeatures: {list(X.columns)}")
    
    # ========================================================================
    # STEP 2: TRAIN-TEST SPLIT
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 2: TRAIN-TEST SPLIT")
    print("=" * 80)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        random_state=args.random_state
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # ========================================================================
    # STEP 3: TRAIN AND COMPARE ALL MODELS
    # ========================================================================
    results = compare_all_models(X_train, y_train, X_test, y_test, args.output_dir)
    
    # ========================================================================
    # STEP 4: CREATE VISUALIZATIONS
    # ========================================================================
    create_comparison_visualization(results, y_test, args.output_dir)
    
    # ========================================================================
    # STEP 5: FINAL SUMMARY
    # ========================================================================
    print_final_summary(results)
    
    print("\n" + "=" * 80)
    print("TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"\nAll models and results saved to: {args.output_dir}/")
    print("\nNext steps:")
    print("  1. Review model comparison plot")
    print("  2. Test predictions: python scripts/evaluate.py --model <model_path>")
    print("  3. Launch dashboard: python deployment/api/app.py")
    print("=" * 80)


if __name__ == "__main__":
    main()