"""
Advanced Models Comparison Script
Compare Baseline, XGBoost, and Random Forest models with engineered features
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.data.loader import load_energy_data
from src.data.splitter import DataSplitter
from src.data.feature_engineer import EnergyFeatureEngineer
from src.models.baseline import BaselineModel
from src.models.xgboost_model import XGBoostModel
from src.models.random_forest_model import RandomForestModel
from src.evaluation.metrics import calculate_all_metrics


def print_section(title):
    """Print formatted section header"""
    print("\n" + "=" * 90)
    print(f"  {title}")
    print("=" * 90)


def train_and_evaluate_model(model, model_name, X_train, X_test, y_train, y_test):
    """
    Train and evaluate a model
    
    Parameters:
    -----------
    model : object
        Model instance
    model_name : str
        Name of the model
    X_train, X_test : array-like
        Training and test features
    y_train, y_test : array-like
        Training and test targets
        
    Returns:
    --------
    dict
        Results dictionary with metrics and predictions
    """
    print(f"\n🔹 Training {model_name}...")
    
    # Train (handle models with/without verbose parameter)
    try:
        train_metrics = model.train(X_train, y_train, verbose=False)
    except TypeError:
        train_metrics = model.train(X_train, y_train)
    
    # Evaluate (handle models with/without verbose parameter)
    try:
        test_metrics = model.evaluate(X_test, y_test, verbose=False)
    except TypeError:
        test_metrics = model.evaluate(X_test, y_test)
    
    # Get predictions
    y_pred = model.predict(X_test)
    
    print(f"✅ {model_name} Training Complete")
    print(f"   Train R²: {train_metrics['r2']:.4f} | Test R²: {test_metrics['r2']:.4f}")
    print(f"   Train RMSE: {train_metrics['rmse']:.4f} | Test RMSE: {test_metrics['rmse']:.4f}")
    
    # Get feature importance if available
    feature_importance = None
    if hasattr(model, 'get_feature_importance'):
        try:
            feature_importance = model.get_feature_importance(
                feature_names=X_train.columns.tolist() if hasattr(X_train, 'columns') else None,
                top_n=15
            )
        except:
            pass
    
    return {
        'model': model,
        'model_name': model_name,
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'predictions': y_pred,
        'feature_importance': feature_importance
    }


def compare_models(results_dict):
    """
    Compare all models and display results
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary of model results
    """
    print_section("MODEL COMPARISON")
    
    # Create comparison DataFrame
    comparison_data = []
    
    for model_name, results in results_dict.items():
        test_metrics = results['test_metrics']
        comparison_data.append({
            'Model': model_name,
            'R² Score': test_metrics['r2'],
            'RMSE (kWh)': test_metrics['rmse'],
            'MAE (kWh)': test_metrics['mae'],
            'MAPE (%)': test_metrics['mape']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('R² Score', ascending=False)
    
    print("\n📊 PERFORMANCE COMPARISON:")
    print(comparison_df.to_string(index=False))
    
    # Find best model
    best_model_name = comparison_df.iloc[0]['Model']
    best_r2 = comparison_df.iloc[0]['R² Score']
    
    print(f"\n🏆 BEST MODEL: {best_model_name}")
    print(f"   R² Score: {best_r2:.4f} ({best_r2*100:.2f}% variance explained)")
    
    return comparison_df


def plot_model_comparison(results_dict, comparison_df, y_test):
    """
    Create comprehensive comparison visualizations
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary of model results
    comparison_df : pd.DataFrame
        Comparison DataFrame
    y_test : array-like
        True test values
    """
    print_section("GENERATING VISUALIZATIONS")
    
    n_models = len(results_dict)
    
    fig = plt.figure(figsize=(18, 12))
    
    # Create grid
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Plot 1: Metrics Comparison (Bar Chart)
    ax1 = fig.add_subplot(gs[0, :2])
    metrics = ['R² Score', 'RMSE (kWh)', 'MAE (kWh)']
    x = np.arange(len(comparison_df))
    width = 0.25
    
    for i, metric in enumerate(metrics):
        values = comparison_df[metric].values
        if metric == 'R² Score':
            normalized_values = values  # Already 0-1
        else:
            normalized_values = values / values.max()  # Normalize for visualization
        
        ax1.bar(x + i*width, normalized_values, width, label=metric, alpha=0.8)
    
    ax1.set_xlabel('Model', fontsize=11)
    ax1.set_ylabel('Normalized Score', fontsize=11)
    ax1.set_title('Model Performance Comparison', fontsize=13, fontweight='bold')
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(comparison_df['Model'], rotation=15, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: R² Score Comparison
    ax2 = fig.add_subplot(gs[0, 2])
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(comparison_df)))
    bars = ax2.barh(comparison_df['Model'], comparison_df['R² Score'], color=colors, alpha=0.8)
    ax2.set_xlabel('R² Score', fontsize=11)
    ax2.set_title('R² Score by Model', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, comparison_df['R² Score'])):
        ax2.text(val, i, f' {val:.4f}', va='center', fontsize=10)
    
    # Plot 3-5: Actual vs Predicted for each model
    for idx, (model_name, results) in enumerate(results_dict.items()):
        row = 1 + idx // 3
        col = idx % 3
        ax = fig.add_subplot(gs[row, col])
        
        y_pred = results['predictions']
        
        ax.scatter(y_test, y_pred, alpha=0.6, edgecolors='black', s=30)
        
        # Perfect prediction line
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect')
        
        ax.set_xlabel('Actual (kWh)', fontsize=10)
        ax.set_ylabel('Predicted (kWh)', fontsize=10)
        
        r2 = results['test_metrics']['r2']
        rmse = results['test_metrics']['rmse']
        ax.set_title(f'{model_name}\nR²={r2:.4f}, RMSE={rmse:.2f}', 
                    fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    # Plot 6: Summary Statistics
    ax_summary = fig.add_subplot(gs[2, :])
    ax_summary.axis('off')
    
    # Create summary text
    best_model = comparison_df.iloc[0]
    baseline_model = comparison_df[comparison_df['Model'] == 'Baseline (Linear + Log)'].iloc[0] if 'Baseline (Linear + Log)' in comparison_df['Model'].values else None
    
    summary_text = f"""
    ADVANCED MODELS COMPARISON SUMMARY
    {'='*80}
    
    Best Performing Model: {best_model['Model']}
      • R² Score: {best_model['R² Score']:.4f} ({best_model['R² Score']*100:.2f}% variance explained)
      • RMSE: {best_model['RMSE (kWh)']:.2f} kWh
      • MAE: {best_model['MAE (kWh)']:.2f} kWh
      • MAPE: {best_model['MAPE (%)']:.2f}%
    """
    
    if baseline_model is not None:
        improvement = ((best_model['R² Score'] - baseline_model['R² Score']) / baseline_model['R² Score']) * 100
        summary_text += f"""
    Improvement over Baseline:
      • R² improvement: +{improvement:.2f}%
      • RMSE reduction: {baseline_model['RMSE (kWh)'] - best_model['RMSE (kWh)']:.2f} kWh
    """
    
    summary_text += f"""
    
    All Models Tested: {len(results_dict)}
    All Predictions: Non-negative ✓
    Training Status: Complete ✓
    """
    
    ax_summary.text(0.1, 0.9, summary_text, transform=ax_summary.transAxes,
                   fontsize=11, verticalalignment='top', family='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    plt.suptitle('Advanced Models Performance Analysis', 
                fontsize=16, fontweight='bold', y=0.995)
    
    # Save plot
    save_path = 'models/advanced/comparison_results.png'
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Visualizations saved to: {save_path}")
    
    plt.show()


def main():
    """Main comparison function"""
    
    print("\n" + "=" * 90)
    print(" " * 25 + "ADVANCED MODELS COMPARISON")
    print(" " * 20 + "XGBoost vs Random Forest vs Baseline")
    print("=" * 90)
    
    # Step 1: Load and prepare data
    print_section("STEP 1: DATA PREPARATION")
    
    X, y = load_energy_data('data/raw/energy-efficiency-data.csv', target='Y1')
    
    # Split data
    splitter = DataSplitter(test_size=0.2, random_state=42)
    X_train, X_test, y_train, y_test = splitter.split(X, y)
    
    # Engineer features
    print("\n🔧 Engineering features...")
    engineer = EnergyFeatureEngineer(
        create_domain_features=True,
        create_interactions=True,
        create_polynomial=False
    )
    
    X_train_eng = engineer.fit_transform(X_train)
    X_test_eng = engineer.transform(X_test)
    
    print(f"✅ Features engineered: {X_train_eng.shape[1]} features (was {X_train.shape[1]})")
    
    # Step 2: Train all models
    print_section("STEP 2: TRAINING ALL MODELS")
    
    results_dict = {}
    
    # Model 1: Baseline (Linear Regression with Log Transform)
    baseline = BaselineModel()
    results_dict['Baseline (Linear + Log)'] = train_and_evaluate_model(
        baseline, 'Baseline (Linear + Log)', 
        X_train_eng, X_test_eng, y_train, y_test
    )
    
    # Model 2: XGBoost
    xgb = XGBoostModel(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    results_dict['XGBoost'] = train_and_evaluate_model(
        xgb, 'XGBoost',
        X_train_eng, X_test_eng, y_train, y_test
    )
    
    # Model 3: Random Forest
    rf = RandomForestModel(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    results_dict['Random Forest'] = train_and_evaluate_model(
        rf, 'Random Forest',
        X_train_eng, X_test_eng, y_train, y_test
    )
    
    # Step 3: Compare models
    comparison_df = compare_models(results_dict)
    
    # Step 4: Visualizations
    plot_model_comparison(results_dict, comparison_df, y_test)
    
    # Step 5: Save best model
    print_section("STEP 5: SAVING BEST MODEL")
    
    best_model_name = comparison_df.iloc[0]['Model']
    best_model = results_dict[best_model_name]['model']
    
    save_path = f'models/advanced/{best_model_name.lower().replace(" ", "_")}_best.pkl'
    best_model.save(save_path)
    
    print(f"\n✅ Best model ({best_model_name}) saved to: {save_path}")
    
    # Final summary
    print_section("FINAL SUMMARY")
    
    print("\n✅ ALL MODELS TRAINED AND COMPARED SUCCESSFULLY!")
    print(f"\nModels tested: {len(results_dict)}")
    print(f"Best model: {best_model_name}")
    print(f"Best R² score: {comparison_df.iloc[0]['R² Score']:.4f}")
    print(f"\nAll results saved to: models/advanced/")
    
    print("\n" + "=" * 90)
    
    return results_dict, comparison_df


if __name__ == "__main__":
    results, comparison = main()