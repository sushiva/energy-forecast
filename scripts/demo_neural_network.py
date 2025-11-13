"""
Demo: Neural Network vs Tree-Based Models with Synthetic Data

This script demonstrates the Neural Network model with synthetic energy data.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.models.neural_network_model import NeuralNetworkModel
from src.models.xgboost_model import XGBoostModel
from src.models.random_forest_model import RandomForestModel

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def generate_synthetic_energy_data(n_samples=1000, random_state=42):
    """
    Generate synthetic energy consumption data with realistic patterns.
    
    Features:
    - Temperature
    - Hour of day
    - Day of week  
    - Building area
    - Occupancy
    - Various interactions
    """
    np.random.seed(random_state)
    
    # Base features
    temperature = np.random.normal(20, 5, n_samples)  # Celsius
    hour = np.random.randint(0, 24, n_samples)
    day_of_week = np.random.randint(0, 7, n_samples)
    building_area = np.random.uniform(1000, 5000, n_samples)  # sq ft
    occupancy = np.random.uniform(10, 100, n_samples)  # number of people
    
    # Engineered features
    is_weekend = (day_of_week >= 5).astype(int)
    is_business_hours = ((hour >= 8) & (hour <= 18)).astype(int)
    temp_squared = temperature ** 2
    area_occupancy = building_area * occupancy / 1000
    
    # Create DataFrame
    df = pd.DataFrame({
        'temperature': temperature,
        'hour': hour,
        'day_of_week': day_of_week,
        'building_area': building_area,
        'occupancy': occupancy,
        'is_weekend': is_weekend,
        'is_business_hours': is_business_hours,
        'temp_squared': temp_squared,
        'area_occupancy': area_occupancy,
        'temp_hour': temperature * hour,
        'temp_occupancy': temperature * occupancy,
        'hour_sin': np.sin(2 * np.pi * hour / 24),
        'hour_cos': np.cos(2 * np.pi * hour / 24),
        'day_sin': np.sin(2 * np.pi * day_of_week / 7),
        'day_cos': np.cos(2 * np.pi * day_of_week / 7),
    })
    
    # Generate target: Energy consumption (complex non-linear relationship)
    energy = (
        5.0 +  # base consumption
        0.3 * temperature +
        0.02 * temperature ** 2 +  # non-linear temperature effect
        0.001 * building_area +
        0.1 * occupancy +
        2.0 * is_business_hours +
        -1.5 * is_weekend +
        0.01 * area_occupancy +
        0.05 * temperature * occupancy +
        np.random.normal(0, 0.5, n_samples)  # noise
    )
    
    # Ensure positive energy
    energy = np.maximum(energy, 0.1)
    
    df['energy_consumption'] = energy
    
    print(f"✅ Generated {n_samples} samples of synthetic energy data")
    print(f"   Features: {len(df.columns) - 1}")
    print(f"   Energy range: {energy.min():.2f} - {energy.max():.2f} kWh")
    
    return df


def compare_all_models(X_train, y_train, X_test, y_test):
    """Train and compare all models."""
    print("\n" + "=" * 80)
    print("COMPARING ALL MODELS")
    print("=" * 80)
    
    results = {}
    
    # 1. Train Neural Network
    print("\n🧠 Training Neural Network...")
    print("-" * 80)
    nn_model = NeuralNetworkModel(
        hidden_layers=(64, 32, 16),
        learning_rate=0.001,
        batch_size=32,
        epochs=100,
        early_stopping_patience=10,
        random_state=42
    )
    nn_model.fit(X_train, y_train)
    nn_metrics = nn_model.evaluate(X_test, y_test)
    results['Neural Network'] = {
        'model': nn_model,
        'metrics': nn_metrics
    }
    
    # Plot training history
    nn_model.plot_training_history(save_path='models/advanced/nn_training_history.png')
    
    # 2. Train XGBoost
    print("\n" + "-" * 80)
    print("🌲 Training XGBoost...")
    print("-" * 80)
    xgb_model = XGBoostModel(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    xgb_model.fit(X_train, y_train)
    xgb_metrics = xgb_model.evaluate(X_test, y_test)
    results['XGBoost'] = {
        'model': xgb_model,
        'metrics': xgb_metrics
    }
    
    # 3. Train Random Forest
    print("\n" + "-" * 80)
    print("🌳 Training Random Forest...")
    print("-" * 80)
    rf_model = RandomForestModel(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    rf_model.fit(X_train, y_train)
    rf_metrics = rf_model.evaluate(X_test, y_test)
    results['Random Forest'] = {
        'model': rf_model,
        'metrics': rf_metrics
    }
    
    return results


def create_comparison_table(results):
    """Create and display comparison table."""
    print("\n" + "=" * 80)
    print("PERFORMANCE COMPARISON")
    print("=" * 80)
    
    comparison_data = []
    for model_name, result in results.items():
        metrics = result['metrics']
        comparison_data.append({
        'Model': model_name,
        'R² Score': metrics.get('r2_score', metrics.get('r2')),  # Handle both keys
        'RMSE (kWh)': metrics['rmse'],
        'MAE (kWh)': metrics['mae'],
        'MAPE (%)': metrics['mape']
    })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('R² Score', ascending=False)
    comparison_df['Rank'] = range(1, len(comparison_df) + 1)
    comparison_df = comparison_df[['Rank', 'Model', 'R² Score', 'RMSE (kWh)', 'MAE (kWh)', 'MAPE (%)']]
    
    print("\n📊 Model Performance Comparison:")
    print(comparison_df.to_string(index=False))
    
    best_model_name = comparison_df.iloc[0]['Model']
    best_r2 = comparison_df.iloc[0]['R² Score']
    
    medals = {1: '🥇', 2: '🥈', 3: '🥉'}
    print(f"\n{medals.get(1, '')} WINNER: {best_model_name}")
    print(f"   R² Score: {best_r2:.4f} ({best_r2*100:.2f}% variance explained)")
    
    return comparison_df, best_model_name


def plot_comparison_charts(results, comparison_df):
    """Create comparison visualizations."""
    print("\n📊 Creating comparison visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    model_names = comparison_df['Model'].tolist()
    r2_scores = comparison_df['R² Score'].tolist()
    rmse_scores = comparison_df['RMSE (kWh)'].tolist()
    mae_scores = comparison_df['MAE (kWh)'].tolist()
    mape_scores = comparison_df['MAPE (%)'].tolist()
    
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    
    # R² Score
    ax1 = axes[0, 0]
    bars1 = ax1.barh(model_names, r2_scores, color=colors)
    ax1.set_xlabel('R² Score')
    ax1.set_title('R² Score Comparison', fontweight='bold')
    for i, (bar, score) in enumerate(zip(bars1, r2_scores)):
        ax1.text(score - 0.02, i, f'{score:.4f}', ha='right', va='center', 
                fontweight='bold', color='white')
    ax1.grid(axis='x', alpha=0.3)
    
    # RMSE
    ax2 = axes[0, 1]
    bars2 = ax2.barh(model_names, rmse_scores, color=colors)
    ax2.set_xlabel('RMSE (kWh)')
    ax2.set_title('RMSE (Lower is Better)', fontweight='bold')
    for i, (bar, score) in enumerate(zip(bars2, rmse_scores)):
        ax2.text(score + 0.01, i, f'{score:.2f}', ha='left', va='center', fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    
    # MAE
    ax3 = axes[1, 0]
    bars3 = ax3.barh(model_names, mae_scores, color=colors)
    ax3.set_xlabel('MAE (kWh)')
    ax3.set_title('MAE (Lower is Better)', fontweight='bold')
    for i, (bar, score) in enumerate(zip(bars3, mae_scores)):
        ax3.text(score + 0.01, i, f'{score:.2f}', ha='left', va='center', fontweight='bold')
    ax3.grid(axis='x', alpha=0.3)
    
    # MAPE
    ax4 = axes[1, 1]
    bars4 = ax4.barh(model_names, mape_scores, color=colors)
    ax4.set_xlabel('MAPE (%)')
    ax4.set_title('MAPE (Lower is Better)', fontweight='bold')
    for i, (bar, score) in enumerate(zip(bars4, mape_scores)):
        ax4.text(score + 0.1, i, f'{score:.2f}%', ha='left', va='center', fontweight='bold')
    ax4.grid(axis='x', alpha=0.3)
    
    plt.suptitle('Neural Network vs Tree-Based Models', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save
    save_dir = Path('models/advanced')
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig('models/advanced/nn_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✅ Comparison chart saved: models/advanced/nn_comparison.png")
    plt.close()


def plot_predictions_comparison(results, X_test, y_test):
    """Plot actual vs predicted for all models."""
    print("\n📊 Creating predictions comparison...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, (model_name, result) in enumerate(results.items()):
        model = result['model']
        y_pred = model.predict(X_test)
        r2 = result['metrics'].get('r2_score', result['metrics'].get('r2'))
        rmse = result['metrics']['rmse']
        
        ax = axes[idx]
        ax.scatter(y_test, y_pred, alpha=0.5, s=30, edgecolors='k', linewidth=0.5)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
                'r--', lw=2, label='Perfect Prediction')
        ax.set_xlabel('Actual Energy (kWh)', fontsize=11)
        ax.set_ylabel('Predicted Energy (kWh)', fontsize=11)
        ax.set_title(f'{model_name}\nR²={r2:.4f}, RMSE={rmse:.2f}',
                    fontweight='bold', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Actual vs Predicted: All Models', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('models/advanced/nn_predictions.png', dpi=300, bbox_inches='tight')
    print(f"✅ Predictions chart saved: models/advanced/nn_predictions.png")
    plt.close()


def save_best_model(results, best_model_name):
    """Save the best model."""
    print("\n" + "=" * 80)
    print("SAVING BEST MODEL")
    print("=" * 80)
    
    best_model = results[best_model_name]['model']
    save_dir = Path('models/advanced')
    save_dir.mkdir(parents=True, exist_ok=True)
    
    if best_model_name == 'Neural Network':
        save_path = save_dir / 'neural_network_best'
        best_model.save_model(str(save_path))  # NN uses save_model()
    elif best_model_name == 'XGBoost':
        save_path = save_dir / 'xgboost_best.pkl'
        best_model.save(str(save_path))  # XGBoost uses save()
    elif best_model_name == 'Random Forest':
        save_path = save_dir / 'random_forest_best.pkl'
    best_model.save(str(save_path))  # RF uses save()


def main():
    """Main execution."""
    print("\n" + "=" * 80)
    print("NEURAL NETWORK DEMO WITH SYNTHETIC DATA")
    print("=" * 80)
    
    # Generate data
    print("\n📊 Generating synthetic energy data...")
    df = generate_synthetic_energy_data(n_samples=1000, random_state=42)
    
    # Split features and target
    X = df.drop('energy_consumption', axis=1)
    y = df['energy_consumption']
    
    # Train/test split (80/20)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"\n📊 Data Split:")
    print(f"   Training: {len(X_train)} samples")
    print(f"   Testing: {len(X_test)} samples")
    print(f"   Features: {X_train.shape[1]}")
    
    # Compare models
    results = compare_all_models(X_train, y_train, X_test, y_test)
    
    # Create comparison table
    comparison_df, best_model_name = create_comparison_table(results)
    
    # Create visualizations
    plot_comparison_charts(results, comparison_df)
    plot_predictions_comparison(results, X_test, y_test)
    
    # Save best model
    save_best_model(results, best_model_name)
    
    # Summary
    print("\n" + "=" * 80)
    print("✅ NEURAL NETWORK DEMO COMPLETE!")
    print("=" * 80)
    print(f"\n🏆 Best Model: {best_model_name}")
    print(f"\n📁 Saved Files:")
    print(f"   - models/advanced/nn_training_history.png")
    print(f"   - models/advanced/nn_comparison.png")
    print(f"   - models/advanced/nn_predictions.png")
    print(f"   - models/advanced/{best_model_name.lower().replace(' ', '_')}_best.*")


if __name__ == "__main__":
    main()