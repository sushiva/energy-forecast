"""
Evaluation Script
Load a trained model and evaluate its performance
Supports: BaselineModel, XGBoostModel, RandomForestModel, NeuralNetworkModel
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import pandas as pd
import joblib
from src.data.loader import load_energy_data
from src.models.baseline import BaselineModel
from src.models.xgboost_model import XGBoostModel
from src.models.random_forest_model import RandomForestModel
from src.models.neural_network_model import NeuralNetworkModel
from src.evaluation.metrics import calculate_all_metrics, print_metrics
from src.evaluation.visualize import (
    plot_actual_vs_predicted,
    plot_residuals,
    plot_residual_distribution,
    plot_comprehensive_evaluation
)


def detect_and_load_model(filepath):
    """
    Automatically detect model type and load appropriately
    
    Parameters:
    -----------
    filepath : str
        Path to the model file
        
    Returns:
    --------
    model : BaselineModel, XGBoostModel, RandomForestModel, or NeuralNetworkModel
        Loaded model instance
    model_type : str
        Type of model loaded
    """
    print(f"Detecting model type from: {filepath}")
    
    # Try to load and inspect the file
    try:
        model_data = joblib.load(filepath)
        
        # Check what's in the saved data
        if isinstance(model_data, dict):
            # Check for XGBoost/RandomForest format (has 'model' key)
            if 'model' in model_data:
                model_obj = model_data['model']
                model_class = str(type(model_obj))
                
                if 'XGBRegressor' in model_class or 'XGBoost' in model_class:
                    print("  → Detected: XGBoost model")
                    model = XGBoostModel.load(filepath)
                    return model, "XGBoost"
                    
                elif 'RandomForest' in model_class:
                    print("  → Detected: Random Forest model")
                    model = RandomForestModel.load(filepath)
                    return model, "RandomForest"
                    
                elif 'Sequential' in model_class or 'keras' in model_class.lower():
                    print("  → Detected: Neural Network model")
                    model = NeuralNetworkModel.load(filepath)
                    return model, "NeuralNetwork"
            
            # Check for baseline model format
            elif 'model' not in model_data:
                print("  → Detected: Baseline model (pickle format)")
                model = BaselineModel.load(filepath)
                return model, "Baseline"
        
        # If we couldn't detect, try baseline as fallback
        print("  → Unknown format, trying Baseline model...")
        model = BaselineModel.load(filepath)
        return model, "Baseline"
        
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Attempting to load as Baseline model...")
        try:
            model = BaselineModel.load(filepath)
            return model, "Baseline"
        except Exception as e2:
            print(f"Failed to load model: {e2}")
            raise


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Evaluate trained model')
    
    parser.add_argument('--model', type=str, 
                       required=True,
                       help='Path to trained model file')
    
    parser.add_argument('--data', type=str, 
                       default='data/raw/energy-efficiency-data.csv',
                       help='Path to data file')
    
    parser.add_argument('--target', type=str, 
                       default='Y1',
                       help='Target column name')
    
    parser.add_argument('--output', type=str, 
                       default='models/baseline/evaluation_results.csv',
                       help='Path to save evaluation results')
    
    parser.add_argument('--plot', action='store_true',
                       help='Generate evaluation plots')
    
    parser.add_argument('--plot-path', type=str, 
                       default='models/baseline/evaluation_plots.png',
                       help='Path to save plots')
    
    return parser.parse_args()


def main():
    """Main evaluation pipeline"""
    
    print("=" * 70)
    print("MODEL EVALUATION PIPELINE")
    print("=" * 70)
    
    args = parse_arguments()
    
    print(f"\nConfiguration:")
    print(f"  Model file: {args.model}")
    print(f"  Data file: {args.data}")
    print(f"  Target: {args.target}")
    
    print("\n" + "=" * 70)
    print("STEP 1: LOAD MODEL")
    print("=" * 70)
    
    try:
        model, model_type = detect_and_load_model(args.model)
        print(f"✓ Successfully loaded {model_type} model")
    except FileNotFoundError:
        print(f"Error: Model file not found at {args.model}")
        print("Please train a model first using train.py")
        return
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    print("\n" + "=" * 70)
    print("STEP 2: LOAD DATA")
    print("=" * 70)
    
    X, y = load_energy_data(args.data, target=args.target)
    print(f"✓ Loaded data: {X.shape[0]} samples, {X.shape[1]} features")
    
    print("\n" + "=" * 70)
    print("STEP 3: MAKE PREDICTIONS")
    print("=" * 70)
    
    y_pred = model.predict(X)
    print(f"✓ Generated {len(y_pred)} predictions")
    
    print("\n" + "=" * 70)
    print("STEP 4: CALCULATE METRICS")
    print("=" * 70)
    
    metrics = calculate_all_metrics(y, y_pred, n_features=X.shape[1])
    print_metrics(metrics, f"{model_type} Model Evaluation")
    
    print("\n" + "=" * 70)
    print("STEP 5: SAVE RESULTS")
    print("=" * 70)
    
    results_df = pd.DataFrame({
        'Actual': y.values,
        'Predicted': y_pred,
        'Error': y.values - y_pred,
        'Absolute_Error': abs(y.values - y_pred),
        'Percentage_Error': ((y.values - y_pred) / y.values) * 100
    })
    
    results_df.to_csv(args.output, index=False)
    print(f"✓ Results saved to: {args.output}")
    
    if args.plot:
        print("\n" + "=" * 70)
        print("STEP 6: GENERATE PLOTS")
        print("=" * 70)
        
        plot_comprehensive_evaluation(y, y_pred, 
                                     model_name=f"{model_type} Model",
                                     save_path=args.plot_path)
        print(f"✓ Plots saved to: {args.plot_path}")
    
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETED")
    print("=" * 70)
    
    print(f"\nSummary ({model_type} Model):")
    print(f"  Samples evaluated: {len(y)}")
    print(f"  R² Score: {metrics['r2']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  MAE: {metrics['mae']:.4f}")
    print(f"  MAPE: {metrics['mape']:.2f}%")
    print(f"  Results saved to: {args.output}")
    if args.plot:
        print(f"  Plots saved to: {args.plot_path}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()