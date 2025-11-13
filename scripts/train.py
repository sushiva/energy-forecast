"""
Training Script
Main script to train the baseline energy forecasting model
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
from src.data.loader import load_energy_data
from src.data.splitter import DataSplitter
from src.models.baseline import BaselineModel
from src.evaluation.metrics import calculate_all_metrics, print_metrics
from src.evaluation.visualize import plot_comprehensive_evaluation


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train energy forecasting model')
    
    parser.add_argument('--data', type=str, 
                       default='data/raw/energy-efficiency-data.csv',
                       help='Path to data file')
    
    parser.add_argument('--target', type=str, 
                       default='Y1',
                       help='Target column name')
    
    parser.add_argument('--test-size', type=float, 
                       default=0.2,
                       help='Test set size (0.0 to 1.0)')
    
    parser.add_argument('--random-state', type=int, 
                       default=42,
                       help='Random seed for reproducibility')
    
    parser.add_argument('--model-path', type=str, 
                       default='models/baseline/trained_model.pkl',
                       help='Path to save the trained model')
    
    parser.add_argument('--plot-path', type=str, 
                       default='models/baseline/evaluation_plots.png',
                       help='Path to save evaluation plots')
    
    parser.add_argument('--no-plot', action='store_true',
                       help='Skip generating plots')
    
    return parser.parse_args()


def main():
    """Main training pipeline"""
    
    print("=" * 70)
    print("ENERGY FORECASTING MODEL - TRAINING PIPELINE")
    print("=" * 70)
    
    args = parse_arguments()
    
    print(f"\nConfiguration:")
    print(f"  Data file: {args.data}")
    print(f"  Target: {args.target}")
    print(f"  Test size: {args.test_size}")
    print(f"  Random state: {args.random_state}")
    print(f"  Model save path: {args.model_path}")
    
    print("\n" + "=" * 70)
    print("STEP 1: LOAD DATA")
    print("=" * 70)
    
    X, y = load_energy_data(args.data, target=args.target)
    
    print("\n" + "=" * 70)
    print("STEP 2: SPLIT DATA")
    print("=" * 70)
    
    splitter = DataSplitter(test_size=args.test_size, 
                           random_state=args.random_state)
    X_train, X_test, y_train, y_test = splitter.split(X, y)
    
    print("\n" + "=" * 70)
    print("STEP 3: TRAIN MODEL")
    print("=" * 70)
    
    model = BaselineModel()
    train_metrics = model.train(X_train, y_train)
    
    print_metrics(train_metrics, "Training Set")
    
    print("\n" + "=" * 70)
    print("STEP 4: EVALUATE ON TEST SET")
    print("=" * 70)
    
    test_metrics = model.evaluate(X_test, y_test)
    
    print("\n" + "=" * 70)
    print("STEP 5: SAVE MODEL")
    print("=" * 70)
    
    model.save(args.model_path)
    
    if not args.no_plot:
        print("\n" + "=" * 70)
        print("STEP 6: GENERATE EVALUATION PLOTS")
        print("=" * 70)
        
        y_pred = model.predict(X_test)
        plot_comprehensive_evaluation(y_test, y_pred, 
                                     model_name="Baseline Model",
                                     save_path=args.plot_path)
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 70)
    
    print("\nSummary:")
    print(f"  Model saved to: {args.model_path}")
    if not args.no_plot:
        print(f"  Plots saved to: {args.plot_path}")
    print(f"\n  Final Test Performance:")
    print(f"    R² Score: {test_metrics['r2']:.4f}")
    print(f"    RMSE: {test_metrics['rmse']:.4f}")
    print(f"    MAE: {test_metrics['mae']:.4f}")
    print(f"    MAPE: {test_metrics['mape']:.2f}%")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()