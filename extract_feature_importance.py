#!/usr/bin/env python3
"""
Extract feature importance from XGBoost model - handles multiple save formats
"""

import os
import json

def try_load_model():
    """Try different methods to load the model"""
    model_path = 'models/advanced/xgboost_best.pkl'
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return None
    
    print(f"Found model file: {model_path}")
    print(f"File size: {os.path.getsize(model_path)} bytes\n")
    
    # Method 1: Try pickle
    print("Method 1: Trying pickle...")
    try:
        import pickle
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        print("✓ Loaded with pickle")
        return model_data, 'pickle'
    except Exception as e:
        print(f"✗ Pickle failed: {e}")
    
    # Method 2: Try joblib
    print("\nMethod 2: Trying joblib...")
    try:
        import joblib
        model_data = joblib.load(model_path)
        print("✓ Loaded with joblib")
        return model_data, 'joblib'
    except Exception as e:
        print(f"✗ Joblib failed: {e}")
    
    # Method 3: Try XGBoost native load
    print("\nMethod 3: Trying XGBoost native load...")
    try:
        import xgboost as xgb
        model = xgb.XGBRegressor()
        model.load_model(model_path)
        print("✓ Loaded with XGBoost native")
        return model, 'xgboost_native'
    except Exception as e:
        print(f"✗ XGBoost native failed: {e}")
    
    # Method 4: Check if it's JSON
    print("\nMethod 4: Checking if it's JSON...")
    try:
        with open(model_path, 'r') as f:
            model_data = json.load(f)
        print("✓ Loaded as JSON")
        return model_data, 'json'
    except Exception as e:
        print(f"✗ JSON failed: {e}")
    
    return None, None

def extract_importance(model_data, load_method):
    """Extract feature importance from model"""
    import numpy as np
    import pandas as pd
    
    print("\n" + "="*60)
    print("EXTRACTING FEATURE IMPORTANCE")
    print("="*60)
    
    try:
        # Handle different model formats
        if isinstance(model_data, dict):
            print("Model is a dictionary, looking for model key...")
            xgb_model = model_data.get('model', model_data.get('xgboost', model_data))
        else:
            xgb_model = model_data
        
        # Get feature importance
        if hasattr(xgb_model, 'feature_importances_'):
            importance = xgb_model.feature_importances_
        elif hasattr(xgb_model, 'get_score'):
            # XGBoost native format
            importance_dict = xgb_model.get_score(importance_type='gain')
            # Convert to array
            importance = np.array([importance_dict.get(f'f{i}', 0) for i in range(8)])
        else:
            print("Could not find feature_importances_ attribute")
            print(f"Model type: {type(xgb_model)}")
            print(f"Available attributes: {dir(xgb_model)}")
            return
        
        # Feature names
        feature_names = [f'X{i}' for i in range(1, len(importance) + 1)]
        
        # Create dataframe
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance,
            'Percentage': (importance / importance.sum() * 100)
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('Importance', ascending=False)
        
        print("\n" + "="*60)
        print("FEATURE IMPORTANCE RANKING")
        print("="*60)
        
        for idx, row in importance_df.iterrows():
            print(f"{row['Feature']}: {row['Percentage']:.1f}% (importance: {row['Importance']:.4f})")
        
        print("\n" + "="*60)
        print("TOP 5 FEATURES FOR README")
        print("="*60)
        
        for i, (idx, row) in enumerate(importance_df.head(5).iterrows(), 1):
            print(f"{i}. **{row['Feature']}** - {row['Percentage']:.1f}%")
        
        print("\n")
        
    except Exception as e:
        print(f"Error extracting importance: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("XGBoost Feature Importance Extractor")
    print("="*60)
    
    model_data, load_method = try_load_model()
    
    if model_data is not None:
        extract_importance(model_data, load_method)
    else:
        print("\n" + "="*60)
        print("ALTERNATIVE: CHECK YOUR TRAINING SCRIPT")
        print("="*60)
        print("\nCould not load the model file.")
        print("Please check your training script (scripts/demo_neural_network.py)")
        print("or look at the console output when training completed.")
        print("\nYou should see feature importance printed there.")
        print("\nOr run the training script again and capture the output:")
        print("  python scripts/demo_neural_network.py | grep -A 10 'Feature Importance'")