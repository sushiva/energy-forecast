"""
Diagnostic script to determine actual feature importance
Handles models saved as dictionaries
"""
import joblib
import numpy as np
import pandas as pd

model_path = 'models/advanced/xgboost_best.pkl'

print("Loading model...")
print(f"Model path: {model_path}")
print("=" * 60)

# Load the model
loaded_object = joblib.load(model_path)

# Check what we loaded
print("\nType of loaded object:", type(loaded_object))
print("\nContents of loaded object:")
print("-" * 60)

if isinstance(loaded_object, dict):
    print("✓ Loaded object is a dictionary")
    print("\nKeys in dictionary:", list(loaded_object.keys()))
    print("\n")
    
    # Print what's in each key
    for key, value in loaded_object.items():
        print(f"Key: '{key}'")
        print(f"  Type: {type(value)}")
        if hasattr(value, 'shape'):
            print(f"  Shape: {value.shape}")
        print()
    
    # Try to find the model
    model = None
    if 'model' in loaded_object:
        model = loaded_object['model']
        print("✓ Found model in 'model' key")
    elif 'xgboost_model' in loaded_object:
        model = loaded_object['xgboost_model']
        print("✓ Found model in 'xgboost_model' key")
    else:
        # Try to find XGBoost model by type
        for key, value in loaded_object.items():
            if 'XGB' in str(type(value)) or 'Booster' in str(type(value)):
                model = value
                print(f"✓ Found XGBoost model in '{key}' key")
                break
    
    if model is None:
        print("\n❌ Could not find XGBoost model in dictionary")
        print("\nPlease tell me which key contains the model, or")
        print("share the training script that saves this file")
        exit(1)
else:
    model = loaded_object
    print("✓ Loaded object is the model directly")

print("\n" + "=" * 60)
print("MODEL INFORMATION")
print("=" * 60)
print(f"Model type: {type(model)}")

# Now extract feature importance
print("\n" + "=" * 60)
print("FEATURE IMPORTANCE DIAGNOSTICS")
print("=" * 60)

# Try different methods to get feature importance
try:
    print("\n1. XGBoost Built-in Feature Importance (gain):")
    print("-" * 60)
    
    # Get the booster
    if hasattr(model, 'get_booster'):
        booster = model.get_booster()
    elif hasattr(model, 'booster'):
        booster = model.booster
    else:
        booster = model
    
    importance_gain = booster.get_score(importance_type='gain')
    importance_gain_sorted = sorted(importance_gain.items(), key=lambda x: x[1], reverse=True)
    
    total_gain = sum([v for k, v in importance_gain_sorted])
    
    print("\nRANKING BY GAIN (most important):")
    for i, (feature, score) in enumerate(importance_gain_sorted, 1):
        percentage = (score / total_gain) * 100
        print(f"  {i}. {feature}: {score:.2f} ({percentage:.1f}%)")
    
except Exception as e:
    print(f"❌ Error getting gain importance: {e}")

try:
    print("\n2. XGBoost Feature Importance (weight/frequency):")
    print("-" * 60)
    
    if hasattr(model, 'get_booster'):
        booster = model.get_booster()
    elif hasattr(model, 'booster'):
        booster = model.booster
    else:
        booster = model
        
    importance_weight = booster.get_score(importance_type='weight')
    importance_weight_sorted = sorted(importance_weight.items(), key=lambda x: x[1], reverse=True)
    
    print("\nRANKING BY WEIGHT (most used):")
    for i, (feature, score) in enumerate(importance_weight_sorted, 1):
        print(f"  {i}. {feature}: {score:.0f} times")
        
except Exception as e:
    print(f"❌ Error getting weight importance: {e}")

# Try to get feature names
try:
    print("\n3. Feature Names:")
    print("-" * 60)
    
    if hasattr(model, 'feature_names_in_'):
        feature_names = list(model.feature_names_in_)
        print(f"Feature names from model: {feature_names}")
    elif hasattr(model, 'feature_names'):
        feature_names = list(model.feature_names)
        print(f"Feature names from model: {feature_names}")
    elif isinstance(loaded_object, dict) and 'feature_names' in loaded_object:
        feature_names = loaded_object['feature_names']
        print(f"Feature names from dict: {feature_names}")
    else:
        print("Could not find feature names in model")
        
except Exception as e:
    print(f"❌ Error getting feature names: {e}")

print("\n" + "=" * 60)
print("✅ DIAGNOSTIC COMPLETE")
print("=" * 60)
print("\nNEXT STEPS:")
print("1. Compare rankings above with your feature importance graph")
print("2. Share this output with me")
print("3. Tell me which features (X1-X8) correspond to which building attributes")