"""
Complete Training Script - Saves Model Properly for SHAP
Run this to re-train your model and enable SHAP visualizations
"""
import joblib   
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

print("="*70)
print("TRAINING XGBOOST MODEL WITH PROPER SAVING FOR SHAP")
print("="*70)

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
print("\n1. Loading data...")
df = pd.read_csv('data/raw/energy-efficiency-data.csv')
print(f"✓ Loaded data: {df.shape}")
print(f"  Columns: {list(df.columns)}")

# ============================================================================
# STEP 2: DEFINE FEATURES AND TARGET
# ============================================================================
print("\n2. Defining features and target...")

# Features: X1-X8
feature_columns = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8']

# Target: Y1 (Heating Load) or Y2 (Cooling Load)
target_column = 'Y1'  # Change to 'Y2' if you want cooling load

print(f"✓ Features ({len(feature_columns)}): {feature_columns}")
print(f"✓ Target: {target_column}")

# ============================================================================
# STEP 3: CREATE TRAIN/TEST SPLIT
# ============================================================================
print("\n3. Creating train/test split (80/20)...")

train_size = int(len(df) * 0.8)

X_train = df[feature_columns].iloc[:train_size]
X_test = df[feature_columns].iloc[train_size:]
y_train = df[target_column].iloc[:train_size]
y_test = df[target_column].iloc[train_size:]

print(f"✓ Train set: {X_train.shape}")
print(f"✓ Test set: {X_test.shape}")

# ============================================================================
# STEP 4: TRAIN MODEL
# ============================================================================
print("\n4. Training XGBoost model...")

model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)
print("✓ Model trained!")

# ============================================================================
# STEP 5: EVALUATE MODEL
# ============================================================================
print("\n5. Evaluating model...")

y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

print(f"\n📊 MODEL PERFORMANCE:")
print(f"   R² Score:  {r2:.4f} ({r2*100:.2f}%)")
print(f"   RMSE:      {rmse:.2f} kWh")
print(f"   MAE:       {mae:.2f} kWh")
print(f"   MAPE:      {mape:.2f}%")

# ============================================================================
# STEP 6: SAVE MODEL WITH DATA (CRITICAL FOR SHAP!)
# ============================================================================
print("\n6. Saving model with training data...")

# Create models directory
output_dir = Path('models/advanced')
output_dir.mkdir(parents=True, exist_ok=True)

# Save model WITH data - this is the key for SHAP!
model_data = {
    'model': model,
    'X_train': X_train.values,        # Training features as numpy array
    'X_test': X_test.values,          # Test features as numpy array
    'y_train': y_train.values,        # Training target
    'y_test': y_test.values,          # Test target
    'feature_names': feature_columns, # Feature names for SHAP
    'target_name': target_column,     # Target name
    'performance': {                   # Model metrics
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
        'mape': mape
    }
}

model_path = output_dir / 'xgboost_best.pkl'
with open(model_path, 'wb') as f:
    joblib.dump(model_data, model_path)

print(f"✓ Model saved to: {model_path}")
print(f"  Includes: model, training data, test data, feature names")

# ============================================================================
# STEP 7: VERIFY MODEL CAN BE LOADED
# ============================================================================
print("\n7. Verifying saved model...")


loaded_data = joblib.load(model_path)

loaded_model = loaded_data['model']
test_pred = loaded_model.predict(loaded_data['X_test'][:5])

print(f"✓ Model loaded successfully")
print(f"  Sample predictions: {test_pred[:3]}")

# ============================================================================
# DONE!
# ============================================================================
print("\n" + "="*70)
print("✨ SUCCESS! MODEL READY FOR SHAP")
print("="*70)

print(f"""
Your model has been trained and saved with all necessary data!

📁 Saved to: {model_path}

📊 Performance:
   • R² Score: {r2*100:.2f}%
   • RMSE: {rmse:.2f} kWh
   • MAE: {mae:.2f} kWh

🎯 Next Step: Generate SHAP Visualizations

Run this command:
    python simple_shap.py

This will generate ~25 beautiful SHAP visualizations in about 5 minutes!

📁 Output location: visualizations/shap_analysis/

🎉 You're all set!
""")

print("="*70)
