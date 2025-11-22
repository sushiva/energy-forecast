"""
Fix inverted X1 (Relative Compactness) and retrain model

This script:
1. Loads the current model data
2. Inverts X1 to correct definition (higher = more compact)
3. Retrains XGBoost model
4. Saves corrected model
5. Validates the fix worked
"""
import joblib
import numpy as np
import xgboost as xgb
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import shap

print("=" * 70)
print("FIXING INVERTED X1 AND RETRAINING MODEL")
print("=" * 70)

# Load current model data
print("\n1. Loading current model data...")
model_data = joblib.load('models/advanced/xgboost_best.pkl')
X_train = model_data['X_train'].copy()
X_test = model_data['X_test'].copy()
y_train = model_data['y_train']
y_test = model_data['y_test']
feature_names = model_data['feature_names']
target_name = model_data['target_name']

print(f"   Training samples: {len(X_train)}")
print(f"   Test samples: {len(X_test)}")

# Show current X1 range
print(f"\n   Current X1 range: {X_train[:, 0].min():.3f} to {X_train[:, 0].max():.3f}")
print(f"   Current correlation: {np.corrcoef(X_train[:, 0], y_train)[0, 1]:.3f}")

# Fix X1: Invert it
print("\n2. Inverting X1 (Column 0)...")
X_train_fixed = X_train.copy()
X_test_fixed = X_test.copy()

# Invert X1: new = 1 / old
# This makes: high value = compact (correct), low value = elongated (correct)
X_train_fixed[:, 0] = 1.0 / X_train[:, 0]
X_test_fixed[:, 0] = 1.0 / X_test[:, 0]

print(f"   NEW X1 range: {X_train_fixed[:, 0].min():.3f} to {X_train_fixed[:, 0].max():.3f}")
print(f"   NEW correlation: {np.corrcoef(X_train_fixed[:, 0], y_train)[0, 1]:.3f}")
print("   ✅ Correlation should now be NEGATIVE (higher X1 → lower energy)")

# Retrain XGBoost model with same hyperparameters
print("\n3. Retraining XGBoost model...")

# Use same hyperparameters as original model
original_model = model_data['model']
params = original_model.get_params()

print(f"   Using hyperparameters: n_estimators={params.get('n_estimators', 100)}, "
      f"max_depth={params.get('max_depth', 6)}, learning_rate={params.get('learning_rate', 0.1)}")

# Create and train new model
model_fixed = xgb.XGBRegressor(
    n_estimators=params.get('n_estimators', 100),
    max_depth=params.get('max_depth', 6),
    learning_rate=params.get('learning_rate', 0.1),
    random_state=42,
    objective='reg:squarederror'
)

model_fixed.fit(X_train_fixed, y_train)
print("   ✅ Model training complete")

# Evaluate fixed model
print("\n4. Evaluating fixed model...")
y_pred_train = model_fixed.predict(X_train_fixed)
y_pred_test = model_fixed.predict(X_test_fixed)

r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)
mae_test = mean_absolute_error(y_test, y_pred_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

print(f"   Training R²: {r2_train:.4f} ({r2_train*100:.2f}%)")
print(f"   Test R²: {r2_test:.4f} ({r2_test*100:.2f}%)")
print(f"   Test MAE: {mae_test:.2f} kWh")
print(f"   Test RMSE: {rmse_test:.2f} kWh")

# Compare with original model performance
if 'performance' in model_data:
    orig_perf = model_data['performance']
    print(f"\n   Original model R²: {orig_perf.get('test_r2', 'N/A')}")
    print("   ✅ Performance should be similar (data just transformed)")

# Check feature importance
print("\n5. Checking feature importance...")
booster = model_fixed.get_booster()
importance_gain = booster.get_score(importance_type='gain')
importance_sorted = sorted(importance_gain.items(), key=lambda x: x[1], reverse=True)

total_gain = sum([v for k, v in importance_sorted])
print("\n   Top 5 Features (by gain):")
for i, (feature, score) in enumerate(importance_sorted[:5], 1):
    percentage = (score / total_gain) * 100
    print(f"   {i}. {feature}: {percentage:.1f}%")

print("\n   ✅ X1 should still be dominant (~85%)")

# Validate the fix: Test X1 behavior
print("\n6. Validating X1 behavior correction...")

# Create test cases
test_compact = X_test_fixed[0].copy()
test_compact[0] = X_train_fixed[:, 0].max()  # High X1 = compact

test_elongated = X_test_fixed[0].copy()
test_elongated[0] = X_train_fixed[:, 0].min()  # Low X1 = elongated

pred_compact = model_fixed.predict([test_compact])[0]
pred_elongated = model_fixed.predict([test_elongated])[0]

print(f"\n   Compact building (X1={test_compact[0]:.3f}): {pred_compact:.2f} kWh")
print(f"   Elongated building (X1={test_elongated[0]:.3f}): {pred_elongated:.2f} kWh")
print(f"   Difference: {pred_elongated - pred_compact:+.2f} kWh")

if pred_compact < pred_elongated:
    print("   ✅ CORRECT: Compact buildings use LESS energy")
else:
    print("   ❌ ERROR: Still inverted somehow!")

# Calculate SHAP values for validation
print("\n7. Validating SHAP behavior...")
explainer = shap.TreeExplainer(model_fixed)

shap_compact = explainer.shap_values([test_compact])
shap_elongated = explainer.shap_values([test_elongated])

print(f"\n   SHAP for X1 in compact building: {shap_compact[0][0]:+.2f} kWh")
print(f"   SHAP for X1 in elongated building: {shap_elongated[0][0]:+.2f} kWh")

if shap_compact[0][0] < 0 and shap_elongated[0][0] > 0:
    print("   ✅ CORRECT: X1 SHAP values have correct signs")
else:
    print("   ⚠️  Check: SHAP signs may vary depending on other features")

# Save corrected model
print("\n8. Saving corrected model...")

model_data_fixed = {
    'model': model_fixed,
    'X_train': X_train_fixed,
    'X_test': X_test_fixed,
    'y_train': y_train,
    'y_test': y_test,
    'feature_names': feature_names,
    'target_name': target_name,
    'performance': {
        'train_r2': r2_train,
        'test_r2': r2_test,
        'test_mae': mae_test,
        'test_rmse': rmse_test
    },
    'note': 'X1 has been corrected: higher value = more compact (standard definition)'
}

# Save as new file first (don't overwrite original)
joblib.dump(model_data_fixed, 'models/advanced/xgboost_best_FIXED.pkl')
print("   ✅ Saved as: models/advanced/xgboost_best_FIXED.pkl")

print("\n" + "=" * 70)
print("FIX COMPLETE!")
print("=" * 70)

print("\nNEXT STEPS:")
print("1. ✅ Test the fixed model: python test_x1_behavior.py")
print("      (Update script to load xgboost_best_FIXED.pkl)")
print("\n2. ✅ Update dashboard to use: models/advanced/xgboost_best_FIXED.pkl")
print("\n3. ✅ Backup original model:")
print("      mv models/advanced/xgboost_best.pkl models/advanced/xgboost_best_INVERTED_backup.pkl")
print("\n4. ✅ Use fixed model as main:")
print("      mv models/advanced/xgboost_best_FIXED.pkl models/advanced/xgboost_best.pkl")
print("\n5. ✅ README is already correct! (X1 = Relative Compactness, higher = more compact)")
print("\n6. ✅ Regenerate feature importance graph with fixed model")

print("\n" + "=" * 70)
print("IMPORTANT: X1 Range Has Changed!")
print("=" * 70)
print(f"OLD range: 0.620 to 0.980")
print(f"NEW range: {X_train_fixed[:, 0].min():.3f} to {X_train_fixed[:, 0].max():.3f}")
print("\nUpdate your dashboard sliders to use NEW range!")