"""
Test what the model SHOULD predict for X1=1.0 (maximum compactness)
"""
import joblib
import numpy as np
import shap

# Load the actual model
model_data = joblib.load('models/advanced/xgboost_best.pkl')
model = model_data['model']
X_train = model_data['X_train']
X_test = model_data['X_test']

print("=" * 60)
print("TESTING X1 = 1.0 (Maximum Compactness)")
print("=" * 60)

# Get a sample from test set to use as template
sample = X_test[0].copy()
print(f"\nOriginal sample features: {sample}")
print(f"Original sample X1: {sample[0]:.3f}")

# Create a test case with X1 = 1.0 (maximum compactness)
test_case_max_compact = sample.copy()
test_case_max_compact[0] = 1.0  # Set X1 to maximum compactness

print(f"\nTest case (X1=1.0): {test_case_max_compact}")

# Make prediction
prediction_max_compact = model.predict([test_case_max_compact])[0]
print(f"\nPrediction with X1=1.0: {prediction_max_compact:.2f} kWh")

# Calculate SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values([test_case_max_compact])

print(f"\nSHAP values for this prediction:")
feature_names = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8']
for i, (name, shap_val) in enumerate(zip(feature_names, shap_values[0])):
    direction = "↑ INCREASE" if shap_val > 0 else "↓ DECREASE"
    print(f"  {name}: {shap_val:+.2f} kWh ({direction})")

print(f"\nBase value: {explainer.expected_value:.2f} kWh")
print(f"Prediction: {prediction_max_compact:.2f} kWh")
print(f"Calculation check: {explainer.expected_value:.2f} + {sum(shap_values[0]):.2f} = {explainer.expected_value + sum(shap_values[0]):.2f}")

print("\n" + "=" * 60)
print("EXPECTED BEHAVIOR:")
print("=" * 60)
print("✅ X1 SHAP should be NEGATIVE (pushing energy DOWN)")
print("✅ Prediction should be LOW (~10-20 kWh)")
print("\nIf your dashboard shows:")
print("❌ X1 SHAP = POSITIVE (+5.50)")
print("❌ Prediction = HIGH (28.62 kWh)")
print("\nThen there's a bug in your dashboard!")

print("\n" + "=" * 60)
print("NOW TEST X1 = 0.6 (Minimum Compactness)")
print("=" * 60)

# Create a test case with X1 = 0.6 (minimum compactness)
test_case_min_compact = sample.copy()
test_case_min_compact[0] = 0.6  # Set X1 to minimum compactness

print(f"\nTest case (X1=0.6): {test_case_min_compact}")

# Make prediction
prediction_min_compact = model.predict([test_case_min_compact])[0]
print(f"\nPrediction with X1=0.6: {prediction_min_compact:.2f} kWh")

# Calculate SHAP values
shap_values_min = explainer.shap_values([test_case_min_compact])

print(f"\nSHAP values for this prediction:")
for i, (name, shap_val) in enumerate(zip(feature_names, shap_values_min[0])):
    direction = "↑ INCREASE" if shap_val > 0 else "↓ DECREASE"
    print(f"  {name}: {shap_val:+.2f} kWh ({direction})")

print(f"\nPrediction: {prediction_min_compact:.2f} kWh")

print("\n" + "=" * 60)
print("EXPECTED BEHAVIOR:")
print("=" * 60)
print("✅ X1 SHAP should be POSITIVE (pushing energy UP)")
print("✅ Prediction should be HIGH (~30-45 kWh)")

print("\n" + "=" * 60)
print("COMPARISON:")
print("=" * 60)
print(f"X1=1.0 (compact):     {prediction_max_compact:.2f} kWh (should be LOW)")
print(f"X1=0.6 (elongated):   {prediction_min_compact:.2f} kWh (should be HIGH)")
print(f"Difference:           {prediction_min_compact - prediction_max_compact:.2f} kWh")
print("\nThis difference should be ~20-30 kWh (X1 is 85% important!)")