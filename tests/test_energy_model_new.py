import numpy as np
import joblib
import shap

model_data = joblib.load('models/advanced/xgboost_best.pkl')
model = model_data['model']
explainer = shap.TreeExplainer(model)

# Your exact scenario C
test = np.array([[1.30, 637, 318, 147, 5.25, 3, 0.25, 2]])

# Prediction
pred = model.predict(test)[0]
print(f"Prediction: {pred:.2f} kWh")

# SHAP values
shap_values = explainer.shap_values(test)
print(f"\nSHAP values:")
features = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8']
for i, (feat, val) in enumerate(zip(features, shap_values[0])):
    print(f"  {feat}: {val:+.2f} kWh")

print(f"\nBase: {explainer.expected_value:.2f} kWh")
print(f"Sum of SHAP: {sum(shap_values[0]):.2f} kWh")
print(f"Base + SHAP = {explainer.expected_value + sum(shap_values[0]):.2f} kWh")
