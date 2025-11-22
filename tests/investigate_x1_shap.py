"""
Investigate X1 SHAP behavior across its full range
This will show us if there's a non-linear relationship or outlier
"""

import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# Load model
model_data = joblib.load('models/advanced/xgboost_best.pkl')
model = model_data['model']
explainer = shap.TreeExplainer(model)

print("=" * 70)
print("INVESTIGATING X1 SHAP BEHAVIOR")
print("=" * 70)

# Base feature values (holding all others constant)
base_features = [637, 318, 147, 5.25, 3, 0.25, 2]

# Test X1 across its full range
x1_values = np.linspace(1.02, 1.61, 20)
predictions = []
x1_shaps = []

print("\nX1 value | Prediction | X1 SHAP | Expected SHAP Direction")
print("-" * 70)

for x1 in x1_values:
    # Create test sample
    test = np.array([[x1] + base_features])
    
    # Get prediction
    pred = model.predict(test)[0]
    
    # Get SHAP values
    shap_values = explainer.shap_values(test)
    x1_shap = shap_values[0][0]
    
    # Expected direction
    # X1 > 1.35 (average) should be negative (compact)
    # X1 < 1.35 should be positive (elongated)
    expected = "NEGATIVE (compact)" if x1 > 1.35 else "POSITIVE (elongated)"
    status = "✅" if (x1 > 1.35 and x1_shap < 0) or (x1 <= 1.35 and x1_shap > 0) else "❌"
    
    print(f"{x1:.3f}    | {pred:6.2f} kWh | {x1_shap:+7.2f} kWh | {expected} {status}")
    
    predictions.append(pred)
    x1_shaps.append(x1_shap)

# Create visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: X1 vs Prediction
ax1.plot(x1_values, predictions, 'b-', linewidth=2, marker='o')
ax1.axvline(x=1.30, color='r', linestyle='--', label='X1=1.30 (your scenario)')
ax1.set_xlabel('X1 (Relative Compactness)', fontsize=12)
ax1.set_ylabel('Prediction (kWh)', fontsize=12)
ax1.set_title('X1 vs Energy Prediction', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Plot 2: X1 vs SHAP
ax2.plot(x1_values, x1_shaps, 'r-', linewidth=2, marker='o')
ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
ax2.axvline(x=1.30, color='r', linestyle='--', label='X1=1.30')
ax2.set_xlabel('X1 (Relative Compactness)', fontsize=12)
ax2.set_ylabel('X1 SHAP Value (kWh)', fontsize=12)
ax2.set_title('X1 vs SHAP Contribution', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.savefig('x1_behavior_analysis.png', dpi=300, bbox_inches='tight')
print("\n✅ Plot saved as: x1_behavior_analysis.png")

# Summary statistics
print("\n" + "=" * 70)
print("SUMMARY STATISTICS")
print("=" * 70)

print(f"\nPrediction Range:")
print(f"  Min: {min(predictions):.2f} kWh at X1={x1_values[np.argmin(predictions)]:.3f}")
print(f"  Max: {max(predictions):.2f} kWh at X1={x1_values[np.argmax(predictions)]:.3f}")

print(f"\nX1 SHAP Range:")
print(f"  Min: {min(x1_shaps):.2f} kWh at X1={x1_values[np.argmin(x1_shaps)]:.3f}")
print(f"  Max: {max(x1_shaps):.2f} kWh at X1={x1_values[np.argmax(x1_shaps)]:.3f}")

# Check for anomalies
print("\n" + "=" * 70)
print("ANOMALY DETECTION")
print("=" * 70)

# The SHAP value at X1=1.30
idx_130 = np.argmin(np.abs(x1_values - 1.30))
shap_at_130 = x1_shaps[idx_130]

print(f"\nAt X1=1.30:")
print(f"  SHAP: {shap_at_130:+.2f} kWh")
print(f"  Prediction: {predictions[idx_130]:.2f} kWh")

# Check if SHAP is monotonic (should be!)
is_monotonic = all(x1_shaps[i] >= x1_shaps[i+1] for i in range(len(x1_shaps)-1))
if is_monotonic:
    print("\n✅ X1 SHAP is monotonic (decreasing as X1 increases) - GOOD!")
else:
    print("\n❌ X1 SHAP is NOT monotonic - this is WEIRD!")
    
    # Find where it's not monotonic
    for i in range(len(x1_shaps)-1):
        if x1_shaps[i] < x1_shaps[i+1]:
            print(f"  Anomaly at X1={x1_values[i]:.3f} to {x1_values[i+1]:.3f}")

# Check the slope
slope = (x1_shaps[-1] - x1_shaps[0]) / (x1_values[-1] - x1_values[0])
print(f"\nOverall SHAP slope: {slope:.2f} kWh per unit X1")
print(f"(Negative slope is expected: higher X1 → lower SHAP)")

# Check correlation between X1 and prediction
correlation = np.corrcoef(x1_values, predictions)[0, 1]
print(f"\nCorrelation X1 vs Prediction: {correlation:.3f}")
if correlation > 0:
    print("  ⚠️  POSITIVE correlation: Higher X1 → Higher energy (INVERTED!)")
elif correlation < 0:
    print("  ✅ NEGATIVE correlation: Higher X1 → Lower energy (CORRECT!)")

print("\n" + "=" * 70)