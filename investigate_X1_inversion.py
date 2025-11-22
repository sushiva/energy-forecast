"""
Investigate if X1 is inverted in the training data
"""
import joblib
import numpy as np
import pandas as pd

# Load the model data
model_data = joblib.load('models/advanced/xgboost_best.pkl')
X_train = model_data['X_train']
X_test = model_data['X_test']
y_train = model_data['y_train']
y_test = model_data['y_test']

print("=" * 70)
print("INVESTIGATING X1 (Relative Compactness) vs Energy Relationship")
print("=" * 70)

# Combine train and test for full picture
X_all = np.vstack([X_train, X_test])
y_all = np.concatenate([y_train, y_test])

# Extract X1 values
X1_values = X_all[:, 0]

print(f"\nX1 Statistics:")
print(f"  Min: {X1_values.min():.3f}")
print(f"  Max: {X1_values.max():.3f}")
print(f"  Mean: {X1_values.mean():.3f}")
print(f"  Median: {np.median(X1_values):.3f}")

print(f"\nEnergy Statistics:")
print(f"  Min: {y_all.min():.2f} kWh")
print(f"  Max: {y_all.max():.2f} kWh")
print(f"  Mean: {y_all.mean():.2f} kWh")
print(f"  Median: {np.median(y_all):.2f} kWh")

# Calculate correlation
correlation = np.corrcoef(X1_values, y_all)[0, 1]
print(f"\nCorrelation between X1 and Energy: {correlation:.3f}")

if correlation > 0:
    print("  ⚠️  POSITIVE correlation: Higher X1 → Higher Energy (INVERTED!)")
    print("  ⚠️  This means X1 is defined BACKWARDS in your data!")
elif correlation < 0:
    print("  ✅ NEGATIVE correlation: Higher X1 → Lower Energy (CORRECT!)")
else:
    print("  ❓ No correlation detected")

# Show some examples
print("\n" + "=" * 70)
print("SAMPLE DATA POINTS:")
print("=" * 70)

# Sort by X1 to see the pattern
sorted_indices = np.argsort(X1_values)

print("\n5 Buildings with LOWEST X1 (should be most elongated → HIGH energy):")
print("-" * 70)
for i in sorted_indices[:5]:
    print(f"  X1={X1_values[i]:.3f} → Energy={y_all[i]:.2f} kWh")

print("\n5 Buildings with HIGHEST X1 (should be most compact → LOW energy):")
print("-" * 70)
for i in sorted_indices[-5:]:
    print(f"  X1={X1_values[i]:.3f} → Energy={y_all[i]:.2f} kWh")

# Group analysis
print("\n" + "=" * 70)
print("GROUPED ANALYSIS:")
print("=" * 70)

# Split into quartiles
q1, q2, q3 = np.percentile(X1_values, [25, 50, 75])

low_x1_mask = X1_values <= q1
high_x1_mask = X1_values >= q3

low_x1_energy = y_all[low_x1_mask]
high_x1_energy = y_all[high_x1_mask]

print(f"\nBuildings with LOW X1 (≤{q1:.3f}, elongated):")
print(f"  Average Energy: {low_x1_energy.mean():.2f} kWh")
print(f"  Range: {low_x1_energy.min():.2f} - {low_x1_energy.max():.2f} kWh")

print(f"\nBuildings with HIGH X1 (≥{q3:.3f}, compact):")
print(f"  Average Energy: {high_x1_energy.mean():.2f} kWh")
print(f"  Range: {high_x1_energy.min():.2f} - {high_x1_energy.max():.2f} kWh")

energy_difference = high_x1_energy.mean() - low_x1_energy.mean()
print(f"\nDifference: {energy_difference:+.2f} kWh")

if energy_difference > 0:
    print("  ⚠️  Compact buildings use MORE energy (BACKWARDS!)")
elif energy_difference < 0:
    print("  ✅ Compact buildings use LESS energy (CORRECT!)")

print("\n" + "=" * 70)
print("DIAGNOSIS:")
print("=" * 70)

if correlation > 0.5:
    print("❌ PROBLEM CONFIRMED: X1 is INVERTED in your dataset!")
    print("\nPossible causes:")
    print("  1. X1 was calculated as: surface_area / volume (WRONG)")
    print("     Should be: volume / surface_area (CORRECT)")
    print("\n  2. Data source has X1 defined differently than expected")
    print("\n  3. Feature was accidentally inverted during preprocessing")
    
    print("\n" + "=" * 70)
    print("SOLUTIONS:")
    print("=" * 70)
    print("\n1. FIX THE DATA (Recommended):")
    print("   - Invert X1: X1_corrected = 1.0 / X1_original")
    print("   - Or recalculate: X1 = volume / surface_area")
    print("   - Retrain the model with corrected data")
    
    print("\n2. UPDATE YOUR README:")
    print("   - Change description: 'Higher X1 = more elongated' (not compact)")
    print("   - Update all SHAP interpretations accordingly")
    print("   - Document this unusual definition")
    
    print("\n3. FIX YOUR DASHBOARD:")
    print("   - If you keep inverted X1, dashboard is actually CORRECT!")
    print("   - But update labels: 'X1=1.0 means elongated building'")

elif correlation < -0.5:
    print("✅ DATA IS CORRECT: X1 follows standard definition")
    print("   Higher X1 → More compact → Lower energy")
    print("\nBut your MODEL learned it backwards somehow...")
    print("This shouldn't happen. Check your training code!")

else:
    print("❓ WEAK CORRELATION: X1 may not be a good predictor")
    print("   This contradicts the 85.3% importance we saw...")

# Check the original data source
print("\n" + "=" * 70)
print("NEXT STEP:")
print("=" * 70)
print("Check your original data file (data/raw/ or data/processed/)")
print("Look at the column definition for X1 (Relative Compactness)")
print("Share the data source documentation or first few rows")