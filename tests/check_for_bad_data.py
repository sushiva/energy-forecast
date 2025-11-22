import joblib
import numpy as np
import matplotlib.pyplot as plt

model_data = joblib.load('models/advanced/xgboost_best.pkl')
X_train = model_data['X_train']
y_train = model_data['y_train']

x1_train = X_train[:, 0]

# Focus on the anomaly region (X1 = 1.25 to 1.35)
mask_anomaly = (x1_train >= 1.25) & (x1_train <= 1.35)

print("=" * 70)
print("ANALYZING TRAINING DATA IN ANOMALY REGION (X1 = 1.25-1.35)")
print("=" * 70)

print(f"\nSamples in this range: {mask_anomaly.sum()}")
print(f"Percentage of total: {mask_anomaly.sum() / len(x1_train) * 100:.1f}%")

if mask_anomaly.sum() > 0:
    print(f"\nEnergy statistics in X1=1.25-1.35:")
    print(f"  Mean: {y_train[mask_anomaly].mean():.2f} kWh")
    print(f"  Std: {y_train[mask_anomaly].std():.2f} kWh")
    print(f"  Min: {y_train[mask_anomaly].min():.2f} kWh")
    print(f"  Max: {y_train[mask_anomaly].max():.2f} kWh")
    
    print(f"\nAll samples in this range:")
    anomaly_samples = X_train[mask_anomaly]
    anomaly_energy = y_train[mask_anomaly]
    
    for i, (features, energy) in enumerate(zip(anomaly_samples, anomaly_energy)):
        print(f"  Sample {i+1}: X1={features[0]:.3f}, Energy={energy:.2f} kWh")
        print(f"            Other features: X2={features[1]:.0f}, X3={features[2]:.0f}, "
              f"X4={features[3]:.0f}, X5={features[4]:.2f}, X6={features[5]:.0f}, "
              f"X7={features[6]:.2f}, X8={features[7]:.0f}")
else:
    print("\n⚠️ NO TRAINING DATA in X1=1.25-1.35 range!")
    print("   Model is EXTRAPOLATING/INTERPOLATING!")

# Compare with nearby regions
mask_before = (x1_train >= 1.15) & (x1_train < 1.25)
mask_after = (x1_train > 1.35) & (x1_train <= 1.45)

print(f"\nComparison with nearby regions:")
print(f"X1=1.15-1.25: {mask_before.sum()} samples, avg energy = {y_train[mask_before].mean():.2f} kWh")
print(f"X1=1.25-1.35: {mask_anomaly.sum()} samples, avg energy = {y_train[mask_anomaly].mean():.2f} kWh" if mask_anomaly.sum() > 0 else "X1=1.25-1.35: NO DATA")
print(f"X1=1.35-1.45: {mask_after.sum()} samples, avg energy = {y_train[mask_after].mean():.2f} kWh")

# Visualize
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(x1_train, y_train, alpha=0.3, s=20)
plt.axvspan(1.25, 1.35, alpha=0.3, color='red', label='Anomaly region')
plt.axvline(x=1.30, color='red', linestyle='--', linewidth=2, label='X1=1.30')
plt.xlabel('X1 (Relative Compactness)')
plt.ylabel('Energy (kWh)')
plt.title('Training Data Distribution')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.hist(x1_train, bins=50, edgecolor='black')
plt.axvspan(1.25, 1.35, alpha=0.3, color='red', label='Anomaly region')
plt.axvline(x=1.30, color='red', linestyle='--', linewidth=2, label='X1=1.30')
plt.xlabel('X1 (Relative Compactness)')
plt.ylabel('Count')
plt.title('X1 Distribution in Training Data')
plt.legend()

plt.tight_layout()
plt.savefig('training_data_analysis.png', dpi=300)
print("\n✅ Saved: training_data_analysis.png")