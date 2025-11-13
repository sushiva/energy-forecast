"""
Toy Example: Log Transformation to Ensure Non-Negative Predictions
====================================================================
Let's see it with REAL numbers!
"""

import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd

print("=" * 70)
print("TOY EXAMPLE: Why Log Transformation Prevents Negative Predictions")
print("=" * 70)

# ============================================================================
# STEP 1: Create tiny dataset (just 5 rows!)
# ============================================================================
print("\n📊 STEP 1: Our Tiny Dataset")
print("-" * 70)

# Features: Temperature and Hour
X = np.array([
    [20, 10],  # 20°C at 10 AM
    [25, 14],  # 25°C at 2 PM
    [15, 22],  # 15°C at 10 PM
    [30, 12],  # 30°C at noon
    [18, 8]    # 18°C at 8 AM
])

# Target: Energy consumption (kWh)
y = np.array([50, 80, 30, 100, 40])

df = pd.DataFrame(X, columns=['Temperature', 'Hour'])
df['Energy (kWh)'] = y

print(df)
print(f"\n✅ All energy values are positive (as they should be!)")
print(f"   Min: {y.min()} kWh, Max: {y.max()} kWh")

# ============================================================================
# STEP 2: Split into train/test (super simple)
# ============================================================================
print("\n\n📚 STEP 2: Split Data")
print("-" * 70)

X_train, X_test = X[:4], X[4:]  # First 4 for training, last 1 for testing
y_train, y_test = y[:4], y[4:]

print(f"Training data: {len(X_train)} samples")
print(f"Test data: {len(X_test)} sample")
print(f"\nTest sample: Temp={X_test[0][0]}°C, Hour={X_test[0][1]}, Actual Energy={y_test[0]} kWh")

# ============================================================================
# APPROACH A: WITHOUT Log Transformation (Can give negative!)
# ============================================================================
print("\n\n" + "=" * 70)
print("❌ APPROACH A: WITHOUT Log Transformation")
print("=" * 70)

model_regular = LinearRegression()
model_regular.fit(X_train, y_train)

# Predict
y_pred_regular = model_regular.predict(X_test)

print(f"\n🔮 Prediction: {y_pred_regular[0]:.2f} kWh")
print(f"📌 Actual: {y_test[0]} kWh")

# Show the equation
print(f"\n📐 Model equation:")
print(f"   Energy = {model_regular.intercept_:.2f} + "
      f"{model_regular.coef_[0]:.2f}×Temp + "
      f"{model_regular.coef_[1]:.2f}×Hour")

print(f"\n⚠️  Problem: With different data, this COULD predict negative values!")
print(f"   (There's no mathematical constraint preventing it)")

# ============================================================================
# APPROACH B: WITH Log Transformation (ALWAYS positive!)
# ============================================================================
print("\n\n" + "=" * 70)
print("✅ APPROACH B: WITH Log Transformation")
print("=" * 70)

# Transform target to log space
y_train_log = np.log1p(y_train)  # log(1 + y)

print(f"\n🔄 Step 1: Transform training targets to LOG space")
print(f"   Original y_train: {y_train}")
print(f"   Log-transformed:  {y_train_log.round(3)}")

# Train model in log space
model_log = LinearRegression()
model_log.fit(X_train, y_train_log)

print(f"\n🎓 Step 2: Train model to predict LOG(energy)")

# Predict in log space
y_pred_log = model_log.predict(X_test)

print(f"\n🔮 Step 3: Model predicts (in log space): {y_pred_log[0]:.3f}")

# Transform back to original space
y_pred_final = np.expm1(y_pred_log)  # exp(pred) - 1

print(f"   Transform back to original: exp({y_pred_log[0]:.3f}) - 1 = {y_pred_final[0]:.2f} kWh")
print(f"📌 Actual: {y_test[0]} kWh")

print(f"\n✨ Magic: exp() of ANY number is ALWAYS positive!")
print(f"   Even if log prediction was -1000, exp(-1000) ≈ 0 (still positive!)")

# ============================================================================
# COMPARISON
# ============================================================================
print("\n\n" + "=" * 70)
print("📊 COMPARISON")
print("=" * 70)

print(f"\n{'Method':<30} {'Prediction':<15} {'Can be Negative?'}")
print("-" * 70)
print(f"{'Regular Linear Regression':<30} {y_pred_regular[0]:>10.2f} kWh   {'YES ❌'}")
print(f"{'Log-Transformed Regression':<30} {y_pred_final[0]:>10.2f} kWh   {'NO ✅ (guaranteed!)'}")

print("\n\n🎯 KEY TAKEAWAY:")
print("   Log transformation MATHEMATICALLY GUARANTEES non-negative predictions")
print("   because exp(anything) is always positive!\n")
