"""
Toy Example: Why Linear Regression Predicts Negative Energy Values
====================================================================

This demonstrates a fundamental limitation of linear regression for
predicting inherently positive values like energy consumption.

Author: Sudhir Shivaram Bhargav
Date: November 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

print("=" * 70)
print("TOY EXAMPLE: Linear Regression and Negative Predictions")
print("=" * 70)

# ============================================================================
# PART 1: Create Realistic Energy Data
# ============================================================================

print("\n" + "=" * 70)
print("PART 1: Creating Synthetic Energy Data")
print("=" * 70)

np.random.seed(42)

# Create synthetic building data
n_samples = 100

# Features (all positive, realistic values)
building_size = np.random.uniform(500, 5000, n_samples)  # Square feet
num_floors = np.random.randint(1, 20, n_samples)         # Floors
occupancy = np.random.uniform(10, 200, n_samples)        # People

# True energy consumption (always positive!)
# Energy = base + size_factor + floor_factor + occupancy_factor + noise
true_energy = (
    50 +                                    # Base consumption
    0.02 * building_size +                  # Larger building = more energy
    10 * num_floors +                       # More floors = more energy
    0.5 * occupancy +                       # More people = more energy
    np.random.normal(0, 20, n_samples)      # Random noise
)

# Ensure all true values are positive (as they should be in reality)
true_energy = np.maximum(true_energy, 5)

print(f"\nGenerated {n_samples} building samples")
print(f"Energy range: {true_energy.min():.2f} to {true_energy.max():.2f} kWh")
print(f"All energy values positive: {(true_energy > 0).all()}")

# ============================================================================
# PART 2: Train Linear Regression
# ============================================================================

print("\n" + "=" * 70)
print("PART 2: Training Linear Regression")
print("=" * 70)

# Prepare features
X = np.column_stack([building_size, num_floors, occupancy])
y = true_energy

# Train linear regression
model = LinearRegression()
model.fit(X, y)

# Make predictions
y_pred = model.predict(X)

# Evaluate
r2 = r2_score(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))

print(f"\nModel Performance:")
print(f"  R² Score: {r2:.4f}")
print(f"  RMSE: {rmse:.2f} kWh")
print(f"\nModel Coefficients:")
print(f"  Intercept: {model.intercept_:.2f}")
print(f"  Building Size: {model.coef_[0]:.4f}")
print(f"  Num Floors: {model.coef_[1]:.2f}")
print(f"  Occupancy: {model.coef_[2]:.4f}")

# ============================================================================
# PART 3: The Problem - Test on Edge Cases
# ============================================================================

print("\n" + "=" * 70)
print("PART 3: THE PROBLEM - Predicting on Edge Cases")
print("=" * 70)

# Create edge case scenarios (small buildings, low occupancy)
edge_cases = np.array([
    [500, 1, 5],      # Very small building, 1 floor, 5 people
    [600, 1, 3],      # Small building, 1 floor, 3 people
    [450, 1, 2],      # Tiny building, 1 floor, 2 people
    [400, 1, 1],      # Minimal building, 1 floor, 1 person
    [300, 1, 0],      # Empty small building
])

edge_predictions = model.predict(edge_cases)

print("\nEdge Case Predictions:")
print("-" * 70)
print(f"{'Building Size':<15} {'Floors':<10} {'Occupancy':<12} {'Predicted Energy':<20}")
print("-" * 70)

for i, (features, pred) in enumerate(zip(edge_cases, edge_predictions)):
    status = "✓ OK" if pred > 0 else "✗ NEGATIVE!"
    print(f"{features[0]:<15.0f} {features[1]:<10.0f} {features[2]:<12.0f} "
          f"{pred:<10.2f} kWh      {status}")

print("\n" + "!" * 70)
print("PROBLEM IDENTIFIED:")
negative_count = (edge_predictions < 0).sum()
if negative_count > 0:
    print(f"  ✗ {negative_count} out of {len(edge_cases)} predictions are NEGATIVE!")
    print(f"  ✗ Energy consumption cannot be negative in reality!")
    print(f"  ✗ Linear regression doesn't respect physical constraints!")
else:
    print("  ✓ All predictions are positive (got lucky with this data)")
print("!" * 70)

# ============================================================================
# PART 4: Why This Happens
# ============================================================================

print("\n" + "=" * 70)
print("PART 4: WHY LINEAR REGRESSION PREDICTS NEGATIVE VALUES")
print("=" * 70)

print("""
Linear Regression Formula:
──────────────────────────
Energy = Intercept + (coef₁ × Size) + (coef₂ × Floors) + (coef₃ × Occupancy)

The Problem:
────────────
1. Linear regression is UNCONSTRAINED
   → It can predict ANY value from -∞ to +∞
   
2. It only cares about MINIMIZING ERROR
   → It doesn't care if predictions are physically impossible
   
3. For small/edge case inputs:
   → Even with positive coefficients, small inputs can lead to:
   → Intercept + small_positive_terms = potentially negative result

Example from our model:
───────────────────────
""")

example_case = edge_cases[0]
pred = edge_predictions[0]
print(f"Building: {example_case[0]:.0f} sq ft, {example_case[1]:.0f} floors, "
      f"{example_case[2]:.0f} people")
print(f"\nCalculation:")
print(f"  Energy = {model.intercept_:.2f}")
print(f"         + ({model.coef_[0]:.4f} × {example_case[0]:.0f})")
print(f"         + ({model.coef_[1]:.2f} × {example_case[1]:.0f})")
print(f"         + ({model.coef_[2]:.4f} × {example_case[2]:.0f})")
print(f"         = {pred:.2f} kWh")

if pred < 0:
    print(f"\n  ✗ Result is NEGATIVE! Physically impossible!")

# ============================================================================
# PART 5: Solutions
# ============================================================================

print("\n" + "=" * 70)
print("PART 5: SOLUTIONS TO PREVENT NEGATIVE PREDICTIONS")
print("=" * 70)

print("\nOption 1: POST-PROCESSING (Clipping)")
print("-" * 70)
y_pred_clipped = np.maximum(y_pred, 0)
edge_pred_clipped = np.maximum(edge_predictions, 0)

print("Replace negative predictions with zero: max(prediction, 0)")
print("\nPros:")
print("  ✓ Simple to implement")
print("  ✓ Guarantees non-negative predictions")
print("\nCons:")
print("  ✗ Arbitrary fix (why zero specifically?)")
print("  ✗ Model hasn't learned the constraint")
print("  ✗ Discontinuity at zero")

print("\n\nOption 2: LOG TRANSFORMATION (Recommended)")
print("-" * 70)

# Train on log-transformed target
y_log = np.log(y)  # Log of energy
model_log = LinearRegression()
model_log.fit(X, y_log)

# Predict and transform back
y_log_pred = model_log.predict(X)
y_pred_exp = np.exp(y_log_pred)  # Exponentiate to get back to original scale

edge_log_pred = model_log.predict(edge_cases)
edge_pred_exp = np.exp(edge_log_pred)

print("Train on: log(Energy)")
print("Predict: log(Energy)")
print("Transform back: exp(predicted_log_energy)")
print("\nPros:")
print("  ✓ Predictions are ALWAYS positive (exp(x) > 0 for all x)")
print("  ✓ Model learns multiplicative relationships")
print("  ✓ Better for skewed data")
print("  ✓ Physically meaningful")
print("\nCons:")
print("  ✗ Slightly more complex")
print("  ✗ Changes interpretation (multiplicative vs additive)")

print("\n\nOption 3: POISSON REGRESSION")
print("-" * 70)
print("Use Generalized Linear Model with log link function")
print("\nPros:")
print("  ✓ Built-in constraint (predictions always positive)")
print("  ✓ Appropriate for count/rate data")
print("  ✓ Proper statistical framework")
print("\nCons:")
print("  ✗ Assumes Poisson distribution")
print("  ✗ More complex than linear regression")

print("\n\nOption 4: GRADIENT BOOSTING (XGBoost)")
print("-" * 70)
print("Tree-based models with custom constraints")
print("\nPros:")
print("  ✓ Can set explicit constraints")
print("  ✓ Better performance on complex data")
print("  ✓ Handles non-linearities")
print("\nCons:")
print("  ✗ More complex")
print("  ✗ Less interpretable")

# ============================================================================
# PART 6: Comparison of Solutions
# ============================================================================

print("\n" + "=" * 70)
print("PART 6: COMPARISON OF SOLUTIONS ON EDGE CASES")
print("=" * 70)

print("\nEdge Case Comparison:")
print("-" * 70)
print(f"{'Features':<25} {'Linear':<15} {'Clipped':<15} {'Log Transform':<15}")
print("-" * 70)

for i in range(len(edge_cases)):
    features_str = f"Size:{edge_cases[i,0]:.0f} F:{edge_cases[i,1]:.0f} Occ:{edge_cases[i,2]:.0f}"
    lin_pred = f"{edge_predictions[i]:.2f} kWh"
    clip_pred = f"{edge_pred_clipped[i]:.2f} kWh"
    log_pred = f"{edge_pred_exp[i]:.2f} kWh"
    
    if edge_predictions[i] < 0:
        lin_pred += " ✗"
    else:
        lin_pred += " ✓"
    
    print(f"{features_str:<25} {lin_pred:<15} {clip_pred:<15} {log_pred:<15}")

# ============================================================================
# PART 7: Visualization
# ============================================================================

print("\n" + "=" * 70)
print("PART 7: Creating Visualization")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Linear Regression and Negative Predictions', fontsize=16, fontweight='bold')

# Plot 1: Predictions vs Actual
ax = axes[0, 0]
ax.scatter(y, y_pred, alpha=0.6, s=50, label='Training Data')
ax.scatter(true_energy.min(), edge_predictions[edge_predictions < 0], 
           color='red', s=100, marker='x', linewidth=3,
           label='Negative Predictions', zorder=5)
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2, label='Perfect Prediction')
ax.axhline(y=0, color='r', linestyle='--', linewidth=2, alpha=0.5, label='Zero Line')
ax.set_xlabel('Actual Energy (kWh)', fontsize=11)
ax.set_ylabel('Predicted Energy (kWh)', fontsize=11)
ax.set_title('Problem: Linear Regression Can Predict Negative Values', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Distribution of Predictions
ax = axes[0, 1]
ax.hist(y_pred, bins=30, alpha=0.7, color='blue', edgecolor='black', label='Training Predictions')
ax.hist(edge_predictions, bins=10, alpha=0.7, color='red', edgecolor='black', label='Edge Case Predictions')
ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero (Physical Boundary)')
ax.set_xlabel('Predicted Energy (kWh)', fontsize=11)
ax.set_ylabel('Frequency', fontsize=11)
ax.set_title('Distribution: Some Predictions Are Negative!', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Plot 3: Comparison of Methods
ax = axes[1, 0]
x_pos = np.arange(len(edge_cases))
width = 0.25

bars1 = ax.bar(x_pos - width, edge_predictions, width, label='Linear (Original)', alpha=0.8)
bars2 = ax.bar(x_pos, edge_pred_clipped, width, label='Clipped', alpha=0.8)
bars3 = ax.bar(x_pos + width, edge_pred_exp, width, label='Log Transform', alpha=0.8)

ax.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7)
ax.set_xlabel('Edge Case Number', fontsize=11)
ax.set_ylabel('Predicted Energy (kWh)', fontsize=11)
ax.set_title('Solution Comparison on Edge Cases', fontsize=12, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels([f'Case {i+1}' for i in range(len(edge_cases))])
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Highlight negative values
for i, pred in enumerate(edge_predictions):
    if pred < 0:
        bars1[i].set_color('red')
        bars1[i].set_alpha(1.0)

# Plot 4: Why Log Transform Works
ax = axes[1, 1]
x_range = np.linspace(-5, 5, 100)
y_linear = x_range
y_exp = np.exp(x_range)

ax.plot(x_range, y_linear, 'b-', linewidth=2, label='Linear: y = x (can be negative)')
ax.plot(x_range, y_exp, 'g-', linewidth=2, label='Exponential: y = exp(x) (always positive)')
ax.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Zero Line')
ax.axvline(x=0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
ax.fill_between(x_range, -5, 0, alpha=0.2, color='red', label='Physically Impossible Region')
ax.set_xlabel('Model Output (log space)', fontsize=11)
ax.set_ylabel('Final Prediction (original space)', fontsize=11)
ax.set_title('Why Log Transform Guarantees Positive Predictions', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim(-1, 10)

plt.tight_layout()
plt.savefig('linear_regression_negative_predictions.png', dpi=300, bbox_inches='tight')
print("\n✓ Visualization saved: linear_regression_negative_predictions.png")

# ============================================================================
# PART 8: Key Takeaways
# ============================================================================

print("\n" + "=" * 70)
print("PART 8: KEY TAKEAWAYS")
print("=" * 70)

print("""
1. LINEAR REGRESSION PROBLEM:
   ✗ Can predict any value from -∞ to +∞
   ✗ Doesn't respect physical constraints
   ✗ Only cares about minimizing error, not domain validity

2. WHY IT MATTERS:
   ✗ Negative energy consumption is physically impossible
   ✗ Predictions lose credibility with stakeholders
   ✗ Can't deploy a model that gives nonsensical results

3. BEST SOLUTION FOR ENERGY PREDICTION:
   ✓ Log Transformation (train on log(Energy), predict exp(output))
   ✓ Guarantees positive predictions mathematically
   ✓ Better handles skewed energy distributions
   ✓ Physically meaningful (multiplicative relationships)

4. ALTERNATIVE SOLUTIONS:
   • Clipping: Simple but arbitrary
   • Poisson Regression: Good for count data
   • Tree-based models (XGBoost): Best overall performance

5. IN YOUR PROJECT:
   ✓ You used XGBoost (inherently handles this better)
   ✓ Tree-based models naturally respect data ranges
   ✓ No negative predictions in your final model!
""")

print("\n" + "=" * 70)
print("DEMONSTRATION COMPLETE")
print("=" * 70)
print("\nThis toy example illustrates why careful model selection")
print("and domain understanding are crucial for production ML systems.")
print("\nFor your energy-forecast project, XGBoost was the right choice!")
print("=" * 70)
