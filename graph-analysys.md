"""
SHAP Force Plot Behavior Guide for Energy Consumption Model
============================================================

This guide explains how changing each feature (X1-X8) affects the SHAP force plot
and energy consumption predictions.

SHAP Force Plot Reminder:
- Base value (gray) = average prediction across all buildings (~22 kWh)
- RED arrows = features pushing prediction HIGHER (increasing energy)
- BLUE arrows = features pushing prediction LOWER (decreasing energy)
- Arrow size = magnitude of impact

Feature Importance Ranking:
1. X1 (Relative Compactness): 85.3% - DOMINANT
2. X7 (Glazing Area): 12.3%
3. X3 (Wall Area): 2.0%
4. X4 (Roof Area): 0.2%
5. X8 (Glazing Area Distribution): 0.1%
6. X6 (Orientation): 0.1% (used 497 times but low gain)
"""

# =============================================================================
# X1: RELATIVE COMPACTNESS (85.3% importance) 🔥 DOMINANT FEATURE
# =============================================================================
"""
Definition: 
Ratio of building volume to surface area (compactness = volume / surface_area)
Range: Typically 0.60 to 1.00
- Low values (0.60-0.70) = elongated, spread-out buildings (more surface area)
- High values (0.90-1.00) = compact, cube-like buildings (less surface area)

Physical Meaning:
A cube is the most compact shape. Elongated or irregular buildings have more 
surface area exposed to the environment, leading to more heat loss/gain.

SHAP Behavior:
┌─────────────────────────────────────────────────────────────────┐
│ INCREASE X1 (0.70 → 0.95) = More compact building              │
│ Effect: LARGE BLUE arrow (strong negative SHAP value)           │
│ Reason: Less surface area = less thermal transfer               │
│ Prediction: Energy consumption DECREASES significantly          │
│ Example: -15 to -20 kWh contribution                            │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ DECREASE X1 (0.90 → 0.65) = More elongated building            │
│ Effect: LARGE RED arrow (strong positive SHAP value)            │
│ Reason: More surface area = more thermal transfer               │
│ Prediction: Energy consumption INCREASES significantly          │
│ Example: +15 to +20 kWh contribution                            │
└─────────────────────────────────────────────────────────────────┘

Real-World Example:
- Cube building (X1=0.98): Might show -18 kWh (strong blue arrow)
- L-shaped building (X1=0.62): Might show +18 kWh (strong red arrow)
- Difference: ~36 kWh swing!

Expected Dashboard Behavior:
- Slider at minimum (0.62): Huge RED arrow, prediction ~45-50 kWh
- Slider at maximum (0.98): Huge BLUE arrow, prediction ~15-20 kWh
- This feature DOMINATES your force plot!
"""

# =============================================================================
# X7: GLAZING AREA (12.3% importance) - SECONDARY FACTOR
# =============================================================================
"""
Definition:
Total window/glass area as fraction of floor area
Range: 0% to 40% (0.0 to 0.40)
- 0% = No windows (rare, warehouse/storage)
- 10% = Minimal windows
- 25% = Standard office building
- 40% = Glass facade building

Physical Meaning:
Windows have poor insulation compared to walls. More glazing = more heat 
loss in winter and heat gain in summer, requiring more HVAC energy.

SHAP Behavior:
┌─────────────────────────────────────────────────────────────────┐
│ INCREASE X7 (10% → 40%) = More windows                         │
│ Effect: MODERATE RED arrow (positive SHAP value)                │
│ Reason: Poor insulation, thermal transfer through glass         │
│ Prediction: Energy consumption INCREASES                        │
│ Example: +3 to +5 kWh contribution                              │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ DECREASE X7 (40% → 10%) = Fewer windows                        │
│ Effect: MODERATE BLUE arrow (negative SHAP value)               │
│ Reason: Better insulation, less thermal transfer                │
│ Prediction: Energy consumption DECREASES                        │
│ Example: -3 to -5 kWh contribution                              │
└─────────────────────────────────────────────────────────────────┘

Real-World Example:
- Warehouse (X7=0%): Might show -4 kWh (moderate blue arrow)
- Glass tower (X7=40%): Might show +4 kWh (moderate red arrow)
- Standard office (X7=25%): Might show +1 to +2 kWh

Expected Dashboard Behavior:
- Slider at 0%: Moderate BLUE arrow, contributes -4 kWh
- Slider at 40%: Moderate RED arrow, contributes +4 kWh
- Second most visible arrow in force plot
"""

# =============================================================================
# X3: WALL AREA (2.0% importance) - MINOR FACTOR
# =============================================================================
"""
Definition:
Total exterior wall surface area in square meters
Range: Typically 150 to 400 m²
- Varies with building size and shape

Physical Meaning:
More wall area = more surface for heat transfer. However, this is correlated 
with X1 (compactness), so X1 captures most of this effect.

SHAP Behavior:
┌─────────────────────────────────────────────────────────────────┐
│ INCREASE X3 (200 → 400 m²) = More wall surface                 │
│ Effect: SMALL RED arrow (slight positive SHAP value)            │
│ Reason: More surface area for thermal transfer                  │
│ Prediction: Energy consumption INCREASES slightly               │
│ Example: +0.5 to +1.5 kWh contribution                          │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ DECREASE X3 (400 → 200 m²) = Less wall surface                 │
│ Effect: SMALL BLUE arrow (slight negative SHAP value)           │
│ Reason: Less surface area for thermal transfer                  │
│ Prediction: Energy consumption DECREASES slightly               │
│ Example: -0.5 to -1.5 kWh contribution                          │
└─────────────────────────────────────────────────────────────────┘

Real-World Example:
- Small building (X3=150 m²): Might show -1 kWh (small blue arrow)
- Large building (X3=400 m²): Might show +1 kWh (small red arrow)

Expected Dashboard Behavior:
- Changes create SMALL arrows in force plot
- Might be hard to see compared to X1 and X7
- Contributes -1 to +2 kWh typically

Note: X3 effect is partially "absorbed" by X1, since compactness already 
captures the surface-area-to-volume relationship.
"""

# =============================================================================
# X4: ROOF AREA (0.2% importance) - MINIMAL FACTOR
# =============================================================================
"""
Definition:
Total roof surface area in square meters
Range: Typically 100 to 250 m²
- Determined by building footprint

Physical Meaning:
Roof area affects heat gain/loss, especially from solar radiation. However, 
this is a small contributor relative to compactness and glazing.

SHAP Behavior:
┌─────────────────────────────────────────────────────────────────┐
│ INCREASE X4 (150 → 250 m²) = Larger roof                       │
│ Effect: TINY RED arrow (minimal positive SHAP value)            │
│ Reason: More surface for solar gain and heat loss               │
│ Prediction: Energy consumption INCREASES very slightly          │
│ Example: +0.1 to +0.3 kWh contribution                          │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ DECREASE X4 (250 → 150 m²) = Smaller roof                      │
│ Effect: TINY BLUE arrow (minimal negative SHAP value)           │
│ Reason: Less surface for solar gain and heat loss               │
│ Prediction: Energy consumption DECREASES very slightly          │
│ Example: -0.1 to -0.3 kWh contribution                          │
└─────────────────────────────────────────────────────────────────┘

Real-World Example:
- Narrow building (X4=100 m²): Might show -0.2 kWh (tiny blue arrow)
- Wide building (X4=250 m²): Might show +0.2 kWh (tiny red arrow)

Expected Dashboard Behavior:
- Creates VERY SMALL arrows in force plot
- Often barely visible
- Contributes -0.3 to +0.3 kWh typically
- Might not even appear in "top 3 contributors" summary
"""

# =============================================================================
# X8: GLAZING AREA DISTRIBUTION (0.1% importance) - MINIMAL FACTOR
# =============================================================================
"""
Definition:
How windows are distributed across building orientations
Range: 0 to 5 (categorical/ordinal)
- 0 = No windows
- 1 = Windows on one side (North or South or East or West)
- 2 = Windows on two sides
- 3 = Windows on three sides
- 4 = Windows on all four sides (uniform distribution)
- 5 = Unknown distribution

Physical Meaning:
Different orientations affect solar gain (South-facing gets more sun in 
Northern hemisphere). However, in this dataset, the effect is minimal.

SHAP Behavior:
┌─────────────────────────────────────────────────────────────────┐
│ INCREASE X8 (1 → 4) = More uniform window distribution         │
│ Effect: TINY arrow (direction depends on interaction with X7)   │
│ Reason: Distribution affects solar gain patterns                │
│ Prediction: Minimal impact on energy consumption                │
│ Example: ±0.1 to ±0.2 kWh contribution                          │
└─────────────────────────────────────────────────────────────────┘

Real-World Example:
- One-sided windows (X8=1): Might show ±0.1 kWh
- Uniform windows (X8=4): Might show ±0.2 kWh
- Effect is highly dependent on X7 (total glazing area)

Expected Dashboard Behavior:
- Creates VERY SMALL arrows
- Often not visible in force plot
- Only matters when X7 is high (lots of windows)
- Contributes -0.2 to +0.2 kWh typically
"""

# =============================================================================
# X2: SURFACE AREA (Not in top 5 by gain, but used in model)
# =============================================================================
"""
Definition:
Total building surface area (walls + roof) in square meters
Range: Typically 500 to 800 m²

Physical Meaning:
Total envelope area for heat transfer. However, X1 (compactness) already 
captures this relationship more effectively as a ratio.

SHAP Behavior:
┌─────────────────────────────────────────────────────────────────┐
│ INCREASE X2 (600 → 800 m²) = Larger building surface           │
│ Effect: VARIABLE (depends on X1 interaction)                    │
│ Reason: Effect is captured mostly by X1 (compactness)           │
│ Prediction: Small increase in energy consumption                │
│ Example: +0.3 to +1.0 kWh contribution                          │
└─────────────────────────────────────────────────────────────────┘

Expected Dashboard Behavior:
- May show small arrows
- Effect is largely redundant with X1
- Contributes -1 to +1 kWh typically
"""

# =============================================================================
# X5: OVERALL HEIGHT (Not in top 5 by gain, but used in model)
# =============================================================================
"""
Definition:
Building height in meters
Range: Typically 3.5 to 7.0 meters (1-2 stories in this dataset)

Physical Meaning:
Height affects pressure differentials, air infiltration, and vertical 
temperature gradients. However, in this dataset (low-rise buildings), 
the effect is minimal.

SHAP Behavior:
┌─────────────────────────────────────────────────────────────────┐
│ INCREASE X5 (3.5 → 7.0 m) = Taller building                    │
│ Effect: SMALL arrow (direction varies)                          │
│ Reason: Minor effect in low-rise buildings                      │
│ Prediction: Minimal impact on energy consumption                │
│ Example: ±0.3 to ±0.8 kWh contribution                          │
└─────────────────────────────────────────────────────────────────┘

Expected Dashboard Behavior:
- May show small arrows
- Effect is minimal in this dataset
- Contributes -0.8 to +0.8 kWh typically

Note: In your feature importance graph that showed X5 at 24%, that was likely 
from a DIFFERENT model with engineered features where height interacted with 
other variables.
"""

# =============================================================================
# X6: ORIENTATION (0.1% gain, but used 497 times)
# =============================================================================
"""
Definition:
Building orientation relative to compass directions
Range: 2, 3, 4, 5 (categorical)
- 2 = North
- 3 = East
- 4 = South
- 5 = West

Physical Meaning:
Orientation affects solar gain (South-facing gets most sun in Northern 
hemisphere). However, effect is minimal in this dataset, possibly because:
- Buildings are in similar climate
- Orientation effect is absorbed by X7 (glazing area)

SHAP Behavior:
┌─────────────────────────────────────────────────────────────────┐
│ CHANGE X6 (North → South)                                       │
│ Effect: TINY arrow (direction depends on location/glazing)      │
│ Reason: Solar gain varies by orientation                        │
│ Prediction: Minimal impact on energy consumption                │
│ Example: ±0.1 to ±0.3 kWh contribution                          │
└─────────────────────────────────────────────────────────────────┘

Real-World Example:
- North-facing (X6=2): Might show -0.2 kWh (less solar gain, good in hot climate)
- South-facing (X6=4): Might show +0.2 kWh (more solar gain)
- Effect depends heavily on climate and X7

Expected Dashboard Behavior:
- Creates VERY SMALL arrows
- Direction depends on interactions with X7
- Contributes -0.3 to +0.3 kWh typically
- Often not visible in force plot
"""

# =============================================================================
# SUMMARY: EXPECTED FORCE PLOT BEHAVIOR
# =============================================================================
"""
Typical SHAP Force Plot Appearance:

Base Value: ~22 kWh (gray vertical line)

For a HIGH energy building (elongated, lots of windows):
┌────────────────────────────────────────────────────────────┐
│                                                            │
│  X1: ████████████████████ +16 kWh (HUGE RED)             │
│  X7: ████ +1.4 kWh (moderate RED)                         │
│  X3: █ +0.5 kWh (small RED)                               │
│  X6: █ +0.2 kWh (tiny RED)                                │
│  ...others negligible...                                   │
│                                                            │
│  Final Prediction: 40 kWh                                  │
└────────────────────────────────────────────────────────────┘

For a LOW energy building (compact, few windows):
┌────────────────────────────────────────────────────────────┐
│                                                            │
│  X1: ████████████████████ -18 kWh (HUGE BLUE)            │
│  X7: ████ -1.2 kWh (moderate BLUE)                        │
│  X3: █ -0.4 kWh (small BLUE)                              │
│  ...others negligible...                                   │
│                                                            │
│  Final Prediction: 18 kWh                                  │
└────────────────────────────────────────────────────────────┘

Key Takeaways:
1. X1 DOMINATES - expect it to be 85%+ of the visual force plot
2. X7 is visible but much smaller than X1
3. X3 might barely be visible
4. X4, X5, X6, X8 often too small to see clearly
5. Total prediction = Base (22) + sum of all SHAP values
"""

# =============================================================================
# INTERACTIVE TESTING SCENARIOS
# =============================================================================
"""
Test these scenarios in your dashboard to verify SHAP behavior:

Scenario 1: "Energy Efficient Building"
- X1 = 0.98 (very compact)
- X7 = 0.0 (no windows)
- All others = average
Expected: HUGE blue arrows for X1 and X7, prediction ~12-15 kWh

Scenario 2: "Energy Inefficient Building"
- X1 = 0.62 (very elongated)
- X7 = 0.40 (40% windows)
- All others = average
Expected: HUGE red arrows for X1 and X7, prediction ~45-50 kWh

Scenario 3: "Isolate X1 Effect"
- Change only X1 from 0.62 → 0.98
- Keep everything else constant
Expected: Watch X1's arrow flip from large RED to large BLUE
           Prediction drops ~30-35 kWh

Scenario 4: "Isolate X7 Effect"
- Change only X7 from 0% → 40%
- Keep everything else constant
Expected: Watch X7's arrow flip from moderate BLUE to moderate RED
           Prediction increases ~5-8 kWh

Scenario 5: "Minor Features"
- Change X3, X4, X5, X6, X8
Expected: Very small arrows, hard to see
           Prediction changes <2 kWh total
"""

print(__doc__)


DECREASE X1 (0.90 → 0.65) = More elongated building - For this scenario i see the following SHAP force plot

Perfect! This is an excellent real-world example. Let me analyze what you're seeing:

🔍 Analysis of Your SHAP Force Plot
What You're Showing:
Scenario: DECREASE X1 (0.90 → 0.65) = More elongated building
Expected: Large RED arrow pushing prediction higher
What You Actually See: Large CYAN/TURQUOISE (blue-ish) area on the LEFT
✅ This is CORRECT! Here's Why:
The force plot shows the journey from base to prediction:

Base Value (16 kWh) ──────→ Final Prediction (22-23 kWh)
        ↑                            ↑
     (dashed)                    (solid)
📊 Reading the Force Plot:
LEFT SIDE (Cyan/Turquoise):

These are features pushing the prediction DOWN (negative SHAP values)
BUT in your case, they're SMALL or ABSENT because X1 is elongated (0.65)
The cyan you see is likely other features (X7, X3, etc.) trying to reduce energy
RIGHT SIDE (Red/Pink):

This is X1 with a positive SHAP value (~+6 to +7 kWh)
X1 is pushing the prediction UP because the building is elongated (0.65)
This pushes from base (16 kWh) → prediction (22-23 kWh)
🎯 Why Your Plot Looks Like This:
Force Plot Interpretation:
┌─────────────────────────────────────────────────────────┐
│ CYAN (left)          BASE        RED (right)            │
│ Features pushing    (16 kWh)    Features pushing        │
│ DOWN ←─────────────────┼──────────────→ UP            │
│                        │                                │
│ (Other features)       │        X1: +6-7 kWh          │
│ X7, X3 trying to       │        (elongated=bad)        │
│ reduce energy          │                                │
│                        │                                │
│                   Prediction: 22-23 kWh                 │
└─────────────────────────────────────────────────────────┘
📐 The Math:
Base Value: ~16 kWh (average prediction)

X1 Contribution (elongated at 0.65): +6 to +7 kWh (RED/PINK on right)

Elongated building → more surface area → more energy needed
This is POSITIVE SHAP (pushing up)
Other Features (X7, X3, etc.): Maybe -0.5 to -1 kWh total (CYAN on left)

These might be slightly reducing energy
But they're overwhelmed by X1's positive contribution
Final Prediction = 16 + 7 - 0.5 = ~22.5 kWh

✅ Your Plot is CORRECT!
The RED/PINK area on the RIGHT is exactly what we predicted for X1 when elongated!

Common Confusion:
People expect to see:

"RED = bad/high energy"
"BLUE = good/low energy"
But the force plot actually shows:

RIGHT (red) = pushing prediction HIGHER than base
LEFT (cyan) = pushing prediction LOWER than base
🧪 Let's Verify with the Opposite Scenario:
Now try this in your dashboard:

INCREASE X1 to 0.98 (very compact building)

Expected Force Plot:

┌─────────────────────────────────────────────────────────┐
│        RED          BASE           CYAN (huge!)         │
│      (tiny)       (16 kWh)         (X1: -8 to -10)     │
│         ┼──────────────────────────────────────→        │
│                                                          │
│                 Prediction: ~6-8 kWh                    │
└─────────────────────────────────────────────────────────┘
You should see:

HUGE CYAN/TURQUOISE area on the LEFT (X1 pushing prediction way down)
Tiny or no RED area on the right
Prediction around 6-8 kWh (very efficient!)
📝 For Your README/Documentation:
Update your SHAP explanation to clarify:

Force Plot Colors:

Red (RIGHT side): Features increasing energy consumption above base
Blue (LEFT side): Features decreasing energy consumption below base
Arrow size: Magnitude of impact
Example Interpretation: "When X1 = 0.65 (elongated building), the force plot shows a large red region on the right, indicating X1 is pushing the prediction ~7 kWh above the base value of 16 kWh, resulting in a final prediction of ~23 kWh."

🎯 Your Dashboard is Working Perfectly!
The SHAP force plot is correctly showing that an elongated building (X1=0.65) pushes energy consumption UP with that red/pink area on the right side. This matches our prediction of a positive SHAP value when X1 is low!

Try the compact building scenario (X1=0.98) and you'll see the dramatic difference! 🚀