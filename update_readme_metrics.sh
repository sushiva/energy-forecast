#!/bin/bash

# Script to update README.md with actual real-data results
# XGBoost: 99.82% R², ±0.31 kWh MAE, ±0.43 kWh RMSE
# Baseline: 90.91%, ±2.10 kWh MAE, ±3.08 kWh RMSE

echo "Updating README with real-data metrics..."

# Find the README file (adjust path as needed)
README_PATH="README.md"

if [ ! -f "$README_PATH" ]; then
    echo "Error: README.md not found at $README_PATH"
    echo "Please specify the correct path:"
    read -p "Enter README path: " README_PATH
    if [ ! -f "$README_PATH" ]; then
        echo "File not found. Exiting."
        exit 1
    fi
fi

echo "Found README at: $README_PATH"
echo "Creating backup..."
cp "$README_PATH" "${README_PATH}.backup_$(date +%Y%m%d_%H%M%S)"

# Update the metrics
# 1. Fix baseline percentage from 91% to 90.91%
sed -i 's/91%/90.91%/g' "$README_PATH"
sed -i 's/Baseline.*91\.0%/Baseline: 90.91%/g' "$README_PATH"

# 2. Update error metrics from ±1.12 kWh to actual values
sed -i 's/±1\.12 kWh/±0.31 kWh (MAE)/g' "$README_PATH"
sed -i 's/Error:.*±.*kWh/Error: ±0.31 kWh MAE, ±0.43 kWh RMSE/g' "$README_PATH"

# 3. Add baseline error metrics if missing
if ! grep -q "Baseline.*MAE.*RMSE" "$README_PATH"; then
    sed -i 's/Baseline: 90\.91%/Baseline: 90.91% (MAE: ±2.10 kWh, RMSE: ±3.08 kWh)/g' "$README_PATH"
fi

# 4. Update any performance metrics section
sed -i 's/Mean Absolute Error.*:.*[0-9.]*.*kWh/Mean Absolute Error: 0.31 kWh/g' "$README_PATH"
sed -i 's/Root Mean Square Error.*:.*[0-9.]*.*kWh/Root Mean Square Error: 0.43 kWh/g' "$README_PATH"

# 5. Ensure XGBoost accuracy is correct (should already be 99.82%)
sed -i 's/XGBoost.*Accuracy.*:.*[0-9.]*%/XGBoost Accuracy: 99.82%/g' "$README_PATH"

echo "✓ Updated baseline percentage: 90.91%"
echo "✓ Updated XGBoost MAE: 0.31 kWh"
echo "✓ Updated XGBoost RMSE: 0.43 kWh"
echo "✓ Updated Baseline MAE: 2.10 kWh"
echo "✓ Updated Baseline RMSE: 3.08 kWh"
echo ""
echo "Backup saved to: ${README_PATH}.backup_$(date +%Y%m%d_%H%M%S)"
echo "README updated successfully!"
echo ""
echo "Please review the changes with:"
echo "  diff ${README_PATH}.backup_* $README_PATH"