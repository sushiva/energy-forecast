"""
Generate the CORRECT feature importance graph for the base model (8 features only)
This matches what's actually deployed on HuggingFace
"""
import joblib
import matplotlib.pyplot as plt
import numpy as np

# Load the actual deployed model
model_data = joblib.load('models/advanced/xgboost_best.pkl')
model = model_data['model']

# Feature names and descriptions
FEATURE_NAMES = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8']
DESCRIPTIONS = {
    'X1': 'Relative Compactness',
    'X2': 'Surface Area',
    'X3': 'Wall Area', 
    'X4': 'Roof Area',
    'X5': 'Overall Height',
    'X6': 'Orientation',
    'X7': 'Glazing Area',
    'X8': 'Glazing Area Distribution'
}

# Get feature importance from the model
booster = model.get_booster()
importance_dict = booster.get_score(importance_type='gain')

# Map internal XGBoost names (f0, f1, ...) to your FEATURE_NAMES
xgb_feature_names = [f"f{i}" for i in range(len(FEATURE_NAMES))]

feature_importance = []
for feature, xgb_name in zip(FEATURE_NAMES, xgb_feature_names):
    if xgb_name in importance_dict:
        importance = importance_dict[xgb_name]
        feature_importance.append({
            'feature': feature,
            'description': DESCRIPTIONS[feature],
            'importance': importance
        })

# Sort by importance
feature_importance.sort(key=lambda x: x['importance'], reverse=True)

# Calculate percentages
total_importance = sum([f['importance'] for f in feature_importance])
for f in feature_importance:
    f['percentage'] = (f['importance'] / total_importance) * 100 if total_importance > 0 else 0

# Create the plot
plt.figure(figsize=(12, 8))
plt.style.use('seaborn-v0_8-darkgrid')

features = [f"{f['feature']}: {f['description']}" for f in feature_importance]
importances = [f['percentage'] for f in feature_importance]

if importances:  # Guard against empty list
    # Create horizontal bar chart
    bars = plt.barh(features, importances, color='#5DADE2', edgecolor='#2874A6', linewidth=1.5)

    # Add percentage labels
    for i, (bar, imp) in enumerate(zip(bars, importances)):
        plt.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
                 f'{imp:.1f}%', 
                 va='center', fontsize=11, fontweight='bold')

    plt.xlabel('Feature Importance (%)', fontsize=13, fontweight='bold')
    plt.title('Feature Importance (XGBoost Model - Deployed Version)', 
              fontsize=15, fontweight='bold', pad=20)
    plt.xlim(0, max(importances) * 1.15)

    # Invert y-axis so most important is at top
    plt.gca().invert_yaxis()

    plt.tight_layout()
    plt.savefig('images/architecture/feature_importance_corrected.png', dpi=300, bbox_inches='tight')
    plt.savefig('feature_importance_corrected.png', dpi=300, bbox_inches='tight')
    print("✅ Generated corrected feature importance graph!")
    print("\nFeature Importance Rankings:")
    print("=" * 60)
    for i, f in enumerate(feature_importance, 1):
        print(f"{i}. {f['feature']} ({f['description']}): {f['percentage']:.1f}%")
    print("\n✅ Replace your current graph with: feature_importance_corrected.png")
else:
    print("⚠️ No feature importances found. Check model training or feature mapping.")
