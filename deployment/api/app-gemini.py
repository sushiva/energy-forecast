import gradio as gr
import pandas as pd
import numpy as np
import joblib # CRITICAL FIX: Use joblib for stable loading
import shap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# NOTE: The model path MUST match the path used by train_and_save_for_shap.py
MODEL_PATH = 'models/advanced/xgboost_best.pkl'

# --- 1. Load Model and Data (Runs once at startup) ---

# We remove the problematic decorator and rely on Python's global scope 
# which is generally safe for model loading in Gradio Spaces.
def load_model_and_data():
    """Load model, explainer, and training data for feature ranges."""
    try:
        # Load model using JOBLIB
        with open(MODEL_PATH, 'rb') as f:
            model_data = joblib.load(f) # <--- CRITICAL FIX APPLIED
        
        # Extract model, X_train, and feature names from the dictionary
        model = model_data['model']
        X_train = model_data.get('X_train') # NumPy array of training features
        
        # Fallback for feature names, using the names saved in the dictionary
        feature_names = model_data.get('feature_names', [f'X{i+1}' for i in range(X_train.shape[1])])
        
        # Create SHAP explainer
        explainer = shap.TreeExplainer(model)
        
        return model, explainer, X_train, feature_names
    
    except Exception as e:
        print(f"Error loading model or creating explainer: {e}")
        # Return Nones if loading fails
        return None, None, None, None

# Load resources globally
model, explainer, X_train, feature_names = load_model_and_data()

if model is None:
    # If model loading failed, define dummy components to prevent app crash
    gr.Warning("Model loading failed! Check model file path and joblib serialization.")
    
# Get feature ranges from training data for sliders
if X_train is not None and feature_names is not None:
    X_train_df = pd.DataFrame(X_train, columns=feature_names)
    feature_ranges = {
        col: (float(X_train_df[col].min()), float(X_train_df[col].max()), float(X_train_df[col].mean()))
        for col in feature_names
    }
else:
    # Default placeholder ranges if loading failed
    feature_ranges = {f'X{i+1}': (0.0, 1.0, 0.5) for i in range(8)}

# Feature descriptions (for energy efficiency dataset)
descriptions = {
    'X1': 'Relative Compactness',
    'X2': 'Surface Area',
    'X3': 'Wall Area', 
    'X4': 'Roof Area',
    'X5': 'Overall Height',
    'X6': 'Orientation',
    'X7': 'Glazing Area',
    'X8': 'Glazing Area Distribution'
}


# --- 2. Prediction and Explanation Function (Core Logic) ---

def predict_and_explain(*feature_values):
    """
    Makes a prediction and generates SHAP visualization.
    
    Returns:
        tuple: (prediction_markdown_string, matplotlib_figure_object)
    """
    if model is None or explainer is None:
        return "### ⚠️ Model Error\n\nModel resources not available.", None

    # Convert inputs to NumPy array with the correct shape (1 sample, 8 features)
    X = np.array([feature_values])
    
    # Calculate SHAP values and prediction
    shap_values = explainer.shap_values(X)
    prediction = model.predict(X)[0]
    
    base_value = explainer.expected_value
    contributions = shap_values[0]
    
    # --- Create Matplotlib Figure ---
    fig, axes = plt.subplots(1, 2, figsize=(18, 7)) # Two plots side-by-side
    
    # ----------------------------------
    # Plot A: Waterfall Plot
    # ----------------------------------
    ax1 = axes[0]
    
    # Sort by absolute contribution
    indices = np.argsort(np.abs(contributions))[::-1]
    sorted_contributions = contributions[indices]
    
    y_pos = np.arange(len(feature_names))
    colors = ['#FF6B6B' if c > 0 else '#4ECDC4' for c in sorted_contributions]
    
    ax1.barh(y_pos, sorted_contributions, color=colors, alpha=0.8)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([feature_names[i] for i in indices], fontsize=11)
    ax1.set_xlabel('SHAP Value (Impact on Prediction)', fontsize=12)
    ax1.set_title(f'Feature Contribution | Base Value: {base_value:.2f}', fontsize=14)
    ax1.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax1.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, contrib in enumerate(sorted_contributions):
        x_pos = contrib + (0.1 if contrib > 0 else -0.1)
        ax1.text(x_pos, i, f'{contrib:+.2f}', 
               va='center', ha='left' if contrib > 0 else 'right',
               fontsize=10)

    # ----------------------------------
    # Plot B: Force Plot (Simplified Horizontal Bar)
    # ----------------------------------
    ax2 = axes[1]
    
    # Starting point is the Base Value
    x_current = base_value
    
    # Draw positive contributions
    for i, (shap_val, name) in enumerate(zip(contributions, feature_names)):
        if shap_val > 0:
            ax2.barh(0, shap_val, left=x_current, height=0.5, 
                   color='#FF6B6B', alpha=0.7, label=name if shap_val > 0.5 else None) # Use label for legend clarity
            x_current += shap_val
            
    # Reset starting point for negative contributions
    x_current = base_value
    for i, (shap_val, name) in enumerate(zip(contributions, feature_names)):
        if shap_val < 0:
            ax2.barh(0, shap_val, left=x_current + shap_val, height=0.5, 
                   color='#4ECDC4', alpha=0.7, label=name if shap_val < -0.5 else None)
            x_current += shap_val

    # Draw Base Value and Prediction markers
    ax2.axvline(x=prediction, color='black', linestyle='--', linewidth=2, label='Prediction')
    ax2.axvline(x=base_value, color='gray', linestyle='--', linewidth=1, label='Base Value')
    
    ax2.set_xlabel('Predicted Energy (kWh)', fontsize=12)
    ax2.set_title('Prediction Force Plot', fontsize=14)
    ax2.set_yticks([])
    
    # Create simple legend handles
    red_patch = mpatches.Patch(color='#FF6B6B', label='+ Contribution')
    blue_patch = mpatches.Patch(color='#4ECDC4', label='- Contribution')
    
    ax2.legend(handles=[red_patch, blue_patch], loc='lower right', frameon=False)
    ax2.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    # --- Create Prediction Markdown ---
    prediction_markdown = f"""
    ### 🎯 Predicted Energy
    # **{prediction:.2f}** kWh
    
    Base Value: {base_value:.2f} kWh
    """

    return prediction_markdown, fig


# --- 3. Gradio Interface ---

with gr.Blocks(title="Interactive SHAP Explainer (Gradio)", theme=gr.themes.Soft()) as demo:
    
    gr.Markdown("""
    # 📊 Interactive SHAP Explainer
    
    Adjust the feature sliders below to see the **real-time prediction** and **SHAP explanations** for building energy consumption.
    """)
    
    # Define feature components
    input_components = []
    initial_values = [feature_ranges[feat][2] for feat in feature_names] # Start with mean values
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 🎚️ Feature Controls (Input)")
            
            # Create a slider for each feature
            for i, feat in enumerate(feature_names):
                min_val, max_val, mean_val = feature_ranges[feat]
                
                label = f"{feat} ({descriptions.get(feat, '')})"
                
                slider = gr.Slider(
                    minimum=min_val,
                    maximum=max_val,
                    value=mean_val, # Use the mean as the default starting value
                    step=(max_val - min_val) / 100,
                    label=label,
                    interactive=True
                )
                input_components.append(slider)
        
        with gr.Column(scale=2):
            gr.Markdown("### 📊 Real-time Explanation")
            
            # Define outputs
            prediction_output = gr.Markdown(
                label="Prediction", 
                value="### 🎯 Predicted Energy\n# **--** kWh"
            )
            plot_output = gr.Plot(label="SHAP Visualization (Waterfall + Force)")
            
            # Link all sliders to the prediction function on change
            for component in input_components:
                component.change(
                    fn=predict_and_explain,
                    inputs=input_components,
                    outputs=[prediction_output, plot_output],
                )

    # Initial run to populate the dashboard on load
    if model is not None:
        demo.load(
            fn=predict_and_explain,
            inputs=input_components,
            outputs=[prediction_output, plot_output]
        )

# Launch the app
if __name__ == "__main__":
    demo.launch()