"""
Interactive SHAP Visualization - Gradio Version
Sliders on LEFT, Plots on RIGHT, Summary below sliders
"""

import gradio as gr
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

MODEL_PATH = 'xgboost_best.pkl'

def load_model():
    """Load model and create explainer"""
    try:
        model_data = joblib.load(MODEL_PATH)
        model = model_data['model'] if isinstance(model_data, dict) else model_data
        explainer = shap.TreeExplainer(model)
        print("✓ Model and SHAP loaded")
        return model, explainer
    except Exception as e:
        print(f"Error: {e}")
        return None, None

MODEL, EXPLAINER = load_model()

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

def create_shap_visualization(X1, X2, X3, X4, X5, X6, X7, X8):
    """Create SHAP plots for given feature values"""
    
    if MODEL is None:
        return None, None, "Model not loaded"
    
    try:
        # Create feature array
        features = np.array([[X1, X2, X3, X4, X5, X6, X7, X8]])
        df = pd.DataFrame(features, columns=FEATURE_NAMES)
        
        # Predict
        prediction = MODEL.predict(features)[0]
        # Add this conversion:
        if isinstance(prediction, str):
            prediction = float(prediction.strip('[]'))
        else:
            rediction = float(prediction)

        # SHAP values
        shap_values = EXPLAINER.shap_values(features)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        if len(shap_values.shape) > 1:
            shap_values = shap_values[0]
        
        base_value = EXPLAINER.expected_value
        if isinstance(base_value, (list, np.ndarray)):
            base_value = base_value[0]
        
        # Create waterfall plot
        fig1, ax1 = plt.subplots(figsize=(10, 8))
        
        indices = np.argsort(np.abs(shap_values))[::-1]
        y_pos = np.arange(len(FEATURE_NAMES))
        colors = ['#FF6B6B' if s > 0 else '#4ECDC4' for s in shap_values[indices]]
        
        ax1.barh(y_pos, shap_values[indices], color=colors, alpha=0.8, edgecolor='black')
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels([f"{FEATURE_NAMES[i]} ({DESCRIPTIONS[FEATURE_NAMES[i]]}) = {features[0][i]:.2f}" 
                             for i in indices], fontsize=9)
        ax1.set_xlabel('SHAP Value (Impact)', fontweight='bold')
        ax1.set_title(f'Feature Contributions\nBase: {base_value:.2f} → Prediction: {prediction:.2f} kWh', 
                     fontweight='bold')
        ax1.axvline(x=0, color='black', linewidth=1)
        ax1.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, (idx, shap_val) in enumerate(zip(indices, shap_values[indices])):
            x_pos = shap_val + (0.08 if shap_val > 0 else -0.08)
            ax1.text(x_pos, i, f'{shap_val:+.2f}', va='center', 
                    ha='left' if shap_val > 0 else 'right', fontsize=8, fontweight='bold')
        
        plt.tight_layout()
        
        # Create force plot
        fig2, ax2 = plt.subplots(figsize=(10, 3))
        
        x_current = base_value
        for shap_val in shap_values:
            if shap_val > 0:
                ax2.barh(0, shap_val, left=x_current, height=0.6, color='#FF6B6B', alpha=0.7)
                x_current += shap_val
        
        x_current = base_value
        for shap_val in shap_values:
            if shap_val < 0:
                ax2.barh(0, shap_val, left=x_current + shap_val, height=0.6, color='#4ECDC4', alpha=0.7)
                x_current += shap_val
        
        ax2.axvline(x=prediction, color='black', linestyle='--', linewidth=2, label='Prediction')
        ax2.axvline(x=base_value, color='gray', linestyle='--', linewidth=1, label='Base')
        ax2.set_xlabel('Energy (kWh)', fontweight='bold')
        ax2.set_title('Force Plot: From Base to Prediction', fontweight='bold')
        ax2.set_yticks([])
        ax2.legend(loc='upper right')
        ax2.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        # Create summary
        summary = f"""
**🎯 Prediction:** {prediction:.2f} kWh

**📊 Top 3 Contributors:**
- **{FEATURE_NAMES[indices[0]]}** ({DESCRIPTIONS[FEATURE_NAMES[indices[0]]]}): {shap_values[indices[0]]:+.2f} kWh
- **{FEATURE_NAMES[indices[1]]}** ({DESCRIPTIONS[FEATURE_NAMES[indices[1]]]}): {shap_values[indices[1]]:+.2f} kWh
- **{FEATURE_NAMES[indices[2]]}** ({DESCRIPTIONS[FEATURE_NAMES[indices[2]]]}): {shap_values[indices[2]]:+.2f} kWh
"""
        
        return fig2, fig1, summary
        
    except Exception as e:
        return None, None, f"Error: {e}"

# Gradio Interface
with gr.Blocks(title="SHAP Explorer", theme=gr.themes.Soft()) as demo:
    
    gr.Markdown("# 🎛️ Interactive SHAP Feature Explorer")
    gr.Markdown("**Adjust sliders and click 'Analyze' to see SHAP explanations**")
    
    with gr.Row():
        # LEFT COLUMN: Sliders + Summary
        with gr.Column(scale=1):
            gr.Markdown("### 🎚️ Building Features")
            
            X1 = gr.Slider(0.6, 1.0, 0.79, step=0.01, label=f"X1 ({DESCRIPTIONS['X1']})")
            X2 = gr.Slider(500, 850, 637, step=10, label=f"X2 ({DESCRIPTIONS['X2']})")
            X3 = gr.Slider(200, 450, 318, step=10, label=f"X3 ({DESCRIPTIONS['X3']})")
            X4 = gr.Slider(100, 250, 147, step=10, label=f"X4 ({DESCRIPTIONS['X4']})")
            X5 = gr.Slider(3.5, 7.0, 5.25, step=0.25, label=f"X5 ({DESCRIPTIONS['X5']})")
            X6 = gr.Slider(2, 5, 3, step=1, label=f"X6 ({DESCRIPTIONS['X6']})")
            X7 = gr.Slider(0, 0.4, 0.25, step=0.05, label=f"X7 ({DESCRIPTIONS['X7']})")
            X8 = gr.Slider(0, 5, 2, step=1, label=f"X8 ({DESCRIPTIONS['X8']})")
            
            analyze_btn = gr.Button("⚡ Analyze with SHAP", variant="primary", size="lg")
            
            # Summary below sliders
            gr.Markdown("---")
            gr.Markdown("### 📊 Prediction Summary")
            summary_output = gr.Markdown()
        
        # RIGHT COLUMN: Plots
        with gr.Column(scale=2):

            gr.Markdown("### 🎯 SHAP Force Plot")
            force_plot = gr.Plot()
            
            gr.Markdown("### 🎯 SHAP Waterfall Plot")
            waterfall_plot = gr.Plot()
            
          
    
    gr.Markdown("---")
    gr.Markdown("### 💡 Example Configurations")
    gr.Examples(
        examples=[
            [0.98, 514.5, 294.0, 110.25, 7.0, 2, 0.0, 0],
            [0.62, 808.5, 367.5, 220.5, 3.5, 3, 0.4, 5],
            [0.79, 637.0, 318.5, 147.0, 5.25, 4, 0.25, 3],
        ],
        inputs=[X1, X2, X3, X4, X5, X6, X7, X8],
        outputs=[waterfall_plot, force_plot, summary_output],
        fn=create_shap_visualization,
    )
    
    analyze_btn.click(
        fn=create_shap_visualization,
        inputs=[X1, X2, X3, X4, X5, X6, X7, X8],
        outputs=[force_plot, waterfall_plot,  summary_output]
    )
    
    # Load default values on startup
    demo.load(
        fn=create_shap_visualization,
        inputs=[X1, X2, X3, X4, X5, X6, X7, X8],
        outputs=[ force_plot, waterfall_plot, summary_output]
    )

    gr.Markdown("""
    ---
    ### About
    **Model:** XGBoost | **Performance:** 99.82% R² | **Explainability:** SHAP
    """)

if __name__ == "__main__":
    demo.launch()