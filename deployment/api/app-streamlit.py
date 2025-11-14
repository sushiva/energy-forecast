"""
Interactive SHAP Visualization App
Sliders for each feature + Real-time SHAP plots
"""
import joblib
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
from pathlib import Path


MODEL_PATH = 'models/advanced/xgboost_best.pkl'

# Page config
st.set_page_config(
    page_title="Interactive SHAP Explainer",
    page_icon="📊",
    layout="wide"
)

@st.cache_resource
def load_model_and_data():
    """Load model and data (cached for performance)"""
    try:
        # Load model
        
        model_data = joblib.load('models/advanced/xgboost_best.pkl')
        
        if isinstance(model_data, dict):
            model = model_data['model']
            X_train = model_data.get('X_train')
            X_test = model_data.get('X_test')
            feature_names = model_data.get('feature_names', [f'X{i+1}' for i in range(8)])
        else:
            model = model_data
            # Load data from CSV
            df = pd.read_csv('data/processed/energy_data_processed.csv')
            feature_cols = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8']
            train_size = int(len(df) * 0.8)
            X_train = df[feature_cols].iloc[:train_size].values
            X_test = df[feature_cols].iloc[train_size:].values
            feature_names = feature_cols
        
        # Create SHAP explainer
        explainer = shap.TreeExplainer(model)
        
        return model, explainer, X_train, X_test, feature_names
    
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("Run train_and_save_for_shap.py first!")
        return None, None, None, None, None

def create_shap_plots(model, explainer, feature_values, feature_names):
    """Create SHAP visualizations for given feature values"""
    
    # Convert to right shape
    X = np.array([feature_values])
    
    # Calculate SHAP values
    shap_values = explainer.shap_values(X)
    
    # Get prediction
    prediction = model.predict(X)[0]
    
    # Create three plots side by side
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Prediction")
        st.metric("Predicted Energy", f"{prediction:.2f} kWh")
        
        st.subheader("📊 Feature Contribution (Waterfall)")
        
        # Waterfall plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create waterfall manually
        base_value = explainer.expected_value
        contributions = shap_values[0]
        
        # Sort by absolute contribution
        indices = np.argsort(np.abs(contributions))[::-1]
        
        y_pos = np.arange(len(feature_names))
        colors = ['#FF6B6B' if c > 0 else '#4ECDC4' for c in contributions[indices]]
        
        ax.barh(y_pos, contributions[indices], color=colors, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([feature_names[i] for i in indices])
        ax.set_xlabel('SHAP Value (Impact on Prediction)')
        ax.set_title(f'Base Value: {base_value:.2f} kWh → Prediction: {prediction:.2f} kWh')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, (idx, contrib) in enumerate(zip(indices, contributions[indices])):
            value = feature_values[idx]
            label = f'{feature_names[idx]}={value:.2f}'
            x_pos = contrib + (0.1 if contrib > 0 else -0.1)
            ax.text(x_pos, i, f'{contrib:+.2f}', 
                   va='center', ha='left' if contrib > 0 else 'right',
                   fontsize=9)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.subheader("📈 Feature Values")
        
        # Show current values in a nice table
        values_df = pd.DataFrame({
            'Feature': feature_names,
            'Current Value': feature_values,
            'SHAP Value': shap_values[0],
            'Impact': ['↑ Increases' if s > 0 else '↓ Decreases' for s in shap_values[0]]
        })
        
        # Sort by absolute SHAP value
        values_df['Abs_SHAP'] = np.abs(values_df['SHAP Value'])
        values_df = values_df.sort_values('Abs_SHAP', ascending=False).drop('Abs_SHAP', axis=1)
        
        st.dataframe(
            values_df.style.background_gradient(subset=['SHAP Value'], cmap='RdYlGn', vmin=-2, vmax=2),
            use_container_width=True,
            hide_index=True
        )
        
        st.subheader("🔍 Force Plot")
        
        # Force plot as matplotlib
        fig, ax = plt.subplots(figsize=(10, 3))
        
        # Create horizontal force plot
        positive_contrib = sum([s for s in shap_values[0] if s > 0])
        negative_contrib = sum([s for s in shap_values[0] if s < 0])
        
        # Starting point
        x_start = base_value
        
        # Draw base
        ax.barh(0, base_value, left=0, height=0.5, color='gray', alpha=0.3, label='Base')
        
        # Draw positive contributions
        x_current = base_value
        for i, (feat, val, shap_val) in enumerate(zip(feature_names, feature_values, shap_values[0])):
            if shap_val > 0:
                ax.barh(0, shap_val, left=x_current, height=0.5, 
                       color='#FF6B6B', alpha=0.7)
                x_current += shap_val
        
        # Draw negative contributions
        x_current = base_value
        for i, (feat, val, shap_val) in enumerate(zip(feature_names, feature_values, shap_values[0])):
            if shap_val < 0:
                ax.barh(0, shap_val, left=x_current + shap_val, height=0.5, 
                       color='#4ECDC4', alpha=0.7)
                x_current += shap_val
        
        # Final prediction line
        ax.axvline(x=prediction, color='black', linestyle='--', linewidth=2, label='Prediction')
        ax.axvline(x=base_value, color='gray', linestyle='--', linewidth=1, label='Base Value')
        
        ax.set_xlabel('Energy Consumption (kWh)')
        ax.set_title('How Features Push Prediction from Base Value')
        ax.set_yticks([])
        ax.legend(loc='upper right')
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

def main():
    """Main app"""
    
    st.title("🎛️ Interactive SHAP Feature Explorer")
    st.markdown("**Adjust feature sliders to see how predictions change in real-time**")
    
    # Load model and data
    model, explainer, X_train, X_test, feature_names = load_model_and_data()
    
    if model is None:
        st.stop()
    
    # Get feature ranges from training data
    X_train_df = pd.DataFrame(X_train, columns=feature_names)
    feature_ranges = {
        col: (float(X_train_df[col].min()), float(X_train_df[col].max()), float(X_train_df[col].mean()))
        for col in feature_names
    }
    
    st.markdown("---")
    
    # Create layout: Sliders on left, Plots on right
    slider_col, plot_col = st.columns([1, 2])
    
    with slider_col:
        st.subheader("🎚️ Feature Controls")
        st.markdown("*Adjust sliders to change feature values*")
        
        # Create sliders for each feature
        feature_values = []
        
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
        
        for feat in feature_names:
            min_val, max_val, mean_val = feature_ranges[feat]
            
            # Add description if available
            label = f"{feat}"
            if feat in descriptions:
                label += f" ({descriptions[feat]})"
            
            value = st.slider(
                label,
                min_value=min_val,
                max_value=max_val,
                value=mean_val,
                step=(max_val - min_val) / 100,
                key=feat,
                help=f"Range: [{min_val:.2f}, {max_val:.2f}]"
            )
            feature_values.append(value)
        
        # Add preset buttons
        st.markdown("---")
        st.subheader("⚡ Quick Presets")
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            if st.button("📊 Average Values", use_container_width=True):
                st.rerun()
        
        with col_b:
            if st.button("🎲 Random Sample", use_container_width=True):
                # Set random values (will trigger rerun)
                sample_idx = np.random.randint(0, len(X_test))
                st.session_state.random_sample = X_test[sample_idx]
                st.rerun()
        
        # Apply random sample if exists
        if 'random_sample' in st.session_state:
            feature_values = list(st.session_state.random_sample)
            del st.session_state.random_sample
            st.rerun()
    
    with plot_col:
        # Create SHAP plots with current feature values
        create_shap_plots(model, explainer, feature_values, feature_names)
    
    # Add info section at bottom
    st.markdown("---")
    with st.expander("ℹ️ How to Use This App"):
        st.markdown("""
        ### Understanding the Visualizations
        
        **Waterfall Plot (Left)**
        - Shows how each feature contributes to the final prediction
        - Red bars = Feature increases prediction
        - Blue bars = Feature decreases prediction
        - Longer bars = Stronger effect
        
        **Feature Table (Right Top)**
        - Lists all features with their current values
        - Shows SHAP value (contribution to prediction)
        - Green = Positive impact | Red = Negative impact
        
        **Force Plot (Right Bottom)**
        - Visual representation of prediction process
        - Gray line = Base value (average prediction)
        - Black dashed line = Final prediction
        - Shows how features "push" prediction from base
        
        ### Tips
        - Adjust sliders to see real-time updates
        - Use presets to quickly test scenarios
        - Watch how predictions change with each feature
        """)

if __name__ == "__main__":
    main()