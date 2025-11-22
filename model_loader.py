"""
Model path configuration for local development vs deployment

For HuggingFace Spaces deployment:
- Model file should be in project root: xgboost_best.pkl
- Use relative path: './xgboost_best.pkl' or 'xgboost_best.pkl'

For local development:
- Model file is in: models/advanced/xgboost_best.pkl
- Use full path: 'models/advanced/xgboost_best.pkl'
"""

import os
import joblib

def load_model(deployment=False):
    """
    Load model with correct path for environment
    
    Args:
        deployment (bool): True if running on HuggingFace Spaces, False for local
    
    Returns:
        dict: Model data including model, test data, feature names, etc.
    """
    if deployment:
        # HuggingFace Spaces - model in root
        model_path = 'xgboost_best.pkl'
    else:
        # Local development - model in models/advanced/
        model_path = 'models/advanced/xgboost_best.pkl'
    
    print(f"Loading model from: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at: {model_path}\n"
            f"Current directory: {os.getcwd()}\n"
            f"Files in current directory: {os.listdir('.')}"
        )
    
    model_data = joblib.load(model_path)
    print(f"✅ Model loaded successfully from: {model_path}")
    
    return model_data


def auto_detect_environment():
    """
    Automatically detect if running locally or on HuggingFace Spaces
    
    Returns:
        bool: True if deployment environment, False if local
    """
    # Check for HuggingFace Spaces environment variable
    if os.getenv('SPACE_ID') is not None:
        print("🚀 Detected HuggingFace Spaces environment")
        return True
    
    # Check if model exists in root (deployment) or models/advanced/ (local)
    if os.path.exists('xgboost_best.pkl'):
        print("🚀 Found model in root - assuming deployment")
        return True
    elif os.path.exists('models/advanced/xgboost_best.pkl'):
        print("💻 Found model in models/advanced/ - assuming local development")
        return False
    else:
        raise FileNotFoundError(
            "Model file not found in either location!\n"
            "Local: models/advanced/xgboost_best.pkl\n"
            "Deployment: xgboost_best.pkl"
        )


# Example usage in your app.py:
if __name__ == "__main__":
    # Auto-detect environment
    is_deployment = auto_detect_environment()
    model_data = load_model(deployment=is_deployment)
    
    print("\nModel info:")
    print(f"  Features: {model_data['feature_names']}")
    print(f"  Target: {model_data['target_name']}")
    if 'performance' in model_data:
        print(f"  Test R²: {model_data['performance'].get('test_r2', 'N/A')}")