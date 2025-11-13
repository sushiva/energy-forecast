# Run this locally where you have your original trained model object
import pickle
import xgboost as xgb # Or scikit-learn compatible wrapper

with open('models/advanced/xgboost_best.pkl', 'rb') as f:
    model_data = pickle.load(f)

# Assuming the actual model object is stored under the 'model' key
model = model_data['model'] 

# Re-save the model object to a new file format (e.g., JSON)
model.save_model('models/advanced/xgboost_best.json')