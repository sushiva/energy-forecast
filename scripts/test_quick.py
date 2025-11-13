import sys
sys.path.append('.')

from src.models.neural_network_model import NeuralNetworkModel
from src.models.xgboost_model import XGBoostModel
import numpy as np
import pandas as pd

print("Generating data...")
X = pd.DataFrame(np.random.rand(100, 5))
y = pd.Series(np.random.rand(100) * 10)

X_train, X_test = X[:80], X[80:]
y_train, y_test = y[:80], y[80:]

print("\n1. Testing Neural Network...")
nn = NeuralNetworkModel(epochs=5)  # Removed verbose
nn.fit(X_train, y_train)
print("✅ NN Done!")

print("\n2. Testing XGBoost...")
xgb = XGBoostModel()
xgb.fit(X_train, y_train)
print("✅ XGBoost Done!")

print("\n✅ All models work!")