"""
Unit Tests for Energy Consumption Forecasting Model

Tests validate:
1. Model loads correctly
2. X1 feature has correct relationship with predictions
3. SHAP values have correct signs
4. Feature importance is as expected
5. Predictions are within reasonable ranges
6. Model performance metrics meet thresholds
"""

import unittest
import joblib
import numpy as np
import shap
from pathlib import Path


class TestEnergyForecastModel(unittest.TestCase):
    """Test suite for energy forecasting model"""
    
    @classmethod
    def setUpClass(cls):
        """Load model once for all tests"""
        cls.model_path = 'models/advanced/xgboost_best.pkl'
        print(f"\nLoading model from: {cls.model_path}")
        
        cls.model_data = joblib.load(cls.model_path)
        cls.model = cls.model_data['model']
        cls.feature_names = cls.model_data['feature_names']
        cls.explainer = shap.TreeExplainer(cls.model)
        
        print(f"✓ Model loaded successfully")
        print(f"✓ Features: {cls.feature_names}")
    
    def test_01_model_file_exists(self):
        """Test that model file exists"""
        self.assertTrue(
            Path(self.model_path).exists(),
            f"Model file not found at {self.model_path}"
        )
    
    def test_02_model_loads_correctly(self):
        """Test that model loads without errors"""
        self.assertIsNotNone(self.model)
        self.assertIsNotNone(self.model_data)
        self.assertIn('model', self.model_data)
        self.assertIn('feature_names', self.model_data)
    
    def test_03_feature_names_correct(self):
        """Test that feature names are as expected"""
        expected_features = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8']
        self.assertEqual(self.feature_names, expected_features)
    
    def test_04_x1_compact_lower_energy(self):
        """
        Test: Compact building (high X1) should predict LOWER energy
        This is the CRITICAL test - validates X1 is not inverted
        """
        # Create two identical buildings except for X1
        base_features = np.array([637, 318, 147, 5.25, 3, 0.0, 2])
        
        # Compact building (X1 = 1.61)
        compact = np.concatenate([[1.61], base_features]).reshape(1, -1)
        
        # Elongated building (X1 = 1.02)
        elongated = np.concatenate([[1.02], base_features]).reshape(1, -1)
        
        pred_compact = self.model.predict(compact)[0]
        pred_elongated = self.model.predict(elongated)[0]
        
        print(f"\n  Compact (X1=1.61): {pred_compact:.2f} kWh")
        print(f"  Elongated (X1=1.02): {pred_elongated:.2f} kWh")
        
        # CRITICAL ASSERTION: Compact should use LESS energy
        self.assertLess(
            pred_compact, pred_elongated,
            f"FAIL: Compact building ({pred_compact:.2f} kWh) should use "
            f"LESS energy than elongated ({pred_elongated:.2f} kWh). "
            f"X1 may still be inverted!"
        )
        
        # Energy difference should be significant (at least 3 kWh)
        energy_diff = pred_elongated - pred_compact
        self.assertGreater(
            energy_diff, 3.0,
            f"Energy difference ({energy_diff:.2f} kWh) should be >3 kWh. "
            f"X1 may not be contributing as expected."
        )
    
    def test_05_x1_shap_sign_compact(self):
        """
        Test: Compact building (high X1) should have NEGATIVE X1 SHAP value
        """
        # Compact building (X1 = 1.61)
        compact = np.array([[1.61, 637, 318, 147, 5.25, 3, 0.0, 2]])
        
        shap_values = self.explainer.shap_values(compact)
        x1_shap = shap_values[0][0]  # First feature (X1), first sample
        
        print(f"\n  Compact building X1 SHAP: {x1_shap:.2f} kWh")
        
        # X1 SHAP should be NEGATIVE (pushing energy DOWN)
        self.assertLess(
            x1_shap, 0,
            f"FAIL: Compact building X1 SHAP ({x1_shap:.2f}) should be "
            f"NEGATIVE (blue arrow, pushing energy down)"
        )
        
        # Should be substantial (at least -3 kWh)
        self.assertLess(
            x1_shap, -3.0,
            f"X1 SHAP magnitude ({abs(x1_shap):.2f}) should be >3 kWh"
        )
    
    def test_06_x1_shap_sign_elongated(self):
        """
        Test: Elongated building (low X1) should have POSITIVE X1 SHAP value
        """
        # Elongated building (X1 = 1.02)
        elongated = np.array([[1.02, 637, 318, 147, 5.25, 3, 0.0, 2]])
        
        shap_values = self.explainer.shap_values(elongated)
        x1_shap = shap_values[0][0]  # First feature (X1), first sample
        
        print(f"\n  Elongated building X1 SHAP: {x1_shap:.2f} kWh")
        
        # X1 SHAP should be POSITIVE (pushing energy UP)
        self.assertGreater(
            x1_shap, 0,
            f"FAIL: Elongated building X1 SHAP ({x1_shap:.2f}) should be "
            f"POSITIVE (red arrow, pushing energy up)"
        )
        
        # Should be substantial (at least +2 kWh)
        self.assertGreater(
            x1_shap, 2.0,
            f"X1 SHAP magnitude ({x1_shap:.2f}) should be >2 kWh"
        )
    
    def test_07_x1_feature_importance_dominant(self):
        """Test that X1 is the most important feature (~85%)"""
        booster = self.model.get_booster()
        importance_gain = booster.get_score(importance_type='gain')
        
        # Calculate percentages
        total_gain = sum(importance_gain.values())
        importance_pct = {
            feat: (gain / total_gain) * 100 
            for feat, gain in importance_gain.items()
        }
        
        # Sort by importance
        sorted_importance = sorted(
            importance_pct.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        print(f"\n  Feature Importance (top 3):")
        for feat, pct in sorted_importance[:3]:
            print(f"    {feat}: {pct:.1f}%")
        
        # X1 should be most important
        most_important = sorted_importance[0][0]
        self.assertEqual(
            most_important, 'f0',  # f0 = X1
            f"X1 should be most important feature, but {most_important} is"
        )
        
        # X1 should be >80% importance
        x1_importance = importance_pct.get('f0', 0)
        self.assertGreater(
            x1_importance, 80.0,
            f"X1 importance ({x1_importance:.1f}%) should be >80%"
        )
    
    def test_08_x7_glazing_higher_more_energy(self):
        """Test that more glazing (X7) increases energy consumption"""
        # Building with no windows
        no_windows = np.array([[1.30, 637, 318, 147, 5.25, 3, 0.0, 2]])
        
        # Building with 40% windows
        many_windows = np.array([[1.30, 637, 318, 147, 5.25, 3, 0.40, 2]])
        
        pred_no_windows = self.model.predict(no_windows)[0]
        pred_many_windows = self.model.predict(many_windows)[0]
        
        print(f"\n  No windows (X7=0.0): {pred_no_windows:.2f} kWh")
        print(f"  40% windows (X7=0.4): {pred_many_windows:.2f} kWh")
        
        # More windows should increase energy
        self.assertGreater(
            pred_many_windows, pred_no_windows,
            f"More glazing should increase energy consumption"
        )
    
    def test_09_predictions_reasonable_range(self):
        """Test that predictions are within reasonable energy ranges"""
        # Test various building configurations
        test_cases = [
            # [X1, X2, X3, X4, X5, X6, X7, X8]
            np.array([[1.61, 637, 318, 147, 5.25, 3, 0.0, 2]]),  # Very efficient
            np.array([[1.02, 637, 318, 147, 5.25, 3, 0.40, 2]]),  # Inefficient
            np.array([[1.30, 700, 350, 180, 6.0, 4, 0.25, 3]]),  # Average
        ]
        
        for i, test_case in enumerate(test_cases):
            pred = self.model.predict(test_case)[0]
            
            # Predictions should be between 5 and 50 kWh (reasonable range)
            self.assertGreater(pred, 5.0, f"Prediction too low: {pred:.2f} kWh")
            self.assertLess(pred, 50.0, f"Prediction too high: {pred:.2f} kWh")
    
    def test_10_model_performance_metrics(self):
        """Test that model meets performance thresholds"""
        if 'performance' in self.model_data:
            perf = self.model_data['performance']
            
            # R² should be >95%
            test_r2 = perf.get('test_r2', 0)
            self.assertGreater(
                test_r2, 0.95,
                f"Test R² ({test_r2:.4f}) should be >0.95 (95%)"
            )
            
            # MAE should be <1.0 kWh
            test_mae = perf.get('test_mae', float('inf'))
            self.assertLess(
                test_mae, 1.0,
                f"Test MAE ({test_mae:.2f} kWh) should be <1.0 kWh"
            )
            
            print(f"\n  Model Performance:")
            print(f"    Test R²: {test_r2:.4f} ({test_r2*100:.2f}%)")
            print(f"    Test MAE: {test_mae:.2f} kWh")
            print(f"    Test RMSE: {perf.get('test_rmse', 'N/A'):.2f} kWh")
    
    def test_11_x1_correlation_with_energy(self):
        """Test that X1 has correct correlation with energy in test data"""
        if 'X_test' in self.model_data and 'y_test' in self.model_data:
            X_test = self.model_data['X_test']
            y_test = self.model_data['y_test']
            
            # Get X1 values (first column)
            x1_values = X_test[:, 0]
            
            # Calculate correlation
            correlation = np.corrcoef(x1_values, y_test)[0, 1]
            
            print(f"\n  X1 vs Energy correlation: {correlation:.3f}")
            
            # Correlation should be NEGATIVE (higher X1 → lower energy)
            self.assertLess(
                correlation, -0.5,
                f"X1 correlation ({correlation:.3f}) should be NEGATIVE (<-0.5). "
                f"Positive correlation indicates X1 is inverted!"
            )
    
    def test_12_extreme_cases(self):
        """Test extreme building configurations"""
        # Most efficient possible
        most_efficient = np.array([[1.61, 500, 200, 100, 3.5, 2, 0.0, 0]])
        
        # Least efficient possible
        least_efficient = np.array([[1.02, 850, 450, 250, 7.0, 5, 0.40, 4]])
        
        pred_efficient = self.model.predict(most_efficient)[0]
        pred_inefficient = self.model.predict(least_efficient)[0]
        
        print(f"\n  Most efficient config: {pred_efficient:.2f} kWh")
        print(f"  Least efficient config: {pred_inefficient:.2f} kWh")
        
        # Should have substantial difference
        self.assertLess(pred_efficient, pred_inefficient)
        
        # Difference should be at least 10 kWh
        diff = pred_inefficient - pred_efficient
        self.assertGreater(
            diff, 10.0,
            f"Extreme case difference ({diff:.2f} kWh) should be >10 kWh"
        )
    
    def test_13_model_type(self):
        """Test that model is XGBoost as expected"""
        model_type = type(self.model).__name__
        self.assertIn(
            'XGB', model_type,
            f"Model should be XGBoost, but is {model_type}"
        )
    
    def test_14_shap_calculation_works(self):
        """Test that SHAP calculations complete without errors"""
        test_sample = np.array([[1.30, 637, 318, 147, 5.25, 3, 0.25, 2]])
        
        try:
            shap_values = self.explainer.shap_values(test_sample)
            self.assertEqual(len(shap_values[0]), 8)  # Should have 8 SHAP values
        except Exception as e:
            self.fail(f"SHAP calculation failed: {e}")
    
    def test_15_base_value_reasonable(self):
        """Test that SHAP base value (expected value) is reasonable"""
        base_value = self.explainer.expected_value
        
        print(f"\n  SHAP base value: {base_value:.2f} kWh")
        
        # Base value should be around mean energy consumption (~20-25 kWh)
        self.assertGreater(base_value, 15.0)
        self.assertLess(base_value, 30.0)


class TestModelConsistency(unittest.TestCase):
    """Test that model predictions are consistent and reproducible"""
    
    @classmethod
    def setUpClass(cls):
        cls.model_data = joblib.load('models/advanced/xgboost_best.pkl')
        cls.model = cls.model_data['model']
    
    def test_prediction_reproducibility(self):
        """Test that same input gives same prediction"""
        test_input = np.array([[1.30, 637, 318, 147, 5.25, 3, 0.25, 2]])
        
        pred1 = self.model.predict(test_input)[0]
        pred2 = self.model.predict(test_input)[0]
        pred3 = self.model.predict(test_input)[0]
        
        # All predictions should be identical
        self.assertEqual(pred1, pred2)
        self.assertEqual(pred2, pred3)
    
    def test_batch_vs_single_prediction(self):
        """Test that batch and single predictions are consistent"""
        test_input = np.array([[1.30, 637, 318, 147, 5.25, 3, 0.25, 2]])
        
        # Single prediction
        single_pred = self.model.predict(test_input)[0]
        
        # Batch prediction (duplicate same input)
        batch_input = np.vstack([test_input, test_input])
        batch_preds = self.model.predict(batch_input)
        
        # Should be identical
        np.testing.assert_almost_equal(single_pred, batch_preds[0])
        np.testing.assert_almost_equal(single_pred, batch_preds[1])


def run_tests_with_summary():
    """Run all tests and provide a summary"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestEnergyForecastModel))
    suite.addTests(loader.loadTestsFromTestCase(TestModelConsistency))
    
    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n✅ ALL TESTS PASSED! Model is working correctly.")
    else:
        print("\n❌ SOME TESTS FAILED! Review failures above.")
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests_with_summary()
    exit(0 if success else 1)