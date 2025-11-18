"""
Test script for Energy Forecasting API
Tests all endpoints with sample data
"""

import requests
import json
import numpy as np

# API base URL
BASE_URL = "http://localhost:5000"

def test_health_check():
    """Test the health check endpoint"""
    print("\n" + "="*60)
    print("TEST 1: Health Check")
    print("="*60)
    
    response = requests.get(f"{BASE_URL}/")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    assert response.status_code == 200
    print("✓ Health check passed")

def test_model_info():
    """Test the model info endpoint"""
    print("\n" + "="*60)
    print("TEST 2: Model Info")
    print("="*60)
    
    response = requests.get(f"{BASE_URL}/info")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    assert response.status_code == 200
    print("✓ Model info retrieved successfully")

def test_single_prediction():
    """Test single prediction"""
    print("\n" + "="*60)
    print("TEST 3: Single Prediction")
    print("="*60)
    
    # Sample data point
    sample_data = {
        "features": {
            "temperature": 22.5,
            "hour": 14,
            "day_of_week": 2,
            "building_area": 2500.0,
            "occupancy": 50.0,
            "is_weekend": 0,
            "is_business_hours": 1,
            "temp_squared": 506.25,
            "area_occupancy": 125.0,
            "temp_hour": 315.0,
            "temp_occupancy": 1125.0,
            "hour_sin": np.sin(2 * np.pi * 14 / 24),
            "hour_cos": np.cos(2 * np.pi * 14 / 24),
            "day_sin": np.sin(2 * np.pi * 2 / 7),
            "day_cos": np.cos(2 * np.pi * 2 / 7)
        }
    }
    
    response = requests.post(
        f"{BASE_URL}/predict",
        json=sample_data,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    assert response.status_code == 200
    assert 'predictions' in response.json()
    print(f"✓ Prediction: {response.json()['predictions'][0]:.2f} kWh")

def test_batch_prediction():
    """Test batch predictions"""
    print("\n" + "="*60)
    print("TEST 4: Batch Predictions")
    print("="*60)
    
    # Multiple data points
    batch_data = {
        "features": [
            {
                "temperature": 20.0,
                "hour": 8,
                "day_of_week": 1,
                "building_area": 2000.0,
                "occupancy": 30.0,
                "is_weekend": 0,
                "is_business_hours": 1,
                "temp_squared": 400.0,
                "area_occupancy": 60.0,
                "temp_hour": 160.0,
                "temp_occupancy": 600.0,
                "hour_sin": np.sin(2 * np.pi * 8 / 24),
                "hour_cos": np.cos(2 * np.pi * 8 / 24),
                "day_sin": np.sin(2 * np.pi * 1 / 7),
                "day_cos": np.cos(2 * np.pi * 1 / 7)
            },
            {
                "temperature": 25.0,
                "hour": 18,
                "day_of_week": 5,
                "building_area": 3000.0,
                "occupancy": 80.0,
                "is_weekend": 0,
                "is_business_hours": 1,
                "temp_squared": 625.0,
                "area_occupancy": 240.0,
                "temp_hour": 450.0,
                "temp_occupancy": 2000.0,
                "hour_sin": np.sin(2 * np.pi * 18 / 24),
                "hour_cos": np.cos(2 * np.pi * 18 / 24),
                "day_sin": np.sin(2 * np.pi * 5 / 7),
                "day_cos": np.cos(2 * np.pi * 5 / 7)
            },
            {
                "temperature": 18.0,
                "hour": 22,
                "day_of_week": 6,
                "building_area": 1500.0,
                "occupancy": 10.0,
                "is_weekend": 1,
                "is_business_hours": 0,
                "temp_squared": 324.0,
                "area_occupancy": 15.0,
                "temp_hour": 396.0,
                "temp_occupancy": 180.0,
                "hour_sin": np.sin(2 * np.pi * 22 / 24),
                "hour_cos": np.cos(2 * np.pi * 22 / 24),
                "day_sin": np.sin(2 * np.pi * 6 / 7),
                "day_cos": np.cos(2 * np.pi * 6 / 7)
            }
        ]
    }
    
    response = requests.post(
        f"{BASE_URL}/predict",
        json=batch_data,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"Status Code: {response.status_code}")
    result = response.json()
    print(f"Number of predictions: {result['count']}")
    print("Predictions:")
    for i, pred in enumerate(result['predictions'], 1):
        print(f"  {i}. {pred:.2f} kWh")
    
    assert response.status_code == 200
    assert len(result['predictions']) == 3
    print("✓ Batch predictions successful")

def test_missing_features():
    """Test error handling for missing features"""
    print("\n" + "="*60)
    print("TEST 5: Missing Features Error Handling")
    print("="*60)
    
    incomplete_data = {
        "features": {
            "temperature": 22.5,
            "hour": 14
            # Missing other required features
        }
    }
    
    response = requests.post(
        f"{BASE_URL}/predict",
        json=incomplete_data,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    assert response.status_code == 400
    print("✓ Error handling works correctly")

def test_invalid_endpoint():
    """Test 404 handling"""
    print("\n" + "="*60)
    print("TEST 6: Invalid Endpoint")
    print("="*60)
    
    response = requests.get(f"{BASE_URL}/invalid")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    assert response.status_code == 404
    print("✓ 404 handling works correctly")

def run_all_tests():
    """Run all tests"""
    print("\n" + "="*60)
    print("ENERGY FORECASTING API - TEST SUITE")
    print("="*60)
    print("\nMake sure the API is running on http://localhost:5000")
    print("Start it with: python deployment/api/app.py")
    print("\nStarting tests...")
    
    try:
        test_health_check()
        test_model_info()
        test_single_prediction()
        test_batch_prediction()
        test_missing_features()
        test_invalid_endpoint()
        
        print("\n" + "="*60)
        print("ALL TESTS PASSED!")
        print("="*60)
        print("\nAPI is working correctly and ready for use.")
        
    except requests.exceptions.ConnectionError:
        print("\nERROR: Could not connect to API")
        print("Make sure the API is running on http://localhost:5000")
        print("Start it with: python deployment/api/app.py")
    except AssertionError as e:
        print(f"\nTEST FAILED: {e}")
    except Exception as e:
        print(f"\nERROR: {e}")

if __name__ == "__main__":
    run_all_tests()