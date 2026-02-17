#!/usr/bin/env python3
"""
Test script to verify stone quality model integration with backend API
Tests both /api/predict and /api/predict/ensemble endpoints with stone quality assessment
"""

import requests
import json
from pathlib import Path
import os

# Configuration
BACKEND_URL = "http://127.0.0.1:5000"
TEST_IMAGE_PATH = "stabDataset/Good"  # Use a sample image

def test_predict_endpoint():
    """Test basic predict endpoint with stone quality"""
    print("\n" + "="*80)
    print("Testing /api/predict endpoint with stone quality assessment")
    print("="*80)
    
    # Find a test image
    test_dir = Path(TEST_IMAGE_PATH)
    if not test_dir.exists():
        print(f"✗ Test image directory not found: {TEST_IMAGE_PATH}")
        return False
    
    image_files = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.jpeg")) + list(test_dir.glob("*.png"))
    if not image_files:
        print(f"✗ No images found in {TEST_IMAGE_PATH}")
        return False
    
    test_image = image_files[0]
    print(f"\nUsing test image: {test_image.name}")
    
    # Prepare request
    with open(test_image, 'rb') as f:
        files = {'image': f}
        try:
            response = requests.post(
                f"{BACKEND_URL}/api/predict?model=densenet",
                files=files,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                print("\n✓ Prediction successful!")
                print(f"  Predicted class: {result.get('predicted_class')}")
                print(f"  Confidence: {result.get('confidence'):.2f}%")
                
                # Check for stone quality assessment
                if 'stone_quality_assessment' in result:
                    stone = result['stone_quality_assessment']
                    print(f"\n✓ Stone Quality Assessment:")
                    print(f"    Quality: {stone.get('quality')}")
                    print(f"    Confidence: {stone.get('confidence'):.2f}%")
                    print(f"    Maintenance Required: {stone.get('maintenance_required')}")
                    print(f"    Urgency: {stone.get('maintenance_urgency')}")
                    print(f"    Predictions: {stone.get('all_predictions')}")
                    return True
                else:
                    print("\n✗ Stone quality assessment not in response")
                    return False
            else:
                print(f"✗ Request failed with status {response.status_code}")
                print(f"Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"✗ Error during request: {e}")
            return False

def test_ensemble_endpoint():
    """Test ensemble predict endpoint with stone quality"""
    print("\n" + "="*80)
    print("Testing /api/predict/ensemble endpoint with stone quality assessment")
    print("="*80)
    
    # Find a test image
    test_dir = Path(TEST_IMAGE_PATH)
    if not test_dir.exists():
        print(f"✗ Test image directory not found: {TEST_IMAGE_PATH}")
        return False
    
    image_files = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.jpeg")) + list(test_dir.glob("*.png"))
    if not image_files:
        print(f"✗ No images found in {TEST_IMAGE_PATH}")
        return False
    
    test_image = image_files[0]
    print(f"\nUsing test image: {test_image.name}")
    
    # Prepare request
    with open(test_image, 'rb') as f:
        files = {'image': f}
        try:
            response = requests.post(
                f"{BACKEND_URL}/api/predict/ensemble",
                files=files,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                print("\n✓ Ensemble prediction successful!")
                print(f"  Final prediction: {result.get('final_prediction')}")
                print(f"  Average confidence: {result.get('average_confidence'):.2f}%")
                print(f"  Models used: {result.get('models_used')}")
                
                # Check for stone quality assessment
                if 'stone_quality_assessment' in result:
                    stone = result['stone_quality_assessment']
                    print(f"\n✓ Stone Quality Assessment:")
                    print(f"    Quality: {stone.get('quality')}")
                    print(f"    Confidence: {stone.get('confidence'):.2f}%")
                    print(f"    Maintenance Required: {stone.get('maintenance_required')}")
                    print(f"    Urgency: {stone.get('maintenance_urgency')}")
                    print(f"    Predictions: {stone.get('all_predictions')}")
                    return True
                else:
                    print("\n✗ Stone quality assessment not in ensemble response")
                    return False
            else:
                print(f"✗ Request failed with status {response.status_code}")
                print(f"Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"✗ Error during ensemble request: {e}")
            return False

if __name__ == "__main__":
    print("\n" + "="*80)
    print("STONE QUALITY MODEL API INTEGRATION TEST")
    print("="*80)
    
    # Check if backend is running
    try:
        response = requests.get(f"{BACKEND_URL}/api/chatbot/models", timeout=5)
        print(f"✓ Backend is running on {BACKEND_URL}")
    except Exception as e:
        print(f"⚠ Proceeding with test (backend connectivity check): {e}")
    
    # Run tests
    test1_result = test_predict_endpoint()
    test2_result = test_ensemble_endpoint()
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"✓ /api/predict endpoint: {'PASSED' if test1_result else 'FAILED'}")
    print(f"✓ /api/predict/ensemble endpoint: {'PASSED' if test2_result else 'FAILED'}")
    
    if test1_result and test2_result:
        print("\n✓ All tests passed! Stone quality model integration successful!")
    else:
        print("\n✗ Some tests failed. Check the output above.")
    
    print("="*80 + "\n")
