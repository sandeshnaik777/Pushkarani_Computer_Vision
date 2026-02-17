"""
Unit Tests for Pushkarani Monument Classification System
Tests individual components of the deep learning pipeline
"""

import os
import sys
import numpy as np
import json
from PIL import Image
import io

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

# ============================================================================
# DATA PREPROCESSING UNIT TESTS
# ============================================================================

def test_image_resizing():
    """Test Case: Image Resizing to 224x224"""
    print("\n[UNIT TEST] Image Resizing...")
    
    # Create dummy image
    dummy_image = Image.new('RGB', (1920, 1080), color='red')
    dummy_image_array = np.array(dummy_image)
    
    # Resize to 224x224
    dummy_image_resized = Image.fromarray(dummy_image_array).resize((224, 224))
    resized_array = np.array(dummy_image_resized)
    
    assert resized_array.shape == (224, 224, 3), f"Expected (224, 224, 3), got {resized_array.shape}"
    print("  ✓ Image resizing test PASSED")
    return True

def test_normalization():
    """Test Case: Image Normalization to [0, 1]"""
    print("[UNIT TEST] Image Normalization...")
    
    # Create dummy image with values 0-255
    dummy_image = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    
    # Normalize
    normalized = dummy_image.astype('float32') / 255.0
    
    assert normalized.min() >= 0.0, f"Min value {normalized.min()} is below 0"
    assert normalized.max() <= 1.0, f"Max value {normalized.max()} is above 1"
    print("  ✓ Normalization test PASSED")
    return True

def test_data_type_consistency():
    """Test Case: Data Type Consistency (float32)"""
    print("[UNIT TEST] Data Type Consistency...")
    
    dummy_image = np.array([[[100, 150, 200]]], dtype=np.uint8)
    image_float32 = dummy_image.astype('float32')
    
    assert image_float32.dtype == np.float32, f"Expected float32, got {image_float32.dtype}"
    print("  ✓ Data type test PASSED")
    return True

# ============================================================================
# MODEL LOADING UNIT TESTS
# ============================================================================

def test_model_files_exist():
    """Test Case: Model File Existence"""
    print("[UNIT TEST] Model Files Existence...")
    
    models_to_check = [
        'densenet/best_model.keras',
        'inception/best_model.keras',
        'mobilenet/best_model.keras',
        'stone_quality_model/final_model.keras',
        'densenet/class_indices.json'
    ]
    
    base_path = os.path.dirname(os.path.abspath(__file__))
    missing_files = []
    
    for model_path in models_to_check:
        full_path = os.path.join(base_path, model_path)
        if not os.path.exists(full_path):
            missing_files.append(model_path)
        else:
            print(f"  ✓ Found: {model_path}")
    
    assert len(missing_files) == 0, f"Missing files: {missing_files}"
    print("  ✓ All model files exist - test PASSED")
    return True

def test_model_import():
    """Test Case: Model Imports"""
    print("[UNIT TEST] Model Import...")
    
    try:
        from app import load_single_model, load_stone_quality_model, CONFIG
        print("  ✓ Successfully imported model functions")
        print("  ✓ Model import test PASSED")
        return True
    except Exception as e:
        print(f"  ✗ Import failed: {e}")
        return False

def test_class_indices_format():
    """Test Case: Class Indices JSON Format"""
    print("[UNIT TEST] Class Indices Format...")
    
    base_path = os.path.dirname(os.path.abspath(__file__))
    class_indices_path = os.path.join(base_path, 'densenet/class_indices.json')
    
    try:
        with open(class_indices_path, 'r') as f:
            class_indices = json.load(f)
        
        expected_classes = ['type-1', 'type-2', 'type-3']
        actual_classes = sorted(class_indices.keys())
        
        assert actual_classes == expected_classes, f"Expected {expected_classes}, got {actual_classes}"
        print(f"  ✓ Classes found: {actual_classes}")
        print("  ✓ Class indices format test PASSED")
        return True
    except Exception as e:
        print(f"  ✗ Class indices test failed: {e}")
        return False

# ============================================================================
# CONFIGURATION UNIT TESTS
# ============================================================================

def test_config_paths():
    """Test Case: Configuration Paths Valid"""
    print("[UNIT TEST] Configuration Paths...")
    
    try:
        from app import CONFIG
        
        # Check MODEL_PATHS
        for model_name, path in CONFIG['MODEL_PATHS'].items():
            print(f"  • {model_name}: {path}")
            assert os.path.isabs(path), f"Path not absolute: {path}"
        
        # Check STONE_QUALITY_MODEL_PATH
        stone_path = CONFIG['STONE_QUALITY_MODEL_PATH']
        print(f"  • stone_quality: {stone_path}")
        assert os.path.isabs(stone_path), f"Stone quality path not absolute: {stone_path}"
        
        print("  ✓ All paths are absolute - test PASSED")
        return True
    except Exception as e:
        print(f"  ✗ Config paths test failed: {e}")
        return False

def test_stone_quality_classes():
    """Test Case: Stone Quality Classes Configuration"""
    print("[UNIT TEST] Stone Quality Classes...")
    
    try:
        from app import CONFIG
        
        expected_classes = ['Bad', 'Good', 'Medium']
        actual_classes = CONFIG['STONE_QUALITY_CLASSES']
        
        assert set(actual_classes) == set(expected_classes), \
            f"Expected {expected_classes}, got {actual_classes}"
        
        print(f"  ✓ Stone quality classes: {actual_classes}")
        print("  ✓ Stone quality classes test PASSED")
        return True
    except Exception as e:
        print(f"  ✗ Stone quality classes test failed: {e}")
        return False

# ============================================================================
# API ENDPOINT UNIT TESTS
# ============================================================================

def test_api_endpoints_exist():
    """Test Case: API Endpoints Registered"""
    print("[UNIT TEST] API Endpoints Registration...")
    
    try:
        from app import app
        
        endpoints = []
        for rule in app.url_map.iter_rules():
            if rule.rule.startswith('/api'):
                endpoints.append(rule.rule)
        
        expected_endpoints = [
            '/api/predict',
            '/api/predict/ensemble',
            '/api/chat',
            '/api/health'
        ]
        
        for endpoint in expected_endpoints:
            if endpoint in endpoints:
                print(f"  ✓ Found: {endpoint}")
            else:
                print(f"  ! Optional: {endpoint}")
        
        assert len(endpoints) > 0, "No API endpoints found"
        print("  ✓ API endpoints registration test PASSED")
        return True
    except Exception as e:
        print(f"  ✗ API endpoints test failed: {e}")
        return False

# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

def run_all_unit_tests():
    """Run all unit tests"""
    print("\n" + "="*70)
    print("PUSHKARANI UNIT TESTS - DATA PREPROCESSING & MODEL LOADING")
    print("="*70)
    
    test_functions = [
        # Data Preprocessing Tests
        test_image_resizing,
        test_normalization,
        test_data_type_consistency,
        
        # Model File Tests
        test_model_files_exist,
        test_model_import,
        test_class_indices_format,
        
        # Configuration Tests
        test_config_paths,
        test_stone_quality_classes,
        
        # API Tests
        test_api_endpoints_exist,
    ]
    
    results = {}
    passed = 0
    failed = 0
    
    for test_func in test_functions:
        try:
            result = test_func()
            results[test_func.__name__] = "PASSED" if result else "FAILED"
            if result:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            results[test_func.__name__] = f"FAILED: {e}"
            failed += 1
            print(f"  ✗ Error in {test_func.__name__}: {e}")
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    for test_name, result in results.items():
        status_symbol = "✓" if result == "PASSED" else "✗"
        print(f"{status_symbol} {test_name}: {result}")
    
    print("\n" + "="*70)
    print(f"TOTAL: {passed} PASSED, {failed} FAILED out of {passed + failed}")
    print("="*70 + "\n")
    
    return failed == 0

if __name__ == '__main__':
    success = run_all_unit_tests()
    sys.exit(0 if success else 1)
