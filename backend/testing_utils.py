"""
Testing Utilities Module
Helpers and fixtures for testing
"""

import logging
import time
import random
import string
from typing import Any, Dict, Callable, Optional
from unittest.mock import Mock, patch, MagicMock
import json

logger = logging.getLogger(__name__)


class TestDataGenerator:
    """Generates test data"""
    
    @staticmethod
    def generate_random_string(length: int = 10, 
                               chars: str = string.ascii_letters + string.digits) -> str:
        """Generate random string"""
        return ''.join(random.choice(chars) for _ in range(length))
    
    @staticmethod
    def generate_random_email() -> str:
        """Generate random email"""
        username = TestDataGenerator.generate_random_string(8)
        domain = TestDataGenerator.generate_random_string(6)
        return f"{username}@{domain}.com"
    
    @staticmethod
    def generate_random_dict(num_keys: int = 5) -> Dict[str, Any]:
        """Generate random dictionary"""
        return {
            TestDataGenerator.generate_random_string(5): random.randint(0, 100)
            for _ in range(num_keys)
        }
    
    @staticmethod
    def generate_test_image_data(width: int = 224, height: int = 224) -> bytes:
        """Generate test image data"""
        import io
        from PIL import Image
        import numpy as np
        
        # Create random image
        img_array = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        
        # Convert to bytes
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        return img_bytes.getvalue()
    
    @staticmethod
    def generate_sample_prediction() -> Dict[str, Any]:
        """Generate sample prediction"""
        return {
            'model': 'test_model',
            'predictions': {
                'type-1': random.random(),
                'type-2': random.random(),
                'type-3': random.random()
            },
            'top_class': 'type-1',
            'confidence': random.random(),
            'inference_time': random.uniform(0.1, 1.0)
        }


class PerformanceTester:
    """Tests and measures performance"""
    
    @staticmethod
    def measure_execution_time(func: Callable, *args, **kwargs) -> tuple:
        """
        Measure function execution time
        
        Args:
            func: Function to test
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Tuple of (result, execution_time)
        """
        start_time = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start_time
        return result, duration
    
    @staticmethod
    def load_test(func: Callable, num_iterations: int = 100,
                 *args, **kwargs) -> Dict[str, Any]:
        """
        Perform load test on function
        
        Args:
            func: Function to test
            num_iterations: Number of times to call function
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Dictionary with test results
        """
        times = []
        errors = 0
        
        for _ in range(num_iterations):
            try:
                _, duration = PerformanceTester.measure_execution_time(func, *args, **kwargs)
                times.append(duration)
            except Exception as e:
                errors += 1
                logger.error(f"Error during load test: {str(e)}")
        
        if not times:
            return {
                'iterations': num_iterations,
                'errors': errors,
                'success_rate': 0.0
            }
        
        import statistics
        
        return {
            'iterations': num_iterations,
            'successful': len(times),
            'errors': errors,
            'success_rate': (len(times) / num_iterations) * 100,
            'min_time': min(times),
            'max_time': max(times),
            'avg_time': statistics.mean(times),
            'median_time': statistics.median(times),
            'stdev_time': statistics.stdev(times) if len(times) > 1 else 0
        }


class MockBuilder:
    """Builds mock objects for testing"""
    
    @staticmethod
    def create_mock_request(method: str = 'GET', path: str = '/',
                           data: Dict = None, headers: Dict = None) -> Mock:
        """Create mock request"""
        mock_request = Mock()
        mock_request.method = method
        mock_request.path = path
        mock_request.json = data or {}
        mock_request.headers = headers or {}
        mock_request.args = {}
        return mock_request
    
    @staticmethod
    def create_mock_response(status_code: int = 200,
                            data: Dict = None) -> Mock:
        """Create mock response"""
        mock_response = Mock()
        mock_response.status_code = status_code
        mock_response.json.return_value = data or {}
        mock_response.text = json.dumps(data or {})
        return mock_response
    
    @staticmethod
    def create_mock_model() -> Mock:
        """Create mock model"""
        mock_model = Mock()
        mock_model.predict.return_value = [[0.7, 0.2, 0.1]]
        return mock_model


class AssertionHelper:
    """Helper assertions for testing"""
    
    @staticmethod
    def assert_response_valid(response: Dict) -> bool:
        """Assert API response is valid"""
        assert isinstance(response, dict), "Response must be dictionary"
        assert 'status' in response, "Response missing 'status' field"
        assert 'code' in response, "Response missing 'code' field"
        assert 'timestamp' in response, "Response missing 'timestamp' field"
        return True
    
    @staticmethod
    def assert_prediction_valid(prediction: Dict) -> bool:
        """Assert prediction response is valid"""
        assert isinstance(prediction, dict), "Prediction must be dictionary"
        assert 'model' in prediction, "Prediction missing 'model' field"
        assert 'predictions' in prediction, "Prediction missing 'predictions' field"
        assert 'top_class' in prediction, "Prediction missing 'top_class' field"
        assert 'confidence' in prediction, "Prediction missing 'confidence' field"
        assert 0 <= prediction['confidence'] <= 1, "Confidence must be between 0 and 1"
        return True
    
    @staticmethod
    def assert_performance(duration: float, max_duration: float) -> bool:
        """Assert function performance"""
        assert duration <= max_duration, \
            f"Function took {duration:.4f}s, max was {max_duration:.4f}s"
        return True


__all__ = [
    'TestDataGenerator',
    'PerformanceTester',
    'MockBuilder',
    'AssertionHelper'
]
