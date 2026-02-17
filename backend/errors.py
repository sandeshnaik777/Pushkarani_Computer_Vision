"""
Error Handling Module
Custom exceptions and error handlers for the application
"""

import logging
import traceback
from typing import Dict, Any, Callable
from functools import wraps

logger = logging.getLogger(__name__)


class AppException(Exception):
    """Base exception class for application errors"""
    
    def __init__(self, message: str, code: int = 400, details: Any = None):
        """
        Initialize AppException
        
        Args:
            message: Error message
            code: HTTP status code
            details: Additional error details
        """
        self.message = message
        self.code = code
        self.details = details
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary"""
        return {
            'status': 'error',
            'code': self.code,
            'message': self.message,
            'details': self.details
        }


class ValidationError(AppException):
    """Raised when input validation fails"""
    
    def __init__(self, message: str, details: Any = None):
        super().__init__(message, code=400, details=details)


class ModelLoadError(AppException):
    """Raised when model loading fails"""
    
    def __init__(self, message: str, details: Any = None):
        super().__init__(message, code=503, details=details)


class PredictionError(AppException):
    """Raised when prediction fails"""
    
    def __init__(self, message: str, details: Any = None):
        super().__init__(message, code=500, details=details)


class FileNotFoundError(AppException):
    """Raised when required file is not found"""
    
    def __init__(self, filepath: str):
        message = f"Required file not found: {filepath}"
        super().__init__(message, code=404, details={'filepath': filepath})


class ConfigurationError(AppException):
    """Raised when configuration is invalid"""
    
    def __init__(self, message: str, details: Any = None):
        super().__init__(message, code=500, details=details)


class RateLimitError(AppException):
    """Raised when rate limit is exceeded"""
    
    def __init__(self, message: str = "Rate limit exceeded", details: Any = None):
        super().__init__(message, code=429, details=details)


class UnavailableError(AppException):
    """Raised when service is temporarily unavailable"""
    
    def __init__(self, message: str = "Service temporarily unavailable", details: Any = None):
        super().__init__(message, code=503, details=details)


class ErrorHandler:
    """Centralized error handling"""
    
    @staticmethod
    def handle_error(func: Callable) -> Callable:
        """
        Decorator to handle errors in route functions
        
        Args:
            func: Function to decorate
            
        Returns:
            Decorated function
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except AppException as e:
                logger.warning(f"Application error in {func.__name__}: {e.message}")
                return e.to_dict(), e.code
            except ValueError as e:
                logger.warning(f"Validation error in {func.__name__}: {str(e)}")
                return {
                    'status': 'error',
                    'code': 400,
                    'message': f'Validation error: {str(e)}',
                    'details': None
                }, 400
            except FileNotFoundError as e:
                logger.error(f"File not found in {func.__name__}: {str(e)}")
                return {
                    'status': 'error',
                    'code': 404,
                    'message': str(e),
                    'details': None
                }, 404
            except Exception as e:
                logger.error(f"Unexpected error in {func.__name__}: {str(e)}")
                logger.debug(traceback.format_exc())
                return {
                    'status': 'error',
                    'code': 500,
                    'message': 'Internal server error',
                    'details': str(e) if logger.level == logging.DEBUG else None
                }, 500
        
        return wrapper
    
    @staticmethod
    def safe_execute(func: Callable, *args, **kwargs) -> tuple:
        """
        Safely execute function with error handling
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Tuple of (result, error, success)
        """
        try:
            result = func(*args, **kwargs)
            return result, None, True
        except AppException as e:
            logger.warning(f"Application error: {e.message}")
            return None, e, False
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            logger.debug(traceback.format_exc())
            error = AppException(str(e), code=500)
            return None, error, False


class Logger:
    """Enhanced logging configuration"""
    
    @staticmethod
    def log_prediction_start(model_name: str, image_id: str = None):
        """Log prediction start"""
        logger.info(f"Starting prediction with model: {model_name}" + 
                   (f", image_id: {image_id}" if image_id else ""))
    
    @staticmethod
    def log_prediction_success(model_name: str, result: str, confidence: float):
        """Log successful prediction"""
        logger.info(f"Prediction successful - Model: {model_name}, "
                   f"Result: {result}, Confidence: {confidence:.4f}")
    
    @staticmethod
    def log_prediction_error(model_name: str, error: str):
        """Log prediction error"""
        logger.error(f"Prediction failed - Model: {model_name}, Error: {error}")
    
    @staticmethod
    def log_api_request(endpoint: str, method: str, status_code: int):
        """Log API request"""
        logger.info(f"API Request - Endpoint: {endpoint}, Method: {method}, "
                   f"Status: {status_code}")
    
    @staticmethod
    def log_model_load(model_name: str, success: bool = True):
        """Log model loading"""
        status = "Successfully loaded" if success else "Failed to load"
        logger.info(f"Model '{model_name}': {status}")


# Initialize module-level logger
__all__ = [
    'AppException',
    'ValidationError',
    'ModelLoadError',
    'PredictionError',
    'FileNotFoundError',
    'ConfigurationError',
    'RateLimitError',
    'UnavailableError',
    'ErrorHandler',
    'Logger'
]
