"""
Utilities Module
Common utility functions for image processing, data validation, and error handling
"""

import io
import json
import base64
import logging
from typing import Tuple, Optional, Dict, Any
from PIL import Image
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)


class ImageProcessor:
    """Handles image processing operations"""
    
    @staticmethod
    def load_image_from_file(file) -> Optional[Image.Image]:
        """
        Load image from uploaded file
        
        Args:
            file: File object from request
            
        Returns:
            PIL Image object or None if failed
        """
        try:
            image = Image.open(file.stream)
            # Convert RGBA to RGB if needed
            if image.mode in ('RGBA', 'LA', 'P'):
                rgb_image = Image.new('RGB', image.size, (255, 255, 255))
                rgb_image.paste(image, mask=image.split()[-1] if image.mode == 'RGBA' else None)
                return rgb_image
            elif image.mode != 'RGB':
                return image.convert('RGB')
            return image
        except Exception as e:
            logger.error(f"Error loading image: {str(e)}")
            return None
    
    @staticmethod
    def load_image_from_base64(base64_string: str) -> Optional[Image.Image]:
        """
        Load image from base64 encoded string
        
        Args:
            base64_string: Base64 encoded image string
            
        Returns:
            PIL Image object or None if failed
        """
        try:
            image_data = base64.b64decode(base64_string)
            image = Image.open(io.BytesIO(image_data))
            
            # Convert RGBA to RGB if needed
            if image.mode != 'RGB':
                if image.mode == 'RGBA':
                    rgb_image = Image.new('RGB', image.size, (255, 255, 255))
                    rgb_image.paste(image, mask=image.split()[-1])
                    return rgb_image
                return image.convert('RGB')
            return image
        except Exception as e:
            logger.error(f"Error decoding base64 image: {str(e)}")
            return None
    
    @staticmethod
    def resize_image(image: Image.Image, size: Tuple[int, int]) -> Image.Image:
        """
        Resize image to specified dimensions
        
        Args:
            image: PIL Image object
            size: Target size (height, width)
            
        Returns:
            Resized PIL Image object
        """
        try:
            return image.resize((size[1], size[0]), Image.Resampling.LANCZOS)
        except Exception as e:
            logger.error(f"Error resizing image: {str(e)}")
            return image
    
    @staticmethod
    def normalize_image(image: Image.Image) -> np.ndarray:
        """
        Normalize image to numpy array with pixel values in [0, 1]
        
        Args:
            image: PIL Image object
            
        Returns:
            Normalized numpy array
        """
        try:
            img_array = np.array(image, dtype=np.float32)
            # Normalize to [0, 1]
            return img_array / 255.0
        except Exception as e:
            logger.error(f"Error normalizing image: {str(e)}")
            return None
    
    @staticmethod
    def image_to_base64(image: Image.Image) -> Optional[str]:
        """
        Convert PIL Image to base64 string
        
        Args:
            image: PIL Image object
            
        Returns:
            Base64 encoded string
        """
        try:
            buffer = io.BytesIO()
            image.save(buffer, format='PNG')
            buffer.seek(0)
            return base64.b64encode(buffer.getvalue()).decode()
        except Exception as e:
            logger.error(f"Error converting image to base64: {str(e)}")
            return None


class ValidationHelper:
    """Input validation helper functions"""
    
    @staticmethod
    def validate_image_file(file) -> Tuple[bool, str]:
        """
        Validate uploaded image file
        
        Args:
            file: File object from request
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not file:
            return False, "No file provided"
        
        if not file.filename:
            return False, "File has no name"
        
        # Check file extension
        allowed_extensions = {'jpg', 'jpeg', 'png', 'bmp', 'webp'}
        file_ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
        
        if file_ext not in allowed_extensions:
            return False, f"Invalid file format. Allowed: {', '.join(allowed_extensions)}"
        
        # Check file size (50MB limit)
        file.seek(0, 2)  # Seek to end
        file_size = file.tell()
        file.seek(0)  # Reset to beginning
        
        max_size = 50 * 1024 * 1024  # 50MB
        if file_size > max_size:
            return False, f"File size too large (max 50MB, got {file_size / (1024*1024):.1f}MB)"
        
        return True, ""
    
    @staticmethod
    def validate_base64_image(base64_string: str) -> Tuple[bool, str]:
        """
        Validate base64 encoded image string
        
        Args:
            base64_string: Base64 string to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not base64_string:
            return False, "Empty base64 string"
        
        try:
            # Check length
            if len(base64_string) > 50 * 1024 * 1024:  # 50MB
                return False, "Base64 string too large"
            
            # Try to decode
            base64.b64decode(base64_string, validate=True)
            return True, ""
        except Exception as e:
            return False, f"Invalid base64 string: {str(e)}"
    
    @staticmethod
    def validate_model_name(model_name: str, valid_models: list) -> Tuple[bool, str]:
        """
        Validate model name
        
        Args:
            model_name: Model name to validate
            valid_models: List of valid model names
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not model_name:
            return False, "Model name required"
        
        if model_name not in valid_models:
            return False, f"Invalid model. Available: {', '.join(valid_models)}"
        
        return True, ""


class ResponseFormatter:
    """Format API responses consistently"""
    
    @staticmethod
    def success(data: Any = None, message: str = "Success", code: int = 200) -> Dict[str, Any]:
        """
        Format successful response
        
        Args:
            data: Response data
            message: Response message
            code: HTTP status code
            
        Returns:
            Formatted response dictionary
        """
        return {
            'status': 'success',
            'code': code,
            'message': message,
            'data': data,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    @staticmethod
    def error(message: str, code: int = 400, details: Any = None) -> Dict[str, Any]:
        """
        Format error response
        
        Args:
            message: Error message
            code: HTTP status code
            details: Additional error details
            
        Returns:
            Formatted error response dictionary
        """
        return {
            'status': 'error',
            'code': code,
            'message': message,
            'details': details,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    @staticmethod
    def prediction(prediction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format prediction response
        
        Args:
            prediction: Prediction data
            
        Returns:
            Formatted prediction response
        """
        return ResponseFormatter.success(
            data=prediction,
            message="Prediction completed successfully"
        )


class MetricsCollector:
    """Collect and aggregate metrics"""
    
    def __init__(self):
        self.metrics = {
            'total_requests': 0,
            'successful_predictions': 0,
            'failed_predictions': 0,
            'average_processing_time': 0,
            'model_usage': {},
            'total_processing_time': 0
        }
    
    def record_request(self, success: bool, processing_time: float, model: str = None):
        """
        Record a request metric
        
        Args:
            success: Whether request was successful
            processing_time: Time taken to process
            model: Model used (optional)
        """
        self.metrics['total_requests'] += 1
        
        if success:
            self.metrics['successful_predictions'] += 1
        else:
            self.metrics['failed_predictions'] += 1
        
        self.metrics['total_processing_time'] += processing_time
        self.metrics['average_processing_time'] = (
            self.metrics['total_processing_time'] / self.metrics['total_requests']
        )
        
        if model:
            if model not in self.metrics['model_usage']:
                self.metrics['model_usage'][model] = 0
            self.metrics['model_usage'][model] += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        return self.metrics.copy()
    
    def get_success_rate(self) -> float:
        """Get success rate percentage"""
        if self.metrics['total_requests'] == 0:
            return 0.0
        return (self.metrics['successful_predictions'] / self.metrics['total_requests']) * 100


# Global metrics instance
metrics = MetricsCollector()
