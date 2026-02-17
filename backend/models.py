"""
Data Models Module
Pydantic-like models for request/response validation
"""

from typing import Dict, List, Any, Optional
from datetime import datetime


class PredictionRequest:
    """Request model for prediction endpoint"""
    
    def __init__(self, image_base64: str = None, model_name: str = None, 
                 ensemble: bool = False, metadata: Dict[str, Any] = None):
        self.image_base64 = image_base64
        self.model_name = model_name
        self.ensemble = ensemble
        self.metadata = metadata or {}
        self.timestamp = datetime.utcnow()
    
    def validate(self) -> tuple:
        """
        Validate request
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not self.image_base64:
            return False, "image_base64 is required"
        
        if not self.ensemble and not self.model_name:
            return False, "model_name is required when ensemble is False"
        
        if self.ensemble and self.model_name:
            return False, "Cannot specify model_name when ensemble is True"
        
        return True, ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'model_name': self.model_name,
            'ensemble': self.ensemble,
            'has_metadata': bool(self.metadata),
            'timestamp': self.timestamp.isoformat()
        }


class PredictionResponse:
    """Response model for prediction endpoint"""
    
    def __init__(self, model: str, predictions: Dict[str, float], 
                 top_class: str, confidence: float, 
                 inference_time: float = 0, metadata: Dict[str, Any] = None):
        self.model = model
        self.predictions = predictions
        self.top_class = top_class
        self.confidence = confidence
        self.inference_time = inference_time
        self.metadata = metadata or {}
        self.timestamp = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'model': self.model,
            'predictions': self.predictions,
            'top_class': self.top_class,
            'confidence': self.confidence,
            'inference_time': f"{self.inference_time:.4f}s",
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }


class WaterQualityAnalysis:
    """Water quality analysis result"""
    
    def __init__(self, clarity_score: float, turbidity_level: str, 
                 algae_presence: bool, pollution_index: float,
                 overall_quality: str):
        self.clarity_score = clarity_score
        self.turbidity_level = turbidity_level
        self.algae_presence = algae_presence
        self.pollution_index = pollution_index
        self.overall_quality = overall_quality
        self.timestamp = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'clarity_score': self.clarity_score,
            'turbidity_level': self.turbidity_level,
            'algae_presence': self.algae_presence,
            'pollution_index': self.pollution_index,
            'overall_quality': self.overall_quality,
            'timestamp': self.timestamp.isoformat()
        }


class ChatbotQuery:
    """Chatbot query request"""
    
    def __init__(self, query: str, context: Optional[Dict[str, Any]] = None):
        self.query = query
        self.context = context or {}
        self.timestamp = datetime.utcnow()
    
    def validate(self) -> tuple:
        """
        Validate query
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not self.query:
            return False, "Query cannot be empty"
        
        if len(self.query) > 1000:
            return False, "Query too long (max 1000 characters)"
        
        return True, ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'query': self.query,
            'context_provided': bool(self.context),
            'timestamp': self.timestamp.isoformat()
        }


class ChatbotResponse:
    """Chatbot response"""
    
    def __init__(self, answer: str, confidence: float = 1.0,
                 sources: List[str] = None, related_topics: List[str] = None):
        self.answer = answer
        self.confidence = confidence
        self.sources = sources or []
        self.related_topics = related_topics or []
        self.timestamp = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'answer': self.answer,
            'confidence': self.confidence,
            'sources': self.sources,
            'related_topics': self.related_topics,
            'timestamp': self.timestamp.isoformat()
        }


class HealthCheckResponse:
    """Health check response"""
    
    def __init__(self, status: str, models_loaded: int, 
                 models_available: int, uptime_seconds: float,
                 memory_usage_mb: float = 0, cpu_usage_percent: float = 0):
        self.status = status
        self.models_loaded = models_loaded
        self.models_available = models_available
        self.uptime_seconds = uptime_seconds
        self.memory_usage_mb = memory_usage_mb
        self.cpu_usage_percent = cpu_usage_percent
        self.timestamp = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'status': self.status,
            'models_loaded': self.models_loaded,
            'models_available': self.models_available,
            'uptime': f"{self.uptime_seconds:.2f}s",
            'memory_usage_mb': f"{self.memory_usage_mb:.2f}",
            'cpu_usage_percent': f"{self.cpu_usage_percent:.2f}",
            'timestamp': self.timestamp.isoformat()
        }


class APIMetrics:
    """API metrics and statistics"""
    
    def __init__(self, total_requests: int = 0, successful_requests: int = 0,
                 failed_requests: int = 0, avg_response_time: float = 0,
                 total_predictions: int = 0, models_used: Dict[str, int] = None):
        self.total_requests = total_requests
        self.successful_requests = successful_requests
        self.failed_requests = failed_requests
        self.avg_response_time = avg_response_time
        self.total_predictions = total_predictions
        self.models_used = models_used or {}
        self.timestamp = datetime.utcnow()
    
    def get_success_rate(self) -> float:
        """Calculate success rate"""
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_requests': self.total_requests,
            'successful_requests': self.successful_requests,
            'failed_requests': self.failed_requests,
            'success_rate': f"{self.get_success_rate():.2f}%",
            'avg_response_time': f"{self.avg_response_time:.4f}s",
            'total_predictions': self.total_predictions,
            'models_used': self.models_used,
            'timestamp': self.timestamp.isoformat()
        }


class EnvironmentInfo:
    """Application environment information"""
    
    def __init__(self, app_version: str, python_version: str,
                 tensorflow_version: str, flask_version: str,
                 environment: str = 'development'):
        self.app_version = app_version
        self.python_version = python_version
        self.tensorflow_version = tensorflow_version
        self.flask_version = flask_version
        self.environment = environment
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'app_version': self.app_version,
            'python_version': self.python_version,
            'tensorflow_version': self.tensorflow_version,
            'flask_version': self.flask_version,
            'environment': self.environment
        }
