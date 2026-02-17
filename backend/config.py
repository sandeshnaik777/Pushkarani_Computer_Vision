"""
Configuration Management Module
Handles all application configuration with validation and environment support
"""

import os
import json
from typing import Dict, Any
from dataclasses import dataclass, asdict
from datetime import timedelta
import logging

logger = logging.getLogger(__name__)


@dataclass
class ImageConfig:
    """Image processing configuration"""
    height: int = 224
    width: int = 224
    max_size_mb: int = 50
    supported_formats: tuple = ('jpg', 'jpeg', 'png', 'bmp', 'webp')
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CacheConfig:
    """Caching configuration"""
    cache_type: str = 'simple'
    cache_timeout: int = 3600
    cache_default_timeout: int = 300
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ModelConfig:
    """Model configuration and paths"""
    model_paths: Dict[str, str] = None
    class_indices_path: str = '../densenet/class_indices.json'
    default_model: str = 'densenet'
    ensemble_enabled: bool = True
    confidence_threshold: float = 0.7
    
    def __post_init__(self):
        if self.model_paths is None:
            self.model_paths = {
                'densenet': '../densenet/best_model.keras',
                'efficientnetv2': '../efficientnetv2/best_model.keras',
                'convnext': '../convnext/best_model.keras',
                'vgg16': '../vgg16/best_model.keras',
                'resnet50': '../resnet50/best_model.keras',
                'mobilenet': '../mobilenet/best_model.keras',
                'mobilenetv3': '../mobilenetv3/best_model.keras',
                'inception': '../inception/best_model.keras',
                'swin': '../swin/best_model.keras',
                'dinov2': '../dinov2/best_model.keras'
            }
    
    def to_dict(self) -> Dict[str, Any]:
        config_dict = asdict(self)
        config_dict['model_paths'] = self.model_paths
        return config_dict


@dataclass
class APIConfig:
    """API configuration"""
    host: str = '0.0.0.0'
    port: int = 5000
    debug: bool = False
    json_sort_keys: bool = False
    max_request_size: int = 50 * 1024 * 1024  # 50MB
    request_timeout: int = 60
    rate_limit_enabled: bool = True
    rate_limit_requests: int = 100
    rate_limit_period: int = 60  # seconds
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = 'INFO'
    format_string: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    file_enabled: bool = True
    log_file: str = 'app.log'
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class Config:
    """
    Master configuration class
    Loads and manages all application settings from environment variables and defaults
    """
    
    def __init__(self):
        self.env_name = os.getenv('ENVIRONMENT', 'development')
        self.image = ImageConfig()
        self.cache = CacheConfig()
        self.model = ModelConfig()
        self.api = APIConfig()
        self.logging = LoggingConfig()
        
        self._load_from_env()
        self._validate()
    
    def _load_from_env(self):
        """Load settings from environment variables"""
        # API Settings
        self.api.debug = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
        self.api.host = os.getenv('API_HOST', '0.0.0.0')
        self.api.port = int(os.getenv('API_PORT', '5000'))
        
        # Model Settings
        self.model.confidence_threshold = float(os.getenv('CONFIDENCE_THRESHOLD', '0.7'))
        self.model.ensemble_enabled = os.getenv('ENSEMBLE_ENABLED', 'True').lower() == 'true'
        
        # Cache Settings
        self.cache.cache_timeout = int(os.getenv('CACHE_TIMEOUT', '3600'))
        
        # Logging Settings
        self.logging.level = os.getenv('LOG_LEVEL', 'INFO')
        
        logger.info(f"✓ Configuration loaded for {self.env_name} environment")
    
    def _validate(self):
        """Validate configuration values"""
        if self.image.height <= 0 or self.image.width <= 0:
            raise ValueError("Image height and width must be positive")
        
        if self.model.confidence_threshold < 0 or self.model.confidence_threshold > 1:
            raise ValueError("Confidence threshold must be between 0 and 1")
        
        if self.api.port < 1024 or self.api.port > 65535:
            raise ValueError("API port must be between 1024 and 65535")
        
        logger.info("✓ Configuration validation passed")
    
    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        """Convert entire configuration to dictionary"""
        return {
            'environment': self.env_name,
            'image': self.image.to_dict(),
            'cache': self.cache.to_dict(),
            'model': self.model.to_dict(),
            'api': self.api.to_dict(),
            'logging': self.logging.to_dict()
        }
    
    def __repr__(self) -> str:
        return f"<Config environment={self.env_name}>"


# Global config instance
config = Config()
