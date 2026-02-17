"""
Constants and Configuration Values
Central location for all application constants
"""

# API Constants
API_VERSION = "1.0.0"
API_TITLE = "Pushkarani Classification API"
API_DESCRIPTION = "Advanced computer vision API for temple pond classification and water quality analysis"

# Status Codes
SUCCESS_STATUS = 'success'
ERROR_STATUS = 'error'
PENDING_STATUS = 'pending'

# HTTP Status Code
HTTP_OK = 200
HTTP_CREATED = 201
HTTP_BAD_REQUEST = 400
HTTP_UNAUTHORIZED = 401
HTTP_FORBIDDEN = 403
HTTP_NOT_FOUND = 404
HTTP_CONFLICT = 409
HTTP_INTERNAL_ERROR = 500
HTTP_SERVICE_UNAVAILABLE = 503

# Model Configuration
DEFAULT_IMAGE_HEIGHT = 224
DEFAULT_IMAGE_WIDTH = 224
DEFAULT_MODEL = 'densenet'
SUPPORTED_MODELS = [
    'densenet',
    'efficientnetv2',
    'convnext',
    'vgg16',
    'resnet50',
    'mobilenet',
    'mobilenetv3',
    'inception',
    'swin',
    'dinov2'
]

# Classification Classes
CLASSIFICATION_CLASSES = {
    'type-1': 'Teppakulam (Large Step Tank)',
    'type-2': 'Kalyani (Common Tank)',
    'type-3': 'Kunda (Small Tank)'
}

# File Upload
MAX_FILE_SIZE_MB = 50
ALLOWED_IMAGE_EXTENSIONS = {'jpg', 'jpeg', 'png', 'bmp', 'webp'}
ALLOWED_MIME_TYPES = {
    'image/jpeg',
    'image/png',
    'image/bmp',
    'image/webp'
}

# Cache Configuration
DEFAULT_CACHE_TTL = 3600  # 1 hour in seconds
MAX_CACHE_ENTRIES = 1000

# Rate Limiting
DEFAULT_RATE_LIMIT = 100  # requests per minute
RATE_LIMIT_WINDOW = 60  # seconds

# Logging
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_LEVEL = 'INFO'
LOG_FILE = 'app.log'
LOG_MAX_SIZE = 10 * 1024 * 1024  # 10MB

# Performance Thresholds
SLOW_REQUEST_THRESHOLD = 5.0  # seconds
HIGH_MEMORY_USAGE = 80  # percentage
HIGH_CPU_USAGE = 75  # percentage

# Water Quality Analysis
WATER_QUALITY_LEVELS = {
    'excellent': (0.8, 1.0),
    'good': (0.6, 0.8),
    'fair': (0.4, 0.6),
    'poor': (0.0, 0.4)
}

# Error Messages
ERROR_INVALID_IMAGE = "Invalid image file provided"
ERROR_MODEL_NOT_FOUND = "Model not found"
ERROR_MODEL_LOAD_FAILED = "Failed to load model"
ERROR_PREDICTION_FAILED = "Prediction failed"
ERROR_RATE_LIMITED = "Rate limit exceeded"
ERROR_SERVICE_UNAVAILABLE = "Service temporarily unavailable"

# Success Messages
SUCCESS_PREDICTION = "Prediction completed successfully"
SUCCESS_UPLOAD = "File uploaded successfully"
SUCCESS_MODEL_LOADED = "Model loaded successfully"

# For production/development
ENVIRONMENT = 'development'
DEBUG_MODE = True
ENABLE_PROFILING = False

# Database
DATABASE_URL = 'sqlite:///app.db'
DATABASE_ECHO = False

# Security
SECRET_KEY = 'your-secret-key-change-in-production'
JWT_EXPIRATION_HOURS = 24
PASSWORD_MIN_LENGTH = 8
