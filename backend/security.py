"""
Security Module
Implements security best practices and protections
"""

import logging
import hashlib
import secrets
import base64
from typing import Tuple, Optional, Dict, Any
from datetime import datetime, timedelta
import hmac

logger = logging.getLogger(__name__)


class SecurityManager:
    """
    Manages security operations including hashing, validation, and protection
    """
    
    @staticmethod
    def hash_data(data: str, algorithm: str = 'sha256') -> str:
        """
        Hash data using specified algorithm
        
        Args:
            data: Data to hash
            algorithm: Hash algorithm ('sha256', 'sha512', 'md5')
            
        Returns:
            Hex encoded hash string
        """
        if algorithm == 'sha256':
            return hashlib.sha256(data.encode()).hexdigest()
        elif algorithm == 'sha512':
            return hashlib.sha512(data.encode()).hexdigest()
        elif algorithm == 'md5':
            return hashlib.md5(data.encode()).hexdigest()
        else:
            logger.error(f"Unsupported hash algorithm: {algorithm}")
            return hashlib.sha256(data.encode()).hexdigest()
    
    @staticmethod
    def verify_hash(data: str, hash_value: str, algorithm: str = 'sha256') -> bool:
        """
        Verify data against hash
        
        Args:
            data: Original data
            hash_value: Hash to verify against
            algorithm: Hash algorithm used
            
        Returns:
            True if hash matches, False otherwise
        """
        computed_hash = SecurityManager.hash_data(data, algorithm)
        return hmac.compare_digest(computed_hash, hash_value)
    
    @staticmethod
    def generate_token(length: int = 32) -> str:
        """
        Generate secure random token
        
        Args:
            length: Token length in bytes
            
        Returns:
            Base64 encoded token
        """
        random_bytes = secrets.token_bytes(length)
        return base64.b64encode(random_bytes).decode('utf-8')
    
    @staticmethod
    def hash_image(image_bytes: bytes) -> str:
        """
        Generate hash of image data
        
        Args:
            image_bytes: Image data as bytes
            
        Returns:
            Hex encoded hash
        """
        return hashlib.sha256(image_bytes).hexdigest()
    
    @staticmethod
    def sanitize_input(user_input: str, max_length: int = 1000) -> str:
        """
        Sanitize user input to prevent injection attacks
        
        Args:
            user_input: User provided input
            max_length: Maximum allowed length
            
        Returns:
            Sanitized string
        """
        if not isinstance(user_input, str):
            return ""
        
        # Strip whitespace
        sanitized = user_input.strip()
        
        # Limit length
        sanitized = sanitized[:max_length]
        
        # Remove potentially dangerous characters
        dangerous_chars = ['<', '>', '"', "'", '&', ';', '|', '$', '`']
        for char in dangerous_chars:
            sanitized = sanitized.replace(char, '')
        
        return sanitized
    
    @staticmethod
    def validate_api_key(api_key: str, valid_keys: list) -> bool:
        """
        Validate API key
        
        Args:
            api_key: Provided API key
            valid_keys: List of valid API keys
            
        Returns:
            True if valid, False otherwise
        """
        if not api_key or not valid_keys:
            return False
        
        for valid_key in valid_keys:
            if hmac.compare_digest(api_key, valid_key):
                return True
        
        return False


class RateLimiter:
    """
    Implements rate limiting to protect against abuse
    """
    
    def __init__(self, requests_per_minute: int = 60):
        """
        Initialize RateLimiter
        
        Args:
            requests_per_minute: Maximum requests allowed per minute
        """
        self.requests_per_minute = requests_per_minute
        self.client_requests = {}
    
    def is_rate_limited(self, client_id: str) -> bool:
        """
        Check if client is rate limited
        
        Args:
            client_id: Client identifier (IP, user ID, etc.)
            
        Returns:
            True if rate limited, False otherwise
        """
        now = datetime.utcnow()
        
        if client_id not in self.client_requests:
            self.client_requests[client_id] = []
        
        requests = self.client_requests[client_id]
        
        # Remove requests older than 1 minute
        cutoff = now - timedelta(minutes=1)
        requests[:] = [req_time for req_time in requests if req_time > cutoff]
        
        # Check if over limit
        if len(requests) >= self.requests_per_minute:
            logger.warning(f"Rate limit exceeded for client: {client_id}")
            return True
        
        # Add current request
        requests.append(now)
        return False
    
    def get_requests_count(self, client_id: str) -> int:
        """
        Get current request count for client
        
        Args:
            client_id: Client identifier
            
        Returns:
            Number of requests in current minute
        """
        if client_id not in self.client_requests:
            return 0
        
        now = datetime.utcnow()
        cutoff = now - timedelta(minutes=1)
        
        requests = self.client_requests[client_id]
        valid_requests = [req for req in requests if req > cutoff]
        
        return len(valid_requests)
    
    def reset_client(self, client_id: str):
        """
        Reset rate limit for a client
        
        Args:
            client_id: Client identifier
        """
        if client_id in self.client_requests:
            del self.client_requests[client_id]
    
    def cleanup(self):
        """Clean up old entries"""
        now = datetime.utcnow()
        cutoff = now - timedelta(minutes=5)
        
        clients_to_remove = []
        
        for client_id, requests in self.client_requests.items():
            valid_requests = [req for req in requests if req > cutoff]
            
            if not valid_requests:
                clients_to_remove.append(client_id)
            else:
                self.client_requests[client_id] = valid_requests
        
        for client_id in clients_to_remove:
            del self.client_requests[client_id]


class InputValidator:
    """
    Validates various input types for security and correctness
    """
    
    @staticmethod
    def validate_filename(filename: str) -> Tuple[bool, str]:
        """
        Validate filename for security
        
        Args:
            filename: Filename to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not filename:
            return False, "Filename cannot be empty"
        
        # Check length
        if len(filename) > 255:
            return False, "Filename too long"
        
        # Check for path traversal
        if '..' in filename or '/' in filename or '\\' in filename:
            return False, "Invalid filename - path traversal detected"
        
        # Check for null bytes
        if '\x00' in filename:
            return False, "Invalid characters in filename"
        
        return True, ""
    
    @staticmethod
    def validate_json(json_string: str) -> Tuple[bool, Optional[Dict]]:
        """
        Validate and parse JSON
        
        Args:
            json_string: JSON string to validate
            
        Returns:
            Tuple of (is_valid, parsed_json or error_message)
        """
        import json
        
        try:
            parsed = json.loads(json_string)
            return True, parsed
        except json.JSONDecodeError as e:
            return False, f"Invalid JSON: {str(e)}"
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """
        Basic email validation
        
        Args:
            email: Email address to validate
            
        Returns:
            True if valid format, False otherwise
        """
        import re
        
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))
    
    @staticmethod
    def validate_url(url: str) -> bool:
        """
        Basic URL validation
        
        Args:
            url: URL to validate
            
        Returns:
            True if valid format, False otherwise
        """
        import re
        
        url_pattern = r'^https?://[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(/.*)?$'
        return bool(re.match(url_pattern, url))


__all__ = [
    'SecurityManager',
    'RateLimiter',
    'InputValidator'
]
