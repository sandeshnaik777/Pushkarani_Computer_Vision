"""
Validators Module
Comprehensive input and data validation
"""

import logging
import re
from typing import Any, Tuple, Optional, List, Dict
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class Validator(ABC):
    """Abstract base validator class"""
    
    @abstractmethod
    def validate(self, value: Any) -> Tuple[bool, Optional[str]]:
        """
        Validate a value
        
        Args:
            value: Value to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        pass


class StringValidator(Validator):
    """Validates string values"""
    
    def __init__(self, min_length: int = 0, max_length: int = None,
                 pattern: str = None, allowed_chars: str = None):
        self.min_length = min_length
        self.max_length = max_length
        self.pattern = pattern
        self.allowed_chars = allowed_chars
    
    def validate(self, value: Any) -> Tuple[bool, Optional[str]]:
        """Validate string"""
        if not isinstance(value, str):
            return False, f"Expected string, got {type(value).__name__}"
        
        if len(value) < self.min_length:
            return False, f"String too short (min {self.min_length})"
        
        if self.max_length and len(value) > self.max_length:
            return False, f"String too long (max {self.max_length})"
        
        if self.pattern and not re.match(self.pattern, value):
            return False, f"String doesn't match pattern: {self.pattern}"
        
        if self.allowed_chars:
            for char in value:
                if char not in self.allowed_chars:
                    return False, f"Invalid character: {char}"
        
        return True, None


class NumberValidator(Validator):
    """Validates numeric values"""
    
    def __init__(self, min_value: float = None, max_value: float = None,
                 allow_negative: bool = True):
        self.min_value = min_value
        self.max_value = max_value
        self.allow_negative = allow_negative
    
    def validate(self, value: Any) -> Tuple[bool, Optional[str]]:
        """Validate number"""
        if not isinstance(value, (int, float)):
            return False, f"Expected number, got {type(value).__name__}"
        
        if not self.allow_negative and value < 0:
            return False, "Negative numbers not allowed"
        
        if self.min_value is not None and value < self.min_value:
            return False, f"Value below minimum ({self.min_value})"
        
        if self.max_value is not None and value > self.max_value:
            return False, f"Value above maximum ({self.max_value})"
        
        return True, None


class EmailValidator(Validator):
    """Validates email addresses"""
    
    def validate(self, value: Any) -> Tuple[bool, Optional[str]]:
        """Validate email"""
        if not isinstance(value, str):
            return False, "Email must be string"
        
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        
        if not re.match(pattern, value):
            return False, "Invalid email format"
        
        return True, None


class URLValidator(Validator):
    """Validates URLs"""
    
    def validate(self, value: Any) -> Tuple[bool, Optional[str]]:
        """Validate URL"""
        if not isinstance(value, str):
            return False, "URL must be string"
        
        pattern = r'^https?://[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(/.*)?$'
        
        if not re.match(pattern, value):
            return False, "Invalid URL format"
        
        return True, None


class DateValidator(Validator):
    """Validates dates"""
    
    def validate(self, value: Any) -> Tuple[bool, Optional[str]]:
        """Validate date"""
        from datetime import datetime
        
        if isinstance(value, str):
            for fmt in ('%Y-%m-%d', '%Y/%m/%d', '%d-%m-%Y', '%d/%m/%Y'):
                try:
                    datetime.strptime(value, fmt)
                    return True, None
                except ValueError:
                    continue
            return False, "Invalid date format"
        
        elif isinstance(value, datetime):
            return True, None
        
        else:
            return False, f"Expected string or datetime, got {type(value).__name__}"


class ListValidator(Validator):
    """Validates lists"""
    
    def __init__(self, min_items: int = 0, max_items: int = None,
                 item_validator: Validator = None):
        self.min_items = min_items
        self.max_items = max_items
        self.item_validator = item_validator
    
    def validate(self, value: Any) -> Tuple[bool, Optional[str]]:
        """Validate list"""
        if not isinstance(value, list):
            return False, f"Expected list, got {type(value).__name__}"
        
        if len(value) < self.min_items:
            return False, f"Too few items (min {self.min_items})"
        
        if self.max_items and len(value) > self.max_items:
            return False, f"Too many items (max {self.max_items})"
        
        if self.item_validator:
            for i, item in enumerate(value):
                is_valid, error = self.item_validator.validate(item)
                if not is_valid:
                    return False, f"Item {i}: {error}"
        
        return True, None


class DictValidator(Validator):
    """Validates dictionaries"""
    
    def __init__(self, required_keys: List[str] = None,
                 key_validators: Dict[str, Validator] = None):
        self.required_keys = required_keys or []
        self.key_validators = key_validators or {}
    
    def validate(self, value: Any) -> Tuple[bool, Optional[str]]:
        """Validate dictionary"""
        if not isinstance(value, dict):
            return False, f"Expected dict, got {type(value).__name__}"
        
        # Check required keys
        for key in self.required_keys:
            if key not in value:
                return False, f"Missing required key: {key}"
        
        # Validate keys with validators
        for key, validator in self.key_validators.items():
            if key in value:
                is_valid, error = validator.validate(value[key])
                if not is_valid:
                    return False, f"Key '{key}': {error}"
        
        return True, None


class ChainValidator(Validator):
    """Chains multiple validators"""
    
    def __init__(self, *validators: Validator):
        self.validators = validators
    
    def validate(self, value: Any) -> Tuple[bool, Optional[str]]:
        """Validate with chain"""
        for validator in self.validators:
            is_valid, error = validator.validate(value)
            if not is_valid:
                return False, error
        
        return True, None


__all__ = [
    'Validator',
    'StringValidator',
    'NumberValidator',
    'EmailValidator',
    'URLValidator',
    'DateValidator',
    'ListValidator',
    'DictValidator',
    'ChainValidator'
]
