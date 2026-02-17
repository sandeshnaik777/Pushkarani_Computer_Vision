"""
Helper Utilities Module
Common helper functions and utilities
"""

import logging
import os
import json
import hashlib
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import pickle
import gzip

logger = logging.getLogger(__name__)


class FileHelper:
    """File operations helper"""
    
    @staticmethod
    def ensure_directory_exists(directory: str):
        """
        Ensure directory exists, create if needed
        
        Args:
            directory: Directory path
        """
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            logger.info(f"Created directory: {directory}")
    
    @staticmethod
    def safe_read_json(filepath: str) -> Optional[Dict[str, Any]]:
        """
        Safely read JSON file
        
        Args:
            filepath: Path to JSON file
            
        Returns:
            Parsed JSON or None
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error reading JSON file {filepath}: {str(e)}")
            return None
    
    @staticmethod
    def safe_write_json(filepath: str, data: Dict[str, Any], indent: int = 2):
        """
        Safely write JSON file
        
        Args:
            filepath: Path to JSON file
            data: Data to write
            indent: JSON indentation
        """
        try:
            FileHelper.ensure_directory_exists(os.path.dirname(filepath))
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=indent)
            logger.info(f"JSON file written: {filepath}")
        except Exception as e:
            logger.error(f"Error writing JSON file {filepath}: {str(e)}")
    
    @staticmethod
    def safe_read_pickle(filepath: str) -> Optional[Any]:
        """
        Safely read pickle file
        
        Args:
            filepath: Path to pickle file
            
        Returns:
            Unpickled object or None
        """
        try:
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"Error reading pickle file {filepath}: {str(e)}")
            return None
    
    @staticmethod
    def safe_write_pickle(filepath: str, data: Any):
        """
        Safely write pickle file
        
        Args:
            filepath: Path to pickle file
            data: Data to pickle
        """
        try:
            FileHelper.ensure_directory_exists(os.path.dirname(filepath))
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
            logger.info(f"Pickle file written: {filepath}")
        except Exception as e:
            logger.error(f"Error writing pickle file {filepath}: {str(e)}")
    
    @staticmethod
    def get_file_size(filepath: str) -> int:
        """
        Get file size in bytes
        
        Args:
            filepath: Path to file
            
        Returns:
            File size in bytes
        """
        try:
            return os.path.getsize(filepath)
        except Exception as e:
            logger.error(f"Error getting file size: {str(e)}")
            return 0
    
    @staticmethod
    def file_exists(filepath: str) -> bool:
        """
        Check if file exists
        
        Args:
            filepath: Path to file
            
        Returns:
            True if file exists
        """
        return os.path.isfile(filepath)


class StringHelper:
    """String manipulation helper"""
    
    @staticmethod
    def truncate(text: str, max_length: int = 100, suffix: str = "...") -> str:
        """
        Truncate string to max length
        
        Args:
            text: Text to truncate
            max_length: Maximum length
            suffix: Suffix to add if truncated
            
        Returns:
            Truncated string
        """
        if len(text) <= max_length:
            return text
        
        return text[:max_length - len(suffix)] + suffix
    
    @staticmethod
    def slugify(text: str) -> str:
        """
        Convert text to URL-friendly slug
        
        Args:
            text: Text to slugify
            
        Returns:
            Slugified string
        """
        import re
        
        # Convert to lowercase
        text = text.lower()
        
        # Replace spaces with hyphens
        text = re.sub(r'\s+', '-', text)
        
        # Remove non-alphanumeric characters except hyphens
        text = re.sub(r'[^a-z0-9\-]', '', text)
        
        # Remove consecutive hyphens
        text = re.sub(r'-+', '-', text)
        
        # Strip hyphens from start and end
        text = text.strip('-')
        
        return text
    
    @staticmethod
    def camel_case_to_snake_case(name: str) -> str:
        """
        Convert camelCase to snake_case
        
        Args:
            name: CamelCase string
            
        Returns:
            snake_case string
        """
        import re
        
        name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()


class DateTimeHelper:
    """DateTime manipulation helper"""
    
    @staticmethod
    def get_timestamp() -> int:
        """Get current Unix timestamp"""
        return int(datetime.utcnow().timestamp())
    
    @staticmethod
    def format_duration(seconds: float, precision: int = 2) -> str:
        """
        Format duration in seconds to human-readable string
        
        Args:
            seconds: Duration in seconds
            precision: Decimal precision
            
        Returns:
            Formatted duration string
        """
        if seconds < 60:
            return f"{seconds:.{precision}f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.{precision}f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.{precision}f}h"
    
    @staticmethod
    def format_size(bytes_size: int, precision: int = 2) -> str:
        """
        Format bytes to human-readable size
        
        Args:
            bytes_size: Size in bytes
            precision: Decimal precision
            
        Returns:
            Formatted size string
        """
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if bytes_size < 1024:
                return f"{bytes_size:.{precision}f}{unit}"
            bytes_size /= 1024
        
        return f"{bytes_size:.{precision}f}PB"
    
    @staticmethod
    def get_age(timestamp: datetime) -> str:
        """
        Get human-readable age of timestamp
        
        Args:
            timestamp: DateTime object
            
        Returns:
            Age string
        """
        now = datetime.utcnow()
        delta = now - timestamp
        
        if delta.days > 0:
            return f"{delta.days}d ago"
        elif delta.seconds >= 3600:
            hours = delta.seconds // 3600
            return f"{hours}h ago"
        elif delta.seconds >= 60:
            minutes = delta.seconds // 60
            return f"{minutes}m ago"
        else:
            return f"{delta.seconds}s ago"


class DictHelper:
    """Dictionary helper functions"""
    
    @staticmethod
    def deep_merge(dict1: Dict, dict2: Dict) -> Dict:
        """
        Deep merge two dictionaries
        
        Args:
            dict1: Base dictionary
            dict2: Dictionary to merge
            
        Returns:
            Merged dictionary
        """
        result = dict1.copy()
        
        for key, value in dict2.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = DictHelper.deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    @staticmethod
    def flatten(nested_dict: Dict, parent_key: str = '', sep: str = '.') -> Dict:
        """
        Flatten nested dictionary
        
        Args:
            nested_dict: Nested dictionary
            parent_key: Parent key prefix
            sep: Separator for keys
            
        Returns:
            Flattened dictionary
        """
        items = []
        
        for key, value in nested_dict.items():
            new_key = f"{parent_key}{sep}{key}" if parent_key else key
            
            if isinstance(value, dict):
                items.extend(
                    DictHelper.flatten(value, new_key, sep=sep).items()
                )
            else:
                items.append((new_key, value))
        
        return dict(items)


__all__ = [
    'FileHelper',
    'StringHelper',
    'DateTimeHelper',
    'DictHelper'
]
