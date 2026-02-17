"""
Database Utilities Module
Provides database connectivity and query utilities
"""

import logging
import json
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class DatabaseRepository(ABC):
    """Abstract base class for database repositories"""
    
    @abstractmethod
    def create(self, data: Dict[str, Any]) -> Optional[str]:
        """Create a new record"""
        pass
    
    @abstractmethod
    def read(self, record_id: str) -> Optional[Dict[str, Any]]:
        """Read a record by ID"""
        pass
    
    @abstractmethod
    def update(self, record_id: str, data: Dict[str, Any]) -> bool:
        """Update a record"""
        pass
    
    @abstractmethod
    def delete(self, record_id: str) -> bool:
        """Delete a record"""
        pass
    
    @abstractmethod
    def list_all(self) -> List[Dict[str, Any]]:
        """List all records"""
        pass


class PredictionHistory:
    """
    Manages prediction history
    Can be extended to use actual database
    """
    
    def __init__(self, max_records: int = 10000):
        self.max_records = max_records
        self.history = {}
        self.counter = 0
    
    def save_prediction(self, image_hash: str, model_name: str,
                       prediction: Dict[str, Any]) -> str:
        """
        Save a prediction to history
        
        Args:
            image_hash: Hash of the image
            model_name: Model used for prediction
            prediction: Prediction result
            
        Returns:
            Record ID
        """
        self.counter += 1
        record_id = f"pred_{self.counter}_{int(datetime.utcnow().timestamp())}"
        
        record = {
            'id': record_id,
            'image_hash': image_hash,
            'model_name': model_name,
            'prediction': prediction,
            'timestamp': datetime.utcnow().isoformat(),
            'created_at': datetime.utcnow()
        }
        
        # Keep only max_records
        if len(self.history) >= self.max_records:
            # Remove oldest record
            oldest_key = min(self.history.keys(),
                           key=lambda k: self.history[k]['created_at'])
            del self.history[oldest_key]
        
        self.history[record_id] = record
        logger.info(f"Prediction saved: {record_id}")
        
        return record_id
    
    def get_prediction(self, record_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a prediction by ID
        
        Args:
            record_id: ID of the prediction record
            
        Returns:
            Prediction record or None
        """
        return self.history.get(record_id)
    
    def get_predictions_by_model(self, model_name: str) -> List[Dict[str, Any]]:
        """
        Get all predictions for a specific model
        
        Args:
            model_name: Name of the model
            
        Returns:
            List of prediction records
        """
        return [
            record for record in self.history.values()
            if record['model_name'] == model_name
        ]
    
    def get_recent_predictions(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get recent predictions
        
        Args:
            limit: Maximum number of records to return
            
        Returns:
            List of recent prediction records
        """
        sorted_records = sorted(
            self.history.values(),
            key=lambda x: x['created_at'],
            reverse=True
        )
        return sorted_records[:limit]
    
    def get_predictions_for_image(self, image_hash: str) -> List[Dict[str, Any]]:
        """
        Get all predictions for a specific image
        
        Args:
            image_hash: Hash of the image
            
        Returns:
            List of prediction records
        """
        return [
            record for record in self.history.values()
            if record['image_hash'] == image_hash
        ]
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about predictions
        
        Returns:
            Dictionary with statistics
        """
        if not self.history:
            return {
                'total_predictions': 0,
                'unique_images': 0,
                'unique_models': 0,
                'models_used': {}
            }
        
        models_used = {}
        image_hashes = set()
        
        for record in self.history.values():
            model = record['model_name']
            models_used[model] = models_used.get(model, 0) + 1
            image_hashes.add(record['image_hash'])
        
        return {
            'total_predictions': len(self.history),
            'unique_images': len(image_hashes),
            'unique_models': len(models_used),
            'models_used': models_used
        }
    
    def export_to_json(self) -> str:
        """
        Export all predictions to JSON
        
        Returns:
            JSON string of predictions
        """
        records = [
            {**record, 'created_at': record['created_at'].isoformat()}
            for record in self.history.values()
        ]
        return json.dumps(records, indent=2)
    
    def clear_old_records(self, days: int = 7):
        """
        Clear records older than specified days
        
        Args:
            days: Days to keep (remove older)
        """
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        records_to_delete = [
            record_id for record_id, record in self.history.items()
            if record['created_at'] < cutoff_date
        ]
        
        for record_id in records_to_delete:
            del self.history[record_id]
        
        logger.info(f"Deleted {len(records_to_delete)} old records")
    
    def clear_all(self):
        """Clear all prediction history"""
        self.history.clear()
        logger.info("All prediction history cleared")


class CacheManager:
    """
    Simple in-memory cache manager
    Can be extended to use Redis or other backends
    """
    
    def __init__(self, ttl_seconds: int = 3600):
        self.cache = {}
        self.ttl_seconds = ttl_seconds
        self.timestamps = {}
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """
        Set cache value
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (uses default if None)
        """
        self.cache[key] = value
        self.timestamps[key] = datetime.utcnow()
        logger.debug(f"Cache set: {key}")
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get cache value
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found or expired
        """
        if key not in self.cache:
            return None
        
        # Check if expired
        timestamp = self.timestamps.get(key)
        if timestamp:
            age = (datetime.utcnow() - timestamp).total_seconds()
            if age > self.ttl_seconds:
                del self.cache[key]
                del self.timestamps[key]
                return None
        
        logger.debug(f"Cache hit: {key}")
        return self.cache[key]
    
    def delete(self, key: str):
        """
        Delete cache entry
        
        Args:
            key: Cache key
        """
        if key in self.cache:
            del self.cache[key]
            del self.timestamps[key]
            logger.debug(f"Cache deleted: {key}")
    
    def clear(self):
        """Clear all cache"""
        self.cache.clear()
        self.timestamps.clear()
        logger.info("Cache cleared")
    
    def cleanup_expired(self):
        """Remove all expired entries"""
        expired_keys = []
        
        for key, timestamp in self.timestamps.items():
            age = (datetime.utcnow() - timestamp).total_seconds()
            if age > self.ttl_seconds:
                expired_keys.append(key)
        
        for key in expired_keys:
            if key in self.cache:
                del self.cache[key]
            del self.timestamps[key]
        
        if expired_keys:
            logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics
        
        Returns:
            Dictionary with cache stats
        """
        return {
            'total_entries': len(self.cache),
            'ttl_seconds': self.ttl_seconds,
            'memory_usage_estimate': sum(
                len(str(v)) for v in self.cache.values()
            ) / 1024  # KB
        }
