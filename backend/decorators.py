"""
Decorators Module
Common decorators for caching, timing, error handling
"""

import logging
import time
import functools
from typing import Callable, Any, Optional, Dict
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


def timer_decorator(func: Callable) -> Callable:
    """
    Decorator to time function execution
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            logger.info(f"{func.__name__} completed in {duration:.4f} seconds")
            return result
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"{func.__name__} failed after {duration:.4f} seconds: {str(e)}")
            raise
    
    return wrapper


def cache_result(ttl_seconds: int = 300) -> Callable:
    """
    Decorator to cache function results
    
    Args:
        ttl_seconds: Time to live for cache in seconds
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        cache = {}
        timestamps = {}
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from arguments
            cache_key = f"{func.__name__}_{args}_{kwargs}"
            
            # Check if cached and not expired
            if cache_key in cache:
                timestamp = timestamps.get(cache_key)
                if timestamp:
                    age = (datetime.utcnow() - timestamp).total_seconds()
                    if age < ttl_seconds:
                        logger.debug(f"Cache hit for {func.__name__}")
                        return cache[cache_key]
                    else:
                        del cache[cache_key]
                        del timestamps[cache_key]
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache[cache_key] = result
            timestamps[cache_key] = datetime.utcnow()
            logger.debug(f"Cache set for {func.__name__}")
            
            return result
        
        return wrapper
    
    return decorator


def retry_on_exception(max_retries: int = 3, delay_seconds: float = 1.0,
                      backoff: float = 2.0) -> Callable:
    """
    Decorator to retry function on exception
    
    Args:
        max_retries: Maximum number of retries
        delay_seconds: Initial delay between retries
        backoff: Backoff multiplier for each retry
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay_seconds
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    if attempt < max_retries:
                        logger.warning(
                            f"{func.__name__} attempt {attempt + 1} failed: {str(e)}. "
                            f"Retrying in {current_delay:.2f} seconds..."
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(
                            f"{func.__name__} failed after {max_retries} retries: {str(e)}"
                        )
            
            raise last_exception
        
        return wrapper
    
    return decorator


def validate_arguments(**validators: Callable) -> Callable:
    """
    Decorator to validate function arguments
    
    Args:
        **validators: Validator functions for each argument
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Validate arguments
            for arg_name, validator in validators.items():
                if arg_name in kwargs:
                    value = kwargs[arg_name]
                    if not validator(value):
                        raise ValueError(f"Invalid value for {arg_name}: {value}")
                
            return func(*args, **kwargs)
        
        return wrapper
    
    return decorator


def measure_performance(func: Callable) -> Callable:
    """
    Decorator to measure performance metrics
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = 0
        
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            
            logger.info(
                f"Performance - {func.__name__}: "
                f"Duration: {duration:.4f}s"
            )
            
            return result
        except Exception as e:
            duration = time.time() - start_time
            logger.error(
                f"Performance - {func.__name__} FAILED: "
                f"Duration: {duration:.4f}s, "
                f"Error: {str(e)}"
            )
            raise
    
    return wrapper


def deprecated(reason: str = "") -> Callable:
    """
    Decorator to mark function as deprecated
    
    Args:
        reason: Reason for deprecation
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            message = f"{func.__name__} is deprecated"
            if reason:
                message += f": {reason}"
            
            logger.warning(message)
            return func(*args, **kwargs)
        
        wrapper.deprecated = True
        wrapper.deprecation_reason = reason
        
        return wrapper
    
    return decorator


def require_fields(*required_fields: str) -> Callable:
    """
    Decorator to require certain dictionary fields
    
    Args:
        *required_fields: Field names that must be present
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(data_dict: dict, *args, **kwargs):
            missing_fields = []
            
            for field in required_fields:
                if field not in data_dict:
                    missing_fields.append(field)
            
            if missing_fields:
                raise ValueError(f"Missing required fields: {missing_fields}")
            
            return func(data_dict, *args, **kwargs)
        
        return wrapper
    
    return decorator


def synchronized(func: Callable) -> Callable:
    """
    Decorator for thread-safe function execution
    Uses locking to ensure only one thread executes at a time
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    import threading
    
    func._lock = threading.Lock()
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with func._lock:
            return func(*args, **kwargs)
    
    return wrapper


__all__ = [
    'timer_decorator',
    'cache_result',
    'retry_on_exception',
    'validate_arguments',
    'measure_performance',
    'deprecated',
    'require_fields',
    'synchronized'
]
