"""
API Routes Module
Centralized route handlers for the Flask application
"""

import logging
from typing import Tuple, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class APIRoutes:
    """
    Centralized API route handlers
    Provides reusable functions for common endpoint operations
    """
    
    @staticmethod
    def handle_health_check(monitor=None, model_manager=None) -> Tuple[Dict[str, Any], int]:
        """
        Handle health check request
        
        Args:
            monitor: Performance monitor instance
            model_manager: Model manager instance
            
        Returns:
            Tuple of (response_dict, status_code)
        """
        try:
            response = {
                'status': 'healthy',
                'timestamp': datetime.utcnow().isoformat(),
                'message': 'API is operational'
            }
            
            if monitor:
                metrics = monitor.get_metrics_summary()
                response['metrics'] = metrics
            
            if model_manager:
                response['models'] = {
                    'available': model_manager.get_available_models(),
                    'loaded': model_manager.get_loaded_models()
                }
            
            return response, 200
        
        except Exception as e:
            logger.error(f"Error in health check: {str(e)}")
            return {
                'status': 'unhealthy',
                'message': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }, 503
    
    @staticmethod
    def handle_models_info(model_manager) -> Tuple[Dict[str, Any], int]:
        """
        Handle models info request
        
        Args:
            model_manager: Model manager instance
            
        Returns:
            Tuple of (response_dict, status_code)
        """
        try:
            available = model_manager.get_available_models()
            loaded = model_manager.get_loaded_models()
            stats = model_manager.get_all_stats()
            
            return {
                'status': 'success',
                'models': {
                    'available': available,
                    'loaded': loaded,
                    'total_available': len(available),
                    'total_loaded': len(loaded),
                    'stats': stats
                },
                'timestamp': datetime.utcnow().isoformat()
            }, 200
        
        except Exception as e:
            logger.error(f"Error getting models info: {str(e)}")
            return {'status': 'error', 'message': str(e)}, 500
    
    @staticmethod
    def handle_load_model(model_name: str, model_manager) -> Tuple[Dict[str, Any], int]:
        """
        Handle model loading request
        
        Args:
            model_name: Name of model to load
            model_manager: Model manager instance
            
        Returns:
            Tuple of (response_dict, status_code)
        """
        try:
            if model_name not in model_manager.get_available_models():
                return {
                    'status': 'error',
                    'message': f'Model not found: {model_name}',
                    'available_models': model_manager.get_available_models()
                }, 404
            
            success = model_manager.load_model(model_name)
            
            if success:
                return {
                    'status': 'success',
                    'message': f'Model loaded: {model_name}',
                    'model': model_name
                }, 200
            else:
                return {
                    'status': 'error',
                    'message': f'Failed to load model: {model_name}'
                }, 500
        
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return {'status': 'error', 'message': str(e)}, 500
    
    @staticmethod
    def handle_unload_model(model_name: str, model_manager) -> Tuple[Dict[str, Any], int]:
        """
        Handle model unloading request
        
        Args:
            model_name: Name of model to unload
            model_manager: Model manager instance
            
        Returns:
            Tuple of (response_dict, status_code)
        """
        try:
            model_manager.unload_model(model_name)
            
            return {
                'status': 'success',
                'message': f'Model unloaded: {model_name}',
                'still_loaded': model_manager.get_loaded_models()
            }, 200
        
        except Exception as e:
            logger.error(f"Error unloading model: {str(e)}")
            return {'status': 'error', 'message': str(e)}, 500
    
    @staticmethod
    def handle_metrics_request(monitor) -> Tuple[Dict[str, Any], int]:
        """
        Handle metrics request
        
        Args:
            monitor: Performance monitor instance
            
        Returns:
            Tuple of (response_dict, status_code)
        """
        try:
            metrics = monitor.get_metrics_summary()
            recent_errors = monitor.get_recent_errors(limit=5)
            
            return {
                'status': 'success',
                'metrics': metrics,
                'recent_errors': recent_errors,
                'timestamp': datetime.utcnow().isoformat()
            }, 200
        
        except Exception as e:
            logger.error(f"Error getting metrics: {str(e)}")
            return {'status': 'error', 'message': str(e)}, 500
    
    @staticmethod
    def handle_reset_metrics(monitor) -> Tuple[Dict[str, Any], int]:
        """
        Handle metrics reset request
        
        Args:
            monitor: Performance monitor instance
            
        Returns:
            Tuple of (response_dict, status_code)
        """
        try:
            monitor.reset_metrics()
            
            return {
                'status': 'success',
                'message': 'Metrics reset successfully',
                'timestamp': datetime.utcnow().isoformat()
            }, 200
        
        except Exception as e:
            logger.error(f"Error resetting metrics: {str(e)}")
            return {'status': 'error', 'message': str(e)}, 500
    
    @staticmethod
    def handle_version_info() -> Tuple[Dict[str, Any], int]:
        """
        Handle version info request
        
        Returns:
            Tuple of (response_dict, status_code)
        """
        import sys
        import flask
        
        try:
            import tensorflow as tf
            tf_version = tf.__version__
        except ImportError:
            tf_version = "Not installed"
        
        return {
            'status': 'success',
            'version_info': {
                'app_version': '1.0.0',
                'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                'flask_version': flask.__version__,
                'tensorflow_version': tf_version,
                'timestamp': datetime.utcnow().isoformat()
            }
        }, 200
    
    @staticmethod
    def handle_config_info(config) -> Tuple[Dict[str, Any], int]:
        """
        Handle config info request
        
        Args:
            config: Application configuration object
            
        Returns:
            Tuple of (response_dict, status_code)
        """
        try:
            return {
                'status': 'success',
                'config': config.to_dict(),
                'timestamp': datetime.utcnow().isoformat()
            }, 200
        
        except Exception as e:
            logger.error(f"Error getting config: {str(e)}")
            return {'status': 'error', 'message': str(e)}, 500


__all__ = ['APIRoutes']
