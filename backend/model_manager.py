"""
Model Manager Module
Handles model loading, caching, and inference
"""

import os
import json
import logging
import time
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from threading import Lock

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Manages model loading, caching, and inference
    Supports lazy loading and thread-safe operations
    """
    
    def __init__(self, model_paths: Dict[str, str], class_indices_path: str):
        """
        Initialize ModelManager
        
        Args:
            model_paths: Dictionary mapping model names to paths
            class_indices_path: Path to class indices JSON file
        """
        self.model_paths = model_paths
        self.class_indices_path = class_indices_path
        self.loaded_models = {}
        self.class_indices = None
        self.load_lock = Lock()
        self.inference_lock = Lock()
        self.model_load_times = {}
        self.inference_times = {}
        
        self._load_class_indices()
    
    def _load_class_indices(self):
        """Load class indices from JSON file"""
        try:
            if os.path.exists(self.class_indices_path):
                with open(self.class_indices_path, 'r') as f:
                    self.class_indices = json.load(f)
                logger.info(f"✓ Loaded {len(self.class_indices)} classes from {self.class_indices_path}")
            else:
                logger.warning(f"Class indices file not found: {self.class_indices_path}")
                # Use default classes
                self.class_indices = {
                    'type-1': 0,
                    'type-2': 1,
                    'type-3': 2
                }
                logger.info("Using default class indices")
        except Exception as e:
            logger.error(f"Error loading class indices: {str(e)}")
            self.class_indices = {'type-1': 0, 'type-2': 1, 'type-3': 2}
    
    def load_model(self, model_name: str, force_reload: bool = False):
        """
        Load model with lazy loading and caching
        
        Args:
            model_name: Name of model to load
            force_reload: Force reload even if cached
        """
        if model_name in self.loaded_models and not force_reload:
            logger.info(f"Using cached model: {model_name}")
            return True
        
        if model_name not in self.model_paths:
            logger.error(f"Model not found: {model_name}")
            return False
        
        model_path = self.model_paths[model_name]
        
        if not os.path.exists(model_path):
            logger.warning(f"Model file not found: {model_path}")
            return False
        
        with self.load_lock:
            try:
                logger.info(f"Loading model: {model_name} from {model_path}")
                start_time = time.time()
                
                # Lazy import TensorFlow
                import tensorflow as tf
                model = tf.keras.models.load_model(model_path)
                
                load_time = time.time() - start_time
                self.loaded_models[model_name] = model
                self.model_load_times[model_name] = load_time
                
                logger.info(f"✓ Model '{model_name}' loaded successfully in {load_time:.2f}s")
                return True
            
            except Exception as e:
                logger.error(f"Error loading model '{model_name}': {str(e)}")
                return False
    
    def get_model(self, model_name: str):
        """
        Get model, loading if necessary
        
        Args:
            model_name: Name of model
            
        Returns:
            Loaded model or None
        """
        if model_name not in self.loaded_models:
            self.load_model(model_name)
        
        return self.loaded_models.get(model_name)
    
    def predict(self, model_name: str, image_array: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Perform prediction with specified model
        
        Args:
            model_name: Name of model
            image_array: Image as numpy array
            
        Returns:
            Prediction result dictionary or None
        """
        model = self.get_model(model_name)
        if model is None:
            logger.error(f"Model not available: {model_name}")
            return None
        
        with self.inference_lock:
            try:
                start_time = time.time()
                
                # Add batch dimension if needed
                if len(image_array.shape) == 3:
                    image_array = np.expand_dims(image_array, axis=0)
                
                # Perform prediction
                predictions = model.predict(image_array, verbose=0)
                
                inference_time = time.time() - start_time
                if model_name not in self.inference_times:
                    self.inference_times[model_name] = []
                self.inference_times[model_name].append(inference_time)
                
                # Get class indices
                class_list = sorted(self.class_indices.items(), key=lambda x: x[1])
                class_names = [name for name, _ in class_list]
                
                # Get predictions for each class
                pred_dict = {}
                for idx, class_name in enumerate(class_names):
                    if idx < len(predictions[0]):
                        pred_dict[class_name] = float(predictions[0][idx])
                
                # Get top prediction
                top_class_idx = np.argmax(predictions[0])
                top_class_name = class_names[top_class_idx] if top_class_idx < len(class_names) else 'Unknown'
                top_confidence = float(predictions[0][top_class_idx])
                
                return {
                    'model': model_name,
                    'predictions': pred_dict,
                    'top_class': top_class_name,
                    'confidence': top_confidence,
                    'inference_time': inference_time,
                    'timestamp': time.time()
                }
            
            except Exception as e:
                logger.error(f"Error during inference with {model_name}: {str(e)}")
                return None
    
    def ensemble_predict(self, image_array: np.ndarray, 
                        models: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
        """
        Perform ensemble prediction using multiple models
        
        Args:
            image_array: Image as numpy array
            models: List of model names to use (uses all if None)
            
        Returns:
            Ensemble prediction result
        """
        if models is None:
            models = list(self.model_paths.keys())
        
        results = []
        predictions_by_class = {}
        
        for model_name in models:
            result = self.predict(model_name, image_array)
            if result:
                results.append(result)
                
                # Aggregate predictions
                for class_name, confidence in result['predictions'].items():
                    if class_name not in predictions_by_class:
                        predictions_by_class[class_name] = []
                    predictions_by_class[class_name].append(confidence)
        
        if not results:
            logger.error("No models produced predictions for ensemble")
            return None
        
        # Calculate average confidence for each class
        ensemble_predictions = {}
        for class_name, confidences in predictions_by_class.items():
            ensemble_predictions[class_name] = float(np.mean(confidences))
        
        # Get top class
        top_class = max(ensemble_predictions.items(), key=lambda x: x[1])
        
        return {
            'model': 'ensemble',
            'models_used': len(results),
            'predictions': ensemble_predictions,
            'top_class': top_class[0],
            'confidence': top_class[1],
            'individual_results': results,
            'timestamp': time.time()
        }
    
    def get_available_models(self) -> List[str]:
        """Get list of available models"""
        return list(self.model_paths.keys())
    
    def get_loaded_models(self) -> List[str]:
        """Get list of currently loaded models"""
        return list(self.loaded_models.keys())
    
    def unload_model(self, model_name: str):
        """
        Unload model from memory
        
        Args:
            model_name: Name of model to unload
        """
        if model_name in self.loaded_models:
            del self.loaded_models[model_name]
            logger.info(f"Model unloaded: {model_name}")
    
    def unload_all_models(self):
        """Unload all loaded models"""
        with self.load_lock:
            model_count = len(self.loaded_models)
            self.loaded_models.clear()
            logger.info(f"Unloaded {model_count} models")
    
    def get_model_stats(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get statistics for a model
        
        Args:
            model_name: Name of model
            
        Returns:
            Statistics dictionary
        """
        if model_name not in self.loaded_models:
            return None
        
        load_time = self.model_load_times.get(model_name, 0)
        inference_times = self.inference_times.get(model_name, [])
        
        return {
            'model_name': model_name,
            'loaded': True,
            'load_time': load_time,
            'inference_count': len(inference_times),
            'avg_inference_time': float(np.mean(inference_times)) if inference_times else 0,
            'min_inference_time': float(np.min(inference_times)) if inference_times else 0,
            'max_inference_time': float(np.max(inference_times)) if inference_times else 0
        }
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all models"""
        stats = {}
        for model_name in self.get_available_models():
            model_stats = self.get_model_stats(model_name)
            if model_stats:
                stats[model_name] = model_stats
        return stats
