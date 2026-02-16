"""
Model caching utilities for improved performance
"""
import os
import logging
import hashlib
import time
from functools import lru_cache
import threading
from config import get_config
config_class = get_config()
logger = logging.getLogger(__name__)
class ModelCache:
    """Thread-safe model caching system"""
    def __init__(self):
        self._lock = threading.Lock()
        self._models = {}
        self._model_paths = {
            'cnn': config_class.CNN_MODEL_PATH,
            'autoencoder': config_class.AUTOENCODER_MODEL_PATH
        }
    def get_model(self, model_type):
        """Get cached model or load if not available"""
        with self._lock:
            if model_type not in self._models:
                self._load_model(model_type)
            return self._models.get(model_type)
    def _load_model(self, model_type):
        """Load model from disk with GPU acceleration if available"""
        try:
            import torch
            from models.cnn_model import UltraFastNet, FastCATNet
            from models.autoencoder import Autoencoder
            model_path = self._model_paths.get(model_type)
            if not model_path or not os.path.exists(model_path):
                logger.warning(f"Model {model_type} not found at {model_path}")
                self._models[model_type] = None
                return
            device = torch.device('cuda' if torch.cuda.is_available() and config_class.GPU_ACCELERATION else 'cpu')
            logger.info(f"Loading model on device: {device}")
            if model_type == 'cnn':
                try:
                    model = UltraFastNet()
                    logger.info("Using UltraFastNet for maximum speed")
                except Exception as e:
                    logger.warning(f"UltraFastNet failed, falling back to FastCATNet: {e}")
                    model = FastCATNet()
            elif model_type == 'autoencoder':
                model = Autoencoder()
            else:
                logger.error(f"Unknown model type: {model_type}")
                return
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()
            self._models[model_type] = model
            logger.info(f"Loaded {model_type} model from {model_path} on {device}")
        except Exception as e:
            logger.error(f"Failed to load {model_type} model: {e}")
            self._models[model_type] = None
    def clear_cache(self):
        """Clear all cached models"""
        with self._lock:
            self._models.clear()
            logger.info("Model cache cleared")
    def reload_model(self, model_type):
        """Force reload a specific model"""
        with self._lock:
            if model_type in self._models:
                del self._models[model_type]
            self._load_model(model_type)
model_cache = ModelCache()
@lru_cache(maxsize=100)
def cached_image_preprocessing(image_path):
    """
    Cached image preprocessing to avoid redundant processing
    """
    from utils.image_processing import preprocess_image
    return preprocess_image(image_path)
class ResultCache:
    """Cache for analysis results with TTL"""
    def __init__(self, ttl_seconds=300):
        self._cache = {}
        self._lock = threading.Lock()
        self.ttl = ttl_seconds
    def _get_cache_key(self, image_path, analysis_type, params=None):
        """Generate cache key from image path and analysis parameters"""
        key_data = f"{image_path}:{analysis_type}"
        if params:
            key_data += f":{str(sorted(params.items()))}"
        return hashlib.md5(key_data.encode()).hexdigest()
    def get(self, image_path, analysis_type, params=None):
        """Get cached result if available and not expired"""
        with self._lock:
            key = self._get_cache_key(image_path, analysis_type, params)
            if key in self._cache:
                result, timestamp = self._cache[key]
                if time.time() - timestamp < self.ttl:
                    logger.debug(f"Cache hit for {analysis_type} on {image_path}")
                    return result
                else:
                    del self._cache[key]
            return None
    def set(self, image_path, analysis_type, result, params=None):
        """Cache analysis result"""
        with self._lock:
            key = self._get_cache_key(image_path, analysis_type, params)
            self._cache[key] = (result, time.time())
            logger.debug(f"Cached result for {analysis_type} on {image_path}")
    def clear_expired(self):
        """Clear expired cache entries"""
        with self._lock:
            current_time = time.time()
            expired_keys = [
                key for key, (_, timestamp) in self._cache.items()
                if current_time - timestamp >= self.ttl
            ]
            for key in expired_keys:
                del self._cache[key]
            if expired_keys:
                logger.info(f"Cleared {len(expired_keys)} expired cache entries")
result_cache = ResultCache()
@lru_cache(maxsize=50)
def cached_forgery_detection(image_path, fast_mode=False):
    """Cached forgery detection results"""
    from utils.image_processing import detect_image_forgery, enhanced_forgery_detection
    cache_key = "fast_forgery" if fast_mode else "full_forgery"
    cached_result = result_cache.get(image_path, cache_key)
    if cached_result:
        return cached_result
    if fast_mode:
        ai_compression_detected = detect_ai_compression_artifacts(image_path)
        result = enhanced_forgery_detection(image_path, ai_compression_detected)
    else:
        result, _ = detect_image_forgery(image_path)
    result_cache.set(image_path, cache_key, result)
    return result
@lru_cache(maxsize=50)
def cached_copy_move_detection(image_path):
    """Cached copy-move detection results"""
    from utils.image_processing import detect_copy_move
    cached_result = result_cache.get(image_path, "copy_move")
    if cached_result:
        return cached_result
    mask, regions = detect_copy_move(image_path)
    result = (mask, regions)
    result_cache.set(image_path, "copy_move", result)
    return result
def get_cached_model(model_type):
    """Convenience function to get cached model"""
    return model_cache.get_model(model_type)
def clear_all_caches():
    """Clear all caches"""
    model_cache.clear_cache()
    result_cache.clear_expired()
    cached_image_preprocessing.cache_clear()
    cached_forgery_detection.cache_clear()
    cached_copy_move_detection.cache_clear()
    logger.info("All caches cleared")