"""
Configuration settings for the Image Forensics Application
"""
import os
class Config:
    """Base configuration class"""
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
    UPLOAD_FOLDER = 'static/uploads'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'tif'}
    MODEL_DIR = 'models'
    CNN_MODEL_PATH = os.path.join(MODEL_DIR, 'cnn_model.pth')
    AUTOENCODER_MODEL_PATH = os.path.join(MODEL_DIR, 'autoencoder.pth')
    IMAGE_SIZE = (224, 224)
    BATCH_SIZE = 32
    COPY_MOVE_BLOCK_SIZE = 16
    COPY_MOVE_THRESHOLD = 0.8
    COPY_MOVE_MAX_BLOCKS = 1000
    MAX_WORKERS = 4
    CACHE_TIMEOUT = 3600
    FAST_MODE_ENABLED = True
    GPU_ACCELERATION = True
    PARALLEL_PROCESSING = True
    PERFORMANCE_LOGGING = True
    LOG_LEVEL = 'INFO'
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    LOG_FILE = 'app.log'
class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    LOG_LEVEL = 'DEBUG'
class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    SECRET_KEY = os.environ.get('SECRET_KEY')
class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    UPLOAD_FOLDER = 'static/test_uploads'
    LOG_LEVEL = 'DEBUG'
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}
def get_config(config_name=None):
    """Get configuration class based on environment"""
    if config_name is None:
        config_name = os.environ.get('FLASK_ENV', 'development')
    return config.get(config_name, config['default'])