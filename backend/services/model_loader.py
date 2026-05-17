"""
Model Loader Service
Singleton pattern to load models once and reuse across requests
"""

import os
import logging
import threading

logger = logging.getLogger(__name__)

# Thread-safe singleton lock
_lock = threading.Lock()

# Module-level model cache
_image_model = None
_video_model = None
_audio_model = None
_text_model = None

MODEL_FOLDER = os.path.join(os.path.dirname(__file__), '..', 'models')


def _get_model_path(filename):
    """Return absolute path to model file, or None if not found"""
    path = os.path.join(MODEL_FOLDER, filename)
    if os.path.isfile(path):
        return path
    logger.warning(f"Model file not found: {path}")
    return None


def get_image_model():
    """Get or initialize the image deepfake model (singleton)"""
    global _image_model
    if _image_model is None:
        with _lock:
            if _image_model is None:
                from models.image_model import load_image_model
                path = _get_model_path('model_epoch_1.pth')
                _image_model = load_image_model(path)
                logger.info("Image model initialized")
    return _image_model


def get_video_model():
    """Get or initialize the video deepfake model (singleton)"""
    global _video_model
    if _video_model is None:
        with _lock:
            if _video_model is None:
                from models.video_model import load_video_model
                path = _get_model_path('model_epoch_3.pth')
                _video_model = load_video_model(path)
                logger.info("Video model initialized")
    return _video_model


def get_audio_model():
    """Get or initialize the audio deepfake model (singleton)"""
    global _audio_model
    if _audio_model is None:
        with _lock:
            if _audio_model is None:
                from models.audio_model import load_audio_model
                path = _get_model_path('audio_model.pth')
                _audio_model = load_audio_model(path)
                logger.info("Audio model initialized")
    return _audio_model


def get_text_model():
    """Get or initialize the text detection model (singleton)"""
    global _text_model
    if _text_model is None:
        with _lock:
            if _text_model is None:
                from models.text_model import load_text_model
                path = _get_model_path('ai_human_model.pkl')
                _text_model = load_text_model(path)
                logger.info("Text model initialized")
    return _text_model
