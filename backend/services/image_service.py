"""
Image Detection Service
Handles image preprocessing and inference
"""

import os
import logging
import torch
from services.model_loader import get_image_model
from models.image_model import preprocess_image

logger = logging.getLogger(__name__)


def predict_image(image_path):
    """
    Run deepfake detection on an image file.

    Args:
        image_path: Path to uploaded image

    Returns:
        dict with prediction, confidence, and details
    """
    try:
        model = get_image_model()
        tensor = preprocess_image(image_path)

        # Move tensor to model device
        device = next(model.parameters()).device
        tensor = tensor.to(device)

        # Inference
        probs = model.predict(tensor)
        real_prob = float(probs[0][0])
        fake_prob = float(probs[0][1])

        label = 'Fake' if fake_prob > 0.5 else 'Real'
        confidence = max(real_prob, fake_prob)

        # File metadata
        file_size = os.path.getsize(image_path)

        return {
            'prediction': label,
            'confidence': round(confidence, 4),
            'details': {
                'real_probability': round(real_prob, 4),
                'fake_probability': round(fake_prob, 4),
                'file_size_bytes': file_size,
                'model': 'ResNet50',
                'input_size': '224x224'
            }
        }

    except Exception as e:
        logger.error(f"Image prediction error: {e}")
        raise
