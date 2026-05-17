"""
Audio Detection Service
MobileNetV2-based inference (correct version)
"""

import os
import logging
import torch
from services.model_loader import get_audio_model
from models.audio_model import audio_to_melspectrogram, get_audio_features

logger = logging.getLogger(__name__)


def predict_audio(audio_path):
    """
    Run deepfake detection on an audio file.
    """

    try:
        # Load model (already cached)
        model = get_audio_model()
        device = next(model.parameters()).device

        # Preprocess
        spectrogram_tensor = audio_to_melspectrogram(audio_path)
        spectrogram_tensor = spectrogram_tensor.to(device)

        # Inference (MobileNet)
        with torch.no_grad():
            outputs = model(spectrogram_tensor)
            probs = torch.softmax(outputs, dim=1)

        real_prob = float(probs[0][0])
        fake_prob = float(probs[0][1])

        label = 'Fake' if fake_prob > real_prob else 'Real'
        confidence = max(real_prob, fake_prob)

        # Extra features
        audio_features = get_audio_features(audio_path)
        file_size = os.path.getsize(audio_path)

        return {
            'prediction': label,
            'confidence': round(confidence, 4),
            'details': {
                'real_probability': round(real_prob, 4),
                'fake_probability': round(fake_prob, 4),
                'file_size_bytes': file_size,
                'audio_features': audio_features,

                # ✅ FIXED METADATA
                'model': 'MobileNetV2',
                'input_type': 'Mel Spectrogram (3-channel)',
                'sample_rate': 16000,
                'duration_sec': 3
            }
        }

    except Exception as e:
        logger.error(f"Audio prediction error: {e}", exc_info=True)
        raise