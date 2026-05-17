"""
Audio Deepfake Detection Model
MobileNetV2-based (matches trained model)
Flask-ready version
"""

import torch
import torch.nn as nn
import numpy as np
import librosa
import logging
from torchvision import models
import cv2


logger = logging.getLogger(__name__)

# =========================
# CONSTANTS (MATCH TRAINING)
# =========================

SAMPLE_RATE = 16000
DURATION = 7
N_MELS = 128
IMG_SIZE = 224  # MobileNet expects 224x224


# =========================
# LOAD MODEL (.pth)
# =========================

def load_audio_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1️⃣ SAME architecture as training
    model = models.mobilenet_v2(pretrained=False)

    # 2️⃣ Replace classifier
    model.classifier[1] = nn.Linear(model.last_channel, 2)

    # 3️⃣ Load weights
    state_dict = torch.load(model_path, map_location=device)

    # Fix DataParallel issue
    if list(state_dict.keys())[0].startswith("module."):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    logger.info("✅ Audio model (MobileNetV2) loaded successfully")

    return model


# =========================
# PREPROCESS AUDIO (UPLOAD)
# =========================

def audio_to_melspectrogram(file):
    """
    Convert uploaded audio file → Mel spectrogram → tensor
    """
    try:
        # Load audio
        y, sr = librosa.load(file, sr=SAMPLE_RATE, duration=DURATION)

        # Pad/trim
        target_length = SAMPLE_RATE * DURATION
        if len(y) < target_length:
            y = np.pad(y, (0, target_length - len(y)))
        else:
            y = y[:target_length]

        # Mel spectrogram
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
        mel = librosa.power_to_db(mel)

        # Normalize (same as training)
        mel -= mel.min()
        mel /= mel.max()

        # Convert to 3 channels
        mel = np.stack([mel, mel, mel])

        # Resize to 224x224
        mel = np.transpose(mel, (1, 2, 0))  # (H,W,C)
        mel = cv2.resize(mel, (IMG_SIZE, IMG_SIZE))
        mel = np.transpose(mel, (2, 0, 1))  # (C,H,W)

        # Convert to tensor
        tensor = torch.tensor(mel).float().unsqueeze(0)

        return tensor

    except Exception as e:
        logger.error(f"Audio preprocessing error: {e}")
        raise ValueError(f"Could not process audio file: {e}")



def get_audio_features(audio_path, sr=16000):
    """
    Extract simple audio features for UI display
    """
    try:
        y, sr = librosa.load(audio_path, sr=sr, duration=3)

        duration = librosa.get_duration(y=y, sr=sr)

        rms = float(np.mean(librosa.feature.rms(y=y)))
        zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))
        spec_centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))

        return {
            'duration_sec': round(duration, 2),
            'sample_rate': sr,
            'rms_energy': round(rms, 6),
            'zero_crossing_rate': round(zcr, 4),
            'spectral_centroid_hz': round(spec_centroid, 2)
        }

    except Exception as e:
        return {"error": str(e)}
# =========================
# PREDICTION FUNCTION
# =========================

def predict_audio(model, file):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Preprocess
    spec = audio_to_melspectrogram(file)
    spec = spec.to(device)

    # Predict
    with torch.no_grad():
        outputs = model(spec)
        probs = torch.softmax(outputs, dim=1)

    real = probs[0][0].item()
    fake = probs[0][1].item()

    return {
        "prediction": "Fake" if fake > real else "Real",
        "confidence": float(max(real, fake)),
        "real_score": float(real),
        "fake_score": float(fake)
    }