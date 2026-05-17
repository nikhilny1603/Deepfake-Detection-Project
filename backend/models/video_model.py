"""
Video Deepfake Detection Model
ResNet18 + LSTM + FC (matches your trained model architecture)
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
import numpy as np
import logging
from PIL import Image

logger = logging.getLogger(__name__)

VIDEO_FRAME_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


class DeepfakeVideoClassifier(nn.Module):
    """
    ResNet18 backbone + LSTM + FC classifier.
    Processes a sequence of frames through LSTM for temporal modelling.
    """

    def __init__(self, hidden_size=256, num_classes=2):
        super(DeepfakeVideoClassifier, self).__init__()

        self.cnn = models.resnet18(weights=None)
        self.cnn.fc = nn.Identity()

        self.lstm = nn.LSTM(512, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward_frames(self, frames_tensor):
        """
        frames_tensor: (N, C, H, W) — N frames
        Returns: (num_classes,) logits
        """
        # Extract CNN features for all frames: (N, 512)
        features = self.cnn(frames_tensor)

        # Add batch dimension: (1, N, 512)
        features = features.unsqueeze(0)

        # LSTM over frame sequence
        lstm_out, _ = self.lstm(features)

        # Take last frame output: (1, hidden_size)
        out = lstm_out[:, -1, :]

        return self.fc(out)

    def predict_frame(self, frame_tensor):
        """Single-frame prediction (for per-frame scores)"""
        self.eval()
        with torch.no_grad():
            features = self.cnn(frame_tensor)
            features = features.unsqueeze(1)
            lstm_out, _ = self.lstm(features)
            out = lstm_out[:, -1, :]
            logits = self.fc(out)
            return torch.softmax(logits, dim=1)


def load_video_model(model_path=None):
    """Load video deepfake detection model from .pth file"""
    if model_path is None:
        logger.warning("No video model path — running in demo mode")
        model = DeepfakeVideoClassifier()
        model.eval()
        return model

    state_dict = torch.load(model_path, map_location='cpu')

    hidden_size = state_dict['fc.weight'].shape[1]
    num_classes  = state_dict['fc.weight'].shape[0]
    logger.info(f"Detected: hidden_size={hidden_size}, num_classes={num_classes}")

    model = DeepfakeVideoClassifier(hidden_size=hidden_size, num_classes=num_classes)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    logger.info(f"Video model loaded from {model_path}")
    return model


def extract_frames(video_path, max_frames=32, sample_interval=None):
    """Extract frames from a video file"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if sample_interval is None:
        sample_interval = max(1, total_frames // max_frames)

    frames, extracted_indices = [], []
    frame_idx = 0

    while cap.isOpened() and len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % sample_interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
            extracted_indices.append(frame_idx)
        frame_idx += 1

    cap.release()

    if not frames:
        raise ValueError("No frames could be extracted from video")

    logger.info(f"Extracted {len(frames)} frames from {total_frames} total")
    return frames, extracted_indices, fps


def preprocess_frame(pil_image):
    """Preprocess a PIL image frame"""
    return VIDEO_FRAME_TRANSFORMS(pil_image).unsqueeze(0)


def aggregate_frame_predictions(predictions):
    """Aggregate per-frame predictions into a final score"""
    if not predictions:
        return {'label': 'Unknown', 'confidence': 0.0, 'frame_scores': []}

    fake_probs = [p[1] for p in predictions]
    real_probs = [p[0] for p in predictions]

    avg_fake = float(np.mean(fake_probs))
    avg_real = float(np.mean(real_probs))
    std_fake  = float(np.std(fake_probs))

    label = 'Fake' if avg_fake > 0.5 else 'Real'
    confidence = max(avg_fake, avg_real)

    return {
        'label': label,
        'confidence': round(confidence, 4),
        'fake_probability': round(avg_fake, 4),
        'real_probability': round(avg_real, 4),
        'std_deviation': round(std_fake, 4),
        'frame_scores': [round(p, 4) for p in fake_probs]
    }