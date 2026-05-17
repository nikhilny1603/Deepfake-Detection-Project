"""
Video Detection Service
Handles video frame extraction, per-frame inference, and temporal aggregation
"""

import os
import logging
from services.model_loader import get_video_model
from models.video_model import (
    extract_frames,
    preprocess_frame,
    aggregate_frame_predictions
)

logger = logging.getLogger(__name__)


def predict_video(video_path, max_frames=32):
    """
    Run deepfake detection on a video file.

    Args:
        video_path: Path to uploaded video
        max_frames: Maximum number of frames to sample

    Returns:
        dict with prediction, confidence, and per-frame details
    """
    try:
        model = get_video_model()
        device = next(model.parameters()).device

        # Extract frames
        frames, frame_indices, fps = extract_frames(video_path, max_frames=max_frames)

        # Per-frame inference
        frame_predictions = []
        for frame in frames:
            tensor = preprocess_frame(frame).to(device)
            probs = model.predict_frame(tensor)
            real_p = float(probs[0][0])
            fake_p = float(probs[0][1])
            frame_predictions.append((real_p, fake_p))

        # Aggregate results
        result = aggregate_frame_predictions(frame_predictions)

        # Video metadata
        file_size = os.path.getsize(video_path)

        return {
            'prediction': result['label'],
            'confidence': result['confidence'],
            'details': {
                'real_probability': result['real_probability'],
                'fake_probability': result['fake_probability'],
                'std_deviation': result['std_deviation'],
                'frames_analyzed': len(frames),
                'total_fps': round(fps, 2),
                'frame_indices': frame_indices[:len(result['frame_scores'])],
                'frame_scores': result['frame_scores'],
                'file_size_bytes': file_size,
                'model': 'MobileNetV3-Small + Temporal Averaging'
            }
        }

    except Exception as e:
        logger.error(f"Video prediction error: {e}")
        raise
