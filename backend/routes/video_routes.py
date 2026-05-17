"""
Video Detection Routes
POST /predict/video
"""

import os
import uuid
import logging
from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
from services.video_service import predict_video

logger = logging.getLogger(__name__)
video_bp = Blueprint('video', __name__)

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm', 'flv', 'm4v'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@video_bp.route('/video', methods=['POST'])
def predict_video_endpoint():
    """
    POST /predict/video
    Accept: multipart/form-data with field 'file'
    Optional form field: max_frames (int, default 32)
    Returns: JSON with prediction, confidence, per-frame scores
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided. Use multipart/form-data with field "file".'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': f'Unsupported file type. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'}), 400

    max_frames = request.form.get('max_frames', 32, type=int)
    max_frames = max(8, min(max_frames, 64))  # Clamp between 8 and 64

    filename = f"{uuid.uuid4()}_{secure_filename(file.filename)}"
    filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)

    try:
        file.save(filepath)
        logger.info(f"Video uploaded: {filename}, max_frames={max_frames}")

        result = predict_video(filepath, max_frames=max_frames)
        logger.info(f"Video prediction complete: {result['prediction']} ({result['confidence']:.2%})")

        return jsonify(result), 200

    except ValueError as e:
        return jsonify({'error': str(e)}), 422

    except Exception as e:
        logger.error(f"Unexpected error in video prediction: {e}", exc_info=True)
        return jsonify({'error': 'Prediction failed. Please try again.'}), 500

    finally:
        if os.path.exists(filepath):
            os.remove(filepath)
