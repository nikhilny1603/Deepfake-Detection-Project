"""
Audio Detection Routes
POST /predict/audio
"""

import os
import uuid
import logging
from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
from services.audio_service import predict_audio

logger = logging.getLogger(__name__)
audio_bp = Blueprint('audio', __name__)

ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'ogg', 'm4a', 'aac', 'opus'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@audio_bp.route('/audio', methods=['POST'])
def predict_audio_endpoint():

    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Unsupported file type'}), 400

    filename = f"{uuid.uuid4()}_{secure_filename(file.filename)}"
    filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)

    try:
        file.save(filepath)

        # 🔥 This now uses preloaded model
        result = predict_audio(filepath)

        return jsonify(result), 200

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return jsonify({'error': 'Prediction failed'}), 500

    finally:
        if os.path.exists(filepath):
            os.remove(filepath)