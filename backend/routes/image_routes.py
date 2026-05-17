"""
Image Detection Routes
POST /predict/image
"""

import os
import uuid
import logging
from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
from services.image_service import predict_image

logger = logging.getLogger(__name__)
image_bp = Blueprint('image', __name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'bmp', 'tiff'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@image_bp.route('/image', methods=['POST'])
def predict_image_endpoint():
    """
    POST /predict/image
    Accept: multipart/form-data with field 'file'
    Returns: JSON with prediction, confidence, details
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided. Use multipart/form-data with field "file".'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': f'Unsupported file type. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'}), 400

    # Save to temp location
    filename = f"{uuid.uuid4()}_{secure_filename(file.filename)}"
    filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)

    try:
        file.save(filepath)
        logger.info(f"Image uploaded: {filename}")

        # Run detection
        result = predict_image(filepath)
        logger.info(f"Image prediction complete: {result['prediction']} ({result['confidence']:.2%})")

        return jsonify(result), 200

    except ValueError as e:
        return jsonify({'error': str(e)}), 422

    except Exception as e:
        logger.error(f"Unexpected error in image prediction: {e}", exc_info=True)
        return jsonify({'error': 'Prediction failed. Please try again.'}), 500

    finally:
        # Clean up uploaded file
        if os.path.exists(filepath):
            os.remove(filepath)
