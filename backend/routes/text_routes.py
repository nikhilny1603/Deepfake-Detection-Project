"""
Text Detection Routes
POST /predict/text
"""

import logging
from flask import Blueprint, request, jsonify
from services.text_service import predict_text

logger = logging.getLogger(__name__)
text_bp = Blueprint('text', __name__)

MAX_TEXT_LENGTH = 10000  # Characters


@text_bp.route('/text', methods=['POST'])
def predict_text_endpoint():
    """
    POST /predict/text
    Accept: JSON body { "text": "..." }
         OR multipart/form-data with field 'text'
    Returns: JSON with prediction, confidence, details
    """
    text = None

    # Support both JSON body and form data
    if request.is_json:
        data = request.get_json(silent=True)
        if data:
            text = data.get('text', '')
    else:
        text = request.form.get('text', '')

    if not text or not text.strip():
        return jsonify({'error': 'No text provided. Send JSON {"text": "..."} or form field "text".'}), 400

    if len(text) > MAX_TEXT_LENGTH:
        return jsonify({'error': f'Text too long. Maximum {MAX_TEXT_LENGTH} characters allowed.'}), 400

    if len(text.strip()) < 20:
        return jsonify({'error': 'Text too short. Please provide at least 20 characters.'}), 400

    try:
        logger.info(f"Text prediction request: {len(text)} chars")
        result = predict_text(text)
        logger.info(f"Text prediction complete: {result['prediction']} ({result['confidence']:.2%})")

        return jsonify(result), 200

    except Exception as e:
        logger.error(f"Unexpected error in text prediction: {e}", exc_info=True)
        return jsonify({'error': 'Prediction failed. Please try again.'}), 500
