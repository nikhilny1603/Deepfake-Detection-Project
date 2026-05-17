"""
Text Detection Service
Handles AI-generated text detection using NLP pipeline
"""

import logging
from services.model_loader import get_text_model

logger = logging.getLogger(__name__)


def predict_text(text):
    """
    Detect whether text was AI-generated.

    Args:
        text: Raw text string

    Returns:
        dict with prediction, confidence, and linguistic details
    """
    try:
        model = get_text_model()

        # Run prediction
        result = model.predict(text)

        # Additional text stats
        word_count = len(text.split())
        char_count = len(text)
        sentence_count = len([s for s in text.split('.') if s.strip()])

        return {
            'prediction': result['label'],
            'confidence': result['confidence'],
            'details': {
                'real_probability': result['real_probability'],
                'fake_probability': result['fake_probability'],
                'word_count': word_count,
                'character_count': char_count,
                'sentence_count': sentence_count,
                'model': 'TF-IDF + MultinomialNB'
            }
        }

    except Exception as e:
        logger.error(f"Text prediction error: {e}")
        raise
