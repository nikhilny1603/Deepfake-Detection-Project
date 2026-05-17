"""
Deepfake Detection System - Flask Backend
Main application entry point
"""

import os
import logging
from flask import Flask
from flask_cors import CORS

from routes.image_routes import image_bp
from routes.video_routes import video_bp
from routes.audio_routes import audio_bp
from routes.text_routes import text_bp

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_app():
    """Application factory pattern"""
    app = Flask(__name__)

    # Configuration
    app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100 MB max upload
    app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')
    app.config['MODEL_FOLDER'] = os.path.join(os.path.dirname(__file__), '..', 'models')

    # Create upload folder if it doesn't exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    # Enable CORS for all routes (allow React frontend)
    CORS(app, origins='*')

    # Register blueprints
    app.register_blueprint(image_bp, url_prefix='/predict')
    app.register_blueprint(video_bp, url_prefix='/predict')
    app.register_blueprint(audio_bp, url_prefix='/predict')
    app.register_blueprint(text_bp, url_prefix='/predict')

    @app.route('/health', methods=['GET'])
    def health_check():
        """Health check endpoint"""
        return {'status': 'healthy', 'message': 'Deepfake Detection API is running'}, 200

    @app.route('/status', methods=['GET'])
    def model_status():
        """Check which models are loaded"""
        from services.model_loader import _get_model_path

        models_info = {
            'image_model':  _get_model_path('../models/model_epoch_1.pth'),
            'video_model':  _get_model_path('../models/model_epoch_3.pth'),
            'audio_model':  _get_model_path('../models/audio_model.pth'),
            'text_model':   _get_model_path('../models/ai_human_model.pkl'),
        }

        status = {}
        for name, path in models_info.items():
            if path:
                status[name] = {'loaded': True, 'path': path}
            else:
                status[name] = {'loaded': False, 'path': 'NOT FOUND'}

        all_loaded = all(v['loaded'] for v in status.values())
        return {'all_models_ready': all_loaded, 'models': status}, 200

    @app.route('/debug-path', methods=['GET'])
    def debug_path():
        import os
        model_folder = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', '..', 'models')
        )
        files_found = os.listdir(model_folder) if os.path.exists(model_folder) else []
        return {
            'looking_in': model_folder,
            'folder_exists': os.path.exists(model_folder),
            'files_found': files_found
        }, 200

    @app.errorhandler(413)
    def file_too_large(e):
        return {'error': 'File too large. Maximum size is 100MB.'}, 413

    @app.errorhandler(500)
    def internal_error(e):
        logger.error(f"Internal server error: {str(e)}")
        return {'error': 'Internal server error'}, 500

    logger.info("Deepfake Detection API initialized successfully")
    return app


if __name__ == '__main__':
    app = create_app()
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    logger.info(f"Starting server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=debug)
