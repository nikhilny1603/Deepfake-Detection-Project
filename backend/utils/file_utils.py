"""
File Utility Functions
"""

import os
import hashlib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Allowed file types by modality
ALLOWED_IMAGE_TYPES = {'png', 'jpg', 'jpeg', 'webp', 'bmp', 'tiff'}
ALLOWED_VIDEO_TYPES = {'mp4', 'avi', 'mov', 'mkv', 'webm', 'flv', 'm4v'}
ALLOWED_AUDIO_TYPES = {'wav', 'mp3', 'flac', 'ogg', 'm4a', 'aac', 'opus'}


def get_file_extension(filename):
    """Return lowercase extension without dot"""
    return Path(filename).suffix.lstrip('.').lower()


def get_file_hash(filepath, algorithm='sha256'):
    """Compute hash of file for integrity checking"""
    h = hashlib.new(algorithm)
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def get_file_size_mb(filepath):
    """Return file size in megabytes"""
    return os.path.getsize(filepath) / (1024 * 1024)


def cleanup_file(filepath):
    """Safely delete a file"""
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
            logger.debug(f"Cleaned up: {filepath}")
    except Exception as e:
        logger.warning(f"Could not clean up {filepath}: {e}")
