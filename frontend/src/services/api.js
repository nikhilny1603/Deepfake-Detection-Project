/**
 * API service layer
 * All HTTP calls to the Flask backend
 */

import axios from 'axios';

const API_BASE = "https://deepfake-backend.onrender.com";

const api = axios.create({
  baseURL: API_BASE,
  timeout: 120000, // 2 min for heavy video/audio
});

/**
 * Detect deepfake in an image
 * @param {File} file
 * @returns {Promise<{prediction, confidence, details}>}
 */
export async function detectImage(file) {
  const formData = new FormData();
  formData.append('file', file);

  const response = await api.post('/predict/image', formData, {
    headers: { 'Content-Type': 'multipart/form-data' }
  });
  return response.data;
}

/**
 * Detect deepfake in a video
 * @param {File} file
 * @param {number} maxFrames
 * @returns {Promise<{prediction, confidence, details}>}
 */
export async function detectVideo(file, maxFrames = 32) {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('max_frames', String(maxFrames));

  const response = await api.post('/predict/video', formData, {
    headers: { 'Content-Type': 'multipart/form-data' }
  });
  return response.data;
}

/**
 * Detect deepfake in an audio file
 * @param {File} file
 * @returns {Promise<{prediction, confidence, details}>}
 */
export async function detectAudio(file) {
  const formData = new FormData();
  formData.append('file', file);

  const response = await api.post('/predict/audio', formData, {
    headers: { 'Content-Type': 'multipart/form-data' }
  });
  return response.data;
}

/**
 * Detect AI-generated text
 * @param {string} text
 * @returns {Promise<{prediction, confidence, details}>}
 */
export async function detectText(text) {
  const response = await api.post('/predict/text', { text }, {
    headers: { 'Content-Type': 'application/json' }
  });
  return response.data;
}

/**
 * Health check
 */
export async function healthCheck() {
  const response = await api.get('/health');
  return response.data;
}
