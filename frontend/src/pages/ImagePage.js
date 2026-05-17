import React, { useState } from 'react';
import FileUploader from '../components/FileUploader';
import ResultDisplay from '../components/ResultDisplay';
import Loader from '../components/Loader';
import { detectImage } from '../services/api';
import './DetectionPage.css';

const ACCEPT = { 'image/*': ['.png', '.jpg', '.jpeg', '.webp', '.bmp'] };

export default function ImagePage() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  function handleFileSelect(selected) {
    setFile(selected);
    setResult(null);
    setError(null);
    setPreview(URL.createObjectURL(selected));
  }

  async function handleSubmit() {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const data = await detectImage(file);
      setResult(data);
    } catch (err) {
      setError(err.response?.data?.error || err.message || 'Detection failed');
    } finally {
      setLoading(false);
    }
  }

  function handleReset() {
    setFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
  }

  return (
    <div className="page-wrapper">
      <div className="page-header">
        <h1>🖼️ Image Deepfake Detection</h1>
        <p>Upload a photo to detect face-swap or AI-generated manipulation</p>
      </div>

      <div className="card">
        <FileUploader
          onFileSelect={handleFileSelect}
          accept={ACCEPT}
          label="Drag & drop an image, or click to browse"
          hint="Supported: PNG, JPG, JPEG, WebP, BMP"
          icon="🖼️"
          selectedFile={file}
        />

        {preview && !result && (
          <div className="preview-wrapper">
            <img src={preview} alt="Preview" className="image-preview" />
          </div>
        )}

        {error && <div className="error-banner">{error}</div>}

        <div className="action-row">
          <button
            className="btn btn-primary"
            onClick={handleSubmit}
            disabled={!file || loading}
          >
            {loading ? 'Analyzing...' : '🔍 Detect Deepfake'}
          </button>
          {(file || result) && (
            <button className="btn btn-secondary" onClick={handleReset}>
              ↩ Reset
            </button>
          )}
        </div>
      </div>

      {loading && <Loader message="Running CNN analysis on your image..." />}
      {result && <ResultDisplay result={result} modality="image" />}
    </div>
  );
}
