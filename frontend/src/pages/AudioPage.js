import React, { useState } from 'react';
import FileUploader from '../components/FileUploader';
import ResultDisplay from '../components/ResultDisplay';
import Loader from '../components/Loader';
import { detectAudio } from '../services/api';
import './DetectionPage.css';

const ACCEPT = { 'audio/*': ['.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac'] };

export default function AudioPage() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  function handleFileSelect(selected) {
    setFile(selected);
    setResult(null);
    setError(null);
  }

  async function handleSubmit() {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const data = await detectAudio(file);
      setResult(data);
    } catch (err) {
      setError(err.response?.data?.error || err.message || 'Detection failed');
    } finally {
      setLoading(false);
    }
  }

  function handleReset() {
    setFile(null);
    setResult(null);
    setError(null);
  }

  return (
    <div className="page-wrapper">
      <div className="page-header">
        <h1>🎵 Audio Deepfake Detection</h1>
        <p>Detect AI voice cloning or synthetic speech via Mel spectrogram analysis</p>
      </div>

      <div className="card">
        <FileUploader
          onFileSelect={handleFileSelect}
          accept={ACCEPT}
          label="Drag & drop an audio file, or click to browse"
          hint="Supported: WAV, MP3, FLAC, OGG, M4A, AAC"
          icon="🎵"
          selectedFile={file}
        />

        {file && !result && !loading && (
          <div className="audio-info-box">
            <span>🎵 {file.name}</span>
            <span>({(file.size / (1024 * 1024)).toFixed(2)} MB)</span>
          </div>
        )}

        {error && <div className="error-banner">{error}</div>}

        <div className="action-row">
          <button
            className="btn btn-primary"
            onClick={handleSubmit}
            disabled={!file || loading}
          >
            {loading ? 'Analyzing audio...' : '🔍 Detect Deepfake'}
          </button>
          {(file || result) && (
            <button className="btn btn-secondary" onClick={handleReset}>↩ Reset</button>
          )}
        </div>
      </div>

      {loading && <Loader message="Converting audio to spectrogram and running CNN..." />}
      {result && <ResultDisplay result={result} modality="audio" />}
    </div>
  );
}
