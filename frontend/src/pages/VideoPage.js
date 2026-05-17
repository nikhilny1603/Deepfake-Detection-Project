import React, { useState } from 'react';
import FileUploader from '../components/FileUploader';
import ResultDisplay from '../components/ResultDisplay';
import Loader from '../components/Loader';
import { detectVideo } from '../services/api';
import './DetectionPage.css';

const ACCEPT = { 'video/*': ['.mp4', '.avi', '.mov', '.mkv', '.webm'] };

export default function VideoPage() {
  const [file, setFile] = useState(null);
  const [maxFrames, setMaxFrames] = useState(32);
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
      const data = await detectVideo(file, maxFrames);
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
        <h1>🎬 Video Deepfake Detection</h1>
        <p>Upload a video to analyze frames for deepfake manipulation</p>
      </div>

      <div className="card">
        <FileUploader
          onFileSelect={handleFileSelect}
          accept={ACCEPT}
          label="Drag & drop a video, or click to browse"
          hint="Supported: MP4, AVI, MOV, MKV, WebM (max 100MB)"
          icon="🎬"
          selectedFile={file}
        />

        <div className="settings-row">
          <label className="settings-label">
            Frames to analyze:
            <select
              value={maxFrames}
              onChange={e => setMaxFrames(Number(e.target.value))}
              className="settings-select"
            >
              <option value={16}>16 (Fast)</option>
              <option value={32}>32 (Balanced)</option>
              <option value={48}>48 (Thorough)</option>
              <option value={64}>64 (Maximum)</option>
            </select>
          </label>
        </div>

        {error && <div className="error-banner">{error}</div>}

        <div className="action-row">
          <button
            className="btn btn-primary"
            onClick={handleSubmit}
            disabled={!file || loading}
          >
            {loading ? 'Analyzing frames...' : '🔍 Detect Deepfake'}
          </button>
          {(file || result) && (
            <button className="btn btn-secondary" onClick={handleReset}>↩ Reset</button>
          )}
        </div>
      </div>

      {loading && <Loader message={`Extracting and analyzing ${maxFrames} frames...`} />}
      {result && <ResultDisplay result={result} modality="video" />}
    </div>
  );
}
