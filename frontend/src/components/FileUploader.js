import React, { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import './FileUploader.css';

export default function FileUploader({
  onFileSelect,
  accept,
  label,
  hint,
  icon = '📁',
  selectedFile = null
}) {
  const onDrop = useCallback((acceptedFiles) => {
    if (acceptedFiles.length > 0) {
      onFileSelect(acceptedFiles[0]);
    }
  }, [onFileSelect]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept,
    maxFiles: 1,
    multiple: false
  });

  return (
    <div className="uploader-wrapper">
      <div
        {...getRootProps()}
        className={`dropzone ${isDragActive ? 'drag-active' : ''} ${selectedFile ? 'has-file' : ''}`}
      >
        <input {...getInputProps()} />
        <div className="dropzone-icon">{selectedFile ? '✅' : icon}</div>
        {selectedFile ? (
          <div className="dropzone-file-info">
            <p className="file-name">{selectedFile.name}</p>
            <p className="file-size">{formatFileSize(selectedFile.size)}</p>
          </div>
        ) : (
          <div className="dropzone-text">
            <p className="dropzone-label">
              {isDragActive ? 'Drop it here!' : label}
            </p>
            {hint && <p className="dropzone-hint">{hint}</p>}
          </div>
        )}
      </div>
    </div>
  );
}

function formatFileSize(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}
