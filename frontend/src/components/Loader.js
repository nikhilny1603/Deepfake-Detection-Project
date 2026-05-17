import React from 'react';
import './Loader.css';

export default function Loader({ message = 'Analyzing...' }) {
  return (
    <div className="loader-wrapper">
      <div className="loader-spinner">
        <div className="spinner-ring" />
        <div className="spinner-ring" />
        <div className="spinner-ring" />
      </div>
      <p className="loader-message">{message}</p>
    </div>
  );
}
