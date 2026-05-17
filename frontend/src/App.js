import React, { useState } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import Navbar from './components/Navbar';
import HomePage from './pages/HomePage';
import ImagePage from './pages/ImagePage';
import VideoPage from './pages/VideoPage';
import AudioPage from './pages/AudioPage';
import TextPage from './pages/TextPage';
import './App.css';

function App() {
  return (
    <Router>
      <div className="app">
        <Navbar />
        <main className="main-content">
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/image" element={<ImagePage />} />
            <Route path="/video" element={<VideoPage />} />
            <Route path="/audio" element={<AudioPage />} />
            <Route path="/text" element={<TextPage />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;
