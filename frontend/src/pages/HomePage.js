import React from 'react';
import { Link } from 'react-router-dom';
import './HomePage.css';

const MODALITIES = [
  {
    to: '/image',
    icon: '🖼️',
    title: 'Image Detection',
    desc: 'Analyze photos to detect face-swap deepfakes using CNN (ResNet50).',
    tag: 'ResNet50',
    color: 'purple'
  },
  {
    to: '/video',
    icon: '🎬',
    title: 'Video Detection',
    desc: 'Frame-by-frame video analysis with temporal aggregation over up to 64 frames.',
    tag: 'MobileNetV3 + Temporal',
    color: 'blue'
  },
  {
    to: '/audio',
    icon: '🎵',
    title: 'Audio Detection',
    desc: 'Spectrogram CNN to catch AI-synthesized or cloned voices.',
    tag: 'SpectrogramCNN',
    color: 'green'
  },
  {
    to: '/text',
    icon: '📝',
    title: 'Text Detection',
    desc: 'NLP pipeline to detect GPT-generated or AI-written text.',
    tag: 'TF-IDF + NaiveBayes',
    color: 'orange'
  }
];

export default function HomePage() {
  return (
    <div className="home-page">
      {/* Hero */}
      <p></p>
      <section className="hero">
        <div className="hero-badge">AI-Powered Detection</div>
        <h1 className="hero-title">
          Deepfake <span className="gradient-text">Detection</span> System
        </h1>
        <p className="hero-desc">
          Upload images, videos, audio, or paste text to instantly detect deepfakes and AI-generated
          content with deep learning models.
        </p>
        <div className="hero-actions">
          <Link to="/image" className="btn btn-primary btn-lg">Start Detecting →</Link>
          {/* <a
            href="https://github.com/"
            className="btn btn-secondary btn-lg"
            target="_blank" rel="noopener noreferrer"
          >GitHub</a> */}
        </div>
      </section>

      {/* Cards */}
      <section className="modality-grid container">
        {MODALITIES.map(m => (
          <Link key={m.to} to={m.to} className={`modality-card color-${m.color}`}>
            <div className="mc-icon">{m.icon}</div>
            <div className="mc-body">
              <h3 className="mc-title">{m.title}</h3>
              <p className="mc-desc">{m.desc}</p>
              <span className="mc-tag">{m.tag}</span>
            </div>
          </Link>
        ))}
      </section>

      {/* How it works */}
      <section className="how-it-works container">
        <h2 className="section-title">How It Works</h2>
        <div className="steps">
          {['Upload your file or paste text', 'AI model analyzes the content', 'Get instant Real/Fake verdict with confidence score'].map((step, i) => (
            <div className="step" key={i}>
              <div className="step-num">{i + 1}</div>
              <p className="step-text">{step}</p>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
