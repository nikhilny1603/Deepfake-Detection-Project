# 🔍 DeepFake Detection System

An end-to-end AI-powered deepfake detection system that analyzes **images**, **videos**, **audio**, and **text** to determine if content is real or fake.

---

## 🎯 Features

| Modality | Model | Input |
|----------|-------|-------|
| Image | ResNet50 (CNN) | PNG, JPG, WebP, BMP |
| Video | MobileNetV3 + Temporal Aggregation | MP4, AVI, MOV, MKV |
| Audio | SpectrogramCNN (Mel spectrogram) | WAV, MP3, FLAC, OGG |
| Text | TF-IDF + MultinomialNB | Raw text (≥20 chars) |

---

## 📁 Project Structure

```
deepfake-detection/
├── backend/               # Flask API server
│   ├── models/            # Model definitions (PyTorch + sklearn)
│   │   ├── image_model.py
│   │   ├── video_model.py
│   │   ├── audio_model.py
│   │   └── text_model.py
│   ├── routes/            # Flask Blueprint routes
│   │   ├── image_routes.py
│   │   ├── video_routes.py
│   │   ├── audio_routes.py
│   │   └── text_routes.py
│   ├── services/          # Business logic & model loaders
│   │   ├── model_loader.py
│   │   ├── image_service.py
│   │   ├── video_service.py
│   │   ├── audio_service.py
│   │   └── text_service.py
│   ├── utils/
│   │   └── file_utils.py
│   ├── app.py             # Flask application factory
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/              # React.js UI
│   ├── src/
│   │   ├── components/    # Navbar, FileUploader, ResultDisplay, Loader
│   │   ├── pages/         # HomePage, ImagePage, VideoPage, AudioPage, TextPage
│   │   └── services/      # Axios API client
│   ├── public/
│   ├── package.json
│   └── Dockerfile
├── models/                # Trained model weights (place here)
│   ├── image_model.pth    ← train & place here
│   ├── video_model.pth    ← train & place here
│   ├── audio_model.pth    ← train & place here
│   └── text_model.pkl     ← train & place here
├── training/              # Training scripts
│   ├── train_image_model.py
│   ├── train_audio_model.py
│   └── train_text_model.py
├── docker-compose.yml
└── README.md
```

---

## 🚀 Quick Start

### Option 1: Local Development

#### Backend

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run development server
python app.py
# → API available at http://localhost:5000
```

#### Frontend

```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm start
# → UI available at http://localhost:3000
```

### Option 2: Docker Compose (recommended)

```bash
# Build and start all services
docker-compose up --build

# API: http://localhost:5000
# UI:  http://localhost:3000
```

---

## 🌐 API Reference

All endpoints are under the Flask server (default: `http://localhost:5000`).

### `GET /health`
Returns server health status.

```json
{ "status": "healthy", "message": "Deepfake Detection API is running" }
```

---

### `POST /predict/image`
**Content-Type:** `multipart/form-data`

| Field | Type | Description |
|-------|------|-------------|
| `file` | File | Image (PNG/JPG/WebP/BMP) |

**Response:**
```json
{
  "prediction": "Fake",
  "confidence": 0.91,
  "details": {
    "real_probability": 0.09,
    "fake_probability": 0.91,
    "file_size_bytes": 204800,
    "model": "ResNet50",
    "input_size": "224x224"
  }
}
```

---

### `POST /predict/video`
**Content-Type:** `multipart/form-data`

| Field | Type | Description |
|-------|------|-------------|
| `file` | File | Video (MP4/AVI/MOV/MKV/WebM) |
| `max_frames` | int | Frames to sample (8–64, default 32) |

**Response:**
```json
{
  "prediction": "Fake",
  "confidence": 0.87,
  "details": {
    "real_probability": 0.13,
    "fake_probability": 0.87,
    "frames_analyzed": 32,
    "total_fps": 29.97,
    "frame_scores": [0.82, 0.91, 0.78, ...],
    "std_deviation": 0.06,
    "model": "MobileNetV3-Small + Temporal Averaging"
  }
}
```

---

### `POST /predict/audio`
**Content-Type:** `multipart/form-data`

| Field | Type | Description |
|-------|------|-------------|
| `file` | File | Audio (WAV/MP3/FLAC/OGG/M4A) |

**Response:**
```json
{
  "prediction": "Real",
  "confidence": 0.83,
  "details": {
    "real_probability": 0.83,
    "fake_probability": 0.17,
    "audio_features": {
      "duration_sec": 5.0,
      "sample_rate": 22050,
      "spectral_centroid_hz": 2341.5
    },
    "model": "SpectrogramCNN",
    "spectrogram": "Mel (128 bins, 5s window)"
  }
}
```

---

### `POST /predict/text`
**Content-Type:** `application/json`

```json
{ "text": "Your text here..." }
```

**Response:**
```json
{
  "prediction": "Fake",
  "confidence": 0.78,
  "details": {
    "real_probability": 0.22,
    "fake_probability": 0.78,
    "word_count": 120,
    "character_count": 750,
    "model": "TF-IDF + MultinomialNB"
  }
}
```

---

## 🧪 Training Your Own Models

### Image Model

```bash
# Prepare dataset:
# data/train/real/*.jpg
# data/train/fake/*.jpg
# data/val/real/*.jpg
# data/val/fake/*.jpg

python training/train_image_model.py \
  --data-dir ./data \
  --output-dir ./models \
  --epochs 20 \
  --batch-size 32 \
  --balance
```

### Audio Model

```bash
# Prepare dataset:
# data/train/real/*.wav
# data/train/fake/*.wav
# data/val/real/*.wav
# data/val/fake/*.wav

python training/train_audio_model.py \
  --data-dir ./data \
  --output-dir ./models \
  --epochs 30 \
  --batch-size 16
```

### Text Model

```bash
# From CSV (columns: text, label — 0=human, 1=AI):
python training/train_text_model.py \
  --csv ./data/texts.csv \
  --output-dir ./models

# From separate files (one text per line):
python training/train_text_model.py \
  --real-file ./data/human_texts.txt \
  --fake-file ./data/ai_texts.txt \
  --output-dir ./models
```

### Recommended Datasets

| Modality | Dataset |
|----------|---------|
| Image | [FaceForensics++](https://github.com/ondyari/FaceForensics) |
| Video | [DFDC (Deepfake Detection Challenge)](https://ai.facebook.com/datasets/dfdc/) |
| Audio | [ASVspoof 2019/2021](https://www.asvspoof.org/) |
| Text | [HC3](https://github.com/Hello-SimpleAI/chatgpt-comparison-detection) / [GROVER](https://rowanzellers.com/grover/) |

---

## ⚙️ Configuration

### Backend (`backend/.env`)

```env
PORT=5000
DEBUG=False
```

### Frontend (`frontend/.env`)

```env
REACT_APP_API_URL=http://localhost:5000
```

---

## 🔧 Requirements

- **Python** 3.10+
- **Node.js** 18+
- **GPU** (optional but recommended for faster inference)
- **FFmpeg** (required for video frame extraction)

---

## 📝 License

MIT License — free to use, modify, and distribute.
