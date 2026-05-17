# Models Directory

Place trained model files here:

| File              | Description                          | Size (approx) |
|-------------------|--------------------------------------|---------------|
| `image_model.pth` | ResNet50 binary classifier           | ~90 MB        |
| `video_model.pth` | MobileNetV3-Small classifier         | ~10 MB        |
| `audio_model.pth` | SpectrogramCNN weights               | ~5 MB         |
| `text_model.pkl`  | TF-IDF + MultinomialNB pipeline      | ~20 MB        |

## Training

See `../training/` for training scripts.

## Demo Mode

If model files are not present, the system runs in **demo mode**:
- Image/Video/Audio: Uses pretrained ImageNet weights (not tuned for deepfakes)
- Text: Uses heuristic linguistic analysis

For real predictions, train models on a deepfake dataset such as:
- **FaceForensics++** (image/video)
- **ASVspoof** (audio)
- **HC3 / GROVER** datasets (text)
