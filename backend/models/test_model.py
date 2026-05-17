import torch
import sys
sys.path.insert(0, '.')

from image_model import load_image_model, preprocess_image
from video_model import load_video_model

print("=== Testing Image Model ===")
try:
    model = load_image_model('model_epoch_1.pth')
    print("✅ Image model loaded successfully")

    # Create a dummy input (1 image, 3 channels, 224x224)
    dummy = torch.randn(1, 3, 224, 224)
    probs = model.predict(dummy)
    real = round(float(probs[0][0]) * 100, 2)
    fake = round(float(probs[0][1]) * 100, 2)
    print(f"✅ Image model running — Real: {real}%  Fake: {fake}%")
except Exception as e:
    print(f"❌ Image model error: {e}")

print("\n=== Testing Video Model ===")
try:
    model = load_video_model('model_epoch_3.pth')
    print("✅ Video model loaded successfully")

    # Create a dummy input (1 frame, 3 channels, 224x224)
    dummy = torch.randn(1, 3, 224, 224)
    probs = model.predict_frame(dummy)
    real = round(float(probs[0][0]) * 100, 2)
    fake = round(float(probs[0][1]) * 100, 2)
    print(f"✅ Video model running — Real: {real}%  Fake: {fake}%")
except Exception as e:
    print(f"❌ Video model error: {e}")