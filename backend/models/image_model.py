"""
Image Deepfake Detection Model
ResNet18 + LSTM + FC (matches your trained model architecture)
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import logging

logger = logging.getLogger(__name__)


class DeepfakeImageClassifier(nn.Module):
    """
    ResNet18 backbone + LSTM + FC classifier.
    Matches the architecture of the trained .pth model.
    """

    def __init__(self, hidden_size=256, num_classes=2):
        super(DeepfakeImageClassifier, self).__init__()

        # ResNet18 backbone — fc replaced with Identity (no classification head)
        self.cnn = models.resnet18(weights=None)
        self.cnn.fc = nn.Identity()

        # LSTM for temporal/sequence modelling
        self.lstm = nn.LSTM(512, hidden_size, batch_first=True)

        # Final binary classifier
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        """
        x: (B, C, H, W) — single image, treated as sequence of length 1
        """
        # Extract CNN features: (B, 512)
        features = self.cnn(x)

        # Add sequence dimension: (B, 1, 512)
        features = features.unsqueeze(1)

        # LSTM: output shape (B, 1, hidden_size)
        lstm_out, _ = self.lstm(features)

        # Take last timestep: (B, hidden_size)
        out = lstm_out[:, -1, :]

        # Classify: (B, num_classes)
        return self.fc(out)

    def predict(self, image_tensor):
        """Run inference and return class probabilities"""
        self.eval()
        with torch.no_grad():
            logits = self.forward(image_tensor)
            probabilities = torch.softmax(logits, dim=1)
        return probabilities


IMAGE_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def load_image_model(model_path=None):
    """Load the image deepfake detection model from .pth file"""
    if model_path is None:
        logger.warning("No model path — running in demo mode")
        model = DeepfakeImageClassifier()
        model.eval()
        return model

    # Load state dict
    state_dict = torch.load(model_path, map_location='cpu')

    # Auto-detect hidden_size and num_classes from saved weights
    hidden_size = state_dict['fc.weight'].shape[1]
    num_classes  = state_dict['fc.weight'].shape[0]
    logger.info(f"Detected: hidden_size={hidden_size}, num_classes={num_classes}")

    model = DeepfakeImageClassifier(hidden_size=hidden_size, num_classes=num_classes)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    logger.info(f"Image model loaded from {model_path}")
    return model


def preprocess_image(image_path):
    """Load and preprocess image for model input"""
    try:
        img = Image.open(image_path).convert('RGB')
        tensor = IMAGE_TRANSFORMS(img).unsqueeze(0)
        return tensor
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        raise ValueError(f"Could not process image: {e}")