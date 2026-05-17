"""
Training Script — Audio Deepfake Detection
SpectrogramCNN classifier

Dataset structure expected:
  data/
    train/
      real/  ← real audio files (.wav, .mp3, ...)
      fake/  ← AI-generated audio files
    val/
      real/
      fake/
"""

import os
import sys
import argparse
import logging
import json
import glob

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))
from models.audio_model import SpectrogramCNN, audio_to_melspectrogram

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

AUDIO_EXTENSIONS = ('.wav', '.mp3', '.flac', '.ogg', '.m4a')


class AudioDataset(Dataset):
    """Dataset that loads audio files and converts to spectrograms on the fly"""

    def __init__(self, root_dir):
        self.samples = []  # List of (path, label)
        for label_idx, label_name in enumerate(['real', 'fake']):
            folder = os.path.join(root_dir, label_name)
            if not os.path.isdir(folder):
                logger.warning(f"Missing folder: {folder}")
                continue
            for ext in AUDIO_EXTENSIONS:
                for path in glob.glob(os.path.join(folder, f'*{ext}')):
                    self.samples.append((path, label_idx))
        logger.info(f"Loaded {len(self.samples)} samples from {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            tensor = audio_to_melspectrogram(path)
            return tensor.squeeze(0), label  # (1, H, W), label
        except Exception as e:
            logger.warning(f"Error loading {path}: {e}. Using zeros.")
            from models.audio_model import SPEC_HEIGHT, SPEC_WIDTH
            return torch.zeros(1, SPEC_HEIGHT, SPEC_WIDTH), label


def compute_metrics(labels, preds):
    acc  = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, zero_division=0)
    rec  = recall_score(labels, preds, zero_division=0)
    f1   = f1_score(labels, preds, zero_division=0)
    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for specs, labels in loader:
        specs, labels = specs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(specs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * specs.size(0)
        all_preds.extend(outputs.argmax(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return total_loss / len(loader.dataset), compute_metrics(all_labels, all_preds)


@torch.no_grad()
def validate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for specs, labels in loader:
        specs, labels = specs.to(device), labels.to(device)
        outputs = model(specs)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * specs.size(0)
        all_preds.extend(outputs.argmax(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return total_loss / len(loader.dataset), compute_metrics(all_labels, all_preds)


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    train_ds = AudioDataset(os.path.join(args.data_dir, 'train'))
    val_ds   = AudioDataset(os.path.join(args.data_dir, 'val'))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    model = SpectrogramCNN(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    os.makedirs(args.output_dir, exist_ok=True)
    best_acc = 0.0
    history = []

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_m = train_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_m = validate_epoch(model, val_loader, criterion, device)
        scheduler.step()

        logger.info(f"[{epoch}/{args.epochs}] Train Acc: {tr_m['accuracy']:.4f} | Val Acc: {va_m['accuracy']:.4f} F1: {va_m['f1']:.4f}")
        history.append({'epoch': epoch, 'train_loss': tr_loss, **{f'train_{k}': v for k, v in tr_m.items()},
                        'val_loss': va_loss, **{f'val_{k}': v for k, v in va_m.items()}})

        if va_m['accuracy'] > best_acc:
            best_acc = va_m['accuracy']
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'audio_model.pth'))
            logger.info(f"  Best model saved ({best_acc:.4f})")

    with open(os.path.join(args.output_dir, 'audio_training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    logger.info(f"Best val accuracy: {best_acc:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir',   required=True)
    parser.add_argument('--output-dir', default='../models')
    parser.add_argument('--epochs',     type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr',         type=float, default=1e-3)
    parser.add_argument('--workers',    type=int, default=2)
    main(parser.parse_args())
