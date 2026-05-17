"""
Training Script — Image Deepfake Detection
ResNet50 binary classifier

Dataset structure expected:
  data/
    train/
      real/   ← real face images
      fake/   ← deepfake images
    val/
      real/
      fake/
"""

import os
import sys
import argparse
import logging
import json
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Allow importing from backend
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))
from models.image_model import DeepfakeImageClassifier

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


# ─── Data Augmentation ────────────────────────────────────────────────────────

TRAIN_TRANSFORMS = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

VAL_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# ─── Helpers ──────────────────────────────────────────────────────────────────

def compute_metrics(all_labels, all_preds, all_probs):
    """Compute classification metrics"""
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}


def make_weighted_sampler(dataset):
    """Create a WeightedRandomSampler to handle class imbalance"""
    class_counts = [0, 0]
    for _, label in dataset.imgs:
        class_counts[label] += 1
    weights = [1.0 / class_counts[label] for _, label in dataset.imgs]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


# ─── Training Loop ────────────────────────────────────────────────────────────

def train_epoch(model, loader, criterion, optimizer, device, scaler):
    """Run one training epoch with mixed precision"""
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        if batch_idx % 50 == 0:
            logger.info(f"  Batch {batch_idx}/{len(loader)} — loss: {loss.item():.4f}")

    epoch_loss = running_loss / len(loader.dataset)
    metrics = compute_metrics(all_labels, all_preds, None)
    return epoch_loss, metrics


@torch.no_grad()
def validate_epoch(model, loader, criterion, device):
    """Run validation pass"""
    model.eval()
    running_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        probs = torch.softmax(outputs, dim=1)[:, 1]
        preds = outputs.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    metrics = compute_metrics(all_labels, all_preds, all_probs)
    return epoch_loss, metrics


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Datasets
    train_dataset = ImageFolder(os.path.join(args.data_dir, 'train'), transform=TRAIN_TRANSFORMS)
    val_dataset   = ImageFolder(os.path.join(args.data_dir, 'val'),   transform=VAL_TRANSFORMS)

    logger.info(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")
    logger.info(f"Classes: {train_dataset.class_to_idx}")

    sampler = make_weighted_sampler(train_dataset) if args.balance else None
    shuffle = sampler is None

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        sampler=sampler, shuffle=shuffle,
        num_workers=args.workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.workers, pin_memory=True
    )

    # Model
    model = DeepfakeImageClassifier(pretrained=True, freeze_base=args.freeze_base)
    model = model.to(device)

    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    os.makedirs(args.output_dir, exist_ok=True)
    best_val_acc = 0.0
    history = []

    for epoch in range(1, args.epochs + 1):
        logger.info(f"\n{'='*50}")
        logger.info(f"Epoch {epoch}/{args.epochs}")

        train_loss, train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_metrics = validate_epoch(model, val_loader, criterion, device)
        scheduler.step()

        logger.info(f"Train — Loss: {train_loss:.4f} | Acc: {train_metrics['accuracy']:.4f}")
        logger.info(f"Val   — Loss: {val_loss:.4f}   | Acc: {val_metrics['accuracy']:.4f} | F1: {val_metrics['f1']:.4f}")

        history.append({
            'epoch': epoch,
            'train_loss': train_loss, **{f'train_{k}': v for k, v in train_metrics.items()},
            'val_loss': val_loss,   **{f'val_{k}': v for k, v in val_metrics.items()}
        })

        # Save best model
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            save_path = os.path.join(args.output_dir, 'image_model.pth')
            torch.save(model.state_dict(), save_path)
            logger.info(f"*** New best model saved ({best_val_acc:.4f}) → {save_path}")

    # Save training history
    history_path = os.path.join(args.output_dir, 'image_training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    logger.info(f"Training history saved to {history_path}")
    logger.info(f"Best validation accuracy: {best_val_acc:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Image Deepfake Detector')
    parser.add_argument('--data-dir',    type=str, required=True, help='Path to dataset (with train/ and val/)')
    parser.add_argument('--output-dir',  type=str, default='../models', help='Where to save models')
    parser.add_argument('--epochs',      type=int, default=20)
    parser.add_argument('--batch-size',  type=int, default=32)
    parser.add_argument('--lr',          type=float, default=1e-4)
    parser.add_argument('--workers',     type=int, default=4)
    parser.add_argument('--freeze-base', action='store_true', help='Freeze backbone layers')
    parser.add_argument('--balance',     action='store_true', help='Use weighted sampling to balance classes')
    args = parser.parse_args()
    main(args)
