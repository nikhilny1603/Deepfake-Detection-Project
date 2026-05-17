"""
Training Script — Text Deepfake Detection
TF-IDF + Multinomial Naive Bayes

Dataset: CSV file with columns: text, label (0=human, 1=AI)
Or two separate text files: real_texts.txt and fake_texts.txt
"""

import os
import sys
import argparse
import logging
import json
import csv
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))
from models.text_model import DeepfakeTextDetector

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def load_from_csv(csv_path):
    """Load texts and labels from CSV"""
    texts, labels = [], []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row.get('text', row.get('content', '')).strip()
            label = int(row.get('label', row.get('is_ai', 0)))
            if text:
                texts.append(text)
                labels.append(label)
    return texts, labels


def load_from_txt_files(real_path, fake_path):
    """Load from separate real/fake text files (one sample per line)"""
    texts, labels = [], []
    with open(real_path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                texts.append(line)
                labels.append(0)
    with open(fake_path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                texts.append(line)
                labels.append(1)
    return texts, labels


def evaluate(detector, texts, labels):
    """Evaluate the model on a set of texts"""
    preds, probs = [], []
    for text in texts:
        result = detector.predict(text)
        preds.append(1 if result['label'] == 'Fake' else 0)
        probs.append(result['fake_probability'])

    acc  = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, zero_division=0)
    rec  = recall_score(labels, preds, zero_division=0)
    f1   = f1_score(labels, preds, zero_division=0)

    logger.info("\n" + classification_report(labels, preds, target_names=['Human', 'AI-generated']))
    logger.info(f"Confusion matrix:\n{confusion_matrix(labels, preds)}")

    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}


def main(args):
    # Load data
    if args.csv:
        texts, labels = load_from_csv(args.csv)
    elif args.real_file and args.fake_file:
        texts, labels = load_from_txt_files(args.real_file, args.fake_file)
    else:
        logger.error("Provide --csv OR both --real-file and --fake-file")
        return

    logger.info(f"Loaded {len(texts)} samples | Real: {labels.count(0)} | AI: {labels.count(1)}")

    # Split
    tr_texts, va_texts, tr_labels, va_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # Train
    detector = DeepfakeTextDetector()
    logger.info("Training TF-IDF + MultinomialNB model...")
    detector.train(tr_texts, tr_labels)

    # Evaluate
    logger.info("\n=== Validation Results ===")
    metrics = evaluate(detector, va_texts, va_labels)
    logger.info(f"Val Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1']:.4f}")

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, 'text_model.pkl')
    detector.save(save_path)

    # Save metrics
    with open(os.path.join(args.output_dir, 'text_training_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Model saved to {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv',        type=str, help='CSV file with columns: text, label')
    parser.add_argument('--real-file',  type=str, help='Text file with human texts (one per line)')
    parser.add_argument('--fake-file',  type=str, help='Text file with AI texts (one per line)')
    parser.add_argument('--output-dir', type=str, default='../models')
    main(parser.parse_args())
