"""Trains the ISLens LSTM on recorded samples in data/raw/<phrase>/*.json.

Usage:
    python train.py
"""

import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from features import sequence_to_tensor, SEQ_LEN
from labels import PHRASES
from model import build_model, save_checkpoint

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "islens_lstm.pt"
SEED = 42
HIDDEN_SIZE = 64
NUM_LAYERS = 1
EPOCHS = 200
LR = 1e-3
VAL_FRACTION = 0.2
AUGMENT_COPIES = 4  # extra noisy/time-jittered copies per real sample
NOISE_STD = 0.02
MAX_TIME_SHIFT = 2


def load_raw_samples():
    samples = []  # list of (tensor(SEQ_LEN, FRAME_DIM), label_index)
    counts = {}
    for label_idx, phrase in enumerate(PHRASES):
        phrase_dir = DATA_DIR / phrase
        files = sorted(phrase_dir.glob("*.json")) if phrase_dir.exists() else []
        counts[phrase] = len(files)
        for f in files:
            with open(f, "r", encoding="utf-8") as fh:
                record = json.load(fh)
            tensor = sequence_to_tensor(record["frames"], seq_len=SEQ_LEN)
            samples.append((tensor, label_idx))
    return samples, counts


def augment(tensor, rng):
    copies = [tensor]
    for _ in range(AUGMENT_COPIES):
        t = tensor.copy()
        # small gaussian jitter on non-presence features (first 2 cols are 0/1 flags)
        noise = rng.normal(0, NOISE_STD, size=t.shape).astype(np.float32)
        noise[:, :2] = 0.0
        t = t + noise
        # small random shift in time (roll), padding edges by repeating
        shift = rng.integers(-MAX_TIME_SHIFT, MAX_TIME_SHIFT + 1)
        if shift != 0:
            t = np.roll(t, shift, axis=0)
        copies.append(t.astype(np.float32))
    return copies


def stratified_split(samples, val_fraction, rng):
    by_class = {}
    for tensor, label in samples:
        by_class.setdefault(label, []).append((tensor, label))

    train, val = [], []
    for label, items in by_class.items():
        rng.shuffle(items)
        n_val = max(1, int(round(len(items) * val_fraction))) if len(items) > 1 else 0
        val.extend(items[:n_val])
        train.extend(items[n_val:])
    return train, val


def to_batch(items):
    X = torch.tensor(np.stack([t for t, _ in items]), dtype=torch.float32)
    y = torch.tensor([l for _, l in items], dtype=torch.long)
    return X, y


def main():
    random.seed(SEED)
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)

    samples, counts = load_raw_samples()
    print("Recorded samples per phrase:")
    for phrase, n in counts.items():
        print(f"  {phrase:15s} {n}")
    total = len(samples)
    if total == 0:
        print("\nNo recorded samples found in data/raw/. Use the recorder tool first.")
        return
    if any(n == 0 for n in counts.values()):
        missing = [p for p, n in counts.items() if n == 0]
        print(f"\nWarning: no samples for: {', '.join(missing)}. Training will skip those classes' accuracy meaningfully.")

    train_raw, val_raw = stratified_split(samples, VAL_FRACTION, rng)
    print(f"\nTrain samples (before augmentation): {len(train_raw)}  Val samples: {len(val_raw)}")

    train_items = []
    for tensor, label in train_raw:
        for aug_tensor in augment(tensor, rng):
            train_items.append((aug_tensor, label))
    print(f"Train samples (after augmentation): {len(train_items)}")

    Xtrain, ytrain = to_batch(train_items)
    Xval, yval = to_batch(val_raw) if val_raw else (None, None)

    model = build_model(PHRASES, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = -1.0
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(Xtrain)
        loss = criterion(logits, ytrain)
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0 or epoch == EPOCHS:
            model.eval()
            with torch.no_grad():
                train_acc = (logits.argmax(1) == ytrain).float().mean().item()
                msg = f"epoch {epoch:4d}  loss {loss.item():.4f}  train_acc {train_acc:.2f}"
                if Xval is not None:
                    val_logits = model(Xval)
                    val_acc = (val_logits.argmax(1) == yval).float().mean().item()
                    msg += f"  val_acc {val_acc:.2f}"
                    if val_acc >= best_val_acc:
                        best_val_acc = val_acc
                        best_state = {k: v.clone() for k, v in model.state_dict().items()}
                print(msg)

    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        best_val_acc = None

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(str(MODEL_PATH), model, PHRASES, HIDDEN_SIZE, NUM_LAYERS, SEQ_LEN)
    print(f"\nSaved model to {MODEL_PATH}")
    if best_val_acc is not None:
        print(f"Best validation accuracy: {best_val_acc:.2f} (chance = {1/len(PHRASES):.2f})")
        if best_val_acc < 0.4:
            print("This is low for a 5-class demo. Consider recording more/cleaner samples per phrase.")


if __name__ == "__main__":
    main()
