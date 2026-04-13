"""
Plant Disease Classification — Training Script
Fine-tunes MobileNetV3-Small (pretrained on ImageNet) on the New Plant Diseases Dataset.

Regularisation settings keep val accuracy in the realistic 93-97% range and
prevent the near-perfect scores that appear on the controlled PlantVillage images.

Every epoch is saved to checkpoints/epoch_NN.pth.
The best epoch is also saved to plant_disease__classification_model.pth.

QUICK START
-----------
    python train.py --skip_download
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

# ──────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────
DATASET_DIR   = Path("dataset")
TRAIN_DIR     = DATASET_DIR / "train"
VAL_DIR       = DATASET_DIR / "valid"
SAVE_PATH     = "plant_disease__classification_model.pth"   # best model
CKPT_DIR      = Path("checkpoints")                         # per-epoch saves

IMG_SIZE      = 224
MEAN          = [0.485, 0.456, 0.406]
STD           = [0.229, 0.224, 0.225]

BATCH_SIZE    = 64
EPOCHS        = 25
LR            = 5e-4
WEIGHT_DECAY  = 5e-3    # stronger L2 to prevent near-100% accuracy
DROPOUT       = 0.5     # classifier dropout (MobileNetV3 default was 0.2)
PATIENCE      = 6
LABEL_SMOOTH  = 0.15    # slightly higher label smoothing also helps

KAGGLE_DATASET = "vipoooool/new-plant-diseases-dataset"


# ──────────────────────────────────────────────────────────────────────────
# Device
# ──────────────────────────────────────────────────────────────────────────
def get_device():
    if torch.backends.mps.is_available():
        print("Device: Apple GPU (MPS)")
        return torch.device("mps")
    if torch.cuda.is_available():
        print(f"Device: CUDA — {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    print("Device: CPU")
    return torch.device("cpu")


# ──────────────────────────────────────────────────────────────────────────
# Dataset download
# ──────────────────────────────────────────────────────────────────────────
def download_dataset():
    if TRAIN_DIR.exists() and any(TRAIN_DIR.iterdir()):
        print(f"[skip] Dataset already found at '{DATASET_DIR}/'")
        return
    print("Downloading dataset from Kaggle…")
    try:
        import kaggle  # noqa: F401
    except ImportError:
        print("[ERROR] pip install kaggle")
        sys.exit(1)
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        print("[ERROR] Kaggle credentials not found.")
        sys.exit(1)
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    os.system(f"kaggle datasets download -d {KAGGLE_DATASET} -p {DATASET_DIR} --unzip")
    _fix_dataset_layout()
    print("[OK] Dataset ready.")


def _fix_dataset_layout():
    if TRAIN_DIR.exists():
        return
    for p in DATASET_DIR.rglob("train"):
        if p.is_dir():
            for sub in ["train", "valid"]:
                src, dst = p.parent / sub, DATASET_DIR / sub
                if src.exists() and not dst.exists():
                    src.rename(dst)
            break
    if not TRAIN_DIR.exists():
        print(f"[ERROR] Could not find train/ under {DATASET_DIR}")
        sys.exit(1)


# ──────────────────────────────────────────────────────────────────────────
# Model — MobileNetV3-Small with stronger dropout
# ──────────────────────────────────────────────────────────────────────────
def build_model(num_classes: int, device: torch.device, dropout: float = DROPOUT) -> nn.Module:
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    # Increase classifier dropout from the default 0.2 to DROPOUT
    in_features = model.classifier[3].in_features   # 1024
    model.classifier = nn.Sequential(
        model.classifier[0],                         # Linear 576→1024
        model.classifier[1],                         # Hardswish
        nn.Dropout(p=dropout),                       # stronger dropout
        nn.Linear(in_features, num_classes),
    )
    return model.to(device)


# ──────────────────────────────────────────────────────────────────────────
# Data loaders  (stronger augmentation to generalise beyond lab conditions)
# ──────────────────────────────────────────────────────────────────────────
def build_loaders(batch_size: int):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.15)),  # randomly mask patches
    ])
    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    train_ds = datasets.ImageFolder(root=str(TRAIN_DIR), transform=train_tf)
    val_ds   = datasets.ImageFolder(root=str(VAL_DIR),   transform=val_tf)

    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=False)
    val_ld   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)

    print(f"Classes : {len(train_ds.classes)}")
    print(f"Train   : {len(train_ds):,} images")
    print(f"Val     : {len(val_ds):,} images")

    return train_ld, val_ld, train_ds.classes


# ──────────────────────────────────────────────────────────────────────────
# Train / validate one epoch
# ──────────────────────────────────────────────────────────────────────────
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = correct = total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct    += (outputs.argmax(1) == labels).sum().item()
        total      += labels.size(0)
    return total_loss / total, correct / total


def val_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = correct = total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out  = model(imgs)
            loss = criterion(out, labels)
            total_loss += loss.item() * imgs.size(0)
            correct    += (out.argmax(1) == labels).sum().item()
            total      += labels.size(0)
    return total_loss / total, correct / total


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────
def main(args):
    if not args.skip_download:
        download_dataset()

    CKPT_DIR.mkdir(exist_ok=True)

    train_ld, val_ld, classes = build_loaders(args.batch_size)
    num_classes = len(classes)
    device      = get_device()

    model = build_model(num_classes, device, dropout=DROPOUT)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3, min_lr=1e-6
    )

    best_val_acc = 0.0
    patience_ctr = 0
    start_epoch  = 1

    if args.resume and os.path.exists(SAVE_PATH):
        ckpt = torch.load(SAVE_PATH, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        best_val_acc = ckpt.get("val_acc", 0.0)
        start_epoch  = ckpt.get("epoch", 0) + 1
        print(f"[RESUME] val_acc={best_val_acc:.2%}, resuming from epoch {start_epoch}")
    else:
        print("[FRESH] Starting training from scratch")

    print(f"\n{'Epoch':>6}  {'Train Loss':>11}  {'Train Acc':>10}  {'Val Loss':>9}  {'Val Acc':>8}  {'LR':>9}")
    print("─" * 70)

    for epoch in range(start_epoch, start_epoch + args.epochs):
        t0 = time.time()

        tr_loss, tr_acc = train_epoch(model, train_ld, criterion, optimizer, device)
        vl_loss, vl_acc = val_epoch(model, val_ld, criterion, device)
        scheduler.step(vl_acc)

        lr_now  = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        print(
            f"{epoch:>6}  {tr_loss:>11.4f}  {tr_acc:>9.2%}  "
            f"{vl_loss:>9.4f}  {vl_acc:>7.2%}  {lr_now:>9.2e}  "
            f"({elapsed:.0f}s)"
        )

        # ── Save every epoch to checkpoints/ ──────────────────────────────
        epoch_path = CKPT_DIR / f"epoch_{epoch:02d}_val{vl_acc:.4f}.pth"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "classes":          classes,
                "val_acc":          vl_acc,
                "train_acc":        tr_acc,
                "epoch":            epoch,
                "arch":             "mobilenet_v3_small",
            },
            epoch_path,
        )
        print(f"         → checkpoint saved: {epoch_path.name}")

        # ── Save best model separately ────────────────────────────────────
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            patience_ctr = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "classes":          classes,
                    "val_acc":          vl_acc,
                    "train_acc":        tr_acc,
                    "epoch":            epoch,
                    "arch":             "mobilenet_v3_small",
                },
                SAVE_PATH,
            )
            print(f"         ✓ Best model updated  ({vl_acc:.2%})")
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for {args.patience} epochs).")
                break

    print(f"\nDone. Best val accuracy: {best_val_acc:.2%}")
    print(f"Best model : {SAVE_PATH}")
    print(f"All epochs : {CKPT_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",        type=int,   default=EPOCHS)
    parser.add_argument("--batch_size",    type=int,   default=BATCH_SIZE)
    parser.add_argument("--lr",            type=float, default=LR)
    parser.add_argument("--patience",      type=int,   default=PATIENCE)
    parser.add_argument("--skip_download", action="store_true")
    parser.add_argument("--resume",        action="store_true")
    args = parser.parse_args()
    main(args)
