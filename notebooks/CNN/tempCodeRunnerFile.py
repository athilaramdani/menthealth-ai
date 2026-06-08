import torch
import torch.nn as nn
import torch.optim as optim
import logging
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, classification_report, f1_score
from tqdm import tqdm
from pathlib import Path
import sys
from collections import Counter
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
DL_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = DL_DIR.parent

if str(DL_DIR) not in sys.path:
    sys.path.append(str(DL_DIR))

from dataloader import get_dataloaders

RESULTS_DIR = PROJECT_ROOT / "results"
(RESULTS_DIR / "metrics").mkdir(parents=True, exist_ok=True)
(RESULTS_DIR / "plots").mkdir(parents=True, exist_ok=True)
(RESULTS_DIR / "confusion_matrix").mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(SCRIPT_DIR / "train_2dcnn.log"),
        logging.StreamHandler(),
    ],
)

CLASS_NAMES = ["NORMAL", "DEPRESI"]


class MelSpectrogram2DCNN(nn.Module):
    def __init__(self, num_classes=2, dropout_rate=0.3):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(4, 4),
            nn.Dropout2d(0.2),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.global_pool(x)
        x = self.flatten(x)
        return self.fc(x)


def spec_augment(spectrograms, freq_mask_param=10, time_mask_param=30, num_masks=1):
    augmented = spectrograms.clone()
    _, _, n_mels, time_steps = augmented.shape

    for _ in range(num_masks):
        f = torch.randint(0, freq_mask_param + 1, (1,)).item()
        if f > 0:
            f0 = torch.randint(0, max(1, n_mels - f), (1,)).item()
            augmented[:, :, f0 : f0 + f, :] = 0

    for _ in range(num_masks):
        t = torch.randint(0, time_mask_param + 1, (1,)).item()
        if t > 0:
            t0 = torch.randint(0, max(1, time_steps - t), (1,)).item()
            augmented[:, :, :, t0 : t0 + t] = 0

    return augmented


def save_evaluation_results(y_true, y_pred, history, run_name="2class"):
    logging.info("Menyimpan hasil evaluasi...")

    epochs_range = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(15, 4))
    plt.subplot(1, 3, 1)
    plt.plot(epochs_range, history["train_loss"], label="Train Loss")
    plt.plot(epochs_range, history["val_loss"], label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(epochs_range, history["train_acc"], label="Train Accuracy")
    plt.plot(epochs_range, history["val_acc"], label="Validation Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(epochs_range, history["val_macro_f1"], label="Validation Macro F1", color="green")
    plt.title("Validation Macro F1")
    plt.xlabel("Epochs")
    plt.ylabel("Macro F1")
    plt.ylim(0, 1.0)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "plots" / f"2d_cnn_learning_curves_{run_name}.png")
    plt.close()

    labels = list(range(len(CLASS_NAMES)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
    )
    plt.title("Confusion Matrix - 2D CNN (Mel-Spectrogram, 2 classes)")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.savefig(RESULTS_DIR / "confusion_matrix" / f"2d_cnn_cm_{run_name}.png")
    plt.close()

    report = classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        labels=labels,
        zero_division=0,
    )

    out_report = RESULTS_DIR / "metrics" / f"2d_cnn_classification_report_{run_name}.txt"
    with open(out_report, "w", encoding="utf-8") as f:
        f.write("=== 2D CNN Classification Report (2 classes) ===\n\n")
        f.write(report)

    logging.info(f"Classification report tersimpan di {out_report}")


def train_model(epochs=100, batch_size=16, learning_rate=5e-4, max_len=800):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")

    model = MelSpectrogram2DCNN(num_classes=len(CLASS_NAMES)).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Total parameters: {total_params:,} (~{total_params/1000:.1f}K)")

    train_loader, val_loader, _ = get_dataloaders(
        batch_size=batch_size,
        max_len=max_len,
        use_weighted_sampler=False,
    )

    # Class weights
    train_dataset = train_loader.dataset
    if hasattr(train_dataset, "dataset"):
        train_labels = [train_dataset.dataset.labels[i] for i in train_dataset.indices]
    else:
        train_labels = list(train_dataset.labels)

    label_tensor = torch.tensor(train_labels, dtype=torch.long)
    class_counts = torch.bincount(label_tensor, minlength=len(CLASS_NAMES)).float().clamp(min=1.0)

    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum() * len(CLASS_NAMES)
    class_weights = class_weights.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    warmup_epochs = 5
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.2, end_factor=1.0, total_iters=warmup_epochs
    )
    reduce_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=12, min_lr=1e-6
    )

    best_macro_f1 = -1.0
    best_epoch = 0
    early_stop_patience = 30
    epochs_no_improve = 0

    model_out_dir = PROJECT_ROOT / "models" / "dl" / "cnn"
    model_out_dir.mkdir(parents=True, exist_ok=True)
    output_model_path = model_out_dir / "best_model.pt"

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "val_macro_f1": [],
    }

    best_preds, best_labels = [], []
    last_lr = learning_rate

    for epoch in range(epochs):
        current_lr = optimizer.param_groups[0]["lr"]
        logging.info(f"--- Epoch {epoch + 1}/{epochs} | LR: {current_lr:.6f} ---")

        # TRAIN
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        for spectrograms, labels in tqdm(train_loader, desc="Training", leave=False):
            spectrograms, labels = spectrograms.to(device), labels.to(device)
            spectrograms = spec_augment(spectrograms)

            optimizer.zero_grad()
            outputs = model(spectrograms)
            loss = criterion(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        epoch_train_loss = train_loss / max(1, len(train_loader))
        epoch_train_acc = 100.0 * correct_train / max(1, total_train)

        # VAL
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for spectrograms, labels in val_loader:
                spectrograms, labels = spectrograms.to(device), labels.to(device)
                outputs = model(spectrograms)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        epoch_val_loss = val_loss / max(1, len(val_loader))
        epoch_val_acc = 100.0 * correct_val / max(1, total_val)
        epoch_macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_acc"].append(epoch_val_acc)
        history["val_macro_f1"].append(epoch_macro_f1)

        logging.info(f"Train  → Loss: {epoch_train_loss:.4f} | Acc: {epoch_train_acc:.2f}%")
        logging.info(
            f"Val    → Loss: {epoch_val_loss:.4f} | Acc: {epoch_val_acc:.2f}% | Macro F1: {epoch_macro_f1:.4f}"
        )

        pred_dist = dict(sorted(Counter(all_preds).items()))
        pred_dist_named = {CLASS_NAMES[k]: v for k, v in pred_dist.items()}
        logging.info(f"Dist prediksi val: {pred_dist_named} ({len(pred_dist)}/2 kelas terdeteksi)")

        if epoch_macro_f1 > best_macro_f1:
            best_macro_f1 = epoch_macro_f1
            best_epoch = epoch + 1
            torch.save(model.state_dict(), output_model_path)
            best_preds = list(all_preds)
            best_labels = list(all_labels)
            epochs_no_improve = 0
            logging.info(f"✅ Model terbaik disimpan (Macro F1: {epoch_macro_f1:.4f}) → {output_model_path}")
        else:
            epochs_no_improve += 1

        if epoch < warmup_epochs:
            warmup_scheduler.step()
        else:
            reduce_scheduler.step(epoch_macro_f1)

        new_lr = optimizer.param_groups[0]["lr"]
        if new_lr != last_lr:
            logging.info(f"⚠️ LR berubah: {last_lr:.6f} → {new_lr:.6f}")
        last_lr = new_lr

        if epochs_no_improve >= early_stop_patience:
            logging.info(f"Early stopping pada epoch {epoch + 1} (no improve selama {early_stop_patience} epoch).")
            break

    if len(best_preds) == 0:
        best_preds = list(all_preds)
        best_labels = list(all_labels)

    logging.info(f"Training selesai. Best epoch {best_epoch}, Macro F1: {best_macro_f1:.4f}.")
    save_evaluation_results(y_true=best_labels, y_pred=best_preds, history=history, run_name="2class")


if __name__ == "__main__":
    train_model(epochs=100, batch_size=16, learning_rate=5e-4, max_len=800)

