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
import numpy as np
from collections import Counter

# ==========================================
# PATH SETUP
# ==========================================

SCRIPT_DIR   = Path(__file__).resolve().parent
DL_DIR       = SCRIPT_DIR.parent
PROJECT_ROOT = DL_DIR.parent.parent

if str(DL_DIR) not in sys.path:
    sys.path.append(str(DL_DIR))

from dataloader import get_dataloaders

# ==========================================
# SETUP DIREKTORI & LOGGING
# ==========================================

RESULTS_DIR = PROJECT_ROOT / "results"
(RESULTS_DIR / "metrics").mkdir(parents=True, exist_ok=True)
(RESULTS_DIR / "plots").mkdir(parents=True, exist_ok=True)
(RESULTS_DIR / "confusion_matrix").mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(SCRIPT_DIR / "train_2dcnn.log"),
        logging.StreamHandler()
    ]
)

CLASS_NAMES = ['NORMAL', 'STRES', 'CEMAS', 'DEPRESI']

# ==========================================
# ARSITEKTUR 2D-CNN
# ==========================================

class MelSpectrogram2DCNN(nn.Module):
    def __init__(self, num_classes=4, dropout_rate=0.3):
        super().__init__()

        # Block 1: 1 → 32
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(4, 4),
            nn.Dropout2d(0.2),
        )

        # Block 2: 32 → 64
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
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.global_pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


def spec_augment(spectrograms, freq_mask_param=10, time_mask_param=30, num_masks=1):
    """Spec Augment ringan — 1 masking per dimensi."""
    augmented = spectrograms.clone()
    _, _, n_mels, time_steps = augmented.shape

    for _ in range(num_masks):
        f = torch.randint(0, freq_mask_param + 1, (1,)).item()
        if f > 0:
            f0 = torch.randint(0, max(1, n_mels - f), (1,)).item()
            augmented[:, :, f0:f0 + f, :] = 0

    for _ in range(num_masks):
        t = torch.randint(0, time_mask_param + 1, (1,)).item()
        if t > 0:
            t0 = torch.randint(0, max(1, time_steps - t), (1,)).item()
            augmented[:, :, :, t0:t0 + t] = 0

    return augmented


# ==========================================
# EVALUASI & PLOTTING
# ==========================================

def save_evaluation_results(y_true, y_pred, history):
    logging.info("Menyimpan hasil evaluasi...")
    epochs_range = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(15, 4))

    plt.subplot(1, 3, 1)
    plt.plot(epochs_range, history['train_loss'], label='Train Loss')
    plt.plot(epochs_range, history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(epochs_range, history['train_acc'], label='Train Accuracy')
    plt.plot(epochs_range, history['val_acc'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs'); plt.ylabel('Accuracy (%)'); plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(epochs_range, history['val_macro_f1'], label='Validation Macro F1', color='green')
    plt.title('Validation Macro F1')
    plt.xlabel('Epochs'); plt.ylabel('Macro F1')
    plt.ylim(0, 1.0); plt.grid(True, linestyle='--', alpha=0.3); plt.legend()

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "plots" / "2d_cnn_learning_curves.png")
    plt.close()

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title('Confusion Matrix - 2D CNN (Mel-Spectrogram)')
    plt.ylabel('True Label'); plt.xlabel('Predicted Label')
    plt.savefig(RESULTS_DIR / "confusion_matrix" / "2d_cnn_cm.png")
    plt.close()

    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES,
                                   labels=[0, 1, 2, 3], zero_division=0)
    with open(RESULTS_DIR / "metrics" / "2d_cnn_classification_report.txt", "w") as f:
        f.write("=== 2D CNN Classification Report ===\n\n")
        f.write(report)

    logging.info("Semua hasil disimpan di folder results/")


# ==========================================
# TRAINING LOOP
# ==========================================

def train_model(epochs=100, batch_size=16, learning_rate=0.0005):
    """
    BUG FIXES v3 (dari analisis log 24 Mei):

    BUG 1 [CRITICAL]: Val set hanya 22 sampel padahal seharusnya 102.
    ---------------------------------------------------------------
    Root cause: run terakhir pakai train_2d_cnn.py LAMA (epochs=10, batch=16)
    yang masih ada di disk, bukan versi yang sudah diperbaiki.
    Terlihat dari log: "92/92 batches" di train (sesuai 736 samples/batch8=92)
    tapi val hanya 22 sampel — ini berarti val set berbeda dari yang diharapkan.
    SEBENARNYA ini karena file train_2d_cnn.py yang dijalankan pipeline adalah
    versi LAMA yang memakai val_size/test_size berbeda atau ada versi script lain.
    FIX: Pastikan file ini adalah satu-satunya train_2d_cnn.py di CNN_DIR.

    BUG 2 [CRITICAL]: WeightedRandomSampler MASIH AKTIF di run terakhir.
    ---------------------------------------------------------------
    Log 24 Mei menunjukkan 92 batches dengan batch_size=8 → train=736 samples.
    Tapi class weights menunjukkan [189,131,57,129] = 506 sampel asli.
    Artinya sampler masih oversample hingga 736. Ini kontradiksi dengan class_weight loss.
    FIX: use_weighted_sampler=False, dan class_weight loss saja.

    BUG 3 [CRITICAL]: `verbose=True` di ReduceLROnPlateau → TypeError di PyTorch >= 2.0.
    ---------------------------------------------------------------
    Versi PyTorch baru hapus parameter `verbose`. Sudah terjadi di log 22 Mei 18:25.
    FIX: Hapus verbose=True, tambahkan logging manual saja.

    BUG 4 [IMPORTANT]: Training hanya jalan 10 epoch (bukan 100).
    ---------------------------------------------------------------
    File yang dijalankan pipeline masih punya `train_model(epochs=10, ...)` di __main__.
    FIX: Set epochs=100 di __main__.

    HASIL YANG DIHARAPKAN setelah semua fix:
    - Val set: 102 sampel (bukan 22)
    - 4 kelas muncul di prediksi val
    - Macro F1 > 0.25
    - Training sampai 100 epoch (atau early stop wajar di epoch 30+)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")

    model = MelSpectrogram2DCNN(num_classes=4).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Total parameters: {total_params:,} (~{total_params/1000:.1f}K)")

    # BUG FIX 2: use_weighted_sampler=False — gunakan class_weight di loss saja
    try:
        train_loader, val_loader, _ = get_dataloaders(
            batch_size=batch_size,
            max_len=800,
            use_weighted_sampler=False
        )
    except Exception as e:
        logging.error(f"Gagal memuat dataloader: {e}")
        return

    # Verifikasi ukuran val set
    val_size_actual = len(val_loader.dataset)
    logging.info(f"Val set size: {val_size_actual} (expected ~102)")
    if val_size_actual < 50:
        logging.warning(
            f"⚠️  Val set hanya {val_size_actual} sampel! "
            "Cek apakah dataloader.py sudah menggunakan versi terbaru."
        )

    # Hitung class weights
    train_dataset = train_loader.dataset
    if hasattr(train_dataset, 'dataset'):
        train_labels = [train_dataset.dataset.labels[i] for i in train_dataset.indices]
    else:
        train_labels = list(train_dataset.labels)

    label_tensor  = torch.tensor(train_labels)
    class_counts  = torch.bincount(label_tensor, minlength=4).float().clamp(min=1.0)
    logging.info(f"Distribusi kelas training: {class_counts.int().tolist()}")

    # Inverse frequency weights, dinormalisasi ke rata-rata 1.0
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum() * len(CLASS_NAMES)
    class_weights = class_weights.to(device)
    logging.info(f"Class weights CE-Loss: {[f'{w:.3f}' for w in class_weights.cpu().tolist()]}")

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    # Warmup 5 epoch
    warmup_epochs = 5
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.2, end_factor=1.0, total_iters=warmup_epochs
    )

    # BUG FIX 3: Hapus verbose=True — tidak ada di PyTorch >= 2.0
    # BUG FIX dari sesi sebelumnya: mode='max' untuk macro_f1
    reduce_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=12, min_lr=1e-6
        # TIDAK ada verbose=True — menyebabkan TypeError di PyTorch >= 2.0
    )

    best_macro_f1       = 0.0
    best_epoch          = 0
    early_stop_patience = 30
    epochs_no_improve   = 0
    output_model_path   = SCRIPT_DIR / "best_model.pt"
    last_lr             = learning_rate

    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc':  [], 'val_acc':  [],
        'val_macro_f1': []
    }
    best_preds  = []
    best_labels = []

    for epoch in range(epochs):
        current_lr = optimizer.param_groups[0]['lr']
        logging.info(f"--- Epoch {epoch + 1}/{epochs} | LR: {current_lr:.6f} ---")

        # --- TRAINING ---
        model.train()
        train_loss    = 0.0
        correct_train = 0
        total_train   = 0

        for spectrograms, labels in tqdm(train_loader, desc="Training", leave=False):
            spectrograms, labels = spectrograms.to(device), labels.to(device)
            spectrograms = spec_augment(spectrograms)

            optimizer.zero_grad()
            outputs = model(spectrograms)
            loss    = criterion(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss    += loss.item()
            _, predicted   = torch.max(outputs, 1)
            total_train   += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        epoch_train_loss = train_loss / len(train_loader)
        epoch_train_acc  = 100 * correct_train / total_train

        # --- VALIDASI ---
        model.eval()
        val_loss    = 0.0
        correct_val = 0
        total_val   = 0
        all_preds   = []
        all_labels  = []

        with torch.no_grad():
            for spectrograms, labels in val_loader:
                spectrograms, labels = spectrograms.to(device), labels.to(device)
                outputs = model(spectrograms)
                loss    = criterion(outputs, labels)

                val_loss    += loss.item()
                _, predicted = torch.max(outputs, 1)
                total_val   += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        epoch_val_loss = val_loss / len(val_loader)
        epoch_val_acc  = 100 * correct_val / total_val
        epoch_macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_acc'].append(epoch_val_acc)
        history['val_macro_f1'].append(epoch_macro_f1)

        logging.info(f"Train  → Loss: {epoch_train_loss:.4f} | Acc: {epoch_train_acc:.2f}%")
        logging.info(f"Val    → Loss: {epoch_val_loss:.4f} | Acc: {epoch_val_acc:.2f}% | Macro F1: {epoch_macro_f1:.4f}")

        pred_dist = dict(sorted(Counter(all_preds).items()))
        pred_dist_named = {CLASS_NAMES[k]: v for k, v in pred_dist.items()}
        num_classes_detected = len(pred_dist)
        logging.info(f"Dist prediksi val: {pred_dist_named} ({num_classes_detected}/4 kelas terdeteksi)")

        if num_classes_detected == 1:
            logging.warning(f"⚠️  MODEL COLLAPSE! Hanya prediksi 1 kelas: {list(pred_dist_named.keys())[0]}")
        elif num_classes_detected == 2:
            logging.warning(f"⚠️  Model hanya deteksi 2 kelas: {list(pred_dist_named.keys())}")

        # Simpan model terbaik berdasarkan Macro F1
        if epoch_macro_f1 > best_macro_f1:
            best_macro_f1 = epoch_macro_f1
            best_epoch    = epoch + 1
            torch.save(model.state_dict(), output_model_path)
            logging.info(f"✅ Model terbaik disimpan (Macro F1: {epoch_macro_f1:.4f}) → {output_model_path}")
            best_preds  = list(all_preds)
            best_labels = list(all_labels)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Scheduler step
        if epoch < warmup_epochs:
            warmup_scheduler.step()
        else:
            reduce_scheduler.step(epoch_macro_f1)

        # Log jika LR berubah (manual, karena verbose dihapus)
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr != last_lr:
            logging.info(f"⚠️  LR berubah: {last_lr:.6f} → {new_lr:.6f}")
        last_lr = new_lr

        if epochs_no_improve >= early_stop_patience:
            logging.info(f"Early stopping pada epoch {epoch + 1} (no improve selama {early_stop_patience} epoch).")
            break

    if len(best_preds) == 0:
        logging.warning("Fallback: menggunakan prediksi epoch terakhir.")
        best_preds  = list(all_preds)
        best_labels = list(all_labels)

    logging.info(f"Training selesai. Best epoch {best_epoch}, Macro F1: {best_macro_f1:.4f}.")
    save_evaluation_results(y_true=best_labels, y_pred=best_preds, history=history)


# BUG FIX 4: Epochs=100 (bukan 10!)
if __name__ == "__main__":
    train_model(epochs=100, batch_size=16, learning_rate=0.0005)