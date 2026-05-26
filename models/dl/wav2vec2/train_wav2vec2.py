import os
import math
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from tqdm import tqdm
import pandas as pd
import soundfile as sf
from transformers import Wav2Vec2Processor, Wav2Vec2Model

# Import Dataloader yang sudah sukses kita validasi sebelumnya
from dataloader_raw import Wav2Vec2RawDataset

# Konfigurasi Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

CLASS_NAMES = ["NORMAL", "STRES", "CEMAS", "DEPRESI"]
CLASS_TO_IDX = {n: i for i, n in enumerate(CLASS_NAMES)}

@dataclass
class TrainConfig:
    raw_root: Path = Path(r"C:\Users\raiha\OneDrive\Documents\GitHub\menthealth-ai\data\raw").resolve()
    label_csv: Path = Path(r"C:\Users\raiha\OneDrive\Documents\GitHub\menthealth-ai\data\splits\custom_4class_labels.csv").resolve()

    pretrained_name: str = "facebook/wav2vec2-base-960h"

    sample_rate: int = 16000
    max_duration_seconds: float = 8.0

    batch_size: int = 4
    epochs: int = 5
    lr: float = 1e-5
    weight_decay: float = 0.01

    # Patient-wise split untuk mencegah kebocoran data (Anti-Leakage Protection)
    val_ratio: float = 0.15
    test_ratio: float = 0.10
    random_state: int = 42

    fine_tune_encoder: bool = True
    max_grad_norm: float = 1.0

    results_dir: Path = Path(r"C:\Users\raiha\OneDrive\Documents\GitHub\menthealth-ai\results").resolve()
    output_best_path: Path = Path("best_model.pt")

class Wav2Vec2ForAudioClassification(nn.Module):
    def __init__(self, wav2vec2_model: Wav2Vec2Model, num_classes: int = 4, dropout: float = 0.2):
        super().__init__()
        self.wav2vec2 = wav2vec2_model
        hidden_size = wav2vec2_model.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_values, attention_mask=None):
        outputs = self.wav2vec2(input_values=input_values, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # Dimensi: [B, T_h, H]

        if attention_mask is not None:
            # Masked Mean Pooling untuk menyelaraskan dimensi waktu downsampled
            B, T_h, H = hidden_states.shape
            T_in = attention_mask.shape[1]
            if T_h != T_in:
                mask = torch.nn.functional.interpolate(
                    attention_mask.unsqueeze(1).float(),
                    size=T_h,
                    mode="nearest",
                ).squeeze(1).to(hidden_states.dtype)
            else:
                mask = attention_mask.to(hidden_states.dtype)

            mask = mask.unsqueeze(-1)  # Dimensi: [B, T_h, 1]
            hidden_states = hidden_states * mask
            denom = mask.sum(dim=1).clamp(min=1.0)
            pooled = hidden_states.sum(dim=1) / denom
        else:
            pooled = hidden_states.mean(dim=1)

        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits

def patient_wise_split(dataset: Wav2Vec2RawDataset, val_ratio: float, test_ratio: float, seed: int = 42):
    """Memisahkan data berdasarkan ID Partisipan agar tidak ada kebocoran informasi."""
    rng = np.random.default_rng(seed)

    patient_to_indices: Dict[str, List[int]] = {}
    for i, s in enumerate(dataset.samples):
        name = Path(s.audio_path).parent.name  # Mengambil folder parent (cth: '302_P')
        pid = name.split("_")[0] if "_" in name else name
        patient_to_indices.setdefault(pid, []).append(i)

    patients = sorted(patient_to_indices.keys())
    patient_labels = []
    for pid in patients:
        idx0 = patient_to_indices[pid][0]
        patient_labels.append(int(dataset.samples[idx0].label_idx))

    labels_arr = np.array(patient_labels)
    class_to_patients = {c: [p for p, y in zip(patients, labels_arr) if y == c] for c in np.unique(labels_arr)}

    def split_grouped(ratio, remaining_patients):
        selected = []
        for c, plist in class_to_patients.items():
            plist = [p for p in plist if p in remaining_patients]
            n = len(plist)
            k = int(round(n * ratio))
            rng.shuffle(plist)
            selected.extend(plist[:k])
        return selected

    all_set = set(patients)
    test_patients = split_grouped(test_ratio, all_set)
    trainval_patients = sorted(list(all_set - set(test_patients)))

    val_patients = split_grouped(val_ratio / (1 - test_ratio), set(trainval_patients))
    train_patients = sorted(list(set(trainval_patients) - set(val_patients)))

    def indices_from_patients(pids: List[str]):
        idxs = []
        for pid in pids:
            idxs.extend(patient_to_indices[pid])
        return sorted(idxs)

    train_idx = indices_from_patients(train_patients)
    val_idx = indices_from_patients(val_patients)
    test_idx = indices_from_patients(sorted(list(all_set - set(train_patients) - set(val_patients))))

    return train_idx, val_idx, test_idx

def collate_fn_builder(processor: Wav2Vec2Processor):
    def collate_fn(batch: List[Dict[str, Any]]):
        waveforms = []
        for b in batch:
            w = b["waveform"]
            if isinstance(w, torch.Tensor):
                w = w.detach().cpu().get_device() if w.is_cuda else w.float()
                while w.dim() > 1:
                    w = w.squeeze(0)
                waveforms.append(w.numpy())
            else:
                w = np.asarray(w, dtype=np.float32)
                w = np.squeeze(w)
                waveforms.append(w)

        labels = torch.stack([b["label"] for b in batch])
        inputs = processor(
            waveforms,
            sampling_rate=processor.feature_extractor.sampling_rate,
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        inputs["labels"] = labels
        return inputs
    return collate_fn

def save_confusion_and_report(y_true, y_pred, out_dir: Path, prefix: str = "wav2vec2"):
    out_dir.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])

    import matplotlib.pyplot as plt
    import seaborn as sns

    (out_dir / "confusion_matrix").mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics").mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title(f"Confusion Matrix - {prefix}")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix" / f"{prefix}_cm.png")
    plt.close()

    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, labels=[0, 1, 2, 3], zero_division=0)
    with open(out_dir / "metrics" / f"{prefix}_classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)


def save_wav2vec2_learning_curves(history: Dict[str, List[float]], results_dir: Path, prefix: str = "wav2vec2"):
    """Simpan plot learning curves untuk wav2vec2."""
    import matplotlib.pyplot as plt

    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    epochs_range = range(1, len(history.get("train_loss", [])) + 1)

    plt.figure(figsize=(15, 5))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history.get("train_loss", []), label="Train Loss")
    plt.plot(epochs_range, history.get("val_loss", []), label="Val Loss")
    plt.title("Wav2Vec2 Training & Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)

    # Macro F1
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history.get("train_f1", []), label="Train Macro F1")
    plt.plot(epochs_range, history.get("val_f1", []), label="Val Macro F1", color="green")
    plt.title("Wav2Vec2 Training & Validation Macro F1")
    plt.xlabel("Epochs")
    plt.ylabel("Macro F1")
    plt.ylim(0, 1.0)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()

    plt.tight_layout()
    out_path = plots_dir / f"{prefix}_learning_curves.png"
    plt.savefig(out_path)
    plt.close()


def main():
    cfg = TrainConfig()
    script_dir = Path(__file__).resolve().parent
    cfg.output_best_path = script_dir / "best_model.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"🚀 Menggunakan Device: {device}")

    logging.info("⏳ Memuat Wav2Vec2 Processor & Pretrained Model...")
    processor = Wav2Vec2Processor.from_pretrained(cfg.pretrained_name)
    wav2vec2 = Wav2Vec2Model.from_pretrained(cfg.pretrained_name)

    model = Wav2Vec2ForAudioClassification(wav2vec2, num_classes=4).to(device)

    # 1. Inisialisasi Dataset Terlebih Dahulu (FIXED: Menghindari NameError)
    logging.info("📦 Membuka dan menganalisis dataset lokal...")
    train_dataset = Wav2Vec2RawDataset(
        raw_root=cfg.raw_root,
        label_csv=cfg.label_csv,
        max_duration_seconds=cfg.max_duration_seconds,
        sample_rate=cfg.sample_rate,
        limit_samples=None,
    )

    # 2. Guard Check: Validasi jangkauan indeks kelas
    if len(train_dataset) > 0:
        bad_labels = [s.label_idx for s in train_dataset.samples if not (0 <= s.label_idx < 4)]
        if len(bad_labels) > 0:
            logging.error(f"❌ Terjadi kesalahan target label luar batas (0-3): {set(bad_labels)}")
            return

    if not cfg.fine_tune_encoder:
        for p in model.wav2vec2.parameters():
            p.requires_grad = False

    # Split dataset anti-leakage
    train_idx, val_idx, test_idx = patient_wise_split(
        dataset=train_dataset,
        val_ratio=cfg.val_ratio,
        test_ratio=cfg.test_ratio,
        seed=cfg.random_state,
    )

    train_ds = Subset(train_dataset, train_idx)
    val_ds = Subset(train_dataset, val_idx)
    test_ds = Subset(train_dataset, test_idx)

    collate_fn = collate_fn_builder(processor)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn)

    # Hitung bobot penyeimbang kelas (Class Weights)
    class_counts = torch.zeros(4)
    for idx in train_idx:
        class_counts[train_dataset.samples[idx].label_idx] += 1
    class_counts = class_counts.clamp(min=1.0)
    class_weights = (1.0 / class_counts)
    class_weights = class_weights / class_weights.sum() * 4
    class_weights = class_weights.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_macro_f1 = -1.0
    best_state = None
    epochs_no_improve = 0
    patience = 5

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_f1": [],
        "val_f1": [],
    }


    logging.info("🏋️ Memulai Proses Training Pipeline (4 Kelas)...")
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_loss = 0.0
        all_preds_train = []
        all_labels_train = []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.epochs} [Train]", leave=False):
            input_values = batch["input_values"].to(device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            logits = model(input_values=input_values, attention_mask=attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()

            train_loss += loss.item()
            preds = logits.argmax(dim=-1)
            all_preds_train.extend(preds.detach().cpu().numpy().tolist())
            all_labels_train.extend(labels.detach().cpu().numpy().tolist())

        train_macro_f1 = f1_score(all_labels_train, all_preds_train, average="macro", zero_division=0)

        # Evaluasi Validasi
        model.eval()
        val_loss = 0.0
        all_preds_val = []
        all_labels_val = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch}/{cfg.epochs} [Val]", leave=False):
                input_values = batch["input_values"].to(device)
                attention_mask = batch.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)
                labels = batch["labels"].to(device)

                logits = model(input_values=input_values, attention_mask=attention_mask)
                loss = criterion(logits, labels)

                val_loss += loss.item()
                preds = logits.argmax(dim=-1)
                all_preds_val.extend(preds.detach().cpu().numpy().tolist())
                all_labels_val.extend(labels.detach().cpu().numpy().tolist())

        val_macro_f1 = f1_score(all_labels_val, all_preds_val, average="macro", zero_division=0)

        # Simpan history untuk learning curves
        history["train_loss"].append(train_loss / len(train_loader))
        history["val_loss"].append(val_loss / len(val_loader))
        history["train_f1"].append(train_macro_f1)
        history["val_f1"].append(val_macro_f1)

        logging.info(

            f"📊 Epoch {epoch:02d} -> train_loss={train_loss/len(train_loader):.4f} | train_f1={train_macro_f1:.4f} || "
            f"val_loss={val_loss/len(val_loader):.4f} | val_f1={val_macro_f1:.4f}"
        )

        # Early Stopping & Simpan Model Terbaik
        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            torch.save(best_state, cfg.output_best_path)
            logging.info(f"🎯 Model terbaik diperbarui & disimpan -> F1-Score: {best_macro_f1:.4f}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logging.info(f"⏹️ Early stopping aktif. Tidak ada peningkatan performa selama {patience} epoch.")
                break

    # Simpan learning curves sebelum testing (berdasarkan history per epoch)
    try:
        save_wav2vec2_learning_curves(history=history, results_dir=cfg.results_dir, prefix="wav2vec2")
        logging.info("📈 Learning curves wav2vec2 disimpan.")
    except Exception as e:
        logging.warning(f"Gagal menyimpan learning curves wav2vec2: {e}")

    # Pengujian Tahap Akhir (Testing)

    if best_state is not None:
        logging.info("🧪 Memuat model terbaik untuk pengujian final (Test Dataset)...")
        model.load_state_dict(best_state, strict=True)

    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing Final", leave=False):
            input_values = batch["input_values"].to(device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            labels = batch["labels"].to(device)

            logits = model(input_values=input_values, attention_mask=attention_mask)
            preds = logits.argmax(dim=-1)
            y_true.extend(labels.detach().cpu().numpy().tolist())
            y_pred.extend(preds.detach().cpu().numpy().tolist())

    # Generate visualisasi matriks kebingungan dan report txt
    save_confusion_and_report(y_true=y_true, y_pred=y_pred, out_dir=cfg.results_dir, prefix="wav2vec2")
    final_test_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    logging.info(f"✨ Proses selesai! Nilai Akhir Test Macro F1-Score: {final_test_f1:.4f}")

if __name__ == "__main__":
    main()