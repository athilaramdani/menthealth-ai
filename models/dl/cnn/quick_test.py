"""
Quick test untuk memverifikasi model tidak collapse ke satu kelas.
Jalankan setelah training untuk cek distribusi prediksi.
"""
import torch
import sys
from pathlib import Path
from collections import Counter

SCRIPT_DIR   = Path(__file__).resolve().parent
DL_DIR       = SCRIPT_DIR.parent
PROJECT_ROOT = DL_DIR.parent.parent

if str(DL_DIR) not in sys.path:
    sys.path.append(str(DL_DIR))

from dataloader import get_dataloaders
from train_2d_cnn import MelSpectrogram2DCNN

CLASS_NAMES = ['NORMAL', 'STRES', 'CEMAS', 'DEPRESI']

def quick_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = MelSpectrogram2DCNN(num_classes=4).to(device)
    model_path = SCRIPT_DIR / "best_model.pt"
    
    if not model_path.exists():
        print(f"❌ Model tidak ditemukan di {model_path}")
        print("   Jalankan train_2d_cnn.py terlebih dahulu!")
        return
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Load validation set
    _, val_loader, _ = get_dataloaders(batch_size=8, max_len=800)
    
    all_preds = []
    all_labels = []
    
    print("\n🔍 Testing model pada validation set...")
    
    with torch.no_grad():
        for spectrograms, labels in val_loader:
            spectrograms = spectrograms.to(device)
            outputs = model(spectrograms)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Analisis distribusi
    pred_counts = Counter(all_preds)
    label_counts = Counter(all_labels)
    
    print("\n" + "="*60)
    print("📊 DISTRIBUSI PREDIKSI vs GROUND TRUTH")
    print("="*60)
    
    print("\n Ground Truth (Val Set):")
    for i in range(4):
        count = label_counts.get(i, 0)
        print(f"   {CLASS_NAMES[i]:8s}: {count:3d} sampel")
    
    print("\n Prediksi Model:")
    for i in range(4):
        count = pred_counts.get(i, 0)
        pct = 100 * count / len(all_preds) if all_preds else 0
        print(f"   {CLASS_NAMES[i]:8s}: {count:3d} sampel ({pct:.1f}%)")
    
    # Deteksi collapse
    max_pred_pct = max(pred_counts.values()) / len(all_preds) * 100 if all_preds else 0
    unique_classes = len(pred_counts)
    
    print("\n" + "="*60)
    if max_pred_pct > 80:
        print("❌ MODEL COLLAPSE TERDETEKSI!")
        print(f"   {max_pred_pct:.1f}% prediksi ke satu kelas")
    elif unique_classes < 3:
        print(f"⚠️  MODEL HANYA PREDIKSI {unique_classes} KELAS")
        print("   Seharusnya 4 kelas")
    else:
        print("✅ MODEL TIDAK COLLAPSE")
        print(f"   Prediksi tersebar di {unique_classes} kelas")
    print("="*60 + "\n")
    
    # Hitung accuracy per kelas
    correct_per_class = {i: 0 for i in range(4)}
    total_per_class = {i: 0 for i in range(4)}
    
    for pred, true in zip(all_preds, all_labels):
        total_per_class[true] += 1
        if pred == true:
            correct_per_class[true] += 1
    
    print("📈 RECALL PER KELAS:")
    for i in range(4):
        total = total_per_class[i]
        correct = correct_per_class[i]
        recall = 100 * correct / total if total > 0 else 0
        print(f"   {CLASS_NAMES[i]:8s}: {recall:5.1f}% ({correct}/{total})")
    
    overall_acc = 100 * sum(correct_per_class.values()) / len(all_preds)
    print(f"\n🎯 Overall Accuracy: {overall_acc:.2f}%\n")

if __name__ == "__main__":
    quick_test()
