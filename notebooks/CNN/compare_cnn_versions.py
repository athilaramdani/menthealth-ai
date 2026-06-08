"""
Script untuk membandingkan hasil CNN v2, v3, dan (nanti) v4a
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

# Data comparison
data = {
    "Model": ["CNN v2", "CNN v3"],
    "CV_Mean_F1": [0.5203, 0.5222],
    "CV_Std_F1": [0.0639, 0.0446],
    "Test_F1": [0.4571, 0.3667],
    "Test_Recall_NORMAL": [0.83, 0.92],
    "Test_Recall_DEPRESI": [0.14, 0.00],  # KRITIS!
    "Test_Precision_NORMAL": [0.62, 0.61],
    "Test_Precision_DEPRESI": [0.33, 0.00],
}

df = pd.DataFrame(data)

# Create visualization
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Perbandingan CNN v2 vs v3 (DAIC-WOZ Dataset)", 
             fontsize=16, fontweight='bold')

# 1. CV Mean F1 with error bars
ax = axes[0, 0]
x = np.arange(len(df))
bars = ax.bar(x, df['CV_Mean_F1'], yerr=df['CV_Std_F1'], 
              capsize=10, color=['steelblue', 'tomato'], alpha=0.7)
ax.set_xticks(x)
ax.set_xticklabels(df['Model'])
ax.set_ylabel('Macro F1')
ax.set_title('CV Performance (Mean ± Std)')
ax.set_ylim(0, 0.7)
ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Baseline 0.5')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Add value labels
for i, (m, s) in enumerate(zip(df['CV_Mean_F1'], df['CV_Std_F1'])):
    ax.text(i, m + s + 0.02, f'{m:.4f}\n±{s:.4f}', 
            ha='center', va='bottom', fontsize=9)

# 2. Test F1
ax = axes[0, 1]
bars = ax.bar(df['Model'], df['Test_F1'], color=['steelblue', 'tomato'], alpha=0.7)
ax.set_ylabel('Test Macro F1')
ax.set_title('Test Set Performance')
ax.set_ylim(0, 0.7)
ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
ax.grid(axis='y', alpha=0.3)

for i, v in enumerate(df['Test_F1']):
    color = 'green' if v > 0.45 else 'red'
    ax.text(i, v + 0.02, f'{v:.4f}', ha='center', va='bottom', 
            fontsize=10, fontweight='bold', color=color)

# 3. Test Recall Comparison
ax = axes[0, 2]
x = np.arange(len(df))
width = 0.35
ax.bar(x - width/2, df['Test_Recall_NORMAL'], width, 
       label='NORMAL', color='steelblue', alpha=0.7)
ax.bar(x + width/2, df['Test_Recall_DEPRESI'], width, 
       label='DEPRESI', color='tomato', alpha=0.7)
ax.set_xticks(x)
ax.set_xticklabels(df['Model'])
ax.set_ylabel('Recall')
ax.set_title('Test Recall per Class')
ax.set_ylim(0, 1.0)
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Highlight DEPRESI=0.00
for i, val in enumerate(df['Test_Recall_DEPRESI']):
    if val == 0.0:
        ax.text(i + width/2, val + 0.05, '❌ 0.00\nCOLLAPSE!', 
                ha='center', va='bottom', fontsize=9, color='red', fontweight='bold')

# 4. Test Precision Comparison
ax = axes[1, 0]
x = np.arange(len(df))
ax.bar(x - width/2, df['Test_Precision_NORMAL'], width, 
       label='NORMAL', color='steelblue', alpha=0.7)
ax.bar(x + width/2, df['Test_Precision_DEPRESI'], width, 
       label='DEPRESI', color='tomato', alpha=0.7)
ax.set_xticks(x)
ax.set_xticklabels(df['Model'])
ax.set_ylabel('Precision')
ax.set_title('Test Precision per Class')
ax.set_ylim(0, 1.0)
ax.legend()
ax.grid(axis='y', alpha=0.3)

# 5. Stability (STD)
ax = axes[1, 1]
bars = ax.bar(df['Model'], df['CV_Std_F1'], color=['steelblue', 'tomato'], alpha=0.7)
ax.set_ylabel('Standard Deviation')
ax.set_title('CV Stability (Lower = Better)')
ax.set_ylim(0, 0.1)
ax.grid(axis='y', alpha=0.3)

for i, v in enumerate(df['CV_Std_F1']):
    color = 'green' if v < 0.05 else 'orange' if v < 0.07 else 'red'
    ax.text(i, v + 0.003, f'{v:.4f}', ha='center', va='bottom', 
            fontsize=10, color=color, fontweight='bold')

# 6. Summary Table
ax = axes[1, 2]
ax.axis('off')

summary_text = """
┌─────────────────────────────────────┐
│ VERDICT: CNN v2 WINS!              │
├─────────────────────────────────────┤
│ Alasan:                             │
│ • Test F1 lebih baik (0.46 vs 0.37)│
│ • Recall DEPRESI > 0 (v3 = 0!)     │
│ • Generalisasi lebih baik          │
│ • Production ready                  │
│                                     │
│ CNN v3 GAGAL TOTAL di test:        │
│ • Model collapse ke NORMAL         │
│ • Recall DEPRESI = 0.00            │
│ • Tidak dapat digunakan!           │
│                                     │
│ NEXT: Train CNN v4a                │
│ • Focal Loss + Mixup               │
│ • Target Test F1 ≥ 0.55            │
│ • Target Recall DEPRESI ≥ 0.40     │
└─────────────────────────────────────┘
"""

ax.text(0.5, 0.5, summary_text, 
        fontsize=10, 
        family='monospace',
        ha='center', va='center',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()
plt.savefig(RESULTS_DIR / "plots" / "cnn_v2_vs_v3_comparison.png", 
            dpi=150, bbox_inches='tight')
print(f"✅ Saved: {RESULTS_DIR / 'plots' / 'cnn_v2_vs_v3_comparison.png'}")

# Save comparison table
df.to_csv(RESULTS_DIR / "metrics" / "cnn_v2_v3_comparison.csv", index=False)
print(f"✅ Saved: {RESULTS_DIR / 'metrics' / 'cnn_v2_v3_comparison.csv'}")

print("\n" + "="*60)
print("SUMMARY:")
print("="*60)
print(df.to_string(index=False))
print("="*60)
print("\n🏆 WINNER: CNN v2 (Baseline)")
print("❌ LOSER:  CNN v3 (Model Collapse)")
print("🚀 NEXT:   Train CNN v4a with Focal Loss + Mixup")
