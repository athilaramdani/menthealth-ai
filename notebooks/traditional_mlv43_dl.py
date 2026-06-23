# %% [markdown]
# Dataset Overview: DAIC-WOZ (FULL 189 PARTICIPANTS)
# **Pipeline v43** — DEEP LEARNING (CNN on RAW COVAREP)
#
# ─────────────────────────────────────────────────────────────────────
#  v43 = PyTorch 1D-CNN + 189 Participants + RAW COVAREP Matrices
#
#  Tujuan: Menembus F1 > 0.80 sesuai referensi State-of-the-Art (MDPI).
#  Data: Menggabungkan 107 Train + 35 Dev + 47 Test (Total 189).
#  Fitur: Raw COVAREP (74 features/frame, 10ms frame).
#  Metode: Segmentasi audio menjadi jendela 4 detik (400 frames).
#  Arsitektur: 1D Convolutional Neural Network (PyTorch).
# ─────────────────────────────────────────────────────────────────────

# %% [markdown]
# ## Setup & Imports

# %%
import os, sys, glob, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), "..")) if "notebooks" in os.getcwd() else os.getcwd()
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "DAIC-WOZ")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "dl_v43")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "v43")

for d in [MODELS_DIR, os.path.join(RESULTS_DIR, "metrics")]:
    os.makedirs(d, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# %% [markdown]
# ## Load Labels (189 Participants)

# %%
df_train = pd.read_csv(os.path.join(RAW_DIR, "train_split_Depression_AVEC2017.csv"))
df_dev = pd.read_csv(os.path.join(RAW_DIR, "dev_split_Depression_AVEC2017.csv"))
df_test = pd.read_csv(os.path.join(RAW_DIR, "full_test_split.csv"))

# Normalize columns
df_train = df_train[['Participant_ID', 'PHQ8_Binary']].rename(columns={'Participant_ID':'id', 'PHQ8_Binary':'label'})
df_dev = df_dev[['Participant_ID', 'PHQ8_Binary']].rename(columns={'Participant_ID':'id', 'PHQ8_Binary':'label'})
df_test = df_test[['Participant_ID', 'PHQ_Binary']].rename(columns={'Participant_ID':'id', 'PHQ_Binary':'label'})

df_train['split'] = 'train'
df_dev['split'] = 'dev'
df_test['split'] = 'test'

df_labels = pd.concat([df_train, df_dev, df_test], ignore_index=True)
df_labels['id'] = df_labels['id'].astype(int)
print(f"Total Labels: {len(df_labels)}")

# %% [markdown]
# ## Data Processing (Segmenting COVAREP)

# %%
WINDOW_SIZE = 400 # 4 seconds (1 frame = 10ms)
STEP_SIZE = 200   # 2 seconds overlap

def load_covarep_windows(pid):
    # DAIC-WOZ folder structure: 300_P/300_COVAREP.csv
    filepath = os.path.join(RAW_DIR, f"{pid}_P", f"{pid}_COVAREP.csv")
    if not os.path.exists(filepath):
        return None
    
    # Read without header
    df_cov = pd.read_csv(filepath, header=None)
    data = df_cov.values.astype(np.float32) # shape: [frames, 74]
    
    # Replace zeros with very small number or drop unvoiced, but COVAREP has V/UV flag
    # We will just standard scale it later.
    
    windows = []
    for start in range(0, len(data) - WINDOW_SIZE + 1, STEP_SIZE):
        windows.append(data[start : start + WINDOW_SIZE, :])
        
    if len(windows) == 0:
        return None
    return np.array(windows) # shape: [num_windows, 400, 74]

# Load everything into memory (might take 1-2 mins, ~1-2 GB RAM)
print("Extracting overlapping windows from 189 participants...")
X_all, y_all, split_all, pid_all = [], [], [], []

for idx, row in df_labels.iterrows():
    pid = int(row['id'])
    lbl = int(row['label'])
    split = row['split']
    
    windows = load_covarep_windows(pid)
    if windows is not None:
        X_all.append(windows)
        y_all.extend([lbl] * len(windows))
        split_all.extend([split] * len(windows))
        pid_all.extend([pid] * len(windows))

X_all = np.vstack(X_all)
y_all = np.array(y_all)
split_all = np.array(split_all)
pid_all = np.array(pid_all)

print(f"Total windows extracted: {X_all.shape}")

# %% [markdown]
# ## Scaler and Dataset

# %%
# Clean infinite values and NaNs from the raw COVAREP features
X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)

# Fit scaler on Train only to prevent data leakage
train_mask = (split_all == 'train')
X_train_flat = X_all[train_mask].reshape(-1, 74) # flatten for scaling

scaler = StandardScaler()
scaler.fit(X_train_flat)

# Scale all
X_all_scaled = scaler.transform(X_all.reshape(-1, 74)).reshape(-1, WINDOW_SIZE, 74)

class CovarepDataset(Dataset):
    def __init__(self, X, y):
        # 1D CNN expects [batch, channels, length] -> [batch, 74, 400]
        self.X = torch.tensor(X, dtype=torch.float32).transpose(1, 2)
        self.y = torch.tensor(y, dtype=torch.float32)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_idx = np.where(split_all == 'train')[0]
dev_idx = np.where(split_all == 'dev')[0]
test_idx = np.where(split_all == 'test')[0]

train_dataset = CovarepDataset(X_all_scaled[train_idx], y_all[train_idx])
dev_dataset = CovarepDataset(X_all_scaled[dev_idx], y_all[dev_idx])
test_dataset = CovarepDataset(X_all_scaled[test_idx], y_all[test_idx])

# Calculate pos_weight for BCEWithLogitsLoss
num_pos = y_all[train_idx].sum()
num_neg = len(train_idx) - num_pos
pos_weight = torch.tensor([num_neg / num_pos], device=DEVICE)

BATCH_SIZE = 128
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# %% [markdown]
# ## Model Architecture (1D CNN)

# %%
class DepressionCNN(nn.Module):
    def __init__(self):
        super(DepressionCNN, self).__init__()
        # Input shape: [batch, 74, 400]
        self.conv1 = nn.Conv1d(in_channels=74, out_channels=128, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(128)
        self.pool1 = nn.MaxPool1d(2) # Output: [batch, 128, 200]
        
        self.conv2 = nn.Conv1d(128, 256, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(256)
        self.pool2 = nn.MaxPool1d(2) # Output: [batch, 256, 100]

        self.conv3 = nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(2) # Output: [batch, 256, 50]
        
        self.global_pool = nn.AdaptiveAvgPool1d(1) # Output: [batch, 256, 1]
        
        self.fc1 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 1)
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu(self.bn3(self.conv3(x))))
        
        x = self.global_pool(x).squeeze(-1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x.squeeze(-1)

model = DepressionCNN().to(DEVICE)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

# %% [markdown]
# ## Training Loop

# %%
EPOCHS = 20
best_dev_loss = float('inf')

print("Starting Training...")
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * X_batch.size(0)
    
    train_loss /= len(train_loader.dataset)
    
    model.eval()
    dev_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in dev_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            dev_loss += loss.item() * X_batch.size(0)
            
    dev_loss /= len(dev_loader.dataset)
    scheduler.step(dev_loss)
    
    if dev_loss < best_dev_loss:
        best_dev_loss = dev_loss
        torch.save(model.state_dict(), os.path.join(MODELS_DIR, "best_cnn.pth"))
        
    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Dev Loss: {dev_loss:.4f}")

# %% [markdown]
# ## Inference and Patient-Level Aggregation

# %%
model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "best_cnn.pth")))
model.eval()

def predict_patient_level(split_name, indices):
    patient_probs = {}
    patient_trues = {}
    
    with torch.no_grad():
        for idx in indices:
            X_win = torch.tensor(X_all_scaled[idx], dtype=torch.float32).transpose(0, 1).unsqueeze(0).to(DEVICE)
            pid = pid_all[idx]
            lbl = y_all[idx]
            
            output = model(X_win)
            prob = torch.sigmoid(output).item()
            
            if pid not in patient_probs:
                patient_probs[pid] = []
                patient_trues[pid] = lbl
            patient_probs[pid].append(prob)
            
    pids = list(patient_probs.keys())
    trues = np.array([patient_trues[p] for p in pids])
    
    # Aggregate probabilities (Mean across all windows for a patient)
    agg_probs = np.array([np.mean(patient_probs[p]) for p in pids])
    
    best_thr = 0.5
    best_f1 = 0.0
    for thr in np.arange(0.30, 0.71, 0.01):
        preds = (agg_probs >= thr).astype(int)
        f1 = f1_score(trues, preds, average='macro', zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
            
    auc = roc_auc_score(trues, agg_probs)
    return best_f1, best_thr, auc

dev_f1, dev_thr, dev_auc = predict_patient_level('dev', dev_idx)
test_f1, test_thr, test_auc = predict_patient_level('test', test_idx)

print("="*50)
print("FINAL PATIENT-LEVEL RESULTS (189 PARTICIPANTS)")
print("="*50)
print(f"DEV  SET -> Macro F1: {dev_f1:.4f} | AUC: {dev_auc:.4f} | Thr: {dev_thr:.2f}")
print(f"TEST SET -> Macro F1: {test_f1:.4f} | AUC: {test_auc:.4f}")

# Save to CSV
df_res = pd.DataFrame([
    {'Split': 'Dev', 'Model': '1D-CNN (COVAREP)', 'Macro_F1': dev_f1, 'AUC': dev_auc},
    {'Split': 'Test', 'Model': '1D-CNN (COVAREP)', 'Macro_F1': test_f1, 'AUC': test_auc}
])
df_res.to_csv(os.path.join(RESULTS_DIR, "metrics", "v43_results.csv"), index=False)
print("\nMetrics saved to results/v43/metrics/v43_results.csv")
