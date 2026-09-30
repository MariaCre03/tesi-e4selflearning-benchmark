import argparse
import os
import sys
import glob
import re
import json
import csv
import pickle
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

# =====================================================================
# ARCHITETTURE: BIOT, SimMTM, FEMBA (Pure PyTorch)
# =====================================================================

# --- 1. BIOT (Biosignal Transformer) ---
class SinusoidalPE(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class BIOTEncoder(nn.Module):
    def __init__(self, n_channels=6, seg_len=640, chunk_size=64, d_model=256, n_heads=8, n_layers=4, dropout=0.1):
        super().__init__()
        self.n_channels = n_channels
        self.chunk_size = chunk_size
        self.n_chunks = seg_len // chunk_size
        self.d_model = d_model
        self.patch_embed = nn.Linear(chunk_size, d_model)
        self.channel_emb = nn.Embedding(n_channels, d_model)
        self.pos_enc = SinusoidalPE(d_model, max_len=self.n_chunks * n_channels + 1)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model*4, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        B, C, T = x.shape
        x = x[:, :, :self.n_chunks * self.chunk_size].contiguous().view(B, C, self.n_chunks, self.chunk_size)
        x = self.patch_embed(x) + self.channel_emb(torch.arange(C, device=x.device)).unsqueeze(0).unsqueeze(2)
        x = x.view(B, C * self.n_chunks, self.d_model)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        return self.norm(self.transformer(self.pos_enc(x)))[:, 0, :]

class BIOTClassifier(nn.Module):
    def __init__(self, encoder, n_classes=1, dropout=0.5):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(encoder.d_model, encoder.d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(encoder.d_model // 2, n_classes)
        )
    def forward(self, x):
        return self.head(self.encoder(x))

# --- 2. SimMTM (Masked Time-Series Modeling) ---
class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, d_model):
        super().__init__()
        self.proj = nn.Linear(patch_size, d_model)
        self.norm = nn.LayerNorm(d_model)
    def forward(self, x):
        return self.norm(self.proj(x))

class SimMTMEncoder(nn.Module):
    def __init__(self, seg_len=640, patch_size=16, d_model=128, n_heads=4, n_layers=3, dropout=0.1):
        super().__init__()
        self.patch_size = patch_size
        self.d_model = d_model
        self.n_patches = seg_len // patch_size
        self.patch_emb = PatchEmbedding(patch_size, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, self.n_patches, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model*4,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

    def patchify(self, x):
        B, C, T = x.shape
        x_cut = x[:, :, :self.n_patches * self.patch_size]
        return x_cut.contiguous().view(B * C, self.n_patches, self.patch_size)

    def forward(self, x):
        emb = self.patch_emb(self.patchify(x)) + self.pos_emb
        return self.norm(self.transformer(emb))

class SimMTMClassifier(nn.Module):
    def __init__(self, encoder, n_channels=6, n_classes=1, dropout=0.3):
        super().__init__()
        self.encoder = encoder
        self.n_channels = n_channels
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(encoder.d_model * n_channels, encoder.d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(encoder.d_model, n_classes)
        )
    def forward(self, x):
        B, C, T = x.shape
        enc = self.encoder(x).mean(dim=1)
        enc = enc.view(B, self.n_channels * self.encoder.d_model)
        return self.head(enc)

# --- 3. FEMBA (Foundational Mamba Biosignals) ---
class PureMamba(nn.Module):
    def __init__(self, d_model=384, d_state=16, d_conv=4, expand=4, dt_rank=24):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = d_model * expand
        self.dt_rank = dt_rank
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, kernel_size=d_conv,
            groups=self.d_inner, padding=d_conv - 1
        )
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * self.d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        self.A_log = nn.Parameter(torch.randn(self.d_inner, self.d_state))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, u):
        B, L, _ = u.shape
        xz = self.in_proj(u)
        x, z = xz.chunk(2, dim=-1)
        x_conv = self.conv1d(x.transpose(1, 2))[:, :, :L].transpose(1, 2)
        x_act = F.silu(x_conv)
        x_dbl = self.x_proj(x_act)
        delta, B_mat, C_mat = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(delta))
        A = -torch.exp(self.A_log.float())
        deltaA = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
        deltaB = delta.unsqueeze(-1) * B_mat.unsqueeze(2)
        h = torch.zeros(B, self.d_inner, self.d_state, device=u.device, dtype=u.dtype)
        ys = []
        for t in range(L):
            h = deltaA[:, t] * h + deltaB[:, t] * x_act[:, t].unsqueeze(-1)
            y_t = (h * C_mat[:, t].unsqueeze(1)).sum(dim=-1)
            ys.append(y_t)
        y = torch.stack(ys, dim=1) + x_act * self.D.unsqueeze(0).unsqueeze(0)
        return self.out_proj(y * F.silu(z))

class MambaWrapper(nn.Module):
    def __init__(self, d_model=384):
        super().__init__()
        self.mamba_fwd = PureMamba(d_model=d_model)
        self.mamba_rev = PureMamba(d_model=d_model)
    def forward(self, x):
        fwd = self.mamba_fwd(x)
        rev = self.mamba_rev(x.flip(dims=[1])).flip(dims=[1])
        return fwd + rev

class FEMBA(nn.Module):
    def __init__(self, seq_length=640, num_channels=6, num_classes=1, embed_dim=128, num_blocks=4, dropout=0.3):
        super().__init__()
        self.patch_embed = nn.Sequential()
        self.patch_embed.add_module("proj", nn.Conv2d(1, embed_dim, kernel_size=(2, 16), stride=(2, 16)))
        self.pos_embed = nn.Parameter(torch.randn(1, 40, 384) * 0.02)
        self.mamba_blocks = nn.ModuleList([MambaWrapper(d_model=384) for _ in range(num_blocks)])
        self.norm_layers = nn.ModuleList([nn.LayerNorm(384) for _ in range(num_blocks)])
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(384, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        B, C, L = x.shape
        x_2d = x.unsqueeze(1)
        feat = self.patch_embed(x_2d)
        B, D, H, W = feat.shape
        feat = feat.view(B, D * H, W).permute(0, 2, 1)
        feat = feat + self.pos_embed
        for block, norm in zip(self.mamba_blocks, self.norm_layers):
            res = feat
            feat = norm(block(feat)) + res
        pooled = feat.mean(dim=1)
        return self.head(pooled)

# =====================================================================
# DATASET PYTORCH (Resampling 640 canali)
# =====================================================================
class WearableStressDataset(Dataset):
    def __init__(self, file_paths, labels, target_len=640):
        self.file_paths = file_paths
        self.labels = labels
        self.target_len = target_len
        self.channel_keys = ['ACC_x', 'ACC_y', 'ACC_z', 'BVP', 'EDA', 'TEMP']

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        data_dict = {}
        if path.endswith('.h5'):
            with h5py.File(path, 'r') as hf:
                for ch in self.channel_keys:
                    if ch in hf:
                        data_dict[ch] = torch.from_numpy(hf[ch][:]).float()
        else:
            with open(path, "rb") as f:
                obj = pickle.load(f)
            d = obj["data"] if isinstance(obj, dict) and "data" in obj else obj
            for ch in self.channel_keys:
                if ch in d:
                    val = d[ch]
                    data_dict[ch] = torch.tensor(val, dtype=torch.float32) if not isinstance(val, torch.Tensor) else val.float()

        tensors = []
        for ch in self.channel_keys:
            raw_val = data_dict.get(ch, torch.zeros(self.target_len))
            if raw_val.ndim == 1:
                raw_val = raw_val.unsqueeze(0).unsqueeze(0)
            elif raw_val.ndim == 2:
                raw_val = raw_val.unsqueeze(0)
            resampled = F.interpolate(raw_val, size=self.target_len, mode="linear", align_corners=False).squeeze()
            mean = torch.mean(resampled)
            std = torch.std(resampled) + 1e-6
            tensors.append((resampled - mean) / std)

        x_stacked = torch.stack(tensors, dim=0)
        y_val = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x_stacked, y_val

# =====================================================================
# RUNNER FOLD CON LOGGING DELLE EPOCHE
# =====================================================================
def train_and_eval_fold(model, loader_train, loader_test, epochs, lr, device, pos_weight=None, fold_name=""):
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    epoch_records = []

    for ep in range(1, epochs + 1):
        model.train()
        train_losses = []
        tr_y_true, tr_y_pred = [], []
        for xb, yb in loader_train:
            xb = xb.to(device)
            yb = yb.to(device).unsqueeze(1)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            preds = (torch.sigmoid(out) >= 0.5).long().cpu().numpy().flatten()
            tr_y_pred.extend(preds)
            tr_y_true.extend(yb.cpu().numpy().flatten())

        tr_loss_mean = np.mean(train_losses)
        tr_acc = accuracy_score(tr_y_true, tr_y_pred)
        tr_f1 = f1_score(tr_y_true, tr_y_pred, zero_division=0)

        # Valutazione test ad ogni epoca
        model.eval()
        val_losses = []
        val_y_true, val_y_pred = [], []
        with torch.no_grad():
            for xb, yb in loader_test:
                xb = xb.to(device)
                yb_gpu = yb.to(device).unsqueeze(1)
                out = model(xb)
                loss = criterion(out, yb_gpu)
                val_losses.append(loss.item())
                preds = (torch.sigmoid(out) >= 0.5).long().cpu().numpy().flatten()
                val_y_pred.extend(preds)
                val_y_true.extend(yb.numpy().flatten())

        val_loss_mean = np.mean(val_losses)
        val_acc = accuracy_score(val_y_true, val_y_pred)
        val_f1 = f1_score(val_y_true, val_y_pred, zero_division=0)

        epoch_records.append({
            "fold": fold_name,
            "epoch": ep,
            "train_loss": round(tr_loss_mean, 4),
            "train_acc": round(tr_acc * 100, 2),
            "train_f1_stress": round(tr_f1 * 100, 2),
            "val_loss": round(val_loss_mean, 4),
            "val_acc": round(val_acc * 100, 2),
            "val_f1_stress": round(val_f1 * 100, 2),
        })

    # Metriche finali sul fold
    f1_m = f1_score(val_y_true, val_y_pred, average="macro", zero_division=0)
    prec = precision_score(val_y_true, val_y_pred, zero_division=0)
    rec = recall_score(val_y_true, val_y_pred, zero_division=0)
    return val_y_true, val_y_pred, val_acc, val_f1, f1_m, prec, rec, epoch_records

# =====================================================================
# CARICAMENTO DEI PESI PRE-TRAINED
# =====================================================================
def load_weights(model_type, model, weight_path, device):
    if not weight_path or not os.path.exists(weight_path):
        print(f"[AVVISO] Nessun checkpoint pre-trained caricato per {model_type} (partenza random)")
        return
    print(f"[INFO] Caricamento pesi pre-trained da: {weight_path}")
    ckpt = torch.load(weight_path, map_location=device)
    sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    
    if model_type == "biot":
        encoder_sd = {k.replace("encoder.", ""): v for k, v in sd.items()}
        missing, unexpected = model.encoder.load_state_dict(encoder_sd, strict=False)
        print(f"  -> BIOT Encoder caricato. (Missing: {len(missing)}, Unexpected: {len(unexpected)})")
    elif model_type == "simmtm":
        encoder_sd = {k.replace("encoder.", ""): v for k, v in sd.items()}
        missing, unexpected = model.encoder.load_state_dict(encoder_sd, strict=False)
        print(f"  -> SimMTM Encoder caricato. (Missing: {len(missing)}, Unexpected: {len(unexpected)})")
    elif model_type == "femba":
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"  -> FEMBA caricato. (Missing: {len(missing)}, Unexpected: {len(unexpected)})")

def build_model(model_type):
    if model_type == "biot":
        enc = BIOTEncoder(n_channels=6, seg_len=640, chunk_size=64, d_model=256, n_heads=8, n_layers=4)
        return BIOTClassifier(enc, n_classes=1, dropout=0.5)
    elif model_type == "simmtm":
        enc = SimMTMEncoder(seg_len=640, patch_size=16, d_model=128, n_heads=4, n_layers=3)
        return SimMTMClassifier(enc, n_channels=6, n_classes=1, dropout=0.3)
    elif model_type == "femba":
        return FEMBA(seq_length=640, num_channels=6, num_classes=1, embed_dim=128, num_blocks=4, dropout=0.3)
    else:
        raise ValueError(f"Modello non supportato: {model_type}")

# =====================================================================
# FUNZIONE PRINCIPALE DI BENCHMARK
# =====================================================================
def run_benchmark(model_name, dataset_name, data_dir, weight_path, epochs, lr, batch_size, out_root, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    exp_tag = f"{model_name.upper()}_{dataset_name.upper()}_LOSOCV"
    exp_dir = os.path.join(out_root, f"{model_name}_{dataset_name}")
    os.makedirs(exp_dir, exist_ok=True)

    print("\n" + "="*80)
    print(f" AVVIO BENCHMARK: {exp_tag}")
    print(f" Dispositivo: {device} | Epoche: {epochs} | LR: {lr} | Output: {exp_dir}")
    print("="*80)

    # Carica metadati
    meta_p = os.path.join(data_dir, "metadata.pkl")
    with open(meta_p, "rb") as f:
        meta = pickle.load(f)

    corrected_paths = []
    for p in meta["sessions_paths"]:
        parts = p.replace("\\", "/").split("/")
        candidate = os.path.join(data_dir, parts[-2], parts[-1])
        if not os.path.exists(candidate):
            candidate = p
        corrected_paths.append(candidate)
    all_paths = np.array(corrected_paths)

    labels_dict = meta["sessions_labels"]
    lbl_key = next((k for k in ["status", "label", "binary_stress", "stress"] if k in labels_dict), list(labels_dict.keys())[0])
    raw_labels = np.array(labels_dict[lbl_key], dtype=np.float32)
    uniq = set(np.unique(raw_labels))
    if uniq.issubset({0.0, 1.0}):
        all_labels = raw_labels
    elif 2.0 in uniq and 0.0 not in uniq:
        all_labels = np.array([1.0 if y == 2.0 else 0.0 for y in raw_labels], dtype=np.float32)
    else:
        all_labels = np.array([1.0 if y == 1.0 else 0.0 for y in raw_labels], dtype=np.float32)

    # Riconoscimento soggetti
    if "wesad" in dataset_name.lower():
        all_subs = np.array([re.search(r'[sS]\d+', p).group(0).upper() if re.search(r'[sS]\d+', p) else "S0" for p in all_paths])
        unique_subs = sorted(list(np.unique(all_subs)), key=lambda x: int(re.search(r'\d+', x).group(0)) if re.search(r'\d+', x) else 0)
    else:
        all_subs = np.array([p.replace("\\", "/").split("/")[-2].split("_")[0] for p in all_paths])
        unique_subs = sorted(list(np.unique(all_subs)))

    n_folds = len(unique_subs)
    print(f"[INFO] Trovati {len(all_paths)} campioni su {n_folds} soggetti (LOSOCV): {unique_subs}")

    all_epoch_records = []
    fold_summary = []
    global_y_true, global_y_pred = [], []

    for idx, test_sub in enumerate(unique_subs, 1):
        fold_tag = f"Fold_{idx:02d}_{test_sub}"
        train_mask = (all_subs != test_sub)
        test_mask = (all_subs == test_sub)

        ds_train = WearableStressDataset(all_paths[train_mask], all_labels[train_mask])
        ds_test = WearableStressDataset(all_paths[test_mask], all_labels[test_mask])

        loader_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
        loader_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False)

        # Calcolo class weight
        tr_targets = all_labels[train_mask]
        n_pos = sum(1 for y in tr_targets if y == 1.0)
        n_neg = sum(1 for y in tr_targets if y == 0.0)
        pos_weight = torch.tensor([n_neg / max(1, n_pos)], device=device, dtype=torch.float32)

        # Inizializza modello pulito + carica pesi pre-trained per ogni fold
        model = build_model(model_name).to(device)
        load_weights(model_name, model, weight_path, device)

        y_true, y_pred, acc, f1, f1_m, prec, rec, ep_recs = train_and_eval_fold(
            model, loader_train, loader_test, epochs, lr, device, pos_weight=pos_weight, fold_name=fold_tag
        )

        all_epoch_records.extend(ep_recs)
        global_y_true.extend(y_true)
        global_y_pred.extend(y_pred)

        fold_res = {
            "fold": idx, "subject": test_sub, "samples": len(y_true),
            "accuracy": round(acc * 100, 2),
            "f1_stress": round(f1 * 100, 2),
            "macro_f1": round(f1_m * 100, 2),
            "precision": round(prec * 100, 2),
            "recall": round(rec * 100, 2)
        }
        fold_summary.append(fold_res)
        print(f" -> [{idx:02d}/{n_folds}] Soggetto {test_sub:>3s} ({len(y_true):>2d} campioni) | Acc: {acc*100:5.2f}% | F1-Stress: {f1*100:5.2f}% | Macro-F1: {f1_m*100:5.2f}%")

    # SALVATAGGIO 1: Storia completa di tutte le epoche (CSV)
    csv_path = os.path.join(exp_dir, "epoch_history.csv")
    if all_epoch_records:
        keys = all_epoch_records[0].keys()
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(all_epoch_records)
    print(f"\n[OK] Storico epoche salvato in: {csv_path}")

    # SALVATAGGIO 2: Risultati dettagliati per ciascun soggetto (JSON per statistica)
    json_path = os.path.join(exp_dir, "losocv_folds.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(fold_summary, f, indent=2)
    print(f"[OK] Metriche fold-by-fold salvate in: {json_path}")

    # SALVATAGGIO 3: Report Finale (TXT)
    accs = [r["accuracy"] for r in fold_summary]
    f1s = [r["f1_stress"] for r in fold_summary]
    f1ms = [r["macro_f1"] for r in fold_summary]
    precs = [r["precision"] for r in fold_summary]
    recs = [r["recall"] for r in fold_summary]

    cm = confusion_matrix(global_y_true, global_y_pred)
    glob_acc = accuracy_score(global_y_true, global_y_pred)
    glob_f1 = f1_score(global_y_true, global_y_pred, zero_division=0)
    glob_f1m = f1_score(global_y_true, global_y_pred, average="macro", zero_division=0)

    report_lines = [
        "="*70,
        f"       REPORT FINALE LOSOCV: {exp_tag}",
        "="*70,
        f" Modello: {model_name.upper()} | Dataset: {dataset_name.upper()} ({n_folds} Soggetti)",
        f" Pretrained Weights: {weight_path}",
        "-"*70,
        f" Accuratezza Media (Fold) : {np.mean(accs):6.2f}% +- {np.std(accs):5.2f}%",
        f" F1-Stress Medio (Fold)   : {np.mean(f1s):6.2f}% +- {np.std(f1s):5.2f}%",
        f" Macro-F1 Medio (Fold)    : {np.mean(f1ms):6.2f}% +- {np.std(f1ms):5.2f}%",
        f" Richiamo Medio (Fold)    : {np.mean(recs):6.2f}% +- {np.std(recs):5.2f}%",
        f" Precisione Media (Fold)  : {np.mean(precs):6.2f}% +- {np.std(precs):5.2f}%",
        "-"*70,
        f" POOLED (Globale su tutti i {len(global_y_true)} campioni):",
        f"  Acc: {glob_acc*100:.2f}% | F1-Stress: {glob_f1*100:.2f}% | Macro-F1: {glob_f1m*100:.2f}%",
        "-"*70,
        " MATRICE DI CONFUSIONE POOLED:",
        str(cm),
        "="*70
    ]
    report_txt = "\n".join(report_lines)
    txt_path = os.path.join(exp_dir, "report_finale.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(report_txt)
    print("\n" + report_txt)
    print(f"\n[OK] Report completo salvato in: {txt_path}")

def main():
    parser = argparse.ArgumentParser(description="Esegui Benchmark LOSOCV con salvataggio completo delle epoche")
    parser.add_argument("--model", type=str, default="biot", choices=["biot", "simmtm", "femba", "all"])
    parser.add_argument("--dataset", type=str, default="wesad", choices=["wesad", "hosseini", "all"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--out_dir", type=str, default="risultati_benchmark")
    args = parser.parse_args()

    models = ["biot", "simmtm", "femba"] if args.model == "all" else [args.model]
    datasets = ["wesad", "hosseini"] if args.dataset == "all" else [args.dataset]

    # Mappatura percorsi pesi standard sul server
    weight_map = {
        "biot": "biot_encoder_pretrained.pt",
        "simmtm": "simmtm_encoder_pretrained.pt",
        "femba": "femba_encoder_pretrained.pt"
    }

    data_map = {
        "wesad": r"downstream_data\wesad_segmented",
        "hosseini": r"downstream_data\hosseini_segmented"
    }

    for m in models:
        for d in datasets:
            w_path = weight_map.get(m, "")
            d_path = data_map.get(d, "")
            if not os.path.exists(d_path):
                # Fallback path alternativo
                d_path = os.path.join("downstream_data", f"{d}_segmented")
            run_benchmark(
                model_name=m,
                dataset_name=d,
                data_dir=d_path,
                weight_path=w_path,
                epochs=args.epochs,
                lr=args.lr,
                batch_size=args.batch_size,
                out_root=args.out_dir
            )

if __name__ == "__main__":
    main()
