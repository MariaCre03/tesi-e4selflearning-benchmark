import argparse
import os
import sys
import glob
import re
import json
import csv
import pickle
import h5py
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

# =====================================================================
# 1. DEFINIZIONE ARCHITETTURE (BIOT, SimMTM, FEMBA)
# =====================================================================

# --- BIOT ---
class SinusoidalPE(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x): return x + self.pe[:, :x.size(1)]

class BIOTEncoder(nn.Module):
    def __init__(self, n_channels=6, seg_len=640, chunk_size=64, d_model=256, n_heads=8, n_layers=4, dropout=0.1):
        super().__init__()
        self.n_channels, self.chunk_size, self.d_model = n_channels, chunk_size, d_model
        self.n_chunks = seg_len // chunk_size
        self.patch_embed = nn.Linear(chunk_size, d_model)
        self.channel_emb = nn.Embedding(n_channels, d_model)
        self.pos_enc = SinusoidalPE(d_model, max_len=self.n_chunks * n_channels + 1)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_model*4, dropout=dropout, batch_first=True)
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
            nn.Dropout(dropout), nn.Linear(encoder.d_model, encoder.d_model // 2),
            nn.ReLU(), nn.Dropout(dropout), nn.Linear(encoder.d_model // 2, n_classes)
        )
    def forward(self, x): return self.head(self.encoder(x))

# --- SimMTM ---
class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, d_model):
        super().__init__()
        self.proj = nn.Linear(patch_size, d_model)
        self.norm = nn.LayerNorm(d_model)
    def forward(self, x): return self.norm(self.proj(x))

class SimMTMEncoder(nn.Module):
    def __init__(self, seg_len=640, patch_size=16, d_model=128, n_heads=4, n_layers=3, dropout=0.1):
        super().__init__()
        self.patch_size, self.d_model = patch_size, d_model
        self.n_patches = seg_len // patch_size
        self.patch_emb = PatchEmbedding(patch_size, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, self.n_patches, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_model*4, dropout=dropout, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        B, C, T = x.shape
        x_cut = x[:, :, :self.n_patches * self.patch_size].contiguous().view(B * C, self.n_patches, self.patch_size)
        emb = self.patch_emb(x_cut) + self.pos_emb
        return self.norm(self.transformer(emb))

class SimMTMClassifier(nn.Module):
    def __init__(self, encoder, n_channels=6, n_classes=1, dropout=0.3):
        super().__init__()
        self.encoder, self.n_channels = encoder, n_channels
        self.head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(encoder.d_model * n_channels, encoder.d_model),
            nn.ReLU(), nn.Dropout(dropout), nn.Linear(encoder.d_model, n_classes)
        )
    def forward(self, x):
        B, C, T = x.shape
        enc = self.encoder(x).mean(dim=1).view(B, self.n_channels * self.encoder.d_model)
        return self.head(enc)

# --- FEMBA ---
class PureMamba(nn.Module):
    def __init__(self, d_model=384, d_state=16, d_conv=4, expand=4, dt_rank=24):
        super().__init__()
        self.d_model, self.d_state, self.d_conv, self.expand, self.dt_rank = d_model, d_state, d_conv, expand, dt_rank
        self.d_inner = d_model * expand
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv, groups=self.d_inner, padding=d_conv - 1)
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
            ys.append((h * C_mat[:, t].unsqueeze(1)).sum(dim=-1))
        y = torch.stack(ys, dim=1) + x_act * self.D.unsqueeze(0).unsqueeze(0)
        return self.out_proj(y * F.silu(z))

class MambaWrapper(nn.Module):
    def __init__(self, d_model=384):
        super().__init__()
        self.mamba_fwd = PureMamba(d_model=d_model)
        self.mamba_rev = PureMamba(d_model=d_model)
    def forward(self, x):
        return self.mamba_fwd(x) + self.mamba_rev(x.flip(dims=[1])).flip(dims=[1])

class FEMBA(nn.Module):
    def __init__(self, seq_length=640, num_channels=6, num_classes=1, embed_dim=128, num_blocks=4, dropout=0.3):
        super().__init__()
        self.patch_embed = nn.Sequential()
        self.patch_embed.add_module("proj", nn.Conv2d(1, embed_dim, kernel_size=(2, 16), stride=(2, 16)))
        self.pos_embed = nn.Parameter(torch.randn(1, 40, 384) * 0.02)
        self.mamba_blocks = nn.ModuleList([MambaWrapper(d_model=384) for _ in range(num_blocks)])
        self.norm_layers = nn.ModuleList([nn.LayerNorm(384) for _ in range(num_blocks)])
        self.head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(384, 128), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(128, num_classes)
        )

    def forward(self, x):
        feat = self.patch_embed(x.unsqueeze(1))
        B, D, H, W = feat.shape
        feat = feat.view(B, D * H, W).permute(0, 2, 1) + self.pos_embed
        for block, norm in zip(self.mamba_blocks, self.norm_layers):
            feat = norm(block(feat)) + feat
        return self.head(feat.mean(dim=1))

# =====================================================================
# 2. DATASET PYTORCH (Resampling a 640 canali)
# =====================================================================
class WearableDataset(Dataset):
    def __init__(self, file_paths, labels, target_len=640):
        self.file_paths = file_paths
        self.labels = labels
        self.target_len = target_len
        self.channels = ['ACC_x', 'ACC_y', 'ACC_z', 'BVP', 'EDA', 'TEMP']

    def __len__(self): return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        data_dict = {}
        if path.endswith('.h5'):
            with h5py.File(path, 'r') as hf:
                for ch in self.channels:
                    if ch in hf: data_dict[ch] = torch.from_numpy(hf[ch][:]).float()
        else:
            with open(path, "rb") as f:
                obj = pickle.load(f)
            d = obj["data"] if isinstance(obj, dict) and "data" in obj else obj
            for ch in self.channels:
                if ch in d:
                    v = d[ch]
                    data_dict[ch] = torch.tensor(v, dtype=torch.float32) if not isinstance(v, torch.Tensor) else v.float()

        tensors = []
        for ch in self.channels:
            raw = data_dict.get(ch, torch.zeros(self.target_len))
            if raw.ndim == 1: raw = raw.unsqueeze(0).unsqueeze(0)
            elif raw.ndim == 2: raw = raw.unsqueeze(0)
            res = F.interpolate(raw, size=self.target_len, mode="linear", align_corners=False).squeeze()
            mean, std = torch.mean(res), torch.std(res) + 1e-6
            tensors.append((res - mean) / std)
        return torch.stack(tensors, dim=0), torch.tensor(self.labels[idx], dtype=torch.float32)

# =====================================================================
# 3. RUNNER CON SALVATAGGIO INCREMENTALE (AD OGNI SINGOLO FOLD!)
# =====================================================================
def save_incremental(exp_dir, all_epoch_records, fold_summary, global_y_true, global_y_pred, exp_tag, model_name, dataset_name, weight_path):
    # 1. Salva Epoch History CSV
    csv_path = os.path.join(exp_dir, "epoch_history.csv")
    if all_epoch_records:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_epoch_records[0].keys())
            writer.writeheader()
            writer.writerows(all_epoch_records)

    # 2. Salva Folds JSON
    json_path = os.path.join(exp_dir, "losocv_folds.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(fold_summary, f, indent=2)

    # 3. Salva Report TXT intermedio/finale
    if fold_summary:
        accs = [r["accuracy"] for r in fold_summary]
        f1s = [r["f1_stress"] for r in fold_summary]
        f1ms = [r["macro_f1"] for r in fold_summary]
        cm = confusion_matrix(global_y_true, global_y_pred)
        glob_acc = accuracy_score(global_y_true, global_y_pred)
        glob_f1 = f1_score(global_y_true, global_y_pred, zero_division=0)
        glob_f1m = f1_score(global_y_true, global_y_pred, average="macro", zero_division=0)

        report = [
            "="*70,
            f" REPORT LOSOCV: {exp_tag} (Folds completati: {len(fold_summary)})",
            "="*70,
            f" Modello: {model_name.upper()} | Dataset: {dataset_name.upper()}",
            f" Pesi pre-trained: {weight_path}",
            "-"*70,
            f" Accuratezza Media : {np.mean(accs):6.2f}% +- {np.std(accs):5.2f}%",
            f" F1-Stress Medio   : {np.mean(f1s):6.2f}% +- {np.std(f1s):5.2f}%",
            f" Macro-F1 Medio    : {np.mean(f1ms):6.2f}% +- {np.std(f1ms):5.2f}%",
            "-"*70,
            f" POOLED ({len(global_y_true)} campioni): Acc={glob_acc*100:.2f}% | F1-Stress={glob_f1*100:.2f}% | Macro-F1={glob_f1m*100:.2f}%",
            "-"*70,
            " MATRICE DI CONFUSIONE:", str(cm),
            "="*70
        ]
        with open(os.path.join(exp_dir, "report_finale.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(report))

def run_experiment(model_name, dataset_name, data_dir, weight_path, epochs=20, lr=0.0001, batch_size=16, out_root="risultati_benchmark"):
    exp_tag = f"{model_name.upper()}_{dataset_name.upper()}_LOSOCV"
    exp_dir = os.path.join(out_root, f"{model_name}_{dataset_name}")
    os.makedirs(exp_dir, exist_ok=True)

    print("\n" + "="*80)
    print(f"[{time.strftime('%H:%M:%S')}] AVVIO: {exp_tag}")
    print(f" Cartella dati: {data_dir} | Pesi: {weight_path}")
    print("="*80, flush=True)

    with open(os.path.join(data_dir, "metadata.pkl"), "rb") as f: meta = pickle.load(f)

    corrected = []
    for p in meta["sessions_paths"]:
        parts = p.replace("\\", "/").split("/")
        cand = os.path.join(data_dir, parts[-2], parts[-1])
        corrected.append(cand if os.path.exists(cand) else p)
    all_paths = np.array(corrected)

    lbls = meta["sessions_labels"]
    lbl_k = next((k for k in ["status", "label", "binary_stress", "stress"] if k in lbls), list(lbls.keys())[0])
    raw_y = np.array(lbls[lbl_k], dtype=np.float32)
    uniq = set(np.unique(raw_y))
    all_labels = raw_y if uniq.issubset({0.0, 1.0}) else np.array([1.0 if y in [1.0, 2.0] else 0.0 for y in raw_y], dtype=np.float32)

    if "wesad" in dataset_name.lower():
        all_subs = np.array([re.search(r'[sS]\d+', p).group(0).upper() if re.search(r'[sS]\d+', p) else "S0" for p in all_paths])
        unique_subs = sorted(list(np.unique(all_subs)), key=lambda x: int(re.search(r'\d+', x).group(0)) if re.search(r'\d+', x) else 0)
    else:
        all_subs = np.array([p.replace("\\", "/").split("/")[-2].split("_")[0] for p in all_paths])
        unique_subs = sorted(list(np.unique(all_subs)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_folds = len(unique_subs)
    print(f"[INFO] Trovati {len(all_paths)} campioni su {n_folds} soggetti: {unique_subs}", flush=True)

    all_epoch_records = []
    fold_summary = []
    global_y_true, global_y_pred = [], []

    for idx, test_sub in enumerate(unique_subs, 1):
        fold_tag = f"Fold_{idx:02d}_{test_sub}"
        tr_mask = (all_subs != test_sub)
        te_mask = (all_subs == test_sub)

        ds_train = WearableDataset(all_paths[tr_mask], all_labels[tr_mask])
        ds_test = WearableDataset(all_paths[te_mask], all_labels[te_mask])
        loader_tr = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
        loader_te = DataLoader(ds_test, batch_size=batch_size, shuffle=False)

        tr_targets = all_labels[tr_mask]
        n_pos = sum(1 for y in tr_targets if y == 1.0)
        n_neg = sum(1 for y in tr_targets if y == 0.0)
        pos_weight = torch.tensor([n_neg / max(1, n_pos)], device=device, dtype=torch.float32)

        if model_name == "biot": model = BIOTClassifier(BIOTEncoder()).to(device)
        elif model_name == "simmtm": model = SimMTMClassifier(SimMTMEncoder()).to(device)
        elif model_name == "femba": model = FEMBA().to(device)

        if weight_path and os.path.exists(weight_path):
            ckpt = torch.load(weight_path, map_location=device)
            sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
            if model_name in ["biot", "simmtm"]:
                model.encoder.load_state_dict({k.replace("encoder.", ""): v for k, v in sd.items()}, strict=False)
            else:
                model.load_state_dict(sd, strict=False)

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

        for ep in range(1, epochs + 1):
            model.train()
            tr_losses, tr_y_t, tr_y_p = [], [], []
            for xb, yb in loader_tr:
                xb, yb_gpu = xb.to(device), yb.to(device).unsqueeze(1)
                opt.zero_grad()
                out = model(xb)
                loss = criterion(out, yb_gpu)
                loss.backward(); opt.step()
                tr_losses.append(loss.item())
                tr_y_p.extend((torch.sigmoid(out) >= 0.5).long().cpu().numpy().flatten())
                tr_y_t.extend(yb.numpy().flatten())

            model.eval()
            val_losses, val_y_t, val_y_p = [], [], []
            with torch.no_grad():
                for xb, yb in loader_te:
                    xb, yb_gpu = xb.to(device), yb.to(device).unsqueeze(1)
                    out = model(xb)
                    val_losses.append(criterion(out, yb_gpu).item())
                    val_y_p.extend((torch.sigmoid(out) >= 0.5).long().cpu().numpy().flatten())
                    val_y_t.extend(yb.numpy().flatten())

            all_epoch_records.append({
                "fold": fold_tag, "epoch": ep,
                "train_loss": round(np.mean(tr_losses), 4),
                "train_acc": round(accuracy_score(tr_y_t, tr_y_p) * 100, 2),
                "val_loss": round(np.mean(val_losses), 4),
                "val_acc": round(accuracy_score(val_y_t, val_y_p) * 100, 2),
                "val_f1_stress": round(f1_score(val_y_t, val_y_p, zero_division=0) * 100, 2),
            })

        acc = accuracy_score(val_y_t, val_y_p)
        f1 = f1_score(val_y_t, val_y_p, zero_division=0)
        f1_m = f1_score(val_y_t, val_y_p, average="macro", zero_division=0)
        prec = precision_score(val_y_t, val_y_p, zero_division=0)
        rec = recall_score(val_y_t, val_y_p, zero_division=0)

        global_y_true.extend(val_y_t)
        global_y_pred.extend(val_y_p)

        fold_res = {
            "fold": idx, "subject": test_sub, "samples": len(val_y_t),
            "accuracy": round(acc * 100, 2), "f1_stress": round(f1 * 100, 2),
            "macro_f1": round(f1_m * 100, 2), "precision": round(prec * 100, 2), "recall": round(rec * 100, 2)
        }
        fold_summary.append(fold_res)
        print(f"[{time.strftime('%H:%M:%S')}] -> [{idx:02d}/{n_folds}] Soggetto {test_sub:>3s} | Acc: {acc*100:5.2f}% | F1-Stress: {f1*100:5.2f}% | Macro-F1: {f1_m*100:5.2f}%", flush=True)

        # SALVATAGGIO INCREMENTALE DOPO OGNI SINGOLO FOLD!
        save_incremental(exp_dir, all_epoch_records, fold_summary, global_y_true, global_y_pred, exp_tag, model_name, dataset_name, weight_path)

    print(f"\n[COMPLETATO] {exp_tag} salvato con successo in: {exp_dir}\n", flush=True)

# =====================================================================
# MAIN: GESTISCE TUTTI I MODELLI RIMANENTI IN SEQUENZA
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Runner per i modelli rimanenti con salvataggio incrementale")
    parser.add_argument("--mode", type=str, default="remaining", choices=["remaining", "femba_wesad", "simmtm_hosseini", "biot_hosseini"])
    args = parser.parse_args()

    weight_map = {
        "biot": "biot_encoder_pretrained.pt",
        "simmtm": "simmtm_encoder_pretrained.pt",
        "femba": "femba_encoder_pretrained.pt"
    }

    queue = []
    if args.mode == "remaining":
        # I tre esperimenti che mancano alla lista
        queue = [
            ("femba", "wesad", r"downstream_data\wesad_segmented"),
            ("simmtm", "hosseini", r"downstream_data\hosseini_segmented"),
            ("biot", "hosseini", r"downstream_data\hosseini_segmented"),
        ]
    elif args.mode == "femba_wesad":
        queue = [("femba", "wesad", r"downstream_data\wesad_segmented")]
    elif args.mode == "simmtm_hosseini":
        queue = [("simmtm", "hosseini", r"downstream_data\hosseini_segmented")]
    elif args.mode == "biot_hosseini":
        queue = [("biot", "hosseini", r"downstream_data\hosseini_segmented")]

    print("\n" + "#"*80)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] CODA ESPERIMENTI DA ESEGUIRE: {len(queue)}")
    for m, d, path in queue:
        print(f"  * Modello: {m.upper():<8} su Dataset: {d.upper():<10}")
    print("#"*80 + "\n", flush=True)

    for m, d, path in queue:
        run_experiment(
            model_name=m,
            dataset_name=d,
            data_dir=path,
            weight_path=weight_map[m],
            epochs=20,
            lr=0.0001,
            batch_size=16
        )

    print("\n" + "="*80)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] TUTTI GLI ESPERIMENTI IN CODA SONO STATI COMPLETATI!")
    print("="*80 + "\n", flush=True)

if __name__ == "__main__":
    main()
