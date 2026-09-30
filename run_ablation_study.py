import argparse
import os
import sys
import glob
import re
import json
import pickle
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

# =====================================================================
# 1. ARCHITETTURA SimMTM (Transformer)
# =====================================================================
class PatchEmbedding(nn.Module):
    def __init__(self, patch_size=16, d_model=128):
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

# =====================================================================
# 2. ARCHITETTURA FEMBA (Mamba / State-Space Model)
# =====================================================================
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

    def extract_features(self, x):
        B, C, T = x.shape
        x_in = x.unsqueeze(1)
        feat = self.patch_embed.proj(x_in)
        feat = feat.permute(0, 3, 1, 2).contiguous().view(B, 40, 384)
        feat = feat + self.pos_embed
        for mamba, norm in zip(self.mamba_blocks, self.norm_layers):
            feat = norm(mamba(feat)) + feat
        return feat

    def forward(self, x):
        feat = self.extract_features(x)
        pooled = feat.mean(dim=1)
        return self.head(pooled)

# =====================================================================
# 3. DATASET PYTORCH (Pretraining & Fine-tuning)
try:
    import h5py
except ImportError:
    h5py = None

CHANNELS = ['ACC_x', 'ACC_y', 'ACC_z', 'BVP', 'EDA', 'TEMP']

def load_data_dict(path):
    try:
        if h5py is not None and (path.endswith('.h5') or h5py.is_hdf5(path)):
            d = {}
            with h5py.File(path, 'r') as hf:
                for ch in CHANNELS:
                    if ch in hf:
                        d[ch] = torch.from_numpy(hf[ch][:]).float()
            return d
    except Exception:
        pass
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj["data"] if isinstance(obj, dict) and "data" in obj else obj

class PretrainDataset(Dataset):
    def __init__(self, file_paths, target_len=640):
        self.file_paths = file_paths
        self.target_len = target_len

    def __len__(self): return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        d = load_data_dict(path)
        tensors = []
        for ch in CHANNELS:
            raw = d.get(ch, torch.zeros(self.target_len))
            if not isinstance(raw, torch.Tensor): raw = torch.tensor(raw, dtype=torch.float32)
            if raw.ndim == 1: raw = raw.unsqueeze(0).unsqueeze(0)
            elif raw.ndim == 2: raw = raw.unsqueeze(0)
            res = F.interpolate(raw, size=self.target_len, mode="linear", align_corners=False).squeeze()
            mean, std = torch.mean(res), torch.std(res) + 1e-6
            tensors.append((res - mean) / std)
        return torch.stack(tensors, dim=0)

class WearableDownstreamDataset(Dataset):
    def __init__(self, file_paths, labels, target_len=640):
        self.file_paths = file_paths
        self.labels = labels
        self.target_len = target_len

    def __len__(self): return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        d = load_data_dict(path)
        tensors = []
        for ch in CHANNELS:
            raw = d.get(ch, torch.zeros(self.target_len))
            if not isinstance(raw, torch.Tensor): raw = torch.tensor(raw, dtype=torch.float32)
            if raw.ndim == 1: raw = raw.unsqueeze(0).unsqueeze(0)
            elif raw.ndim == 2: raw = raw.unsqueeze(0)
            res = F.interpolate(raw, size=self.target_len, mode="linear", align_corners=False).squeeze()
            mean, std = torch.mean(res), torch.std(res) + 1e-6
            tensors.append((res - mean) / std)
        return torch.stack(tensors, dim=0), torch.tensor(self.labels[idx], dtype=torch.float32)

# =====================================================================
# 4. PRETRAINING (SimMTM & FEMBA)
# =====================================================================
def pretrain_simmtm(file_paths, out_ckpt_path, epochs=10, batch_size=32, lr=0.0001, mask_ratio=0.4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[PRE-TRAINING SimMTM] {len(file_paths):,} finestre | Epoche: {epochs} | Device: {device}", flush=True)

    encoder = SimMTMEncoder().to(device)
    decoder = nn.Linear(encoder.d_model, encoder.patch_size).to(device)
    opt = torch.optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()), lr=lr, weight_decay=1e-4)
    loader = DataLoader(PretrainDataset(file_paths), batch_size=batch_size, shuffle=True, drop_last=True)

    t0 = time.time()
    for ep in range(1, epochs + 1):
        encoder.train(); decoder.train()
        losses = []
        for xb in loader:
            xb = xb.to(device)
            B, C, T = xb.shape
            B_C = B * C
            x_cut = xb[:, :, :encoder.n_patches * encoder.patch_size].contiguous().view(B_C, encoder.n_patches, encoder.patch_size)
            mask = torch.rand(B_C, encoder.n_patches, device=device) < mask_ratio
            if not mask.any(): mask[:, 0] = True
            
            x_masked = x_cut.clone()
            x_masked[mask] = 0.0

            emb = encoder.patch_emb(x_masked) + encoder.pos_emb
            rep = encoder.norm(encoder.transformer(emb))
            recon = decoder(rep)

            loss = F.mse_loss(recon[mask], x_cut[mask])
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())

        print(f"  -> SimMTM Epoca {ep:02d}/{epochs:02d} ({time.time()-t0:.1f}s) | MSE Loss: {np.mean(losses):.4f}", flush=True)

    os.makedirs(os.path.dirname(out_ckpt_path), exist_ok=True)
    torch.save({"model": encoder.state_dict()}, out_ckpt_path)
    print(f"[OK] Pesi pre-trained SimMTM salvati in: {out_ckpt_path}\n", flush=True)

def pretrain_femba(file_paths, out_ckpt_path, epochs=10, batch_size=32, lr=0.0001, mask_ratio=0.4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[PRE-TRAINING FEMBA Mamba] {len(file_paths):,} finestre | Epoche: {epochs} | Device: {device}", flush=True)

    femba = FEMBA().to(device)
    decoder = nn.Linear(384, 6 * 16).to(device)
    opt = torch.optim.AdamW(list(femba.parameters()) + list(decoder.parameters()), lr=lr, weight_decay=1e-4)
    loader = DataLoader(PretrainDataset(file_paths), batch_size=batch_size, shuffle=True, drop_last=True)

    t0 = time.time()
    for ep in range(1, epochs + 1):
        femba.train(); decoder.train()
        losses = []
        for xb in loader:
            xb = xb.to(device) # (B, 6, 640)
            B, C, T = xb.shape
            
            # Mask temporal patches
            mask = torch.rand(B, 40, device=device) < mask_ratio
            if not mask.any(): mask[:, 0] = True

            xb_masked = xb.clone()
            for t_idx in range(40):
                m_t = mask[:, t_idx]
                xb_masked[m_t, :, t_idx*16:(t_idx+1)*16] = 0.0

            feats = femba.extract_features(xb_masked) # (B, 40, 384)
            recon = decoder(feats) # (B, 40, 96)
            recon = recon.view(B, 40, 6, 16).permute(0, 2, 1, 3).contiguous().view(B, 6, 640)

            loss = F.mse_loss(recon, xb)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())

        print(f"  -> FEMBA Epoca {ep:02d}/{epochs:02d} ({time.time()-t0:.1f}s) | MSE Loss: {np.mean(losses):.4f}", flush=True)

    os.makedirs(os.path.dirname(out_ckpt_path), exist_ok=True)
    torch.save({"model": femba.state_dict()}, out_ckpt_path)
    print(f"[OK] Pesi pre-trained FEMBA salvati in: {out_ckpt_path}\n", flush=True)

# =====================================================================
# 5. VALUTAZIONE DOWNSTREAM IN LOSOCV (WESAD & Hosseini)
# =====================================================================
def evaluate_downstream_losocv(model_type, dataset_name, data_dir, ckpt_path, epochs=20, lr=0.0001, batch_size=16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    fold_accs, fold_f1s, fold_mf1s = [], [], []

    for idx, test_sub in enumerate(unique_subs, 1):
        tr_m = (all_subs != test_sub)
        te_m = (all_subs == test_sub)

        ds_tr = WearableDownstreamDataset(all_paths[tr_m], all_labels[tr_m])
        ds_te = WearableDownstreamDataset(all_paths[te_m], all_labels[te_m])
        loader_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True)
        loader_te = DataLoader(ds_te, batch_size=batch_size, shuffle=False)

        tr_targets = all_labels[tr_m]
        n_pos = sum(1 for y in tr_targets if y == 1.0)
        n_neg = sum(1 for y in tr_targets if y == 0.0)
        pos_weight = torch.tensor([n_neg / max(1, n_pos)], device=device, dtype=torch.float32)

        if model_type == "femba":
            model = FEMBA().to(device)
            if os.path.exists(ckpt_path):
                sd = torch.load(ckpt_path, map_location=device)
                sd = sd["model"] if "model" in sd else sd
                model.load_state_dict(sd, strict=False)
        else:
            model = SimMTMClassifier(SimMTMEncoder()).to(device)
            if os.path.exists(ckpt_path):
                sd = torch.load(ckpt_path, map_location=device)
                sd = sd["model"] if "model" in sd else sd
                model.encoder.load_state_dict({k.replace("encoder.", ""): v for k, v in sd.items()}, strict=False)

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

        for ep in range(1, epochs + 1):
            model.train()
            for xb, yb in loader_tr:
                xb, yb_g = xb.to(device), yb.to(device).unsqueeze(1)
                opt.zero_grad()
                loss = criterion(model(xb), yb_g)
                loss.backward(); opt.step()

        model.eval()
        val_y_t, val_y_p = [], []
        with torch.no_grad():
            for xb, yb in loader_te:
                xb = xb.to(device)
                out = model(xb)
                val_y_p.extend((torch.sigmoid(out) >= 0.5).long().cpu().numpy().flatten())
                val_y_t.extend(yb.numpy().flatten())

        fold_accs.append(accuracy_score(val_y_t, val_y_p) * 100)
        fold_f1s.append(f1_score(val_y_t, val_y_p, zero_division=0) * 100)
        fold_mf1s.append(f1_score(val_y_t, val_y_p, average="macro", zero_division=0) * 100)

    mean_acc = round(float(np.mean(fold_accs)), 2)
    mean_f1 = round(float(np.mean(fold_f1s)), 2)
    mean_mf1 = round(float(np.mean(fold_mf1s)), 2)
    return {"acc": mean_acc, "f1_stress": mean_f1, "macro_f1": mean_mf1}

# =====================================================================
# 6. GESTIONE ABLATION: LODO E QUANTITY VS DIVERSITY
# =====================================================================
KNOWN_DATASETS = [
    "adarp", "big-ideas", "dati_preelaborati", "in-gauge_en-gage",
    "ppg_dalia", "pgg_dalia", "spd", "stress_detection_nurses_hospital",
    "toadstool", "ue4w", "weee", "wesad", "wesd"
]

def map_path_to_dataset(p):
    norm = p.replace("\\", "/").lower()
    for k in sorted(KNOWN_DATASETS, key=lambda x: -len(x)):
        if k in norm:
            # Normalizza chiavi note
            if k in ["ppg_dalia", "pgg_dalia"]: return "ppg_dalia"
            if k in ["big-ideas", "dati_preelaborati"]: return "big-ideas"
            return k
    return "unknown"

def run_ablation(args):
    print("\n" + "="*85)
    print(f"   AVVIO SUITE ABLATION STUDY - MODELLO: {args.model.upper()} ({args.mode.upper()})")
    print("="*85)

    meta_file = args.meta_path
    if not os.path.exists(meta_file):
        alt = r"data\preprocessed\unsegmented_unlabelled\metadata.pkl"
        if os.path.exists(alt): meta_file = alt

    with open(meta_file, "rb") as f:
        meta = pickle.load(f)
    all_raw_paths = meta["sessions_paths"]

    base_dir = os.path.dirname(meta_file)
    norm_paths = []
    for p in all_raw_paths:
        p_c = p.replace("\\", "/")
        if "sl512_ss256_unlabelled" in p_c:
            rel = p_c.split("sl512_ss256_unlabelled/")[1]
            cand = os.path.join(base_dir, rel)
            norm_paths.append(cand if os.path.exists(cand) else p)
        else:
            norm_paths.append(p)

    dataset_buckets = {k: [] for k in KNOWN_DATASETS}
    for p in norm_paths:
        ds_k = map_path_to_dataset(p)
        if ds_k in dataset_buckets:
            dataset_buckets[ds_k].append(p)

    out_root = "risultati_ablation"
    os.makedirs(out_root, exist_ok=True)
    out_json = os.path.join(out_root, "ablation_summary.json")

    summary_results = []
    if os.path.exists(out_json):
        try:
            with open(out_json, "r", encoding="utf-8") as f:
                summary_results = json.load(f)
        except Exception:
            summary_results = []

    # Inserisci baseline FEMBA se assente
    femba_baseline_name = "FEMBA - ALL (Tutti gli 11 Dataset)"
    if args.model == "femba" and not any(r.get("configurazione") == femba_baseline_name for r in summary_results):
        summary_results.append({
            "modello": "femba", "tipo": "BASELINE", "configurazione": femba_baseline_name, "escluso": "Nessuno (Completo)",
            "wesad_acc": 82.58, "wesad_f1_stress": 52.85, "wesad_macro_f1": 71.00,
            "hosseini_acc": 55.93, "hosseini_f1_stress": 65.34, "hosseini_macro_f1": 51.76
        })

    def save_summary():
        with open(out_json, "w", encoding="utf-8") as fj:
            json.dump(summary_results, fj, indent=2)

    # -------------------------------------------------------------
    # ESPERIMENTO: QUANTITY VS DIVERSITY (Budget Fisso 8.000 finestre)
    # -------------------------------------------------------------
    if args.mode in ["all", "diversity"]:
        print("\n" + "#"*80)
        print(f" [ESPERIMENTO QUANTITY VS DIVERSITY] MODELLO: {args.model.upper()}")
        print("#"*80)
        
        target_budget = 8000
        prefix = f"{args.model}_"
        label_prefix = f"{args.model.upper()} - " if args.model == "femba" else ""

        # Caso A: Single Source (100% da Big-Ideas/dati_preelaborati)
        ckpt_single = os.path.join(out_root, "checkpoints", f"{prefix}single_source.pt")
        single_files = random.sample(dataset_buckets["dati_preelaborati"], min(len(dataset_buckets["dati_preelaborati"]), target_budget))
        if not os.path.exists(ckpt_single):
            print(f"\n>>> CONFIGURAZIONE: {label_prefix}Single-Source (8.000 finestre da 1 solo dataset)")
            if args.model == "femba":
                pretrain_femba(single_files, ckpt_single, epochs=args.pretrain_epochs, batch_size=32)
            else:
                pretrain_simmtm(single_files, ckpt_single, epochs=args.pretrain_epochs, batch_size=32)
        else:
            print(f" [SKIP] Checkpoint esistente per Single-Source: {ckpt_single}")

        # Caso B: Multi-Source Diverse (800 finestre da 10 dataset)
        ckpt_multi = os.path.join(out_root, "checkpoints", f"{prefix}multi_source.pt")
        multi_files = []
        per_ds = target_budget // len(KNOWN_DATASETS)
        for k, files in dataset_buckets.items():
            multi_files.extend(random.sample(files, min(len(files), per_ds)))
        if not os.path.exists(ckpt_multi):
            print(f"\n>>> CONFIGURAZIONE: {label_prefix}Multi-Source Diverse (8.000 finestre da 10 dataset)")
            if args.model == "femba":
                pretrain_femba(multi_files, ckpt_multi, epochs=args.pretrain_epochs, batch_size=32)
            else:
                pretrain_simmtm(multi_files, ckpt_multi, epochs=args.pretrain_epochs, batch_size=32)
        else:
            print(f" [SKIP] Checkpoint esistente per Multi-Source: {ckpt_multi}")

        # Valutazione downstream
        print(f"\n>>> Valutazione Downstream LOSOCV...")
        wesad_dir = os.path.join("downstream_data", "wesad_segmented")
        hoss_dir = os.path.join("downstream_data", "hosseini_segmented")
        res_s_wesad = evaluate_downstream_losocv(args.model, "wesad", wesad_dir, ckpt_single)
        res_s_hoss = evaluate_downstream_losocv(args.model, "hosseini", hoss_dir, ckpt_single)
        res_m_wesad = evaluate_downstream_losocv(args.model, "wesad", wesad_dir, ckpt_multi)
        res_m_hoss = evaluate_downstream_losocv(args.model, "hosseini", hoss_dir, ckpt_multi)

        cfg_s = f"{label_prefix}Fixed Budget (Single-Source)"
        cfg_m = f"{label_prefix}Fixed Budget (Multi-Source Diverse)"

        summary_results = [r for r in summary_results if r.get("configurazione") not in [cfg_s, cfg_m]]
        summary_results.append({
            "modello": args.model, "tipo": "QvD", "configurazione": cfg_s, "escluso": "Nessuno (solo BigIdeas)",
            "wesad_acc": res_s_wesad["acc"], "wesad_f1_stress": res_s_wesad["f1_stress"], "wesad_macro_f1": res_s_wesad["macro_f1"],
            "hosseini_acc": res_s_hoss["acc"], "hosseini_f1_stress": res_s_hoss["f1_stress"], "hosseini_macro_f1": res_s_hoss["macro_f1"]
        })
        summary_results.append({
            "modello": args.model, "tipo": "QvD", "configurazione": cfg_m, "escluso": "Nessuno (10 dataset)",
            "wesad_acc": res_m_wesad["acc"], "wesad_f1_stress": res_m_wesad["f1_stress"], "wesad_macro_f1": res_m_wesad["macro_f1"],
            "hosseini_acc": res_m_hoss["acc"], "hosseini_f1_stress": res_m_hoss["f1_stress"], "hosseini_macro_f1": res_m_hoss["macro_f1"]
        })
        save_summary()

    # -------------------------------------------------------------
    # ESPERIMENTO: LEAVE-ONE-DATASET-OUT (LODO)
    # -------------------------------------------------------------
    if args.mode in ["all", "lodo"]:
        print("\n" + "#"*80)
        print(f" [ESPERIMENTO LEAVE-ONE-DATASET-OUT] MODELLO: {args.model.upper()}")
        print("#"*80)

        lodo_targets = [args.target_ds] if args.target_ds != "all" else KNOWN_DATASETS
        label_prefix = f"{args.model.upper()} - " if args.model == "femba" else ""

        for exclude_ds in lodo_targets:
            exp_name = f"{args.model}_no_{exclude_ds}"
            ckpt_path = os.path.join(out_root, "checkpoints", f"{exp_name}.pt")
            
            train_files = []
            for k, files in dataset_buckets.items():
                if k != exclude_ds:
                    train_files.extend(files)

            cfg_name = f"{label_prefix}ALL - {exclude_ds}"
            print(f"\n>>> CONFIGURAZIONE: {cfg_name} ({len(train_files):,} finestre)")

            if not os.path.exists(ckpt_path):
                random.seed(args.seed)
                sampled_files = random.sample(train_files, min(len(train_files), args.max_pretrain_windows))
                if args.model == "femba":
                    pretrain_femba(sampled_files, ckpt_path, epochs=args.pretrain_epochs, batch_size=32)
                else:
                    pretrain_simmtm(sampled_files, ckpt_path, epochs=args.pretrain_epochs, batch_size=32)
            else:
                print(f" [SKIP] Checkpoint esistente: {ckpt_path}")

            wesad_dir = os.path.join("downstream_data", "wesad_segmented")
            hoss_dir = os.path.join("downstream_data", "hosseini_segmented")
            res_wesad = evaluate_downstream_losocv(args.model, "wesad", wesad_dir, ckpt_path, epochs=20)
            res_hosseini = evaluate_downstream_losocv(args.model, "hosseini", hoss_dir, ckpt_path, epochs=20)

            summary_results = [r for r in summary_results if r.get("configurazione") != cfg_name]
            summary_results.append({
                "modello": args.model, "tipo": "LODO", "configurazione": cfg_name, "escluso": exclude_ds,
                "wesad_acc": res_wesad["acc"], "wesad_f1_stress": res_wesad["f1_stress"], "wesad_macro_f1": res_wesad["macro_f1"],
                "hosseini_acc": res_hosseini["acc"], "hosseini_f1_stress": res_hosseini["f1_stress"], "hosseini_macro_f1": res_hosseini["macro_f1"]
            })
            save_summary()

    save_summary()
    print("\n" + "="*95)
    print("                      TABELLA FINALE ABLATION STUDY")
    print("="*95)
    print(f"{'Configurazione':<40} | {'WESAD Acc':<12} | {'WESAD F1':<12} | {'HOSSEINI Acc':<14} | {'HOSSEINI F1'}")
    print("-" * 95)
    for r in summary_results:
        print(f"{r['configurazione']:<40} | {r['wesad_acc']:>5.2f}%       | {r['wesad_f1_stress']:>5.2f}%       | {r['hosseini_acc']:>5.2f}%         | {r['hosseini_f1_stress']:>5.2f}%")
    print("="*95 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ablation Study Runner")
    parser.add_argument("--model", type=str, default="simmtm", choices=["simmtm", "femba"], help="Modello da usare (simmtm o femba)")
    parser.add_argument("--mode", type=str, default="diversity", choices=["all", "lodo", "diversity"])
    parser.add_argument("--target_ds", type=str, default="adarp", help="Dataset da escludere in LODO (oppure 'all')")
    parser.add_argument("--meta_path", type=str, default=r"data\preprocessed\sl512_ss256_unlabelled\metadata.pkl")
    parser.add_argument("--pretrain_epochs", type=int, default=10)
    parser.add_argument("--max_pretrain_windows", type=int, default=15000)
    parser.add_argument("--seed", type=int, default=42)
    run_ablation(parser.parse_args())
