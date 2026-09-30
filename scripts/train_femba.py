import argparse, os, sys, pickle, numpy as np, torch, re, h5py
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

# ==========================================
# Architettura FEMBA (Foundational EEG/Biosignal Mamba)
# Implementazione 100% Pure PyTorch
# ==========================================

class PureMamba(nn.Module):
    def __init__(self, d_model=384, d_state=16, d_conv=4, expand=4, dt_rank=24):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = d_model * expand  # 1536
        self.dt_rank = dt_rank  # 24

        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, kernel_size=d_conv,
            groups=self.d_inner, padding=d_conv - 1
        )
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * self.d_state, bias=False)
        # Canonical Mamba S4D-Real diagonal decay initialization
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
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
        # x: (B, C=6, T=640)
        B, C, T = x.shape
        x_in = x.unsqueeze(1)  # (B, 1, 6, 640)
        feat = self.patch_embed.proj(x_in)  # (B, 128, 3, 40)
        feat = feat.permute(0, 3, 1, 2).contiguous().view(B, 40, 384)
        feat = feat + self.pos_embed

        for mamba, norm in zip(self.mamba_blocks, self.norm_layers):
            feat = norm(mamba(feat)) + feat

        pooled = feat.mean(dim=1)  # (B, 384)
        return self.head(pooled)

# ==========================================
# Dataset PyTorch per FEMBA (Resampling 640)
# ==========================================

class FEMBA_Dataset(Dataset):
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

        # Resampling dei 6 canali a 640 campioni
        tensors = []
        for ch in self.channel_keys:
            raw_val = data_dict[ch]
            if raw_val.ndim == 1:
                raw_val = raw_val.unsqueeze(0).unsqueeze(0)
            elif raw_val.ndim == 2:
                raw_val = raw_val.unsqueeze(0)

            resampled = F.interpolate(raw_val, size=self.target_len, mode='linear', align_corners=False).squeeze()
            mean = resampled.mean()
            std = resampled.std()
            if std < 1e-7:
                std = 1.0
            normalized = (resampled - mean) / std
            tensors.append(normalized)

        x = torch.stack(tensors, dim=0)  # (6, 640)
        y = float(self.labels[idx])
        return x, torch.tensor(y, dtype=torch.float32)

# ==========================================
# Training di un singolo Fold
# ==========================================

def run_single_fold(args, fold_name, ds_train, ds_test, verbose=True):
    loader_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True)
    loader_test = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False)

    model = FEMBA(seq_length=640, num_channels=6, num_classes=1, embed_dim=128, num_blocks=4, dropout=args.dropout)
    if args.pretrained_weights and os.path.exists(args.pretrained_weights):
        ckpt = torch.load(args.pretrained_weights, map_location="cpu")
        model.load_state_dict(ckpt, strict=False)
        if verbose:
            print(f" [OK] Pesi pre-addestrati FEMBA caricati da: {args.pretrained_weights}")

    model = model.to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.weighted_loss:
        train_y = [ds_train[i][1].item() for i in range(len(ds_train))]
        n_pos = sum(1 for y in train_y if y == 1.0)
        n_neg = sum(1 for y in train_y if y == 0.0)
        pos_weight = torch.tensor([n_neg / max(1, n_pos)], device=args.device, dtype=torch.float32)
        if verbose:
            print(f" [{fold_name}] Weighted Loss: {n_neg} non-stress vs {n_pos} stress (pos_weight = {pos_weight.item():.2f})")
    else:
        pos_weight = None

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for xb, yb in loader_train:
            xb = xb.to(args.device)
            yb = yb.to(args.device).unsqueeze(1)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        if epoch % 2 == 0 or epoch == 1 or epoch == args.epochs:
            print(f"   --> {fold_name} | Epoca {epoch:02d}/{args.epochs} - Loss: {np.mean(losses):.4f}", flush=True)

    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in loader_test:
            xb = xb.to(args.device)
            out = model(xb)
            preds = (torch.sigmoid(out) >= 0.5).long().cpu().numpy().flatten()
            y_pred.extend(preds)
            y_true.extend(yb.numpy().flatten())

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    f1_m = f1_score(y_true, y_pred, average="macro", zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)

    del model, loader_train, loader_test, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return y_true, y_pred, acc, f1, f1_m, prec, rec

# ==========================================
# MAIN
# ==========================================

def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[INFO] Modello: FEMBA (Foundational EEG/Biosignal Mamba - State Space Model)")
    print(f"[INFO] Dispositivo in uso: {args.device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    os.makedirs(args.output_dir, exist_ok=True)

    meta_file = os.path.join(args.dataset, "metadata.pkl")
    with open(meta_file, "rb") as f:
        meta = pickle.load(f)

    corrected_paths = []
    for p in meta["sessions_paths"]:
        parts = p.replace("\\", "/").split("/")
        corrected_paths.append(os.path.join(args.dataset, parts[-2], parts[-1]))
    all_paths = np.array(corrected_paths)

    labels_dict = meta["sessions_labels"]
    possible_keys = ["status", "label", "binary_stress", "stress"]
    lbl_key = next((k for k in possible_keys if k in labels_dict), None)
    if lbl_key is None:
        for k, v in labels_dict.items():
            u = np.unique(v)
            if len(u) <= 3 and set(u).issubset({0.0, 1.0, 2.0, -9.0}):
                lbl_key = k
                break
        if lbl_key is None:
            lbl_key = list(labels_dict.keys())[0]

    raw_labels = np.array(labels_dict[lbl_key], dtype=np.float32)
    uniq = set(np.unique(raw_labels))
    print(f"[INFO] Chiave label selezionata: '{lbl_key}' con valori unici: {sorted(list(uniq))}")

    if uniq.issubset({0.0, 1.0}):
        all_labels = raw_labels
    elif 2.0 in uniq and 0.0 not in uniq:
        all_labels = np.array([1.0 if y == 2.0 else 0.0 for y in raw_labels], dtype=np.float32)
    else:
        all_labels = np.array([1.0 if y == 1.0 else 0.0 for y in raw_labels], dtype=np.float32)

    n_pos = int(np.sum(all_labels == 1.0))
    n_neg = int(np.sum(all_labels == 0.0))
    print(f"[INFO] Distribuzione classi: {n_neg} Non-Stress (0.0) | {n_pos} Stress (1.0)")

    # Identificazione soggetti
    if "physionet" in args.dataset.lower():
        args.ds_type = "PhysioNet"
        all_subs = np.array([p.replace("\\", "/").split("/")[-2] for p in all_paths])
        unique_subs = sorted(list(np.unique(all_subs)))
    elif any("s" in p.lower() and re.search(r'[sS]\d+', p) for p in all_paths[:20]) and "wesad" in args.dataset.lower():
        args.ds_type = "WESAD"
        all_subs = np.array([re.search(r'[sS]\d+', p).group(0).upper() for p in all_paths])
        unique_subs = sorted(list(np.unique(all_subs)), key=lambda x: int(re.search(r'\d+', x).group(0)))
    else:
        args.ds_type = "Hosseini"
        all_subs = np.array([p.replace("\\", "/").split("/")[-2].split("_")[0] for p in all_paths])
        unique_subs = sorted(list(np.unique(all_subs)))

    print(f"[INFO] Dataset rilevato: {args.ds_type} | {len(all_paths)} campioni totali su {len(unique_subs)} soggetti: {unique_subs}")

    if args.losocv:
        n_folds = len(unique_subs)
        print(f"\n=======================================================")
        print(f"   AVVIO LOSOCV FEMBA SU {args.ds_type} ({n_folds} FOLD SUBJECT-INDEPENDENT)")
        print(f"=======================================================")
        fold_results = []
        all_y_true, all_y_pred = [], []

        for idx, test_sub in enumerate(unique_subs, 1):
            train_mask = (all_subs != test_sub)
            test_mask = (all_subs == test_sub)

            ds_train = FEMBA_Dataset(all_paths[train_mask], all_labels[train_mask])
            ds_test = FEMBA_Dataset(all_paths[test_mask], all_labels[test_mask])

            y_t, y_p, acc, f1, f1_m, prec, rec = run_single_fold(
                args, f"Fold {idx:02d}/{n_folds} ({test_sub})", ds_train, ds_test, verbose=False
            )
            all_y_true.extend(y_t); all_y_pred.extend(y_p)
            fold_results.append({
                "sub": test_sub, "acc": acc, "f1": f1, "f1_macro": f1_m, "prec": prec, "rec": rec, "n": len(y_t)
            })
            print(f" Fold {idx:02d}/{n_folds} - Test Sub: {test_sub:>3s} ({len(y_t):>2d} campioni) -> Acc: {acc*100:5.2f}% | F1-Stress: {f1*100:5.2f}% | Rec: {rec*100:5.2f}% | Prec: {prec*100:5.2f}%")

        accs = [r["acc"] for r in fold_results]
        f1s = [r["f1"] for r in fold_results]
        f1ms = [r["f1_macro"] for r in fold_results]
        precs = [r["prec"] for r in fold_results]
        recs = [r["rec"] for r in fold_results]

        cm = confusion_matrix(all_y_true, all_y_pred)
        global_acc = accuracy_score(all_y_true, all_y_pred)
        global_f1 = f1_score(all_y_true, all_y_pred, zero_division=0)
        global_f1m = f1_score(all_y_true, all_y_pred, average="macro", zero_division=0)
        global_prec = precision_score(all_y_true, all_y_pred, zero_division=0)
        global_rec = recall_score(all_y_true, all_y_pred, zero_division=0)

        print("\n" + "="*60)
        print(f"       REPORT FINALE LOSOCV FEMBA ({args.ds_type} {n_folds}-FOLD)")
        print("="*60)
        print(f" Accuratezza Media (Fold) : {np.mean(accs)*100:6.2f}% +- {np.std(accs)*100:5.2f}%")
        print(f" F1-Stress Medio (Fold)   : {np.mean(f1s)*100:6.2f}% +- {np.std(f1s)*100:5.2f}%")
        print(f" Macro-F1 Medio (Fold)    : {np.mean(f1ms)*100:6.2f}% +- {np.std(f1ms)*100:5.2f}%")
        print(f" Richiamo Medio (Fold)    : {np.mean(recs)*100:6.2f}% +- {np.std(recs)*100:5.2f}%")
        print(f" Precisione Media (Fold)  : {np.mean(precs)*100:6.2f}% +- {np.std(precs)*100:5.2f}%")
        print("-"*60)
        print(f" POOLED (Globale su tutti i {len(all_y_true)} campioni):")
        print(f"  Acc: {global_acc*100:5.2f}% | F1-Stress: {global_f1*100:5.2f}% | Macro-F1: {global_f1m*100:5.2f}% | Rec: {global_rec*100:5.2f}% | Prec: {global_prec*100:5.2f}%")
        print("-"*60)
        print(" MATRICE DI CONFUSIONE POOLED:")
        print(cm)
        print("="*60)

        # Salvataggio su disco per statistiche e report
        os.makedirs(args.output_dir, exist_ok=True)
        losocv_data = []
        for r in fold_results:
            losocv_data.append({
                "subject": r["sub"],
                "fold": r["sub"],
                "accuracy": round(r["acc"] * 100, 2),
                "f1_stress": round(r["f1"] * 100, 2),
                "macro_f1": round(r["f1_macro"] * 100, 2),
                "precision": round(r["prec"] * 100, 2),
                "recall": round(r["rec"] * 100, 2),
                "samples": r["n"]
            })
        import json
        with open(os.path.join(args.output_dir, "losocv_folds.json"), "w", encoding="utf-8") as f_json:
            json.dump(losocv_data, f_json, indent=2)

        rep_lines = [
            "="*60,
            f"       REPORT FINALE LOSOCV FEMBA ({args.ds_type} {n_folds}-FOLD)",
            "="*60,
            f" Accuratezza Media (Fold) : {np.mean(accs)*100:6.2f}% +- {np.std(accs)*100:5.2f}%",
            f" F1-Stress Medio (Fold)   : {np.mean(f1s)*100:6.2f}% +- {np.std(f1s)*100:5.2f}%",
            f" Macro-F1 Medio (Fold)    : {np.mean(f1ms)*100:6.2f}% +- {np.std(f1ms)*100:5.2f}%",
            f" Richiamo Medio (Fold)    : {np.mean(recs)*100:6.2f}% +- {np.std(recs)*100:5.2f}%",
            f" Precisione Media (Fold)  : {np.mean(precs)*100:6.2f}% +- {np.std(precs)*100:5.2f}%",
            "-"*60,
            f" POOLED (Globale su tutti i {len(all_y_true)} campioni):",
            f"  Acc: {global_acc*100:5.2f}% | F1-Stress: {global_f1*100:5.2f}% | Macro-F1: {global_f1m*100:5.2f}%",
            "-"*60,
            " MATRICE DI CONFUSIONE POOLED:",
            str(cm),
            "="*60
        ]
        with open(os.path.join(args.output_dir, "report_finale.txt"), "w", encoding="utf-8") as f_rep:
            f_rep.write("\n".join(rep_lines))
        print(f" [OK] Risultati salvati in: {args.output_dir}/losocv_folds.json e report_finale.txt\n")

    else:
        print(f"\n[INFO] Avvio addestramento su Split Fisso ({args.ds_type})...")
        if args.ds_type == "WESAD":
            train_subs = ["S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "S11"]
            test_subs = ["S13", "S14", "S15", "S16", "S17"]
        else:
            np.random.seed(args.seed)
            shuffled = np.random.permutation(unique_subs)
            split_idx = int(len(unique_subs) * 0.70)
            train_subs, test_subs = list(shuffled[:split_idx]), list(shuffled[split_idx:])

        m_train = np.isin(all_subs, train_subs)
        m_test = np.isin(all_subs, test_subs)
        ds_train = FEMBA_Dataset(all_paths[m_train], all_labels[m_train])
        ds_test = FEMBA_Dataset(all_paths[m_test], all_labels[m_test])

        y_true, y_pred, acc, f1, f1_m, prec, rec = run_single_fold(
            args, f"Split Fisso ({args.ds_type})", ds_train, ds_test, verbose=True
        )
        cm = confusion_matrix(y_true, y_pred)
        print("\n" + "="*50)
        print(f"     REPORT FINALE FEMBA (SPLIT FISSO {args.ds_type})")
        print("="*50)
        print(f" Accuratezza  : {acc*100:6.2f}%")
        print(f" F1-Stress    : {f1*100:6.2f}%")
        print(f" Macro-F1     : {f1_m*100:6.2f}%")
        print(f" Richiamo     : {rec*100:6.2f}%")
        print(f" Precisione   : {prec*100:6.2f}%")
        print("-"*50)
        print(" MATRICE DI CONFUSIONE:")
        print(cm)
        print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tuning FEMBA su WESAD / Hosseini")
    parser.add_argument("--dataset", type=str, required=True, help="Percorso a wesad_segmented o hosseini_segmented")
    parser.add_argument("--pretrained_weights", type=str, default="", help="Pesi pre-trained dell'encoder FEMBA")
    parser.add_argument("--epochs", type=int, default=20, help="Numero di epoche (default 20)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size (default 16)")
    parser.add_argument("--lr", type=float, default=0.0003, help="Learning rate (default 0.0003)")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate (default 0.3)")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay per AdamW")
    parser.add_argument("--output_dir", type=str, default="downstream/femba/run", help="Cartella output")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--weighted_loss", action="store_true", help="Usa perdita pesata (pos_weight)")
    parser.add_argument("--losocv", action="store_true", help="Esegui Leave-One-Subject-Out Cross-Validation")
    main(parser.parse_args())
