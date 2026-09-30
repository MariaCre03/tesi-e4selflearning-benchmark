import argparse, os, sys, pickle, numpy as np, torch, re, h5py
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

# ==========================================
# Architettura BIOT (Biosignal Transformer)
# ==========================================

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

# ==========================================
# Dataset PyTorch per BIOT (Resampling 640)
# ==========================================

class BIOT_Dataset(Dataset):
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
        
        # Resampling di ciascun canale a target_len (640 campioni)
        tensors = []
        for ch in self.channel_keys:
            raw_val = data_dict[ch]
            if raw_val.ndim == 1:
                raw_val = raw_val.unsqueeze(0).unsqueeze(0)  # (1, 1, L)
            elif raw_val.ndim == 2:
                raw_val = raw_val.unsqueeze(0)

            resampled = F.interpolate(raw_val, size=self.target_len, mode='linear', align_corners=False).squeeze()
            
            # Z-Score normalization
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

    encoder = BIOTEncoder(seg_len=640, d_model=256, n_heads=8, n_layers=4)
    if args.pretrained_weights and os.path.exists(args.pretrained_weights):
        ckpt = torch.load(args.pretrained_weights, map_location="cpu")
        encoder.load_state_dict(ckpt, strict=False)
        if verbose:
            print(f" [OK] Pesi pre-addestrati caricati da: {args.pretrained_weights}")

    model = BIOTClassifier(encoder, n_classes=1, dropout=args.dropout).to(args.device)
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
    return y_true, y_pred, acc, f1, f1_m, prec, rec

# ==========================================
# MAIN
# ==========================================

def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[INFO] Modello: BIOT (Biosignal Transformer)")
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
        print(f"   AVVIO LOSOCV BIOT SU {args.ds_type} ({n_folds} FOLD SUBJECT-INDEPENDENT)")
        print(f"=======================================================")
        fold_results = []
        all_y_true, all_y_pred = [], []

        for idx, test_sub in enumerate(unique_subs, 1):
            train_mask = (all_subs != test_sub)
            test_mask = (all_subs == test_sub)

            ds_train = BIOT_Dataset(all_paths[train_mask], all_labels[train_mask])
            ds_test = BIOT_Dataset(all_paths[test_mask], all_labels[test_mask])

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
        print(f"       REPORT FINALE LOSOCV BIOT ({args.ds_type} {n_folds}-FOLD)")
        print("="*60)
        print(f" Accuratezza Media (Fold) : {np.mean(accs)*100:6.2f}% +- {np.std(accs)*100:5.2f}%")
        print(f" F1-Stress Medio (Fold)   : {np.mean(f1s)*100:6.2f}% +- {np.std(f1s)*100:5.2f}%")
        print(f" Macro-F1 Medio (Fold)    : {np.mean(f1ms)*100:6.2f}% +- {np.std(f1ms)*100:5.2f}%")
        print(f" Richiamo Medio (Fold)    : {np.mean(recs)*100:6.2f}% +- {np.std(recs)*100:5.2f}%")
        print(f" Precisione Media (Fold)  : {np.mean(precs)*100:6.2f}% +- {np.std(precs)*100:5.2f}%")
        print("-" * 60)
        print(f" POOLED (Globale su tutti i {len(all_y_true)} campioni):")
        print(f"  Acc: {global_acc*100:.2f}% | F1-Stress: {global_f1*100:.2f}% | Macro-F1: {global_f1m*100:.2f}% | Rec: {global_rec*100:.2f}% | Prec: {global_prec*100:.2f}%")
        print("-" * 60)
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
            f"       REPORT FINALE LOSOCV BIOT ({args.ds_type} {n_folds}-FOLD)",
            "="*60,
            f" Accuratezza Media (Fold) : {np.mean(accs)*100:6.2f}% +- {np.std(accs)*100:5.2f}%",
            f" F1-Stress Medio (Fold)   : {np.mean(f1s)*100:6.2f}% +- {np.std(f1s)*100:5.2f}%",
            f" Macro-F1 Medio (Fold)    : {np.mean(f1ms)*100:6.2f}% +- {np.std(f1ms)*100:5.2f}%",
            f" Richiamo Medio (Fold)    : {np.mean(recs)*100:6.2f}% +- {np.std(recs)*100:5.2f}%",
            f" Precisione Media (Fold)  : {np.mean(precs)*100:6.2f}% +- {np.std(precs)*100:5.2f}%",
            "-"*60,
            f" POOLED (Globale su tutti i {len(all_y_true)} campioni):",
            f"  Acc: {global_acc*100:.2f}% | F1-Stress: {global_f1*100:.2f}% | Macro-F1: {global_f1m*100:.2f}%",
            "-"*60,
            " MATRICE DI CONFUSIONE POOLED:",
            str(cm),
            "="*60
        ]
        with open(os.path.join(args.output_dir, "report_finale.txt"), "w", encoding="utf-8") as f_rep:
            f_rep.write("\n".join(rep_lines))
        print(f" [OK] Risultati salvati in: {args.output_dir}/losocv_folds.json e report_finale.txt\n")

    else:
        # Split Fisso Storico Tesi
        if args.ds_type == "WESAD":
            train_subs = ["S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "S11"]
            test_subs  = ["S13", "S14", "S15", "S16", "S17"]
        else:
            train_subs = ['G', 'H', 'I', 'L', 'P', 'Q', 'S', 'T']
            test_subs  = ['J', 'K', 'M', 'O']

        print(f"\n[INFO] Esecuzione Split Fisso Tesi ({len(train_subs)} Train / {len(test_subs)} Test)")
        print(f"  Train: {train_subs}")
        print(f"  Test:  {test_subs}")

        mask_train = np.isin(all_subs, train_subs)
        mask_test  = np.isin(all_subs, test_subs)

        ds_train = BIOT_Dataset(all_paths[mask_train], all_labels[mask_train])
        ds_test  = BIOT_Dataset(all_paths[mask_test], all_labels[mask_test])

        y_true, y_pred, acc, f1, f1_m, prec, rec = run_single_fold(
            args, f"Split Fisso ({args.ds_type})", ds_train, ds_test, verbose=True
        )
        cm = confusion_matrix(y_true, y_pred)

        print("\n" + "="*50)
        print(f"    CLASSIFICATION REPORT BIOT ({args.ds_type} SPLIT FISSO)")
        print("="*50)
        print(f" Accuratezza (Accuracy) : {acc*100:>6.2f}%")
        print(f" F1-Score (Stress)      : {f1*100:>6.2f}%")
        print(f" Macro F1-Score         : {f1_m*100:>6.2f}%")
        print(f" Precisione (Precision) : {prec*100:>6.2f}%")
        print(f" Richiamo (Recall)      : {rec*100:>6.2f}%")
        print("-" * 50)
        print(" MATRICE DI CONFUSIONE:")
        print(cm)
        print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="downstream_data/wesad_segmented")
    parser.add_argument("--pretrained_weights", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--output_dir", type=str, default="downstream/biot/run")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weighted_loss", action="store_true", help="Loss pesata")
    parser.add_argument("--losocv", action="store_true", help="Esegue LOSOCV subject-independent")
    main(parser.parse_args())
