import argparse
import os
import sys
import glob
import re
import json
import csv
import pickle
import time
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.getcwd(), 'src'))
from timebase.models.models import get_models
from timebase.data.dataset import ClassificationDataset
from timebase.utils import utils

class WearableClsDataset(ClassificationDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.positive_classes = [1.0]

    def segment_id(self, filename: str) -> int:
        clean = os.path.basename(filename).replace(".h5", "").replace("win_", "")
        match = re.search(r'\d+', clean)
        return int(match.group(0)) if match else 0

    def session_id(self, filename: str) -> int:
        match = re.search(r'[sS](\d+)', filename)
        if match:
            return int(match.group(1))
        match_f = re.search(r'[fF](\d+)', filename)
        if match_f:
            return int(match_f.group(1)) + 100
        match_hosseini = re.search(r'([A-Za-z]+)_T\d+', filename)
        if match_hosseini:
            sub = match_hosseini.group(1).upper()
            return ord(sub[0]) - 64 if sub[0].isalpha() else 0
        return 0

def load_pre_trained(args, classifier):
    target_file = None
    for r, d, f in os.walk(args.path2pretraining_res):
        for candidate in ["model_best.pt", "model_state.pt"]:
            if candidate in f:
                target_file = os.path.join(r, candidate)
                break
        if target_file: break
    if not target_file:
        raise FileNotFoundError(f"Pesi pre-trained non trovati in {args.path2pretraining_res}")
    ckpt = torch.load(target_file, map_location=args.device)
    sd = classifier.sslearner.state_dict()
    sd.update({k: v for k, v in ckpt["model"].items() if any(m in k for m in ["channel_embedding", "feature_encoder"])})
    classifier.sslearner.load_state_dict(sd)
    return target_file

def save_fold_results(exp_dir, fold_summary, global_y_true, global_y_pred, ds_name, n_folds):
    os.makedirs(exp_dir, exist_ok=True)
    with open(os.path.join(exp_dir, "losocv_folds.json"), "w", encoding="utf-8") as f:
        json.dump(fold_summary, f, indent=2)

    accs = [r["accuracy"] for r in fold_summary]
    f1s = [r["f1_stress"] for r in fold_summary]
    f1ms = [r["macro_f1"] for r in fold_summary]
    cm = confusion_matrix(global_y_true, global_y_pred)
    glob_acc = accuracy_score(global_y_true, global_y_pred)
    glob_f1 = f1_score(global_y_true, global_y_pred, zero_division=0)
    glob_f1m = f1_score(global_y_true, global_y_pred, average="macro", zero_division=0)

    report = [
        "="*70,
        f" REPORT FINALE LOSOCV: CORPONI_{ds_name.upper()}_LOSOCV (Fold: {len(fold_summary)}/{n_folds})",
        "="*70,
        f" Modello: CORPONI (E4mer) | Dataset: {ds_name.upper()}",
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

def run_corponi(ds_name, data_dir, pretrain_dir, out_dir, epochs=20, lr=0.0001, batch_size=16):
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "="*75)
    print(f"[{time.strftime('%H:%M:%S')}] AVVIO CORPONI (E4mer) LOSOCV: {ds_name.upper()}")
    print(f" Cartella: {data_dir} | Pesi: {pretrain_dir} | Device: {device}")
    print("="*75, flush=True)

    with open(os.path.join(data_dir, "metadata.pkl"), "rb") as f:
        meta = pickle.load(f)

    stats_p = os.path.join(data_dir, "stats.pkl")
    stats = None
    if os.path.exists(stats_p):
        with open(stats_p, "rb") as f:
            sl = pickle.load(f)
            stats = sl[list(sl.keys())[0]] if isinstance(sl, dict) and not any(k in sl for k in ['ACC_x', 'BVP']) else sl
    if stats is None:
        stats = {c: {'min': -10, 'max': 10, 'mean': 0, 'std': 1, 'median': 0, 'iqr': 1} for c in ['ACC_x', 'ACC_y', 'ACC_z', 'BVP', 'EDA', 'TEMP']}

    corrected = []
    for p in meta["sessions_paths"]:
        parts = p.replace("\\", "/").split("/")
        cand = os.path.join(data_dir, parts[-2], parts[-1])
        corrected.append(cand if os.path.exists(cand) else p)
    all_paths = np.array(corrected)
    lbls = meta["sessions_labels"]

    if "wesad" in ds_name.lower():
        all_subs = np.array([re.search(r'[sS]\d+', p).group(0).upper() if re.search(r'[sS]\d+', p) else "S0" for p in all_paths])
        unique_subs = sorted(list(np.unique(all_subs)), key=lambda x: int(re.search(r'\d+', x).group(0)) if re.search(r'\d+', x) else 0)
        seg_len = 512
        ch_freq = meta["ds_info"]["channel_freq"]
    elif "physionet" in ds_name.lower():
        all_subs = np.array([p.replace("\\", "/").split("/")[-2] for p in all_paths])
        unique_subs = sorted(list(np.unique(all_subs)))
        seg_len = 60
        ch_freq = {'ACC_x': 32, 'ACC_y': 32, 'ACC_z': 32, 'BVP': 64, 'EDA': 4, 'TEMP': 4}
    else:
        all_subs = np.array([p.replace("\\", "/").split("/")[-2].split("_")[0] for p in all_paths])
        unique_subs = sorted(list(np.unique(all_subs)))
        seg_len = 60
        ch_freq = {'ACC_x': 32, 'ACC_y': 32, 'ACC_z': 32, 'BVP': 64, 'EDA': 4, 'TEMP': 4}

    n_folds = len(unique_subs)
    print(f"[INFO] Trovati {len(all_paths)} campioni su {n_folds} soggetti: {unique_subs}", flush=True)

    fold_summary, global_y_true, global_y_pred = [], [], []

    for idx, test_sub in enumerate(unique_subs, 1):
        tr_m = (all_subs != test_sub)
        te_m = (all_subs == test_sub)

        args = argparse.Namespace(
            device=device, path2pretraining_res=pretrain_dir, dataset=data_dir,
            batch_size=batch_size, epochs=epochs, lr=lr, task_mode=1, scaling_mode=2,
            e4selflearning=True, verbose=0, use_wandb=False, save_plots=False,
            critic_score_lambda=0.0, reuse_stats=False, clear_output_dir=False
        )
        utils.load_args(args, dir=pretrain_dir)
        args.dataset = data_dir
        args.task_mode = 1
        args.scaling_mode = 2
        args.e4selflearning = True
        args.output_dir = out_dir
        args.ds_info = {"channel_freq": ch_freq, "segment_length": seg_len}
        args.input_shapes = {c: [seg_len * ch_freq[c]] for c in ch_freq.keys()}
        args.num_train_subjects = len(np.unique(all_subs[tr_m]))

        # Parametri di default per Reconstructor/Transformer se omessi dal pretraining
        defaults = {
            "drop_path": 0.0, "a_dropout": 0.0, "m_dropout": 0.0, "disable_bias": False,
            "num_blocks": 3, "num_heads": 3, "num_units": 64, "mlp_dim": 64,
            "emb_num_filters": 4, "split_mode": 0
        }
        for k, v in defaults.items():
            if not hasattr(args, k) or getattr(args, k) is None:
                setattr(args, k, v)

        ds_tr = WearableClsDataset(
            args=args, filenames=all_paths[tr_m],
            labels={k: np.array(v)[tr_m] for k, v in lbls.items()},
            rec_ids=all_paths[tr_m], stats=stats,
            recording_id_str_to_num={p: i for i, p in enumerate(np.unique(all_paths[tr_m]))}
        )
        ds_te = WearableClsDataset(
            args=args, filenames=all_paths[te_m],
            labels={k: np.array(v)[te_m] for k, v in lbls.items()},
            rec_ids=all_paths[te_m], stats=stats,
            recording_id_str_to_num={p: i for i, p in enumerate(np.unique(all_paths[te_m]))}
        )

        loader_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True)
        loader_te = DataLoader(ds_te, batch_size=batch_size, shuffle=False)

        tr_targets = [ds_tr[i]["target"].item() for i in range(len(ds_tr))]
        n_pos = sum(1 for y in tr_targets if y == 1.0)
        n_neg = sum(1 for y in tr_targets if y == 0.0)
        pos_weight = torch.tensor([n_neg / max(1, n_pos)], device=device, dtype=torch.float32)

        model, _ = get_models(args, summary=None)
        load_pre_trained(args, model)
        model = model.to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr)

        for ep in range(1, epochs + 1):
            model.train()
            for b in loader_tr:
                inputs = {k: v.float().to(device) for k, v in b["data"].items()}
                targets = b["target"].to(device).float()
                out, _ = model(inputs)
                if out.shape != targets.shape:
                    targets = targets.view_as(out)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(out, targets, pos_weight=pos_weight)
                loss.backward(); opt.step(); opt.zero_grad()

        model.eval()
        val_y_t, val_y_p = [], []
        with torch.no_grad():
            for b in loader_te:
                inputs = {k: v.float().to(device) for k, v in b["data"].items()}
                out, _ = model(inputs)
                val_y_t.extend(b["target"].cpu().numpy().flatten())
                val_y_p.extend((torch.sigmoid(out) >= 0.5).long().cpu().numpy().flatten())

        acc = accuracy_score(val_y_t, val_y_p)
        f1 = f1_score(val_y_t, val_y_p, zero_division=0)
        f1_m = f1_score(val_y_t, val_y_p, average="macro", zero_division=0)
        global_y_true.extend(val_y_t)
        global_y_pred.extend(val_y_p)

        f_res = {
            "fold": idx, "subject": test_sub, "samples": len(val_y_t),
            "accuracy": round(acc * 100, 2), "f1_stress": round(f1 * 100, 2), "macro_f1": round(f1_m * 100, 2)
        }
        fold_summary.append(f_res)
        print(f" [{idx:02d}/{n_folds}] Soggetto {test_sub:>3s} ({len(val_y_t):>2d} campioni) | Acc: {acc*100:5.2f}% | F1-Stress: {f1*100:5.2f}% | Macro-F1: {f1_m*100:5.2f}%", flush=True)

        save_fold_results(out_dir, fold_summary, global_y_true, global_y_pred, ds_name, n_folds)

    print(f"\n[OK] Completato {ds_name.upper()} per Corponi! Report salvato in {out_dir}")

def main():
    parser = argparse.ArgumentParser(description="Corponi LOSOCV Runner")
    parser.add_argument("--mode", type=str, default="all", choices=["all", "wesad", "hosseini", "physionet"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    if args.mode in ["all", "wesad"]:
        run_corponi(
            ds_name="wesad",
            data_dir=r"downstream_data\wesad_segmented",
            pretrain_dir="output_pretrain_full",
            out_dir=r"risultati_benchmark\corponi_wesad",
            epochs=args.epochs, lr=args.lr, batch_size=args.batch_size
        )

    if args.mode in ["all", "hosseini"]:
        run_corponi(
            ds_name="hosseini",
            data_dir=r"downstream_data\hosseini_segmented",
            pretrain_dir="output_pretrain_full",
            out_dir=r"risultati_benchmark\corponi_hosseini",
            epochs=args.epochs, lr=args.lr, batch_size=args.batch_size
        )

    if args.mode in ["all", "physionet"]:
        run_corponi(
            ds_name="physionet",
            data_dir=r"downstream_data\physionet_segmented",
            pretrain_dir="output_pretrain_full",
            out_dir=r"risultati_benchmark\corponi_physionet",
            epochs=args.epochs, lr=args.lr, batch_size=args.batch_size
        )

if __name__ == "__main__":
    main()
