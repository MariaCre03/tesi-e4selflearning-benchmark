import argparse, os, sys, pickle, numpy as np, torch, re
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from torch.utils.data import DataLoader
from tqdm import tqdm
from timebase.models.models import get_models
from timebase.data.dataset import ClassificationDataset
from timebase.utils import utils
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

class WESAD_ClassificationDataset(ClassificationDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.positive_classes = [1.0]
        
    def session_id(self, filename: str) -> int:
        match = re.search(r'[sS](\d+)', filename)
        return int(match.group(1)) if match else 0

def load_pre_trained(args, classifier):
    target_file = None
    for r, d, f in os.walk(args.path2pretraining_res):
        for candidate in ["model_best.pt", "model_state.pt"]:
            if candidate in f:
                target_file = os.path.join(r, candidate)
                break
        if target_file: break
    if not target_file: raise FileNotFoundError(f"ERROR: Weights not found in {args.path2pretraining_res}!")
    ckpt = torch.load(target_file, map_location=args.device)
    sd = classifier.sslearner.state_dict()
    sd.update({k: v for k, v in ckpt["model"].items() if any(m in k for m in ["channel_embedding", "feature_encoder"])})
    classifier.sslearner.load_state_dict(sd)
    return target_file

def run_single_fold(args, fold_name, ds_train, ds_test, verbose=True):
    train_subs = np.unique([re.search(r'[sS]\d+', p).group(0).upper() for p in ds_train.filenames if re.search(r'[sS]\d+', p)])
    args.num_train_subjects = len(train_subs)

    loader_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True)
    loader_test = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False)

    classifier, _ = get_models(args, summary=None)
    load_pre_trained(args, classifier)
    classifier.to(args.device)
    opt = torch.optim.AdamW(classifier.parameters(), lr=args.lr)

    if args.weighted_loss:
        train_targets = [ds_train[i]["target"].item() for i in range(len(ds_train))]
        n_pos = sum(1 for y in train_targets if y == 1.0)
        n_neg = sum(1 for y in train_targets if y == 0.0)
        pos_weight = torch.tensor([n_neg / max(1, n_pos)], device=args.device, dtype=torch.float32)
        if verbose:
            print(f" [{fold_name}] Weighted Loss: {n_neg} non-stress vs {n_pos} stress (pos_weight = {pos_weight.item():.2f})")
    else:
        pos_weight = None

    for epoch in range(1, args.epochs + 1):
        classifier.train()
        losses = []
        for batch in loader_train:
            inputs = {k: v.float().to(args.device) for k, v in batch["data"].items()}
            targets = batch["target"].to(args.device).float()
            out, _ = classifier(inputs)
            if out.shape != targets.shape:
                targets = targets.view_as(out)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                input=out, target=targets, pos_weight=pos_weight
            )
            loss.backward(); opt.step(); opt.zero_grad()
            losses.append(loss.item())

    classifier.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in loader_test:
            inputs = {k: v.float().to(args.device) for k, v in batch["data"].items()}
            out, _ = classifier(inputs)
            y_true.extend(batch["target"].cpu().numpy().flatten())
            y_pred.extend((torch.sigmoid(out) >= 0.5).long().cpu().numpy().flatten())

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    return y_true, y_pred, acc, f1, f1_macro, prec, rec

def main(args):
    utils.set_random_seed(args.seed)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[INFO] Dispositivo in uso: {args.device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    os.makedirs(args.output_dir, exist_ok=True)

    args.verbose = 0; args.use_wandb = False; args.save_plots = False
    args.critic_score_lambda = 0.0; args.reuse_stats = False; args.clear_output_dir = False

    cli_dataset_path = args.dataset
    args.task_mode = 2
    utils.load_args(args, dir=args.path2pretraining_res)
    args.dataset = cli_dataset_path
    args.task_mode = 1
    args.scaling_mode = 2
    args.e4selflearning = True

    meta_file = os.path.join(args.dataset, "metadata.pkl")
    with open(meta_file, "rb") as f:
        meta = pickle.load(f)
    args.ds_info = meta["ds_info"]
    args.input_shapes = {c: [512 * args.ds_info["channel_freq"][c]] for c in ['ACC_x', 'ACC_y', 'ACC_z', 'BVP', 'EDA', 'TEMP']}

    stats = None
    stats_path = os.path.join(args.dataset, "stats.pkl")
    if os.path.exists(stats_path):
        with open(stats_path, "rb") as f:
            sl = pickle.load(f)
            stats = sl[list(sl.keys())[0]]
    if stats is None:
        stats = {c: {'min': -10, 'max': 10, 'mean': 0, 'std': 1, 'median': 0, 'iqr': 1} for c in ['ACC_x', 'ACC_y', 'ACC_z', 'BVP', 'EDA', 'TEMP']}

    corrected_paths = []
    for p in meta["sessions_paths"]:
        parts = p.replace("\\", "/").split("/")
        corrected_paths.append(os.path.join(args.dataset, parts[-2], parts[-1]))
    all_paths = np.array(corrected_paths)
    labels_dict = {k: np.array(v) for k, v in meta["sessions_labels"].items()}

    # Estrazione soggetti
    all_subs = np.array([re.search(r'[sS]\d+', p).group(0).upper() for p in all_paths])
    unique_subs = sorted(list(np.unique(all_subs)), key=lambda x: int(re.search(r'\d+', x).group(0)))
    print(f"[INFO] Dataset WESAD: {len(all_paths)} segmenti totali su {len(unique_subs)} soggetti: {unique_subs}")

    if args.losocv:
        print(f"\n=======================================================")
        print(f"      AVVIO LOSOCV SU WESAD (15 FOLD SUBJECT-INDEPENDENT)")
        print(f"=======================================================")
        fold_results = []
        all_y_true, all_y_pred = [], []

        for idx, test_sub in enumerate(unique_subs, 1):
            train_mask = (all_subs != test_sub)
            test_mask = (all_subs == test_sub)

            ds_train = WESAD_ClassificationDataset(
                args=args, filenames=all_paths[train_mask],
                labels={k: v[train_mask] for k, v in labels_dict.items()},
                rec_ids=all_paths[train_mask], stats=stats,
                recording_id_str_to_num={p: i for i, p in enumerate(np.unique(all_paths[train_mask]))}
            )
            ds_test = WESAD_ClassificationDataset(
                args=args, filenames=all_paths[test_mask],
                labels={k: v[test_mask] for k, v in labels_dict.items()},
                rec_ids=all_paths[test_mask], stats=stats,
                recording_id_str_to_num={p: i for i, p in enumerate(np.unique(all_paths[test_mask]))}
            )

            y_t, y_p, acc, f1, f1_m, prec, rec = run_single_fold(
                args, f"Fold {idx:02d}/15 (Soggetto {test_sub})", ds_train, ds_test, verbose=False
            )
            all_y_true.extend(y_t); all_y_pred.extend(y_p)
            fold_results.append({
                "sub": test_sub, "acc": acc, "f1": f1, "f1_macro": f1_m, "prec": prec, "rec": rec, "n": len(y_t)
            })
            print(f" Fold {idx:02d}/15 - Test Sub: {test_sub:>3s} ({len(y_t):>2d} campioni) -> Acc: {acc*100:5.2f}% | F1-Stress: {f1*100:5.2f}% | Rec: {rec*100:5.2f}% | Prec: {prec*100:5.2f}%")

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
        print("          REPORT FINALE LOSOCV (WESAD 15-FOLD)")
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

    else:
        train_subs = ["S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "S11"]
        test_subs  = ["S13", "S14", "S15", "S16", "S17"]
        print(f"\n[INFO] Esecuzione Split Fisso Tesi (10 Train / 5 Test)")
        print(f"  Train: {train_subs}")
        print(f"  Test:  {test_subs}")

        mask_train = np.isin(all_subs, train_subs)
        mask_test  = np.isin(all_subs, test_subs)

        ds_train = WESAD_ClassificationDataset(
            args=args, filenames=all_paths[mask_train],
            labels={k: v[mask_train] for k, v in labels_dict.items()},
            rec_ids=all_paths[mask_train], stats=stats,
            recording_id_str_to_num={p: i for i, p in enumerate(np.unique(all_paths[mask_train]))}
        )
        ds_test = WESAD_ClassificationDataset(
            args=args, filenames=all_paths[mask_test],
            labels={k: v[mask_test] for k, v in labels_dict.items()},
            rec_ids=all_paths[mask_test], stats=stats,
            recording_id_str_to_num={p: i for i, p in enumerate(np.unique(all_paths[mask_test]))}
        )

        y_true, y_pred, acc, f1, f1_m, prec, rec = run_single_fold(
            args, "Split Fisso", ds_train, ds_test, verbose=True
        )
        cm = confusion_matrix(y_true, y_pred)

        print("\n" + "="*50)
        print("        CLASSIFICATION REPORT (WESAD SPLIT FISSO)")
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
    parser.add_argument("--path2pretraining_res", type=str, default="output_pretrain_full")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--output_dir", type=str, default="downstream/wesad/run_weighted")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weighted_loss", action="store_true", help="Abilita loss pesata per bilanciare lo stress")
    parser.add_argument("--losocv", action="store_true", help="Esegue Leave-One-Subject-Out Cross-Validation su tutti i 15 soggetti")
    main(parser.parse_args())
