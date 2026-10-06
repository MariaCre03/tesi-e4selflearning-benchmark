import os
import re

target = "run_ablation_study.py"
if not os.path.exists(target):
    print(f"[ERRORE] File non trovato: {target}")
    exit(1)

with open(target, "r", encoding="utf-8") as f:
    code = f.read()

# 1. FIX SOGGETTI PHYSIONET
physio_check = 'elif "physionet" in dataset_name.lower():'
if physio_check not in code:
    old_else = '    else:\n        all_subs = np.array([p.replace("\\\\", "/").split("/")[-2].split("_")[0] for p in all_paths])\n        unique_subs = sorted(list(np.unique(all_subs)))'
    new_branch = '    elif "physionet" in dataset_name.lower():\n        all_subs = np.array([p.replace("\\\\", "/").split("/")[-2] for p in all_paths])\n        unique_subs = sorted(list(np.unique(all_subs)))\n' + old_else
    if old_else in code:
        code = code.replace(old_else, new_branch, 1)
        print("[OK] 1. Fix estrazione soggetti PhysioNet applicato!")
    else:
        # Fallback regex
        pattern = r'(\s+else:\s+all_subs = np\.array\(\[p\.replace\([^\n]+\n\s+unique_subs = sorted[^\n]+\))'
        m = re.search(pattern, code)
        if m:
            indent = "    "
            repl = f'{indent}elif "physionet" in dataset_name.lower():\n{indent}    all_subs = np.array([p.replace("\\\\", "/").split("/")[-2] for p in all_paths])\n{indent}    unique_subs = sorted(list(np.unique(all_subs)))\n' + m.group(1)
            code = code[:m.start()] + repl + code[m.end():]
            print("[OK] 1. Fix PhysioNet applicato via regex!")
        else:
            print("[AVVISO] Non è stato possibile applicare il fix PhysioNet automaticamente.")
else:
    print("[INFO] 1. Fix PhysioNet già presente!")

# 2. FIX CACHE FOLDS
cache_check = 'Fold gia\' calcolati e presenti in'
if cache_check not in code and 'Fold già calcolati e presenti in' not in code:
    marker = '    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")\n    with open(os.path.join(data_dir, "metadata.pkl"), "rb") as f:'
    cache_code = '''    if folds_out_path and os.path.exists(folds_out_path):
        try:
            with open(folds_out_path, "r", encoding="utf-8") as fp:
                saved_folds = json.load(fp)
            if len(saved_folds) > 0:
                print(f"    [SKIP] Fold gia' calcolati e presenti in: {folds_out_path}", flush=True)
                accs = [f["accuracy"] for f in saved_folds if "accuracy" in f and f["accuracy"] is not None]
                f1s  = [f["f1_stress"] for f in saved_folds if "f1_stress" in f and f["f1_stress"] is not None]
                mf1s = [f["macro_f1"] for f in saved_folds if "macro_f1" in f and f["macro_f1"] is not None]
                if len(accs) > 0:
                    return {"acc": round(float(np.mean(accs)), 2),
                            "f1_stress": round(float(np.mean(f1s)), 2),
                            "macro_f1": round(float(np.mean(mf1s)), 2),
                            "folds": saved_folds}
        except Exception:
            pass

''' + marker
    if marker in code:
        code = code.replace(marker, cache_code, 1)
        print("[OK] 2. Cache salvataggio fold applicata (salterà i fold già fatti)!")
    else:
        print("[AVVISO] Marker per cache fold non trovato.")
else:
    print("[INFO] 2. Cache fold già presente!")

with open(target, "w", encoding="utf-8") as f:
    f.write(code)

print("\n[SUCCESSO] run_ablation_study.py aggiornato e pronto per l'esecuzione!")
