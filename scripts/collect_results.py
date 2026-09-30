import os
import sys
import glob
import json
import pickle
import re

def print_header(title):
    print("\n" + "="*85)
    print(f"   {title}")
    print("="*85)

def scan_datasets():
    print_header("1. STATO DATASET PREPROCESSATI SUL SERVER")
    candidate_meta = [
        r"data\preprocessed\unsegmented_unlabelled\metadata.pkl",
        r"data\preprocessed\sl512_ss256_unlabelled\metadata.pkl",
        r"..\data\preprocessed\unsegmented_unlabelled\metadata.pkl",
        r"C:\Users\Admin\data\preprocessed\unsegmented_unlabelled\metadata.pkl",
        r"C:\Users\Admin\tesi_maria_aggiustata\data\preprocessed\unsegmented_unlabelled\metadata.pkl"
    ]
    meta_path = None
    for p in candidate_meta:
        if os.path.exists(p):
            meta_path = p
            break

    if meta_path:
        print(f"[OK] metadata.pkl trovato in: {meta_path}")
        try:
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
            sessions = meta.get("sessions_info", meta.get("sessions_paths", []))
            print(f"  -> Totale sessioni registrate: {len(sessions)}")
            
            # Conta per dataset
            ds_counts = {}
            for item in (meta.get("sessions_paths") or list(sessions.keys())):
                p_norm = item.replace("\\", "/")
                parts = p_norm.split("/")
                ds_name = parts[-3] if len(parts) >= 3 else parts[0]
                ds_counts[ds_name] = ds_counts.get(ds_name, 0) + 1
            
            print(f"  -> Dataset distinti rilevati ({len(ds_counts)}):")
            for ds, count in sorted(ds_counts.items()):
                print(f"     * {ds:<30} : {count:>5} file/sessioni")
        except Exception as e:
            print(f"  [WARN] Errore lettura metadata: {e}")
    else:
        print("[AVVISO] Nessun metadata.pkl trovato nei percorsi standard.")
        # Cerca cartelle in data/
        for root, dirs, files in os.walk("data"):
            if "channels.h5" in files or "metadata.pkl" in files:
                print(f"  Trovati dati in: {root} ({len(files)} file)")

def scan_checkpoints():
    print_header("2. CHECKPOINT MODELLI (.pt) SALVATI")
    pts = glob.glob("**/*.pt", recursive=True)
    if not pts:
        # Cerca anche nella cartella superiore
        pts = glob.glob("../*.pt") + glob.glob("../**/*.pt")
    
    if pts:
        print(f"Trovati {len(pts)} file di pesi/checkpoint:")
        for p in sorted(pts):
            sz_mb = os.path.getsize(p) / (1024 * 1024)
            mtime = os.path.getmtime(p)
            import datetime
            dt = datetime.datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M')
            print(f"  * {p:<60} | {sz_mb:>8.2f} MB | {dt}")
    else:
        print("  Nessun file .pt trovato.")

def scan_logs_and_results():
    print_header("3. RISULTATI SPERIMENTALI (LOG, TXT, JSON, CSV)")
    log_files = glob.glob("**/*.log", recursive=True) + glob.glob("**/*.txt", recursive=True) + glob.glob("**/results*.json", recursive=True) + glob.glob("**/metrics*.csv", recursive=True)
    found_any = False
    for lf in sorted(log_files):
        try:
            with open(lf, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
            # Cerca righe rilevanti
            metric_lines = []
            for l in lines:
                lower = l.lower()
                if any(k in lower for k in ["report finale", "accuratezza media", "f1-stress medio", "macro-f1 medio", "pooled", "classification report", "acc:"]):
                    metric_lines.append(l.strip())
            if metric_lines:
                found_any = True
                print(f"\n[FILE] {lf}")
                for ml in metric_lines[-10:]:
                    print(f"   {ml}")
        except Exception:
            pass

    if not found_any:
        print("  Nessun log con metriche testuali trovato in cartelle standard.")

def scan_notebooks():
    print_header("4. RISULTATI SALVATI NEI NOTEBOOK JUPYTER (.ipynb)")
    nbs = glob.glob("**/*.ipynb", recursive=True)
    for nb_path in sorted(nbs):
        if ".ipynb_checkpoints" in nb_path:
            continue
        try:
            with open(nb_path, "r", encoding="utf-8") as f:
                nb = json.load(f)
            tables = []
            for cell in nb.get("cells", []):
                for out in cell.get("outputs", []):
                    txt = "".join(out.get("text", []))
                    if any(k in txt for k in ["Accuracy", "Acc=", "F1-Score", "Macro F1", "REPORT FINALE"]):
                        for line in txt.split("\n"):
                            if any(k in line for k in ["Acc", "F1", "Model", "only_", "bigideas", "all_datasets", "Corponi", "FEMBA", "BIOT", "SimMTM"]):
                                line_clean = line.strip()
                                if line_clean and len(line_clean) > 3:
                                    tables.append(line_clean)
            if tables:
                print(f"\n[NOTEBOOK] {nb_path}")
                # Mostra le righe più significative
                for row in tables[:25]:
                    print(f"   {row}")
                if len(tables) > 25:
                    print(f"   ... (altre {len(tables)-25} righe)")
        except Exception:
            pass

def main():
    print("\n" + "#"*85)
    print("      REPORT AUTOMATICO STATO ESPERIMENTI E RISULTATI")
    print("#"*85)
    print(f"Cartella di esecuzione: {os.getcwd()}")
    scan_datasets()
    scan_checkpoints()
    scan_logs_and_results()
    scan_notebooks()
    print("\n" + "="*85)
    print("   SCANSIONE COMPLETATA")
    print("="*85 + "\n")

if __name__ == "__main__":
    main()
