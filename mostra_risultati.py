import os
import sys
import glob
import pickle
from collections import defaultdict

OUT_FILE = "REPORT_COMPLETO_RISULTATI.txt"

def log(msg, f_out):
    print(msg)
    f_out.write(msg + "\n")

def main():
    with open(OUT_FILE, "w", encoding="utf-8") as f_out:
        log("=" * 85, f_out)
        log("       REPORT COMPLETO ESPERIMENTI, RUN E DATASET SUL SERVER", f_out)
        log("=" * 85, f_out)

        # 1. TABELLA DEGLI 11 DATASET
        log("\n" + "#" * 85, f_out)
        log(" 1. TABELLA CORPUS PRE-TRAINING (11 DATASET E4SELFLEARNING)", f_out)
        log("#" * 85, f_out)

        meta_file = r"data\preprocessed\unsegmented_unlabelled\metadata.pkl"
        if os.path.exists(meta_file):
            try:
                with open(meta_file, "rb") as f:
                    meta = pickle.load(f)
                paths = meta.get("sessions_paths", list(meta.get("sessions_info", {}).keys()))
                log(f"[OK] metadata.pkl caricato: {len(paths)} sessioni totali.", f_out)
                
                contexts = {
                    "adarp": "Free-living (Alcohol & Stress) | Real-world",
                    "big-ideas": "Ambulatory Glycemic Study | Real-world",
                    "dati_preelaborati": "BigIdeas Lab Glycemic Study | Real-world",
                    "in-gauge_en-gage": "Indoor Office Engagement | Real-world",
                    "k_emocon": "Debate Interaction | Laboratory",
                    "pgg_dalia": "Daily Activities Protocol | Real-world",
                    "spd": "Physiological Stress Protocol | Laboratory",
                    "stress_detection_nurses_hospital": "Hospital Nurses Shift | Real-world",
                    "toadstool": "Video Game Play | Laboratory",
                    "ue4w": "Everyday Wearable Monitoring | Real-world",
                    "weee": "Workplace Emotion Evaluation | Real-world",
                    "wesad": "Controlled TSST Stress Protocol | Laboratory",
                    "wesd": "Wearable Stress Dataset | Laboratory"
                }
                ds_stats = defaultdict(lambda: {"windows": 0, "subjects": set()})
                for p in paths:
                    norm_p = p.replace("\\", "/")
                    parts = norm_p.split("/")
                    ds_name = "unknown"
                    for k in contexts.keys():
                        if k in norm_p.lower():
                            ds_name = k
                            break
                    if ds_name == "unknown":
                        ds_name = parts[-3] if len(parts) >= 3 else parts[0]
                    sub = parts[-2] if len(parts) >= 2 else "s0"
                    ds_stats[ds_name]["windows"] += 1
                    ds_stats[ds_name]["subjects"].add(sub)

                header = f"{'Dataset':<32} | {'Soggetti':<10} | {'Finestre/File':<15} | {'Contesto & Setting'}"
                log("\n" + header, f_out)
                log("-" * 85, f_out)
                tot_win = 0
                for ds, st in sorted(ds_stats.items()):
                    n_w = st["windows"]
                    n_s = len(st["subjects"])
                    tot_win += n_w
                    ctx = contexts.get(ds, "General Wearable")
                    log(f"{ds:<32} | {n_s:<10} | {n_w:<15} | {ctx}", f_out)
                log("-" * 85, f_out)
                log(f"TOTALE: {len(ds_stats)} Dataset | {tot_win} Finestre/Sessioni complessive", f_out)
            except Exception as e:
                log(f"[ERRORE lettura metadata]: {e}", f_out)
        else:
            log(f"[AVVISO] metadata.pkl non trovato in {meta_file}", f_out)

        # 2. RISULTATI UFFICIALI BENCHMARK (risultati_benchmark/)
        log("\n" + "#" * 85, f_out)
        log(" 2. RISULTATI UFFICIALI BENCHMARK (LOSOCV SALVATI IN risultati_benchmark)", f_out)
        log("#" * 85, f_out)

        bench_dirs = glob.glob("risultati_benchmark/*")
        if bench_dirs:
            summary_table = []
            for b_dir in sorted(bench_dirs):
                if not os.path.isdir(b_dir):
                    continue
                exp_name = os.path.basename(b_dir)
                rep_file = os.path.join(b_dir, "report_finale.txt")
                json_file = os.path.join(b_dir, "losocv_folds.json")
                csv_file = os.path.join(b_dir, "epoch_history.csv")
                
                log(f"\n--- ESPERIMENTO: {exp_name.upper()} ({b_dir}) ---", f_out)
                
                # Leggi report_finale se presente
                if os.path.exists(rep_file):
                    with open(rep_file, "r", encoding="utf-8", errors="ignore") as f_rep:
                        rep_text = f_rep.read().strip()
                        for r_line in rep_text.split("\n"):
                            log(f"   {r_line}", f_out)
                
                # Leggi losocv_folds.json per statistiche aggregate
                if os.path.exists(json_file):
                    try:
                        import json
                        with open(json_file, "r", encoding="utf-8") as f_j:
                            folds_data = json.load(f_j)
                        accs = [f["acc"] for f in folds_data]
                        f1_stresses = [f["f1_stress"] for f in folds_data]
                        macro_f1s = [f["macro_f1"] for f in folds_data]
                        
                        import numpy as np
                        m_acc, s_acc = np.mean(accs), np.std(accs)
                        m_f1, s_f1 = np.mean(f1_stresses), np.std(f1_stresses)
                        m_mf1, s_mf1 = np.mean(macro_f1s), np.std(macro_f1s)
                        
                        summary_table.append({
                            "exp": exp_name,
                            "folds": len(folds_data),
                            "acc": f"{m_acc:.2f}% ± {s_acc:.2f}%",
                            "f1": f"{m_f1:.2f}% ± {s_f1:.2f}%",
                            "macro_f1": f"{m_mf1:.2f}% ± {s_mf1:.2f}%"
                        })
                    except Exception as e:
                        log(f"   [Errore lettura json folds: {e}]", f_out)
                elif os.path.exists(csv_file):
                    log(f"   [Trovato solo epoch_history.csv ({os.path.getsize(csv_file)} byte)]", f_out)

            if summary_table:
                log("\n" + "=" * 85, f_out)
                log("   TABELLA RIASSUNTIVA COMPARATIVA DEI MODELLI (LOSOCV)", f_out)
                log("=" * 85, f_out)
                h_fmt = f"{'Esperimento':<25} | {'Folds':<6} | {'Accuracy Media':<18} | {'F1-Stress Medio':<18} | {'Macro-F1 Medio':<18}"
                log(h_fmt, f_out)
                log("-" * 85, f_out)
                for row in summary_table:
                    r_fmt = f"{row['exp']:<25} | {row['folds']:<6} | {row['acc']:<18} | {row['f1']:<18} | {row['macro_f1']:<18}"
                    log(r_fmt, f_out)
                log("-" * 85, f_out)
        else:
            log("[AVVISO] Nessuna cartella trovata in 'risultati_benchmark/'.", f_out)

        # 3. RUN DOWNSTREAM STORICI (WESAD & HOSSEINI in downstream/)
        log("\n" + "#" * 85, f_out)
        log(" 3. RUN DOWNSTREAM ARCHIVIATI (in downstream/)", f_out)
        log("#" * 85, f_out)

        downstream_dirs = glob.glob("downstream/*/*")
        if downstream_dirs:
            for run_dir in sorted(downstream_dirs):
                if not os.path.isdir(run_dir):
                    continue
                files = os.listdir(run_dir)
                non_txt = [fn for fn in files if not fn.endswith(".txt")]
                txt_files = [fn for fn in files if fn.endswith(".txt")]
                log(f" * {run_dir:<40} : {len(files)} file ({', '.join(non_txt) if non_txt else 'solo architetture .txt'})", f_out)

        # 4. CHECKPOINT MODELLI PRE-TRAINED
        log("\n" + "#" * 85, f_out)
        log(" 4. MODELLI PRE-TRAINED (.pt) PRESENTI", f_out)
        log("#" * 85, f_out)
        pts = glob.glob("*.pt") + glob.glob("output_pretrain*/**/*.pt", recursive=True) + glob.glob("checkpoints/**/*.pt", recursive=True)
        for p in pts:
            sz = os.path.getsize(p) / (1024*1024)
            log(f"  * {p:<65} ({sz:>7.2f} MB)", f_out)

        log("\n" + "=" * 85, f_out)
        log(f" REPORT GENERATO! Salvato nel file: {OUT_FILE}", f_out)
        log("=" * 85 + "\n", f_out)

if __name__ == "__main__":
    main()
