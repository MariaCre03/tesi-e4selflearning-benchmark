import os
import sys
import glob
import pickle
import pandas as pd
from collections import defaultdict

OUT_FILE = "REPORT_COMPLETO_RISULTATI.txt"

def log(msg, f_out):
    print(msg)
    f_out.write(msg + "\n")

def main():
    with open(OUT_FILE, "w", encoding="utf-8") as f_out:
        log("=" * 90, f_out)
        log("          REPORT COMPLETO DI TUTTI GLI ESPERIMENTI E DATASET SUL SERVER", f_out)
        log("=" * 90, f_out)

        # -------------------------------------------------------------
        # 1. TABELLA RIASSUNTIVA DEGLI 11 DATASET (METADATA.PKL)
        # -------------------------------------------------------------
        log("\n" + "#" * 90, f_out)
        log(" 1. ANALISI DEL CORPUS DEI DATASET (11 DATASET E4SELFLEARNING)", f_out)
        log("#" * 90, f_out)

        meta_candidates = [
            r"data\preprocessed\unsegmented_unlabelled\metadata.pkl",
            r"data\preprocessed\sl512_ss256_unlabelled\metadata.pkl",
            r"..\data\preprocessed\unsegmented_unlabelled\metadata.pkl",
        ]
        meta_file = None
        for p in meta_candidates:
            if os.path.exists(p):
                meta_file = p
                break

        if meta_file:
            log(f"[OK] Trovato metadata.pkl in: {meta_file}", f_out)
            try:
                with open(meta_file, "rb") as f:
                    meta = pickle.load(f)

                paths = meta.get("sessions_paths", [])
                if not paths and "sessions_info" in meta:
                    paths = list(meta["sessions_info"].keys())

                log(f"Totale sessioni/file registrati: {len(paths)}", f_out)

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

                log("\n--- TABELLA DATASET PREPROCESSATI ---", f_out)
                header = f"{'Dataset':<32} | {'Soggetti':<10} | {'Finestre/File':<15} | {'Contesto & Setting'}"
                log(header, f_out)
                log("-" * 90, f_out)
                tot_win = 0
                for ds, st in sorted(ds_stats.items()):
                    n_w = st["windows"]
                    n_s = len(st["subjects"])
                    tot_win += n_w
                    ctx = contexts.get(ds, "General Wearable")
                    log(f"{ds:<32} | {n_s:<10} | {n_w:<15} | {ctx}", f_out)
                log("-" * 90, f_out)
                log(f"TOTALE: {len(ds_stats)} Dataset | {tot_win} Finestre/Sessioni complessive", f_out)

            except Exception as e:
                log(f"[ERRORE lettura metadata]: {e}", f_out)
        else:
            log("[AVVISO] metadata.pkl non trovato nei percorsi locali.", f_out)

        # -------------------------------------------------------------
        # 2. CONTENUTO E RISULTATI DEI RUN DOWNSTREAM (WESAD & HOSSEINI)
        # -------------------------------------------------------------
        log("\n" + "#" * 90, f_out)
        log(" 2. DETTAGLIO RUN E RISULTATI DOWNSTREAM (WESAD & HOSSEINI)", f_out)
        log("#" * 90, f_out)

        downstream_dirs = glob.glob("downstream/*/*")
        if not downstream_dirs:
            downstream_dirs = glob.glob("downstream/*")

        if downstream_dirs:
            for run_dir in sorted(downstream_dirs):
                if not os.path.isdir(run_dir):
                    continue
                log(f"\n" + "-" * 75, f_out)
                log(f" CARTELLE RUN: {run_dir}", f_out)
                log("-" * 75, f_out)

                files = os.listdir(run_dir)
                log(f" File presenti: {', '.join(files)}", f_out)

                for fn in files:
                    fp = os.path.join(run_dir, fn)
                    if fn.endswith(".txt") or fn.endswith(".log") or fn.endswith(".json") or fn.endswith(".yaml") or fn.endswith(".yml"):
                        try:
                            with open(fp, "r", encoding="utf-8", errors="ignore") as f_sub:
                                content = f_sub.read().strip()
                            log(f"\n   >>> [FILE: {fn}] (Lunghezza: {len(content)} caratteri):", f_out)
                            lines = content.split("\n")
                            if len(lines) <= 25:
                                for l in lines:
                                    log(f"       {l}", f_out)
                            else:
                                for l in lines[:15]:
                                    log(f"       {l}", f_out)
                                log(f"       ... [saltate {len(lines)-25} righe centrali] ...", f_out)
                                for l in lines[-10:]:
                                    log(f"       {l}", f_out)
                        except Exception as e:
                            log(f"       [Impossibile leggere {fn}: {e}]", f_out)
                    elif fn.endswith(".pt"):
                        sz_mb = os.path.getsize(fp) / (1024 * 1024)
                        log(f"   >>> [CHECKPOINT: {fn}] Dimensione: {sz_mb:.2f} MB", f_out)

        else:
            log(" Nessuna cartella downstream trovata.", f_out)

        # -------------------------------------------------------------
        # 3. STATO PRETRAIN FULL E ALTRI CHECKPOINT
        # -------------------------------------------------------------
        log("\n" + "#" * 90, f_out)
        log(" 3. CHECKPOINT ROOT E OUTPUT PRETRAIN", f_out)
        log("#" * 90, f_out)
        pts = glob.glob("*.pt") + glob.glob("output_pretrain*/**/*.pt", recursive=True) + glob.glob("checkpoints/**/*.pt", recursive=True)
        for p in pts:
            sz = os.path.getsize(p) / (1024 * 1024)
            log(f"  * {p:<65} ({sz:>7.2f} MB)", f_out)

        log("\n" + "=" * 90, f_out)
        log(f" REPORT GENERATO CON SUCCESSO! Salvato nel file: {OUT_FILE}", f_out)
        log("=" * 90 + "\n", f_out)

if __name__ == "__main__":
    main()
