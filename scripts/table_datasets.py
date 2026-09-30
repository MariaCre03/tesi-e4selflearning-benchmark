import os
import sys
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict

# Dizionario dei contesti di acquisizione scientifici per ciascun dataset di E4SelfLearning
DATASET_CONTEXTS = {
    "adarp": {
        "full_name": "ADARP",
        "context": "Free-living / Ambulatory (Alcohol Craving & Stress)",
        "setting": "Real-world",
        "description": "Monitoraggio continuo di soggetti con disturbo da uso di alcol nella vita quotidiana"
    },
    "big-ideas": {
        "full_name": "Big-Ideas Lab",
        "context": "Free-living / Ambulatory (Glycemic & Wearable Study)",
        "setting": "Real-world",
        "description": "Variabilità glicemica e biosensori in condizioni di vita libera"
    },
    "in-gauge_en-gage": {
        "full_name": "In-Gauge & En-Gage",
        "context": "Office / Indoor Work Environment",
        "setting": "Real-world / Office",
        "description": "Comportamento, comfort ed emozioni di occupanti in ambienti di lavoro chiusi"
    },
    "k_emocon": {
        "full_name": "K-EmoCon",
        "context": "Semi-controlled / Debate Interaction",
        "setting": "Laboratory",
        "description": "Riconoscimento delle emozioni durante dibattiti interpersonali tra partecipanti"
    },
    "pgg_dalia": {
        "full_name": "PPG-DaLiA",
        "context": "Protocolled Daily Life Activities",
        "setting": "Real-world",
        "description": "Monitoraggio delle attività quotidiane guidate (guida, lavoro al PC, ciclismo, pranzo, camminata)"
    },
    "stress_detection_nurses_hospital": {
        "full_name": "Nurse Stress (Hospital)",
        "context": "Hospital Work Shifts (High Stress Environment)",
        "setting": "Real-world",
        "description": "Rilevamento dello stress continuo durante i turni lavorativi degli infermieri in ospedale"
    },
    "spd": {
        "full_name": "SPD",
        "context": "Physiological Stress Detection",
        "setting": "Laboratory",
        "description": "Protocollo sperimentale di induzione dello stress e monitoraggio fisiologico"
    },
    "toadstool": {
        "full_name": "Toadstool",
        "context": "Video Game Play (Super Mario Bros)",
        "setting": "Laboratory",
        "description": "Risposte fisiologiche ed emotive a fallimenti e successi durante il gameplay"
    },
    "ue4w": {
        "full_name": "UE4W",
        "context": "Everyday Life Wearable Monitoring",
        "setting": "Real-world",
        "description": "Monitoraggio esteso di segnali fisiologici con Empatica E4 in contesto non vincolato"
    },
    "weee": {
        "full_name": "WEEE",
        "context": "Workplace Emotional Evaluation",
        "setting": "Real-world / Office",
        "description": "Valutazione dell'esperienza emotiva in contesti occupazionali"
    },
    "wesad": {
        "full_name": "WESAD (Pre-train)",
        "context": "Controlled Stress Protocol (TSST & Amusement)",
        "setting": "Laboratory",
        "description": "Stress cognitivo/sociale (Trier Social Stress Test) e rilassamento controllato in laboratorio"
    },
    "wesd": {
        "full_name": "WESD",
        "context": "Wearable Stress Dataset",
        "setting": "Laboratory",
        "description": "Studio fisiologico guidato per rilevazione dello stress"
    }
}

DEFAULT_CHANNELS = "ACC (x,y,z), BVP, EDA, TEMP"

def find_metadata_path(base_paths):
    for p in base_paths:
        if os.path.exists(p):
            return p
    return None

def analyze_dataset_corpus(meta_path):
    print(f"[INFO] Caricamento metadati da: {meta_path}...")
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    paths = meta.get("sessions_paths", [])
    if len(paths) == 0:
        print("[ERRORE] Nessun percorso trovato in metadata.pkl")
        return None

    print(f"[INFO] Trovati {len(paths):,} segmenti totali nel corpus.")

    # Raggruppamento per dataset
    ds_stats = defaultdict(lambda: {"windows": 0, "subjects": set(), "sample_file": None})

    for p in paths:
        norm_p = p.replace("\\", "/")
        parts = norm_p.split("/")
        
        # Identificazione del nome del dataset e del soggetto
        # Esempio: .../sl512_ss256_unlabelled/<dataset_name>/<subject_folder>/segment_XXX.pkl
        # Oppure: .../sl512_ss256_unlabelled/<dataset_name>_<subject_id>/segment_XXX.pkl
        ds_name = "unknown"
        sub_id = "unknown"
        
        if "sl512_ss256_unlabelled" in parts:
            idx = parts.index("sl512_ss256_unlabelled")
            if len(parts) > idx + 1:
                rel = parts[idx + 1]
                # Verifica se rel corrisponde a un dataset noto
                matched = False
                for known_k in sorted(DATASET_CONTEXTS.keys(), key=lambda x: -len(x)):
                    if rel == known_k or rel.startswith(known_k + "_") or rel.startswith(known_k + "/"):
                        ds_name = known_k
                        # Il soggetto e' la parte restante o la cartella successiva
                        if len(parts) > idx + 2 and not parts[idx+2].endswith(".pkl"):
                            sub_id = parts[idx+2]
                        else:
                            sub_id = rel.replace(known_k, "").strip("_-")
                        matched = True
                        break
                if not matched:
                    if "dati_preelaborati" in norm_p:
                        ds_name = "big-ideas"
                        sub_id = parts[-2]
                    else:
                        ds_name = rel.split("_")[0]
                        sub_id = parts[idx + 2] if len(parts) > idx + 2 and not parts[idx+2].endswith(".pkl") else "sub_0"
        else:
            # Fallback generico
            ds_name = parts[-3] if len(parts) >= 3 else parts[-2]
            sub_id = parts[-2] if len(parts) >= 3 else "sub"

        ds_stats[ds_name]["windows"] += 1
        ds_stats[ds_name]["subjects"].add(sub_id)
        if ds_stats[ds_name]["sample_file"] is None:
            ds_stats[ds_name]["sample_file"] = p

    # Calcolo metriche
    rows = []
    window_length_sec = 512
    step_size_sec = 256  # 50% overlap

    total_windows = 0
    total_hours = 0.0
    total_subjects = 0

    for ds_name, d in sorted(ds_stats.items()):
        n_win = d["windows"]
        n_subs = len(d["subjects"])
        # Ogni finestra consecutiva aggiunge 'step_size_sec' secondi (512s prima + (n-1)*256s)
        # Stima conservativa standard per dataset a finestre sovrapposte:
        dur_sec = n_win * step_size_sec
        dur_hours = dur_sec / 3600.0

        ctx_info = DATASET_CONTEXTS.get(ds_name.lower(), {
            "full_name": ds_name.upper(),
            "context": "General Wearable Study",
            "setting": "Unspecified",
            "description": "Biosignals monitoring"
        })

        total_windows += n_win
        total_hours += dur_hours
        total_subjects += n_subs

        rows.append({
            "Dataset": ctx_info["full_name"],
            "Key": ds_name,
            "Setting": ctx_info["setting"],
            "Soggetti": n_subs,
            "Finestre (512s)": n_win,
            "Ore Totali": round(dur_hours, 1),
            "Segnali": DEFAULT_CHANNELS,
            "Contesto di Acquisizione": ctx_info["context"]
        })

    df = pd.DataFrame(rows)
    return df, total_subjects, total_windows, total_hours

def main():
    candidate_paths = [
        r"C:\Users\Admin\data\preprocessed\sl512_ss256_unlabelled\metadata.pkl",
        r"C:\Users\Admin\data\preprocessed\unsegmented_unlabelled\metadata.pkl",
        r"data\preprocessed\sl512_ss256_unlabelled\metadata.pkl",
        r"..\data\preprocessed\sl512_ss256_unlabelled\metadata.pkl"
    ]

    meta_file = find_metadata_path(candidate_paths)
    if not meta_file:
        print("[AVVISO] metadata.pkl non trovato nei percorsi locali predefiniti.")
        print("Uso: python table_datasets.py [percorso_a_metadata.pkl]")
        if len(sys.argv) > 1:
            meta_file = sys.argv[1]
        else:
            meta_file = candidate_paths[0]

    df, tot_sub, tot_win, tot_hr = analyze_dataset_corpus(meta_file)
    if df is None:
        return

    print("\n" + "="*110)
    print("           TABELLA SINTETICA CORPUS DI PRE-TRAINING (11 DATASET E4SELFLEARNING)")
    print("="*110)
    # Stampa formattata
    format_df = df[["Dataset", "Setting", "Soggetti", "Finestre (512s)", "Ore Totali", "Segnali", "Contesto di Acquisizione"]]
    print(format_df.to_string(index=False))
    print("-" * 110)
    print(f" TOTALE COMPLESSIVO: {len(df)} Dataset | {tot_sub} Soggetti (non-disgiunti) | {tot_win:,} Finestre | ~{tot_hr:,.1f} Ore di Registrazione")
    print("="*110)

    # Salvataggio su file markdown e csv
    out_md = "dataset_pretraining_table.md"
    out_csv = "dataset_pretraining_table.csv"
    
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# Tabella Corpus di Pre-training E4SelfLearning\n\n")
        f.write(format_df.to_markdown(index=False))
        f.write(f"\n\n**Totale**: {len(df)} Dataset, {tot_win:,} finestre (512s, step 256s), ~{tot_hr:,.1f} ore complessive.\n")

    format_df.to_csv(out_csv, index=False)
    print(f"\n[OK] Tabella salvata con successo in:\n  -> {out_md}\n  -> {out_csv}")

if __name__ == "__main__":
    main()
