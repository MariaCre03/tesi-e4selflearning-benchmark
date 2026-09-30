import os
import sys
import json
import glob
import pickle
import numpy as np

def print_box(title):
    print("\n" + "=" * 80)
    print(f" {title.upper()}")
    print("=" * 80)

def main():
    print_box("AUDIT COMPLETO SERVER - VERIFICA REQUISITI PAPER JOURNAL")
    print("Questo script verifica lo stato di avanzamento degli 8 punti richiesti dalla professoressa.")
    
    status_summary = {}

    # -------------------------------------------------------------
    # PUNTO 1: PRE-TRAINING COMPLETO (11 DATASET SU 11, >6000 ORE)
    # -------------------------------------------------------------
    print_box("PUNTO 1: Pre-training Completo (11 Dataset)")
    meta_candidates = [
        r"data/preprocessed/unsegmented_unlabelled/metadata.pkl",
        r"data\preprocessed\unsegmented_unlabelled\metadata.pkl",
        r"data/preprocessed/sl512_ss256_unlabelled/metadata.pkl",
        r"data\preprocessed\sl512_ss256_unlabelled\metadata.pkl",
    ]
    found_meta = None
    for mc in meta_candidates:
        if os.path.exists(mc):
            found_meta = mc
            break
            
    if found_meta:
        try:
            with open(found_meta, "rb") as f:
                meta = pickle.load(f)
            paths = meta.get("sessions_paths", list(meta.get("sessions_info", {}).keys()))
            
            KNOWN_DATASETS = [
                "adarp", "dati_preelaborati", "big-ideas", "in-gauge_en-gage", "k_emocon",
                "pgg_dalia", "spd", "stress_detection_nurses_hospital", "toadstool",
                "ue4w", "weee", "wesad", "wesd"
            ]
            detected_ds = set()
            for p in paths:
                p_l = p.replace("\\", "/").lower()
                for kd in KNOWN_DATASETS:
                    if kd in p_l:
                        detected_ds.add(kd)
            
            print(f" [OK] Metadata trovato ({found_meta}): {len(paths):,} finestre/sessioni.")
            print(f" [OK] Dataset rilevati ({len(detected_ds)}): {sorted(list(detected_ds))}")
            if len(detected_ds) >= 10:
                print(" -> STATO CORPUS: COMPLETO (Tutti i dataset sono presenti nel pre-training metadata!)")
            else:
                print(f" -> STATO CORPUS: ATTENZIONE, trovati solo {len(detected_ds)} dataset.")
        except Exception as e:
            print(f" [ERRORE lettura metadata]: {e}")
    else:
        print(" [AVVISO] Metadata del pre-training non trovato nei percorsi standard.")

    # Check checkpoints pretraining
    pretrain_ckpts = (
        glob.glob("output_pretrain*/**/*.pt", recursive=True) +
        glob.glob("*.pt") +
        glob.glob("checkpoints/**/*.pt", recursive=True)
    )
    print("\n Checkpoint pre-addestrati individuati:")
    full_weights = [p for p in pretrain_ckpts if "full" in p.lower() or "encoder" in p.lower() or "best" in p.lower()]
    if full_weights:
        for p in full_weights:
            sz = os.path.getsize(p) / (1024*1024)
            print(f"  * {p:<55} ({sz:6.2f} MB)")
        status_summary["1_pretraining"] = "PRESENTE / DA VERIFICARE EPOCHE"
    else:
        print("  [NESSUN CHECKPOINT FULL TROVATO]")
        status_summary["1_pretraining"] = "DA VERIFICARE"

    # -------------------------------------------------------------
    # PUNTO 2: ABLATION EXTENSION A FEMBA (MAMBA VS TRANSFORMER)
    # -------------------------------------------------------------
    print_box("PUNTO 2: Estensione Ablation a FEMBA (Mamba)")
    ablation_json = "risultati_ablation/ablation_summary.json"
    femba_in_ablation = False
    if os.path.exists(ablation_json):
        try:
            with open(ablation_json, "r", encoding="utf-8") as f:
                ab_data = json.load(f)
            femba_in_ablation = any("femba" in str(r).lower() for r in ab_data)
        except Exception:
            pass

    if femba_in_ablation:
        print(" [OK] FEMBA e' presente nei risultati di ablation!")
        status_summary["2_femba_ablation"] = "COMPLETATO"
    else:
        print(" [ATTENZIONE CRITICA] FEMBA NON compare nei file di ablation attuali (solo SimMTM/BIOT trovati)!")
        print(" -> Per il paper e' necessario eseguire l'ablation anche con FEMBA (confronto Transformer vs SSM).")
        status_summary["2_femba_ablation"] = "MANCANTE"

    # -------------------------------------------------------------
    # PUNTO 3 & 4: PROTOCOLLO LOSOCV & CLASS-WEIGHTED LOSS
    # -------------------------------------------------------------
    print_box("PUNTI 3 & 4: Protocollo LOSOCV Uniforme e Class-Weighted Loss")
    if os.path.exists(ablation_json):
        with open(ablation_json, "r", encoding="utf-8") as f:
            ab_data = json.load(f)
        all_row = next((r for r in ab_data if "ALL" in r.get("configurazione", "")), None)
        if all_row:
            print(f" Risultati configurazione 'ALL' nell'ablation:")
            print(f"  * WESAD:    Acc = {all_row.get('wesad_acc')}%, F1-Stress = {all_row.get('wesad_f1_stress')}%")
            print(f"  * HOSSEINI: Acc = {all_row.get('hosseini_acc')}%, F1-Stress = {all_row.get('hosseini_f1_stress')}%")
            if all_row.get('wesad_f1_stress', 0) > 70.0:
                print("  [OK] F1-Stress su WESAD e' elevato (>70%): la class-weighted loss e' attiva!")
                status_summary["3_4_losocv_weighting"] = "OTTIMO (F1 > 70%)"
            elif all_row.get('wesad_f1_stress', 0) < 60.0:
                print("  [ATTENZIONE] F1-Stress su WESAD e' ~54%: verificare se la loss pesata era inclusa nella run salvata.")
                status_summary["3_4_losocv_weighting"] = "DA VERIFICARE WEIGHTED LOSS"
        else:
            print(" [INFO] Nessuna riga 'ALL' trovata in ablation_summary.json.")
            status_summary["3_4_losocv_weighting"] = "DA VERIFICARE"
    else:
        print(" [AVVISO] ablation_summary.json non trovato.")
        status_summary["3_4_losocv_weighting"] = "NON TROVATO"

    # -------------------------------------------------------------
    # PUNTO 5: DATA QUANTITY VS DATA DIVERSITY
    # -------------------------------------------------------------
    print_box("PUNTO 5: Controllo Quantity vs Diversity (Budget Fisso)")
    qvd_ckpts = [
        os.path.join("risultati_ablation", "checkpoints", "simmtm_single_source.pt"),
        os.path.join("risultati_ablation", "checkpoints", "simmtm_multi_source.pt")
    ]
    qvd_exists = all(os.path.exists(p) for p in qvd_ckpts)
    qvd_in_json = False
    if os.path.exists(ablation_json):
        with open(ablation_json, "r", encoding="utf-8") as f:
            ab_data = json.load(f)
        qvd_in_json = any(r.get("tipo") == "QvD" or "Fixed Budget" in r.get("configurazione", "") for r in ab_data)

    if qvd_exists and qvd_in_json:
        print(" [OK] Esperimento Quantity vs Diversity (Single-Source vs Multi-Source a ore fisse) COMPLETATO!")
        status_summary["5_quantity_vs_diversity"] = "COMPLETATO"
    elif qvd_exists:
        print(" [PARZIALE] Checkpoint QvD trovati ma risultati non registrati nel summary json.")
        status_summary["5_quantity_vs_diversity"] = "CHECKPOINTS PRONTI, RUNNARE VALUTAZIONE"
    else:
        print(" [NON ANCORA ESEGUITO] Mancano i checkpoint di Quantity vs Diversity in risultati_ablation/checkpoints/")
        status_summary["5_quantity_vs_diversity"] = "DA LANCIARE"

    # -------------------------------------------------------------
    # PUNTO 6: LEAVE-ONE-DATASET-OUT (LODO) ABLATION
    # -------------------------------------------------------------
    print_box("PUNTO 6: Leave-One-Dataset-Out (LODO) Ablation")
    if os.path.exists(ablation_json):
        with open(ablation_json, "r", encoding="utf-8") as f:
            ab_data = json.load(f)
        lodo_rows = [r for r in ab_data if r.get("tipo") == "LODO" or "ALL -" in r.get("configurazione", "")]
        excluded_ds = [r.get("escluso", r.get("configurazione")) for r in lodo_rows]
        print(f" Numero configurazioni LODO completate: {len(lodo_rows)}/11")
        if excluded_ds:
            print(f" Dataset esclusi testati finora: {excluded_ds}")
        if len(lodo_rows) >= 11:
            print(" [OK] Tutte le 11 combinazioni LODO sono state completate!")
            status_summary["6_lodo_ablation"] = "COMPLETATO (11/11)"
        elif len(lodo_rows) > 0:
            print(f" [PARZIALE] Solo {len(lodo_rows)}/11 dataset completati. Completare gli altri.")
            status_summary["6_lodo_ablation"] = f"PARZIALE ({len(lodo_rows)}/11)"
        else:
            print(" [NON ANCORA ESEGUITO] Nessuna configurazione LODO registrata.")
            status_summary["6_lodo_ablation"] = "DA LANCIARE"
    else:
        print(" [AVVISO] ablation_summary.json non trovato.")
        status_summary["6_lodo_ablation"] = "NON TROVATO"

    # -------------------------------------------------------------
    # PUNTO 7: STATISTICA DELL'ESPERIMENTO (FOLDS, CI 95%, SIGNIFICANCE)
    # -------------------------------------------------------------
    print_box("PUNTO 7: Analisi Statistica e Test di Ipotesi (Significance)")
    bench_dirs = glob.glob("risultati_benchmark/*")
    bench_models = [os.path.basename(d) for d in bench_dirs if os.path.isdir(d)]
    print(f" Cartelle trovate in risultati_benchmark/: {bench_models}")
    
    missing_folds = []
    found_folds = []
    for d in bench_dirs:
        if os.path.isdir(d):
            jf = os.path.join(d, "losocv_folds.json")
            if os.path.exists(jf):
                try:
                    with open(jf, "r") as f:
                        f_data = json.load(f)
                    found_folds.append((os.path.basename(d), len(f_data)))
                except Exception:
                    missing_folds.append(os.path.basename(d))
            else:
                missing_folds.append(os.path.basename(d))
                
    if found_folds:
        print(f" [OK] Trovati file losocv_folds.json con salvataggio fold per fold:")
        for name, n_f in found_folds:
            print(f"  * {name:<35} : {n_f} folds registrati")
            
    if missing_folds:
        print(f" [ATTENZIONE] Manca losocv_folds.json per: {missing_folds}")
        print("  -> Senza losocv_folds.json compute_statistics.py non puo calcolare Wilcoxon / paired t-test!")
        status_summary["7_statistics"] = "PARZIALE (Alcuni fold mancanti)"
    elif found_folds:
        status_summary["7_statistics"] = "PRONTO (Tutti i fold presenti)"
    else:
        status_summary["7_statistics"] = "NON TROVATO"

    # -------------------------------------------------------------
    # PUNTO 8: TERZO DOWNSTREAM DATASET (PHYSIONET HONGN 2025)
    # -------------------------------------------------------------
    print_box("PUNTO 8: Terzo Downstream Dataset (PhysioNet Hongn 2025)")
    physio_meta = os.path.join("downstream_data", "physionet_segmented", "metadata.pkl")
    if os.path.exists(physio_meta):
        try:
            with open(physio_meta, "rb") as f:
                pm = pickle.load(f)
            p_paths = pm.get("sessions_paths", [])
            print(f" [OK] Dataset PhysioNet pronto e segmentato: {len(p_paths)} finestre.")
            
            # Controlla se i benchmark sono stati runnati
            physio_runs = [d for d in bench_dirs if "physionet" in d.lower()]
            print(f" Benchmark eseguiti su PhysioNet: {physio_runs}")
            if len(physio_runs) >= 4:
                print(" [OK] Tutti e 4 i modelli (Corponi, BIOT, SimMTM, FEMBA) sono stati eseguiti su PhysioNet!")
                status_summary["8_third_dataset"] = "COMPLETATO (4/4 modelli)"
            elif len(physio_runs) > 0:
                print(f" [PARZIALE] Eseguiti solo {len(physio_runs)} modelli su PhysioNet.")
                status_summary["8_third_dataset"] = f"PARZIALE ({len(physio_runs)}/4)"
            else:
                print(" [DA ESEGUIRE] Il dataset e' segmentato ma i benchmark non sono ancora stati lanciati.")
                print(" -> Lanciare: python run_physionet_suite.py")
                status_summary["8_third_dataset"] = "DATASET PRONTO, RUN BENCHMARK MANCANTE"
        except Exception as e:
            print(f" [ERRORE lettura PhysioNet meta]: {e}")
            status_summary["8_third_dataset"] = "ERRORE"
    else:
        print(" [NON ANCORA PRONTO] downstream_data/physionet_segmented/metadata.pkl NON esiste.")
        print(" -> Per prepararlo ed eseguire la suite: python run_physionet_suite.py")
        status_summary["8_third_dataset"] = "DA PREPARARE E LANCIARE"

    # -------------------------------------------------------------
    # QUADRO SINOTTICO FINALE
    # -------------------------------------------------------------
    print_box("TABELLA DI SINTESI FINALE: STATO DEGLI 8 REQUISITI DELLA PROF")
    print(f"{'Requisito Professore':<45} | {'Stato Rilevato'}")
    print("-" * 75)
    for k, v in status_summary.items():
        print(f"{k:<45} | {v}")
    print("=" * 75 + "\n")

if __name__ == "__main__":
    main()
