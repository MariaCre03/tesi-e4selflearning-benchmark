import os
import sys
import io

# Forza stdout e stderr su UTF-8 con fallback sicuro per evitare errori charmap/cp1252
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import pickle
import argparse
import gc
import shutil
import zipfile
import numpy as np

# 1. Configura percorsi
BASE_DIR = r"C:\Users\Admin"
SRC_DIR = os.path.join(BASE_DIR, "test_wesad", "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

RAW_DIR = os.path.join(BASE_DIR, "data", "raw_data", "unlabelled_data")
os.makedirs(RAW_DIR, exist_ok=True)

PREPROCESSED_DIR = os.path.join(BASE_DIR, "data", "preprocessed", "unsegmented_unlabelled")
meta_p = os.path.join(PREPROCESSED_DIR, "metadata.pkl")

print("="*60)
print("[START] AVVIO PREPROCESSING FASE 0 - ADARP & WESAD (11/11 DATASET)")
print("="*60)

# STEP 1: Estrazione archivi principali se presenti in C:\Users\Admin
for z_name, folder_name in [("ADARP.zip", "ADARP"), ("WESAD.zip", "WESAD")]:
    z_path = os.path.join(BASE_DIR, z_name)
    target_f = os.path.join(RAW_DIR, folder_name)
    if os.path.exists(z_path):
        print(f"\n[ESTRAZIONE ARCHIVIO] {z_name} -> {target_f}...")
        with zipfile.ZipFile(z_path, 'r') as z:
            z.extractall(RAW_DIR)
        print(f"  -> {z_name} estratto con successo.")
    else:
        print(f"\n[INFO] {z_name} non in C:\\Users\\Admin (verifico se gia' estratto in {target_f})")

# STEP 2: Preparazione interna ADARP
adarp_base = os.path.join(RAW_DIR, "ADARP")
if os.path.exists(adarp_base):
    # Se c'e' Data.zip interno ad ADARP, estrailo
    adarp_data_zip = os.path.join(adarp_base, "Data.zip")
    if os.path.exists(adarp_data_zip):
        print("  -> Estrazione Data.zip interno ad ADARP...")
        with zipfile.ZipFile(adarp_data_zip, 'r') as z:
            z.extractall(adarp_base)
        print("  -> Data.zip estratto.")

# STEP 3: Preparazione interna WESAD (estrazione SXX_E4_Data.zip)
wesad_base = os.path.join(RAW_DIR, "WESAD")
if os.path.exists(wesad_base):
    print("  -> Controllo ed estrazione degli archivi E4 interni per ciascun soggetto WESAD...")
    wesad_count = 0
    for root, dirs, files in os.walk(wesad_base):
        for f in files:
            if f.endswith("_E4_Data.zip"):
                z_inner = os.path.join(root, f)
                # Estrai nella stessa cartella del soggetto
                try:
                    with zipfile.ZipFile(z_inner, 'r') as z:
                        z.extractall(root)
                    wesad_count += 1
                except Exception as e:
                    print(f"    [ERRORE] Estrazione {f}: {e}")
    if wesad_count > 0:
        print(f"  -> Estratti {wesad_count} archivi E4 interni in WESAD.")
    else:
        print("  -> Archivi E4 interni WESAD gia' estratti o non trovati.")

# STEP 4: Import timebase
from timebase.data import preprocessing as prep
from timebase.data.static import LABEL_COLS, UNLABELLED_DATA_PATHS, CSV_CHANNELS, CORRUPTED_FILES
from timebase.data.preprocessing import REFORMAT_COLLECTION_DICT, get_channel_from_filename, check_faulty_folder
from timebase.utils import h5

# Patch Windows per recast_collection
def windows_recast_collection(args, collection: str, path: str):
    dirs = []
    output_dir_collection = os.path.join(args.path2unlabelled_data, "recast", collection)
    root_dir = os.path.join(args.path2unlabelled_data, path)
    files2dismiss = []
    print(f"  [Scansione file in: {root_dir}]")
    count_processed = 0
    all_csvs = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for f in filenames:
            if f.endswith(".csv") and get_channel_from_filename(f, CSV_CHANNELS):
                all_csvs.append((dirpath, f))
                
    print(f"  [Trovati {len(all_csvs)} file CSV da convertire. Inizio recasting...]")
    for dirpath, filename in all_csvs:
        if dirpath in CORRUPTED_FILES:
            continue
        check_faulty_folder(dirpath=dirpath, files2dismiss=files2dismiss)
        if os.path.join(dirpath, filename) in files2dismiss:
            continue
        
        dirpath_clean = dirpath.replace("\\", "/")
        path_clean = path.replace("\\", "/")
        rel_dir = dirpath_clean.rsplit(f"{path_clean}/", 1)[-1] if f"{path_clean}/" in dirpath_clean else os.path.relpath(dirpath, root_dir)
        
        output_file = os.path.join(
            output_dir_collection,
            rel_dir,
            f"{get_channel_from_filename(filename, CSV_CHANNELS)}.csv",
        ).replace(" ", "_")
        
        out_parent = os.path.dirname(output_file)
        if not os.path.exists(out_parent):
            os.makedirs(out_parent, exist_ok=True)
            dirs.append(out_parent)
            
        if collection in REFORMAT_COLLECTION_DICT.keys():
            REFORMAT_COLLECTION_DICT[collection](
                filename=os.path.join(dirpath, filename),
                output_file=output_file,
            )
        else:
            shutil.copyfile(os.path.join(dirpath, filename), output_file)
            
        count_processed += 1
        if count_processed % 100 == 0:
            print(f"    -> Riformattati {count_processed}/{len(all_csvs)} file CSV...")

    unique_dirs = list(dict.fromkeys(dirs))
    return unique_dirs

# STEP 5: Carica metadata esistente
if os.path.exists(meta_p):
    with open(meta_p, "rb") as f:
        combined_meta = pickle.load(f)
    print(f"\n[METADATA] Caricato metadata esistente con {len(combined_meta['sessions_info'])} sessioni.")
else:
    combined_meta = {"invalid_sessions": [], "sessions_info": {}, "sleep_algorithm": "van_hees"}

args = argparse.Namespace(
    path2unlabelled_data=RAW_DIR,
    output_dir=PREPROCESSED_DIR,
    e4selflearning=True,
    overwrite=True,
    sleep_algorithm="van_hees",
    wear_minimum_minutes=5,
    minimum_recorded_time=15,
    num_workers=2,
    chunksize=1,
    verbose=0,
    seed=1234
)

target_collections = ["adarp", "wesad"]

for col in target_collections:
    path = UNLABELLED_DATA_PATHS[col]
    full_path = os.path.join(RAW_DIR, path)
    if not os.path.exists(full_path):
        print(f"\n[SALTATO] Cartella {col} non trovata in {full_path}")
        continue
        
    print("\n" + "="*45)
    print(f"[PROCESS] INIZIO PREPROCESSING: {col}")
    print("="*45)
    
    s_dirs = windows_recast_collection(args, collection=col, path=path)
    print(f"  [Recast completato] Trovate {len(s_dirs)} cartelle di sessione.")
    
    count_saved = 0
    recast_prefix = os.path.join(RAW_DIR, "recast").replace("\\", "/") + "/"
    
    for i, s_id in enumerate(s_dirs, 1):
        s_id_clean = s_id.replace("\\", "/")
        name = s_id_clean.replace(recast_prefix, "")
        
        if name in combined_meta["sessions_info"]:
            continue
            
        try:
            data, info, too_short = prep.preprocess_dir(args, recording_dir=s_id, labelled=False)
            if not too_short:
                data["labels"] = np.full(len(LABEL_COLS), np.nan, dtype=np.float32)
                sess_dir = os.path.join(PREPROCESSED_DIR, name)
                os.makedirs(sess_dir, exist_ok=True)
                h5.write(os.path.join(sess_dir, "channels.h5"), data, True)
                combined_meta["sessions_info"][name] = info
                count_saved += 1
                if count_saved % 10 == 0 or i == len(s_dirs):
                    with open(meta_p, "wb") as f:
                        pickle.dump(combined_meta, f)
                    print(f"    [{i}/{len(s_dirs)}] Salvata sessione: {name}")
            else:
                combined_meta["invalid_sessions"].append(name)
            del data
            gc.collect()
        except Exception as e:
            print(f"    [WARN] Errore elaborazione sessione {name}: {e}")

    with open(meta_p, "wb") as f:
        pickle.dump(combined_meta, f)
        
    recast_col = os.path.join(RAW_DIR, "recast", col)
    shutil.rmtree(recast_col, ignore_errors=True)
    print(f"[OK] Completato {col}. Sessioni salvate in questo step: {count_saved}")

print("\n" + "="*60)
print(f"[FINE] PREPROCESSING COMPLETATO! Totale sessioni nel metadata: {len(combined_meta['sessions_info'])}")
print("="*60)
