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
print("[START] AVVIO PREPROCESSING FASE 0 - IN-GAUGE_EN-GAGE")
print("="*60)

# 2. Localizza ed estrai raw_wearable_data per In-Gauge
print("\n[STEP 1] Organizzazione ed estrazione cartella raw_wearable_data...")
ingauge_base = os.path.join(RAW_DIR, "in-gauge_en-gage")

sub_folders = [f for f in os.listdir(ingauge_base) if f.startswith("in-gauge-and-en-gage") and os.path.isdir(os.path.join(ingauge_base, f))]
if sub_folders:
    ingauge_folder = os.path.join(ingauge_base, sub_folders[0])
    target_raw = os.path.join(ingauge_folder, "raw_wearable_data")
    os.makedirs(target_raw, exist_ok=True)
    
    # 1. Estrai raw_wearable_data.zip direttamente dentro target_raw
    raw_zip = os.path.join(ingauge_folder, "raw_wearable_data.zip")
    if os.path.exists(raw_zip) and not any(f.startswith("Week") for f in os.listdir(target_raw)):
        print(f"  -> Estrazione {raw_zip} dentro raw_wearable_data...")
        with zipfile.ZipFile(raw_zip, "r") as z:
            z.extractall(target_raw)
        print("  -> File estratti con successo in raw_wearable_data!")
        
    # Se alcuni Week*.zip sono finiti nella cartella padre, spostali dentro target_raw
    for f in os.listdir(ingauge_folder):
        if f.startswith("Week") and f.endswith(".zip"):
            dest_f = os.path.join(target_raw, f)
            if not os.path.exists(dest_f):
                shutil.move(os.path.join(ingauge_folder, f), dest_f)
                
    # 2. Scompatta ogni Week*.zip dentro target_raw
    print("  -> Estrazione automatica dei singoli archivi Week*.zip...")
    count_weeks = 0
    for f in os.listdir(target_raw):
        if f.endswith(".zip"):
            w_zip = os.path.join(target_raw, f)
            w_name = f.replace(".zip", "")
            w_dir = os.path.join(target_raw, w_name)
            if not os.path.exists(w_dir):
                try:
                    with zipfile.ZipFile(w_zip, "r") as z:
                        z.extractall(target_raw)
                    count_weeks += 1
                except Exception as e:
                    print(f"    [ERRORE] Estrazione {f}: {e}")
    if count_weeks > 0:
        print(f"  -> Scompattati {count_weeks} archivi settimanali (file CSV ora visibili).")
    else:
        print("  -> Archivi settimanali già scompattati.")
else:
    print(f"[ATTENZIONE] Cartella in-gauge non trovata in {ingauge_base}")

# 3. Import delle librerie timebase
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
        os.makedirs(out_parent, exist_ok=True)
        if out_parent not in dirs:
            dirs.append(out_parent)
            
        if collection in REFORMAT_COLLECTION_DICT.keys():
            REFORMAT_COLLECTION_DICT[collection](
                filename=os.path.join(dirpath, filename),
                output_file=output_file,
            )
        else:
            shutil.copyfile(os.path.join(dirpath, filename), output_file)
            
        count_processed += 1
        if count_processed % 50 == 0 or count_processed == len(all_csvs):
            print(f"    -> Riformattati {count_processed}/{len(all_csvs)} file CSV...")
            
    # Assicura il recupero di tutte le cartelle sessione presenti
    existing_dirs = [root for root, d, f in os.walk(output_dir_collection) if any(x.endswith('.csv') for x in f)]
    dirs = sorted(list(set(dirs + existing_dirs)))
    return dirs

prep.recast_collection = windows_recast_collection

# 4. Caricamento metadata
print("\n[STEP 2] Caricamento indice metadata...")
if not os.path.exists(meta_p):
    raise FileNotFoundError(f"Errore: metadata.pkl non trovato in {meta_p}!")

with open(meta_p, "rb") as f:
    combined_meta = pickle.load(f)

initial_count = len(combined_meta["sessions_info"])
print(f"  -> Sessioni attuali mappate nel metadata: {initial_count}")

# 5. Esegui Preprocessing per in-gauge_en-gage
args = argparse.Namespace(
    path2unlabelled_data=RAW_DIR,
    output_dir=PREPROCESSED_DIR,
    e4selflearning=True,
    overwrite=False,
    sleep_algorithm="van_hees",
    wear_minimum_minutes=5,
    minimum_recorded_time=15,
    num_workers=4,
    chunksize=1,
    verbose=0,
    seed=1234
)

col = "in-gauge_en-gage"
print(f"\n==========================================")
print(f"[PROCESS] INIZIO PREPROCESSING: {col}")
print(f"==========================================")

path = UNLABELLED_DATA_PATHS[col]
source_full = os.path.join(RAW_DIR, path)

if not os.path.exists(source_full):
    print(f"[ATTENZIONE] Cartella sorgente non trovata: {source_full}")
else:
    s_dirs = prep.recast_collection(args, collection=col, path=path)
    output_dir_col = os.path.join(args.path2unlabelled_data, "recast", col)
    if os.path.exists(output_dir_col):
        recovered = [root for root, d, f in os.walk(output_dir_col) if any(x.endswith('.csv') for x in f)]
        s_dirs = sorted(list(set(s_dirs + recovered)))
    print(f"  -> Sessioni rilevate da elaborare: {len(s_dirs)}")
    
    added_col = 0
    for s_id in s_dirs:
        norm_s = s_id.replace("\\", "/")
        norm_raw = RAW_DIR.replace("\\", "/")
        name = norm_s.replace(f"{norm_raw}/recast/", "")
        name = name.replace("data/raw_data/unlabelled_data/recast/", "")
        
        if name in combined_meta["sessions_info"]:
            continue
            
        try:
            data, info, too_short = prep.preprocess_dir(args, recording_dir=s_id, labelled=False)
            if not too_short:
                data["labels"] = np.full(len(LABEL_COLS), np.nan, dtype=np.float32)
                sess_dir = os.path.join(PREPROCESSED_DIR, name)
                os.makedirs(sess_dir, exist_ok=True)
                h5.write(os.path.join(sess_dir, "channels.h5"), data, overwrite=True)
                combined_meta["sessions_info"][name] = info
                with open(meta_p, "wb") as f:
                    pickle.dump(combined_meta, f)
                added_col += 1
                if added_col % 5 == 0 or added_col == 1:
                    print(f"     [+] Mappata nuova sessione ({added_col}): {name}")
            del data
            gc.collect()
        except Exception as e:
            print(f"     [WARN] Errore su {name}: {e}")
            
    print(f"[OK] Concluso {col}: aggiunte {added_col} nuove sessioni!")
    shutil.rmtree(os.path.join(RAW_DIR, "recast", col), ignore_errors=True)

final_count = len(combined_meta["sessions_info"])
print("\n" + "="*60)
print(f"[SUCCESS] COMPLETATO CON SUCCESSO!")
print(f"Sessioni prima di questo step: {initial_count}")
print(f"Nuove sessioni aggiunte: {final_count - initial_count}")
print(f"Totale complessivo sessioni nel metadata: {final_count}")
print("="*60)
