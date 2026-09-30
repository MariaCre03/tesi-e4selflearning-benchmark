import os
import sys
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
print("🚀 AVVIO PREPROCESSING INCREMENTALE (Fase 0 - Aggiunta Dataset)")
print("="*60)

# Funzione per estrarre tutti gli zip di sessione interni
def unpack_nested_zips(target_dir):
    count = 0
    for root, dirs, files in os.walk(target_dir):
        for f in files:
            if f.endswith(".zip"):
                zip_path = os.path.join(root, f)
                out_sub = os.path.join(root, f.replace(".zip", ""))
                if not os.path.exists(out_sub) or not any(x.endswith('.csv') for x in os.listdir(out_sub)):
                    os.makedirs(out_sub, exist_ok=True)
                    try:
                        with zipfile.ZipFile(zip_path, "r") as z:
                            z.extractall(out_sub)
                        count += 1
                    except Exception as e:
                        print(f"    Errore estrazione {f}: {e}")
    if count > 0:
        print(f"  -> Scompattati {count} archivi di sessione E4 (file CSV ora visibili).")

# 2. Configurazione e correzione percorsi cartelle
print("\n[STEP 1] Controllo e organizzazione cartelle raw...")

# PPG_DaLiA
dalia_base = os.path.join(RAW_DIR, "PPG_DaLiA")
dalia_target = os.path.join(dalia_base, "PPG_FieldStudy")
if not os.path.exists(dalia_target):
    dalia_zip = os.path.join(dalia_base, "data.zip")
    if os.path.exists(dalia_zip):
        print("  -> Estrazione PPG_DaLiA data.zip...")
        with zipfile.ZipFile(dalia_zip, "r") as z:
            z.extractall(dalia_base)

if os.path.exists(dalia_target):
    print("  -> Unpacking sessioni interne PPG_DaLiA...")
    unpack_nested_zips(dalia_target)

# Stress nurses
nurses_base = os.path.join(RAW_DIR, "stress_detection_nurses_hospital")
nurses_target = os.path.join(nurses_base, "Stress_dataset")
os.makedirs(nurses_target, exist_ok=True)

# Se le cartelle dei soggetti sono state estratte fuori da Stress_dataset, spostiamole dentro
for item in os.listdir(nurses_base):
    item_p = os.path.join(nurses_base, item)
    if os.path.isdir(item_p) and item != "Stress_dataset":
        dest_p = os.path.join(nurses_target, item)
        if not os.path.exists(dest_p):
            shutil.move(item_p, dest_p)

# Se c'è lo zip, estrailo direttamente in Stress_dataset se non è ancora fatto
nurses_zip = os.path.join(nurses_base, "Stress_dataset.zip")
if os.path.exists(nurses_zip) and len(os.listdir(nurses_target)) == 0:
    print("  -> Estrazione Stress_dataset.zip...")
    with zipfile.ZipFile(nurses_zip, "r") as z:
        z.extractall(nurses_target)

print("  -> Unpacking sessioni interne Stress nurses...")
unpack_nested_zips(nurses_target)

# 3. Import delle librerie timebase
from timebase.data import preprocessing as prep
from timebase.data.static import LABEL_COLS, UNLABELLED_DATA_PATHS, CSV_CHANNELS, CORRUPTED_FILES
from timebase.data.preprocessing import REFORMAT_COLLECTION_DICT, get_channel_from_filename, check_faulty_folder
from timebase.utils import h5

# Patch robusta per i percorsi Windows in recast_collection
def windows_recast_collection(args, collection: str, path: str):
    dirs = []
    output_dir_collection = os.path.join(args.path2unlabelled_data, "recast", collection)
    root_dir = os.path.join(args.path2unlabelled_data, path)
    files2dismiss = []
    print(f"  [Scansione file in: {root_dir}]")
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".csv") and get_channel_from_filename(filename, CSV_CHANNELS):
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
    return dirs

prep.recast_collection = windows_recast_collection

# 4. Carica metadata esistente
print("\n[STEP 2] Caricamento indice metadata...")
if not os.path.exists(meta_p):
    raise FileNotFoundError(f"Errore: metadata.pkl non trovato in {meta_p}!")

with open(meta_p, "rb") as f:
    combined_meta = pickle.load(f)

initial_count = len(combined_meta["sessions_info"])
print(f"  -> Sessioni attuali mappate nel metadata: {initial_count}")

# 5. Esegui Preprocessing
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

datasets_to_process = ["pgg_dalia", "stress_detection_nurses_hospital"]

for col in datasets_to_process:
    print(f"\n==========================================")
    print(f"🚀 INIZIO PREPROCESSING: {col}")
    print(f"==========================================")
    
    path = UNLABELLED_DATA_PATHS[col]
    source_full = os.path.join(RAW_DIR, path)
    if not os.path.exists(source_full):
        print(f"⚠️ Cartella non trovata: {source_full}")
        continue
        
    s_dirs = prep.recast_collection(args, collection=col, path=path)
    print(f"  -> Sessioni raw trovate: {len(s_dirs)}")
    
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
                print(f"     [+] Mappata nuova sessione ({added_col}): {name}")
            del data
            gc.collect()
        except Exception as e:
            print(f"     ⚠️ Errore su {name}: {e}")
            
    print(f"✅ Concluso {col}: aggiunte {added_col} nuove sessioni!")
    shutil.rmtree(os.path.join(RAW_DIR, "recast", col), ignore_errors=True)

final_count = len(combined_meta["sessions_info"])
print("\n" + "="*60)
print(f"🎉 COMPLETATO CON SUCCESSO!")
print(f"Sessioni iniziali: {initial_count}")
print(f"Nuove sessioni aggiunte: {final_count - initial_count}")
print(f"Totale complessivo sessioni nel metadata: {final_count}")
print("="*60)
