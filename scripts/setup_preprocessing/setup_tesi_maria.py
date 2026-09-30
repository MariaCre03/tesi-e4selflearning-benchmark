import os
import sys
import io
import shutil
import pickle

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)

BASE_DIR = r"C:\Users\Admin"
DEST_DIR = os.path.join(BASE_DIR, "tesi_maria_aggiustata")

print("="*65)
print("[START] CREAZIONE ED ORGANIZZAZIONE: tesi_maria_aggiustata")
print("="*65)

# 1. Pulizia file zip di transito (recupero ~20 GB)
print("\n[STEP 1] Pulizia archivi zip di transito...")
zip_files = ["group4.zip", "raw_two_datasets.zip", "ADARP.zip", "WESAD.zip", "test_wesad.zip"]
freed_bytes = 0
for z in zip_files:
    p = os.path.join(BASE_DIR, z)
    if os.path.exists(p):
        sz = os.path.getsize(p)
        try:
            os.remove(p)
            freed_bytes += sz
            print(f"  -> Rimosso {z} ({sz / (1024**3):.2f} GB recuperati)")
        except Exception as e:
            print(f"  [WARN] Impossibile rimuovere {z}: {e}")
print(f"  [OK] Totale spazio recuperato: {freed_bytes / (1024**3):.2f} GB!")

# 2. Creazione cartelle principali in tesi_maria_aggiustata
print("\n[STEP 2] Creazione struttura cartelle pulita...")
dirs_to_create = [
    os.path.join(DEST_DIR, "data", "preprocessed"),
    os.path.join(DEST_DIR, "src"),
    os.path.join(DEST_DIR, "checkpoints"),
    os.path.join(DEST_DIR, "downstream_data"),
    os.path.join(DEST_DIR, "scripts"),
    os.path.join(DEST_DIR, "runs"),
]
for d in dirs_to_create:
    os.makedirs(d, exist_ok=True)
    print(f"  -> Cartella pronta: {os.path.relpath(d, BASE_DIR)}")

# 3. Spostamento / Collegamento dati pre-elaborati (1167 sessioni)
print("\n[STEP 3] Configurazione dati pre-elaborati (1167 sessioni)...")
src_prep = os.path.join(BASE_DIR, "data", "preprocessed", "unsegmented_unlabelled")
dst_prep = os.path.join(DEST_DIR, "data", "preprocessed", "unsegmented_unlabelled")

if os.path.exists(src_prep) and not os.path.exists(dst_prep):
    print("  -> Spostamento unsegmented_unlabelled in tesi_maria_aggiustata/data/preprocessed...")
    shutil.move(src_prep, dst_prep)
    print("  -> Spostamento completato con successo.")
elif os.path.exists(dst_prep):
    print("  -> unsegmented_unlabelled gia' presente nella cartella di destinazione.")

# Verifica integrita' metadata nella nuova posizione
meta_file = os.path.join(dst_prep, "metadata.pkl")
if os.path.exists(meta_file):
    with open(meta_file, "rb") as f:
        meta = pickle.load(f)
    print(f"  [VERIFICA METADATA] OK! Presenti {len(meta['sessions_info'])} sessioni valide su 11 dataset.")
else:
    print("  [WARN] metadata.pkl non trovato in destinazione!")

# 4. Copia del codice 'src' (modulo timebase patchato)
print("\n[STEP 4] Copia del codice sorgente src/timebase...")
src_code = os.path.join(BASE_DIR, "test_wesad", "src")
dst_code = os.path.join(DEST_DIR, "src")
if os.path.exists(src_code):
    for item in os.listdir(src_code):
        s = os.path.join(src_code, item)
        d = os.path.join(dst_code, item)
        if os.path.isdir(s):
            if not os.path.exists(d):
                shutil.copytree(s, d)
        else:
            if not os.path.exists(d):
                shutil.copy2(s, d)
    print("  -> Codice src/timebase copiato e pronto.")

# 5. Copia dei checkpoint e dei dati downstream WESAD da test_wesad
print("\n[STEP 5] Copia checkpoint e dati downstream...")
src_ckpt = os.path.join(BASE_DIR, "test_wesad", "ckpt_sslearner")
dst_ckpt = os.path.join(DEST_DIR, "checkpoints", "ckpt_sslearner")
if os.path.exists(src_ckpt) and not os.path.exists(dst_ckpt):
    shutil.copytree(src_ckpt, dst_ckpt)
    print("  -> Checkpoint base Corponi copiato in checkpoints/ckpt_sslearner.")

src_wesad_seg = os.path.join(BASE_DIR, "test_wesad", "wesad_segmented")
dst_wesad_seg = os.path.join(DEST_DIR, "downstream_data", "wesad_segmented")
if os.path.exists(src_wesad_seg) and not os.path.exists(dst_wesad_seg):
    shutil.copytree(src_wesad_seg, dst_wesad_seg)
    print("  -> Dati WESAD etichettati per Fase 2 copiati in downstream_data/wesad_segmented.")

# 6. Copia degli script di esecuzione delle fasi
print("\n[STEP 6] Copia degli script eseguibili principali...")
scripts_to_copy = [
    "segment.py",
    "pre_train.py",
    "train_ann.py"
]
# cerca in BASE_DIR o in repo/e4selflearning
for s_name in scripts_to_copy:
    # cerca prima in BASE_DIR
    src_file = os.path.join(BASE_DIR, s_name)
    if not os.path.exists(src_file):
        # cerca in repo
        repo_candidate = os.path.join(BASE_DIR, "repo", "e4selflearning", s_name)
        if os.path.exists(repo_candidate):
            src_file = repo_candidate
    if os.path.exists(src_file):
        dst_file = os.path.join(DEST_DIR, s_name)
        shutil.copy2(src_file, dst_file)
        print(f"  -> Copiato {s_name} in tesi_maria_aggiustata/")
    else:
        print(f"  [INFO] {s_name} non trovato nella radice (lo creeremo/copieremo noi).")

# Copia anche train_ann_fixed.py
fixed_ann = os.path.join(BASE_DIR, "test_wesad", "train_ann_fixed.py")
if os.path.exists(fixed_ann):
    shutil.copy2(fixed_ann, os.path.join(DEST_DIR, "train_ann_fixed.py"))
    print("  -> Copiato train_ann_fixed.py in tesi_maria_aggiustata/")

print("\n" + "="*65)
print("[COMPLETATO] La cartella C:\\Users\\Admin\\tesi_maria_aggiustata e' configurata!")
print("="*65)
