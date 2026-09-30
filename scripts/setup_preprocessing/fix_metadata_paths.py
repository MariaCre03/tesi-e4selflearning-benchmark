import os
import sys
import io
import pickle

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)

BASE_DIR = r"C:\Users\Admin\tesi_maria_aggiustata"
TARGET_DIR = os.path.join(BASE_DIR, "data", "preprocessed", "unsegmented_unlabelled")
meta_p = os.path.join(TARGET_DIR, "metadata.pkl")

print("="*65)
print("[VERIFICA ED ALLINEAMENTO PERCORSI METADATA]")
print("="*65)

if not os.path.exists(meta_p):
    print(f"[ERRORE] File {meta_p} non trovato!")
    sys.exit(1)

with open(meta_p, "rb") as f:
    meta = pickle.load(f)

old_sessions = meta.get("sessions_info", {})
print(f"Sessioni attualmente registrate nel metadata: {len(old_sessions)}")

# Scansiona tutti i file channels.h5 presenti su disco
print(f"\nScansione dei file channels.h5 reali in {TARGET_DIR}...")
disk_channels = {}
for root, dirs, files in os.walk(TARGET_DIR):
    if "channels.h5" in files:
        rel_path = os.path.relpath(root, TARGET_DIR).replace("\\", "/")
        disk_channels[rel_path] = root

print(f"Trovati {len(disk_channels)} file channels.h5 su disco!")

# Mappatura e allineamento
new_sessions = {}
matched = 0

# 1. Mappatura esatta
for rel_path, full_path in disk_channels.items():
    if rel_path in old_sessions:
        new_sessions[rel_path] = old_sessions[rel_path]
        matched += 1

# 2. Mappatura per match finale (es. cartella 001, 002, S1, ecc.) per percorsi nidificati
unmatched_disk = set(disk_channels.keys()) - set(new_sessions.keys())
unmatched_old = set(old_sessions.keys()) - set(new_sessions.keys())

if unmatched_disk and unmatched_old:
    print(f"\nAllineamento di {len(unmatched_disk)} percorsi nidificati...")
    for rel_path in list(unmatched_disk):
        clean_name = os.path.basename(rel_path)
        for old_id in list(unmatched_old):
            # Controlla se il nome finale o la sottocartella corrisponde
            if old_id.endswith("/" + clean_name) or old_id == clean_name or old_id.split("/")[-1] == clean_name:
                # se appartengono allo stesso dataset (es. big-ideas o wesd)
                dataset_disk = rel_path.split("/")[0]
                dataset_old = old_id.split("/")[0]
                if dataset_disk == dataset_old or "big-ideas" in old_id or "dati_preelaborati" in rel_path:
                    new_sessions[rel_path] = old_sessions[old_id]
                    unmatched_disk.discard(rel_path)
                    unmatched_old.discard(old_id)
                    matched += 1
                    print(f"  -> Allineato: {old_id}  ==>  {rel_path}")
                    break

# Se ci sono ancora vecchie sessioni che non esistono su disco, avvisa
still_missing = set(old_sessions.keys()) - set(new_sessions.keys())
print(f"\n[RISULTATO]")
print(f"  Totale sessioni allineate e verificate su disco: {len(new_sessions)}")

# Salva metadata aggiornato
meta["sessions_info"] = new_sessions
with open(meta_p, "wb") as f:
    pickle.dump(meta, f)

print(f"  Metadata salvato con successo in {meta_p}!")
print("="*65)
