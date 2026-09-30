import os
import io
import re
import zipfile
import datetime
import urllib.request
import numpy as np
import pandas as pd
import h5py
import pickle
from tqdm import tqdm

ZIP_URL = "https://physionet.org/static/published-projects/wearable-device-dataset/wearable-device-dataset-from-induced-stress-and-structured-exercise-sessions-1.0.1.zip"
LOCAL_DOWNLOADS = os.path.expanduser("~/Downloads/wearable-device-dataset-from-induced-stress-and-structured-exercise-sessions-1.0.1.zip")
SERVER_ZIP = "wearable-device-dataset-from-induced-stress-and-structured-exercise-sessions-1.0.1.zip"

OUTPUT_DIR = "downstream_data/physionet_segmented"

# Channels & Frequencies from Empatica E4
CH_FREQ = {
    'ACC_x': 32, 'ACC_y': 32, 'ACC_z': 32,
    'BVP': 64, 'EDA': 4, 'TEMP': 4
}

WINDOW_SEC = 60
STRIDE_SEC = 30

def get_zip_path():
    if os.path.exists(SERVER_ZIP):
        return SERVER_ZIP
    if os.path.exists(LOCAL_DOWNLOADS):
        return LOCAL_DOWNLOADS
    print(f"[INFO] Download del dataset da PhysioNet in corso: {ZIP_URL}")
    urllib.request.urlretrieve(ZIP_URL, SERVER_ZIP)
    return SERVER_ZIP

def parse_time(dt_str):
    for fmt in ('%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M:%S.%f'):
        try:
            return datetime.datetime.strptime(dt_str.strip(), fmt)
        except ValueError:
            pass
    raise ValueError(f"Formato timestamp non riconosciuto: {dt_str}")

def prepare():
    zip_path = get_zip_path()
    print(f"[INFO] Apertura archivio: {zip_path}")
    z = zipfile.ZipFile(zip_path, 'r')

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Trova tutti i file tags.csv per la cartella STRESS
    stress_tag_files = [f for f in z.namelist() if 'STRESS' in f and f.endswith('tags.csv')]
    stress_tag_files = sorted(stress_tag_files)

    print(f"[INFO] Trovate {len(stress_tag_files)} cartelle di soggetti STRESS.")

    sessions_paths = []
    sessions_labels = []
    subject_counts = {}

    excluded_subs = {'f07', 'f14_a', 'f14_b'} # problematiche hardware note

    for tag_file in tqdm(stress_tag_files, desc="Elaborazione soggetti"):
        sub_dir = tag_file.rsplit('/', 1)[0] + '/'
        sub_name = sub_dir.split('/')[-2]
        
        if sub_name in excluded_subs:
            print(f" [SKIP] Escluso {sub_name} (problemi hardware documentati).")
            continue

        try:
            # Lettura segnali E4
            eda_raw = pd.read_csv(io.BytesIO(z.read(sub_dir + 'EDA.csv')), header=None)
            t0_str = str(eda_raw.iloc[0, 0])
            t0_dt = parse_time(t0_str)
            fs_eda = float(eda_raw.iloc[1, 0])
            eda_sig = eda_raw.iloc[2:, 0].values.astype(np.float32)

            bvp_raw = pd.read_csv(io.BytesIO(z.read(sub_dir + 'BVP.csv')), header=None)
            fs_bvp = float(bvp_raw.iloc[1, 0])
            bvp_sig = bvp_raw.iloc[2:, 0].values.astype(np.float32)

            temp_raw = pd.read_csv(io.BytesIO(z.read(sub_dir + 'TEMP.csv')), header=None)
            fs_temp = float(temp_raw.iloc[1, 0])
            temp_sig = temp_raw.iloc[2:, 0].values.astype(np.float32)

            acc_raw = pd.read_csv(io.BytesIO(z.read(sub_dir + 'ACC.csv')), header=None)
            fs_acc = float(acc_raw.iloc[1, 0])
            acc_sig = acc_raw.iloc[2:, :3].values.astype(np.float32)

            # Durata minima comune in secondi
            dur_sec = min(
                len(eda_sig) / fs_eda,
                len(bvp_sig) / fs_bvp,
                len(temp_sig) / fs_temp,
                len(acc_sig) / fs_acc
            )

            # Lettura tags
            tags_raw = pd.read_csv(io.BytesIO(z.read(tag_file)), header=None)
            t_secs = [0.0] + [(parse_time(str(row[0])) - t0_dt).total_seconds() for _, row in tags_raw.iterrows()]

            # Definizione intervalli secondo il protocollo ufficiale
            stress_spans = []
            rest_spans = []

            if sub_name.startswith('S'): # Versione 1
                if len(t_secs) >= 13:
                    rest_spans.append((t_secs[1], t_secs[3]))    # Baseline
                    stress_spans.append((t_secs[3], t_secs[4]))  # Stroop
                    rest_spans.append((t_secs[4], t_secs[5]))    # Rest 1
                    stress_spans.append((t_secs[5], t_secs[6]))  # TMCT
                    rest_spans.append((t_secs[6], t_secs[7]))    # Rest 2
                    stress_spans.append((t_secs[7], t_secs[8]))  # Real Opinion
                    stress_spans.append((t_secs[9], t_secs[10])) # Opposite Opinion
                    stress_spans.append((t_secs[11], t_secs[12]))# Subtract Test
            else: # Versione 2 (f)
                if len(t_secs) >= 10:
                    rest_spans.append((t_secs[1], t_secs[2]))    # Baseline
                    stress_spans.append((t_secs[2], t_secs[3]))  # TMCT
                    rest_spans.append((t_secs[3], t_secs[4]))    # Rest 1
                    stress_spans.append((t_secs[4], t_secs[5]))  # Real Opinion
                    stress_spans.append((t_secs[6], t_secs[7]))  # Opposite Opinion
                    rest_spans.append((t_secs[7], t_secs[8]))    # Rest 2
                    stress_spans.append((t_secs[8], t_secs[9]))  # Subtract Test

            if not stress_spans or not rest_spans:
                print(f" [WARN] Tag non sufficienti per {sub_name}, saltato.")
                continue

            sub_out_dir = os.path.join(OUTPUT_DIR, f"SUB_{sub_name}")
            os.makedirs(sub_out_dir, exist_ok=True)

            w_start = 0.0
            win_idx = 0
            sub_stress_cnt, sub_rest_cnt = 0, 0

            while w_start + WINDOW_SEC <= dur_sec:
                w_end = w_start + WINDOW_SEC

                # Controllo se la finestra cade interamente in Stress o Rest
                is_stress = any(s <= w_start and w_end <= e for s, e in stress_spans)
                is_rest = any(s <= w_start and w_end <= e for s, e in rest_spans)

                if is_stress and not is_rest:
                    lbl = 1.0
                    sub_stress_cnt += 1
                elif is_rest and not is_stress:
                    lbl = 0.0
                    sub_rest_cnt += 1
                else:
                    # Finestra transitoria o mista -> scartata per pulizia
                    w_start += STRIDE_SEC
                    continue

                # Estrazione fette temporali esatte
                acc_s, acc_e = int(w_start * fs_acc), int(w_end * fs_acc)
                bvp_s, bvp_e = int(w_start * fs_bvp), int(w_end * fs_bvp)
                eda_s, eda_e = int(w_start * fs_eda), int(w_end * fs_eda)
                tmp_s, tmp_e = int(w_start * fs_temp), int(w_end * fs_temp)

                acc_win = acc_sig[acc_s:acc_e]
                bvp_win = bvp_sig[bvp_s:bvp_e]
                eda_win = eda_sig[eda_s:eda_e]
                tmp_win = temp_sig[tmp_s:tmp_e]

                # Target shapes
                target_len_acc = int(WINDOW_SEC * CH_FREQ['ACC_x'])
                target_len_bvp = int(WINDOW_SEC * CH_FREQ['BVP'])
                target_len_eda = int(WINDOW_SEC * CH_FREQ['EDA'])
                target_len_tmp = int(WINDOW_SEC * CH_FREQ['TEMP'])

                acc_x = acc_win[:target_len_acc, 0] if len(acc_win) >= target_len_acc else np.pad(acc_win[:, 0], (0, target_len_acc - len(acc_win)))
                acc_y = acc_win[:target_len_acc, 1] if len(acc_win) >= target_len_acc else np.pad(acc_win[:, 1], (0, target_len_acc - len(acc_win)))
                acc_z = acc_win[:target_len_acc, 2] if len(acc_win) >= target_len_acc else np.pad(acc_win[:, 2], (0, target_len_acc - len(acc_win)))
                bvp = bvp_win[:target_len_bvp] if len(bvp_win) >= target_len_bvp else np.pad(bvp_win, (0, target_len_bvp - len(bvp_win)))
                eda = eda_win[:target_len_eda] if len(eda_win) >= target_len_eda else np.pad(eda_win, (0, target_len_eda - len(eda_win)))
                tmp = tmp_win[:target_len_tmp] if len(tmp_win) >= target_len_tmp else np.pad(tmp_win, (0, target_len_tmp - len(tmp_win)))

                h5_file = os.path.join(sub_out_dir, f"win_{win_idx:04d}.h5")
                with h5py.File(h5_file, 'w') as hf:
                    hf.create_dataset('ACC_x', data=acc_x.astype(np.float32))
                    hf.create_dataset('ACC_y', data=acc_y.astype(np.float32))
                    hf.create_dataset('ACC_z', data=acc_z.astype(np.float32))
                    hf.create_dataset('BVP', data=bvp.astype(np.float32))
                    hf.create_dataset('EDA', data=eda.astype(np.float32))
                    hf.create_dataset('TEMP', data=tmp.astype(np.float32))
                    
                    # Labels array (standard Corponi/WESAD compatibility)
                    lbl_arr = np.zeros(41, dtype=np.float32)
                    lbl_arr[3] = lbl # status
                    hf.create_dataset('labels', data=lbl_arr)

                rel_path = os.path.join(f"SUB_{sub_name}", f"win_{win_idx:04d}.h5").replace('\\', '/')
                sessions_paths.append(rel_path)
                sessions_labels.append(lbl)

                win_idx += 1
                w_start += STRIDE_SEC

            subject_counts[sub_name] = {'stress': sub_stress_cnt, 'rest': sub_rest_cnt, 'total': win_idx}

        except Exception as e:
            print(f" [ERRORE] Elaborazione fallita per {sub_name}: {e}")

    # Creazione metadata.pkl
    meta = {
        'sessions_paths': sessions_paths,
        'sessions_labels': {'status': sessions_labels},
        'ds_info': {
            'channel_freq': CH_FREQ,
            'segment_length': WINDOW_SEC,
            'subject_counts': subject_counts
        }
    }

    meta_path = os.path.join(OUTPUT_DIR, "metadata.pkl")
    with open(meta_path, 'wb') as f:
        pickle.dump(meta, f)

    # Creazione stats.pkl
    stats = {c: {'min': -10, 'max': 10, 'mean': 0, 'std': 1, 'median': 0, 'iqr': 1} for c in CH_FREQ.keys()}
    with open(os.path.join(OUTPUT_DIR, "stats.pkl"), 'wb') as f:
        pickle.dump(stats, f)

    n_tot = len(sessions_paths)
    n_stress = sum(1 for y in sessions_labels if y == 1.0)
    n_rest = sum(1 for y in sessions_labels if y == 0.0)

    print("\n" + "="*70)
    print(" PREPARAZIONE PHYSIONET COMPLETATA CON SUCCESSO!")
    print("="*70)
    print(f" Cartella output : {OUTPUT_DIR}")
    print(f" Soggetti validi : {len(subject_counts)}")
    print(f" Finestre Totali : {n_tot}")
    print(f" Finestre Stress : {n_stress} ({n_stress/max(1,n_tot)*100:.1f}%)")
    print(f" Finestre Rest   : {n_rest} ({n_rest/max(1,n_tot)*100:.1f}%)")
    print("="*70)

if __name__ == "__main__":
    prepare()
