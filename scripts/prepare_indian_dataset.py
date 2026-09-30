import os
import zipfile
import pandas as pd
import numpy as np
import h5py
import pickle
import neurokit2 as nk
from tqdm import tqdm
import gdown

# --- Configuration ---
FILE_ID = '1qY-3qOPzPWm3rtrOmfOBfqVe_xcWfziM'
ZIP_NAME = 'indian_dataset.zip'
EXTRACT_DIR = 'data/indian_raw'
PROCESSED_DIR = 'data/indian_preprocessed'
METADATA_FILE = os.path.join(PROCESSED_DIR, 'metadata.pkl')

# Mapping: Tasks T1, T2, T3 -> Stress (1), Tasks T4, T5 -> Baseline (0)
TASK_MAP = {
    'T1': 1, 'T2': 1, 'T3': 1,
    'T4': 0, 'T5': 0
}

CHANNELS_FREQ = {
    "BVP": 64, "EDA": 4, "HR": 1, "TEMP": 4,
    "ACC_x": 32, "ACC_y": 32, "ACC_z": 32,
    "EDA_tonic": 4, "EDA_phasic": 4, "EDA_smna": 4
}

def download_dataset():
    if not os.path.exists(ZIP_NAME):
        print(f"Downloading dataset from Google Drive (ID: {FILE_ID})...")
        url = f'https://drive.google.com/uc?id={FILE_ID}'
        gdown.download(url, ZIP_NAME, quiet=False)
    
    if not os.path.exists(EXTRACT_DIR):
        print("Extracting dataset...")
        with zipfile.ZipFile(ZIP_NAME, 'r') as zip_ref:
            zip_ref.extractall(EXTRACT_DIR)

def load_e4_csv(path):
    """Loads a standard E4 CSV file (line 0: t0, line 1: freq, line 2+: data)"""
    try:
        df = pd.read_csv(path, header=None)
        t0 = float(df.iloc[0, 0])
        freq = float(df.iloc[1, 0])
        data = df.iloc[2:, :].values.astype(np.float32)
        return t0, freq, data
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None, None, None

def process_eda(eda_signal, freq):
    """Decomposes EDA into Tonic and Phasic components using NeuroKit2"""
    try:
        # NeuroKit2 eda_process returns a dataframe with components
        signals, info = nk.eda_process(eda_signal.flatten(), sampling_rate=int(freq))
        tonic = signals["EDA_Tonic"].values.astype(np.float32)
        phasic = signals["EDA_Phasic"].values.astype(np.float32)
        # SMNA approximation (using Phasic as proxy if not using cvxEDA)
        smna = phasic.copy() 
        return tonic, phasic, smna
    except:
        # Fallback to zeros if decomposition fails
        return np.zeros_like(eda_signal), np.zeros_like(eda_signal), np.zeros_like(eda_signal)

def prepare():
    download_dataset()
    
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    sessions_info = {}
    subjects = [d for d in os.listdir(EXTRACT_DIR) if os.path.isdir(os.path.join(EXTRACT_DIR, d))]
    
    print(f"Found {len(subjects)} subjects. Processing...")
    
    for sub in tqdm(subjects):
        sub_path = os.path.join(EXTRACT_DIR, sub)
        # Tasks are subdirectories like T1, T2...
        tasks = [t for t in os.listdir(sub_path) if t in TASK_MAP and os.path.isdir(os.path.join(sub_path, t))]
        
        for task in tasks:
            session_id = f"{sub}_{task}"
            session_dir = os.path.join(PROCESSED_DIR, session_id)
            os.makedirs(session_dir, exist_ok=True)
            
            task_path = os.path.join(sub_path, task)
            
            # Identify files (e.g. T1ACC.csv)
            files = os.listdir(task_path)
            channel_data = {}
            t0s = {}
            
            # Map Indian files to Corponi channels
            mapping = {
                'ACC': 'ACC', 'BVP': 'BVP', 'EDA': 'EDA', 'HR': 'HR', 'TEMP': 'TEMP'
            }
            
            found_all = True
            for key, corponi_key in mapping.items():
                target_file = next((f for f in files if key in f and f.endswith('.csv')), None)
                if not target_file:
                    found_all = False
                    break
                
                t0, freq, data = load_e4_csv(os.path.join(task_path, target_file))
                if data is None:
                    found_all = False
                    break
                
                channel_data[corponi_key] = data
                t0s[corponi_key] = t0
            
            if not found_all:
                continue
            
            # Split ACC
            acc = channel_data['ACC']
            channel_data['ACC_x'] = acc[:, 0]
            channel_data['ACC_y'] = acc[:, 1]
            channel_data['ACC_z'] = acc[:, 2]
            del channel_data['ACC']
            
            # Decompose EDA
            tonic, phasic, smna = process_eda(channel_data['EDA'], 4)
            channel_data['EDA_tonic'] = tonic
            channel_data['EDA_phasic'] = phasic
            channel_data['EDA_smna'] = smna
            
            # Save to HDF5
            h5_path = os.path.join(session_dir, 'channels.h5')
            with h5py.File(h5_path, 'w') as f:
                for c, v in channel_data.items():
                    f.create_dataset(c, data=v)
                
                # Add status label
                label_val = float(TASK_MAP[task])
                # Corponi labels are a vector of length 41 (LABEL_COLS)
                # We only care about 'status' (index 3) and 'Sub_ID' (index 0)
                labels = np.zeros(41, dtype=np.float32)
                labels[3] = label_val # status
                # Try to extract numeric sub id
                try:
                    labels[0] = float(''.join(filter(str.isdigit, sub)))
                except:
                    labels[0] = -1
                
                f.create_dataset('labels', data=labels)
                
                # Sleep status (placeholder for wake)
                sleep_mask = np.zeros(len(channel_data['ACC_x']), dtype=np.float32)
                f.create_dataset('SLEEP', data=sleep_mask)

            sessions_info[session_id] = {
                'channel_names': list(channel_data.keys()),
                'sampling_rates': {c: CHANNELS_FREQ[c] for c in channel_data.keys() if c in CHANNELS_FREQ},
                'unix_t0': {c: t0s.get(c, t0s['HR']) for c in channel_data.keys()},
                'labelled': True,
                'seconds_per_status': {0.0: len(channel_data['ACC_x'])/32.0}
            }

    # Save metadata
    metadata = {
        'sessions_info': sessions_info,
        'invalid_sessions': [],
        'sleep_algorithm': 'van_hees'
    }
    with open(METADATA_FILE, 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"Preparation complete. {len(sessions_info)} sessions processed.")

if __name__ == "__main__":
    prepare()
