import os
import sys
import subprocess
import argparse
import time

def run_cmd(cmd, desc):
    print("\n" + "="*80)
    print(f" [RUN] {desc}")
    print(f" Comando: {cmd}")
    print("="*80)
    t0 = time.time()
    res = subprocess.run(cmd, shell=True)
    el = time.time() - t0
    if res.returncode != 0:
        print(f" [ERRORE] Il comando e' fallito con codice {res.returncode}")
        sys.exit(res.returncode)
    print(f" [OK] {desc} completato in {el:.1f}s.")

def main():
    parser = argparse.ArgumentParser(description="PhysioNet (Hongn 2025) Complete Benchmark Suite")
    parser.add_argument("--skip_prep", action="store_true", help="Salta la preparazione se già presente")
    parser.add_argument("--models", nargs="+", default=["corponi", "biot", "simmtm", "femba"],
                        choices=["corponi", "biot", "simmtm", "femba"], help="Modelli da eseguire")
    args = parser.parse_args()

    data_dir = os.path.join("downstream_data", "physionet_segmented")
    meta_file = os.path.join(data_dir, "metadata.pkl")

    # 1. Preparazione Dataset
    if not os.path.exists(meta_file) or not args.skip_prep:
        print("\n>>> FASE 1: Preparazione e Segmentazione del Dataset PhysioNet 2025...")
        run_cmd(f"{sys.executable} scripts/prepare_physionet_dataset.py", "Segmentazione PhysioNet (34 soggetti)")
    else:
        print(f"\n[INFO] Dataset PhysioNet gia' pronto in {data_dir}.")

    # 2. Esecuzione Benchmark per ciascun modello
    def get_ckpt(name):
        candidates = [
            f"{name}_encoder_pretrained.pt",
            os.path.join("output_pretrain_full", f"{name}_full.pt"),
            f"{name}_full.pt"
        ]
        for c in candidates:
            if os.path.exists(c):
                return c
        return ""

    # CORPONI
    if "corponi" in args.models:
        run_cmd(
            f"{sys.executable} run_corponi_benchmark.py --mode physionet --epochs 20 --batch_size 16",
            "LOSOCV Benchmark: CORPONI (E4mer Baseline) su PhysioNet"
        )

    # BIOT
    if "biot" in args.models:
        biot_ckpt = get_ckpt("biot")
        biot_flag = f"--pretrained_weights {biot_ckpt}" if biot_ckpt else ""
        run_cmd(
            f"{sys.executable} scripts/train_biot.py --dataset {data_dir} --output_dir risultati_benchmark/biot_physionet_losocv --losocv --weighted_loss --epochs 20 --batch_size 16 {biot_flag}",
            "LOSOCV Benchmark: BIOT Transformer su PhysioNet"
        )

    # SIMMTM
    if "simmtm" in args.models:
        simmtm_ckpt = get_ckpt("simmtm")
        simmtm_flag = f"--pretrained_weights {simmtm_ckpt}" if simmtm_ckpt else ""
        run_cmd(
            f"{sys.executable} scripts/train_simmtm.py --dataset {data_dir} --output_dir risultati_benchmark/simmtm_physionet_losocv --losocv --weighted_loss --epochs 20 --batch_size 16 {simmtm_flag}",
            "LOSOCV Benchmark: SimMTM Transformer su PhysioNet"
        )

    # FEMBA
    if "femba" in args.models:
        femba_ckpt = get_ckpt("femba")
        femba_flag = f"--pretrained_weights {femba_ckpt}" if femba_ckpt else ""
        run_cmd(
            f"{sys.executable} scripts/train_femba.py --dataset {data_dir} --output_dir risultati_benchmark/femba_physionet_losocv --losocv --weighted_loss --epochs 20 --batch_size 16 {femba_flag}",
            "LOSOCV Benchmark: FEMBA (Mamba) su PhysioNet"
        )

    print("\n" + "="*80)
    print(" TUTTI I BENCHMARK SU PHYSIONET COMPLETATI CON SUCCESSO!")
    print("="*80)
    print("I risultati fold-by-fold sono salvati in:")
    print(" - risultati_benchmark/corponi_physionet/")
    print(" - risultati_benchmark/biot_physionet_losocv/")
    print(" - risultati_benchmark/simmtm_physionet_losocv/")
    print(" - risultati_benchmark/femba_physionet_losocv/")
    print("="*80)

if __name__ == "__main__":
    main()
