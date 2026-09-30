import os
import json
import numpy as np
import scipy.stats as stats

BENCH_DIR = "risultati_benchmark"
DATASETS = ["wesad", "hosseini", "physionet"]
MODELS = ["corponi", "biot", "simmtm", "femba"]

def load_model_folds(ds, model):
    candidates = [
        os.path.join(BENCH_DIR, f"{model}_{ds}"),
        os.path.join(BENCH_DIR, f"{model}_{ds}_losocv")
    ]
    json_path = None
    for c in candidates:
        cand_json = os.path.join(c, "losocv_folds.json")
        if os.path.exists(cand_json):
            json_path = cand_json
            break
    if not json_path:
        return None
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Ordina per subject/fold per garantire allineamento paired
    data_sorted = sorted(data, key=lambda x: str(x.get("subject", x.get("fold"))))
    accs = [f["accuracy"] for f in data_sorted]
    f1_stresses = [f["f1_stress"] for f in data_sorted]
    macro_f1s = [f["macro_f1"] for f in data_sorted]
    subs = [str(f.get("subject", f.get("fold"))) for f in data_sorted]
    return {"acc": np.array(accs), "f1": np.array(f1_stresses), "macro_f1": np.array(macro_f1s), "subs": subs}

def compute_ci(data, confidence=0.95):
    n = len(data)
    m = np.mean(data)
    std_err = stats.sem(data)
    h = std_err * stats.t.ppf((1 + confidence) / 2., n - 1)
    return m, h

def run_stats():
    print("\n" + "="*95)
    print("        ANALISI STATISTICA COMPLETA DEL BENCHMARK (LOSOCV SIGNIFICANCE TESTING)")
    print("="*95)

    for ds in DATASETS:
        print(f"\n" + "#"*70)
        print(f" DATASET: {ds.upper()}")
        print("#"*70)
        
        results = {}
        for m in MODELS:
            res = load_model_folds(ds, m)
            if res is not None:
                results[m] = res
        
        if not results:
            print(" Nessun dato trovato per questo dataset.")
            continue

        # Tabella Medie +- CI 95%
        print("\n--- 1. METRICHE CON INTERVALLO DI CONFIDENZA AL 95% ---")
        print(f"{'Modello':<15} | {'Accuratezza (95% CI)':<25} | {'F1-Stress (95% CI)':<25} | {'Macro-F1 (95% CI)'}")
        print("-" * 90)
        for m in MODELS:
            if m in results:
                m_acc, h_acc = compute_ci(results[m]["acc"])
                m_f1, h_f1 = compute_ci(results[m]["f1"])
                m_mf1, h_mf1 = compute_ci(results[m]["macro_f1"])
                print(f"{m.upper():<15} | {m_acc:5.2f}% ± {h_acc:4.2f}%          | {m_f1:5.2f}% ± {h_f1:4.2f}%          | {m_mf1:5.2f}% ± {h_mf1:4.2f}%")

        # Test di Significatività: Wilcoxon Signed-Rank Test & Paired t-test
        print("\n--- 2. TEST DI SIGNIFICATIVITÀ STATISTICA (PAIRED P-VALUES) ---")
        print(f"{'Confronto (A vs B)':<25} | {'Metrica':<12} | {'Wilcoxon p-val':<16} | {'t-test p-val':<14} | {'Significativo?'}")
        print("-" * 85)

        pairs = [
            ("femba", "corponi"),
            ("simmtm", "corponi"),
            ("biot", "corponi"),
            ("femba", "biot"),
            ("femba", "simmtm"),
            ("simmtm", "biot"),
        ]

        for m1, m2 in pairs:
            if m1 in results and m2 in results:
                # Confronto su accuratezza e Macro-F1
                for metric_key, metric_name in [("acc", "Accuracy"), ("macro_f1", "Macro-F1"), ("f1", "F1-Stress")]:
                    v1 = results[m1][metric_key]
                    v2 = results[m2][metric_key]
                    diff = v1 - v2
                    if np.all(diff == 0):
                        continue
                    try:
                        _, p_wilc = stats.wilcoxon(v1, v2)
                    except Exception:
                        p_wilc = 1.0
                    _, p_ttest = stats.ttest_rel(v1, v2)
                    sig = "SI (p < 0.05) *" if p_wilc < 0.05 or p_ttest < 0.05 else "NO (p >= 0.05)"
                    if p_wilc < 0.01 or p_ttest < 0.01:
                        sig = "SI (p < 0.01) **"
                    
                    pair_label = f"{m1.upper()} vs {m2.upper()}"
                    print(f"{pair_label:<25} | {metric_name:<12} | {p_wilc:<16.4f} | {p_ttest:<14.4f} | {sig}")
                print("-" * 85)

    print("\n" + "="*95)
    print(" NOTE METODOLOGICHE PER IL PAPER:")
    print("  * Wilcoxon signed-rank test e Paired t-test sono stati calcolati sui fold appaiati (soggetto per soggetto).")
    print("  * Risultati con p < 0.05 sono contrassegnati con *, con p < 0.01 con **.")
    print("="*95 + "\n")

if __name__ == "__main__":
    run_stats()
