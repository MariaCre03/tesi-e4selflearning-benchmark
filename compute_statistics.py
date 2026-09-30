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
    
    # Mappa per subject per garantire allineamento paired esatto
    sub_map = {str(f.get("subject", f.get("fold"))): f for f in data}
    return sub_map

def compute_ci(data, confidence=0.95):
    n = len(data)
    m = np.mean(data)
    std_err = stats.sem(data)
    h = std_err * stats.t.ppf((1 + confidence) / 2., n - 1)
    return m, h

def holm_bonferroni(p_values, alpha=0.05):
    """Correzione sequenziale di Holm-Bonferroni per confronti multipli."""
    sorted_indices = np.argsort(p_values)
    m = len(p_values)
    adjusted_sig = [False] * m
    for rank, idx in enumerate(sorted_indices):
        threshold = alpha / (m - rank)
        if p_values[idx] <= threshold:
            adjusted_sig[idx] = True
        else:
            break
    return adjusted_sig

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

        # Trova soggetti comuni a tutti i modelli disponibili
        common_subs = sorted(list(set.intersection(*[set(results[m].keys()) for m in results])))
        print(f" [INFO] Soggetti comuni allineati per LOSOCV: {len(common_subs)} ({', '.join(common_subs[:5])}...)")

        # Tabella Medie +- CI 95%
        print("\n--- 1. METRICHE CON INTERVALLO DI CONFIDENZA AL 95% ---")
        print(f"{'Modello':<15} | {'Accuratezza (95% CI)':<25} | {'F1-Stress (95% CI)':<25} | {'Macro-F1 (95% CI)'}")
        print("-" * 90)
        for m in MODELS:
            if m in results:
                accs = [results[m][s]["accuracy"] for s in common_subs]
                f1s = [results[m][s]["f1_stress"] for s in common_subs]
                mf1s = [results[m][s]["macro_f1"] for s in common_subs]
                m_acc, h_acc = compute_ci(accs)
                m_f1, h_f1 = compute_ci(f1s)
                m_mf1, h_mf1 = compute_ci(mf1s)
                print(f"{m.upper():<15} | {m_acc:5.2f}% ± {h_acc:4.2f}%          | {m_f1:5.2f}% ± {h_f1:4.2f}%          | {m_mf1:5.2f}% ± {h_mf1:4.2f}%")

        # Test di Significatività: Wilcoxon Signed-Rank Test con correzione Holm-Bonferroni
        print("\n--- 2. TEST DI SIGNIFICATIVITÀ STATISTICA (PAIRED WILCOXON & HOLM-BONFERRONI) ---")
        print(f"{'Confronto (A vs B)':<25} | {'Metrica':<12} | {'Wilcoxon p-val':<16} | {'t-test p-val':<14} | {'Significativo (Holm)'}")
        print("-" * 92)

        pairs = [
            ("femba", "corponi"),
            ("simmtm", "corponi"),
            ("biot", "corponi"),
            ("femba", "biot"),
            ("femba", "simmtm"),
            ("simmtm", "biot"),
        ]

        for metric_key, metric_name in [("accuracy", "Accuracy"), ("macro_f1", "Macro-F1"), ("f1_stress", "F1-Stress")]:
            active_pairs = []
            wilc_pvals = []
            ttest_pvals = []

            for m1, m2 in pairs:
                if m1 in results and m2 in results:
                    v1 = np.array([results[m1][s][metric_key] for s in common_subs])
                    v2 = np.array([results[m2][s][metric_key] for s in common_subs])
                    diff = v1 - v2
                    if np.all(diff == 0):
                        continue
                    try:
                        _, p_wilc = stats.wilcoxon(v1, v2)
                    except Exception:
                        p_wilc = 1.0
                    try:
                        _, p_ttest = stats.ttest_rel(v1, v2)
                    except Exception:
                        p_ttest = 1.0

                    active_pairs.append((m1, m2))
                    wilc_pvals.append(p_wilc)
                    ttest_pvals.append(p_ttest)

            # Correzione di Holm-Bonferroni sui confronti del dataset
            sig_flags = holm_bonferroni(wilc_pvals, alpha=0.05) if wilc_pvals else []

            for (m1, m2), p_w, p_t, is_sig in zip(active_pairs, wilc_pvals, ttest_pvals, sig_flags):
                pair_label = f"{m1.upper()} vs {m2.upper()}"
                sig_str = "SI (p_corr < 0.05) *" if is_sig else "NO"
                print(f"{pair_label:<25} | {metric_name:<12} | {p_w:<16.4f} | {p_t:<14.4f} | {sig_str}")
            print("-" * 92)

    print("\n" + "="*95)
    print(" NOTE METODOLOGICHE PER IL PAPER:")
    print("  * Tutti i test appaiati garantiscono che il soggetto i-esimo del Modello A corrisponda esattamente al soggetto i-esimo del Modello B.")
    print("  * I p-value di Wilcoxon sono corretti per confronti multipli tramite la procedura sequenziale di Holm-Bonferroni (alfa = 0.05).")
    print("="*95 + "\n")

if __name__ == "__main__":
    run_stats()
