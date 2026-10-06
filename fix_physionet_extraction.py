import os

target_file = "run_ablation_study.py"
if not os.path.exists(target_file):
    print(f"[ERRORE] File non trovato: {target_file}")
    exit(1)

with open(target_file, "r", encoding="utf-8") as f:
    content = f.read()

target = '    else:\n        all_subs = np.array([p.replace("\\\\", "/").split("/")[-2].split("_")[0] for p in all_paths])\n        unique_subs = sorted(list(np.unique(all_subs)))'

replacement = '    elif "physionet" in dataset_name.lower():\n        all_subs = np.array([p.replace("\\\\", "/").split("/")[-2] for p in all_paths])\n        unique_subs = sorted(list(np.unique(all_subs)))\n    else:\n        all_subs = np.array([p.replace("\\\\", "/").split("/")[-2].split("_")[0] for p in all_paths])\n        unique_subs = sorted(list(np.unique(all_subs)))'

if 'elif "physionet" in dataset_name.lower():' in content:
    print("[OK] Correzione PhysioNet già presente in run_ablation_study.py!")
elif target in content:
    content = content.replace(target, replacement, 1)
    with open(target_file, "w", encoding="utf-8") as f:
        f.write(content)
    print("[OK] Correzione PhysioNet applicata con successo in run_ablation_study.py!")
else:
    # Prova fallback con line replace
    lines = content.splitlines()
    new_lines = []
    patched = False
    for i, line in enumerate(lines):
        if line.strip() == "else:" and i > 0 and "extract_wesad_sub" in "".join(lines[max(0, i-15):i]):
            new_lines.append('    elif "physionet" in dataset_name.lower():')
            new_lines.append('        all_subs = np.array([p.replace("\\\\", "/").split("/")[-2] for p in all_paths])')
            new_lines.append('        unique_subs = sorted(list(np.unique(all_subs)))')
            new_lines.append(line)
            patched = True
        else:
            new_lines.append(line)
    if patched:
        with open(target_file, "w", encoding="utf-8") as f:
            f.write("\n".join(new_lines) + "\n")
        print("[OK] Correzione PhysioNet applicata tramite fallback!")
    else:
        print("[AVVISO] Impossibile trovare il blocco else da sostituire. Verifica manualmente.")
