import os
import sys
import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser(description="Lancia un comando in background su Windows persistente a disconnessioni SSH")
    parser.add_argument("cmd", type=str, help="Il comando da eseguire (es. 'python run_ablation_study.py')")
    parser.add_argument("--log", type=str, default="background_log.txt", help="File di log dove reindirizzare l'output")
    args = parser.parse_args()

    cwd = os.getcwd()
    python_exe = sys.executable

    # Assicura che venga usato l'interprete Python dell'ambiente Conda attivo!
    cmd = args.cmd.strip()
    if cmd.startswith("python "):
        resolved_cmd = f'"{python_exe}" ' + cmd[7:]
    elif cmd == "python":
        resolved_cmd = f'"{python_exe}"'
    else:
        resolved_cmd = cmd

    cmd_to_run = f"cmd.exe /c cd /d \"{cwd}\" && {resolved_cmd} > \"{args.log}\" 2>&1"
    
    ps_script = f"""
    $res = Invoke-WmiMethod -Class Win32_Process -Name Create -ArgumentList @('{cmd_to_run.replace("'", "''")}', '{cwd.replace("'", "''")}')
    Write-Output $res.ProcessId
    """

    print("\n" + "=" * 70)
    print(f" [LAUNCHER BACKGROUND WINDOWS]")
    print(f" Python attivo (Conda): {python_exe}")
    print(f" Comando da lanciare  : {resolved_cmd}")
    print(f" Cartella di lavoro   : {cwd}")
    print(f" File di log          : {args.log}")
    print("=" * 70)

    res = subprocess.run(["powershell", "-NoProfile", "-Command", ps_script], capture_output=True, text=True)
    lines = [line.strip() for line in res.stdout.strip().splitlines() if line.strip()]
    pid = lines[-1] if lines else None

    if pid and pid.isdigit() and int(pid) > 0:
        print(f" [OK] Processo avviato con successo con PID: {pid}")
        print(f" Il processo e' COMPLETAMENTE STACCATO dalla sessione SSH.")
        print(f" Puoi chiudere la connessione e spegnere il tuo PC: il server continuera' a lavorare!")
        print(f"\n Per seguire l'output quando ti ricolleghi, digita:")
        print(f"   type {args.log}")
        print(f" Per verificare se e' ancora attivo:")
        print(f"   tasklist /fi \"PID eq {pid}\"")
    else:
        print(f" [ERRORE] Impossibile avviare il processo:")
        print(f" STDOUT: {res.stdout}")
        print(f" STDERR: {res.stderr}")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()
