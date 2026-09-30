import time
import sys

print(f"=== Avvio test persistenza (durata 3 minuti) alle {time.strftime('%Y-%m-%d %H:%M:%S')} ===", flush=True)

# 36 passi da 5 secondi = 180 secondi (3 minuti completi)
for i in range(1, 37):
    time.sleep(5)
    print(f"Passo {i:02d}/36 - Server attivo alle {time.strftime('%H:%M:%S')} (secondo {i*5})", flush=True)

print(f"=== Test completato con successo alle {time.strftime('%H:%M:%S')} ===", flush=True)
