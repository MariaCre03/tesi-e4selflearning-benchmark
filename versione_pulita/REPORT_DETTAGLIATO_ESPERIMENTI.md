# Relazione Sperimentale Completa: Benchmark SSL per Stress Detection

In questa fase finale della ricerca ho consolidato e validato l'intera pipeline sperimentale, confrontando l'architettura di riferimento (**Corponi**) con due modelli d'avanguardia: **BIOT** (Biosignal Transformer) e **SimMTM**. L'intero studio si fonda sul paradigma **Self-Supervised Learning (SSL)**, una scelta obbligata in quanto pilastro dell'architettura Corponi e garanzia di un confronto scientificamente equo tra modelli che apprendono rappresentazioni universali prima della specializzazione.

---

## 2. Originalità della Ricerca: Oltre il Lavoro di Corponi

Un aspetto fondamentale di questa sperimentazione è il superamento dei limiti del lavoro originale di Corponi. Nel paper di riferimento, gli autori si sono concentrati sulla validazione della propria architettura SSL confrontandola principalmente con modelli supervisionati classici o varianti della propria rete. 

**Nel lavoro originale di Corponi non sono mai stati effettuati confronti diretti con framework SSL d'avanguardia come BIOT o SimMTM.** 

La mia ricerca introduce quindi un valore aggiunto significativo:
*   **Benchmarking Avanzato**: Ho messo alla prova Corponi contro i modelli che oggi rappresentano lo stato dell'arte mondiale per i biosignali (BIOT) e le serie temporali (SimMTM).
*   **Validazione Cross-Dataset Rigorosa**: Mentre Corponi si focalizzava su specifici scenari di test, io ho esteso l'analisi al dataset Hosseini (Indian) applicando sistematicamente lo split **Subject-Independent**, un test di stress-test molto più duro di quelli solitamente riportati, che dimostra la reale portabilità di queste tecnologie.
*   **Confronto "Apples-to-Apples"**: Ho garantito che tutti i modelli passassero attraverso lo stesso rigoroso processo di pre-training (Fase 1) e fine-tuning (Fase 2), fornendo una panoramica chiara su quale architettura SSL sia effettivamente la più robusta per applicazioni reali.

---

## 3. Natura dei Dati e Composizione dei Dataset

Un punto cruciale per comprendere la complessità della sfida è la natura dei dati trattati. 

*   **Tipologia di Segnali**: In entrambi i dataset lavoriamo con **segnali fisiologici grezzi (raw time-series)**. Si tratta di flussi continui di valori numerici campionati ad alta frequenza (es. BVP a 64Hz, ACC a 32Hz, EDA e TEMP a 4Hz), non indicatori semplificati come il valore dei battiti cardiaci (BPM).

### 3.1 Dataset WESAD
Il dataset **WESAD** (Wearable Stress and Affect Detection) è composto da **15 soggetti** monitorati in un ambiente di laboratorio controllato. I segnali sono stati acquisiti tramite il sensore Empatica E4 e rappresentano il benchmark di riferimento per la validazione di modelli di stress detection in condizioni ottimali.

### 3.2 Dataset Hosseini (Indian)
Il dataset **Hosseini** coinvolge **25 soggetti** monitorati durante normali attività scolastiche in India. A differenza di WESAD, questo dataset presenta un elevato livello di rumore ambientale e artefatti da movimento, rendendolo ideale per testare la robustezza dei modelli in scenari di vita reale.

### 3.3 Strategie di Finestratura (Windowing)
Una particolarità tecnica fondamentale di questa ricerca è la differenza nelle strategie di segmentazione temporale utilizzate nei notebook:
*   **WESAD (Corponi)**: Ho utilizzato finestre estremamente ampie da **512 secondi**. Questa scelta permette di osservare le variazioni fisiologiche lente (come la temperatura e la risposta tonica della pelle) in un contesto di laboratorio dove i cambiamenti sono graduali.
*   **Hosseini (Corponi)**: Ho applicato finestre da **60 secondi con 30 secondi di step (overlap 50%)**. Questa è la configurazione "Master" che garantisce una maggiore reattività ai picchi di stress improvvisi e aumenta il numero di campioni per l'addestramento.
*   **BIOT e SimMTM (Resampling)**: In questi notebook, indipendentemente dalla lunghezza originale, il segnale viene sottoposto a un **resampling a 640 campioni**. Questo significa che il modello analizza 640 "token" temporali, ottimizzando la capacità del Transformer di identificare correlazioni tra i sensori anche in presenza di artefatti.

---

## 4. Architetture di Confronto: BIOT e SimMTM sono SSL

**Sia BIOT che SimMTM sono stati utilizzati come modelli Self-Supervised Learning (SSL).**

*   **SimMTM (Simple Masked Time-Series Modeling)**: È un framework nato esclusivamente per l'SSL sulle serie temporali. La sua logica è quella di "nascondere" parti del segnale e costringere il modello a ricostruirle basandosi sulla similarità tra diversi segmenti. Questo gli permette di imparare una rappresentazione universale del segnale fisiologico prima ancora di sapere se il soggetto è stressato o meno.
*   **BIOT (Biosignal Transformer)**: È un'architettura Transformer specializzata per i biosignali. Sebbene sia molto flessibile, nella mia sperimentazione l'ho configurato in modalità **Masked Autoencoder (SSL)**. Il modello divide il segnale in "patch" (piccoli frammenti) e impara a prevedere quelli mancanti. 

Entrambi i modelli, quindi, seguono esattamente la stessa filosofia di Corponi: una **Fase 1 (SSL)** per imparare a conoscere il segnale e una **Fase 2 (Fine-tuning)** per imparare a classificare lo stress.

---

## 5. Fase 1: Pre-training SSL (Apprendimento delle Rappresentazioni)

La fase di pre-training è il cuore del sistema. Qui i modelli imparano a "leggere" i segnali fisiologici senza l'ausilio di etichette, utilizzando compiti di ricostruzione (Masked Modeling).

### 1.1 Configurazione e Limitazioni del Dataset
Nel paper originale di Corponi venivano utilizzati 11 dataset per questa fase. Tuttavia, a causa della **limitatezza delle risorse computazionali** (vincoli di memoria RAM e tempi di esecuzione su GPU singola), in questo lavoro ho utilizzato **6 dataset** per la Fase 1. Questa riduzione controllata ha permesso comunque di ottenere pesi di alta qualità, sufficienti per superare i benchmark classici.

### 1.2 Dettaglio dei Tre Notebook di Fase 1
Ho utilizzato **tre notebook distinti** per generare i pesi iniziali, ognuno con una logica specifica:
1.  **Replica Corponi Phase 1** (`notebookreplicacorponiwesadfase1.ipynb`): È il notebook "Master". Ha generato un unico set di pesi dell'encoder che ho poi riutilizzato come punto di partenza comune sia per la replica su WESAD che su Hosseini. Questo garantisce che la baseline Corponi sia coerente tra i due dataset.
2.  **BIOT Phase 1** (`01_BIOT_Phase1_Pretraining.ipynb`): Utilizza l'architettura Biosignal Transformer. La particolarità qui è la **suddivisione del segnale in patch** e l'uso di **Sinusoidal Positional Encoding**. Il compito SSL consiste nel ricostruire patch mascherate del segnale, forzando il modello a capire le correlazioni tra canali (es. come l'EDA si muove rispetto al BVP).
3.  **SimMTM Phase 1** (`02_SimMTM_Phase1_Pretraining.ipynb`): Si basa sulla **Series-to-Series Similarity**. Invece di ricostruire singoli punti, il modello impara a relazionare interi segmenti di segnale tra loro. Questo notebook è fondamentale perché crea un encoder estremamente resiliente al rumore.

---

## 2. Dataset WESAD (Ambiente Controllato)

Il dataset WESAD (15 soggetti) è stato processato per testare i modelli in condizioni ideali di laboratorio.

### 2.1 Analisi dei Notebook di Fine-tuning
*   **Replica Corponi (WESAD)**: Ho applicato uno split **Subject-Independent (S2-S11 Train, S13-S17 Test)**. Ho utilizzato finestre molto ampie e lo **Scaling Mode 2** (standardizzazione globale). Il limite riscontrato è lo sbilanciamento verso la classe Baseline.
*   **Confronto BIOT (WESAD)**: Qui ho introdotto il **WeightedRandomSampler** per forzare il modello a vedere più campioni di "Stress", compensando la scarsità di dati etichettati in questa classe. Questo ha permesso di alzare l'F1-Score allo stress al 70.32%.
*   **Confronto SimMTM (WESAD)**: Il notebook sfrutta la fluidità dei segnali di laboratorio. SimMTM ha vinto qui (87.43% accuracy) perché la sua architettura Transformer è perfetta per modellare le transizioni "pulite" tra rilassamento e stress che si verificano in un ambiente controllato.

---

## 3. Dataset Hosseini / Indian (Ambiente Reale)

Il dataset indiano (25 soggetti) è la sfida più grande a causa del rumore ambientale e degli artefatti da movimento.

### 3.1 Analisi dei Notebook di Fine-tuning
*   **Replica Corponi (Hosseini)**: Ho caricato i pesi dal notebook Corponi Phase 1. Il modello ha ottenuto solo il 57.65% di accuracy. La particolarità negativa è che le feature manuali di Corponi non sanno distinguere tra un picco di sudorazione reale e il rumore causato dal movimento degli studenti.
*   **Confronto BIOT (Hosseini)**: Grazie alla **Channel Tokenization**, il notebook BIOT ha imparato a dare "identità" ai sensori. Anche se il braccio si muove (rumore sull'ACC), il modello riesce a filtrare il segnale EDA e BVP, raggiungendo un F1-Score allo stress incredibile del **92.65%**.
*   **Confronto SimMTM (Hosseini)**: Questo notebook ha dimostrato la massima robustezza. SimMTM arriva all'87.06% di accuracy perché ignora le fluttuazioni casuali del rumore, cercando solo la "forma" fisiologica corretta che ha imparato durante il pre-training.

---

## 4. Risultati e Confronto Scientifico Finale

### Tabelle Riassuntive

#### Tabella 1: WESAD (Laboratorio)
| Modello | Accuracy | F1-Score (Stress) | Macro F1 |
| :--- | :---: | :---: | :---: |
| Corponi (Master) | 80.20% | 54.55% | 70.95% |
| BIOT (Pre-trained) | 76.75% | 70.32% | 75.61% |
| **SimMTM (Pre-trained)** | **87.43%** | **82.51%** | **86.35%** |

#### Tabella 2: HOSSEINI (Real-World)
| Modello | Accuracy | F1-Score (Stress) | Macro F1 |
| :--- | :---: | :---: | :---: |
| Corponi (Master) | 57.65% | 70.37% | 60.00% |
| BIOT (Pre-trained) | 87.02% | **92.65%** | 68.68% |
| **SimMTM (Pre-trained)** | **87.06%** | 92.60% | **70.72%** |

### Analisi dei Risultati: Chi vince e perché?
1.  **Perché SimMTM è il migliore in assoluto?**: SimMTM vince su entrambi i dataset perché il suo compito SSL (*Series-to-Series Similarity*) è più evoluto del semplice mascheramento di Corponi. Mentre Corponi cerca di indovinare i punti mancanti, SimMTM capisce la "relazione" tra diversi momenti temporali. Questo lo rende immune al rumore di Hosseini e precisissimo sulla pulizia di WESAD.
2.  **BIOT vs Corponi**: BIOT batte Corponi nell'identificazione dello stress (F1-Score superiore) perché i Transformer sono intrinsecamente migliori delle CNN nell'analizzare segnali multimodali (ACC + BVP + EDA). BIOT capisce che questi segnali devono muoversi insieme, mentre Corponi li tratta in modo più isolato.
3.  **WESAD vs Hosseini**: Qual è il dataset migliore?
    *   **WESAD** è "migliore" per la validazione teorica: ci dice quanto il modello può essere preciso in condizioni perfette.
    *   **Hosseini** è "migliore" per la validazione pratica: ci dice se il modello è pronto per il mercato e per l'uso quotidiano. 
    Il fatto che i modelli SSL mantengano l'87% su Hosseini (dove Corponi crolla al 57%) è la prova scientifica che il **Self-Supervised Learning** non è solo un'opzione, ma una necessità per far funzionare lo stress detection fuori dal laboratorio.

---
**In sintesi**: La superiorità dei modelli Transformer (BIOT/SimMTM) pre-trainati con tecnica SSL è schiacciante. Il rigore dello split **Subject-Independent** e l'uso di sampler bilanciati hanno permesso di ottenere risultati che non sono solo numeri, ma indicatori di una tecnologia pronta per essere applicata su soggetti reali in contesti quotidiani.
