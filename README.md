# ProjectBioInformatic

# Project details

## datasetManager.py
Le reti neurali basate su funzioni di attivazione come ReLU o Sigmoide soffrono immensamente se le feature hanno scale diverse. Se l'età varia tra 10 e 90 e la conta dei CD4 tra 0 e 1000, i gradienti della rete si concentreranno quasi esclusivamente sui CD4, ignorando l'età. Per questo applichiamo lo StandardScaler (Z-score normalization), che trasforma ogni feature in modo da avere media 0 e varianza 1.
Un'altra scelta cruciale qui è definire formalmente l'evento di pericolo. Nel dataset ACTG 175, l'inefficacia è legata al fallimento della terapia (censor) e la tossicità è legata all'interruzione del trattamento per eventi avversi (**offtrt**). Creiamo un'etichetta binaria sintetica: il paziente è "Unsafe" (1) se sperimenta almeno una di queste due condizioni negative.

## siamesedataset.py

Una rete neurale standard accetta un input $X$ e predice un output $Y$. Una rete Siamese lavora per confronto. Ha bisogno di ricevere in input due campioni distinti ($X_1, X_2$) e una label di similarità che dice se i due campioni appartengono alla stessa classe (label = 1) o a classi diverse (label = 0); Creiamo un generatore che costruisce queste **coppie** (l'input della rete) in modo stocastico. Durante l'addestramento, per ogni paziente estratto, decidiamo casualmente se accoppiarlo con un paziente con lo stesso esito clinico (coppia genuina) o con un esito opposto (coppia impostore). Questo costringe la rete a mappare le distanze nello spazio latente in modo significativo per il nostro obiettivo.

## network.py

Qui strutturiamo la rete, non come due reti diverse, ma con un'unica sottorete che viene istanziata una volta. Quando passiamo la coppia di input (X_1, X_2), invochiamo la stessa identica sequenza di strati e pesi su entrambi gli input, e questo garantisce che se anche volessimo invertire l'ordine  le coordinate rimangano le stesse.

## losses.py

Qui è dove utilizziamo una funzione di loss per comprendere e il comportamento geometrico attorno allo spazio latente (spazio degli input caratterizzanti). La funzione utilizzata è quella di Contrastive Loss la cui equazione è : $$L(E_1, E_2, Y) = Y \cdot D^2 + (1 - Y) \cdot \max(0, M - D)^2$$

dove M è il _margin_

Se la label _Y=1_ (pazienti simili), la seconda parte si annulla e la loss è semplicemente _D^2_. Il gradiente spingerà i vettori ad avvicinarsi (distanza tende a zero).Se la label _Y=0_ (pazienti diversi), la prima parte si annulla. Se la distanza _D_ è già maggiore del margine _M_, il termine diventa zero (la rete non viene penalizzata). Se la distanza è inferiore al margine, la rete riceve una penalità e i gradienti spingono i vettori ad allontanarsi fino a superare la distanza di sicurezza _M_.

## trainer.py

Allena e traina tramite backpropagation e validation.

## policyevaluation.py

Qui si cerca finalmente, dopo aver impostato e settato i vari parametri, aver chiarito come è ordinato lo spazio latente, identificato qual è il **centroide** di Pericolo, isolato i pazienti con tossicità ad alto rischio storicamente (is_unsafe == 1), allora possiamo identificare su un nuovo input, la distanza euclidea dal centroide, e se è inferiore al parametro critico ($\rho$), allora viene identificato "Ad altissimo rischio di tossicità"

## main.py

Esegue l'intero programma.

# Installazione e esecuzione

### 1. Clonare il repository o scaricare i file
Assicurati che tutti i file `.py` si trovino all'interno della stessa cartella di lavoro.

### 2. Creare un ambiente virtuale (Consigliato)
Apri il terminale nella cartella del progetto e digita:

Creazione dell'ambiente virtuale
```
python -m venv venv
```
### Attivazione (Windows)
```
venv\Scripts\activate
```

### Attivazione (Linux / MacOS)
```
source venv/bin/activate
```

### Installare le dipendenze
```
pip install -r requirements.txt
```

# Dataset ACTG 175
L'**ACTG 175** (AIDS Clinical Trials Group Study 175) è un dataset storico e famosissimo nell'ambito del Causal Inference e della medicina. Rappresenta un trial clinico randomizzato del 1996 su pazienti affetti da HIV.

L'obiettivo originale era confrontare la monoterapia (un solo farmaco) con terapie combinate (più farmaci) analizzando la progressione della malattia attraverso il conteggio delle cellule CD4.

All'interno del progetto, l'**obiettivo** è quello di mappare le variabili del dataset per isolare l'**efficacia** e la **tossicità**.
Le informazioni e la documentazione è possibile visionarla al loro github: https://github.com/uci-ml-repo/ucimlrepo

# Esempi
* ## Uber: Ottimizzazione delle Promozioni e Ritenzione degli Utenti
  **Uber** deve decidere se e quanto sconto (trattamento) offrire a un utente per incentivarlo a usare l'app dopo che ha vissuto un'esperienza negativa (es. un forte ritardo del conducente). Dare sconti a tutti è costosissimo, mentre darne troppi pochi causa l'abbandono dell'utente (churn).
  ### L'Applicazione Causale MTL
  I data scientist di Uber hanno implementato modelli di Uplift Modeling basati su reti neurali Multi-Task. Nello specifico, hanno integrato e reso open-source l'architettura [DragonNet](https://towardsdatascience.com/tarnet-and-dragonnet-causal-inference-between-s-and-t-learners-0444b8cc65bd/). La rete ha una base condivisa che apprende una rappresentazione dell'utente (cronologia viaggi, zona, livello di insoddisfazione) e si dirama in task/teste diverse per prevedere l'esito (se l'utente abbandona o meno) con o senza la promozione, calcolando l'effetto causale netto condizionale (CATE).
  ### Fonti
  - [Paper scientifico](https://proceedings.mlr.press/v104/du19a/du19a.pdf)
   - [Codice Sorgente](https://gemini.google.com/app/ee3024598ce79bf1#:~:text=Vedi%20il%20codice%20di%20DragonNet%20su%20GitHub%20)
* ## Booking: Valutare l'Impatto Reale delle Nuove Funzionalità (Voluntary Adoption)
  Quando Booking rilascia una nuova funzionalità (ad esempio, un nuovo sistema di pagamento flessibile opzionale per gli hotel partner), non può obbligare nessuno a usarla. Gli hotel che decidono autonomamente di adottarla sono solitamente quelli più moderni, grandi o digitalizzati. Se Booking confrontasse semplicemente i partner che usano la feature con quelli che non la usano, vedrebbe che chi la usa guadagna di più. Ma è merito della feature (Causa) o del fatto che quegli hotel erano già più forti in partenza (Confondimento/Bias di selezione)?
  ### L'Applicazione Causale MTL
  **Booking**  utilizza il [Machine Learning Causale](https://arxiv.org/abs/2206.15475) e architetture neurali **Multi-Task** per stimare l'impatto reale isolando il bias. Tramite un approccio controfattuale, il modello stima contemporaneamente due task (i potenziali outcome): l'andamento delle prenotazioni dell'hotel se decidesse di adottare la feature e l'andamento se decidesse di non farlo, bilanciando le caratteristiche di partenza dei partner per simulare un perfetto "A/B test" anche dove l'esperimento puro è impossibile.
  ### Fonti
  - [Causality Gap](https://booking.ai/the-causality-gap-measuring-the-true-impact-of-voluntary-adoption-in-digital-marketplaces-ea68b5a35120)
* ## Piattaforme di Streaming e E-Commerce (es. Netflix, Alibaba): Uplift Modeling Multi-Trattamento per le Raccomandazioni
  Nelle moderne piattaforme di e-commerce o streaming, non esiste un solo trattamento binario (Sconto Sì / Sconto No), ma esistono trattamenti multipli (es. Mostrare la locandina A, la locandina B, inviare una **notifica push** aggressiva o una soft). L'obiettivo è massimizzare l'engagement totale minimizzando il disturbo all'utente.
  ### L'Applicazione Causale MTL
  In questi contesti si usano estensioni di **TARNet** e modelli Multi-Task in cui la rete neurale non ha solo 2 teste, ma ha "N" teste (una per ogni possibile trattamento/incentivo disponibile).Lo strato condiviso estrae i pattern storici dell'utente. Ogni "testa" predice l'esito specifico sotto quel preciso trattamento (es. la probabilità di cliccare sul video vedendo la locandina X).Questo approccio, ampiamente documentato nelle conferenze di settore come [KDD](https://www.isti.cnr.it/it/ricerca/ricerca/laboratories/9/Knowledge_Discovery_and_Data_Mining_KDD)(Knowledge Database and Data Mining) o [RecSys](https://recsys.acm.org/), permette di fare decisioni in tempo reale (**Online Dynamic Decision-Making**) calcolando quale specifico trattamento genererà il maggior incremento (**uplift**) di conversioni rispetto allo scenario di controllo (nessun intervento).

  ### Fonti
  - [Paper di Ricerca sull'Uplift Multi-Trattamento](https://www.researchgate.net/publication/338797004_Uplift_Modeling_for_Multiple_Treatments_with_Cost_Optimization)
   - [tarNet e arXiv](https://arxiv.org/abs/1606.03976)

> **Supportate** i miei lavori con una stellina! ⭐️ 
> Grazie mille! 🤖